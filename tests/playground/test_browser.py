"""Browser tests for the playground (spec section 9). Opt-in: python -m pytest tests/playground/test_browser.py -m browser -v

Setup once: python -m pip install 'playwright>=1.40' && python -m playwright install chromium
Each test opens a fresh browser context, so each one downloads Pyodide again (about 10 MB).
The module takes about 8 minutes. Every number that depends on the machine (load, lap speed, Stop, tick gap) is printed on each run
and asserted only under PLAYGROUND_TIMING=1, on an idle, wired Mac.
"""
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))  # the course scripts are importable modules

from playwright.sync_api import expect, sync_playwright  # noqa: E402

from _course import REPO, load_solution  # noqa: E402
from build_playground import exercise_records, write_playground  # noqa: E402
from playground_browser import P_STEER, exercise_url, open_ready, paint_colours, pasteable, serve, set_code, wait_state  # noqa: E402
from zero2fsd.car import Car, run  # noqa: E402
from zero2fsd.grade import check  # noqa: E402
from zero2fsd.preview import get as get_preview  # noqa: E402
from zero2fsd.sim.camera import Camera  # noqa: E402
from zero2fsd.sim.dynamics import DT  # noqa: E402

pytestmark = pytest.mark.browser

TIMING = bool(os.environ.get("PLAYGROUND_TIMING"))
LOAD_BUDGET_S = 3.0  # spec S-1
REALTIME_FACTOR = 5  # spec §7: a lap of gentle with camera runs at >= 5x real time on a Mac (measured 10.3x)
STOP_BUDGET_S = 3.0  # spec S-4
TICK_GAP_BUDGET_MS = 500  # the longest the page may go between two 20 ms timer ticks while a `while True` runs
STEER_ID = "0.1.4.a"
# A literal on purpose: this key is the contract with the site (src/lib/learnfsd-progress.ts:7), not something to read back from progress.js.
PROGRESS_KEY = "learnfsd.v2.progress"
PASSING = ("0.1.4.a", "0.1.3.a")  # two units, so the second pass has a first one to leave alone
SAVED_COPY = "Your progress is saved"  # app.js's status line for a pass that was stored
STREAMING = ("0.1.4.b", "0.1.3.b")  # spec 5.3: the two checks that drive the car, so their rows stream
# What a learner holds before writing anything, pasted and as the Mac holds it. check() refuses it in require_own before it
# drives, so a Check on it streams no rows. (Not `provided.estimate_lane`: the shim rejects a pasted provided box as a NameError.)
UNDRIVEN = {
    "0.1.4.b": ("from zero2fsd.car import Car\nmy_car = Car()\n", Car()),
    "0.1.3.b": ("estimate_lane = None\n", None),
}
# The exercises S-2 covers, one case each. A registry that lost one would shrink the loop without failing it.
WEEK_1 = ("0.1.1.a", "0.1.1.b", "0.1.2.a", "0.1.3.a", "0.1.3.b", "0.1.4.a", "0.1.4.b")
# The worker message the page's Run sends for each exercise kind (app.js KINDS), and the same call made in-process on the shim.
# Keyed by kind, never by exercise id: a new exercise of a known kind needs no edit here.
RUN_MESSAGE = {"drive": "run", "stats": "stats", "image": "image"}
NATIVE_RUN = {
    "drive": lambda shim, record, learner: shim.preview_run(learner, record["id"], record["road"], lambda row: None),
    "stats": lambda shim, record, learner: shim.preview_stats(learner, record["id"]),
    "image": lambda shim, record, learner: shim.preview_image(learner, record["id"]),
}
PNG_MAGIC = [137, 80, 78, 71]
CAMERA_SIZE = [Camera().width, Camera().height]  # the worker reads these two from this same class to size a frame

RECORDS = exercise_records(REPO)
BY_ID = {record["id"]: record for record in RECORDS}
RUN_KINDS = tuple(RUN_MESSAGE)  # zero2fsd.preview.Kind: Week 1 has an exercise of each
STATS = [record for record in RECORDS if record["kind"] == "stats"]
IMAGES = [record for record in RECORDS if record["kind"] == "image"]
PLAYER_CONTROLS = ("#play", "#scrub", "#to-end")  # visible on a drive exercise, hidden on an image exercise
READOUTS = ("#r-t", "#r-lat", "#r-speed", "#r-steer", "#r-progress")


def first_of_kind(kind):
    found = [record for record in RECORDS if record["kind"] == kind]
    assert found, f"no exercise has kind {kind!r}; these tests need one per preview kind"
    return found[0]


ONE_PER_KIND = [first_of_kind(kind) for kind in RUN_KINDS]
# Per kind: the exercise whose answer is a number, and the plain-language text the page must show when it is not (Task 7's
# `_at_boundary` and `_as_stats`). The other drive exercises answer with a LaneEstimate or a Car: Task 7 tests those messages.
WRONG_RETURN = {
    "drive": (BY_ID[STEER_ID], "must return one number"),
    "stats": (first_of_kind("stats"), "expected a number or a dict of numbers"),
}

RETURN_ZERO = "def p_steer(est, k_off, k_head):\n    return 0.0\n"
FOREVER = "def p_steer(est, k_off, k_head):\n    while True:\n        pass\n"
DIVIDES_BY_ZERO = "def p_steer(est, k_off, k_head):\n    return 1 / 0\n"
RAISES_AFTER_50_STEPS = '''calls = 0


def p_steer(est, k_off, k_head):
    global calls
    calls += 1
    if calls > 50:
        raise ValueError("steering table exhausted")
    return 0.0
'''

# Records every data-state the status line LEAVES, so a test can read the order the page went through. Reading the new value
# at delivery would lose a state the page passes through inside one task (`ready` straight into `running` on an auto-run).
STATE_RECORDER = """
window.__states = [];
new MutationObserver((records) => {
  for (const r of records) if (r.target.id === "status") window.__states.push(r.oldValue);
}).observe(document, { subtree: true, attributes: true, attributeOldValue: true, attributeFilter: ["data-state"] });
"""
# Records the type of every message the page posts to its worker, so a test can count the requests a page action made.
POSTED_RECORDER = """
window.__posted = [];
const postMessage = Worker.prototype.postMessage;
Worker.prototype.postMessage = function (message, ...rest) {
  window.__posted.push(message.type);
  return postMessage.call(this, message, ...rest);
};
"""
# Records every text the status line shows (it reads the dashboard's download, which sets no state of its own).
STATUS_TEXT_RECORDER = """() => {
  window.__texts = [];
  const status = document.getElementById("status");
  new MutationObserver(() => window.__texts.push(status.textContent)).observe(status, { childList: true, characterData: true, subtree: true });
}"""
# localStorage exists but every read throws, as in a browser with site data blocked.
BLOCK_STORAGE = """Object.defineProperty(window, "localStorage", {
  configurable: true,
  get() { throw new DOMException("storage is blocked", "SecurityError"); },
});"""
# How many pixels of a canvas differ from its corner pixel (the background).
DRAWN_PIXELS_JS = """(selector) => {
  const c = document.querySelector(selector), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  const key = (i) => ((d[i] << 24) | (d[i + 1] << 16) | (d[i + 2] << 8) | d[i + 3]) >>> 0, corner = key(0);
  let drawn = 0;
  for (let i = 0; i < d.length; i += 4) if (key(i) !== corner) drawn++;
  return drawn;
}"""
# On a canvas: how many pixels are not black, and for each colour how many pixels are within 8 per channel of it
# (a canvas that blends the overlay with alpha rounds the colour a little).
MASK_PIXELS_JS = """({ selector, colours }) => {
  const c = document.querySelector(selector), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  const near = (i, rgb) => rgb.every((v, k) => Math.abs(d[i + k] - v) <= 8);
  const counts = { nonBlack: 0, colour: colours.map(() => 0) };
  for (let i = 0; i < d.length; i += 4) {
    if (d[i] || d[i + 1] || d[i + 2]) counts.nonBlack++;
    colours.forEach((rgb, k) => { if (near(i, rgb)) counts.colour[k]++; });
  }
  return counts;
}"""
FRAME_SUM_JS = """() => {
  const c = document.querySelector('#frame');
  return c.getContext('2d').getImageData(0, 0, c.width, c.height).data.reduce((a, b) => a + b, 0);
}"""
# The pixel size of the camera-frame canvas.
FRAME_SIZE_JS = """() => {
  const c = document.querySelector('#frame');
  return [c.width, c.height];
}"""
# Counts the 20 ms timer ticks on the page's main thread and keeps the longest gap between two: a frozen page ticks no more.
TICKER_JS = """() => {
  window.__ticks = { last: performance.now(), maxGap: 0, count: 0 };
  setInterval(() => {
    const now = performance.now();
    window.__ticks.count++;
    window.__ticks.maxGap = Math.max(window.__ticks.maxGap, now - window.__ticks.last);
    window.__ticks.last = now;
  }, 20);
}"""
# (progress text, number of verdict lines on screen) every time the progress readout changes.
TRACE_PROGRESS_JS = """() => {
  window.__trace = [];
  const readout = document.querySelector('#r-progress');
  new MutationObserver(() => window.__trace.push([readout.textContent, document.querySelectorAll('#verdict li').length]))
    .observe(readout, { childList: true, characterData: true, subtree: true });
}"""
# The worker started by hand on a same-origin document (the page's own worker.js, no page): every message it posts is kept in order.
WORKER_JS = """(init) => {
  window.messages = [];
  window.worker = new Worker("./worker.js", { type: "module" });
  window.worker.onmessage = ({ data }) => {
    const copy = { ...data };
    if (data.png) { copy.png_magic = Array.from(new Uint8Array(data.png).slice(0, 4)); delete copy.png; }
    window.messages.push(copy);
  };
  window.worker.onerror = (event) => window.messages.push({ type: "onerror", message: event.message });
  if (init) window.worker.postMessage({ type: "init" });
}"""
SENTINEL = {"type": "sentinel"}  # a request no worker has a handler for: its error closes the list of replies before it


@pytest.fixture(scope="module")
def base_url(tmp_path_factory):
    dist = tmp_path_factory.mktemp("playground-dist")
    write_playground(REPO, dist)
    with serve(dist) as url:
        yield url


@pytest.fixture(scope="module")
def chromium():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        yield browser
        browser.close()


@pytest.fixture
def page_errors():
    return []


@pytest.fixture
def page(chromium, page_errors):
    context = chromium.new_context()
    context.add_init_script(STATE_RECORDER)
    context.add_init_script(POSTED_RECORDER)
    page = context.new_page()
    page.set_default_timeout(60_000)
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    yield page
    context.close()


def visited(page):
    """The data-state values the status line went through (the ones it left, then the one it is in), repeats collapsed."""
    states = page.evaluate("[...window.__states, document.getElementById('status').dataset.state]")
    return [state for i, state in enumerate(states) if i == 0 or state != states[i - 1]]


def line_of(code, needle):
    """1-based line number of the first line containing `needle`: the number the page must report for that line."""
    return next(i for i, line in enumerate(code.splitlines(), 1) if needle in line)


def first_number(text):
    match = re.search(r"\d+(?:\.\d+)?", text)
    assert match, f"no number in {text!r}"
    return float(match.group())


def press_check(page, code):
    """Put `code` in the editor, press Check, return the verdict's lines once the page reports done."""
    set_code(page, code)
    page.click("#check")
    wait_state(page, "done")
    return page.locator("#verdict li").all_text_contents()


def native_check(record):
    """The verdict the Mac prints for the reference solution of `record`."""
    return check(record["id"], getattr(load_solution(REPO, record["unit_id"]), record["name"]), show=False)


def native_error(shim, record, code):
    """The exception the Mac's shim raises running `code` as the page runs it (a Run press), or None if it runs."""
    try:
        NATIVE_RUN[record["kind"]](shim, record, shim.learner_object(code, record["name"]))
    except Exception as exc:  # noqa: BLE001 - whatever `code` raises is the expectation
        return exc
    return None


def stored_progress(page):
    return page.evaluate("key => localStorage.getItem(key)", PROGRESS_KEY)


def scroll_width(page):
    return page.evaluate("document.documentElement.scrollWidth")


def freeze_playhead(page):
    """Pause the replay and return the five readouts, which no longer change."""
    if page.inner_text("#play") == "Pause":
        page.click("#play")
    expect(page.locator("#play")).to_have_text("Play")
    return [page.inner_text(readout) for readout in READOUTS]


# ---- opening ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["id"])
def test_every_exercise_opens_ready_with_its_label_and_links(page, base_url, record):
    open_ready(page, base_url, record["id"])
    expect(page.locator("#exercise-label")).to_have_text(record["label"])
    assert not re.search(r"\d+\.\d+\.\d+", page.inner_text("#exercise-label")), "a dotted id reached the learner"
    expect(page.locator("#unit-link")).to_have_attribute("href", f"../unit/{record['unit_key']}/")  # relative: spec §12
    expect(page.locator("#open-colab")).to_have_attribute("href", record["colab_url"])


def test_the_registry_holds_the_seven_week_1_exercises():
    """S-2 and the opening test loop over the registry, so this is what keeps both loops at seven."""
    assert set(WEEK_1) <= set(BY_ID)
    assert {BY_ID[exercise_id]["kind"] for exercise_id in WEEK_1} == set(RUN_KINDS)


@pytest.mark.parametrize("bad_id", ["9.9.9.z", "nonsense"])
def test_unknown_exercise_opens_the_first_exercise_and_says_why(page, base_url, bad_id):
    page.goto(exercise_url(base_url, bad_id))
    wait_state(page, "ready")
    expect(page.locator("#status")).to_contain_text("Unknown exercise")
    expect(page.locator("#exercise-label")).to_have_text(RECORDS[0]["label"])


@pytest.mark.skipif(not TIMING, reason="timing budgets need an idle, wired machine: set PLAYGROUND_TIMING=1")
def test_cold_load_reaches_ready_within_3s(page, base_url):
    started = time.monotonic()
    page.goto(exercise_url(base_url, STEER_ID))
    wait_state(page, "ready", timeout_ms=int(LOAD_BUDGET_S * 1000))
    print(f"load_s={time.monotonic() - started:.2f}")


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["id"])
def test_preset_starter_runs_automatically(page, base_url, shim, record):
    """?preset=starter loads the starter and runs it once after ready. What the run reports is what the Mac's shim reports
    for that starter: the expectation is derived by running it in-process, so a new exercise needs no table entry."""
    open_ready(page, base_url, record["id"])
    assert "running" not in visited(page), "a plain URL started a run on its own"
    page.goto(exercise_url(base_url, record["id"], "&preset=starter"))
    page.wait_for_function("() => ['done', 'error'].includes(document.getElementById('status').dataset.state)")
    states = visited(page)
    assert states.index("ready") < states.index("running")
    assert page.evaluate("window.__getCode()") == record["presets"]["starter"], "the editor does not hold the starter verbatim"
    requests = [type_ for type_ in page.evaluate("window.__posted") if type_ in {*RUN_MESSAGE.values(), "check"}]
    assert requests == [RUN_MESSAGE[record["kind"]]], f"the starter must run exactly once, by this kind's own message: {requests}"
    expected = native_error(shim, record, record["presets"]["starter"])
    assert page.get_attribute("#status", "data-state") == ("error" if expected else "done")
    if expected:
        name = type(expected).__name__
        status = " ".join(page.inner_text("#status").split())  # #status collapses spaces; the shim's empty message leaves two
        assert status.startswith(f"{name}:"), f"the page names a different exception than the Mac's shim ({name}): {status!r}"
        assert " ".join(page.inner_text("#stdout").split()).count(status) == 1, "the error line is not in #stdout exactly once"


# ---- running ------------------------------------------------------------------------------------------------


def test_run_provided_steer_completes_and_draws(page, base_url):
    record = BY_ID[STEER_ID]
    open_ready(page, base_url, STEER_ID)
    expect(page.locator("#editor .cm-content")).to_have_count(1)  # CodeMirror itself loaded, not the textarea fallback
    set_code(page, P_STEER)
    started = time.monotonic()
    page.click("#run")
    wait_state(page, "done")
    elapsed = time.monotonic() - started
    assert "running" in visited(page)
    expect(page.locator("#status")).to_contain_text(f"Completed {record['road']}")
    assert first_number(page.inner_text("#r-progress")) > 0
    expect(page.locator("#mask")).to_be_hidden()  # the image preview's canvas belongs to image exercises only
    for control in PLAYER_CONTROLS:
        expect(page.locator(control)).to_be_visible()
    for canvas in ("#canvas", "#strips"):
        assert page.evaluate(DRAWN_PIXELS_JS, canvas) > 0, f"{canvas} has only background pixels"
    steps = int(page.get_attribute("#scrub", "max")) + 1
    assert abs(steps - len(run(Car(), record["road"]).t)) <= 2  # the page drove the same lap the Mac drives
    print(f"real_time_x={steps * DT / elapsed:.1f}")  # recorded in the acceptance PR (step C9)
    if TIMING:
        assert elapsed * REALTIME_FACTOR < steps * DT, f"the lap ran at {steps * DT / elapsed:.1f}x real time; spec section 7 wants {REALTIME_FACTOR}x"
    page.click("#to-end")
    expect(page.locator("#scrub")).to_have_value(str(steps - 1))


def test_scrubbing_redraws_the_camera_frame(page, base_url):
    open_ready(page, base_url, STEER_ID)
    set_code(page, P_STEER)
    page.click("#run")
    wait_state(page, "done")
    freeze_playhead(page)  # hold the playhead still so only the scrubber moves it
    page.click("#to-end")
    page.wait_for_function(f"({FRAME_SUM_JS})() > 0")
    assert page.evaluate(FRAME_SIZE_JS) == CAMERA_SIZE, "the camera frame is not painted at the camera's own width and height"
    at_end = page.evaluate(FRAME_SUM_JS)
    page.fill("#scrub", "0")
    page.wait_for_function(f"({FRAME_SUM_JS})() !== {at_end}")


def test_play_button_label_follows_the_playhead(page, base_url):
    open_ready(page, base_url, STEER_ID)
    set_code(page, P_STEER)
    page.click("#run")
    wait_state(page, "done")
    freeze_playhead(page)  # leaves the label at "Play"
    page.click("#play")
    expect(page.locator("#play")).to_have_text("Pause")
    page.click("#play")
    expect(page.locator("#play")).to_have_text("Play")


def test_dashboard_library_loads_only_on_demand(page, base_url):
    requested = []
    page.on("request", lambda request: requested.append(request.url))
    open_ready(page, base_url, STEER_ID)
    set_code(page, P_STEER)
    page.click("#run")
    wait_state(page, "done")
    assert not [url for url in requested if "matplotlib" in url.lower()], "matplotlib was fetched before Dashboard was pressed"
    page.evaluate(STATUS_TEXT_RECORDER)
    page.click("#dashboard")
    expect(page.locator("#dashboard-img")).to_be_visible(timeout=120_000)
    assert [url for url in requested if "matplotlib" in url.lower()], "pressing Dashboard did not fetch matplotlib"
    # After `ready`, the worker's one download is a lazy dependency: the page words it as the dashboard's, not as start-up.
    assert "Loading matplotlib for the dashboard..." in page.evaluate("window.__texts")


def test_dashboard_is_enabled_only_by_a_completed_run(page, base_url):
    """The worker keeps the last completed run for the Dashboard; the page enables the button only while it holds one."""
    open_ready(page, base_url, STEER_ID)
    expect(page.locator("#dashboard")).to_be_disabled()  # nothing has run
    set_code(page, P_STEER)
    page.click("#run")
    wait_state(page, "done")
    expect(page.locator("#dashboard")).to_be_enabled()
    set_code(page, DIVIDES_BY_ZERO)
    page.click("#run")
    wait_state(page, "error")
    expect(page.locator("#dashboard")).to_be_disabled()  # a failed run leaves no run to draw
    set_code(page, P_STEER)
    page.click("#run")
    wait_state(page, "done")
    expect(page.locator("#dashboard")).to_be_enabled()
    page.click("#check")
    wait_state(page, "done")
    expect(page.locator("#dashboard")).to_be_disabled()  # a check is not a run


@pytest.mark.parametrize("record", STATS, ids=lambda record: record["id"])
def test_stats_exercise_shows_native_values(page, base_url, record):
    reference = getattr(load_solution(REPO, record["unit_id"]), record["name"])
    preview = get_preview(record["id"])
    native = preview.build(reference, run(Car(), preview.road))  # the canned drive is the preview's own road
    open_ready(page, base_url, record["id"])
    set_code(page, pasteable(record["unit_id"]))
    page.click("#run")
    wait_state(page, "done")
    shown = dict(zip(page.locator("#stats dt").all_text_contents(), page.locator("#stats dd").all_text_contents()))
    assert shown.keys() == native.keys()
    for label, value in native.items():
        assert math.isclose(float(shown[label]), value, rel_tol=1e-3), f"{label}: page {shown[label]}, Mac {value}"
    expect(page.locator("#mask")).to_be_hidden()


@pytest.mark.parametrize("record", IMAGES, ids=lambda record: record["id"])
def test_image_exercise_paints_the_masks_and_hides_the_player(page, base_url, record):
    """Week 1's paint-mask exercise: one camera frame in #frame, the learner's masks in #mask, no player."""
    open_ready(page, base_url, record["id"])
    set_code(page, pasteable(record["unit_id"]))
    page.click("#run")
    wait_state(page, "done")
    expect(page.locator("#frame")).to_be_visible()
    expect(page.locator("#mask")).to_be_visible()
    for control in PLAYER_CONTROLS:
        expect(page.locator(control)).to_be_hidden()
    painted = page.evaluate(MASK_PIXELS_JS, {"selector": "#mask", "colours": paint_colours(base_url)})
    assert painted["nonBlack"] > 0
    assert all(count > 0 for count in painted["colour"]), f"a mask was not painted in its colour: {painted}"


@pytest.mark.parametrize("record", IMAGES, ids=lambda record: record["id"])
def test_image_exercise_with_empty_masks_paints_no_overlay(page, base_url, record):
    """The control for the test above: a learner who marks nothing gets no overlay colour, so the count can tell the two apart."""
    open_ready(page, base_url, record["id"])
    nothing = f"import numpy as np\n\n\ndef {record['name']}(frame):\n    none = np.zeros(frame.shape[:2], dtype=bool)\n    return none, none\n"
    set_code(page, nothing)
    page.click("#run")
    wait_state(page, "done")
    colours = paint_colours(base_url)
    painted = page.evaluate(MASK_PIXELS_JS, {"selector": "#mask", "colours": colours})
    assert painted["colour"] == [0] * len(colours), painted


# ---- check --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["id"])
def test_check_matches_native_verdict_on_the_reference(page, base_url, record):
    """S-2: the verdict Pyodide prints for the reference solution is the verdict the Mac prints, line for line.

    One case per exercise in the registry: the seven of Week 1, the image exercise 0.1.2.a among them. No kind is skipped.
    """
    native = native_check(record)
    assert native.passed, f"the reference for {record['id']} fails natively: {native.details}"
    open_ready(page, base_url, record["id"])
    lines = press_check(page, pasteable(record["unit_id"]))
    expect(page.locator("#verdict")).to_have_attribute("data-passed", "true")
    assert lines == native.details
    expect(page.locator("#stdout")).to_have_text("")  # the report is the verdict; a check posts no grader output


def test_a_learners_own_print_reaches_stdout_during_a_check(page, base_url):
    """The control for the line above: #stdout is empty after a check because the grader prints nothing, not because it is never shown."""
    open_ready(page, base_url, STEER_ID)
    press_check(page, "print('mine')\n" + P_STEER)
    expect(page.locator("#verdict")).to_have_attribute("data-passed", "true")
    expect(page.locator("#stdout")).to_have_text("mine\n")


def test_failing_check_has_the_native_shape(page, base_url):
    """A failing verdict carries numpy-formatted arrays, which differ by version: compare headline, line count and indentation."""
    namespace = {}
    exec(RETURN_ZERO, namespace)
    native = check(STEER_ID, namespace["p_steer"], show=False)
    assert not native.passed
    open_ready(page, base_url, STEER_ID)
    lines = press_check(page, RETURN_ZERO)
    expect(page.locator("#verdict")).to_have_attribute("data-passed", "false")
    assert lines[0] == native.details[0]
    indent = lambda text: len(text) - len(text.lstrip())  # noqa: E731
    assert [indent(line) for line in lines] == [indent(line) for line in native.details]


def test_pass_is_recorded_in_progress_store(page, base_url):
    for exercise_id in PASSING:
        open_ready(page, base_url, exercise_id)
        press_check(page, pasteable(BY_ID[exercise_id]["unit_id"]))
        expect(page.locator("#verdict")).to_have_attribute("data-passed", "true")
        expect(page.locator("#status")).to_contain_text(SAVED_COPY)  # the control for test_storage_blocked's "not shown"
    store = json.loads(stored_progress(page))  # read-modify-write: the second pass must not erase the first
    assert store["version"] == 1
    for exercise_id in PASSING:
        entry = store["units"][BY_ID[exercise_id]["unit_key"]]["exercises"][exercise_id]
        assert entry["via"] == "playground"
        assert entry["passedAt"]


def test_failing_check_records_nothing(page, base_url):
    open_ready(page, base_url, STEER_ID)
    press_check(page, RETURN_ZERO)
    expect(page.locator("#verdict")).to_have_attribute("data-passed", "false")
    assert STEER_ID not in (stored_progress(page) or "")


@pytest.mark.parametrize("exercise_id", STREAMING)
def test_check_streams_rows_before_the_verdict(page, base_url, exercise_id):
    """Pastes the reference solution, which holds the learner's own steer and perception. An empty Car would not do:
    check("0.1.4.b", Car()) fails in require_own before it drives, so nothing would stream."""
    open_ready(page, base_url, exercise_id)
    page.evaluate(TRACE_PROGRESS_JS)
    press_check(page, pasteable(BY_ID[exercise_id]["unit_id"]))
    before_verdict = {text for text, verdict_lines in page.evaluate("window.__trace") if verdict_lines == 0}
    assert len(before_verdict) >= 2, f"#r-progress never moved while the check drove: {before_verdict}"
    expect(page.locator("#verdict")).to_have_attribute("data-passed", "true")


@pytest.mark.parametrize("exercise_id", STREAMING)
def test_check_that_fails_before_driving_streams_nothing(page, base_url, exercise_id):
    """The control for the test above: the provided boxes are refused before the drive, so #r-progress holds still."""
    pasted, held_natively = UNDRIVEN[exercise_id]
    native = check(exercise_id, held_natively, show=False)
    assert not native.passed and "still uses the provided" in native.details[0], native.details
    open_ready(page, base_url, exercise_id)
    page.evaluate(TRACE_PROGRESS_JS)
    lines = press_check(page, pasted)
    expect(page.locator("#verdict")).to_have_attribute("data-passed", "false")
    assert lines[0] == native.details[0]
    before_verdict = {text for text, verdict_lines in page.evaluate("window.__trace") if verdict_lines == 0}
    assert len(before_verdict) <= 1, f"#r-progress moved although nothing drove: {before_verdict}"  # one reset at the start is allowed


@pytest.mark.parametrize("road, named", [
    (None, "names none"),  # a drive record with no road is not given the first one
    ("nowhere", 'names road "nowhere", which roads.json does not have'),
], ids=["omitted", "unknown"])
def test_a_drive_record_with_no_road_or_an_unknown_one_fails_on_load_naming_the_problem_and_offers_retry(page, base_url, road, named):
    """The page names the problem at load (before any run) and offers Retry; it never plays the exercise on a road it was not given."""

    def with_road(route):
        records = route.fetch().json()
        for record in records:
            record.pop("road", None)
            if road is not None:
                record["road"] = road
        route.fulfill(json=records)

    page.route(re.compile(r"/exercises\.json$"), with_road)
    page.goto(exercise_url(base_url, STREAMING[0]))
    wait_state(page, "error")
    expect(page.locator("#status")).to_contain_text(named)
    expect(page.locator("#retry")).to_be_visible()


# ---- stop ---------------------------------------------------------------------------------------------------


def test_stop_halts_while_true_and_page_stays_responsive(page, base_url):
    open_ready(page, base_url, STEER_ID)
    page.evaluate(TICKER_JS)
    set_code(page, FOREVER)
    page.click("#run")
    wait_state(page, "running")
    ticks_at_start = page.evaluate("window.__ticks.count")
    page.wait_for_timeout(1000)  # let the worker spin inside the loop before Stop
    ticks_while_spinning = page.evaluate("window.__ticks.count") - ticks_at_start
    started = time.monotonic()
    page.click("#stop")
    wait_state(page, "ready")
    restart_s = time.monotonic() - started
    max_gap_ms = page.evaluate("window.__ticks.maxGap")
    print(f"restart_s={restart_s:.2f} max_tick_gap_ms={max_gap_ms:.0f}")
    assert ticks_while_spinning > 1, "the page's main thread did not tick while the loop ran"
    if TIMING:
        assert restart_s < STOP_BUDGET_S, f"Stop took {restart_s:.2f} s; spec S-4 wants under {STOP_BUDGET_S} s"
        assert max_gap_ms < TICK_GAP_BUDGET_MS, f"the page froze for {max_gap_ms:.0f} ms while the loop ran"
    assert visited(page)[-3:] == ["running", "loading", "ready"]
    set_code(page, P_STEER)
    page.click("#run")
    wait_state(page, "done")


def test_stop_during_a_normal_run_keeps_the_rows(page, base_url):
    open_ready(page, base_url, STEER_ID)
    set_code(page, P_STEER)
    page.click("#run")
    page.wait_for_function("() => /[1-9]/.test(document.querySelector('#r-progress').textContent)")
    page.click("#stop")
    wait_state(page, "ready")
    assert first_number(page.inner_text("#r-progress")) > 0
    assert int(page.get_attribute("#scrub", "max")) > 0


# ---- learner errors -----------------------------------------------------------------------------------------


def test_learner_exception_reports_class_and_line(page, base_url):
    open_ready(page, base_url, STEER_ID)
    set_code(page, DIVIDES_BY_ZERO)
    page.click("#run")
    wait_state(page, "error")
    status = page.inner_text("#status")
    assert "ZeroDivisionError" in status
    assert f"line {line_of(DIVIDES_BY_ZERO, '1 / 0')}" in status


def test_raise_at_step_50(page, base_url):
    open_ready(page, base_url, STEER_ID)
    page.evaluate("window.__survives = true")  # a reload would clear it
    set_code(page, RAISES_AFTER_50_STEPS)
    page.click("#run")
    wait_state(page, "error")
    assert f"line {line_of(RAISES_AFTER_50_STEPS, 'raise ValueError')}" in page.inner_text("#stdout")
    assert page.locator("#verdict li").count() == 0
    assert first_number(page.inner_text("#r-progress")) > 0
    set_code(page, P_STEER)
    page.click("#run")
    wait_state(page, "done")
    assert page.evaluate("window.__survives") is True


@pytest.mark.parametrize(
    "kind, returned, type_name",
    [("drive", "(0.1, 0.2)", "tuple"), ("drive", "None", "NoneType"), ("drive", '"left"', "str"), ("stats", '"big"', "str")],
    ids=["drive-tuple", "drive-none", "drive-text", "stats-text"],
)
def test_wrong_return_type(page, base_url, kind, returned, type_name):
    record, message = WRONG_RETURN[kind]
    open_ready(page, base_url, record["id"])
    set_code(page, f"def {record['name']}(*args, **kwargs):\n    return {returned}\n")  # one signature fits every kind
    page.click("#run")
    wait_state(page, "error")
    shown = page.inner_text("#stdout")
    assert message in shown
    assert record["id"] in shown  # the shim's `_at_boundary` prefixes the exercise id
    assert type_name in shown
    assert "numpy" not in shown.lower()


def test_storage_blocked(page, base_url, page_errors):
    page.add_init_script(BLOCK_STORAGE)
    open_ready(page, base_url, STEER_ID)
    press_check(page, P_STEER)
    expect(page.locator("#verdict")).to_have_attribute("data-passed", "true")
    expect(page.locator("#status")).to_contain_text("blocked saving")
    assert SAVED_COPY not in page.inner_text("#status"), "the page says the pass is saved although storage is blocked"
    assert page_errors == []


# ---- load failures ------------------------------------------------------------------------------------------


# The editor modules come from the import map in index.html:12-33, confirmed:
#   $ curl -sI https://cdn.jsdelivr.net/npm/@codemirror/view@6.43.14/dist/index.js | head -1
#   HTTP/2 200
def test_editor_import_blocked_falls_back_to_textarea(page, base_url):
    page.route(re.compile(r"/npm/@codemirror/"), lambda route: route.abort())
    open_ready(page, base_url, STEER_ID)
    expect(page.locator("#editor-fallback")).to_be_visible()
    set_code(page, P_STEER)
    page.click("#run")
    wait_state(page, "done")


def abort_then_retry(page, base_url, pattern, named):
    """Abort every request matching `pattern`: the page must say which file failed and offer Retry; Retry must recover."""
    page.route(pattern, lambda route: route.abort())
    page.goto(exercise_url(base_url, STEER_ID))
    wait_state(page, "error")
    expect(page.locator("#status")).to_contain_text(re.compile(named))
    expect(page.locator("#retry")).to_be_visible()
    page.unroute(pattern)
    page.click("#retry")
    wait_state(page, "ready")


# The URL the worker imports (worker.js:9, :85), confirmed:
#   $ curl -sI https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs | head -1
#   HTTP/2 200
# The pattern takes any version, so a Pyodide bump does not turn this test into one that aborts nothing.
def test_route_abort_pyodide_js(page, base_url):
    abort_then_retry(page, base_url, re.compile(r"/pyodide/v[^/]+/full/pyodide\.m?js$"), r"pyodide\.m?js")


@pytest.mark.parametrize("filename", ["bundle.zip", "exercises.json"])
def test_route_abort_names_the_failed_file(page, base_url, filename):
    abort_then_retry(page, base_url, re.compile(rf"/{re.escape(filename)}$"), re.escape(filename))


def dashboard_wheel(name):
    """The route pattern of the wheel `name` that a Dashboard press downloads (worker.js, the `dashboard` handler). Confirmed in
    headless Chromium on 2026-10-11 for matplotlib, whose dependency kiwisolver is a wheel of its own:
      https://cdn.jsdelivr.net/pyodide/v314.0.7/full/matplotlib-3.10.8-cp314-cp314-pyemscripten_2026_0_wasm32.whl
    The pattern takes any version and any wheel tag."""
    return re.compile(rf"/pyodide/v[^/]+/full/{name}-[^/]+\.whl$")


def block_the_dashboard_wheel_then_retry(page, base_url, name):
    """Spec section 6: a charting library that fails to load is reported by file name, the run behind it stands, the rest of the
    page keeps working, and the next Dashboard press fetches the wheel again without a reload."""
    open_ready(page, base_url, STEER_ID)
    set_code(page, P_STEER)
    page.click("#run")
    wait_state(page, "done")
    readouts = freeze_playhead(page)
    page.route(dashboard_wheel(name), lambda route: route.abort())
    page.click("#dashboard")
    wait_state(page, "error")
    for surface in ("#status", "#stdout", "#dashboard-error"):  # the three places Task 11's `error` handler writes the learner-facing line
        shown = page.inner_text(surface)
        assert re.search(rf"{name}-\S+\.whl", shown), f"{surface}: {shown!r}"
    expect(page.locator("#dashboard-error")).to_be_visible()
    expect(page.locator("#dashboard-img")).to_be_hidden()
    expect(page.locator("#dashboard")).to_be_enabled()  # the run behind it stands: pressing again retries
    expect(page.locator("#traceback")).to_be_hidden()  # a failed download is not a failed run
    assert [page.inner_text(readout) for readout in READOUTS] == readouts, "the failed Dashboard changed the run's readouts"
    page.unroute(dashboard_wheel(name))
    page.click("#dashboard")  # the failed download left nothing half-loaded: this press fetches the wheel again
    expect(page.locator("#dashboard-img")).to_be_visible(timeout=120_000)
    expect(page.locator("#dashboard-error")).to_be_hidden()
    page.click("#run")  # the page is not left busy: a new run completes, and the old picture does not stay under it
    wait_state(page, "done")
    expect(page.locator("#dashboard-img")).to_be_hidden()
    expect(page.locator("#dashboard")).to_be_enabled()


def test_route_abort_names_the_matplotlib_wheel(page, base_url):
    block_the_dashboard_wheel_then_retry(page, base_url, "matplotlib")


def test_route_abort_names_a_dependency_wheel_of_matplotlib(page, base_url):
    """A dependency is loaded by the same press: the worker looks at what is loaded, not at the package it was asked for."""
    block_the_dashboard_wheel_then_retry(page, base_url, "kiwisolver")


# ---- phone width --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("record", ONE_PER_KIND, ids=lambda record: record["kind"])
def test_no_horizontal_scroll_at_375(page, base_url, record):
    page.set_viewport_size({"width": 375, "height": 700})
    open_ready(page, base_url, record["id"])
    assert scroll_width(page) <= 375
    set_code(page, pasteable(record["unit_id"]))
    page.click("#run")
    wait_state(page, "done")
    assert scroll_width(page) <= 375


def test_no_horizontal_scroll_with_a_failing_verdict(page, base_url):
    """Failing verdict lines are the longest text the page shows."""
    page.set_viewport_size({"width": 375, "height": 700})
    open_ready(page, base_url, STEER_ID)
    press_check(page, RETURN_ZERO)
    assert scroll_width(page) <= 375


# ---- the worker's protocol, driven directly -----------------------------------------------------------------
# The page never sends these: it disables Dashboard until a run completes, sends only the kinds it knows, and always sends
# init first. The worker answers them all the same, and the page relies on it.


def start_worker(page, base_url, init=True):
    """Start the page's worker by hand on a same-origin document (no page, no editor). With `init`, wait for its `ready`."""
    page.goto(f"{base_url}exercises.json")
    page.evaluate(WORKER_JS, init)
    if init:
        reply(page, "ready", 0, timeout=90_000)


def send(page, message):
    """Post `message` to the worker; returns how many messages had arrived before it."""
    seen = page.evaluate("window.messages.length")
    page.evaluate("message => window.worker.postMessage(message)", message)
    return seen


def reply(page, kind, since, timeout=120_000):
    """The first message after index `since` that is not progress, a row or output, which must be a `kind`: a reply of any
    other type (an error where a result was due, a result where an error was due) fails at once, by name."""
    handle = page.wait_for_function(
        "([since]) => window.messages.slice(since).find((m) => !['progress', 'row', 'stdout'].includes(m.type))",
        arg=[since], timeout=timeout)
    message = handle.json_value()
    assert message["type"] == kind, message
    return message


def exchange(page, message):
    """Every reply to `message`, in order. One worker answers in order, so a request it has no handler for, sent right behind
    `message`, has its error arrive after every reply `message` makes: that error closes the list, with no time to wait out."""
    since = page.evaluate("window.messages.length")
    page.evaluate("([message, sentinel]) => { window.worker.postMessage(message); window.worker.postMessage(sentinel); }", [message, SENTINEL])
    page.wait_for_function(
        "([since, sentinel]) => window.messages.slice(since).some((m) => m.type === 'error' && m.request === sentinel.type)",
        arg=[since, SENTINEL], timeout=120_000)
    replies = page.evaluate("since => window.messages.slice(since)", since)
    return replies[: next(i for i, m in enumerate(replies) if m.get("request") == SENTINEL["type"])]


def test_worker_dashboard_needs_a_completed_run(page, base_url):
    """Spec 5.2: the dashboard draws the last completed run. With no run, or after a run that raised, it is an `error` naming the dashboard."""
    start_worker(page, base_url)
    steer = BY_ID[STEER_ID]
    drive = {"type": "run", "exercise": steer, "road": steer["road"]}
    error = reply(page, "error", send(page, {"type": "dashboard"}), timeout=180_000)  # (b) no run yet; this press also loads matplotlib
    assert error["request"] == "dashboard" and "run the car first" in error["message"], error
    reply(page, "done", send(page, {**drive, "code": P_STEER}))
    png = reply(page, "dashboard", send(page, {"type": "dashboard"}), timeout=180_000)  # (a) a completed run
    assert png["png_magic"] == PNG_MAGIC, png
    error = reply(page, "error", send(page, {**drive, "code": DIVIDES_BY_ZERO}))
    assert error["request"] == "run" and "ZeroDivisionError" in error["message"], error
    error = reply(page, "error", send(page, {"type": "dashboard"}), timeout=180_000)  # (c) the failed run cleared the earlier success
    assert error["request"] == "dashboard" and "run the car first" in error["message"], error


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["id"])
def test_worker_answers_a_learner_that_is_neither_function_nor_car_with_one_reply(page, base_url, shim, record):
    """`name = None` is what a starter holds; `name = 5` is any other value. A run ends in exactly one `done` or one `error`,
    the one the Mac's shim gives (a 0.1.3.x `None` is Car's own "use the provided box": it runs); a check ends in exactly one
    failing `verdict`: never two replies, never none."""
    start_worker(page, base_url)
    for value, type_name in (("None", "NoneType"), ("5", "int")):
        code = f"{record['name']} = {value}\n"
        native = native_error(shim, record, code)
        replies = exchange(page, {"type": RUN_MESSAGE[record["kind"]], "code": code, "exercise": record, "road": record["road"]})
        ended = [r for r in replies if r["type"] in {"done", "error"}]
        assert ended == replies[-1:], f"{value}: not exactly one closing reply, or not last: {[r['type'] for r in replies][-3:]}"
        if native is None:
            assert ended[0]["type"] == "done", ended
        else:
            assert ended[0]["type"] == "error" and ended[0]["request"] == RUN_MESSAGE[record["kind"]], ended
            assert ended[0]["message"] == f"{type(native).__name__}: {native}", ended
            assert record["id"] in ended[0]["message"] and type_name in ended[0]["message"], ended
    held = shim.learner_object(f"{record['name']} = None\n", record["name"])
    native = shim.check_code(held, record["id"])
    replies = exchange(page, {"type": "check", "code": f"{record['name']} = None  # your code\n", "exercise": record})
    ended = [r for r in replies if r["type"] in {"verdict", "error"}]
    assert [r["type"] for r in ended] == ["verdict"] and ended == replies[-1:], replies[-3:]
    assert native["passed"] is False and ended[0]["passed"] is False and ended[0]["details"][0] == native["details"][0], (native, ended)


@pytest.mark.parametrize("code, passes", [(P_STEER, True), (RETURN_ZERO, False)], ids=["passing", "failing"])
def test_worker_check_posts_no_grader_stdout(page, base_url, code, passes):
    """The grader's report is the verdict's `details`; nothing of it is printed. The learner's own print still is."""
    start_worker(page, base_url)
    record = BY_ID[STEER_ID]
    replies = exchange(page, {"type": "check", "code": code, "exercise": record})
    assert [r["type"] for r in replies if r["type"] != "row"] == ["verdict"] and replies[-1]["passed"] is passes, replies[-3:]
    assert not [r for r in replies if r["type"] == "stdout"], "the grader printed its report"
    replies = exchange(page, {"type": "check", "code": "print('mine')\n" + code, "exercise": record})
    assert [r["text"] for r in replies if r["type"] == "stdout"] == ["mine"], replies


@pytest.mark.parametrize("message_type", ["nonsense", "constructor", "toString", "__proto__"])
def test_worker_names_a_message_type_it_has_no_handler_for(page, base_url, message_type):
    """An inherited name (`constructor`, `toString`, `__proto__`) is not a handler: it is reported by name like any unknown type."""
    start_worker(page, base_url, init=False)
    error = reply(page, "error", send(page, {"type": message_type}), timeout=15_000)
    assert error["request"] == message_type and message_type in error["message"], error


def test_worker_refuses_a_request_before_init(page, base_url):
    start_worker(page, base_url, init=False)
    steer = BY_ID[STEER_ID]
    error = reply(page, "error", send(page, {"type": "run", "code": P_STEER, "exercise": steer, "road": steer["road"]}), timeout=15_000)
    assert error["request"] == "run" and 'Send "init" and wait for "ready" first.' in error["message"], error
