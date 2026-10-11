"""Browser helpers shared by tests/playground/test_browser.py, scripts/measure_phone.py and the manual acceptance steps."""
import inspect
import json
import re
import sys
import threading
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote
from urllib.request import urlopen

from playwright.sync_api import expect

from _course import REPO

sys.path.insert(0, str(REPO))  # `python scripts/<name>.py` puts scripts/ first, not the repo root

from zero2fsd.car import provided  # noqa: E402

# The provided steering law under the exercise's function name: learner code that passes 0.1.4.a.
P_STEER = inspect.getsource(provided.steer_p).replace("def steer_p(", "def p_steer(", 1)
# CodeMirror mounted, or the textarea fallback shown: either way the learner can type.
EDITOR_READY = ("() => !!document.querySelector('#editor .cm-content') || "
                "((f) => f !== null && f.offsetParent !== null)(document.querySelector('#editor-fallback'))")
LOADED_SOLUTION = re.compile(r'^(?P<name>\w+) = load_solution\([^\n]*?"(?P<unit>[\d.]+)"\)\.(?P<attr>\w+)$', re.M)


def pasteable(unit_id: str) -> str:
    """A reference solution as one cell a learner could paste: the sibling units it loads by path are inlined above it."""
    source = (REPO / "solutions" / "course" / unit_id / "solution.py").read_text()
    siblings = [m["unit"] for m in LOADED_SOLUTION.finditer(source)]
    source = LOADED_SOLUTION.sub(r"\g<name> = \g<attr>", source).replace("from _course import load_solution\n", "")
    assert "load_solution" not in source, f"solutions/course/{unit_id}/solution.py loads something this helper cannot inline"
    return "".join(pasteable(unit) + "\n" for unit in siblings) + source


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@contextmanager
def serve(directory):
    """Serve `directory` on a free localhost port; yields the base URL (with a trailing slash)."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_Quiet, directory=str(directory)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()


def exercise_url(base_url, exercise_id, query=""):
    """The canonical page URL. Next serves public/ by exact path, so the directory URL does not work on the site."""
    return f"{base_url}index.html?exercise={quote(exercise_id, safe='')}{query}"


def paint_colours(base_url):
    """The two mask paints as [r, g, b] lists: palette.json scene.yellow, then scene.white (what the shim paints and the page draws)."""
    with urlopen(f"{base_url}palette.json") as response:
        scene = json.load(response)["scene"]
    return [[int(scene[name][i:i + 2], 16) for i in (1, 3, 5)] for name in ("yellow", "white")]


def wait_state(page, state, timeout_ms=60_000):
    expect(page.locator("#status")).to_have_attribute("data-state", state, timeout=timeout_ms)


def open_ready(page, base_url, exercise_id, query=""):
    """Open an exercise and wait until the runtime is ready and the editor is usable."""
    page.goto(exercise_url(base_url, exercise_id, query))
    wait_state(page, "ready")
    page.wait_for_function(EDITOR_READY)


def set_code(page, code):
    """Replace the editor's text: the textarea when it is the fallback, else CodeMirror through window.__setCode."""
    if page.locator("#editor-fallback").is_visible():
        page.fill("#editor-fallback", code)
    else:
        page.evaluate("text => window.__setCode(text)", code)
