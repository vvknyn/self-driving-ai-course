import ast
import dataclasses
import json

import numpy as np
import pytest
from _course import REPO, load_yaml, parse_lab, syllabus_units

from zero2fsd.car import Car, provided, run
from zero2fsd.car.runner import COLUMNS
from zero2fsd.grade import _drive as grade_drive
from zero2fsd.grade._report import exercise_label
from zero2fsd.grade.units.u0_1_2 import NOISE_SIGMA
from zero2fsd.palette import SCENE
from zero2fsd.sim.camera import PALETTE

NOT_IN_BROWSER = {"plt", "dashboard", "check", "practice", "ends"}  # notebook-only names: the page has its own buttons

P_STEER = """\
def p_steer(est, k_off, k_head):
    if not est.valid:
        return 0.0
    return float(np.clip(-(k_off * est.offset_m + k_head * est.heading_rad), -MAX_STEER, MAX_STEER))
"""
MY_CAR = """\
def estimate_lane(obs):
    return provided.estimate_lane(obs)


def my_controller(est, obs):
    return steer_p(est, K_OFF, K_HEAD)


my_car = Car(perception=estimate_lane, controller=my_controller)
"""
LANE_ERROR_STATS = """\
def lane_error_stats(lat_err):
    return {"mean_abs": np.abs(lat_err).mean(), "max_abs": np.abs(lat_err).max(), "rms": np.sqrt(np.mean(lat_err**2))}
"""
COUNT_STEPS = "def steps_off_lane(lat_err):\n    return len(lat_err)\n"
PAINT_MASKS = "def paint_masks(frame):\n    return provided.paint_masks(frame)\n"
SWAPPED_MASKS = "def paint_masks(frame):\n    yellow, white = provided.paint_masks(frame)\n    return white, yellow\n"
NO_MASKS = "def paint_masks(frame):\n    return np.zeros(frame.shape[:2], bool), np.zeros(frame.shape[:2], bool)\n"


def prelude_names(shim):
    ns = {}
    exec(shim.PRELUDE, ns)
    return {name for name in ns if not name.startswith("__")}


def lab_cells():
    """(unit entry, unit.yaml, parsed lab cells) for every ready unit of the real syllabus."""
    for entry in syllabus_units(REPO, "ready"):
        folder = REPO / "course" / entry["folder"]
        yield entry, load_yaml(folder / "unit.yaml"), parse_lab((folder / "lab.py").read_text(encoding="utf-8"))


def bound_names(source):
    """Names a module-level code cell binds: imports, defs, classes and assignments."""
    names = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)}
    return names


def test_prelude_binds_every_name_the_lab_setup_cells_bind(shim):
    lab_names = {n for _, _, cells in lab_cells() for c in cells if "setup" in c.tags for n in bound_names(c.source)}
    assert lab_names - NOT_IN_BROWSER <= prelude_names(shim)
    assert NOT_IN_BROWSER.isdisjoint(prelude_names(shim))


def test_learner_object_returns_what_the_learner_defined(shim):
    assert shim.learner_object("def f(x):\n    return x + 1\n", "f")(1) == 2
    assert shim.learner_object("def estimate_lane(obs):\n    return 7\n", "estimate_lane")(None) == 7  # a redefinition counts


def test_a_provided_box_or_a_missing_name_is_not_an_answer(shim):
    with pytest.raises(NameError, match="define `p_steer`"):
        shim.learner_object("x = 1", "p_steer")
    for name in sorted(prelude_names(shim)):  # the class: any name the prelude binds, left untouched
        with pytest.raises(NameError, match=f"define `{name}`"):
            shim.learner_object("pass", name)


def test_format_error_reports_the_learner_line_and_nothing_else(shim):
    fn = shim.learner_object("def p_steer(est, k_off, k_head):\n    raise ValueError('boom')\n    return 0.0\n", "p_steer")
    with pytest.raises(ValueError) as caught:
        fn(None, 0.35, 2.7)
    info = shim.format_error(caught.value)
    assert info["line"] == 2 and info["message"] == "ValueError: boom" and "<learner>" in info["traceback"]
    nested = shim.learner_object("def helper():\n    raise ValueError('deep')\n\n\ndef p_steer(est, k_off, k_head):\n    return helper()\n", "p_steer")
    with pytest.raises(ValueError) as inner:
        nested(None, 0.35, 2.7)
    assert shim.format_error(inner.value)["line"] == 2  # the innermost learner frame, not the call site on line 6
    with pytest.raises(NameError) as elsewhere:
        shim.learner_object("pass", "p_steer")
    assert shim.format_error(elsewhere.value)["line"] is None
    with pytest.raises(SyntaxError) as syntax:
        shim.learner_object("x = 1\ndef f(:\n", "f")
    assert shim.format_error(syntax.value)["line"] == 2


@pytest.mark.parametrize("exc, message", [
    (ValueError("boom"), "ValueError: boom"),
    (NotImplementedError(), "NotImplementedError"),  # an empty message leaves no dangling colon for the page to pad
    (AssertionError(""), "AssertionError"),
    (KeyError("k"), "KeyError: 'k'"),
])
def test_format_error_joins_only_the_non_empty_parts(shim, exc, message):
    assert shim.format_error(exc)["message"] == message


def test_row_dict_is_plain_json(shim):
    rows = []
    run(Car(), "straight", max_steps=3, on_step=rows.append)
    for row in rows:
        plain = shim.row_dict(row)
        assert tuple(plain) == COLUMNS
        assert {type(v) for v in plain.values()} <= {float, int, bool}
        json.dumps(plain)
    with pytest.raises(ValueError):
        shim.row_dict(rows[0][:-1])


def test_summary_of_a_complete_run_and_of_a_run_that_leaves_the_road(shim):
    tel = run(Car(), "straight", max_steps=40)
    kept = shim.summary(tel)
    assert set(kept) == {"completed", "steps", "mean_abs_lat", "max_abs_lat", "progress_m", "off_road"}
    assert kept["steps"] == 40 and kept["completed"] is False and kept["off_road"] is False
    assert kept["progress_m"] == pytest.approx(tel.progress_m[-1]) and kept["progress_m"] > tel.progress_m[0]
    assert kept["mean_abs_lat"] == pytest.approx(np.abs(tel.lat).mean()) and kept["max_abs_lat"] == pytest.approx(np.abs(tel.lat).max())
    left = shim.summary(run(Car(controller=lambda est, obs: 0.5), "straight"))
    assert left["off_road"] is True and left["completed"] is False and left["steps"] < 200
    json.dumps([kept, left])


@pytest.mark.parametrize("road", ["straight", "gentle"])
def test_preview_run_streams_one_row_per_step(shim, road):
    rows = []
    done = shim.preview_run(shim.learner_object(P_STEER, "p_steer"), "0.1.4.a", road, rows.append)
    assert len(rows) == done["steps"] == len(shim.LAST_RUN.t) and done["completed"] is True
    assert rows[0]["t"] == 0.0 and tuple(rows[0]) == COLUMNS


def test_a_learner_that_raises_mid_run_stops_the_run_and_the_next_run_works(shim):
    code = "def p_steer(est, k_off, k_head, _n=[0]):\n    _n[0] += 1\n    if _n[0] > 20:\n        raise ValueError('boom')\n    return 0.0\n"
    good, rows = shim.learner_object(P_STEER, "p_steer"), []
    shim.preview_run(good, "0.1.4.a", "straight", rows.append)
    assert shim.LAST_RUN is not None
    rows.clear()
    with pytest.raises(ValueError) as caught:
        shim.preview_run(shim.learner_object(code, "p_steer"), "0.1.4.a", "straight", rows.append)
    assert len(rows) == 20 and shim.format_error(caught.value)["line"] == 4 and shim.LAST_RUN is None
    assert shim.preview_run(good, "0.1.4.a", "straight", rows.append)["completed"]


@pytest.mark.parametrize("exercise, code, name, expected, got", [
    ("0.1.4.a", "def p_steer(est, k_off, k_head):\n    return (0.1, 0.2)\n", "p_steer", "must return one number", "got tuple"),
    ("0.1.4.a", "def p_steer(est, k_off, k_head):\n    return None\n", "p_steer", "must return one number", "got NoneType"),
    ("0.1.4.a", "def p_steer(est, k_off, k_head):\n    return 'left'\n", "p_steer", "must return one number", "got str"),
    ("0.1.3.a", "def estimate_lane(obs):\n    return (0.0, 0.0, True)\n", "estimate_lane", "must return a LaneEstimate", "got tuple"),
    ("0.1.4.b", "my_car = 5\n", "my_car", "must be a Car", "got int"),
])
def test_a_wrong_return_type_is_a_plain_error_naming_the_exercise(shim, exercise, code, name, expected, got):
    with pytest.raises(TypeError) as caught:
        shim.preview_run(shim.learner_object(code, name), exercise, "straight", lambda row: None)
    message = str(caught.value)
    assert message.startswith(f"{exercise_label(exercise)}: ") and expected in message and got in message and "numpy" not in message.lower()


def test_the_learners_own_type_error_keeps_its_own_message_and_line(shim):
    fn = shim.learner_object("def p_steer(est, k_off, k_head):\n    return 'a' + 1\n", "p_steer")
    with pytest.raises(TypeError) as caught:
        shim.preview_run(fn, "0.1.4.a", "straight", lambda row: None)
    assert "0.1.4.a" not in str(caught.value) and shim.format_error(caught.value)["line"] == 2


def test_a_host_callback_failure_on_preview_run_is_the_hosts_error_not_the_learners(shim):
    def on_row(row):
        raise RuntimeError("host callback failed")

    with pytest.raises(shim.HostCallbackError, match="host callback failed") as caught:
        shim.preview_run(shim.learner_object(P_STEER, "p_steer"), "0.1.4.a", "straight", on_row)
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_a_host_callback_failure_on_check_code_is_raised_once_after_the_run_and_never_blamed_on_the_learner(shim):
    seen = []

    def on_row(row):
        seen.append(row)
        raise TypeError("host callback failed")  # a TypeError is what _at_boundary would otherwise relabel

    with pytest.raises(shim.HostCallbackError) as caught:
        shim.check_code(shim.learner_object(MY_CAR, "my_car"), "0.1.4.b", on_row=on_row)
    assert len(seen) == 1, "the hook goes quiet after the first failure"
    assert isinstance(caught.value.__cause__, TypeError)
    assert str(caught.value).startswith("the playground's row callback failed")  # the host's own sentence, not a learner-facing one


CALLS = {  # the kind of exercise a call is for -> a call of it that never gets as far as the learner's code
    "drive": lambda shim, exercise: shim.preview_run(lambda: None, exercise, "straight", lambda row: None),
    "stats": lambda shim, exercise: shim.preview_stats(lambda lat: 0, exercise),
    "image": lambda shim, exercise: shim.preview_image(lambda frame: None, exercise),
}
EXERCISE_OF = {"drive": "0.1.4.a", "stats": "0.1.1.a", "image": "0.1.2.a"}


@pytest.mark.parametrize("call, kind", [(call, kind) for call in CALLS for kind in EXERCISE_OF if kind != call])
def test_a_call_for_one_kind_refuses_an_exercise_of_another(shim, call, kind):
    with pytest.raises(ValueError, match=f"{EXERCISE_OF[kind]} is a {kind} exercise; this call is for {call} exercises"):
        CALLS[call](shim, EXERCISE_OF[kind])


@pytest.mark.parametrize("call", list(CALLS))
def test_an_unknown_exercise_id_is_a_plain_error_listing_the_valid_ids(shim, call):
    with pytest.raises(ValueError, match=r"unknown exercise '9\.9\.9\.z'; known: .*0\.1\.4\.a"):
        CALLS[call](shim, "9.9.9.z")


def test_preview_stats_summarises_the_canned_drive_by_label(shim):
    lat = np.asarray(shim.canned_telemetry(shim.get("0.1.1.a").road).lat)
    stats = shim.preview_stats(shim.learner_object(LANE_ERROR_STATS, "lane_error_stats"), "0.1.1.a")
    assert stats == pytest.approx({"mean_abs": np.abs(lat).mean(), "max_abs": np.abs(lat).max(), "rms": np.sqrt(np.mean(lat**2))})
    counter = shim.learner_object("def steps_off_lane(lat_err, threshold=0.9):\n    return int((np.abs(lat_err) > threshold).sum())\n", "steps_off_lane")
    assert list(shim.preview_stats(counter, "0.1.1.b").values()) == [float((np.abs(lat) > 0.9).sum())]


def test_stats_run_on_the_road_the_registry_names_and_share_one_drive_per_road(shim, monkeypatch):
    shim.canned_telemetry.cache_clear()
    road, roads, real_run = shim.get("0.1.1.a").road, [], shim.run
    assert shim.get("0.1.1.b").road == road
    monkeypatch.setattr(shim, "run", lambda car, scenario: roads.append(scenario) or real_run(car, scenario))
    steps = shim.learner_object(COUNT_STEPS, "steps_off_lane")
    for exercise in ("0.1.1.a", "0.1.1.b"):
        assert list(shim.preview_stats(steps, exercise).values()) == [float(len(real_run(Car(), road).t))]
    assert roads == [road]  # two exercises, one road: one drive


def test_a_stats_exercise_on_another_road_gets_that_roads_drive(shim, monkeypatch):
    real_get, straight = shim.get, len(run(Car(), "straight").t)
    assert straight != len(run(Car(), real_get("0.1.1.b").road).t)  # the two roads give different drives
    monkeypatch.setattr(shim, "get", lambda exercise_id: dataclasses.replace(real_get(exercise_id), road="straight"))
    assert shim.preview_stats(shim.learner_object(COUNT_STEPS, "steps_off_lane"), "0.1.1.b") == {"steps_off_lane": float(straight)}


@pytest.mark.parametrize("exercise, code, name, what, got", [
    ("0.1.1.a", "def lane_error_stats(lat_err):\n    return 'oops'\n", "lane_error_stats", "a number or a dict of numbers", "got str"),
    ("0.1.1.b", "def steps_off_lane(lat_err, threshold=0.9):\n    return '3'\n", "steps_off_lane", "a number or a dict of numbers", "got str"),
    ("0.1.1.a", "def lane_error_stats(lat_err):\n    return lat_err[:3]\n", "lane_error_stats", "a number or a dict of numbers", "got ndarray"),
    ("0.1.1.b", "def steps_off_lane(lat_err, threshold=0.9):\n    pass\n", "steps_off_lane", "a number or a dict of numbers", "got NoneType"),
    ("0.1.1.a", "def lane_error_stats(lat_err):\n    return {'mean': 'high'}\n", "lane_error_stats", "a number for 'mean'", "got str"),
    ("0.1.1.b", "def steps_off_lane(lat_err, threshold=0.9):\n    return True\n", "steps_off_lane", "a number or a dict of numbers", "got bool"),
])
def test_a_stats_function_returning_the_wrong_type_is_a_plain_error(shim, exercise, code, name, what, got):
    with pytest.raises(TypeError) as caught:
        shim.preview_stats(shim.learner_object(code, name), exercise)
    assert str(caught.value) == f"{exercise_label(exercise)}: expected {what}, {got}"


def test_a_stats_function_that_edits_its_input_leaves_the_shared_drive_alone(shim):
    road = shim.get("0.1.1.b").road
    before = shim.canned_telemetry(road).lat.copy()
    wipe = shim.learner_object("def steps_off_lane(lat_err):\n    lat_err[:] = 99.0\n    return 0\n", "steps_off_lane")
    assert shim.preview_stats(wipe, "0.1.1.b") == {"steps_off_lane": 0.0}
    assert np.array_equal(shim.canned_telemetry(road).lat, before) and shim.canned_telemetry(road) is shim.canned_telemetry(road)


def scene_rgb(name):
    return tuple(int(SCENE[name][i:i + 2], 16) for i in (1, 3, 5))


def picture(shim, code, seed=0):
    """(frame, overlay) arrays of the image preview of 0.1.2.a for the learner's `paint_masks` in `code`."""
    shown = shim.preview_image(shim.learner_object(code, "paint_masks"), "0.1.2.a", seed)
    shape = (shown["height"], shown["width"], 3)
    return tuple(np.frombuffer(shown[key], np.uint8).reshape(shape) for key in ("frame", "overlay"))


@pytest.mark.parametrize("seed", [0, 3])
def test_the_image_frame_is_what_the_car_sees_at_the_start_of_the_straight_road(shim, seed):
    seen = []

    def perception(obs):
        seen.append(obs.frame.copy())
        return provided.estimate_lane(obs)

    run(Car(perception=perception), "straight", max_steps=1, seed=seed, noise_sigma=NOISE_SIGMA)
    frame, overlay = picture(shim, PAINT_MASKS, seed)
    assert frame.shape == overlay.shape == (180, 320, 3) and np.array_equal(frame, seen[0])


def test_the_same_seed_gives_the_same_picture_and_another_seed_other_noise(shim):
    first, again, other = (picture(shim, PAINT_MASKS, seed)[0] for seed in (0, 0, 1))
    assert np.array_equal(first, again) and not np.array_equal(first, other)


@pytest.mark.parametrize("code, order", [(PAINT_MASKS, 1), (SWAPPED_MASKS, -1)], ids=["as returned", "yellow and white swapped"])
def test_the_overlay_paints_each_mask_in_the_scene_colour_of_its_paint(shim, code, order):
    frame, overlay = picture(shim, code)
    first, second = provided.paint_masks(frame.copy())[::order]  # the masks the learner's function returned, in order
    assert first.any() and second.any()
    expected = np.zeros_like(frame)
    expected[first], expected[second] = scene_rgb("yellow"), scene_rgb("white")
    assert np.array_equal(overlay, expected)


def test_a_function_that_marks_nothing_gives_a_black_overlay(shim):
    assert not picture(shim, NO_MASKS)[1].any()


def test_no_camera_colour_is_near_the_markers_the_registry_paints_masks_in(shim):
    nearest = min(np.abs(np.array(marker) - colour).max() for marker in shim.OVERLAY.values() for colour in PALETTE)
    assert nearest > 5 * NOISE_SIGMA  # so reading a mask back from the painted frame cannot pick up a scenery pixel


@pytest.mark.parametrize("code, expected", [
    ("def paint_masks(frame):\n    return 'oops'\n", "returned str, expected (yellow, white)"),
    ("def paint_masks(frame):\n    return provided.paint_masks(frame)[0]\n", "returned ndarray, expected (yellow, white)"),
    ("def paint_masks(frame):\n    return frame[:, :, 0], frame[:, :, 1]\n", "the yellow mask must be a bool array of shape (180, 320)"),
])
def test_a_wrong_mask_answer_is_a_plain_error_naming_the_exercise(shim, code, expected):
    with pytest.raises(TypeError) as caught:
        shim.preview_image(shim.learner_object(code, "paint_masks"), "0.1.2.a")
    assert str(caught.value) == f"Exercise A (Images are arrays): {expected}"


def test_an_error_in_the_mask_function_keeps_the_learners_own_line(shim):
    fn = shim.learner_object("def paint_masks(frame):\n    return frame.nothing\n", "paint_masks")
    with pytest.raises(AttributeError) as caught:
        shim.preview_image(fn, "0.1.2.a")
    assert shim.format_error(caught.value)["line"] == 2


def test_check_code_gives_the_verdict_and_streams_rows_only_for_drive_checks(shim):
    ok = shim.check_code(shim.learner_object(P_STEER, "p_steer"), "0.1.4.a")
    bad = shim.check_code(shim.learner_object("def p_steer(est, k_off, k_head):\n    return 0.0\n", "p_steer"), "0.1.4.a")
    assert ok["passed"] is True and bad["passed"] is False
    assert all(isinstance(line, str) for line in ok["details"] + bad["details"]) and bad["details"]
    assert shim.check_code(shim.learner_object(LANE_ERROR_STATS, "lane_error_stats"), "0.1.1.a")["passed"] is True
    drive_rows, stats_rows = [], []
    assert shim.check_code(shim.learner_object(MY_CAR, "my_car"), "0.1.4.b", on_row=drive_rows.append)["passed"] is True
    shim.check_code(shim.learner_object(LANE_ERROR_STATS, "lane_error_stats"), "0.1.1.a", on_row=stats_rows.append)
    assert len(drive_rows) > 500 and tuple(drive_rows[0]) == COLUMNS and stats_rows == []
    refused_rows = []
    refused = shim.check_code(Car(), "0.1.4.b", on_row=refused_rows.append)  # the provided boxes: refused before any drive
    assert refused["passed"] is False and "still uses the provided" in refused["details"][0] and refused_rows == []


@pytest.mark.parametrize("code, name, exercise_id, passed", [
    (MY_CAR, "my_car", "0.1.4.b", True),
    ("def p_steer(est, k_off, k_head):\n    return 0.0\n", "p_steer", "0.1.4.a", False),
], ids=["passing", "failing"])
def test_check_code_prints_nothing(shim, capsys, code, name, exercise_id, passed):
    """Colab shows the grader's printed report; in the browser the same lines travel as the verdict's details."""
    verdict = shim.check_code(shim.learner_object(code, name), exercise_id)
    assert verdict["passed"] is passed and verdict["details"]
    assert capsys.readouterr().out == ""


def test_check_code_draws_no_figure(shim, monkeypatch):
    drawn = []
    monkeypatch.setattr(grade_drive, "show_figure", drawn.append)  # what a check does with `show=True`: matplotlib has no window here
    assert shim.check_code(shim.learner_object(MY_CAR, "my_car"), "0.1.4.b")["passed"] is True
    assert drawn == []


@pytest.mark.parametrize("seed, noise_sigma", [(0, 0.0), (7, 6.0)])
def test_frame_rgb_is_the_frame_the_car_saw(shim, seed, noise_sigma):
    seen = []

    def perception(obs):
        seen.append(obs.frame.copy())
        return provided.estimate_lane(obs)

    tel = run(Car(perception=perception), "gentle", max_steps=6, seed=seed, noise_sigma=noise_sigma)
    frames = [shim.frame_rgb("gentle", k, tel.x[k], tel.y[k], tel.yaw[k], tel.speed[k], seed, noise_sigma) for k in (0, 5)]
    assert [len(f) for f in frames] == [320 * 180 * 3] * 2 and frames == [seen[0].tobytes(), seen[5].tobytes()]
    assert frames[0] != frames[1]


def test_dashboard_png_shows_the_last_completed_run(shim):
    shim.LAST_RUN = None
    with pytest.raises(RuntimeError, match="run the car first"):
        shim.dashboard_png()
    shim.preview_run(shim.learner_object(P_STEER, "p_steer"), "0.1.4.a", "straight", lambda row: None)
    assert shim.dashboard_png().startswith(b"\x89PNG\r\n\x1a\n")


def test_probe_steps_per_s_is_a_positive_rate(shim):
    assert shim.probe_steps_per_s(30) > 0
