import ast
import dataclasses
import math
import re
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

import zero2fsd.grade as grade_pkg
from _course import REPO, load_solution, parse_lab
from zero2fsd import geom
from zero2fsd.car import Car, LaneEstimate, provided, run
from zero2fsd.car.runner import COLUMNS
from zero2fsd.grade import _drive, check
from zero2fsd.grade._drive import GATE_SCENARIO
from zero2fsd.grade._poses import LOOKAHEAD, sample_poses
from zero2fsd.grade._report import EXERCISES, LESSON_TITLES, NOT_IMPLEMENTED, REPORT, Failure, _lines, call, exercise_label, fmt, fmt_gap, fmt_pair, lesson_titles, matching_mistakes
from zero2fsd.sim.dynamics import MAX_STEER

IDS = [e["id"] for p in sorted(Path("course").glob("*/unit.yaml")) for e in yaml.safe_load(p.read_text())["exercises"]]  # every unit.yaml, so a new unit adds no line here


def as_target(exercise_id, fn):
    """The object an exercise grades: a Car for the all-yours exercise, the (world_to_ego, mount) pair for 1.1.5.b, else the function itself."""
    if exercise_id == "1.1.5.b":
        return (geom.world_to_ego, fn)
    return Car(perception=fn, controller=lambda est, obs: fn()) if exercise_id == "0.1.4.b" else fn


def text(result):
    return "\n".join(result.details)


# --- what the learner reads never contains a dotted id ------------------------------------------------------------

def test_lesson_titles_are_the_unit_yaml_titles():
    units = {y["id"]: y["title"] for y in (yaml.safe_load(p.read_text()) for p in Path("course").glob("*/unit.yaml"))}
    assert LESSON_TITLES == {unit: units[unit] for unit in LESSON_TITLES}


def test_two_unit_modules_claiming_one_lesson_id_raise():
    unit = lambda *lesson: types.SimpleNamespace(LESSON=lesson)  # noqa: E731
    assert lesson_titles([unit("9.9.1", "One"), unit("9.9.2", "Two")]) == {"9.9.1": "One", "9.9.2": "Two"}
    with pytest.raises(ValueError, match="duplicate lesson id '9.9.1'"):
        lesson_titles([unit("9.9.1", "One"), unit("9.9.1", "Again")])


@pytest.mark.parametrize("exercise_id, label", [
    ("0.1.1.a", "Exercise A (Meet your car)"),
    ("0.1.1.b", "Exercise B (Meet your car)"),
    ("0.1.2.a", "Exercise A (Images are arrays)"),
    ("0.1.3.b", "Exercise B (Your first perception box)"),
    ("0.1.4.a", "Exercise A (Closing the loop)"),
])
def test_exercise_label_is_the_suffix_and_the_lesson_title(exercise_id, label):
    assert exercise_label(exercise_id) == label


@pytest.mark.parametrize("exercise_id", IDS)
def test_failing_report_and_hint_name_no_dotted_id(exercise_id):
    result = check(exercise_id, as_target(exercise_id, lambda *a, **k: None), show=False)
    lines = text(result)
    assert exercise_label(exercise_id) in lines and f'lecture "{LESSON_TITLES[exercise_id.rpartition(".")[0]]}"' in lines
    assert not re.search(r"\d\.\d\.\d", lines), lines


@pytest.mark.parametrize("exercise_id", ["0.1.1.a", "0.1.2.a", "0.1.3.a", "0.1.4.a"])
def test_passing_report_names_no_dotted_id(exercise_id):
    assert not re.search(r"\d\.\d\.\d", text(check(exercise_id, GOOD[exercise_id], show=False)))


# --- correct answers, written here independently of the grader and of solutions/ ---------------------------------

def good_lane_error_stats(lat):
    a = np.abs(lat)
    return {"mean_abs": a.mean(), "max_abs": a.max(), "rms": float(np.sqrt(np.mean(lat**2)))}


def good_steps_off_lane(lat, threshold=0.9):
    return int(np.count_nonzero(np.abs(lat) > threshold))


def own_perception(obs):  # a different function object from the provided one, same behaviour
    return provided.estimate_lane(obs)


GOOD = {
    "0.1.1.a": good_lane_error_stats,
    "0.1.1.b": good_steps_off_lane,
    "0.1.2.a": provided.paint_masks,
    "0.1.3.a": provided.estimate_lane,
    "0.1.4.a": provided.steer_p,
}


def test_the_registry_holds_exactly_the_loop_zero_exercises():
    assert sorted(EXERCISES) == IDS


def test_unknown_id_is_a_value_error_that_lists_the_known_ids():
    with pytest.raises(ValueError) as err:
        check("9.9.9.z", lambda x: x)
    assert all(i in str(err.value) for i in IDS)


@pytest.mark.parametrize("exercise_id", IDS)
def test_not_implemented_is_the_red_message(exercise_id, capsys):
    def skeleton(*args, **kwargs):
        raise NotImplementedError

    result = check(exercise_id, as_target(exercise_id, skeleton), show=False)
    assert not result.passed and NOT_IMPLEMENTED in text(result)
    out = capsys.readouterr().out
    assert out.startswith("\033[31m\N{BALLOT X}") and NOT_IMPLEMENTED in out


@pytest.mark.parametrize("exercise_id", IDS)
def test_returning_none_fails(exercise_id):
    assert not check(exercise_id, as_target(exercise_id, lambda *a, **k: None), show=False).passed


@pytest.mark.parametrize("exercise_id", IDS)
def test_a_raising_function_is_reported_with_its_input(exercise_id):
    def broken(*args, **kwargs):
        raise TypeError("unsupported operand")

    result = check(exercise_id, as_target(exercise_id, broken), show=False)
    assert not result.passed
    lines = text(result)
    assert "your code raised TypeError: unsupported operand" in lines and "input:" in lines and "hint:" in lines


@pytest.mark.parametrize("exercise_id", ["0.1.1.a", "0.1.1.b", "0.1.2.a", "0.1.3.a", "0.1.4.a"])
def test_a_correct_answer_passes_and_prints_a_green_summary(exercise_id, capsys):
    result = check(exercise_id, GOOD[exercise_id], show=False)
    assert result.passed, text(result)
    assert capsys.readouterr().out.startswith(f"\033[32m\N{CHECK MARK} {exercise_label(exercise_id)}:")


@pytest.mark.parametrize("before", [True, False])
@pytest.mark.parametrize("report", [True, False])
def test_report_is_restored_after_a_check_that_raises(monkeypatch, before, report):
    def explode(target, show, on_step):
        raise RuntimeError("a bug in the grader")

    monkeypatch.setitem(EXERCISES, "x.y.z", explode)
    token = REPORT.set(before)
    try:
        with pytest.raises(RuntimeError, match="a bug in the grader"):
            check("x.y.z", None, show=False, report=report)
        assert REPORT.get() is before
    finally:
        REPORT.reset(token)


def test_check_report_false_prints_nothing_and_the_default_prints(capsys):
    for exercise_id, target in (("0.1.4.a", GOOD["0.1.4.a"]), ("0.1.3.b", own_perception)):  # a check that only calls, and one that drives
        quiet = check(exercise_id, target, show=False, report=False)
        assert capsys.readouterr().out == "", exercise_id
        loud = check(exercise_id, target, show=False)
        assert quiet.details == loud.details
        assert f"\N{CHECK MARK} {exercise_label(exercise_id)}:" in capsys.readouterr().out


# --- wrong answers: each failure says what was checked, the input, expected, got, and a hint --------------------

def signed_max_stats(lat):
    return {**good_lane_error_stats(lat), "max_abs": float(lat.max())}


def steps_at_or_above(lat, threshold=0.9):
    return int(np.count_nonzero(np.abs(lat) >= threshold))


def default_threshold_only(lat, threshold=0.9):
    return good_steps_off_lane(lat)


def unclipped_p(est, k_off, k_head):
    return -(k_off * est.offset_m + k_head * est.heading_rad) if est.valid else 0.0


def positive_p(est, k_off, k_head):
    return -provided.steer_p(est, k_off, k_head)


def ignores_valid(est, k_off, k_head):
    return float(np.clip(-(k_off * est.offset_m + k_head * est.heading_rad), -MAX_STEER, MAX_STEER))


def flipped_offset(obs):
    est = provided.estimate_lane(obs)
    return LaneEstimate(-est.offset_m, est.heading_rad, est.valid)


def never_valid(obs):
    return LaneEstimate(math.nan, math.nan, False)


@pytest.mark.parametrize(
    "exercise_id, wrong, expect_in_details",
    [
        ("0.1.1.a", signed_max_stats, ["max_abs is wrong", "expected: max_abs =", "got:      max_abs ="]),
        ("0.1.1.b", steps_at_or_above, ["counted", "threshold"]),  # exactly-at-threshold values are not off the lane
        ("0.1.1.b", default_threshold_only, ["threshold=0.5"]),  # a custom threshold must be honoured
        ("0.1.1.a", lambda lat: [1.0, 2.0, 3.0], ["expected a dict"]),
        ("0.1.2.a", lambda f: tuple(m.astype(np.uint8) for m in provided.paint_masks(f)), ["bool array of shape (180, 320)"]),
        ("0.1.2.a", lambda f: provided.paint_masks(f)[::-1], ["IoU"]),  # yellow and white swapped
        ("0.1.2.a", lambda f: provided.paint_masks(f)[0], ["expected (yellow, white)"]),
        ("0.1.3.a", flipped_offset, ["only", "of 40 poses", "first miss"]),
        ("0.1.3.a", never_valid, ["valid=False"]),
        ("0.1.3.a", lambda obs: 0.0, ["expected a LaneEstimate"]),
        ("0.1.4.a", unclipped_p, ["clips"]),
        ("0.1.4.a", positive_p, ["steers"]),
        ("0.1.4.a", ignores_valid, ["no lane seen"]),
        ("0.1.4.a", lambda est, k_off, k_head: "left", ["finite steering angle"]),
    ],
)
def test_a_wrong_answer_fails_with_the_full_explanation(exercise_id, wrong, expect_in_details):
    result = check(exercise_id, wrong, show=False)
    assert not result.passed
    lines = text(result)
    assert all(part in lines for part in expect_in_details), lines
    assert all(label in lines for label in ("checked:", "input:", "hint:")), lines


# --- box labs: never a green from the provided box, the gate decides on the run ----------------------------------

@pytest.mark.parametrize("perception", [None, provided.estimate_lane])
def test_the_perception_gate_never_passes_on_the_provided_box(perception):
    result = check("0.1.3.b", perception, show=False)
    assert not result.passed and "still uses the provided perception" in text(result)


@pytest.mark.parametrize(
    "car, still_provided",
    [
        (Car(), "perception and controller"),
        (Car(perception=own_perception), "controller"),
        (Car(controller=lambda est, obs: provided.steer_p(est)), "perception"),
    ],
)
def test_the_all_yours_car_needs_both_boxes_of_its_own(car, still_provided):
    result = check("0.1.4.b", car, show=False)
    assert not result.passed and f"still uses the provided {still_provided}" in text(result)


def test_the_all_yours_car_must_be_a_car():
    result = check("0.1.4.b", provided.steer_p, show=False)
    assert not result.passed and "must be a Car" in text(result)


def test_a_perception_that_never_sees_the_lane_fails_the_gate_and_the_dashboard_copes_with_nan():
    result = check("0.1.3.b", never_valid, show=True)
    assert not result.passed and "did not pass the gentle gate" in text(result)
    assert "left the lane" in text(result)


def telemetry_that(make_tel, *, completed, mean_lat, off_road=False):
    tel = make_tel(lat=[mean_lat, mean_lat], progress=[0.0, 100.0], scenario="gentle", route_length=100.0,
                   off_road=[False, off_road])
    return dataclasses.replace(tel, completed=completed)


@pytest.mark.parametrize(
    "completed, mean_lat, off_road, verdict",
    [
        (True, 0.30, False, "pass"),
        (True, 0.49, False, "pass"),
        (True, 0.50, False, "mean |lat|"),  # the limit itself is not under the limit
        (True, 0.80, False, "mean |lat|"),
        (False, 0.10, True, "left the lane"),
        (False, 0.10, False, "ran out of time"),
    ],
)
def test_the_gate_decides_on_completion_and_mean_lateral_error(monkeypatch, make_tel, completed, mean_lat, off_road, verdict):
    tel = telemetry_that(make_tel, completed=completed, mean_lat=mean_lat, off_road=off_road)
    monkeypatch.setattr(_drive, "run", lambda car, scenario, on_step=None, camera=None: tel)
    if verdict == "pass":
        assert "completed gentle" in _drive.gate(Car(), show=False)
    else:
        with pytest.raises(_drive.Failure) as err:
            _drive.gate(Car(), show=False)
        assert verdict in f"{err.value.headline} {err.value.shown['got']}"


def test_show_draws_the_dashboard_once_per_run_only_when_asked(monkeypatch, make_tel):
    monkeypatch.setattr(_drive, "run", lambda car, scenario, on_step=None, camera=None: telemetry_that(make_tel, completed=True, mean_lat=0.1))
    shown = []
    monkeypatch.setattr(_drive, "show_figure", shown.append)
    _drive.gate(Car(), show=False)
    assert shown == []
    _drive.gate(Car(), show=True)
    assert len(shown) == 1 and len(shown[0].axes) == 4


def test_a_working_all_yours_car_passes_gentle_and_shows_the_curvy_cliffhanger(capsys):
    car = Car(perception=own_perception, controller=lambda est, obs: provided.steer_p(est))
    result = check("0.1.4.b", car, show=False)
    assert result.passed, text(result)
    assert "curvy (not graded): left the lane after" in text(result)
    out = capsys.readouterr().out
    assert "your car on gentle" in out and "curvy (not graded)" in out


def test_a_working_perception_passes_the_gate_when_placed_in_the_provided_car():
    result = check("0.1.3.b", own_perception, show=False)
    assert result.passed and "completed gentle" in text(result), text(result)


# --- streaming: `on_step` sees every step of the gate drive, and only of a drive ----------------------------------

def own_car():
    return Car(perception=own_perception, controller=lambda est, obs: provided.steer_p(est))


@pytest.mark.parametrize(
    "exercise_id, target, reference",
    [
        ("0.1.3.b", own_perception, lambda: Car(perception=own_perception)),
        ("0.1.4.b", own_car(), own_car),  # the curvy cliffhanger after the gate is not streamed
    ],
)
def test_the_gate_drive_streams_every_step_to_on_step(exercise_id, target, reference):
    rows, ref_rows = [], []
    result = check(exercise_id, target, show=False, on_step=rows.append)
    assert result.passed, text(result)
    run(reference(), GATE_SCENARIO, on_step=ref_rows.append)
    assert ref_rows and rows == ref_rows  # the very rows a plain run of the same car produces, not just as many
    assert all(len(row) == len(COLUMNS) for row in rows)


@pytest.mark.parametrize("exercise_id", ["0.1.1.a", "0.1.1.b", "0.1.2.a", "0.1.3.a", "0.1.4.a"])
def test_a_grader_that_does_not_drive_never_calls_on_step(exercise_id):
    calls = []
    result = check(exercise_id, GOOD[exercise_id], show=False, on_step=calls.append)
    assert result.passed, text(result)
    assert calls == []


def test_a_car_that_fails_before_driving_streams_nothing():
    calls = []
    result = check("0.1.4.b", Car(), show=False, on_step=calls.append)
    assert not result.passed and "still uses the provided" in text(result)
    assert calls == []


# --- the pose sampler and the grader's own hygiene ---------------------------------------------------------------

def test_straight_poses_have_no_turn_in_the_next_lookahead_metres_and_are_seeded():
    poses = sample_poses(np.random.default_rng(5), 30, ("straight", "gentle"), straight_only=True)
    assert {p.scenario for p in poses} == {"straight", "gentle"}
    for p in poses:
        heading = p.road.pose_at(p.s + np.arange(0.0, LOOKAHEAD + 1e-9, 5.0))[2]
        assert np.ptp(heading) < 1e-9 and abs(p.offset) <= 1.0 and abs(p.heading) <= 0.15
        _, offset, head = p.road.ego_truth(p.state.x, p.state.y, p.state.yaw)
        assert offset == pytest.approx(p.offset, abs=1e-6) and head == pytest.approx(p.heading, abs=1e-6)
    again = sample_poses(np.random.default_rng(5), 30, ("straight", "gentle"), straight_only=True)
    assert [str(p) for p in again] == [str(p) for p in poses]


def test_any_road_pose_can_lie_on_a_bend():
    poses = sample_poses(np.random.default_rng(6), 60, ("curvy",), straight_only=False)
    turning = [p for p in poses if np.ptp(p.road.pose_at(p.s + np.arange(0.0, LOOKAHEAD + 1e-9, 5.0))[2]) > 0.1]
    assert turning


def test_the_grader_never_reads_the_reference_solutions():
    sources = list(Path(grade_pkg.__file__).parent.rglob("*.py"))
    assert sources and all("solutions" not in p.read_text() for p in sources)


def test_grading_is_repeatable():
    first = [check("0.1.3.a", provided.estimate_lane, show=False).details for _ in range(2)]
    assert first[0] == first[1]


@pytest.mark.parametrize("got, verdict", [
    (None, "not answered yet"),
    (0.2 + 1e-9, "\033[32m\N{CHECK MARK} mean |e|"),
    (0.25, "\033[31m\N{BALLOT X} mean |e|: expected 0.2, got 0.25"),
    ("0.2", "\033[31m\N{BALLOT X}"),
    (np.array([0.2, 0.2]), "\033[31m\N{BALLOT X}"),
    (np.int64(3), "expected 0.2, got 3\033"),  # a NumPy scalar prints as a plain number
])
def test_practice_prints_whether_an_ungraded_answer_is_right_and_never_raises(got, verdict, capsys):
    assert grade_pkg.practice("mean |e|", got, 0.2) is None
    assert verdict in capsys.readouterr().out


def test_practice_prints_a_tuple_of_numpy_scalars_as_plain_numbers(capsys):
    grade_pkg.practice("point", (np.float64(2.5534472), np.float64(1.2742297)), (2.553, 1.274), tol=0.001)
    assert "point: (2.55345, 1.27423)" in capsys.readouterr().out


# --- a near miss must not read "expected 3.14159, got 3.14159" ---------------------------------------------------


def expected_got(failure):
    lines = _lines(failure, "Exercise A", "checked", "hint")
    return {line.split(":")[0].strip(): line.split(":", 1)[1].strip() for line in lines[1:]}


def test_a_failure_whose_numbers_differ_in_the_7th_digit_shows_two_different_numbers_and_the_gap():
    shown = expected_got(Failure("off", expected=3.14159265, got=3.14159))
    assert shown["expected"] == "3.141593" and shown["got"] == "3.14159 (off by 2.7e-06)"
    arrays = expected_got(Failure("off", expected=np.array([1234.5678, 2.0]), got=np.array([1234.56789, 2.0])))
    assert arrays["expected"] != arrays["got"].split(" (")[0] and arrays["got"].endswith("(off by 9e-05)")
    assert fmt_pair((4, 1), np.array([4.0000001, 1.0]))[1].endswith("(off by 1e-07)")  # a tuple against an array


@pytest.mark.parametrize("expected, got", [(3.14159, 3.14159), (0.2, 0.25), (np.array([1.0, 2.0]), np.array([1.0, 2.0])), (np.nan, np.nan)])
def test_values_that_are_equal_or_already_read_differently_print_as_before(expected, got):
    assert fmt_pair(expected, got) == (fmt(expected), fmt(got))


def test_a_pre_formatted_string_is_refused_by_the_pair_formatter_and_printed_as_text_by_the_report():
    with pytest.raises(TypeError):
        fmt_pair("s 20.000000 m", (20.0, 0.1))
    assert expected_got(Failure("off", expected="nan", got=3.0)) == {"checked": "checked", "expected": "nan", "got": "3", "hint": "hint"}


def test_a_mistake_that_equals_the_right_answer_is_never_matched():
    right = np.array([4.0, 1.0])
    candidates = [("coincides", right.copy()), ("distinct", right + 0.5), ("other shape", np.zeros(3))]
    assert matching_mistakes(right + 5e-7, right, candidates, 1e-6) == []  # the near miss equals "coincides" too: it is not a mistake there
    assert matching_mistakes(right + 0.5, right, candidates, 1e-6) == ["distinct"]
    assert matching_mistakes(np.zeros(3), right, candidates, 1e-6) == ["other shape"]


def test_a_real_near_miss_from_1_1_2_shows_two_different_numbers():
    result = check("1.1.2.a", lambda pose, p: geom.world_to_ego(pose, p) + 1e-7, show=False, report=False)
    shown = {line.split(":")[0].strip(): line.split(":", 1)[1].strip() for line in result.details[1:] if ":" in line}
    assert not result.passed and shown["expected"] != shown["got"].split(" (")[0] and "(off by 1e-07)" in shown["got"]


def _shown(result):
    return {line.split(":")[0].strip(): line.split(":", 1)[1].strip() for line in result.details[1:] if ":" in line}


def test_a_1_1_3_near_miss_and_a_1_1_4_near_miss_each_show_different_expected_and_got_lines():
    from _course import REPO, load_solution
    road_frame, relative = load_solution(REPO, "1.1.3").to_road_frame, load_solution(REPO, "1.1.4").relative_pose

    def s_off(*args):
        s, d = road_frame(*args)
        return s + 2e-6, d

    def translation_off(a, b):
        out = np.array(relative(a, b), float)
        out[0, -1] += 2e-6  # the x of the shift, in a 3x3 or a 4x4
        return out

    for exercise, fn in (("1.1.3.a", s_off), ("1.1.4.a", translation_off)):
        result = check(exercise, fn, show=False, report=False)
        shown = _shown(result)
        assert not result.passed and shown["expected"] != shown["got"], (exercise, shown)


# --- practice() shows float noise as 0 on a pass and the raw numbers on a fail ------------------------------------


def test_a_passing_practice_prints_noise_below_the_tolerance_as_0_and_states_the_tolerance(capsys):
    grade_pkg.practice("A", np.array([[0.0, 1.0], [-1.0, 6.1232e-17]]), [[0, 1], [-1, 0]], tol=1e-9)
    out = capsys.readouterr().out
    assert "e-17" not in out and "all within 1e-09" in out and "\N{CHECK MARK}" in out
    grade_pkg.practice("pair", (6e-17, 0.5), (0, 0.5), tol=1e-9)
    assert "pair: (0, 0.5) (shown as 0: all within 1e-09)" in capsys.readouterr().out


def test_a_failing_practice_prints_its_raw_numbers(capsys):
    grade_pkg.practice("A", np.array([6.1232e-17, 0.5]), np.array([0.0, 0.0]), tol=1e-9)
    out = capsys.readouterr().out
    assert "e-17" in out and "\N{BALLOT X}" in out and "shown as 0" not in out


def test_a_real_1_1_2_lab_practice_prints_no_e_minus_17_noise(capsys):
    """The lab's own `practice(...)` statement for R^T R - I, run on the reference rotation matrix."""
    cells = parse_lab((REPO / "course" / "1.1.2-the-view-from-the-drivers-seat" / "lab.py").read_text())
    statement = next(ast.get_source_segment(cell.source, node) for cell in cells if cell.kind == "code" and "R transposed times R" in cell.source
                     for node in ast.walk(ast.parse(cell.source))
                     if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "practice" and "R transposed times R" in ast.get_source_segment(cell.source, node))
    assert float(np.abs(geom.rotation_matrix(2.2).T @ geom.rotation_matrix(2.2) - np.eye(2)).max()) > 0  # the noise is really there
    eval(statement, {"practice": grade_pkg.practice, "np": np, "R": geom.rotation_matrix(2.2)})
    out = capsys.readouterr().out
    assert "e-17" not in out and "e-18" not in out and "\N{CHECK MARK}" in out and "all within 1e-12" in out


# --- learner code may edit its arguments in place; the grader's arrays must not change ----------------------------


def test_call_hands_the_learner_copies_of_every_array_and_container_of_arrays_and_the_failure_stays_honest():
    road, extra, options = np.arange(6.0).reshape(3, 2), np.array([1.0, 2.0]), {"w": np.ones(2)}
    original = road.copy()

    def mutating(line, parts, opts, scale=None):
        line += 100.0
        parts[0] += 100.0
        opts["w"] += 100.0
        scale += 100.0
        return float(line[0, 0])

    got = call(mutating, road, [extra, 7], options, scale=extra, given="a road")
    assert got == 100.0 and np.array_equal(road, original) and np.array_equal(extra, [1.0, 2.0]) and np.array_equal(options["w"], np.ones(2))
    failure = Failure("wrong", expected=road[0, 0], got=got)
    assert expected_got(failure)["expected"] == "0" and expected_got(failure)["got"] == "100"  # the grader's own value, not the shifted one


def test_a_real_1_1_2_grader_is_not_fooled_by_a_learner_that_wipes_its_input_after_answering():
    solution = load_solution(REPO, "1.1.2")

    def wiping(pose, points):
        answer = solution.world_to_ego(pose, points)
        points[...] = 0.0  # the grader reads its own points again afterwards (the round trip back to the world)
        return answer

    result = check("1.1.2.a", wiping, show=False, report=False)
    assert result.passed, "\n".join(result.details)


def test_fmt_gap_shows_noise_below_the_tolerance_as_0_and_a_real_gap_as_it_is():
    assert fmt_gap(3.6e-14, 1e-9, ".2g") == "0" and fmt_gap(3.6e-14, 1e-9) == "0"
    assert fmt_gap(2.5e-3, 1e-9, ".2g") == "0.0025" and fmt_gap(float("nan"), 1e-9, ".2g") == "nan"
