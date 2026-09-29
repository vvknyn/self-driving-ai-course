import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

import zero2fsd.grade as grade_pkg
from zero2fsd.car import Car, LaneEstimate, provided
from zero2fsd.grade import _drive, check
from zero2fsd.grade._poses import LOOKAHEAD, sample_poses
from zero2fsd.grade._report import EXERCISES, NOT_IMPLEMENTED
from zero2fsd.sim.dynamics import MAX_STEER

IDS = ["0.1.1.a", "0.1.1.b", "0.1.2.a", "0.1.3.a", "0.1.3.b", "0.1.4.a", "0.1.4.b"]


def as_target(exercise_id, fn):
    """The object an exercise grades: a Car for the all-yours exercise, else the function itself."""
    return Car(perception=fn, controller=lambda est, obs: fn()) if exercise_id == "0.1.4.b" else fn


def text(result):
    return "\n".join(result.details)


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
    assert capsys.readouterr().out.startswith(f"\033[32m\N{CHECK MARK} {exercise_id}:")


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
    monkeypatch.setattr(_drive, "run", lambda car, scenario: tel)
    if verdict == "pass":
        assert "completed gentle" in _drive.gate(Car(), show=False)
    else:
        with pytest.raises(_drive.Failure) as err:
            _drive.gate(Car(), show=False)
        assert verdict in f"{err.value.headline} {err.value.shown['got']}"


def test_show_draws_the_dashboard_once_per_run_only_when_asked(monkeypatch, make_tel):
    monkeypatch.setattr(_drive, "run", lambda car, scenario: telemetry_that(make_tel, completed=True, mean_lat=0.1))
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
    assert "cliffhanger (not graded): left the lane after" in text(result)
    out = capsys.readouterr().out
    assert "your car on gentle" in out and "your cliffhanger on curvy" in out


def test_a_working_perception_passes_the_gate_when_placed_in_the_provided_car():
    result = check("0.1.3.b", own_perception, show=False)
    assert result.passed and "completed gentle" in text(result), text(result)


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
