"""Verdict margins: the checks that compare a simulation-derived number with a threshold.

Three checks do (the drive gate, the perception hit rate, the paint IoU).  A verdict that sits too close to its
threshold could read PASS on one computer and FAIL on another, so the reference solution must clear each by
MARGIN, and a learner's answer that lands within BORDERLINE_FRACTION gets the borderline sentence.

The other checks compare the learner's function with a reference computed in the same runtime (tolerance 1e-9),
so the computer cannot split them and they need no margin.
"""
import dataclasses
import operator
import re
from pathlib import Path

import numpy as np
import pytest

import zero2fsd.grade

from _course import REPO, load_solution
from zero2fsd.car import Car, LaneEstimate, provided, run
from zero2fsd.grade import _drive, check
from zero2fsd.grade._drive import GATE_MEAN_LAT, GATE_SCENARIO
from zero2fsd.grade._report import BORDERLINE_FRACTION, BORDERLINE_NOTE, borderline, margin, vs_limit
from zero2fsd.grade.units import u0_1_2, u0_1_3
from zero2fsd.grade.units.u0_1_2 import IOU_MIN, paint_trials
from zero2fsd.grade.units.u0_1_3 import MIN_HIT_RATE, N_POSES, pose_trials
from zero2fsd.score import metrics

MARGIN = 0.05  # the reference solution clears every threshold by at least this fraction of the threshold


def gate_mean_lat():
    tel = run(load_solution(REPO, "0.1.4").my_car, GATE_SCENARIO)
    assert tel.completed
    return metrics(tel)["mean_abs_lat"]


def hit_rate():
    return sum(hit for *_, hit in pose_trials(load_solution(REPO, "0.1.3").estimate_lane)) / N_POSES


def lowest_iou():
    return min(iou for *_, iou in paint_trials(load_solution(REPO, "0.1.2").paint_masks))


# check -> (measure on the reference solution, the grader module that prints its margin, its threshold, the side that passes)
THRESHOLDED = {
    "drive gate: mean |lat|": (gate_mean_lat, _drive, GATE_MEAN_LAT, operator.lt),
    "perception hit rate": (hit_rate, u0_1_3, MIN_HIT_RATE, operator.ge),
    "paint IoU": (lowest_iou, u0_1_2, IOU_MIN, operator.ge),
}


@pytest.mark.parametrize("name", THRESHOLDED)
def test_the_reference_solution_clears_every_thresholded_check_by_the_margin(name):
    measure, _, threshold, passes = THRESHOLDED[name]
    value = measure()
    assert passes(value, threshold), f"{name}: {value} against {threshold}"
    assert margin(value, threshold) >= MARGIN, f"{name}: {value} is only {margin(value, threshold):.1%} from {threshold}"


def test_every_grader_that_prints_a_margin_is_in_the_table():
    """A new thresholded check must land in THRESHOLDED too: every `vs_limit(` caller in the grade package is one row."""
    package = Path(zero2fsd.grade.__file__).resolve().parent
    printers = {
        path for path in package.rglob("*.py")
        if path.name != "_report.py" and re.search(r"\bvs_limit\(", path.read_text())
    }
    assert printers == {Path(module.__file__).resolve() for _, module, _, _ in THRESHOLDED.values()}


# --- the sentence and the printed margin ---------------------------------------------------------------------------

@pytest.mark.parametrize("value, near", [(0.509, True), (0.491, True), (0.5, True), (0.512, False), (0.488, False), (0.0, False)])
def test_a_value_within_two_percent_of_the_threshold_is_borderline_on_either_side(value, near):
    assert BORDERLINE_FRACTION == 0.02
    assert borderline(value, 0.5) == (BORDERLINE_NOTE if near else None)


def test_the_borderline_sentence_is_the_one_the_spec_fixes_and_a_negative_threshold_works():
    assert BORDERLINE_NOTE == "Borderline: tiny differences between computers can flip this. Aim for a clearer margin."
    assert borderline(-1.01, -1.0) == BORDERLINE_NOTE and borderline(-1.1, -1.0) is None


@pytest.mark.parametrize(
    "value, limit, unit, printed",
    [
        (0.376, 0.5, "m", "(limit 0.5 m, margin 25 %)"),
        (0.92, 0.85, "", "(limit 0.85, margin 8.2 %)"),
        (0.0, 0.5, "m", "(limit 0.5 m, margin 100 %)"),
        (92.5, 90, "%", "(limit 90 %, margin 2.8 %)"),
        (0.499, 0.5, "m", f"(limit 0.5 m, margin 0.2 %). {BORDERLINE_NOTE}"),
    ],
)
def test_vs_limit_prints_the_limit_and_the_margin_and_the_sentence_only_when_borderline(value, limit, unit, printed):
    assert vs_limit(value, limit, unit) == printed


# --- every thresholded line of a real check prints metric, limit and margin ---------------------------------------

def text(result):
    return "\n".join(result.details)


def own_perception(obs):
    return provided.estimate_lane(obs)


def blind_for(first: int):
    """Provided perception that sees nothing on its first `first` frames, so exactly 40 - first poses are hits."""
    calls = []

    def perception(obs):
        calls.append(1)
        return LaneEstimate(float("nan"), float("nan"), False) if len(calls) <= first else provided.estimate_lane(obs)

    return perception


def keeping(share: float):
    """Provided paint masks with all but the first `share` of each mask's pixels dropped: IoU is about `share`."""

    def paint_masks(frame):
        kept = []
        for mask in provided.paint_masks(frame):
            out = np.zeros_like(mask)
            out.flat[np.flatnonzero(mask)[: int(share * np.count_nonzero(mask))]] = True
            kept.append(out)
        return tuple(kept)

    return paint_masks


# (check, learner, passes?, borderline?, what the printed line says)
@pytest.mark.parametrize(
    "exercise_id, learner, passed, near, says",
    [
        ("0.1.3.a", provided.estimate_lane, True, False, "hit rate 100.0 % (limit 90 %, margin 11 %)"),
        ("0.1.3.a", blind_for(4), True, True, "hit rate 90.0 % (limit 90 %, margin 0.0 %)"),
        ("0.1.3.a", blind_for(5), False, False, "hit rate 87.5 % (limit 90 %, margin 2.8 %)"),
        ("0.1.2.a", provided.paint_masks, True, False, "IoU 1.000 (limit 0.85, margin 18 %)"),
        ("0.1.2.a", keeping(0.86), True, True, "(limit 0.85, margin 1."),
        ("0.1.2.a", keeping(0.84), False, True, "(limit 0.85, margin 1."),
        ("0.1.2.a", keeping(0.5), False, False, "(limit 0.85, margin 41 %)"),
    ],
)
def test_the_hit_rate_and_the_iou_print_their_limit_and_margin_and_flag_a_thin_one(exercise_id, learner, passed, near, says):
    result = check(exercise_id, learner, show=False)
    assert result.passed is passed, text(result)
    assert says in text(result), text(result)
    assert (BORDERLINE_NOTE in text(result)) is near, text(result)


def tel_that(make_tel, *, completed, mean_lat):
    tel = make_tel(lat=[mean_lat, mean_lat], progress=[0.0, 100.0], scenario="gentle", route_length=100.0)
    return dataclasses.replace(tel, completed=completed)


@pytest.mark.parametrize("exercise_id, subject", [
    ("0.1.3.b", own_perception),
    ("0.1.4.b", Car(perception=own_perception, controller=lambda est, obs: provided.steer_p(est))),
])
@pytest.mark.parametrize(
    "completed, mean_lat, passed, near, says",
    [
        (True, 0.376, True, False, "mean |lat| 0.376 m (limit 0.5 m, margin 25 %)"),
        (True, 0.495, True, True, "mean |lat| 0.495 m (limit 0.5 m, margin 1.0 %)"),
        (True, 0.505, False, True, "mean |lat| 0.505 m (limit 0.5 m, margin 1.0 %)"),
        (False, 0.8, False, False, "mean |lat| 0.800 m (limit 0.5 m, margin 60 %)"),
    ],
)
def test_both_gate_exercises_print_the_gate_limit_and_margin_and_flag_a_thin_one(
        monkeypatch, make_tel, exercise_id, subject, completed, mean_lat, passed, near, says):
    monkeypatch.setattr(_drive, "run", lambda car, scenario, on_step=None: tel_that(make_tel, completed=completed, mean_lat=mean_lat))
    result = check(exercise_id, subject, show=False)
    assert result.passed is passed, text(result)
    assert says in text(result), text(result)
    assert (BORDERLINE_NOTE in text(result)) is near, text(result)
