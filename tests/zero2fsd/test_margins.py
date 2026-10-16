"""Verdict margins: the checks that compare a simulation-derived number with a threshold.

Three checks do (the drive gate, the perception hit rate, the paint IoU).  A verdict that sits too close to its
threshold could read PASS on one computer and FAIL on another, so the reference solution must clear each by
MARGIN, and a learner's answer that lands within BORDERLINE_FRACTION gets the borderline sentence.

The other checks compare the learner's function with a reference computed in the same runtime (tolerance 1e-9),
so the computer cannot split them and they need no margin.
"""
import dataclasses
import importlib
import re
from pathlib import Path

import numpy as np
import pytest

import zero2fsd.grade

from _course import REPO, load_solution
from zero2fsd.car import Car, LaneEstimate, provided
from zero2fsd.grade import _drive, check
from zero2fsd.grade._report import BORDERLINE_FRACTION, BORDERLINE_NOTE, borderline, margin, vs_limit

MARGIN = 0.05  # the reference solution clears every threshold by at least this fraction of the threshold


def graders_calling_vs_limit():
    """The grade-package modules (the unit modules and the shared drive gate) that print a margin."""
    package = Path(zero2fsd.grade.__file__).resolve().parent
    return [path for path in sorted(package.rglob("*.py")) if path.name != "_report.py" and re.search(r"\bvs_limit\(", path.read_text())]


def grader_module(path):
    return importlib.import_module(".".join(("zero2fsd", "grade", *path.relative_to(Path(zero2fsd.grade.__file__).resolve().parent).with_suffix("").parts)))


# "module: label" -> its Threshold, from the module-level THRESHOLDS of each grader that calls vs_limit
THRESHOLDED = {f"{path.stem}: {label}": threshold
               for path in graders_calling_vs_limit() for label, threshold in getattr(grader_module(path), "THRESHOLDS", {}).items()}


@pytest.mark.parametrize("name", THRESHOLDED)
def test_the_reference_solution_clears_every_thresholded_check_by_the_margin(name):
    threshold = THRESHOLDED[name]
    value = threshold.measure(load_solution(REPO, threshold.unit))
    assert threshold.passes(value, threshold.limit), f"{name}: {value} against {threshold.limit}"
    assert margin(value, threshold.limit) >= MARGIN, f"{name}: {value} is only {margin(value, threshold.limit):.1%} from {threshold.limit}"


def test_every_grader_that_prints_a_margin_declares_its_thresholds():
    undeclared = [path.relative_to(REPO).as_posix() for path in graders_calling_vs_limit() if not getattr(grader_module(path), "THRESHOLDS", None)]
    assert not undeclared, "these call vs_limit but declare no module-level THRESHOLDS = {label: Threshold(limit, passes, unit, measure)}: " + ", ".join(undeclared)


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


GATE_EXERCISES = pytest.mark.parametrize("exercise_id, subject", [
    ("0.1.3.b", own_perception),
    ("0.1.4.b", Car(perception=own_perception, controller=lambda est, obs: provided.steer_p(est))),
])


@GATE_EXERCISES
@pytest.mark.parametrize(
    "completed, mean_lat, passed, near, says",
    [
        (True, 0.376, True, False, "mean |lat| 0.376 m (limit 0.5 m, margin 25 %)"),
        (True, 0.495, True, True, "mean |lat| 0.495 m (limit 0.5 m, margin 1.0 %)"),
        (True, 0.505, False, True, "mean |lat| 0.505 m (limit 0.5 m, margin 1.0 %)"),
        (False, 0.8, False, False, "mean |lat| 0.800 m (limit 0.5 m, margin 60 %)"),
        # a run that did not finish fails on completion: a thin mean |lat| did not decide it, so no coin-flip sentence
        (False, 0.495, False, False, "mean |lat| 0.495 m (limit 0.5 m, margin 1.0 %)"),
        (False, 0.505, False, False, "mean |lat| 0.505 m (limit 0.5 m, margin 1.0 %)"),
    ],
)
def test_both_gate_exercises_print_the_gate_limit_and_margin_and_flag_a_thin_one(
        monkeypatch, make_tel, exercise_id, subject, completed, mean_lat, passed, near, says):
    monkeypatch.setattr(_drive, "run", lambda car, scenario, on_step=None, camera=None: tel_that(make_tel, completed=completed, mean_lat=mean_lat))
    result = check(exercise_id, subject, show=False)
    assert result.passed is passed, text(result)
    assert says in text(result), text(result)
    assert (BORDERLINE_NOTE in text(result)) is near, text(result)


@GATE_EXERCISES
@pytest.mark.parametrize("mean_lat, near", [(98.0, False), (98.01, True), (102.0, False), (101.99, True)])
def test_the_borderline_sentence_needs_a_margin_under_two_percent_not_at_it(monkeypatch, make_tel, exercise_id, subject, mean_lat, near):
    monkeypatch.setattr(_drive, "GATE_MEAN_LAT", 100.0)  # a 2 % margin is exact in floating point at 100 and is not at 0.5
    monkeypatch.setattr(_drive, "run", lambda car, scenario, on_step=None, camera=None: tel_that(make_tel, completed=True, mean_lat=mean_lat))
    result = check(exercise_id, subject, show=False)
    assert (BORDERLINE_NOTE in text(result)) is near, text(result)
