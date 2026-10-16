"""Unit 1.1.1 Positions and headings: what its graders refuse, its previews, and its `naive_arctan` preset.

The reference solutions passing, the margin and the generic report rules are covered for every unit by test_solutions.py,
test_margins.py and test_grade.py; this file holds what only this unit has: the wrong answers its lecture warns about.
"""
import math
import re

import numpy as np
import pytest

from _course import REPO, load_solution
from zero2fsd.car import Car, run
from zero2fsd.geom import wrap_angle
from zero2fsd.grade import check
from zero2fsd.grade.units import u1_1_1
from zero2fsd.preview import get

PRESETS = REPO / "course" / "1.1.1-positions-and-headings" / "presets"
SOLUTION = load_solution(REPO, "1.1.1")


def text(result):
    return "\n".join(result.details)


def preset(exercise_id, name):
    """The preset file's function, loaded the way a learner pastes it into a cell."""
    scope = {}
    exec((PRESETS / exercise_id / f"{name}.py").read_text(), scope)
    return scope[{"1.1.1.a": "direction_angle", "1.1.1.b": "forward_left"}[exercise_id]]


@pytest.fixture(scope="module")
def lap():
    return run(Car(), "gentle")


# --- Exercise A: direction_angle ---------------------------------------------------------------------------------

def zero_gives_east(v):  # arctan2 as is: right everywhere except the zero arrow
    return np.arctan2(np.asarray(v, dtype=float)[..., 1], np.asarray(v, dtype=float)[..., 0])


def arguments_swapped(v):
    v = np.asarray(v, dtype=float)
    return np.where((v == 0).all(axis=-1), np.nan, np.arctan2(v[..., 0], v[..., 1]))[()]


def full_turns(v):  # an angle in [0, 2 pi) instead of [-pi, pi]
    return np.mod(SOLUTION.direction_angle(v), 2 * math.pi)


@pytest.mark.parametrize("answer, said", [
    (preset("1.1.1.a", "naive_arctan"), "wrong direction for a single arrow"),
    (arguments_swapped, "wrong direction for a single arrow"),
    (full_turns, "wrong direction for a single arrow"),
    (zero_gives_east, "the zero arrow has no direction"),
    (lambda v: float(SOLUTION.direction_angle(v)[0]) if np.ndim(v) == 2 else SOLUTION.direction_angle(v), "expected one angle per arrow"),
])
def test_a_wrong_direction_angle_is_refused_with_the_reason(answer, said):
    result = check("1.1.1.a", answer, show=False)
    assert not result.passed and said in text(result), text(result)


def test_the_naive_preset_fails_on_the_first_arrow_that_points_west_and_names_the_arrow():
    lines = text(check("1.1.1.a", preset("1.1.1.a", "naive_arctan"), show=False))
    assert "v = (-1, 0)" in lines and "expected: 3.14159" in lines


def test_the_lap_check_wraps_the_difference_so_a_full_turn_more_is_not_a_miss():
    arrows, _ = u1_1_1._lap()
    angles = SOLUTION.direction_angle(arrows)
    assert u1_1_1.miss_from_yaw(angles) == pytest.approx(u1_1_1.miss_from_yaw(angles + 2 * math.pi)) == pytest.approx(0.040994, abs=1e-6)
    assert u1_1_1.miss_from_yaw(angles + 0.2) > u1_1_1.LAP_TOL


# --- Exercise B: forward_left -------------------------------------------------------------------------------------

def cross_the_wrong_way(theta, v):  # v x h: every left part changes sign
    out = SOLUTION.forward_left(theta, v)
    return np.stack([out[..., 0], -out[..., 1]], axis=-1)


def left_arrow_as_sin_cos(theta, v):
    h = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    v = np.asarray(v, dtype=float)
    return np.stack([(v * h).sum(axis=-1), v[..., 0] * np.sin(theta) + v[..., 1] * np.cos(theta)], axis=-1)


def first_heading_only(theta, v):  # a heading per arrow is ignored
    return SOLUTION.forward_left(np.ravel(theta)[0], v)


@pytest.mark.parametrize("answer", [cross_the_wrong_way, left_arrow_as_sin_cos, first_heading_only])
def test_a_wrong_forward_left_is_refused(answer):
    result = check("1.1.1.b", answer, show=False)
    assert not result.passed and ("wrong forward or left part" in text(result) or "row i must use heading i" in text(result)), text(result)


# --- previews -----------------------------------------------------------------------------------------------------

def test_the_direction_preview_counts_misses_and_reports_the_worst_for_the_solution_and_the_preset(lap):
    build = get("1.1.1.a").build
    assert build(SOLUTION.direction_angle, lap) == {"steps where your heading misses yaw by > 0.1 rad": 0.0, "worst miss (rad)": pytest.approx(0.040994, abs=1e-6)}
    naive = build(preset("1.1.1.a", "naive_arctan"), lap)
    assert list(naive.values()) == [888.0, pytest.approx(3.142, abs=1e-3)]


def test_the_forward_left_preview_tells_the_solution_from_each_slip(lap):
    build = get("1.1.1.b").build
    assert list(build(SOLUTION.forward_left, lap).values()) == [pytest.approx(0.3998, abs=1e-4), pytest.approx(0.0164, abs=1e-4)]
    assert build(cross_the_wrong_way, lap)["largest left per step (m)"] == pytest.approx(0.0022, abs=1e-4)
    assert build(left_arrow_as_sin_cos, lap)["largest left per step (m)"] == pytest.approx(0.4000, abs=1e-4)


def test_a_nan_direction_counts_as_a_miss_and_the_telemetry_is_never_changed(lap):
    def scribble(v):
        v.fill(0.0)
        return np.full(len(v), np.nan)

    before = (lap.x.copy(), lap.y.copy(), lap.yaw.copy())
    values = get("1.1.1.a").build(scribble, lap)
    assert values["steps where your heading misses yaw by > 0.1 rad"] == len(lap.x) - 1
    assert all(np.array_equal(a, b) for a, b in zip(before, (lap.x, lap.y, lap.yaw)))


@pytest.mark.parametrize("exercise_id, answer, said", [
    ("1.1.1.a", lambda v: None, "expected one angle per step arrow, shape (1777,), got shape ()"),
    ("1.1.1.a", lambda v: np.zeros(3), "shape (1777,), got shape (3,)"),
    ("1.1.1.a", lambda v: "abc", "expected one angle per step arrow, got str"),
    ("1.1.1.b", lambda theta, v: "3", "expected a (forward, left) pair per step arrow, shape (1777, 2), got shape ()"),
])
def test_a_preview_answer_that_is_not_numbers_of_the_right_shape_is_a_plain_language_type_error(lap, exercise_id, answer, said):
    with pytest.raises(TypeError, match=re.escape(said)):
        get(exercise_id).build(answer, lap)
