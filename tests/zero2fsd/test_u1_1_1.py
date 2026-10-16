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


def wrapped_to_the_cut(v):  # the lecture's own wrap puts west at -pi, not +pi
    return wrap_angle(SOLUTION.direction_angle(v))


def in_degrees(v):
    return np.degrees(SOLUTION.direction_angle(v))


def zero_only_when_all_zero(v):  # nan only when the whole input is the zero arrow: a zero row inside a batch comes out east
    v = np.asarray(v, dtype=float)
    return np.nan if not v.any() else np.arctan2(v[..., 1], v[..., 0])


def np_where_without_the_zero_dim_fix(v):  # a correct answer; for one arrow np.where returns a 0-d array, not a float
    v = np.asarray(v, dtype=float)
    return np.where((v == 0).all(axis=-1), np.nan, np.arctan2(v[..., 1], v[..., 0]))


@pytest.mark.parametrize("answer, said", [
    (preset("1.1.1.a", "naive_arctan"), "np.arctan of the slope"),
    (arguments_swapped, "dx and dy look swapped"),
    (full_turns, "another angle range"),
    (wrapped_to_the_cut, "another angle range"),
    (in_degrees, "degrees"),
    (zero_gives_east, "the zero arrow has no direction"),
    (zero_only_when_all_zero, "the zero arrow inside a batch has no direction: row 17 must be nan"),
    (lambda v: float(SOLUTION.direction_angle(v)[0]) if np.ndim(v) == 2 else SOLUTION.direction_angle(v), "expected one angle per arrow"),
])
def test_a_wrong_direction_angle_is_refused_with_the_mistake_it_matches(answer, said):
    result = check("1.1.1.a", answer, show=False)
    assert not result.passed and said in text(result), text(result)


def test_an_np_where_answer_that_returns_a_zero_dim_array_is_correct_and_passes():  # needs the shared F0c fix (a 0-d array is a number)
    assert check("1.1.1.a", np_where_without_the_zero_dim_fix, show=False).passed


def test_the_naive_preset_fails_on_the_first_arrow_that_points_west_and_names_the_arrow():
    lines = text(check("1.1.1.a", preset("1.1.1.a", "naive_arctan"), show=False))
    assert "v = (-1, 0)" in lines and "expected: 3.14159" in lines


def test_the_lap_check_wraps_the_difference_so_a_full_turn_more_is_not_a_miss():
    arrows, _ = u1_1_1._lap()
    angles = SOLUTION.direction_angle(arrows)
    assert u1_1_1.miss_from_yaw(angles) == pytest.approx(u1_1_1.miss_from_yaw(angles + 2 * math.pi)) == pytest.approx(0.040994, abs=1e-6)
    assert u1_1_1.miss_from_yaw(angles + 0.2) > u1_1_1.LAP_TOL


def test_a_wrong_answer_only_on_step_length_arrows_is_caught_by_the_direct_comparison():
    def off_on_short_arrows(v):  # right everywhere except arrows shorter than 0.5 m, where it is 0.01 rad off: inside the yaw tolerance
        out = np.asarray(SOLUTION.direction_angle(v), dtype=float)
        short = np.hypot(*np.moveaxis(np.asarray(v, dtype=float), -1, 0)) < 0.5
        return np.where(short & (np.hypot(*np.moveaxis(np.asarray(v, dtype=float), -1, 0)) > 0), out + 0.01, out)
    result = check("1.1.1.a", off_on_short_arrows, show=False)
    assert not result.passed and "step arrow" in text(result), text(result)


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


def forward_negated(theta, v):  # a mistake the lab does not list
    out = SOLUTION.forward_left(theta, v)
    return np.stack([-out[..., 0], out[..., 1]], axis=-1)


@pytest.mark.parametrize("answer, said", [
    (cross_the_wrong_way, "wrong sign"),
    (left_arrow_as_sin_cos, "(-sin theta, cos theta), not (sin theta, cos theta)"),
    (forward_negated, "wrong forward or left part"),  # unnamed, but still honest
    (first_heading_only, "row i must use yaw i"),
])
def test_a_wrong_forward_left_is_refused_and_a_known_mistake_is_named(answer, said):
    result = check("1.1.1.b", answer, show=False)
    assert not result.passed and said in text(result), text(result)


def test_a_direction_that_is_a_pair_of_numbers_says_it_wants_one_number():
    result = check("1.1.1.a", lambda v: np.array([0.0, np.nan]), show=False)
    assert not result.passed and "expected one number for one arrow, got shape (2,)" in text(result), text(result)


def test_the_lap_check_reports_a_real_miss_not_a_flip_across_the_cut():
    def smoothed(v):  # each angle averaged with its neighbours: wrong, and flips at the cut
        out = np.asarray(SOLUTION.direction_angle(v), dtype=float) if np.ndim(v) == 2 else SOLUTION.direction_angle(v)
        return out if np.ndim(v) < 2 or len(v) < 100 else np.convolve(out, np.ones(3) / 3, mode="same")
    said = text(check("1.1.1.a", smoothed, show=False))
    assert "step arrow" in said and "-3.1415" not in said, said


# --- previews -----------------------------------------------------------------------------------------------------

def test_the_direction_preview_counts_misses_and_reports_the_worst_for_the_solution_and_the_preset(lap):
    build = get("1.1.1.a").build
    assert build(SOLUTION.direction_angle, lap) == {"steps where your direction misses the yaw by > 0.1 rad": 0.0, "worst miss (rad)": pytest.approx(0.040994, abs=1e-6)}
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
    assert values["steps where your direction misses the yaw by > 0.1 rad"] == len(lap.x) - 1
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
