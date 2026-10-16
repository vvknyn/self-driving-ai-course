"""1.1.4 Poses you can chain: the skeleton fails, the reference passes, the two presets fail for the reasons the lab names."""
import math
import re

import numpy as np
import pytest

from _course import REPO, load_solution
from zero2fsd import geom
from zero2fsd.grade import check
from zero2fsd.grade.units import u1_1_4

PRESETS = REPO / "course" / "1.1.4-poses-you-can-chain" / "presets" / "1.1.4.b"
solution = load_solution(REPO, "1.1.4")


def failing_details(exercise_id, fn):
    result = check(exercise_id, fn, show=False)
    assert not result.passed, f"{exercise_id} passed a function that should fail"
    return "\n".join(result.details)


def preset_function(name):
    namespace = {}
    exec((PRESETS / f"{name}.py").read_text(), namespace)
    return namespace["dead_reckon"]


def test_the_lab_skeletons_fail_as_not_implemented_yet():
    def skeleton(*args):
        raise NotImplementedError

    for exercise_id in ("1.1.4.a", "1.1.4.b"):
        assert "not implemented yet" in failing_details(exercise_id, skeleton)


def test_the_reference_passes_both_exercises():
    assert check("1.1.4.a", solution.relative_pose, show=False).passed
    assert check("1.1.4.b", solution.dead_reckon, show=False).passed


@pytest.mark.parametrize("name", ["add_components", "wrong_order"])
def test_each_preset_fails_the_per_pose_check_and_names_a_pose(name):
    details = failing_details("1.1.4.b", preset_function(name))
    assert "pose " in details and "every pose is checked" in details


def test_the_wrong_order_ends_near_the_truth_yet_fails_in_the_middle():
    """The lecture's point: a check of the end alone would accept this chain."""
    _, position, _ = u1_1_4.rebuild(preset_function("wrong_order"))
    assert position[-1] < 2.0 < position.max()
    assert position.max() == pytest.approx(204.88, abs=0.01)


def test_adding_the_parts_of_each_step_is_far_off_at_the_end():
    _, position, _ = u1_1_4.rebuild(preset_function("add_components"))
    assert position[-1] > 100.0


@pytest.mark.parametrize("name, fn, named", [
    ("the arguments swapped", lambda a, b: geom.invert_pose(b) @ a, "the arguments swapped"),
    ("R instead of R transposed", lambda a, b: slip_inverse(a) @ b, "the inverse built with R instead of its transpose"),
    ("the product the wrong way round", lambda a, b: b @ geom.invert_pose(a), "the product the wrong way round"),
    ("the world difference", lambda a, b: u1_1_4.matrix(b[0, 2] - a[0, 2], b[1, 2] - a[1, 2],
                                                        math.atan2(b[1, 0], b[0, 0]) - math.atan2(a[1, 0], a[0, 0])),
     "the difference of the two poses in world numbers"),
])
def test_each_relative_pose_mistake_the_lesson_names_is_named_in_the_message(name, fn, named):
    """Each is named alone: the first pair the check shows tells the mistakes apart (G-msg)."""
    details = failing_details("1.1.4.a", fn)
    assert "That is what you get from" in details and named in details and " or from " not in details


def slip_inverse(T):
    """The lab's invert_pose_slip: R where R transposed belongs, in both places."""
    inverse = np.eye(3)
    inverse[:2, :2], inverse[:2, 2] = T[:2, :2], -T[:2, :2] @ T[:2, 2]
    return inverse


def inverse_for_3x3_only(T):
    """The closed form with the size 3 baked in."""
    inverse = np.eye(3)
    inverse[:2, :2], inverse[:2, 2] = T[:2, :2].T, -T[:2, :2].T @ T[:2, 2]
    return inverse


def test_an_invert_pose_written_for_3x3_only_fails_on_the_4x4_pair_and_says_so():
    """The lab promises invert_pose works at any size and the next lesson uses it on 4x4 poses."""
    details = failing_details("1.1.4.a", lambda a, b: inverse_for_3x3_only(a) @ b)
    assert "The 3x3 pairs before it passed and this 4x4 one did not" in details and "at any size" in details
    assert check("1.1.4.a", lambda a, b: geom.invert_pose(a) @ b, show=False).passed


def test_a_dead_reckon_with_a_yaw_error_near_pi_reports_that_error_in_radians():
    """Every angle a message prints is an angle: the rotation-entry gap here is 2, the yaw error is 3.13."""
    def turned(T0, steps):
        P = solution.dead_reckon(T0, steps).copy()
        c, s = math.cos(3.13), math.sin(3.13)
        P[:, :2, :2] = np.array([[c, -s], [s, c]]) @ P[:, :2, :2]
        return P
    details = failing_details("1.1.4.b", turned)
    shown = float(re.search(r"([0-9.]+) rad of yaw off", details).group(1))
    assert abs(shown - 3.13) < 0.01


def test_a_wrong_relative_pose_that_matches_no_named_mistake_gets_no_guess():
    details = failing_details("1.1.4.a", lambda a, b: np.eye(3))
    assert "two poses away from the axes" in details and "That is what you get" not in details


def test_a_near_miss_where_the_mistakes_coincide_with_the_right_answer_names_none():
    """For the same pose twice, every mistake of the lesson gives the identity: the right answer. None of them may be named."""
    def near_miss(a, b):
        out = np.array(solution.relative_pose(a, b), float)
        if np.array_equal(a, b):
            out[0, -1] += 5e-7
        return out
    details = failing_details("1.1.4.a", near_miss)
    assert "the same pose twice" in details and "That is what you get" not in details


@pytest.mark.parametrize("break_it, words", [
    (lambda T: np.diag([1.0, 1.0, 2.0]) @ T, "last row"),
    (lambda T: T + np.array([[0, 0, 0], [0, 0, 0], [0, 1e-3, 0]]), "last row"),
    (lambda T: np.diag([1.5, 1.5, 1.0]) @ T, "not a rotation"),
    (lambda T: np.diag([1.0, -1.0, 1.0]) @ T, "mirror"),
    (lambda T: T * np.nan, "nan"),
])
def test_a_relative_pose_that_is_no_pose_matrix_is_told_so_in_plain_words(break_it, words):
    assert words in failing_details("1.1.4.a", lambda a, b: break_it(solution.relative_pose(a, b)))


@pytest.mark.parametrize("break_it, words", [
    (lambda P: P * np.nan, "nan"),
    (lambda P: np.diag([1.0, 1.0, 3.0]) @ P, "last row"),
    (lambda P: np.diag([1.01, 1.01, 1.0]) @ P, "not a rotation"),
])
def test_a_dead_reckon_that_is_no_pose_matrix_names_the_pose_and_the_reason(break_it, words):
    details = failing_details("1.1.4.b", lambda T0, steps: break_it(solution.dead_reckon(T0, steps)))
    assert words in details and "is not a pose matrix" in details


def test_a_dead_reckon_whose_positions_are_right_but_whose_rotation_entries_are_not_fails():
    """Positions and the atan2 angle alone would accept a rotation block scaled by 1.001; the entries do not."""
    def scaled(T0, steps):
        P = solution.dead_reckon(T0, steps).copy()
        P[:, :2, :2] *= 1.0000001
        return P
    assert not check("1.1.4.b", scaled, show=False).passed


@pytest.mark.parametrize("name, named", [
    ("add_components", "adding each step's forward part to x"),
    ("wrong_order", "each step multiplied on the left"),
])
def test_each_dead_reckon_mistake_the_lesson_names_is_named_in_the_message(name, named):
    assert named in failing_details("1.1.4.b", preset_function(name))


def test_a_relative_pose_of_the_wrong_shape_is_a_clear_failure():
    assert "shape" in failing_details("1.1.4.a", lambda a, b: (a @ b)[:2])


def test_invert_pose_the_lab_asks_for_works_at_both_sizes():
    """Problem 3's 4x4 case: the closed form, not np.linalg.inv, at any size."""
    T3 = geom.pose_to_matrix((1.0, 2.0, 0.9))
    T4 = np.eye(4)
    T4[:3, :3] = geom.rot_z(0.7) @ geom.rot_y(0.1)
    T4[:3, 3] = (1.2, 0.3, 1.5)
    for T in (T3, T4):
        assert np.abs(geom.invert_pose(T) @ T - np.eye(len(T))).max() < 1e-12


def test_a_relative_pose_that_is_right_on_the_lap_but_wrong_off_it_still_fails():
    """The seeded pairs reach beyond the lap's own poses: x, y in (-50, 50) and any heading."""
    def lap_only(a, b):
        return solution.relative_pose(a, b) if abs(a[1, 0]) < 0.9 else np.eye(3)
    assert not check("1.1.4.a", lap_only, show=False).passed


def test_the_trailing_one_and_the_wrapped_angle_the_lab_prints():
    """The trailing 1 versus 0, and the wrap trap class-wide: the raw comparison fails at exactly the steps where the yaw is above pi."""
    poses, _ = u1_1_4.lap()
    T = geom.pose_to_matrix(tuple(poses[400]))
    assert np.allclose(T @ [10, 2, 1], (160.987, 27.8514, 1), atol=1e-3)
    assert np.allclose(T @ [10, 2, 0], (6.3282, 7.9971, 0), atol=1e-3)
    gap = np.array([geom.matrix_to_pose(geom.pose_to_matrix(tuple(p)))[2] for p in poses]) - poses[:, 2]
    assert math.isclose(abs(gap[1021]), 2 * math.pi, abs_tol=1e-9) and np.abs(gap[:1021]).max() < 1e-12
    assert np.array_equal(np.nonzero(np.abs(gap) > 1)[0], np.nonzero(poses[:, 2] > math.pi)[0])


def test_a_pass_with_float_noise_reports_a_worst_error_of_0():
    result = check("1.1.4.b", lambda start, steps: solution.dead_reckon(start, steps) + 3.6e-14, show=False)
    assert result.passed and "worst 0 m" in result.details[0]
