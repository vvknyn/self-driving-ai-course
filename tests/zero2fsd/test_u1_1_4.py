"""1.1.4 Poses you can chain: the skeleton fails, the reference passes, the two presets fail for the reasons the lab names."""
import math

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


def test_a_swapped_relative_pose_fails_and_the_message_names_the_case():
    details = failing_details("1.1.4.a", lambda a, b: geom.invert_pose(b) @ a)
    assert "A then B" in details


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
    poses, _ = u1_1_4.lap()
    T = geom.pose_to_matrix(tuple(poses[400]))
    assert np.allclose(T @ [10, 2, 1], (160.987, 27.8514, 1), atol=1e-3)
    assert np.allclose(T @ [10, 2, 0], (6.3282, 7.9971, 0), atol=1e-3)
    gap = np.array([geom.matrix_to_pose(geom.pose_to_matrix(tuple(p)))[2] for p in poses]) - poses[:, 2]
    assert math.isclose(abs(gap[1021]), 2 * math.pi, abs_tol=1e-9) and np.abs(gap[:1021]).max() < 1e-12
