"""1.1.2 The view from the driver's seat: the grader tells the right transform from the ones that look right at yaw 0."""
import math

import numpy as np
import pytest

from _course import REPO, load_solution
from zero2fsd import geom
from zero2fsd.car import Car, run
from zero2fsd.grade import check
from zero2fsd.grade.units.u1_1_2 import renderer_gap
from zero2fsd.preview import get
from zero2fsd.preview.units.u1_1_2 import PICTURES, differing_pictures
from zero2fsd.sim import Camera

PRESETS = REPO / "course" / "1.1.2-the-view-from-the-drivers-seat" / "presets" / "1.1.2.b"
solution = load_solution(REPO, "1.1.2").world_to_ego


def r_not_transposed(pose, p):
    return (np.asarray(p, dtype=float) - np.array(pose[:2])) @ geom.rotation_matrix(pose[2]).T


def no_rotation(pose, p):
    return np.asarray(p, dtype=float) - np.array(pose[:2])


def rotate_first(pose, p):  # R^T p - t: the right rotation, applied before the subtraction
    return np.asarray(p, dtype=float) @ geom.rotation_matrix(pose[2]) - np.array(pose[:2])


def forward_only(pose, p):  # the right answer for one point, and a crash for an array of them
    x, y, theta = pose
    dx, dy = p[0] - x, p[1] - y
    return np.array([dx * math.cos(theta) + dy * math.sin(theta), -dx * math.sin(theta) + dy * math.cos(theta)])


def not_implemented(pose, p):
    raise NotImplementedError


def preset(name):
    namespace = {"np": np, "rotation_matrix": geom.rotation_matrix}  # what the playground's prelude binds
    exec((PRESETS / f"{name}.py").read_text(), namespace)
    return namespace["world_to_ego"]


def run_check(exercise_id, fn):
    return check(exercise_id, fn, show=False, report=False)


def test_the_skeleton_fails_and_says_it_is_not_written_yet():
    for exercise_id in ("1.1.2.a", "1.1.2.b"):
        result = run_check(exercise_id, not_implemented)
        assert not result.passed and "not implemented yet" in result.details[0]


def test_the_reference_solution_passes_both_exercises_and_matches_the_renderer_to_rounding():
    assert run_check("1.1.2.a", solution).passed and run_check("1.1.2.b", solution).passed
    assert renderer_gap(solution) < 1e-12 and renderer_gap(geom.world_to_ego) == 0.0


@pytest.mark.parametrize(("fn", "yaw"), [(r_not_transposed, "pi/2"), (no_rotation, "pi/2"), (rotate_first, "pi/2")])
def test_a_wrong_transform_is_named_at_the_first_yaw_that_catches_it(fn, yaw):
    result = run_check("1.1.2.a", fn)
    assert not result.passed and f"wrong at yaw {yaw}" in result.details[0]
    assert "1.1.2" not in "\n".join(result.details)  # never the dotted id


def test_the_named_yaws_say_what_they_can_and_cannot_tell_apart():
    assert "catches both R in place of its transpose and a missing rotation" in run_check("1.1.2.a", r_not_transposed).details[0]
    assert "pi/2" in run_check("1.1.2.a", no_rotation).details[0]  # no rotation passes yaw 0 only; the swap passes yaw 0 and pi


def test_a_transform_that_only_takes_one_point_fails_the_shape_check():
    result = run_check("1.1.2.a", forward_only)
    assert not result.passed and "shape" in result.details[0] + "".join(result.details)


@pytest.mark.parametrize("fn", [r_not_transposed, no_rotation, rotate_first])
def test_the_forty_pose_test_catches_what_a_yaw_zero_test_misses(fn):
    at_zero = ((5.0, 2.0, 0.0), [[8.0, 3.0]])
    assert np.allclose(fn(*at_zero), geom.world_to_ego(*at_zero), atol=1e-12)  # all three pass at yaw 0
    assert renderer_gap(fn) > 1.0  # and are hundreds of metres off over the 40 seeded poses


def test_the_presets_fail_the_renderer_gate_and_the_matrix_form_passes_it():
    for name in ("r_not_transposed", "no_rotation"):
        result = run_check("1.1.2.b", preset(name))
        assert not result.passed and "did not pass the gentle gate" in result.details[0], name
        assert not run_check("1.1.2.a", preset(name)).passed, name


def test_the_swap_preset_is_the_swap_and_the_other_leaves_the_rotation_out():
    pose, points = (3.0, -2.0, 1.1), np.array([[10.0, 4.0], [-6.0, 7.0]])
    assert np.allclose(preset("r_not_transposed")(pose, points), r_not_transposed(pose, points))
    assert np.allclose(preset("no_rotation")(pose, points), points - np.array(pose[:2]))


@pytest.fixture(scope="module")
def canned():
    return run(Car(), "gentle")


def test_the_preview_counts_pictures_and_pixels_that_differ(canned):
    assert differing_pictures(solution, canned) == (0, 0)
    swap_pictures, swap_pixels = differing_pictures(r_not_transposed, canned)
    none_pictures, none_pixels = differing_pictures(no_rotation, canned)
    assert (swap_pictures, none_pictures) == (17, 17) and swap_pixels > 0 and none_pixels > swap_pixels
    stats = get("1.1.2.b").build(r_not_transposed, canned)
    assert stats == {f"pictures that differ from the simulator's (of {PICTURES} sampled)": 17.0, "pixels that differ": float(swap_pixels)}
    assert get("1.1.2.a").build(solution, canned) == {"largest difference from the renderer's transform (m)": renderer_gap(solution)}


def test_the_provided_transform_in_the_camera_changes_nothing():
    assert Camera(world_to_ego=solution).world_to_ego is solution and Camera().world_to_ego is geom.world_to_ego
