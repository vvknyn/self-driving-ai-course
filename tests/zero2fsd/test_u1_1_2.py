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


def matrix_first(pose, p):  # R^T @ (p - t): right for one point, silently mixed up for 2 points stored as rows
    return geom.rotation_matrix(pose[2]).T @ (np.asarray(p, dtype=float) - np.array(pose[:2]))


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


def degrees(pose, p):  # the right transform with the yaw read as degrees
    return geom.world_to_ego((*pose[:2], math.radians(pose[2])), p)


def inverse(pose, p):  # ego_to_world handed in
    return geom.ego_to_world(pose, p)


def left_first(pose, p):
    return geom.world_to_ego(pose, p)[..., ::-1]


def left_flipped(pose, p):
    return geom.world_to_ego(pose, p) * (1, -1)


def sin_cos_swapped(pose, p):  # R with sin and cos exchanged
    c, s = math.sin(pose[2]), math.cos(pose[2])
    return (np.asarray(p, dtype=float) - np.array(pose[:2])) @ np.array([[c, -s], [s, c]])


def shrunk_yaw(pose, p):  # a mistake the grader does not know by name: the yaw scaled by 0.9
    return geom.world_to_ego((*pose[:2], 0.9 * pose[2]), p)


MISTAKES = {  # wrong answer -> the words its message must hold, and the section it must point to
    "swap": (r_not_transposed, "yaw pi/2: this is the swap", "Undo is the transpose"),
    "no rotation": (no_rotation, "yaw pi/2: this is no rotation", "Two dot products make a frame change"),
    "rotate first": (rotate_first, "turns first and subtracts t afterwards", "Undo is the transpose"),
    "degrees": (degrees, "reads the yaw as degrees", "The rotation matrix"),
    "inverse": (inverse, "this is ego_to_world", "Undo is the transpose"),
    "order": (left_first, "forward and left are in the wrong order", "Two dot products make a frame change"),
    "left flipped": (left_flipped, "left has the wrong sign", "Two dot products make a frame change"),
    "sin and cos swapped": (sin_cos_swapped, "yaw 0 has no rotation at all", "Two dot products make a frame change"),
    "yaw scaled by 0.9": (shrunk_yaw, "a rotation mistake: check R(θ)ᵀ, θ in radians", "The rotation matrix"),  # not in the grader's list
}


@pytest.mark.parametrize(("fn", "yaw"), [(r_not_transposed, "pi/2"), (no_rotation, "pi/2"), (rotate_first, "pi/2"), (left_first, "0")])
def test_a_wrong_transform_is_caught_at_the_first_yaw_that_catches_it(fn, yaw):
    result = run_check("1.1.2.a", fn)
    assert not result.passed and f"wrong at yaw {yaw}" in result.details[0]
    assert "1.1.2" not in "\n".join(result.details)  # never the dotted id


@pytest.mark.parametrize("mistake", MISTAKES)
def test_each_wrong_transform_is_named_for_the_mistake_it_makes_and_points_to_its_own_section(mistake):
    fn, words, section = MISTAKES[mistake]
    headline = run_check("1.1.2.a", fn).details[0]
    assert words in headline and f'the section "{section}"' in headline, headline
    assert "1.1.2" not in headline  # never the dotted id


def test_a_failure_at_yaw_zero_never_says_what_a_pass_there_means():
    headline = run_check("1.1.2.a", left_first).details[0]
    assert "yaw 0" in headline and "cannot tell" not in headline and "identity" not in headline


def test_a_pass_says_what_each_named_yaw_proved():
    message = run_check("1.1.2.a", solution).details[0]
    assert "yaw 0 cannot tell R from its transpose" in message and "yaw pi/2 tells R from its transpose" in message


def test_the_renderer_failure_sends_the_learner_back_to_exercise_a():
    assert "Run Exercise A first" in run_check("1.1.2.b", r_not_transposed).details[0]


def test_matrix_at_rows_is_named_when_two_points_run_without_an_error():
    result = run_check("1.1.2.a", matrix_first)
    assert not result.passed and "matrix @ rows" in result.details[0] and "Every point at once" in result.details[0]
    assert "1.1.2" not in result.details[0]


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
        assert not result.passed and "differs from the renderer's transform" in result.details[0], name
        assert not run_check("1.1.2.a", preset(name)).passed, name


def offset_by_a_hand(pose, p):  # right except 0.05 m too far left everywhere: small enough for the car to still drive
    return solution(pose, p) + np.array([0.0, 0.05])


def test_a_transform_the_drive_forgives_still_fails_the_entry_by_entry_comparison():
    from zero2fsd.grade._drive import gate
    assert gate(Car(), False, None, Camera(world_to_ego=offset_by_a_hand))  # the downstream drive alone passes it
    result = run_check("1.1.2.b", offset_by_a_hand)
    assert not result.passed and "differs from the renderer's transform" in result.details[0]


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
