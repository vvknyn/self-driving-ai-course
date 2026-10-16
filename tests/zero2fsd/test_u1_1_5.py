"""1.1.5 The camera's point of view: the skeleton fails, the solution passes, each preset fails for the check it is named for,
a mount that is merely close is told apart from the real one, and the previews give the numbers the lesson prints."""
import math
import re
from pathlib import Path

import numpy as np
import pytest

from _course import REPO, load_solution
from zero2fsd import geom
from zero2fsd.car import Car, run
from zero2fsd.grade import check
from zero2fsd.grade.units import u1_1_5
from zero2fsd.preview import get
from zero2fsd.sim import Camera

PRESETS = REPO / "course" / "1.1.5-the-cameras-point-of-view" / "presets"
HEIGHT, PITCH = u1_1_5.HEIGHT_M, u1_1_5.PITCH_RAD


@pytest.fixture(scope="module")
def solution():
    return load_solution(REPO, "1.1.5")


def preset(exercise_id, name):
    namespace: dict = {}
    exec(compile((PRESETS / exercise_id / f"{name}.py").read_text(), name, "exec"), namespace)
    return namespace


def failure_text(exercise_id, fn, capsys):
    capsys.readouterr()
    passed = check(exercise_id, fn, show=False).passed
    return passed, re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)


def skeleton(*_args):
    raise NotImplementedError


def pair(mount_fn):
    """Exercise B's target: the learner's world_to_ego (here the provided one) and their mount."""
    return (geom.world_to_ego, mount_fn)


def target(exercise_id, mount_fn):
    return pair(mount_fn) if exercise_id.endswith("b") else mount_fn


@pytest.mark.parametrize("exercise_id", ["1.1.5.a", "1.1.5.b", "1.1.5.c"])
def test_the_untouched_skeleton_fails_and_the_solution_passes(exercise_id, solution, capsys):
    name = "ray_to_ground" if exercise_id.endswith("c") else "T_ego_cam"
    assert failure_text(exercise_id, target(exercise_id, skeleton), capsys)[0] is False
    assert failure_text(exercise_id, solution.camera_mount if exercise_id.endswith("b") else getattr(solution, name), capsys)[0] is True


@pytest.mark.parametrize("exercise_id", ["1.1.5.a", "1.1.5.b"])
@pytest.mark.parametrize("returned, words", [(None, "returned None"), (np.eye(3), "build the 4×4 mount pose"), ([1, 2, 3], "build the 4×4 mount pose")])
def test_a_mount_that_is_not_a_4x4_fails_first_with_what_to_build(exercise_id, returned, words, capsys):
    passed, text = failure_text(exercise_id, target(exercise_id, lambda height, pitch: returned), capsys)
    assert not passed and words in text


# --- a camera tipped up: the lesson's own mistake, on the mount (a), the drive (b) and a direction treated as a point (c) ---

def test_pitch_up_fails_a_by_name_and_b_on_the_drive(capsys):
    mistake = preset("1.1.5.a", "pitch_up")["T_ego_cam"]
    assert preset("1.1.5.b", "pitch_up")["camera_mount"][1](HEIGHT, PITCH).tolist() == mistake(HEIGHT, PITCH).tolist()
    passed, text = failure_text("1.1.5.a", mistake, capsys)
    assert not passed and "your camera tips up; a positive pitch must tip the nose down" in text
    passed, text = failure_text("1.1.5.b", preset("1.1.5.b", "pitch_up")["camera_mount"], capsys)
    assert not passed and "your camera tips up; a positive pitch must tip the nose down" in text and "margin" not in text


def test_a_direction_treated_as_a_point_fails_c_with_every_pixel_nan(capsys):
    passed, text = failure_text("1.1.5.c", preset("1.1.5.c", "direction_with_w1")["ray_to_ground"], capsys)
    assert not passed and "200 of 200 pixels below the horizon came back nan" in text


# --- non-vacuous: a mount that is only a little wrong draws other pictures and is rejected --------------------------------

@pytest.mark.parametrize("height, pitch, why", [(HEIGHT, PITCH + math.radians(1.0), "pitch off by 1 degree"), (HEIGHT + 0.1, PITCH, "height off by 0.1 m")])
def test_a_slightly_wrong_mount_draws_other_pictures_and_is_rejected_by_a_and_b(height, pitch, why, capsys):
    mount = geom.T_ego_cam(height, pitch)
    assert u1_1_5.pictures_that_differ(Camera(T_ego_cam=mount)) > 0, why
    assert u1_1_5.pixel_gap(mount) > u1_1_5.PIXEL_TOL, why
    wrong = lambda h, p: geom.T_ego_cam(height, pitch)  # noqa: E731
    assert failure_text("1.1.5.a", wrong, capsys)[0] is False
    assert failure_text("1.1.5.b", pair(wrong), capsys)[0] is False


def test_the_spec_mount_draws_the_simulators_own_pictures():
    assert u1_1_5.pictures_that_differ(Camera(T_ego_cam=u1_1_5.SPEC_MOUNT)) == 0
    assert u1_1_5.pixel_gap(u1_1_5.SPEC_MOUNT) == 0.0


# --- margins: a within 1e-6 px, c within 1e-9 m, nan above the horizon --------------------------------------------------

@pytest.mark.parametrize("nudge, passes", [(0.0, True), (1e-9, True), (1e-3, False)])
def test_a_clears_1e_minus_6_pixels_and_fails_a_milli_radian_off(nudge, passes, capsys):
    mount = lambda h, p: geom.T_ego_cam(h, p + nudge)  # noqa: E731
    assert failure_text("1.1.5.a", mount, capsys)[0] is passes


@pytest.mark.parametrize("scale, passes", [(1.0, True), (1.0 + 1e-12, True), (1.0 + 1e-6, False)])
def test_c_clears_1e_minus_9_metres_and_fails_a_micro_scale_error(scale, passes, capsys):
    def scaled(T, ray):
        return geom.ray_to_ground(T, ray) * np.array([scale, scale, 1.0])
    assert failure_text("1.1.5.c", scaled, capsys)[0] is passes


def test_c_wants_nan_above_the_horizon(capsys):
    def zeros_for_sky(T, ray):  # right below the horizon; above it, 0 instead of nan
        return np.nan_to_num(geom.ray_to_ground(T, ray))
    passed, text = failure_text("1.1.5.c", zeros_for_sky, capsys)
    assert not passed and "pixels above the horizon got a road point" in text
    horizon = Camera().cy - Camera().f * math.tan(PITCH)
    above = geom.ray_to_ground(u1_1_5.SPEC_MOUNT, geom.pixel_ray(Camera(), 159.5, horizon - 1.0))
    assert bool(np.isnan(above[0])) is True


# --- previews ------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def canned():
    return run(Car(), "gentle", max_steps=20)


@pytest.mark.parametrize("exercise_id, labels", [
    ("1.1.5.a", ["ground distance seen in row 150 (m)", "largest pixel difference from the renderer (px)"]),
    ("1.1.5.b", ["pictures that differ (of 18)"]),
    ("1.1.5.c", ["largest ground difference (m)"]),
])
def test_the_previews_are_labelled_stats_on_gentle(exercise_id, labels, solution, canned):
    preview = get(exercise_id)
    assert (preview.kind, preview.road) == ("stats", "gentle")
    fn = {"c": solution.ray_to_ground, "b": solution.camera_mount}.get(exercise_id[-1], solution.T_ego_cam)
    assert list(preview.build(fn, canned)) == labels


def test_the_previews_give_the_lessons_numbers(solution, canned):
    def stats(exercise_id, fn):
        return get(exercise_id).build(fn, canned)
    spec_a = stats("1.1.5.a", solution.T_ego_cam)
    assert spec_a["ground distance seen in row 150 (m)"] == pytest.approx(2.9073, abs=1e-4)
    assert spec_a["largest pixel difference from the renderer (px)"] == 0.0
    assert stats("1.1.5.b", solution.camera_mount)["pictures that differ (of 18)"] == 0.0
    assert stats("1.1.5.c", solution.ray_to_ground)["largest ground difference (m)"] < 1e-9
    up = preset("1.1.5.a", "pitch_up")["T_ego_cam"]
    mistake = stats("1.1.5.a", up)
    assert mistake["ground distance seen in row 150 (m)"] == pytest.approx(4.976, abs=1e-3)
    assert mistake["largest pixel difference from the renderer (px)"] == pytest.approx(40.25, abs=0.01)
    assert stats("1.1.5.b", preset("1.1.5.b", "pitch_up")["camera_mount"])["pictures that differ (of 18)"] == 18.0
    nan_gap = stats("1.1.5.c", preset("1.1.5.c", "direction_with_w1")["ray_to_ground"])["largest ground difference (m)"]
    assert bool(np.isnan(nan_gap)) is True


# --- the 4×4 is checked as a rigid transform and entry by entry, before any pixel ----------------------------------------

def mount_with(edit):
    def build(height, pitch):
        T = geom.T_ego_cam(height, pitch)
        edit(T)
        return T
    return build


def zero_last_row(T):
    T[3] = 0.0


def scale_turn(T):
    T[:3, :3] *= 1.01


def mirror_turn(T):
    T[:3, 0] *= -1.0


@pytest.mark.parametrize("edit, words", [
    (zero_last_row, "the last row of your 4×4 must be 0 0 0 1 (the trailing 1 is what lets a point be shifted); yours is 0 0 0 0"),
    (scale_turn, "is not a pure turn"),
    (mirror_turn, "mirrors instead of turning"),
    (lambda T: T.__setitem__((0, 3), math.nan), "nan or inf"),
])
@pytest.mark.parametrize("exercise_id", ["1.1.5.a", "1.1.5.b"])
def test_a_4x4_that_is_not_a_rigid_transform_is_named_before_it_is_used(exercise_id, edit, words, capsys):
    passed, text = failure_text(exercise_id, target(exercise_id, mount_with(edit)), capsys)
    assert not passed and words in text and "RuntimeWarning" not in text and "margin" not in text


def test_a_3x3_block_off_by_a_hair_is_told_apart_entry_by_entry(capsys):
    passed, text = failure_text("1.1.5.a", mount_with(lambda T: T.__setitem__((0, 3), 1e-7)), capsys)
    assert not passed and "differs from the reference mount in row 1, column 4" in text


@pytest.mark.parametrize("build, words", [
    (lambda h, p: geom.T_ego_cam(h, -p), "your camera tips up"),
    (lambda h, p: u1_1_5.mount_from(geom.CAM_FROM_EGO_AXES.T @ geom.rot_y(p), h), "the wrong order, so the camera looks level and to the right"),
    (lambda h, p: u1_1_5.mount_from(geom.T_ego_cam(h, p)[:3, :3].T, h), "is transposed"),
    (lambda h, p: geom.invert_pose(geom.T_ego_cam(h, p)), "the inverse mount, camera-from-ego"),
    (lambda h, p: geom.T_ego_cam(h, math.radians(p)), "treats the pitch as degrees"),
])
@pytest.mark.parametrize("exercise_id", ["1.1.5.a", "1.1.5.b"])
def test_each_warned_mistake_is_named_by_its_own_message(exercise_id, build, words, capsys):
    passed, text = failure_text(exercise_id, target(exercise_id, build), capsys)
    assert not passed and words in text


def test_a_wrong_height_is_named_as_a_height(capsys):
    passed, text = failure_text("1.1.5.a", lambda h, p: geom.T_ego_cam(1.5, p), capsys)
    assert not passed and "your lens sits at (0, 0, 1.5)" in text and "1.4 m straight above the ego origin" in text


def test_a_lens_axis_off_the_mount_is_told_in_degrees(capsys):
    passed, text = failure_text("1.1.5.a", lambda h, p: geom.T_ego_cam(h, p + math.radians(2.0)), capsys)
    assert not passed and "your lens looks straight ahead and 7.0° down; it must look straight ahead and 5.0° down" in text


# --- B drives with the learner's world_to_ego as well as the mount -------------------------------------------------------

def world_to_ego_transposed(pose, p_world):  # R in place of Rᵀ: turns the wrong way
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    return (np.asarray(p_world, dtype=float) - (x, y)) @ np.array([[c, -s], [s, c]]).T


def world_to_ego_mirrored(pose, p_world):  # left and right swapped
    return geom.world_to_ego(pose, p_world) * np.array([1.0, -1.0])


@pytest.mark.parametrize("world_to_ego", [world_to_ego_transposed, world_to_ego_mirrored])
def test_b_fails_a_wrong_world_to_ego_with_a_correct_mount(world_to_ego, solution, capsys):
    passed, text = failure_text("1.1.5.b", (world_to_ego, solution.T_ego_cam), capsys)
    assert not passed and "pictures differently from the simulator's own" in text and "look at world_to_ego" in text and "margin" not in text


@pytest.mark.parametrize("answer", [geom.world_to_ego, 5, (geom.world_to_ego,), ("a", "b")])
def test_b_wants_the_pair_of_functions(answer, capsys):
    passed, text = failure_text("1.1.5.b", answer, capsys)
    assert not passed and "the pair of your two functions" in text


def test_a_transposed_mount_in_b_prints_no_runtime_warning(capsys):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        passed, text = failure_text("1.1.5.b", pair(mount_with(lambda T: T.__setitem__((slice(0, 3), slice(0, 3)), T[:3, :3].T))), capsys)
    assert not passed and "RuntimeWarning" not in text and "Warning" not in text


# --- C: a single nan, an off-road point, an all-nan answer ---------------------------------------------------------------

def test_c_names_a_single_nan_above_the_horizon_without_a_shape(capsys):
    passed, text = failure_text("1.1.5.c", lambda T, ray: np.nan_to_num(geom.ray_to_ground(T, ray), nan=0.0) if np.isfinite(geom.ray_to_ground(T, ray)[0]) else math.nan, capsys)  # right below the horizon; above it one nan where three belong
    assert not passed and "a single number" in text and "shape ()" not in text and "np.full(3, np.nan)" in text


def test_c_names_a_point_above_the_road(capsys):
    passed, text = failure_text("1.1.5.c", lambda T, ray: geom.ray_to_ground(T, ray) + np.array([0.0, 0.0, 0.5]), capsys)
    assert not passed and "m above the road" in text


def test_c_gives_a_cause_only_when_every_pixel_is_nan(capsys):
    passed, text = failure_text("1.1.5.c", preset("1.1.5.c", "direction_with_w1")["ray_to_ground"], capsys)
    assert not passed and "(w = 1)" in text
    passed, text = failure_text("1.1.5.c", lambda T, ray: np.full(3, np.nan), capsys)  # all nan, but not the shifted ray's answer
    assert not passed and "(w = 1)" not in text and "return nan only when" in text


# --- the mistakes the lecture names, the pair's order, and C at other mounts -------------------------------------------------

@pytest.mark.parametrize("exercise_id", ["1.1.5.a", "1.1.5.b"])
def test_rot_x_in_place_of_rot_y_is_named_as_a_roll(exercise_id, capsys):
    build = lambda h, p: u1_1_5.mount_from(geom.rot_x(p) @ geom.CAM_FROM_EGO_AXES.T, h)  # noqa: E731
    passed, text = failure_text(exercise_id, target(exercise_id, build), capsys)
    assert not passed and "your turn is about x (a roll)" in text


def test_the_pair_in_the_wrong_order_says_so(solution, capsys):
    passed, text = failure_text("1.1.5.b", (solution.T_ego_cam, geom.world_to_ego), capsys)
    assert not passed and "the pair goes (world_to_ego, T_ego_cam), in that order" in text and "TypeError" not in text


def ignores_the_turn(T, ray):  # the 5° mount's turn written in: right at the spec mount only
    return _ground(u1_1_5.SPEC_MOUNT[:3, :3], T[:3, 3], ray)


def ignores_the_height(T, ray):  # the 1.4 m written in: right at the spec mount only
    return _ground(T[:3, :3], np.array([0.0, 0.0, HEIGHT]), ray)


def _ground(turn, lens, ray):
    r = turn @ np.asarray(ray, dtype=float)
    return np.full(3, np.nan) if r[2] >= 0 else lens + (-lens[2] / r[2]) * r


@pytest.mark.parametrize("stub", [ignores_the_turn, ignores_the_height])
def test_c_fails_a_ray_to_ground_that_ignores_part_of_the_mount_it_is_given(stub, capsys):
    assert max(u1_1_5.ground_gaps(u1_1_5.ground_answers(stub))) < 1e-9  # right at the spec mount: the old grader passed it
    passed, text = failure_text("1.1.5.c", stub, capsys)
    assert not passed and "your road point does not change when the mount changes: use the T you are given" in text


def test_c_names_the_lens_shift_only_when_the_answer_is_the_shifted_ray(capsys):
    passed, text = failure_text("1.1.5.c", preset("1.1.5.c", "direction_with_w1")["ray_to_ground"], capsys)
    assert not passed and "picked up the lens's shift (w = 1)" in text

    def never_turned(T, ray):  # the camera-numbers ray used as it is: its third number is +1
        return _ground(np.eye(3), T[:3, 3], ray)
    passed, text = failure_text("1.1.5.c", never_turned, capsys)
    assert not passed and "w = 1" not in text and "every ray points up or level" not in text and "return nan only when" in text

    def inverted_guard(T, ray):  # nan for the rays that look down, a point for the rest
        r = T[:3, :3] @ ray
        return np.full(3, np.nan) if r[2] < 0 else T[:3, 3] - T[2, 3] / r[2] * r  # above the horizon it still computes lam r
    passed, text = failure_text("1.1.5.c", inverted_guard, capsys)
    assert not passed and "the nan test is the wrong way round" in text and "w = 1" not in text


def test_c_names_a_road_point_that_is_off_the_road_from_what_it_matches(capsys):
    def shift_then_turn(T, ray):
        r = T[:3, :3] @ ray
        return T[:3, :3] @ (-T[2, 3] / r[2] * ray + T[:3, 3])
    def no_lens(T, ray):
        r = T[:3, :3] @ ray
        return -T[2, 3] / r[2] * r
    for stub, words in ((shift_then_turn, "turned the sum"), (no_lens, "left out the lens position")):
        passed, text = failure_text("1.1.5.c", stub, capsys)
        assert not passed and words in text and "guarantees" not in text


def test_c_wants_three_nans_above_the_horizon_not_two(capsys):
    def two_nans(T, ray):
        out = geom.ray_to_ground(T, ray)
        return np.array([np.nan, np.nan, 0.0]) if np.isnan(out[0]) else out
    passed, text = failure_text("1.1.5.c", two_nans, capsys)
    assert not passed and "pixels above the horizon got a road point" in text


def test_every_starter_runs_after_the_playground_prelude(solution):
    """The real build path (scripts/build_playground.exercise_records' starter) for each exercise, run the way the page runs it: no check() defined."""
    import importlib.util
    from _course import parse_lab, load_yaml
    spec = importlib.util.spec_from_file_location("playground_shim", REPO / "playground" / "shim.py")
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    folder = REPO / "course" / "1.1.5-the-cameras-point-of-view"
    cells = parse_lab((folder / "lab.py").read_text(encoding="utf-8"))
    for exercise in load_yaml(folder / "unit.yaml")["exercises"]:
        starter = "\n\n".join(c.source for c in cells if c.kind == "code" and f"exercise:{exercise['id']}" in c.tags)
        assert "check(" not in starter, exercise["id"]
        shim.learner_object(starter, exercise["name"])
