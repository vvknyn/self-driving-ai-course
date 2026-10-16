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


@pytest.mark.parametrize("exercise_id", ["1.1.5.a", "1.1.5.b", "1.1.5.c"])
def test_the_untouched_skeleton_fails_and_the_solution_passes(exercise_id, solution, capsys):
    name = "ray_to_ground" if exercise_id.endswith("c") else "T_ego_cam"
    assert failure_text(exercise_id, skeleton, capsys)[0] is False
    assert failure_text(exercise_id, getattr(solution, name), capsys)[0] is True


@pytest.mark.parametrize("exercise_id", ["1.1.5.a", "1.1.5.b"])
@pytest.mark.parametrize("returned", [None, np.eye(3), [1, 2, 3]])
def test_a_mount_that_is_not_a_4x4_fails_first_with_what_to_build(exercise_id, returned, capsys):
    passed, text = failure_text(exercise_id, lambda height, pitch: returned, capsys)
    assert not passed and "build the 4×4 mount pose" in text


# --- a camera tipped up: the lesson's own mistake, on the mount (a), the drive (b) and a direction treated as a point (c) ---

def test_pitch_up_fails_a_by_name_and_b_on_the_drive(capsys):
    mistake = preset("1.1.5.a", "pitch_up")["T_ego_cam"]
    assert preset("1.1.5.b", "pitch_up")["T_ego_cam"](HEIGHT, PITCH).tolist() == mistake(HEIGHT, PITCH).tolist()
    passed, text = failure_text("1.1.5.a", mistake, capsys)
    assert not passed and "your camera tips up; a positive pitch must tip the nose down" in text
    passed, text = failure_text("1.1.5.b", mistake, capsys)
    assert not passed and "did not pass the gentle gate" in text and "left the lane after 86 of 703 m" in text


def test_a_direction_treated_as_a_point_fails_c_with_every_pixel_nan(capsys):
    passed, text = failure_text("1.1.5.c", preset("1.1.5.c", "direction_with_w1")["ray_to_ground"], capsys)
    assert not passed and "200 of 200 pixels below the horizon came back nan" in text


# --- non-vacuous: a mount that is only a little wrong draws other pictures and is rejected --------------------------------

@pytest.mark.parametrize("height, pitch, why", [(HEIGHT, PITCH + math.radians(1.0), "pitch off by 1 degree"), (HEIGHT + 0.1, PITCH, "height off by 0.1 m")])
def test_a_slightly_wrong_mount_draws_other_pictures_and_is_rejected_by_a_and_b(height, pitch, why, capsys):
    mount = geom.T_ego_cam(height, pitch)
    assert u1_1_5.pictures_that_differ(mount) > 0, why
    assert u1_1_5.pixel_gap(mount) > u1_1_5.PIXEL_TOL, why
    wrong = lambda h, p: geom.T_ego_cam(height, pitch)  # noqa: E731
    assert failure_text("1.1.5.a", wrong, capsys)[0] is False
    assert failure_text("1.1.5.b", wrong, capsys)[0] is False


def test_the_spec_mount_draws_the_simulators_own_pictures():
    assert u1_1_5.pictures_that_differ(u1_1_5.SPEC_MOUNT) == 0
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
    fn = solution.ray_to_ground if exercise_id.endswith("c") else solution.T_ego_cam
    assert list(preview.build(fn, canned)) == labels


def test_the_previews_give_the_lessons_numbers(solution, canned):
    def stats(exercise_id, fn):
        return get(exercise_id).build(fn, canned)
    spec_a = stats("1.1.5.a", solution.T_ego_cam)
    assert spec_a["ground distance seen in row 150 (m)"] == pytest.approx(2.9073, abs=1e-4)
    assert spec_a["largest pixel difference from the renderer (px)"] == 0.0
    assert stats("1.1.5.b", solution.T_ego_cam)["pictures that differ (of 18)"] == 0.0
    assert stats("1.1.5.c", solution.ray_to_ground)["largest ground difference (m)"] < 1e-9
    up = preset("1.1.5.a", "pitch_up")["T_ego_cam"]
    mistake = stats("1.1.5.a", up)
    assert mistake["ground distance seen in row 150 (m)"] == pytest.approx(4.976, abs=1e-3)
    assert mistake["largest pixel difference from the renderer (px)"] == pytest.approx(40.25, abs=0.01)
    assert stats("1.1.5.b", up)["pictures that differ (of 18)"] == 18.0
    nan_gap = stats("1.1.5.c", preset("1.1.5.c", "direction_with_w1")["ray_to_ground"])["largest ground difference (m)"]
    assert bool(np.isnan(nan_gap)) is True
