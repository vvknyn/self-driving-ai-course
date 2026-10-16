"""The camera, run, drive and gate seams: a camera can be handed the learner's transform, and Week 1 does not move."""
import math

import numpy as np
import pytest

from zero2fsd import geom
from zero2fsd.car import Car, run
from zero2fsd.grade._drive import drive, gate
from zero2fsd.sim import BicycleState, Camera, make_scenario


def rng(seed=0):
    return np.random.default_rng(seed)


def test_the_mount_projected_through_a_pinhole_is_the_renderers_camera():
    cam, mounted = Camera(), Camera(T_ego_cam=geom.T_ego_cam(1.4, math.radians(5)))
    r = rng(8)
    F, L = r.uniform(1.0, 60.0, 200), r.uniform(-8.0, 8.0, 200)
    assert np.allclose(mounted.ground_to_pixel(F, L), cam.ground_to_pixel(F, L), atol=1e-6)


def test_a_point_behind_the_lens_has_no_pixel_and_one_in_front_has():
    mounted = Camera(T_ego_cam=geom.T_ego_cam(1.4, math.radians(5)))
    u, v = mounted.ground_to_pixel(np.array([-5.0, 10.0]), np.array([1.0, 1.0]))
    assert np.isnan(u[0]) and np.isnan(v[0])
    assert np.isfinite(u[1]) and np.isfinite(v[1])


def test_the_gentle_drive_with_the_provided_car_is_still_1778_steps():
    tel = run(Car(), "gentle")
    assert tel.completed and len(tel.t) == 1778


def test_run_uses_the_camera_it_is_given():
    seen = []
    cam = Camera(world_to_ego=lambda pose, p: seen.append(pose) or geom.world_to_ego(pose, p))
    run(Car(), "straight", max_steps=2, camera=cam)
    assert len(seen) > 0


def test_a_camera_whose_world_to_ego_is_wrong_draws_a_different_picture():
    road, state = make_scenario("curvy").road, BicycleState(*make_scenario("curvy").road.ego_pose(150.0), 8.0, 0.0)
    right = Camera().render(road, state)
    wrong = Camera(world_to_ego=lambda pose, p: geom.world_to_ego((pose[0], pose[1], 0.0), p)).render(road, state)
    assert np.array_equal(Camera(world_to_ego=geom.world_to_ego).render(road, state), right)
    assert not np.array_equal(right, wrong)




class Reached(BaseException):
    """Raised by a camera hook so a test learns the camera was used without driving a whole lap."""


def reaching_camera():
    def hook(pose, p):
        raise Reached

    return Camera(world_to_ego=hook)


@pytest.mark.parametrize("call", [lambda cam: drive(Car(), "straight", "car", False, camera=cam), lambda cam: gate(Car(), False, camera=cam)], ids=["drive", "gate"])
def test_drive_and_gate_hand_their_camera_to_the_run(call):
    with pytest.raises(Reached):
        call(reaching_camera())
