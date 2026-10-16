"""geom.py is the provided answer to the module 1.1 labs; the simulator's own code is its grader."""
import math

import numpy as np
import pytest

from zero2fsd import geom
from zero2fsd.sim import BicycleState, Camera, make_scenario
from zero2fsd.sim.camera import to_vehicle
from zero2fsd.sim.dynamics import WHEELBASE
from zero2fsd.sim.world import EGO_LANE_CENTER, wrap_angle

TAU = 2 * math.pi


def rng(seed=0):
    return np.random.default_rng(seed)


def centerline_of(road):
    """The polyline a Road was built from (an open road keeps its last vertex in `_b`)."""
    return road._a if road.closed else np.vstack([road._a, road._b[-1]])


ROADS = {name: make_scenario(name).road for name in ("gentle", "curvy")}
L_CORNER = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])


# --- positions and headings ---------------------------------------------------------------------------------------

def test_heading_vector_is_the_unit_vector_and_broadcasts():
    assert np.allclose(geom.heading_vector(math.pi / 2), [0.0, 1.0], atol=1e-15)
    assert geom.heading_vector(np.zeros((3, 4))).shape == (3, 4, 2)


@pytest.mark.parametrize("v, angle", [((1, 0), 0.0), ((0, 1), math.pi / 2), ((-1, 0), math.pi), ((0, -1), -math.pi / 2), ((-1, -1), -3 * math.pi / 4)])
def test_direction_angle_names_the_quadrants_and_both_halves_of_pi(v, angle):
    assert geom.direction_angle(v) == pytest.approx(angle, abs=1e-15)


def test_direction_angle_of_the_zero_vector_is_nan_not_zero():
    assert bool(np.isnan(geom.direction_angle((0.0, 0.0)))) is True
    got = geom.direction_angle(np.array([[0.0, 0.0], [0.0, 2.0]]))
    assert np.isnan(got[0]) and got[1] == pytest.approx(math.pi / 2)


def test_direction_angle_matches_arctan2_on_seeded_vectors():
    v = rng().normal(size=(30, 2))
    assert np.array_equal(geom.direction_angle(v), np.arctan2(v[:, 1], v[:, 0]))


def test_wrap_angle_is_the_simulators_wrap():
    a = rng(1).uniform(-20, 20, 50)
    assert np.array_equal(geom.wrap_angle(a), wrap_angle(a)) and geom.wrap_angle(math.pi) == -math.pi


def test_forward_left_matches_the_cameras_transform_with_the_translation_removed():
    r = rng(2)
    for _ in range(40):
        x, y, yaw = r.uniform(-50, 50, 3)
        v = r.normal(size=(5, 2))
        want = np.stack(to_vehicle(BicycleState(0.0, 0.0, yaw, 0.0, 0.0), v[:, 0], v[:, 1]), axis=-1)
        assert np.allclose(geom.forward_left(yaw, v), want, atol=1e-9)


# --- the view from the driver's seat ------------------------------------------------------------------------------

def test_rotation_matrix_turns_x_into_the_heading_vector():
    for theta in (0.0, 0.7, -2.5):
        R = geom.rotation_matrix(theta)
        assert np.allclose(R @ [1.0, 0.0], geom.heading_vector(theta)) and np.allclose(R.T @ R, np.eye(2))


def test_world_to_ego_is_bit_identical_to_the_old_to_vehicle_arithmetic_on_1000_triples():
    r = rng(3)
    for _ in range(1000):
        x, y, yaw, px, py = r.uniform(-200, 200, 5)
        state = BicycleState(x, y, yaw, 0.0, 0.0)
        c, s = math.cos(yaw), math.sin(yaw)  # the arithmetic camera.to_vehicle had before it delegated to geom
        dx, dy = px - x, py - y
        old = (dx * c + dy * s, -dx * s + dy * c)
        assert tuple(geom.world_to_ego((x, y, yaw), (px, py))) == old
        assert to_vehicle(state, px, py) == old


def test_world_to_ego_takes_any_leading_shape_and_round_trips_through_ego_to_world():
    pose = (3.0, -4.0, -3 * math.pi / 4)
    p = rng(4).normal(size=(2, 5, 2)) * 10
    assert geom.world_to_ego(pose, p).shape == (2, 5, 2)
    assert np.allclose(geom.ego_to_world(pose, geom.world_to_ego(pose, p)), p, atol=1e-12)
    assert np.allclose(geom.world_to_ego((0, 0, math.pi / 2), (0.0, 1.0)), [1.0, 0.0], atol=1e-15)  # straight ahead


# --- the road's own coordinates -----------------------------------------------------------------------------------

def test_project_to_segment_clamps_and_signs_the_distance():
    a, b = (0.0, 0.0), (10.0, 0.0)
    assert geom.project_to_segment(a, b, (4.0, 3.0)) == (4.0, 3.0)  # left of the direction of travel: positive
    assert geom.project_to_segment(a, b, (4.0, -3.0)) == (4.0, -3.0)
    assert geom.project_to_segment(a, b, (4.0, 0.0)) == (4.0, 0.0)  # on the line
    assert geom.project_to_segment(a, b, (13.0, 4.0)) == (10.0, 5.0)  # beyond the end: clamped, d is the distance to the end point


@pytest.mark.parametrize("name", ["gentle", "curvy"])
def test_to_road_frame_matches_road_project_on_seeded_points(name):
    road, r = ROADS[name], rng(5)
    line = centerline_of(road)
    for _ in range(40):
        p = road.offset(r.uniform(0, road.length), r.uniform(-6, 6)) + r.normal(size=2)
        assert np.allclose(geom.to_road_frame(line, road.closed, p), road.project(*p), atol=1e-9)


def test_to_road_frame_in_an_l_corner_and_beyond_an_open_roads_end():
    assert geom.to_road_frame(L_CORNER, False, (5.0, 2.0)) == (5.0, 2.0)  # left of the first leg
    assert geom.to_road_frame(L_CORNER, False, (12.0, 5.0)) == (15.0, -2.0)  # right of the second leg
    assert geom.to_road_frame(L_CORNER, False, (10.0, 14.0)) == (20.0, 4.0)  # past the end: t clamps, d is the distance to the end point


@pytest.mark.parametrize("name", ["gentle", "curvy"])
def test_lane_errors_match_ego_truth_including_poses_a_full_turn_away(name):
    road, r = ROADS[name], rng(6)
    line = centerline_of(road)
    for k in range(40):
        x, y = road.offset(r.uniform(0, road.length), r.uniform(-5, 2))
        yaw = r.uniform(-math.pi, math.pi) + (k % 3 - 1) * TAU  # +2pi and -2pi poses: the wrap trap
        e, psi = geom.lane_errors(line, road.closed, (x, y, yaw))
        _, e_true, psi_true = road.ego_truth(x, y, yaw)
        assert (e, psi) == pytest.approx((e_true, psi_true), abs=1e-9)
        assert -math.pi <= psi < math.pi


def test_at_a_corner_equidistant_from_two_segments_either_adjacent_heading_is_right():
    """A point on the L-corner's bisector is equally near both legs (by design): s and d are unambiguous, psi may be either leg's."""
    p = (10.0 - 3.0, 3.0)  # inside the turn, on the bisector of the corner vertex (10, 0)
    s, d = geom.to_road_frame(L_CORNER, False, p)
    assert (s, d) == pytest.approx((7.0, 3.0))  # nearest to leg 1 at (7, 0) AND to leg 2 at (10, 3): leg 1 wins by 0 in s
    _, psi = geom.lane_errors(L_CORNER, False, (*p, 0.3))
    assert any(psi == pytest.approx(0.3 - heading, abs=1e-9) for heading in (0.0, math.pi / 2))


# --- poses you can chain ------------------------------------------------------------------------------------------

def test_pose_matrix_round_trip():
    pose = (1.5, -2.0, -2.9)
    assert geom.matrix_to_pose(geom.pose_to_matrix(pose)) == pytest.approx(pose, abs=1e-15)
    assert geom.matrix_to_pose(geom.pose_to_matrix((0.0, 0.0, math.pi)))[2] == pytest.approx(math.pi)


def rigid(k, r):
    """A random (k+1, k+1) rigid transform: a rotation (QR of a random matrix, det fixed to +1) and a translation."""
    Q, _ = np.linalg.qr(r.normal(size=(k, k)))
    Q[:, 0] *= np.linalg.det(Q)
    T = np.eye(k + 1)
    T[:k, :k], T[:k, k] = Q, r.normal(size=k) * 5
    return T


@pytest.mark.parametrize("k", [2, 3])
def test_invert_pose_is_the_general_closed_form_and_matches_linalg_inv(k):
    r = rng(7)
    for _ in range(20):
        T = rigid(k, r)
        assert np.allclose(geom.invert_pose(T), np.linalg.inv(T), atol=1e-12)  # only a test may call np.linalg.inv
        assert np.allclose(geom.invert_pose(T) @ T, np.eye(k + 1), atol=1e-12)


def test_relative_pose_names_a_pair_that_does_not_commute_and_the_identity():
    a, b = geom.pose_to_matrix((1.0, 2.0, math.pi / 2)), geom.pose_to_matrix((4.0, 2.0, 0.0))
    assert geom.matrix_to_pose(geom.relative_pose(a, b)) == pytest.approx((0.0, -3.0, -math.pi / 2), abs=1e-12)
    assert not np.allclose(a @ b, b @ a)
    assert np.allclose(geom.relative_pose(a, a), np.eye(3), atol=1e-15)
    assert np.allclose(a @ geom.relative_pose(a, b), b, atol=1e-12)


def test_dead_reckon_keeps_the_start_and_multiplies_on_the_right():
    start = geom.pose_to_matrix((1.0, 1.0, 0.5))
    steps = [geom.pose_to_matrix((1.0, 0.0, 0.3)), geom.pose_to_matrix((0.0, 1.0, -0.2)), geom.pose_to_matrix((2.0, 0.0, 0.1))]
    out = geom.dead_reckon(start, steps)
    assert out.shape == (4, 3, 3) and np.array_equal(out[0], start)
    assert np.allclose(out[3], start @ steps[0] @ steps[1] @ steps[2], atol=1e-15)


def test_t_ego_rear_is_the_wheelbase_behind_the_front_axle():
    assert geom.matrix_to_pose(geom.T_EGO_REAR) == (-WHEELBASE, 0.0, 0.0)


# --- the camera's point of view -----------------------------------------------------------------------------------

def test_rotations_follow_the_right_hand_rule():
    assert np.allclose(geom.rot_z(math.pi / 2) @ [1, 0, 0], [0, 1, 0], atol=1e-15)
    assert np.allclose(geom.rot_x(math.pi / 2) @ [0, 1, 0], [0, 0, 1], atol=1e-15)
    assert np.allclose(geom.rot_y(math.pi / 2) @ [0, 0, 1], [1, 0, 0], atol=1e-15)


def test_t_ego_cam_puts_the_lens_at_height_and_looks_forward_and_down():
    T = geom.T_ego_cam(1.4, math.radians(5))
    assert np.allclose(T[:3, 3], [0, 0, 1.4]) and np.allclose(T[3], [0, 0, 0, 1])
    axis = geom.transform_directions(T, [0.0, 0.0, 1.0])  # the optical axis, in ego
    assert np.allclose(axis, [math.cos(math.radians(5)), 0.0, -math.sin(math.radians(5))], atol=1e-15)
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-15)


def test_transform_points_counts_the_translation_and_directions_do_not():
    T = geom.pose_to_matrix((1.0, 2.0, math.pi / 2))
    assert np.allclose(geom.transform_points(T, [1.0, 0.0]), [1.0, 3.0], atol=1e-15)
    assert np.allclose(geom.transform_directions(T, [1.0, 0.0]), [0.0, 1.0], atol=1e-15)
    assert geom.transform_points(geom.T_ego_cam(1.4, 0.1), np.zeros((4, 5, 3))).shape == (4, 5, 3)


def test_ray_to_ground_matches_pixel_to_ground_and_gives_nan_above_the_horizon():
    cam, T = Camera(), geom.T_ego_cam(1.4, math.radians(5))
    r = rng(9)
    u, v = r.uniform(0, 319, 200), r.uniform(100, 179, 200)  # below the horizon (row ~ 89.5 - f tan 5deg = 74)
    ground = geom.ray_to_ground(T, geom.pixel_ray(cam, u, v))
    assert np.allclose(ground[:, :2], np.stack(cam.pixel_to_ground(u, v), axis=-1), atol=1e-9) and np.allclose(ground[:, 2], 0.0, atol=1e-12)
    assert np.isnan(geom.ray_to_ground(T, geom.pixel_ray(cam, 160.0, 10.0))).all()
    assert geom.pixel_ray(cam, 159.5, 89.5).tolist() == [0.0, 0.0, 1.0]


def test_all_lists_every_public_name_once_so_the_playground_prelude_cannot_drift():
    defined_here = {n for n, v in vars(geom).items() if not n.startswith("_") and (n.isupper() or getattr(v, "__module__", None) == geom.__name__)}
    assert len(geom.__all__) == len(set(geom.__all__)) and defined_here <= set(geom.__all__)
    assert all(hasattr(geom, name) for name in geom.__all__) and "EGO_LANE_CENTER" in geom.__all__
