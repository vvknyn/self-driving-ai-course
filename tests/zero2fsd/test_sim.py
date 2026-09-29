import math
import time

import numpy as np
import pytest

from zero2fsd.sim import (
    BicycleState,
    Camera,
    Road,
    make_scenario,
    render_topdown,
    step,
)
from zero2fsd.sim.camera import PALETTE, ROAD, SKY, GRASS, WHITE, YELLOW
from zero2fsd.sim.scenarios import CENTERLINE_STEP, RUN_OUT
from zero2fsd.sim.world import EGO_LANE_CENTER


@pytest.fixture(scope="module")
def cam():
    return Camera()


def _state_on(road, s, offset=0.0, yaw_err=0.0, speed=8.0):
    x, y, yaw = road.ego_pose(s, offset, yaw_err)
    return BicycleState(x, y, yaw, speed, 0.0)


# --- camera geometry -----------------------------------------------------------------

def test_pixel_ground_round_trip(cam):
    rng = np.random.default_rng(1)
    fwd = rng.uniform(1, 50, 200)
    left = rng.uniform(-8, 8, 200)
    u, v = cam.ground_to_pixel(fwd, left)
    f2, l2 = cam.pixel_to_ground(u, v)
    assert np.allclose(f2, fwd, atol=1e-6) and np.allclose(l2, left, atol=1e-6)


def test_horizon_row_and_nan_above_it(cam):
    _, v_far = cam.ground_to_pixel(1e9, 0.0)
    assert abs(v_far - 76) <= 1
    f, l = cam.pixel_to_ground(np.array([100.0, 100.0]), np.array([10.0, v_far]))
    assert np.isnan(f).all() and np.isnan(l).all()
    f, l = cam.pixel_to_ground(160.0, 150.0)
    assert f > 0 and math.isfinite(l)  # scalar in, scalar out


def test_sign_convention_left_is_lower_column_index(cam):
    (u_left, _), (u_right, _) = cam.ground_to_pixel(10.0, 2.0), cam.ground_to_pixel(10.0, -2.0)
    assert u_left < u_right


# --- rendering -----------------------------------------------------------------------

def test_lane_centre_render_has_yellow_left_of_white(cam):
    road = make_scenario("straight").road
    frame, lab = cam.render(road, _state_on(road, 20.0), return_labels=True)
    assert frame.shape == (180, 320, 3) and frame.dtype == np.uint8 and lab.shape == (180, 320)
    assert set(np.unique(lab)) == {SKY, GRASS, ROAD, YELLOW, WHITE}
    assert np.nonzero(lab == YELLOW)[1].mean() < 160 < np.nonzero(lab == WHITE)[1].mean()
    assert (frame == PALETTE[lab]).all()  # noise off: frame is exactly the palette of the labels


def test_yellow_is_dashed_along_its_length(cam):
    road = make_scenario("straight").road
    _, lab = cam.render(road, _state_on(road, 20.0), return_labels=True)
    rows = np.unique(np.nonzero(lab == YELLOW)[0])
    assert len(rows) < rows.max() - rows.min() + 1  # some rows between first and last have no paint


def test_white_is_solid_and_lines_have_paint_at_every_near_row(cam):
    road = make_scenario("straight").road
    _, lab = cam.render(road, _state_on(road, 20.0), return_labels=True)
    white_rows = np.unique(np.nonzero(lab == WHITE)[0])
    assert (np.diff(white_rows) == 1).all()


@pytest.mark.parametrize("offset,yaw_err", [(0.0, 0.0), (-1.8, 0.0), (0.9, 0.6), (-0.9, -0.6), (0.0, 1.2)])
def test_curvy_never_paints_road_above_the_horizon(cam, offset, yaw_err):
    # Review Focus 4: the road bends behind the camera; near-plane clipping must hold.
    road = make_scenario("curvy").road
    for s in np.arange(0.0, road.length, 1.0)[::10]:
        _, lab = cam.render(road, _state_on(road, s, offset, yaw_err), return_labels=True)
        above = lab[:70]
        assert np.isin(above, [SKY, GRASS]).all(), f"road paint above row 70 at s={s}"


def test_noise_is_seeded_and_bounded_to_uint8(cam):
    road = make_scenario("straight").road
    st = _state_on(road, 20.0)
    a = cam.render(road, st, noise_sigma=6.0, seed=3)
    assert a.dtype == np.uint8
    assert (a == cam.render(road, st, noise_sigma=6.0, seed=3)).all()
    assert (a != cam.render(road, st, noise_sigma=6.0, seed=4)).any()
    assert (a != cam.render(road, st)).any()


def test_closed_road_wraps_at_the_start_line(cam):
    road = make_scenario("gentle").road
    for s in (road.length - 2.0, road.length - 0.1, 0.1):
        _, lab = cam.render(road, _state_on(road, s), return_labels=True)
        assert (lab == YELLOW).any() and (lab == WHITE).any()


@pytest.mark.slow
def test_render_throughput_at_least_100_per_second(cam):
    road = make_scenario("curvy").road
    states = [_state_on(road, s, 0.3, 0.05) for s in np.linspace(0, road.length - 1, 60)]
    cam.render(road, states[0])  # warm up
    rate = 0.0
    for _ in range(3):  # best of three half-second windows: robust to a busy machine
        t0, n = time.perf_counter(), 0
        while time.perf_counter() - t0 < 0.5:
            cam.render(road, states[n % len(states)])
            n += 1
        rate = max(rate, n / (time.perf_counter() - t0))
    print(f"\nRENDER_THROUGHPUT {rate:.0f} renders/s")
    assert rate >= 100


# --- road ----------------------------------------------------------------------------

def _normal(road, s):
    _, _, h = road.pose_at(s)
    return np.array([-math.sin(h), math.cos(h)])


def test_project_inverts_offset_away_from_polyline_vertices():
    road = make_scenario("curvy").road
    rng = np.random.default_rng(0)
    for _ in range(300):
        i = rng.integers(0, len(road._s0))  # a random segment, a point well inside it
        s = road._s0[i] + rng.uniform(0.3, 0.7) * road._seg_len[i]
        d = rng.uniform(-3.6, 3.6)
        x, y, _ = road.pose_at(s)
        p = np.array([x, y]) + _normal(road, s) * d
        s2, d2 = road.project(*p)
        assert abs(s2 - s) < 1e-6 and abs(d2 - d) < 1e-6
        assert road.offset(s, d) == pytest.approx(p, abs=1e-9)


def test_project_error_at_vertices_is_bounded_by_the_polyline_kink():
    # On the inside of a bend, points just past a vertex are nearest the previous segment: the
    # recovered s is off by at most d*(vertex turn angle) and d by d*(1-cos(turn angle)).
    road = make_scenario("curvy").road
    turn = CENTERLINE_STEP / 20.0  # radius 20 m
    rng = np.random.default_rng(1)
    for _ in range(500):
        s, d = rng.uniform(0, road.length), rng.uniform(-3.6, 3.6)
        s2, d2 = road.project(*road.offset(s, d))
        assert abs(s2 - s) <= abs(d) * turn * 1.01 + 1e-9
        assert abs(d2 - d) <= abs(d) * (1 - math.cos(turn)) * 1.01 + 1e-9


def test_lateral_is_left_positive_and_ego_truth_round_trips():
    road = make_scenario("straight").road
    assert road.project(50.0, 1.0)[1] == pytest.approx(1.0)  # +y is left of a road along +x
    for road in (make_scenario("gentle").road, make_scenario("curvy").road):
        for s, off, yaw_err in [(10.3, 0.5, 0.1), (75.2, -0.9, -0.15), (140.1, 0.0, 0.0)]:
            x, y, yaw = road.ego_pose(s, off, yaw_err)
            s2, off2, err2 = road.ego_truth(x, y, yaw)
            assert (s2, off2, err2) == pytest.approx((s, off, yaw_err), abs=1e-6)
    assert EGO_LANE_CENTER == -1.8


def test_pose_at_is_arc_length_parameterised_and_vectorised():
    road = make_scenario("gentle").road
    x, y, h = road.pose_at(np.array([0.0, 50.0, 100.0]))
    assert x.shape == (3,) and h[0] == pytest.approx(0.0)
    assert math.hypot(x[1] - x[0], y[1] - y[0]) == pytest.approx(50.0, abs=1e-6)
    assert isinstance(road.pose_at(3.0)[0], float)


def test_dashes_partition_a_closed_road_evenly():
    road = make_scenario("gentle").road
    spans = road.dashes(0.0, road.length)
    assert all(b > a for a, b in spans)
    assert (road.length / road.dash_period) == pytest.approx(round(road.length / road.dash_period))


# --- scenarios -----------------------------------------------------------------------

def test_scenario_geometry():
    straight = make_scenario("straight")
    assert straight.route_length == pytest.approx(200.0) and straight.road.length == pytest.approx(200.0 + RUN_OUT)
    gentle = make_scenario("gentle")
    assert gentle.road.closed and gentle.road.length == pytest.approx(200 + 2 * math.pi * 60, rel=1e-3)
    assert gentle.route_length == pytest.approx(gentle.road.length) and gentle.run_out == 0.0
    curvy = make_scenario("curvy")
    arc = math.pi / 2 * 20.0
    assert not curvy.road.closed and curvy.run_out == RUN_OUT
    assert curvy.route_length == pytest.approx(30.0 + 6 * arc, rel=1e-3)
    heading = lambda s: float(curvy.road.pose_at(s)[2])
    assert heading(29.0) == pytest.approx(0.0)  # opens with a 30 m straight
    for k in range(6):  # six arcs of radius 20 m, each a 90 degree turn, alternating left/right
        turned = heading(30.0 + (k + 1) * arc - 1e-6) - heading(30.0 + k * arc + 1e-6)
        assert turned == pytest.approx((-1) ** k * math.pi / 2, abs=0.02)
    assert heading(curvy.road.length - 1.0) == pytest.approx(0.0, abs=0.02)  # run-out is straight
    assert straight.target_speed == 8.0


def test_unknown_scenario_lists_the_valid_names():
    with pytest.raises(ValueError, match="straight.*gentle.*curvy"):
        make_scenario("hairpin")


def test_road_accepts_a_plain_centreline():
    road = Road(np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]), closed=False)
    assert road.length == pytest.approx(20.0)
    assert road.pose_at(15.0)[:2] == pytest.approx((10.0, 5.0))


# --- dynamics ------------------------------------------------------------------------

def test_zero_steer_keeps_yaw_and_goes_straight():
    st = BicycleState(0.0, 0.0, 0.3, 8.0, 0.0)
    for _ in range(20):
        st = step(st, 0.0, 0.0)
    assert st.yaw == pytest.approx(0.3)
    assert (st.x, st.y) == pytest.approx((8.0 * math.cos(0.3), 8.0 * math.sin(0.3)), abs=1e-9)


def test_positive_steer_turns_left():
    st = BicycleState(0.0, 0.0, 0.0, 8.0, 0.2)
    assert step(st, 0.2, 0.0).yaw > 0


def test_steer_is_clipped_and_rate_limited():
    st = step(BicycleState(0.0, 0.0, 0.0, 8.0, 0.0), 2.0, 0.0)
    assert st.steer == pytest.approx(0.05)  # 1.0 rad/s * 0.05 s
    for _ in range(100):
        st = step(st, 2.0, 0.0)
    assert st.steer == pytest.approx(0.5)
    for _ in range(100):
        st = step(st, -2.0, 0.0)
    assert st.steer == pytest.approx(-0.5)


def test_accel_is_clipped_and_speed_never_negative():
    st = BicycleState(0.0, 0.0, 0.0, 5.0, 0.0)
    assert step(st, 0.0, 100.0).speed == pytest.approx(5.0 + 3.0 * 0.05)
    assert step(st, 0.0, -100.0).speed == pytest.approx(5.0 - 6.0 * 0.05)
    assert step(BicycleState(0.0, 0.0, 0.0, 0.1, 0.0), 0.0, -6.0).speed == 0.0


def test_constant_steer_traces_a_circle_of_radius_wheelbase_over_sin_steer():
    delta, v = 0.2, 8.0
    st = BicycleState(0.0, 0.0, 0.0, v, delta)
    radius = 2.7 / math.sin(delta)  # front-axle path radius
    centre = np.array([-radius * math.sin(delta), radius * math.cos(delta)])
    for _ in range(200):
        st = step(st, delta, 0.0)
        assert np.hypot(*(np.array([st.x, st.y]) - centre)) == pytest.approx(radius, abs=1e-6)


# --- top-down ------------------------------------------------------------------------

def test_topdown_returns_axes_with_a_line_per_trajectory():
    road = make_scenario("curvy").road
    traj = np.column_stack([np.linspace(0, 30, 20), np.zeros(20)])
    ax = render_topdown(road, [traj, traj + 1.0], labels=("a", "b"))
    assert ax.get_aspect() == 1.0
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b"]
    assert ax.figure is not None
