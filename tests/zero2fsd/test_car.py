import math

import numpy as np
import pytest

from zero2fsd.car import Action, Car, LaneEstimate, Observation, provided, run
from zero2fsd.sim import BicycleState, Camera, make_scenario
from zero2fsd.sim.camera import GRASS, ROAD, SKY, PALETTE, WHITE, YELLOW

STRAIGHT = make_scenario("straight")


def _obs(scenario, s, offset=0.0, yaw_err=0.0, noise_sigma=0.0, seed=0):
    road = scenario.road
    state = BicycleState(*road.ego_pose(s, offset, yaw_err), 8.0, 0.0)
    return Observation(Camera().render(road, state, noise_sigma, seed), 8.0, 0.0)


def _valid(est):
    return LaneEstimate(est.offset_m, est.heading_rad, True)


# --- the shipped property: the provided car holds a gentle bend and fails a tight one ------------

def test_provided_car_completes_gentle():
    tel = run(Car(), "gentle")
    assert tel.completed and not tel.off_road.any()
    assert np.abs(tel.lat).mean() < 0.5
    assert tel.route_length <= tel.progress_m[-1] < tel.route_length + 1.0  # one lap, counted across the wrap


def test_provided_car_leaves_the_lane_on_curvy():
    tel = run(Car(), "curvy")
    assert not tel.completed and tel.off_road[-1] and abs(tel.lat[-1]) > 1.8
    assert not tel.off_road[:-1].any()  # the run stops the moment it leaves


# --- provided perception --------------------------------------------------------------------------

@pytest.mark.parametrize("offset", [-0.8, 0.0, 0.8])
@pytest.mark.parametrize("yaw_err", [-0.1, 0.0, 0.1])
def test_provided_estimate_matches_truth_on_straight(offset, yaw_err):
    est = provided.estimate_lane(_obs(STRAIGHT, 30.0, offset, yaw_err))
    assert est.valid
    assert est.offset_m == pytest.approx(offset, abs=0.15)
    assert est.heading_rad == pytest.approx(yaw_err, abs=0.03)


@pytest.mark.parametrize("colour", [GRASS, SKY, ROAD])
def test_provided_perception_is_invalid_not_a_crash_on_a_paintless_frame(colour):
    frame = np.tile(PALETTE[colour], (180, 320, 1))
    est = provided.estimate_lane(Observation(frame, 8.0, 0.0))
    assert est.valid is False and math.isnan(est.offset_m) and math.isnan(est.heading_rad)


def test_provided_perception_is_invalid_when_the_car_faces_away_from_the_road():
    est = provided.estimate_lane(_obs(STRAIGHT, 30.0, 0.0, yaw_err=1.2))
    assert isinstance(est, LaneEstimate) and est.valid is False


def test_paint_masks_survive_noise():
    scenario = make_scenario("curvy")
    for s, off in [(10.0, 0.0), (45.0, 0.4), (100.0, -0.4)]:
        state = BicycleState(*scenario.road.ego_pose(s, off), 8.0, 0.0)
        frame, labels = Camera().render(scenario.road, state, noise_sigma=8.0, seed=5, return_labels=True)
        for mask, truth in zip(provided.paint_masks(frame), (labels == YELLOW, labels == WHITE)):
            assert mask.shape == (180, 320) and mask.dtype == bool
            assert (mask & truth).sum() / (mask | truth).sum() >= 0.85


# --- provided controller --------------------------------------------------------------------------

def test_steer_p_signs_clip_and_invalid():
    left_of_centre = LaneEstimate(0.5, 0.0, True)
    assert provided.steer_p(left_of_centre, 0.35, 1.2) == pytest.approx(-0.175)  # left of centre -> steer right
    assert provided.steer_p(LaneEstimate(0.0, 0.1, True), 0.35, 1.2) == pytest.approx(-0.12)  # pointing left -> right
    assert provided.steer_p(LaneEstimate(-9.0, 0.0, True)) == 0.5 and provided.steer_p(LaneEstimate(9.0, 0.0, True)) == -0.5
    assert provided.steer_p(LaneEstimate(math.nan, math.nan, False)) == 0.0


def test_hold_speed_pushes_towards_target_and_is_clipped():
    assert provided.hold_speed(8.0) == 0.0
    assert provided.hold_speed(6.0) == pytest.approx(1.0) and provided.hold_speed(10.0) == pytest.approx(-1.0)
    assert provided.hold_speed(-100.0) == 3.0 and provided.hold_speed(100.0) == -6.0


# --- Car: slots, adapters, errors -----------------------------------------------------------------

def test_uses_provided_names_the_slots_still_on_provided_boxes():
    assert Car().uses_provided() == ["perception", "controller"]
    mine = lambda obs: LaneEstimate(0.0, 0.0, True)
    assert Car(perception=mine).uses_provided() == ["controller"]
    assert Car(controller=lambda est, obs: 0.0).uses_provided() == ["perception"]
    assert Car(perception=mine, controller=lambda est, obs: 0.0).uses_provided() == []
    assert Car(perception=provided.estimate_lane).uses_provided() == ["perception", "controller"]  # not a learner box


def test_float_controller_is_adapted_with_the_provided_speed_controller():
    car = Car(perception=lambda obs: LaneEstimate(0.0, 0.0, True), controller=lambda est, obs: 0.1)
    obs = Observation(np.zeros((180, 320, 3), np.uint8), 6.0, 0.0)
    est, action = car.step(obs)
    assert est == LaneEstimate(0.0, 0.0, True)
    assert action == Action(0.1, provided.hold_speed(6.0))


def test_objects_with_estimate_and_act_methods_are_accepted():
    class Perception:
        def estimate(self, obs):
            return LaneEstimate(1.0, 0.0, True)

    class Controller:
        def act(self, est, obs):
            return Action(-est.offset_m / 10, 0.0)

    car = Car(Perception(), Controller())
    assert car.step(Observation(np.zeros((1, 1, 3), np.uint8), 8.0, 0.0)) == (LaneEstimate(1.0, 0.0, True), Action(-0.1, 0.0))
    assert car.uses_provided() == []


@pytest.mark.parametrize("bad, message", [(None, "NoneType"), ("left", "str"), (True, "bool")])
def test_a_controller_returning_the_wrong_type_raises_a_readable_type_error(bad, message):
    car = Car(perception=lambda obs: LaneEstimate(0.0, 0.0, True), controller=lambda est, obs: bad)
    with pytest.raises(TypeError, match=message):
        car.step(Observation(np.zeros((1, 1, 3), np.uint8), 8.0, 0.0))


def test_a_perception_returning_the_wrong_type_raises_a_readable_type_error():
    with pytest.raises(TypeError, match="LaneEstimate.*tuple"):
        Car(perception=lambda obs: (0.0, 0.0)).step(Observation(np.zeros((1, 1, 3), np.uint8), 8.0, 0.0))


def test_a_non_callable_box_is_rejected_when_the_car_is_built():
    with pytest.raises(TypeError, match="perception|estimate"):
        Car(perception=42)


# --- runner ---------------------------------------------------------------------------------------

def test_a_box_raising_propagates_out_of_run():
    def boom(obs):
        raise ValueError("learner bug")

    with pytest.raises(ValueError, match="learner bug"):
        run(Car(perception=boom), "straight")


def test_telemetry_arrays_align_and_max_steps_bounds_the_run():
    tel = run(Car(), "straight", max_steps=7)
    assert len(tel.t) == 7 and not tel.completed and tel.scenario == "straight" and tel.route_length == pytest.approx(200.0)
    for name in ("x", "y", "yaw", "speed", "steer", "accel", "lat", "head_err", "est_offset", "est_heading",
                 "est_valid", "progress_m", "off_road"):
        assert getattr(tel, name).shape == (7,), name
    assert tel.t[1] - tel.t[0] == pytest.approx(0.05)
    assert tel.est_valid.dtype == bool and tel.off_road.dtype == bool


@pytest.mark.parametrize("steer, sign", [(0.03, 1), (-0.03, -1)])
def test_signs_agree_end_to_end_when_the_car_drifts_left_or_right(steer, sign):
    # Drive with a constant steer: truth and the provided estimate must both say the same side.
    tel = run(Car(controller=lambda est, obs: steer), "straight", max_steps=60)
    assert sign * tel.lat[-1] > 0.2 and sign * tel.head_err[-1] > 0.05
    assert tel.est_offset[-1] == pytest.approx(tel.lat[-1], abs=0.15)
    assert tel.est_heading[-1] == pytest.approx(tel.head_err[-1], abs=0.03)


def test_the_run_starts_on_the_lane_centre_at_target_speed_and_completes_the_straight():
    tel = run(Car(), "straight")
    assert tel.completed and tel.lat[0] == pytest.approx(0.0, abs=1e-9) and tel.speed[0] == 8.0
    assert np.abs(tel.lat).max() < 0.1
