"""Drive a Car around a scenario and record telemetry."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..sim.camera import Camera
from ..sim.dynamics import DT, BicycleState, step
from ..sim.scenarios import TARGET_SPEED, Scenario, make_scenario
from .car import Car
from .types import Observation

OFF_ROAD_OFFSET = 1.8  # m of true offset from ego-lane centre that ends the episode
MAX_STEP_FACTOR = 3  # default step budget: this many times the steps needed at target speed


@dataclass(frozen=True)
class Telemetry:
    """Per-step arrays.  `steer` is the applied wheel angle, `accel` the commanded acceleration,
    `lat`/`head_err` the truth (offset from ego-lane centre, heading error) and `est_*` what the car believed."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    speed: np.ndarray
    steer: np.ndarray
    accel: np.ndarray
    lat: np.ndarray
    head_err: np.ndarray
    est_offset: np.ndarray
    est_heading: np.ndarray
    est_valid: np.ndarray
    progress_m: np.ndarray
    off_road: np.ndarray
    scenario: str
    completed: bool
    route_length: float


def run(car: Car, scenario, max_steps: int | None = None, seed: int = 0, noise_sigma: float = 0.0) -> Telemetry:
    """Run until the route is complete, the car is off the road, or `max_steps` (default 3x the ideal).

    Exceptions raised inside the car's boxes propagate.  `scenario` is a Scenario or its name.
    """
    sc = make_scenario(scenario) if isinstance(scenario, str) else scenario
    road, camera = sc.road, Camera()
    if max_steps is None:
        max_steps = math.ceil(MAX_STEP_FACTOR * sc.route_length / (TARGET_SPEED * DT))
    state = BicycleState(*road.ego_pose(sc.start_s), sc.target_speed, 0.0)
    rows, progress, s_prev = [], 0.0, sc.start_s
    for k in range(max_steps):
        s, lat, head_err = road.ego_truth(state.x, state.y, state.yaw)
        ds = s - s_prev
        progress += (ds + road.length / 2) % road.length - road.length / 2 if road.closed else ds
        s_prev = s
        obs = Observation(camera.render(road, state, noise_sigma, seed + k), state.speed, k * DT)
        est, action = car.step(obs)
        off_road = abs(lat) > OFF_ROAD_OFFSET
        rows.append((k * DT, state.x, state.y, state.yaw, state.speed, state.steer, action.accel, lat, head_err,
                     est.offset_m, est.heading_rad, est.valid, progress, off_road))
        if off_road or progress >= sc.route_length:
            break
        state = step(state, action.steer_rad, action.accel)
    cols = [np.array(c) for c in zip(*rows)]
    return Telemetry(*cols, scenario=sc.name, completed=bool(progress >= sc.route_length and not off_road),
                     route_length=sc.route_length)
