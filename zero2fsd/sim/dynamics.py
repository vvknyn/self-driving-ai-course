"""Kinematic bicycle model, referenced at the FRONT axle (the camera point)."""
from __future__ import annotations

import math
from dataclasses import dataclass

WHEELBASE = 2.7
DT = 0.05
MAX_STEER = 0.5
MAX_STEER_RATE = 1.0  # rad/s
ACCEL_RANGE = (-6.0, 3.0)


@dataclass(frozen=True)
class BicycleState:
    """Front-axle position (x, y), heading `yaw`, `speed` along the front wheel, current `steer` angle."""

    x: float
    y: float
    yaw: float
    speed: float
    steer: float


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def step(state: BicycleState, steer_cmd: float, accel: float, dt: float = DT) -> BicycleState:
    """Advance one tick.  Positive steer turns left.  Non-finite commands raise rather than drive."""
    for name, v in (("steer_cmd", steer_cmd), ("accel", accel)):
        if not math.isfinite(v):
            raise ValueError(f"{name} must be a finite number, got {v!r}")
    max_delta = MAX_STEER_RATE * dt
    steer = state.steer + _clip(_clip(steer_cmd, -MAX_STEER, MAX_STEER) - state.steer, -max_delta, max_delta)
    speed = max(0.0, state.speed + _clip(accel, *ACCEL_RANGE) * dt)
    dist = 0.5 * (state.speed + speed) * dt
    dyaw = dist * math.sin(steer) / WHEELBASE
    half = 0.5 * dyaw
    chord = dist * (math.sin(half) / half if half else 1.0)  # exact for constant steer: arc length -> chord
    heading = state.yaw + half + steer  # the front wheel points `steer` off the body axis; midpoint yaw
    return BicycleState(
        state.x + chord * math.cos(heading), state.y + chord * math.sin(heading), state.yaw + dyaw, speed, steer
    )
