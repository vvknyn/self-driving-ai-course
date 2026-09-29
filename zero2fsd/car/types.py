"""The data that flows between the boxes of a car."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Observation:
    frame: np.ndarray  # uint8[H, W, 3] RGB camera image
    speed: float  # m/s
    t: float  # seconds since the start of the run


@dataclass(frozen=True)
class LaneEstimate:
    """Where the car is in its lane, measured at the front axle.

    offset_m > 0: the car is LEFT of lane centre.  heading_rad > 0: the car points LEFT of the lane.
    """

    offset_m: float
    heading_rad: float
    valid: bool


@dataclass(frozen=True)
class Action:
    steer_rad: float  # > 0 turns LEFT
    accel: float  # m/s^2
