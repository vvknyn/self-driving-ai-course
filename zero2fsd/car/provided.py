"""The provided boxes of the Loop 0 car: colour-threshold perception and a P steering controller.

Learners replace these one at a time.  The controller has no lookahead and no curvature
feed-forward, which is deliberate: it holds a gentle bend and fails a tight one.
"""
from __future__ import annotations

import math

import numpy as np

from ..sim.camera import PALETTE, WHITE, YELLOW, Camera
from ..sim.scenarios import TARGET_SPEED
from ..sim.dynamics import MAX_STEER
from .types import Action, LaneEstimate, Observation

PAINT_DISTANCE = 60.0  # RGB distance to a paint colour; the nearest other colour is > 100 away
NEAR_ROWS, FAR_ROWS = (125, 180), (95, 116)  # image row bands [a, b) the lane is measured in
MIN_PIXELS = 3  # paint pixels a band needs before its line counts as seen
_CAMERA = Camera()
# Tuned on the sim: with no curvature feed-forward the steady-state offset grows as 1/radius, so these
# gains hold `gentle` (mean |lat| ~0.4 m) and cut the corner out of the lane on `curvy`.
K_OFF, K_HEAD = 0.35, 2.7


def paint_masks(frame: np.ndarray):
    """(yellow, white) boolean masks of the pixels within `PAINT_DISTANCE` of each paint colour."""
    rgb = frame.astype(float)
    return tuple(np.linalg.norm(rgb - PALETTE[c], axis=-1) < PAINT_DISTANCE for c in (YELLOW, WHITE))


def _lane_point(masks, rows):
    """Ground point (forward, left) midway between the yellow and white paint seen in a row band."""
    uv = []
    for mask in masks:
        v, u = np.nonzero(mask[slice(*rows)])
        if len(u) < MIN_PIXELS:
            return None  # the dashed line can fall out of a band, or the road bends out of view
        uv.append((u.mean(), v.mean() + rows[0]))  # the centroid of a thin line lies on the line
    forward, left = _CAMERA.pixel_to_ground(*np.array(uv).T)
    return forward.mean(), left.mean()


def estimate_lane(obs: Observation) -> LaneEstimate:
    """Offset and heading from two lane-centre points (a near and a far row band) and a straight line."""
    masks = paint_masks(obs.frame)
    near, far = (_lane_point(masks, rows) for rows in (NEAR_ROWS, FAR_ROWS))
    if near is None or far is None:
        return LaneEstimate(math.nan, math.nan, False)
    (f1, c1), (f2, c2) = near, far
    slope = (c2 - c1) / (f2 - f1)  # how fast the lane centre moves left per metre ahead
    return LaneEstimate(offset_m=-(c1 - slope * f1), heading_rad=-math.atan(slope), valid=True)


def hold_speed(speed: float, target: float = TARGET_SPEED) -> float:
    """Acceleration command that nudges `speed` towards `target`."""
    return float(np.clip(0.5 * (target - speed), -6.0, 3.0))


def steer_p(est: LaneEstimate, k_off: float = K_OFF, k_head: float = K_HEAD) -> float:
    """Steering that pushes the car back to lane centre and straight; 0.0 when the estimate is invalid."""
    if not est.valid:
        return 0.0
    return float(np.clip(-(k_off * est.offset_m + k_head * est.heading_rad), -MAX_STEER, MAX_STEER))


def controller(est: LaneEstimate, obs: Observation) -> float:
    """The provided controller box: P steering; the Car pairs it with `hold_speed`."""
    return steer_p(est)
