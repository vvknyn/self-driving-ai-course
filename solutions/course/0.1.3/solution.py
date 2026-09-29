"""Reference answer for 0.1.3 Your first perception box.  Labs never import this; the grader never reads it."""
import math
from pathlib import Path

import numpy as np
from _course import load_solution

from zero2fsd.car import LaneEstimate
from zero2fsd.sim import Camera

__all__ = ["estimate_lane"]

_paint_masks = load_solution(Path(__file__).resolve().parents[3], "0.1.2").paint_masks
_camera = Camera()
NEAR_ROWS, FAR_ROWS = (125, 180), (95, 116)  # image rows [a, b) where the lane is measured, close and far
MIN_PIXELS = 3  # a line seen in fewer pixels than this does not count


def _lane_centre(masks, rows):
    """(forward, left) ground point halfway between the yellow and white line seen in this band of rows."""
    points = []
    for mask in masks:
        v, u = np.nonzero(mask[rows[0]:rows[1]])
        if len(u) < MIN_PIXELS:
            return None
        points.append(_camera.pixel_to_ground(u.mean(), v.mean() + rows[0]))
    return (points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2


def estimate_lane(obs):
    masks = _paint_masks(obs.frame)
    near, far = _lane_centre(masks, NEAR_ROWS), _lane_centre(masks, FAR_ROWS)
    if near is None or far is None:
        return LaneEstimate(math.nan, math.nan, False)
    slope, intercept = np.polyfit([near[0], far[0]], [near[1], far[1]], 1)  # the lane centre as left = slope*forward + intercept
    return LaneEstimate(offset_m=-float(intercept), heading_rad=-math.atan(float(slope)), valid=True)
