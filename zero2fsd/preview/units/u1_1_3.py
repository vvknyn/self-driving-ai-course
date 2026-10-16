"""Previews of 1.1.3's graded exercises (grade/units/u1_1_3.py): the learner's road-frame functions, measured over a lap."""
from __future__ import annotations

import numpy as np

from ...grade.units.u1_1_3 import psi_misfit
from ...sim import make_scenario
from .. import register
from .u0_1 import _as_stats

LAP_ROAD = "gentle"  # the lap the stats are measured on: a closed road, so the learner's function is called with closed=True


def _lap(tel):
    """(road, the (x, y) of every step of the drive)."""
    road = make_scenario(LAP_ROAD).road
    return road, np.column_stack([tel.x, tel.y])


@register("1.1.3.a", "stats", road=LAP_ROAD)
def _road_frame_errors(fn, tel):
    road, points = _lap(tel)
    got = np.array([fn(road.centerline, road.closed, (x, y)) for x, y in points], dtype=float)  # (steps, 2): s, d
    truth = np.array([road.project(x, y) for x, y in points])
    return _as_stats({"largest |d − truth| over the lap (m)": np.abs(got[:, 1] - truth[:, 1]).max(),
                      "largest |s − truth| (m)": np.abs(got[:, 0] - truth[:, 0]).max()}, "to_road_frame")


@register("1.1.3.b", "stats", road=LAP_ROAD)
def _lane_error_errors(fn, tel):
    road, points = _lap(tel)
    poses = [(x, y, yaw) for (x, y), yaw in zip(points, tel.yaw)]
    got = np.array([fn(road.centerline, road.closed, pose) for pose in poses], dtype=float)  # (steps, 2): e, psi
    truth_e = np.array([road.ego_truth(*pose)[1] for pose in poses])
    return _as_stats({"largest |e − truth| (m)": np.abs(got[:, 0] - truth_e).max(),
                      "largest |ψ − truth| (rad)": max(psi_misfit(road, pose, psi) for pose, psi in zip(poses, got[:, 1]))}, "lane_errors")
