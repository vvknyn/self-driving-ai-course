"""One preview per graded exercise of 1.1.4 (grade/units/u1_1_4.py): the learner's code on the canned drive, as numbers."""
from __future__ import annotations

import numpy as np

from ...grade.units.u1_1_4 import POS_TOL, matrix, step_between
from .. import Error, register

ROAD = "gentle"


def _poses(tel) -> np.ndarray:
    return np.column_stack([tel.x, tel.y, tel.yaw])


def _checked(got, shape, what: str) -> np.ndarray:
    got = np.asarray(got, dtype=float)
    if got.shape != shape:
        raise ValueError(f"{what} returned an array of shape {got.shape}, expected {shape}")
    return got


@register("1.1.4.a", "stats", road=ROAD)
def _odometry_steps(fn, tel):
    poses = _poses(tel)
    steps = np.stack([_checked(fn(matrix(*a), matrix(*b)), (3, 3), "relative_pose") for a, b in zip(poses[:-1], poses[1:])])
    return {"mean step forward (m)": float(steps[:, 0, 2].mean()), "mean step turn (rad)": float(np.arctan2(steps[:, 1, 0], steps[:, 0, 0]).mean())}


@register("1.1.4.b", "stats", road=ROAD)
def _rebuilt_lap(fn, tel):
    poses = _poses(tel)
    steps = np.stack([matrix(*step_between(a, b)) for a, b in zip(poses[:-1], poses[1:])])
    got = _checked(fn(matrix(*poses[0]), steps), (len(poses), 3, 3), "dead_reckon")
    error = np.hypot(got[:, 0, 2] - poses[:, 0], got[:, 1, 2] - poses[:, 1])
    return {"worst error over the lap (m)": Error(error.max(), POS_TOL), "error at the end (m)": Error(error[-1], POS_TOL)}
