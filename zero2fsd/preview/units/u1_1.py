"""One preview per graded exercise of module 1.1 (grade/units/u1_1_*.py): the learner's code, run for a look instead of a verdict."""
from __future__ import annotations

import numpy as np

from ...geom import wrap_angle
from .. import register

LAP = "gentle"  # the lab's own lap: the step arrows and yaw the exercises are about
MISS_RAD = 0.1  # a direction this far from the car's yaw counts as a miss in the preview


def _steps(tel):
    """(step arrows, yaw at each step's start), fresh copies the learner's function may scribble on."""
    arrows = np.diff(np.stack([tel.x, tel.y], axis=-1), axis=0)
    return arrows, tel.yaw[:-1].copy()


def _numbers(got, shape: tuple, what: str) -> np.ndarray:
    """The learner's answer as a float array of `shape`; a plain-language TypeError otherwise."""
    try:
        out = np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        raise TypeError(f"expected {what}, got {type(got).__name__}") from None
    if out.shape != shape:
        raise TypeError(f"expected {what}, shape {shape}, got shape {out.shape}")
    return out


@register("1.1.1.a", "stats", road=LAP)
def _direction_angle(fn, tel):
    arrows, yaw = _steps(tel)
    miss = np.abs(wrap_angle(_numbers(fn(arrows), (len(arrows),), "one angle per step arrow") - yaw))
    return {f"steps where your heading misses yaw by > {MISS_RAD} rad": float(np.sum(~(miss <= MISS_RAD))),  # a nan answer is a miss
            "worst miss (rad)": float(miss.max())}


@register("1.1.1.b", "stats", road=LAP)
def _forward_left(fn, tel):
    arrows, yaw = _steps(tel)
    parts = _numbers(fn(yaw, arrows), (len(arrows), 2), "a (forward, left) pair per step arrow")
    return {"mean forward per step (m)": float(parts[:, 0].mean()), "largest left per step (m)": float(parts[:, 1].max())}
