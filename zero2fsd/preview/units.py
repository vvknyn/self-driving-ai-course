"""One preview per graded exercise (grade/units/u0_1_*.py): the learner's code, run for a look instead of a verdict."""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

from ..car import Car, provided
from ..grade._drive import GATE_SCENARIO, require_car
from ..grade.units.u0_1_2 import as_masks
from . import register

STATS_ROAD = "curvy"  # the lab's own stats drive: `gentle` never leaves the lane, so a count of off-lane steps would be 0
_BARE = object()  # the label of a bare number: unlike None, no dict key can be it
OVERLAY = {"yellow": (255, 0, 255), "white": (0, 255, 255)}  # where the learner's (yellow, white) masks say the paint is


def _as_stats(got, name: str) -> dict[str, float]:
    """A function's answer as {label: float}: a dict keeps its keys, a single number is labelled `name`.

    Rejects anything that is not a real number (bool included) with a plain-language TypeError,
    so a learner who returns "3" sees what was expected instead of a float() traceback."""
    pairs = list(got.items()) if isinstance(got, Mapping) else [(_BARE, got)]
    for label, value in pairs:
        if not isinstance(value, Real) or isinstance(value, bool):
            what = "a number or a dict of numbers" if label is _BARE else f"a number for {label!r}"
            raise TypeError(f"expected {what}, got {type(value).__name__}")
    return {str(name if label is _BARE else label): float(value) for label, value in pairs}


@register("0.1.1.a", "stats", road=STATS_ROAD)
def _lane_error_stats(fn, tel):
    return _as_stats(fn(tel.lat.copy()), "lane_error_stats")


@register("0.1.1.b", "stats", road=STATS_ROAD)
def _steps_off_lane(fn, tel):
    return _as_stats(fn(tel.lat.copy()), "steps_off_lane")


@register("0.1.2.a", "image")
def _paint_masks(fn, frame):
    masks = as_masks(fn(frame.copy()), frame.shape[:2], "the preview frame")
    shown = frame.copy()
    for colour, mask in zip(OVERLAY.values(), masks):
        shown[mask] = colour
    return shown


@register("0.1.3.a", "drive", road=GATE_SCENARIO)
@register("0.1.3.b", "drive", road=GATE_SCENARIO)
def _own_perception(fn):
    return Car(perception=fn)


@register("0.1.4.a", "drive", road=GATE_SCENARIO)
def _own_steering(fn):
    return Car(controller=lambda est, obs: fn(est, provided.K_OFF, provided.K_HEAD))


@register("0.1.4.b", "drive", road=GATE_SCENARIO)
def _own_car(car):
    return require_car(car)
