"""A Car is a perception box and a controller box; each slot defaults to the provided one."""
from __future__ import annotations

from numbers import Real

from . import provided
from .types import Action, LaneEstimate, Observation


def _box(box, method: str, default):
    """Accept None (provided), an object with `.method`, or a plain callable."""
    if box is None:
        return default
    fn = getattr(box, method, box)
    if not callable(fn):
        raise TypeError(f"a {method!r} box must be a function or have a .{method}() method, got {type(box).__name__}")
    return fn


class Car:
    def __init__(self, perception=None, controller=None):
        self._slots = {
            "perception": (_box(perception, "estimate", provided.estimate_lane), provided.estimate_lane),
            "controller": (_box(controller, "act", provided.controller), provided.controller),
        }

    def uses_provided(self) -> list[str]:
        """Names of the slots still running the provided box."""
        return [name for name, (fn, default) in self._slots.items() if fn is default]

    def step(self, obs: Observation) -> tuple[LaneEstimate, Action]:
        est = self._slots["perception"][0](obs)
        if not isinstance(est, LaneEstimate):
            raise TypeError(f"the perception box must return a LaneEstimate, got {type(est).__name__}")
        out = self._slots["controller"][0](est, obs)
        if isinstance(out, Action):
            return est, out
        if isinstance(out, Real) and not isinstance(out, bool):  # a bare number is a steering angle
            return est, Action(float(out), provided.hold_speed(obs.speed))
        raise TypeError(f"the controller box must return an Action or a steering angle (float), got {type(out).__name__}")
