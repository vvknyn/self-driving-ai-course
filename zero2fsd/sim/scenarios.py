"""The three course scenarios, built from (length, curvature) pieces."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .world import Road

CENTERLINE_STEP = 0.1  # metres between centreline vertices
TARGET_SPEED = 8.0  # m/s: the speed the provided car holds and every scenario starts at
GENTLE_RADIUS = 80.0  # spec: >= 60 m; wide enough that a P controller with no feed-forward holds it comfortably
RUN_OUT = 40.0  # open roads continue straight past the route end so paint stays visible until completion


@dataclass(frozen=True)
class Scenario:
    name: str
    road: Road
    start_s: float = 0.0
    target_speed: float = TARGET_SPEED
    run_out: float = 0.0  # road beyond the end of the route

    @property
    def route_length(self) -> float:
        """Distance the car must cover: one lap if closed, else start to the end of the route."""
        return self.road.length if self.road.closed else self.road.length - self.start_s - self.run_out


def _trace(pieces, closed: bool) -> Road:
    """Integrate (length, curvature) pieces from the origin heading +x into a Road."""
    x = y = h = 0.0
    pts = []
    for length, kappa in pieces:
        n = math.ceil(length / CENTERLINE_STEP)
        pts.extend(_advance(x, y, h, k * length / n, kappa)[:2] for k in range(n))
        x, y, h = _advance(x, y, h, length, kappa)
    if not closed:
        pts.append((x, y))
    return Road(np.array(pts), closed)


def _advance(x, y, h, dist, kappa):
    """Pose after driving `dist` along an arc of curvature `kappa` (0 = straight)."""
    if kappa == 0.0:
        return x + dist * math.cos(h), y + dist * math.sin(h), h
    return (
        x + (math.sin(h + kappa * dist) - math.sin(h)) / kappa,
        y - (math.cos(h + kappa * dist) - math.cos(h)) / kappa,
        h + kappa * dist,
    )


def _open(name: str, pieces) -> Scenario:
    return Scenario(name, _trace([*pieces, (RUN_OUT, 0.0)], closed=False), run_out=RUN_OUT)


def _straight() -> Scenario:
    return _open("straight", [(200.0, 0.0)])


def _gentle() -> Scenario:
    r = GENTLE_RADIUS
    return Scenario("gentle", _trace([(100.0, 0.0), (math.pi * r, 1 / r), (100.0, 0.0), (math.pi * r, 1 / r)], closed=True))


def _curvy() -> Scenario:
    r, arc = 20.0, math.pi / 2 * 20.0
    return _open("curvy", [(30.0, 0.0)] + [(arc, (-1) ** k / r) for k in range(6)])


SCENARIOS = {"straight": _straight, "gentle": _gentle, "curvy": _curvy}


def make_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]()
    except KeyError:
        raise ValueError(f"unknown scenario {name!r}; valid names: {', '.join(SCENARIOS)}") from None
