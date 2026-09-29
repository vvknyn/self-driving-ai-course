"""Seeded car poses on the course roads, for checks that render frames."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sim import BicycleState, Road, make_scenario
from ..sim.scenarios import TARGET_SPEED

LOOKAHEAD = 45.0  # m of road ahead that must be straight for a straight-road pose (covers what the camera measures)
_GRID, _SAMPLES = 1.0, np.arange(0.0, LOOKAHEAD + 1e-9, 5.0)


@dataclass(frozen=True)
class Pose:
    scenario: str
    road: Road
    s: float
    offset: float  # true offset from lane centre, left positive
    heading: float  # true heading error, left positive
    state: BicycleState

    def __str__(self) -> str:
        return f"{self.scenario} at s={self.s:.0f} m, {self.offset:+.2f} m off lane centre, heading {self.heading:+.3f} rad"


def _starts(road: Road, straight_only: bool) -> np.ndarray:
    """Arc lengths a pose may start at; with `straight_only`, only where the next LOOKAHEAD metres do not turn."""
    s = np.arange(0.0, road.length if road.closed else road.length - LOOKAHEAD, _GRID)
    if not straight_only:
        return s
    heading = road.pose_at(s[:, None] + _SAMPLES)[2]
    return s[np.ptp(heading, axis=1) < 1e-9]


def sample_poses(rng, n, scenarios, *, straight_only, max_offset=1.0, max_heading=0.15) -> list[Pose]:
    """`n` poses, pose k on scenarios[k % len(scenarios)], with offset and heading drawn uniformly."""
    poses = []
    for k in range(n):
        name = scenarios[k % len(scenarios)]
        road = make_scenario(name).road
        s = float(rng.choice(_starts(road, straight_only)))
        offset, heading = rng.uniform(-max_offset, max_offset), rng.uniform(-max_heading, max_heading)
        poses.append(Pose(name, road, s, offset, heading, BicycleState(*road.ego_pose(s, offset, heading), TARGET_SPEED, 0.0)))
    return poses
