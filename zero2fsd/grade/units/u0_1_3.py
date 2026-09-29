"""0.1.3 Your first perception box: a lane estimate from paint."""
from __future__ import annotations

import numpy as np

from ...car import Car, LaneEstimate, Observation
from ...sim import Camera
from ...sim.scenarios import TARGET_SPEED
from .._drive import gate, require_own
from .._poses import sample_poses
from .._report import Failure, call, exercise, learner_code

N_POSES, MIN_HIT_RATE, OFFSET_TOL, HEADING_TOL = 40, 0.9, 0.25, 0.05


@exercise(
    "0.1.3.a",
    checked=(f"offset and heading on {N_POSES} seeded straight-road poses (up to 1 m off centre, up to 0.15 rad yawed): "
             f"at least {MIN_HIT_RATE:.0%} within {OFFSET_TOL} m and {HEADING_TOL} rad and valid"),
    hint='lecture 0.1.3, section "From masks to a lane estimate"',
)
def _estimate_lane(fn, show):
    camera, rng = Camera(), np.random.default_rng(301)
    hits, first_miss = 0, None
    for pose in sample_poses(rng, N_POSES, ("straight", "gentle"), straight_only=True):
        given = f"an Observation from a frame rendered at {pose}"
        est = call(fn, Observation(camera.render(pose.road, pose.state), TARGET_SPEED, 0.0), given=given)
        if not isinstance(est, LaneEstimate):
            raise Failure(f"returned {type(est).__name__}, expected a LaneEstimate", given=given, got=est)
        if est.valid and abs(est.offset_m - pose.offset) < OFFSET_TOL and abs(est.heading_rad - pose.heading) < HEADING_TOL:
            hits += 1
        elif first_miss is None:
            first_miss = (pose, est, given)
    if hits < MIN_HIT_RATE * N_POSES:
        pose, est, given = first_miss
        raise Failure(
            f"only {hits} of {N_POSES} poses were within tolerance (need {MIN_HIT_RATE:.0%}); first miss shown", given=given,
            expected=f"offset {pose.offset:+.2f} m (within {OFFSET_TOL}), heading {pose.heading:+.3f} rad (within {HEADING_TOL}), valid=True",
            got=f"offset {est.offset_m:+.2f} m, heading {est.heading_rad:+.3f} rad, valid={est.valid}",
        )
    return f"{hits} of {N_POSES} poses within {OFFSET_TOL} m and {HEADING_TOL} rad"


@exercise(
    "0.1.3.b",
    checked="your perception inside the provided car on `gentle`: it must complete with mean |lat| < 0.5 m",
    hint='lecture 0.1.3, section "Swap your perception into the car"',
)
def _perception_gate(fn, show):
    with learner_code("Car(perception=your function)"):
        car = Car(perception=fn)
    require_own(car, "perception")  # Car(perception=None) would quietly drive on the provided box
    return gate(car, show)
