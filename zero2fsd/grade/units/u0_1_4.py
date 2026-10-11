"""0.1.4 Closing the loop: a P steering controller and the all-yours car."""
from __future__ import annotations

import math

import numpy as np

from ...car import LaneEstimate
from ...sim.dynamics import MAX_STEER
from .._drive import GATE_MEAN_LAT, GATE_SCENARIO, drive, gate, outcome, require_car, require_own
from .._report import Failure, call, exercise, fmt, is_real

TOL = 1e-9


def _cases():
    """(what is checked, estimate, k_off, k_head, expected steer): named cases first, then seeded random ones."""
    rng = np.random.default_rng(401)
    valid = lambda offset, heading: LaneEstimate(offset, heading, True)
    named = [
        ("left of centre steers right (negative)", valid(0.5, 0.0), 0.4, 1.0, -0.2),
        ("pointing left steers right (negative)", valid(0.0, 0.1), 0.4, 1.0, -0.1),
        ("right of centre steers left (positive)", valid(-0.5, 0.0), 0.4, 1.0, 0.2),
        ("a big error clips at +0.5 rad", valid(-9.0, 0.0), 0.4, 1.0, MAX_STEER),
        ("a big error clips at -0.5 rad", valid(0.0, 4.0), 0.4, 1.0, -MAX_STEER),
        ("no lane seen: steer straight (0.0)", LaneEstimate(math.nan, math.nan, False), 0.4, 1.0, 0.0),
    ]
    random = []
    for _ in range(30):
        off, head, k_off, k_head = rng.uniform(-1.5, 1.5), rng.uniform(-0.3, 0.3), rng.uniform(0.1, 1.0), rng.uniform(0.5, 3.0)
        random.append(("a random case against the formula", valid(off, head), k_off, k_head,
                       float(np.clip(-(k_off * off + k_head * head), -MAX_STEER, MAX_STEER))))
    return named + random


@exercise(
    "0.1.4.a",
    checked=f"p_steer against -(k_off*offset + k_head*heading) clipped to +-{MAX_STEER}: signs, clipping, invalid estimate, 30 random cases",
    hint='lecture 0.1.4, section "The P controller"',
)
def _p_steer(fn, show, on_step):
    cases = _cases()
    for what, est, k_off, k_head, want in cases:
        given = f"est = LaneEstimate(offset_m={fmt(est.offset_m)}, heading_rad={fmt(est.heading_rad)}, valid={est.valid}), k_off={fmt(k_off)}, k_head={fmt(k_head)}"
        got = call(fn, est, k_off, k_head, given=given)
        if not is_real(got):
            raise Failure(f"returned {fmt(got)}, expected a finite steering angle in radians ({what})", given=given, expected=want, got=got)
        if abs(got - want) > TOL:
            raise Failure(f"wrong steering: {what}", given=given, expected=want, got=got)
    return f"all {len(cases)} steering cases match"


@exercise(
    "0.1.4.b",
    checked=f"my_car (your perception and your controller) on `{GATE_SCENARIO}`: it must complete with mean |lat| < {GATE_MEAN_LAT} m; then `curvy` is shown, not graded",
    hint='lecture 0.1.4, section "Closing the loop"',
)
def _my_car(car, show, on_step):
    require_car(car)
    require_own(car)
    passed = gate(car, show, on_step)
    curvy = drive(car, "curvy", "your cliffhanger on curvy", show)
    return f"{passed}; cliffhanger (not graded): {outcome(curvy)}"
