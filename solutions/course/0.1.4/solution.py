"""Reference answers for 0.1.4 Closing the loop.  Labs never import this; the grader never reads it."""
from pathlib import Path

import numpy as np
from _course import load_solution

from zero2fsd.car import Car

__all__ = ["p_steer", "my_car"]

_estimate_lane = load_solution(Path(__file__).resolve().parents[3], "0.1.3").estimate_lane
MAX_STEER = 0.5  # rad, the wheel's limit
K_OFF, K_HEAD = 0.35, 2.7  # tuned on `gentle`: enough to hold a gentle bend, not enough to hold a tight one


def p_steer(est, k_off, k_head):
    if not est.valid:
        return 0.0
    return float(np.clip(-(k_off * est.offset_m + k_head * est.heading_rad), -MAX_STEER, MAX_STEER))


def _controller(est, obs):
    return p_steer(est, K_OFF, K_HEAD)


my_car = Car(perception=_estimate_lane, controller=_controller)
