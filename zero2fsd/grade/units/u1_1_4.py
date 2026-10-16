"""1.1.4 Poses you can chain: relative_pose and dead_reckon, checked against the simulator's own record."""
from __future__ import annotations

import functools
import math
import operator

import numpy as np

from ...car import Car, run
from .._report import Failure, Threshold, call, exercise, vs_limit

LESSON = ("1.1.4", "Poses you can chain")  # (unit id, title): grade/__init__ files it in LESSON_TITLES
ROAD = "gentle"
N_PAIRS, POSE_SEED, REL_TOL = 40, 114, 1e-9  # seeded pairs for relative_pose, and the entry-wise tolerance on its 3x3
POS_TOL, ANGLE_TOL = 1e-6, 1e-9  # dead_reckon: every pose within this many metres and radians of the record


def matrix(x, y, theta) -> np.ndarray:
    """The 3x3 of a pose, written out here so the grader never leans on the geom code it is grading."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def step_between(a, b) -> tuple[float, float, float]:
    """(forward, left, turn): pose b as seen from pose a, each pose (x, y, theta), the turn wrapped to [-pi, pi]."""
    dx, dy, dtheta = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    c, s = math.cos(a[2]), math.sin(a[2])
    return c * dx + s * dy, -s * dx + c * dy, math.atan2(math.sin(dtheta), math.cos(dtheta))


@functools.cache
def lap():
    """The gentle drive of the stock car: its poses (n, 3) and the odometry steps between neighbours (n - 1, 3, 3)."""
    tel = run(Car(), ROAD)
    poses = np.column_stack([tel.x, tel.y, tel.yaw])
    return poses, np.stack([matrix(*step_between(a, b)) for a, b in zip(poses[:-1], poses[1:])])


def rebuild(dead_reckon):
    """(poses, per-pose position error m, per-pose angle error rad) of `dead_reckon` run on the lap's first pose and steps."""
    poses, steps = lap()
    got = np.asarray(call(dead_reckon, matrix(*poses[0]), steps, given=f"the first pose and {len(steps)} odometry steps of {ROAD}"), dtype=float)
    if got.shape != (len(poses), 3, 3):
        raise Failure(f"returned an array of shape {got.shape}, expected {(len(poses), 3, 3)}: the start, then one pose per step",
                      given=f"{len(steps)} steps")
    turn = np.arctan2(got[:, 1, 0], got[:, 0, 0]) - poses[:, 2]
    return got, np.hypot(got[:, 0, 2] - poses[:, 0], got[:, 1, 2] - poses[:, 1]), np.abs(np.arctan2(np.sin(turn), np.cos(turn)))


def named_pairs():
    """(name, T_a, T_b, wanted T_a<-b) for the cases the lecture works by hand."""
    A, B = matrix(1, 0, math.pi / 2), matrix(1, 0, 0)
    yield "A then B", A, A @ B, B
    yield "B then A", B, B @ A, A
    T = matrix(3, 4, 0.7)
    yield "the same pose twice", T, T, np.eye(3)
    yield "a pure turn on the spot", matrix(3, 4, 0), matrix(3, 4, 0.5), matrix(0, 0, 0.5)


def seeded_pairs():
    rng = np.random.default_rng(POSE_SEED)
    for _ in range(N_PAIRS):
        a = (*rng.uniform(-50, 50, 2), rng.uniform(-math.pi, math.pi))
        b = (*rng.uniform(-50, 50, 2), rng.uniform(-math.pi, math.pi))
        yield "a seeded pair", matrix(*a), matrix(*b), matrix(*step_between(a, b))


def relative_pose_error(fn) -> float:
    """The worst entry-wise error of `fn` over the named and the seeded pairs; a Failure for a wrong shape."""
    worst = 0.0
    for name, a, b, want in (*named_pairs(), *seeded_pairs()):
        got = np.asarray(call(fn, a, b, given=f"{name}: two 3x3 poses"), dtype=float)
        if got.shape != (3, 3):
            raise Failure(f"returned an array of shape {got.shape}, expected (3, 3)", given=name)
        worst = max(worst, float(np.abs(got - want).max()))
        if worst >= REL_TOL:
            raise Failure(f"{name}: off by {worst:.3g}, the limit is {REL_TOL:g}",
                          given=f"{name}: T_a =\n{a}\nT_b =\n{b}", expected=want, got=got)
    return worst


THRESHOLDS = {
    "relative_pose, worst entry": Threshold(REL_TOL, operator.lt, LESSON[0], lambda solution: relative_pose_error(solution.relative_pose)),
    "dead_reckon, worst position (m)": Threshold(POS_TOL, operator.lt, LESSON[0], lambda solution: rebuild(solution.dead_reckon)[1].max()),
    "dead_reckon, worst angle (rad)": Threshold(ANGLE_TOL, operator.lt, LESSON[0], lambda solution: rebuild(solution.dead_reckon)[2].max()),
}


@exercise(
    "1.1.4.a",
    checked=(f"relative_pose on four worked cases (A then B, B then A, the same pose twice, a pure turn) and {N_PAIRS} seeded pairs: "
             f"every entry of the 3x3 within {REL_TOL:g} of the closed form"),
    section="What the car felt between two steps",
)
def _relative_pose(fn, show, on_step):
    worst = relative_pose_error(fn)
    return f"all {N_PAIRS + 4} cases within {REL_TOL:g}: worst entry {worst:.2g} {vs_limit(worst, REL_TOL)}"


@exercise(
    "1.1.4.b",
    checked=(f"dead_reckon rebuilds the {ROAD} lap from its odometry steps: every pose within {POS_TOL:g} m and {ANGLE_TOL:g} rad "
             "of the simulator's record, not just the end"),
    section="Dead reckoning",
)
def _dead_reckon(fn, show, on_step):
    poses, steps = lap()
    got, position, angle = rebuild(fn)
    if position.max() >= POS_TOL or angle.max() >= ANGLE_TOL:
        k = int(np.argmax(position / POS_TOL + angle / ANGLE_TOL))
        raise Failure(
            f"pose {k} of {len(poses)} is {position[k]:.4g} m and {angle[k]:.3g} rad off (the end is {position[-1]:.4g} m off): "
            f"every pose is checked, the limit is {POS_TOL:g} m",
            given=f"the first pose and {len(steps)} steps of {ROAD}", expected=f"x, y = ({poses[k, 0]:.3f}, {poses[k, 1]:.3f})",
            got=f"x, y = ({got[k, 0, 2]:.3f}, {got[k, 1, 2]:.3f})",
        )
    return (f"{len(poses)} poses rebuilt from {len(steps)} steps: worst {position.max():.2g} m {vs_limit(position.max(), POS_TOL, 'm')}, "
            f"end {position[-1]:.2g} m, worst angle {angle.max():.2g} rad")
