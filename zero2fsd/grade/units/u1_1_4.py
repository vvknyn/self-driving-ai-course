"""1.1.4 Poses you can chain: relative_pose and dead_reckon, checked against the simulator's own record."""
from __future__ import annotations

import functools
import math
import operator

import numpy as np

from ...car import Car, run
from .._report import Failure, Threshold, call, exercise, fmt_gap, matching_mistakes, vs_limit

LESSON = ("1.1.4", "Poses you can chain")  # (unit id, title): grade/__init__ files it in LESSON_TITLES
ROAD = "gentle"
N_PAIRS, POSE_SEED, REL_TOL = 40, 114, 1e-9  # seeded pairs for relative_pose, and the entry-wise tolerance on its 3x3
POS_TOL, ANGLE_TOL = 1e-6, 1e-9  # dead_reckon: every pose within this many metres, and every rotation entry within this much, of the record
RIGID_TOL = 1e-6  # how far a 3x3 may stray from last row (0, 0, 1) and an orthonormal top-left block before it is not a pose matrix


def matrix(x, y, theta) -> np.ndarray:
    """The 3x3 of a pose, written out here so the grader never leans on the geom code it is grading."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def step_between(a, b) -> tuple[float, float, float]:
    """(forward, left, turn): pose b as seen from pose a, each pose (x, y, theta), the turn wrapped to [-pi, pi]."""
    dx, dy, dtheta = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    c, s = math.cos(a[2]), math.sin(a[2])
    return c * dx + s * dy, -s * dx + c * dy, math.atan2(math.sin(dtheta), math.cos(dtheta))


def not_rigid(m) -> str | None:
    """Plain words for what stops a (..., k+1, k+1) array from being a pose matrix, or None when it is one."""
    m = np.asarray(m, dtype=float)
    if not np.isfinite(m).all():
        return "it contains nan or infinity"
    k = m.shape[-1] - 1
    if np.abs(m[..., k, :] - np.eye(k + 1)[k]).max() > RIGID_TOL:
        return "its last row is not (0, ..., 0, 1)"
    r = m[..., :k, :k]
    if np.abs(r.swapaxes(-1, -2) @ r - np.eye(k)).max() > RIGID_TOL:
        return f"its top-left {k}x{k} block is not a rotation (its columns must be 1 long and at right angles)"
    if (np.linalg.det(r) < 0).any():
        return "its top-left block is a mirror, not a rotation (its determinant is -1)"
    return None


def named_mistake(got, want, candidates: dict[str, np.ndarray]) -> str:
    """The sentence naming every common mistake whose output `got` equals (a case can fit two) and `want` does not, or "" when it fits none."""
    names = matching_mistakes(got, want, candidates.items(), RIGID_TOL)
    return f" That is what you get from {' or from '.join(names)}." if names else ""


def yaw_error(got, poses) -> np.ndarray:
    """Per pose, the angle between the yaw read off `got` (n, 3, 3) and the record's yaw, wrapped into [0, pi]."""
    d = np.arctan2(got[:, 1, 0], got[:, 0, 0]) - poses[:, 2]
    return np.abs(np.arctan2(np.sin(d), np.cos(d)))


def show(m) -> str:
    return np.array2string(np.asarray(m), precision=4, suppress_small=True)


def relative_mistakes(a, b) -> dict[str, np.ndarray]:
    """The outputs of the relative_pose mistakes the lesson names, at the pair (T_a, T_b)."""
    inv = np.linalg.inv  # the grader's own inverse, so it never leans on the learner's invert_pose
    slip_inverse = np.eye(3)  # the lab's invert_pose_slip: R where R transposed belongs, in both places
    slip_inverse[:2, :2], slip_inverse[:2, 2] = a[:2, :2], -a[:2, :2] @ a[:2, 2]
    world_difference = matrix(b[0, 2] - a[0, 2], b[1, 2] - a[1, 2], math.atan2(b[1, 0], b[0, 0]) - math.atan2(a[1, 0], a[0, 0]))
    return {
        "the arguments swapped, which is the step run backwards": inv(b) @ a,
        "the inverse built with R instead of its transpose, in both places": slip_inverse @ b,
        "the product the wrong way round, T_world<-b times the inverse of T_world<-a": b @ inv(a),
        "the difference of the two poses in world numbers, never turned into a's frame": world_difference,
    }


def reckon_mistakes(poses, steps) -> dict[str, np.ndarray]:
    """The outputs of the dead_reckon mistakes the lesson names, run on the lap's first pose and steps."""
    left = [matrix(*poses[0])]
    for step in steps:
        left.append(step @ left[-1])
    parts = np.column_stack([steps[:, 0, 2], steps[:, 1, 2], np.arctan2(steps[:, 1, 0], steps[:, 0, 0])])
    added = poses[0] + np.vstack([np.zeros(3), np.cumsum(parts, axis=0)])
    return {
        "each step multiplied on the left, step @ pose": np.stack(left),
        "adding each step's forward part to x, its left part to y and its turn to the angle": np.stack([matrix(*p) for p in added]),
    }


@functools.cache
def lap():
    """The gentle drive of the stock car: its poses (n, 3) and the odometry steps between neighbours (n - 1, 3, 3)."""
    tel = run(Car(), ROAD)
    poses = np.column_stack([tel.x, tel.y, tel.yaw])
    return poses, np.stack([matrix(*step_between(a, b)) for a, b in zip(poses[:-1], poses[1:])])


def rebuild(dead_reckon):
    """(poses, per-pose position error m, per-pose largest rotation-entry error) of `dead_reckon` run on the lap's first pose and steps."""
    poses, steps = lap()
    got = np.asarray(call(dead_reckon, matrix(*poses[0]), steps, given=f"the first pose and {len(steps)} odometry steps of {ROAD}"), dtype=float)
    if got.shape != (len(poses), 3, 3):
        raise Failure(f"returned an array of shape {got.shape}, expected {(len(poses), 3, 3)}: the start, then one pose per step",
                      given=f"{len(steps)} steps")
    if (why := not_rigid(got)) is not None:
        k = int(np.argmax([not_rigid(m) is not None for m in got]))
        raise Failure(f"pose {k} is not a pose matrix: {why}", given=f"the first pose and {len(steps)} steps of {ROAD}", got=show(got[k]))
    want = np.stack([matrix(*p) for p in poses])
    return got, np.hypot(got[:, 0, 2] - poses[:, 0], got[:, 1, 2] - poses[:, 1]), np.abs(got - want)[:, :2, :2].max(axis=(1, 2))


def matrix_3d(yaw, pitch, position) -> np.ndarray:
    """A 4x4 pose, rot_z(yaw) @ rot_y(pitch) with a position column: the next lesson's shape, written out here."""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    m = np.eye(4)
    m[:3, :3] = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    m[:3, 3] = position
    return m


def named_pairs():
    """(name, T_a, T_b, wanted T_a<-b) for the cases the lecture works by hand, and one 4x4 pair for "invert_pose works at any size"."""
    a, b = (3, 4, 0.7), (-2, 1, -1.2)
    yield "two poses away from the axes", matrix(*a), matrix(*b), matrix(*step_between(a, b))  # first: the mistakes below give three different answers here
    A, B = matrix(1, 0, math.pi / 2), matrix(1, 0, 0)
    yield "A then B", A, A @ B, B
    yield "B then A", B, B @ A, A
    T = matrix(3, 4, 0.7)
    yield "the same pose twice", T, T, np.eye(3)
    yield "a pure turn on the spot", matrix(3, 4, 0), matrix(3, 4, 0.5), matrix(0, 0, 0.5)
    a4, b4 = matrix_3d(0.7, 0.1, (1.2, 0.3, 1.5)), matrix_3d(-0.4, 0.2, (-2.0, 1.0, 0.5))
    yield "two 4x4 poses (the next lesson's size)", a4, b4, np.linalg.inv(a4) @ b4


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
        size = len(a)
        any_size = "" if size == 3 else f" The 3x3 pairs before it passed and this {size}x{size} one did not: relative_pose, and the invert_pose under it, must work at any size."
        try:
            got = np.asarray(call(fn, a, b, given=f"{name}: two {size}x{size} poses"), dtype=float)
        except Failure as failure:
            raise Failure(failure.headline + any_size, **failure.shown) from failure
        if got.shape != a.shape:
            raise Failure(f"returned an array of shape {got.shape}, expected {a.shape}." + any_size, given=name)
        if (why := not_rigid(got)) is not None:
            raise Failure(f"{name}: the result is not a pose matrix: {why}." + any_size, given=f"{name}: T_a =\n{show(a)}\nT_b =\n{show(b)}", got=show(got))
        worst = max(worst, float(np.abs(got - want).max()))
        if worst >= REL_TOL:
            raise Failure(f"{name}: off by {worst:.3g}, the limit is {REL_TOL:g}." + any_size + (named_mistake(got, want, relative_mistakes(a, b)) if size == 3 else ""),
                          given=f"{name}: T_a =\n{show(a)}\nT_b =\n{show(b)}", expected=want, got=got)
    return worst


THRESHOLDS = {
    "relative_pose, worst entry": Threshold(REL_TOL, operator.lt, LESSON[0], lambda solution: relative_pose_error(solution.relative_pose)),
    "dead_reckon, worst position (m)": Threshold(POS_TOL, operator.lt, LESSON[0], lambda solution: rebuild(solution.dead_reckon)[1].max()),
    "dead_reckon, worst rotation entry": Threshold(ANGLE_TOL, operator.lt, LESSON[0], lambda solution: rebuild(solution.dead_reckon)[2].max()),
}


@exercise(
    "1.1.4.a",
    checked=(f"relative_pose on five worked cases (two poses off the axes, A then B, B then A, the same pose twice, a pure turn), one pair of "
             f"4x4 poses and {N_PAIRS} seeded pairs: every entry within {REL_TOL:g} of the closed form"),
    section="What the car felt between two steps",
)
def _relative_pose(fn, show, on_step):
    worst = relative_pose_error(fn)
    return f"all {len(tuple(named_pairs())) + N_PAIRS} cases within {REL_TOL:g}: worst entry {fmt_gap(worst, REL_TOL, '.2g')} {vs_limit(worst, REL_TOL)}"


@exercise(
    "1.1.4.b",
    checked=(f"dead_reckon rebuilds the {ROAD} lap from its odometry steps: every pose within {POS_TOL:g} m and {ANGLE_TOL:g} rad "
             "of the simulator's record, not just the end"),
    section="Dead reckoning",
)
def _dead_reckon(fn, show, on_step):
    poses, steps = lap()
    got, position, entry = rebuild(fn)
    yaw = yaw_error(got, poses)
    if position.max() >= POS_TOL or entry.max() >= ANGLE_TOL:
        k = int(np.argmax(position / POS_TOL + entry / ANGLE_TOL))
        raise Failure(
            f"pose {k} of {len(poses)} is {position[k]:.4g} m and {yaw[k]:.3g} rad of yaw off (the end is {position[-1]:.4g} m off): "
            f"every pose is checked, the limit is {POS_TOL:g} m." + named_mistake(got, np.stack([matrix(*p) for p in poses]), reckon_mistakes(poses, steps)),
            given=f"the first pose and {len(steps)} steps of {ROAD}; expected and got are pose {k}'s (x, y) in m",
            expected=poses[k, :2], got=got[k, :2, 2],
        )
    return (f"{len(poses)} poses rebuilt from {len(steps)} steps: worst {fmt_gap(position.max(), POS_TOL, '.2g')} m {vs_limit(position.max(), POS_TOL, 'm')}, "
            f"end {fmt_gap(position[-1], POS_TOL, '.2g')} m, worst yaw error {fmt_gap(yaw.max(), ANGLE_TOL, '.2g')} rad")
