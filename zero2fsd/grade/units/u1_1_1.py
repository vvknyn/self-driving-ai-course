"""1.1.1 Positions and headings: the direction of an arrow, and its forward and left parts."""
from __future__ import annotations

import math
import operator
from functools import cache

import numpy as np

from ...car import Car, run
from ...sim import BicycleState
from ...sim.camera import to_vehicle
from ...sim.scenarios import TARGET_SPEED
from ...sim.world import wrap_angle
from .._report import Failure, Threshold, call, exercise, fmt, is_real, vs_limit

LESSON = ("1.1.1", "Positions and headings")  # (unit id, title): grade/__init__ files it in LESSON_TITLES
TOL = 1e-12  # direction_angle against np.arctan2
PARTS_TOL = 1e-9  # forward_left against the camera's to_vehicle
LAP_TOL = 0.05  # rad: direction_angle of a step against the car's yaw (the provided car's worst is 0.041)
NAMED = [((1, 0), 0.0), ((0, 1), math.pi / 2), ((-1, 0), math.pi), ((0, -1), -math.pi / 2), ((1, 1), math.pi / 4),
         ((-1, 1), 3 * math.pi / 4), ((-1, -1), -3 * math.pi / 4), ((1, -1), -math.pi / 4),
         ((-1, 1e-12), math.pi - 1e-12), ((-1, -1e-12), -math.pi + 1e-12)]  # one per quarter and axis, and just either side of the cut at pi


@cache
def _lap() -> tuple[np.ndarray, np.ndarray]:
    """(step arrows, yaw wrapped into [-pi, pi)) of the provided car's lap of `gentle`; read-only, hand out copies."""
    tel = run(Car(), "gentle")
    arrows = np.diff(np.stack([tel.x, tel.y], axis=-1), axis=0)
    yaw = wrap_angle(tel.yaw[:-1])
    arrows.setflags(write=False)
    yaw.setflags(write=False)
    return arrows, yaw


def miss_from_yaw(angles) -> float:
    """The largest |wrap(angle - yaw)| over the lap's steps; the difference is wrapped, so an answer near pi is not far off."""
    return float(np.abs(wrap_angle(angles - _lap()[1])).max())


def lap_miss(direction_angle) -> float:
    return miss_from_yaw(direction_angle(_lap()[0].copy()))


THRESHOLDS = {"direction against yaw on gentle (rad)": Threshold(LAP_TOL, operator.lt, LESSON[0], lambda solution: lap_miss(solution.direction_angle))}


def _as_floats(got, given: str) -> np.ndarray:
    """The learner's answer as a float array, or a Failure when it is not numbers."""
    try:
        return np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        raise Failure(f"expected numbers, got {type(got).__name__}", given=given, got=got) from None


def _is_nan(value) -> bool:
    """A single float answer (a 0-d array counts) that is nan; ints, bools, strings, None and containers are not."""
    number = np.asarray(value)
    return number.ndim == 0 and number.dtype.kind == "f" and bool(np.isnan(number))


def _why_wrong(got, v) -> str:
    """The headline for a wrong direction: names the mistake whose output `got` matches at the arrow `v` (G-msg), else the general one."""
    if np.ndim(got) != 0:
        return f"expected one number for one arrow, got shape {np.shape(got)}"
    if not is_real(got):
        return "wrong direction for an arrow"
    x, y = float(v[0]), float(v[1])
    want = math.atan2(y, x)
    near = lambda other: abs(got - other) <= TOL
    with np.errstate(all="ignore"):
        slope = float(np.arctan(y / x)) if x else math.nan
    if near(math.pi / 2 - want) and not near(want):
        return "Δx and Δy look swapped: np.arctan2 takes Δy first"
    if near(math.degrees(want)):
        return "this looks like degrees: give radians"
    if near(slope):
        return "this is np.arctan of the slope, which cannot tell an arrow from its opposite: use np.arctan2 or fix the angle where Δx < 0"
    if abs(wrap_angle(got - want)) <= TOL:
        return "the right direction in another angle range: direction_angle must give np.arctan2's range, −π to π, with west as +π"
    return "wrong direction for an arrow"


@exercise(
    "1.1.1.a",
    checked=(f"{len(NAMED)} named arrows (each quarter, each axis, and just either side of the cut at π) and 30 seeded arrows (one of them the zero arrow) against "
             f"np.arctan2 to {TOL}; the zero arrow gives nan; many arrows give one angle each; and on `gentle` every step arrow "
             f"against np.arctan2, and within {LAP_TOL} rad of the car's yaw (difference wrapped)"),
    section="Which way is it pointing",
)
def _direction_angle(fn, show, on_step):
    for v, want in NAMED:
        given = f"v = {v}"
        got = call(fn, np.array(v, dtype=float), given=given)
        if not is_real(got) or np.ndim(got) != 0 or abs(got - want) > TOL:
            raise Failure(_why_wrong(got, v), given=given, expected=want, got=got)
    given = "v = (0, 0), the zero arrow"
    got = call(fn, np.zeros(2), given=given)
    if not _is_nan(got):
        raise Failure("the zero arrow has no direction: it must give nan", given=given, expected="nan", got=got)
    vectors = np.random.default_rng(1111).normal(0.0, 5.0, (30, 2))
    zero_row = 17
    vectors[zero_row] = 0.0  # the zero arrow inside a batch: it must give nan too, not east
    given = f"{len(vectors)} seeded arrows of shape {vectors.shape}, row {zero_row} the zero arrow"
    got = _as_floats(call(fn, vectors, given=given), given)
    want = np.arctan2(vectors[:, 1], vectors[:, 0])
    want[zero_row] = np.nan
    if np.shape(got) != want.shape:
        raise Failure(f"expected one angle per arrow, shape {want.shape}", given=given, got=f"shape {np.shape(got)}")
    right = np.where(np.isnan(want), np.isnan(got), np.abs(got - want) <= TOL)
    if not right.all():
        worst = int(np.argmin(right))
        if worst == zero_row:
            raise Failure(f"the zero arrow inside a batch has no direction: row {zero_row} must be nan", given=given, expected="nan", got=got[worst])
        raise Failure(_why_wrong(got[worst], vectors[worst]), given=f"arrow {fmt(vectors[worst])}", expected=want[worst], got=got[worst])
    arrows, _ = _lap()
    given = f"the {len(arrows)} step arrows of the provided car's lap of gentle"
    got = _as_floats(call(fn, arrows, given=given), given)
    if np.shape(got) != (len(arrows),):
        raise Failure(f"expected one angle per arrow, shape {(len(arrows),)}", given=given, got=f"shape {np.shape(got)}")
    want_lap = np.arctan2(arrows[:, 1], arrows[:, 0])
    if not np.all(np.abs(got - want_lap) <= TOL):  # entry by entry first; the yaw miss below is extra evidence
        wrong = np.abs(got - want_lap) > TOL
        worst = int(np.argmax(np.where(wrong, np.abs(wrap_angle(got - want_lap)), -1.0)))  # worst by the wrapped miss, so a +pi/-pi flip does not outrank a real error
        raise Failure(_why_wrong(got[worst], arrows[worst]), given=f"step arrow {worst}: {fmt(arrows[worst])}", expected=want_lap[worst], got=got[worst])
    miss = miss_from_yaw(got)
    if not miss < LAP_TOL:
        raise Failure(f"some step's direction is far from the car's yaw: worst miss {fmt(miss)} rad {vs_limit(miss, LAP_TOL, 'rad')}", given=given)
    return f"every case matches np.arctan2; the worst step on gentle is {miss:.4f} rad from the yaw {vs_limit(miss, LAP_TOL, 'rad')}"


def _parts_mistake(theta: float, v, got) -> str:
    """The headline for a wrong (forward, left) at the arrow `v`: names the mistake whose output `got` matches (G-msg), else the general one."""
    c, s, vx, vy = math.cos(theta), math.sin(theta), float(v[0]), float(v[1])
    forward = c * vx + s * vy
    mistakes = [  # (what the left part would be, the message)
        (vx * s - vy * c, "the left part has the wrong sign: it is ĥ × v = cos θ · v_y − sin θ · v_x, not v × ĥ"),
        (s * vx + c * vy, "the left arrow is n̂ = (−sin θ, cos θ), not (sin θ, cos θ)"),
    ]
    for left, message in mistakes:
        if abs(got[0] - forward) <= PARTS_TOL and abs(got[1] - left) <= PARTS_TOL:
            return message
    return "wrong forward or left part"


def _parts(pose, points):
    """(forward, left) of each point (world frame) for a car at `pose` = (x, y, yaw), by the camera's own `to_vehicle`."""
    x, y, theta = pose
    forward, left = to_vehicle(BicycleState(float(x), float(y), float(theta), TARGET_SPEED, 0.0), points[..., 0], points[..., 1])
    return np.stack([forward, left], axis=-1)


@exercise(
    "1.1.1.b",
    checked=(f"forward and left parts of 25 seeded arrows for each of 40 seeded poses (x, y up to 200 m, yaw up to 10 rad), and for 40 "
             f"yaws at once, against the camera's own `to_vehicle` to {PARTS_TOL}"),
    section="Ahead and to the left",
)
def _forward_left(fn, show, on_step):
    rng = np.random.default_rng(1112)
    poses = np.column_stack([rng.uniform(-200, 200, 40), rng.uniform(-200, 200, 40), rng.uniform(-10, 10, 40)])
    points = rng.uniform(-50, 50, (40, 25, 2))
    for pose, p in zip(poses, points):
        arrows, given = p - pose[:2], f"yaw {pose[2]:.4f} rad and 25 arrows of shape (25, 2)"
        got, want = _as_floats(call(fn, float(pose[2]), arrows, given=given), given), _parts(pose, p)
        if np.shape(got) != want.shape:
            raise Failure(f"expected shape {want.shape}: (forward, left) for each arrow", given=given, got=f"shape {np.shape(got)}")
        if not np.all(np.abs(got - want) <= PARTS_TOL):
            bad = int(np.argmax(np.where(np.abs(got - want) <= PARTS_TOL, -1.0, np.abs(got - want)).max(axis=1)))
            raise Failure(_parts_mistake(pose[2], arrows[bad], got[bad]), given=f"yaw {pose[2]:.4f} rad and arrow {fmt(arrows[bad])}", expected=want[bad], got=got[bad])
    given = "40 yaws, one arrow each: θ of shape (40,), v of shape (40, 2)"
    got = _as_floats(call(fn, poses[:, 2], points[:, 0] - poses[:, :2], given=given), given)
    want = np.stack([_parts(pose, p[0]) for pose, p in zip(poses, points)])
    if np.shape(got) != want.shape or not np.all(np.abs(got - want) <= PARTS_TOL):
        raise Failure("with an array of yaws, row i must use yaw i", given=given, expected=f"shape {want.shape}", got=f"shape {np.shape(got)}")
    return "40 poses x 25 arrows, and 40 yaws at once, all match the camera's own forward and left"
