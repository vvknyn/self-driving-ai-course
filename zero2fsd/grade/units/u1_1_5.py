"""1.1.5 The camera's point of view: the camera's mount as one 4×4, and where a pixel's ray meets the road."""
from __future__ import annotations

import math

import numpy as np

from ... import geom
from ...car import Car
from ...sim import BicycleState, Camera, make_scenario
from ...sim.scenarios import TARGET_SPEED
from .._drive import GATE_MEAN_LAT, GATE_SCENARIO, gate
from .._report import Failure, call, exercise, fmt

LESSON = ("1.1.5", "The camera's point of view")  # (unit id, title): grade/__init__ files it in LESSON_TITLES
HEIGHT_M, PITCH_RAD = 1.4, math.radians(5.0)  # the simulator's own mount: what the learner's T_ego_cam must rebuild
SPEC_MOUNT = geom.T_ego_cam(HEIGHT_M, PITCH_RAD)
N_POINTS, N_PICTURES = 200, 18
AXIS_TOL, PIXEL_TOL, GROUND_TOL = 1e-4, 1e-6, 1e-9
ROWS_BELOW, ROWS_ABOVE = (76.0, 179.5), (0.0, 75.0)  # image rows whose rays do, and do not, reach the road (the horizon is row 75.5)


def as_array(raw, shape: tuple[int, ...], what: str, given: str) -> np.ndarray:
    """`raw` as a float array of `shape`; anything else (None, a list of the wrong size, a string) is a Failure that says `what` to build."""
    try:
        array = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        array = None
    if array is None or array.shape != shape:
        got = "None" if raw is None else f"an array of shape {array.shape}" if array is not None else f"a {type(raw).__name__}"
        raise Failure(f"your function returned {got}; {what}", given=given)
    return array


def triple(a) -> str:
    """Three numbers as `(x, y, z)`, the way the lecture writes them."""
    return fmt(tuple(np.asarray(a, dtype=float).tolist()))


def mount(fn, height_m: float = HEIGHT_M, pitch_rad: float = PITCH_RAD) -> np.ndarray:
    """The learner's `T_ego_cam(height_m, pitch_rad)` as a (4, 4) array; a Failure when it is not one."""
    given = f"T_ego_cam({height_m}, {pitch_rad:.4f})"
    return as_array(call(fn, height_m, pitch_rad, given=given), (4, 4), "build the 4×4 mount pose", given)


def pixel_gap(T: np.ndarray) -> float:
    """Largest distance (px) between where `T` projects 200 seeded road points and where the simulator's own camera draws them."""
    rng = np.random.default_rng(0)
    forward, left = rng.uniform(2.0, 60.0, N_POINTS), rng.uniform(-8.0, 8.0, N_POINTS)
    mine, spec = (np.array(cam.ground_to_pixel(forward, left)) for cam in (Camera(T_ego_cam=T), Camera()))
    return float(np.max(np.abs(mine - spec)))  # NaN stays NaN: a mount that loses a point never reads as close


def pictures_that_differ(T: np.ndarray, scenario: str = GATE_SCENARIO) -> int:
    """How many of 18 pictures along the road a camera mounted with `T` draws differently from the simulator's own camera."""
    road, mounted, spec = make_scenario(scenario).road, Camera(T_ego_cam=T), Camera()
    states = [BicycleState(*road.ego_pose(s), TARGET_SPEED, 0.0) for s in np.linspace(0.0, road.length, N_PICTURES)]
    return sum(not np.array_equal(mounted.render(road, st), spec.render(road, st)) for st in states)


@exercise(
    "1.1.5.a",
    checked=(f"your T_ego_cam({HEIGHT_M}, 5°): the lens axis must point (cos φ, 0, −sin φ) in ego numbers, and {N_POINTS} seeded road points, "
             f"2 to 60 m ahead and up to 8 m to either side, must land within {PIXEL_TOL:g} px of where the simulator draws them"),
    section="One 4×4 for the mount",
)
def _mount(fn, show, on_step):
    T = mount(fn)
    axis, want = geom.transform_directions(T, [0.0, 0.0, 1.0]), np.array([math.cos(PITCH_RAD), 0.0, -math.sin(PITCH_RAD)])
    if not np.allclose(axis, want, atol=AXIS_TOL, rtol=0.0):
        up = np.allclose(axis, want * [1.0, 1.0, -1.0], atol=AXIS_TOL, rtol=0.0)
        raise Failure("your camera tips up; a positive pitch must tip the nose down" if up else "the lens axis points the wrong way",
                      given=f"T_ego_cam({HEIGHT_M}, {PITCH_RAD:.4f}) turns the lens axis (0, 0, 1) as a direction",
                      expected=f"the axis in ego numbers: {triple(want)}", got=triple(axis))
    gap = pixel_gap(T)
    if not gap <= PIXEL_TOL:
        raise Failure(f"your mount puts road points up to {fmt(gap)} px from where the simulator draws them",
                      given=f"{N_POINTS} seeded road points through your T_ego_cam and the pinhole step", expected=f"within {PIXEL_TOL:g} px")
    return f"the lens axis is {triple(axis)} and {N_POINTS} road points land within {PIXEL_TOL:g} px of the simulator's (largest gap {fmt(gap)} px)"


@exercise(
    "1.1.5.b",
    checked=(f"your T_ego_cam swapped into the simulator's camera on `{GATE_SCENARIO}`: the provided car must complete it with mean |e| < {GATE_MEAN_LAT} m, "
             f"and none of {N_PICTURES} sampled camera pictures may differ from the simulator's own"),
    section="A camera mounted wrong",
)
def _swap_mount(fn, show, on_step):
    T = mount(fn)
    drove = gate(Car(), show, on_step, camera=Camera(T_ego_cam=T))
    differ = pictures_that_differ(T)
    if differ:
        raise Failure(f"your mount drives, but {differ} of {N_PICTURES} sampled camera pictures differ from the simulator's own",
                      given=f"{N_PICTURES} poses along {GATE_SCENARIO}, drawn through your T_ego_cam and through the simulator's camera",
                      expected="0 pictures that differ", got=f"{differ} pictures that differ")
    return f"{drove}; none of {N_PICTURES} sampled pictures differ"


def ground_point(fn, T: np.ndarray, u: float, v: float) -> np.ndarray:
    """The learner's `ray_to_ground(T, ray)` for the ray through pixel (u, v), as a 3-vector (ego forward, left, up)."""
    given = f"ray_to_ground(T_ego_cam(1.4, 5°), the ray through pixel ({u:.1f}, {v:.1f}))"
    return as_array(call(fn, T, geom.pixel_ray(Camera(), u, v), given=given), (3,), "return the road point as 3 numbers (forward, left, up)", given)


def pixels_in(rows: tuple[float, float]) -> np.ndarray:
    """200 seeded (u, v) pixels with v in `rows`."""
    rng = np.random.default_rng(1)
    return np.column_stack([rng.uniform(-0.5, 319.5, N_POINTS), rng.uniform(*rows, N_POINTS)])


def ground_answers(fn) -> list[tuple[float, float, np.ndarray, np.ndarray]]:
    """(u, v, the learner's road point, the simulator's) for 200 seeded pixels below the horizon."""
    camera = Camera()
    return [(u, v, ground_point(fn, SPEC_MOUNT, u, v), np.array([*camera.pixel_to_ground(u, v), 0.0])) for u, v in pixels_in(ROWS_BELOW)]


def ground_gaps(answers) -> list[float]:
    """Each answer's largest coordinate difference (m); NaN when the learner's point is nan."""
    return [float(np.max(np.abs(got - want))) for _, _, got, want in answers]


@exercise(
    "1.1.5.c",
    checked=(f"your ray_to_ground on {N_POINTS} seeded pixels below the horizon: the road point within {GROUND_TOL:g} m of the simulator's own; "
             f"and on {N_POINTS} pixels above it: nan"),
    section="Where a pixel's ray meets the road",
)
def _ray_to_ground(fn, show, on_step):
    answers = ground_answers(fn)
    missing = [a for a in answers if any(bool(np.isnan(c)) is True for c in a[2])]
    if missing:
        u, v, got, want = missing[0]
        raise Failure(f"{len(missing)} of {N_POINTS} pixels below the horizon came back nan; a ray that points down always meets the road",
                      given=f"the first one: pixel ({u:.1f}, {v:.1f})", expected=triple(want), got=triple(got))
    gaps = ground_gaps(answers)
    worst = int(np.argmax(gaps))
    if gaps[worst] > GROUND_TOL:
        u, v, got, want = answers[worst]
        raise Failure(f"your road points are up to {fmt(gaps[worst])} m from the simulator's own (limit {GROUND_TOL:g} m)",
                      given=f"the worst one: pixel ({u:.1f}, {v:.1f})", expected=triple(want), got=triple(got))
    lands = [(u, v) for u, v in pixels_in(ROWS_ABOVE) if not all(bool(np.isnan(c)) is True for c in ground_point(fn, SPEC_MOUNT, u, v)[:2])]
    if lands:
        raise Failure(f"{len(lands)} of {N_POINTS} pixels above the horizon got a road point; their rays never meet the road, so the answer is nan",
                      given=f"the first one: pixel ({lands[0][0]:.1f}, {lands[0][1]:.1f})", expected="nan", got="a road point")
    return f"{N_POINTS} pixels below the horizon land within {GROUND_TOL:g} m (largest gap {fmt(max(gaps))} m), and {N_POINTS} above it are nan"
