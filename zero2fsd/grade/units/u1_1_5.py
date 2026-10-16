"""1.1.5 The camera's point of view: the camera's mount as one 4×4, and where a pixel's ray meets the road."""
from __future__ import annotations

import math

import numpy as np

from ... import geom
from ...car import Car
from ...sim import BicycleState, Camera, make_scenario
from ...sim.scenarios import TARGET_SPEED
from .._drive import GATE_MEAN_LAT, GATE_SCENARIO, gate
from .._report import Failure, call, exercise, fmt, learner_code

LESSON = ("1.1.5", "The camera's point of view")  # (unit id, title): grade/__init__ files it in LESSON_TITLES
HEIGHT_M, PITCH_RAD = 1.4, math.radians(5.0)  # the simulator's own mount: what the learner's T_ego_cam must rebuild
SPEC_MOUNT = geom.T_ego_cam(HEIGHT_M, PITCH_RAD)
N_POINTS, N_PICTURES = 200, 18
AXIS_TOL, PIXEL_TOL, GROUND_TOL = 1e-4, 1e-6, 1e-9
RIGID_TOL, ENTRY_TOL, MISTAKE_TOL = 1e-9, 1e-8, 1e-6  # a 4×4 is a turn-plus-shift; entries equal the reference; a named mistake's output matches to this
ENTRY_SETTINGS = ((HEIGHT_M, PITCH_RAD), (HEIGHT_M, PITCH_RAD + math.radians(1.0)), (HEIGHT_M + 0.1, PITCH_RAD))  # the settings the 4×4 is compared at, entry by entry
ROWS_BELOW, ROWS_ABOVE = (76.0, 179.5), (0.0, 75.0)  # image rows whose rays do, and do not, reach the road (the horizon is row 75.5)


def as_array(raw, shape: tuple[int, ...], what: str, given: str) -> np.ndarray:
    """`raw` as a float array of `shape`; anything else (None, a list of the wrong size, a string) is a Failure that says `what` to build."""
    try:
        array = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        array = None
    if array is None or array.shape != shape:
        got = ("None" if raw is None else f"a {type(raw).__name__}" if array is None
               else "a single number" if array.ndim == 0 else f"an array of shape {array.shape}")
        raise Failure(f"your function returned {got}; {what}", given=given)
    return array


def triple(a) -> str:
    """Three numbers as `(x, y, z)`, the way the lecture writes them."""
    return fmt(tuple(np.asarray(a, dtype=float).tolist()))


def not_rigid(T: np.ndarray) -> str | None:
    """What is wrong with the (4, 4) `T` as a turn-plus-shift (a rigid transform), in plain words; None when nothing is."""
    if not np.isfinite(T).all():
        return "your 4×4 holds nan or inf"
    if not np.allclose(T[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=RIGID_TOL):
        return f"the last row of your 4×4 must be 0 0 0 1 (the trailing 1 is what lets a point be shifted); yours is {fmt(T[3])}"
    turn = T[:3, :3]
    if not np.allclose(turn @ turn.T, np.eye(3), rtol=0.0, atol=RIGID_TOL):
        return "the top-left 3×3 block of your 4×4 is not a pure turn: its columns must have length 1 and be at right angles (is it rounded, scaled or skewed?)"
    if np.linalg.det(turn) < 0:
        return "the top-left 3×3 block of your 4×4 mirrors instead of turning (its determinant is −1, not +1)"
    return None


def mount(fn, height_m: float = HEIGHT_M, pitch_rad: float = PITCH_RAD) -> np.ndarray:
    """The learner's `T_ego_cam(height_m, pitch_rad)` as a rigid (4, 4) array; a Failure naming what is wrong when it is not one."""
    given = f"T_ego_cam({height_m:g}, {pitch_rad:.4f})"
    T = as_array(call(fn, height_m, pitch_rad, given=given), (4, 4), "build the 4×4 mount pose", given)
    broken = not_rigid(T)
    if broken:
        raise Failure(broken, given=given, got=fmt(T))
    return T


def mount_from(turn: np.ndarray, height_m: float) -> np.ndarray:
    """A 4×4 with `turn` in the top-left block and the lens at (0, 0, height_m)."""
    T = np.eye(4)
    T[:3, :3], T[2, 3] = turn, height_m
    return T


MISTAKES = (  # (what the learner's 4×4 is, the 4×4 that mistake builds): the ones the lecture warns about, and the ones it implies
    ("your camera tips up; a positive pitch must tip the nose down", lambda h, p: geom.T_ego_cam(h, -p)),
    ("your turns are in the wrong order, so the camera looks level and to the right; the pitch goes on the left, rot_y(φ) @ Bᵀ",
     lambda h, p: mount_from(geom.CAM_FROM_EGO_AXES.T @ geom.rot_y(p), h)),
    ("the 3×3 block of your 4×4 turns the opposite way (it is transposed)", lambda h, p: mount_from(geom.T_ego_cam(h, p)[:3, :3].T, h)),
    ("your 4×4 is the inverse mount, camera-from-ego; T_ego_cam is ego-from-camera, with the lens position (0, 0, h) in its last column",
     lambda h, p: geom.invert_pose(geom.T_ego_cam(h, p))),
    ("your 4×4 treats the pitch as degrees; it arrives in radians", lambda h, p: geom.T_ego_cam(h, math.radians(p))),
)


def look_words(axis: np.ndarray) -> str:
    """Where the unit vector `axis` (ego forward, left, up) points, the way the lecture says it: `5.0° down`, `0.5° left of straight ahead and level`."""
    look, dip = math.degrees(math.atan2(axis[1], axis[0])), math.degrees(-math.asin(np.clip(axis[2], -1.0, 1.0)))
    sideways = "straight ahead" if abs(look) < 0.05 else f"{abs(look):.1f}° {'left' if look > 0 else 'right'} of straight ahead"
    vertical = "level" if abs(dip) < 0.05 else f"{abs(dip):.1f}° {'down' if dip > 0 else 'up'}"
    return f"{sideways} and {vertical}"


def check_mount(fn) -> np.ndarray:
    """The learner's `T_ego_cam` as a rigid (4, 4) array, accepted only when it IS the reference mount: the first thing wrong is a Failure
    that names the mistake the 4×4 matches, else which way the lens points, where it sits, or the first entry that differs. The entries are
    compared at three settings, and the pixels last: the pixels alone cannot see a bottom row, and the drive cannot see a wrong sign it never uses."""
    T, given = mount(fn), f"T_ego_cam({HEIGHT_M:g}, {PITCH_RAD:.4f})"
    for headline, build in MISTAKES:
        if np.allclose(T, build(HEIGHT_M, PITCH_RAD), rtol=0.0, atol=MISTAKE_TOL):
            raise Failure(headline, given=given, got=fmt(T))
    axis, want = geom.transform_directions(T, [0.0, 0.0, 1.0]), geom.transform_directions(SPEC_MOUNT, [0.0, 0.0, 1.0])
    if not np.allclose(axis, want, atol=AXIS_TOL, rtol=0.0):
        raise Failure(f"your lens looks {look_words(axis)}; it must look {look_words(want)}", given=f"{given} turns the lens axis (0, 0, 1) as a direction",
                      expected=f"the axis in ego numbers: {triple(want)}", got=triple(axis))
    lens = geom.transform_points(T, [0.0, 0.0, 0.0])
    if not np.allclose(lens, SPEC_MOUNT[:3, 3], atol=AXIS_TOL, rtol=0.0):
        raise Failure(f"your lens sits at {triple(lens)} in ego numbers; it must be {HEIGHT_M:g} m straight above the ego origin", given=f"{given} moves the camera's origin (0, 0, 0) as a point",
                      expected=triple(SPEC_MOUNT[:3, 3]), got=triple(lens))
    for height_m, pitch_rad in ENTRY_SETTINGS:
        mine, spec = mount(fn, height_m, pitch_rad), geom.T_ego_cam(height_m, pitch_rad)
        if not np.allclose(mine, spec, rtol=0.0, atol=ENTRY_TOL):
            row, col = np.unravel_index(np.argmax(np.abs(mine - spec)), spec.shape)
            raise Failure(f"your 4×4 differs from the reference mount in row {row + 1}, column {col + 1}", given=f"T_ego_cam({height_m:g}, {pitch_rad:.4f})",
                          expected=fmt(spec[row, col]), got=fmt(mine[row, col]))
    gap = pixel_gap(T)
    if not gap <= PIXEL_TOL:
        raise Failure(f"your mount puts road points up to {fmt(gap)} px from where the simulator draws them",
                      given=f"{N_POINTS} seeded road points through your T_ego_cam and the pinhole step", expected=f"within {PIXEL_TOL:g} px")
    return T


def pixel_gap(T: np.ndarray) -> float:
    """Largest distance (px) between where `T` projects 200 seeded road points and where the simulator's own camera draws them."""
    rng = np.random.default_rng(0)
    forward, left = rng.uniform(2.0, 60.0, N_POINTS), rng.uniform(-8.0, 8.0, N_POINTS)
    mine, spec = (np.array(cam.ground_to_pixel(forward, left)) for cam in (Camera(T_ego_cam=T), Camera()))
    return float(np.max(np.abs(mine - spec)))  # NaN stays NaN: a mount that loses a point never reads as close


def pictures_that_differ(camera: Camera, scenario: str = GATE_SCENARIO) -> int:
    """How many of 18 pictures along the road `camera` draws differently from the simulator's own camera; the learner's world_to_ego runs here."""
    road, spec = make_scenario(scenario).road, Camera()
    states = [BicycleState(*road.ego_pose(s), TARGET_SPEED, 0.0) for s in np.linspace(0.0, road.length, N_PICTURES)]
    with learner_code("your world_to_ego drawing the road in a camera picture"), np.errstate(invalid="ignore"):  # a nonsense world_to_ego makes nan pixels: drawn off the picture, not a warning
        return sum(not np.array_equal(camera.render(road, st), spec.render(road, st)) for st in states)


@exercise(
    "1.1.5.a",
    checked=(f"your T_ego_cam({HEIGHT_M}, 5°): a rigid 4×4 (last row 0 0 0 1, a pure turn on top) equal entry by entry to the reference mount at three settings, "
             f"the lens axis (cos φ, 0, −sin φ) in ego numbers, and {N_POINTS} seeded road points, 2 to 60 m ahead and up to 8 m to either side, within {PIXEL_TOL:g} px of where the simulator draws them"),
    section="One 4×4 for the mount",
)
def _mount(fn, show, on_step):
    T = check_mount(fn)
    axis = geom.transform_directions(T, [0.0, 0.0, 1.0])
    return f"the 4×4 matches the reference mount, the lens axis is {triple(axis)} and {N_POINTS} road points land within {PIXEL_TOL:g} px of the simulator's (largest gap {fmt(pixel_gap(T))} px)"


def mount_pair(target) -> tuple:
    """`(world_to_ego, T_ego_cam)`, the two pieces of the learner's code the simulator draws a camera picture through; a Failure when `target` is not that pair."""
    if not (isinstance(target, tuple) and len(target) == 2 and all(callable(part) for part in target)):
        raise Failure("your answer must be the pair of your two functions, (world_to_ego, T_ego_cam)", given="the lab builds it as camera_mount = (world_to_ego, T_ego_cam)", got=type(target).__name__)
    return target


@exercise(
    "1.1.5.b",
    checked=(f"your world_to_ego and T_ego_cam swapped into the simulator's camera on `{GATE_SCENARIO}`: the mount must pass Exercise A's checks, "
             f"none of {N_PICTURES} sampled camera pictures may differ from the simulator's own, and the provided car must complete the lap with mean distance from the lane centre < {GATE_MEAN_LAT} m"),
    section="A camera mounted wrong",
)
def _swap_mount(target, show, on_step):
    world_to_ego, fn = mount_pair(target)
    camera = Camera(world_to_ego=world_to_ego, T_ego_cam=check_mount(fn))  # a wrong mount stops here: it never reaches the renderer
    differ = pictures_that_differ(camera)
    if differ:
        raise Failure(f"your world_to_ego draws {differ} of {N_PICTURES} sampled camera pictures differently from the simulator's own; your mount matched the reference, so look at world_to_ego",
                      given=f"{N_PICTURES} poses along {GATE_SCENARIO}, drawn through your world_to_ego and your T_ego_cam and through the simulator's camera",
                      expected="0 pictures that differ", got=f"{differ} pictures that differ")
    with np.errstate(invalid="ignore"):
        drove = gate(Car(), show, on_step, camera=camera)
    return f"{drove}; none of {N_PICTURES} sampled pictures differ, so this is the provided car's own error on {GATE_SCENARIO} (it gets the same with the simulator's camera), not your camera's"


def ground_point(fn, T: np.ndarray, u: float, v: float, what: str = "return the road point as 3 numbers (forward, left, up)") -> np.ndarray:
    """The learner's `ray_to_ground(T, ray)` for the ray through pixel (u, v), as a 3-vector (ego forward, left, up)."""
    given = f"ray_to_ground(T_ego_cam(1.4, 5°), the ray through pixel ({u:.1f}, {v:.1f}))"
    return as_array(call(fn, T, geom.pixel_ray(Camera(), u, v), given=given), (3,), what, given)


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
        hint = "; if all of them are nan, your ray probably picked up the lens's shift (w = 1): a ray is a direction and only turns" if len(missing) == N_POINTS else ""
        raise Failure(f"{len(missing)} of {N_POINTS} pixels below the horizon came back nan; a ray that points down always meets the road{hint}",
                      given=f"the first one: pixel ({u:.1f}, {v:.1f})", expected=triple(want), got=triple(got))
    aloft = max(answers, key=lambda a: abs(a[2][2]))
    if abs(aloft[2][2]) > GROUND_TOL:
        raise Failure(f"your point is {aloft[2][2]:.3g} m above the road; a road point has up = 0, which is what λ = −t_z / r_z guarantees",
                      given=f"pixel ({aloft[0]:.1f}, {aloft[1]:.1f})", expected=triple(aloft[3]), got=triple(aloft[2]))
    gaps = ground_gaps(answers)
    worst = int(np.argmax(gaps))
    if gaps[worst] > GROUND_TOL:
        u, v, got, want = answers[worst]
        raise Failure(f"your road points are up to {fmt(gaps[worst])} m from the simulator's own (limit {GROUND_TOL:g} m)",
                      given=f"the worst one: pixel ({u:.1f}, {v:.1f})", expected=triple(want), got=triple(got))
    lands = [(u, v) for u, v in pixels_in(ROWS_ABOVE) if not all(bool(np.isnan(c)) is True for c in ground_point(fn, SPEC_MOUNT, u, v, "above the horizon there is no road point: return three nans, np.full(3, np.nan)")[:2])]
    if lands:
        raise Failure(f"{len(lands)} of {N_POINTS} pixels above the horizon got a road point; their rays never meet the road, so the answer is nan",
                      given=f"the first one: pixel ({lands[0][0]:.1f}, {lands[0][1]:.1f})", expected="nan", got="a road point")
    return f"{N_POINTS} pixels below the horizon land within {GROUND_TOL:g} m (largest gap {fmt(max(gaps))} m), and {N_POINTS} above it are nan"
