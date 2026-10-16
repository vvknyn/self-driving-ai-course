"""1.1.2 The view from the driver's seat: a world point in the car's own frame."""
from __future__ import annotations

import math

import numpy as np

from ... import geom
from ...car import Car
from ...sim import BicycleState, Camera
from ...sim.camera import to_vehicle
from .._drive import GATE_SCENARIO, gate
from .._report import Failure, call, exercise, fmt

LESSON = ("1.1.2", "The view from the driver's seat")  # (unit id, title): grade/__init__ files it in LESSON_TITLES
TOL, N_POSES, N_POINTS = 1e-9, 40, 25
CAR_AT, AHEAD, LEFT = (10.0, 5.0), 4.0, 1.0  # the named-pose test: a point 4 m ahead and 1 m left must come out as (4, 1)
NAMED_YAWS = {  # yaw label -> (yaw, what a pass at that yaw proves)
    "0": (0.0, "cannot tell R from its transpose, or from no rotation"),
    "pi/2": (math.pi / 2, "tells R from its transpose, and from no rotation"),
    "pi": (math.pi, "tells no rotation from the rest, but not R from its transpose"),
    "-3pi/4": (-3 * math.pi / 4, "tells R from its transpose, and from no rotation"),
}
NAME_TOL = 1e-6  # how close `got` must be to a known wrong answer for the message to name that mistake
SECTION = "Two dot products make a frame change"  # where the general message points: it names no cause


def known_mistakes(pose, point):
    """[(what the learner did, lecture section)] paired with the answer that mistake gives for `point` at `pose`."""
    x, y, yaw = pose
    t, p = np.array([x, y]), np.asarray(point, dtype=float)
    rot = geom.rotation_matrix(yaw)
    right = (p - t) @ rot
    return [
        ((p - t) @ rot.T, ("this is the swap: R(θ) where its transpose belongs", "Undo is the transpose")),
        (p - t, ("this is no rotation: p − t is never turned into the car's axes", "Two dot products make a frame change")),
        (p @ rot - t, ("this turns first and subtracts t afterwards: subtract the car's position first", "Undo is the transpose")),
        ((p - t) @ geom.rotation_matrix(math.radians(yaw)), ("this reads the yaw as degrees: θ is in radians", "The rotation matrix")),
        (p @ rot.T + t, ("this is ego_to_world, which goes the other way: world_to_ego is R(θ)ᵀ(p − t)", "Undo is the transpose")),
        (right[::-1], ("forward and left are in the wrong order: return (forward, left)", "Two dot products make a frame change")),
        (right * (1, -1), ("left has the wrong sign: the car's y axis points to its left", "Two dot products make a frame change")),
        (right * (-1, 1), ("forward has the wrong sign: the car's x axis points ahead", "Two dot products make a frame change")),
        ((p - t[::-1]) @ rot, ("the car's position is swapped: t is (east, north), as in the pose", "Two dot products make a frame change")),
    ]


def name_the_mistake(pose, point, got, label):
    """The headline for a wrong answer `got`: the mistake it matches, else only what the grader saw (how far off, or a constant gap)."""
    for answer, (what, section) in known_mistakes(pose, point):
        if got.shape == answer.shape and np.allclose(got, answer, rtol=0, atol=NAME_TOL):
            return f'wrong at yaw {label}: {what}. See the section "{section}"'
    right = geom.world_to_ego(pose, point)
    if got.shape != right.shape:
        seen = f"returned shape {got.shape}, expected {right.shape}"
    else:
        gap = got - right
        seen = (f"both entries are off by the same {gap[0]:.3g} m: look for a constant added or subtracted"
                if np.allclose(gap, gap[0], rtol=0, atol=NAME_TOL) else f"off by up to {np.abs(gap).max():.3g} m from (forward, left) = {fmt(right)}")
    return f'wrong at yaw {label}: {seen}. See the section "{SECTION}"'


def renderer_gap(fn) -> float:
    """Largest difference (m) between `fn` and the renderer's own transform over 40 seeded poses of 25 points each."""
    rng = np.random.default_rng(1102)
    poses = np.column_stack([rng.uniform(-200, 200, N_POSES), rng.uniform(-200, 200, N_POSES), rng.uniform(-4 * math.pi, 4 * math.pi, N_POSES)])
    points = rng.uniform(-200, 200, (N_POSES, N_POINTS, 2))
    worst = 0.0
    for (x, y, theta), pts in zip(poses, points):
        given = f"pose ({x:.1f}, {y:.1f}, {theta:.3f}) and {N_POINTS} points"
        got = np.asarray(call(fn, (float(x), float(y), float(theta)), pts.copy(), given=given), dtype=float)
        if got.shape != pts.shape:
            raise Failure(f"returned shape {got.shape}, expected {pts.shape}", given=given)
        forward, left = to_vehicle(BicycleState(x, y, theta, 0.0, 0.0), pts[:, 0], pts[:, 1])
        worst = max(worst, float(np.abs(got - np.stack([forward, left], axis=-1)).max()))
    return worst


@exercise(
    "1.1.2.a",
    checked=(f"a point {AHEAD:g} m ahead and {LEFT:g} m left of a car at {CAR_AT} at yaws 0, π/2, π and −3π/4; the (K, 4, 2) shape; "
             f"the round trip with the provided ego_to_world; then {N_POSES} seeded poses against the renderer's own transform, to {TOL:g} m"),
    section="Every point at once",
)
def _world_to_ego(fn, show, on_step):
    car = np.array(CAR_AT)
    for label, (yaw, _) in NAMED_YAWS.items():
        point = car + AHEAD * geom.heading_vector(yaw) + LEFT * np.array([-math.sin(yaw), math.cos(yaw)])
        given = f"pose {(*CAR_AT, round(yaw, 4))}, world point {fmt(point)} (yaw {label})"
        got = np.asarray(call(fn, (*CAR_AT, yaw), point.copy(), given=given), dtype=float)
        if got.shape != (2,) or not np.allclose(got, (AHEAD, LEFT), rtol=0, atol=TOL):
            raise Failure(name_the_mistake((*CAR_AT, yaw), point, got, label), given=given, expected=(AHEAD, LEFT), got=got)
    pose = (31.0, -12.0, 2.2)
    rows = np.array([[40.0, 3.0], [12.0, -20.0]])  # 2 points: the one shape where matrix @ rows runs without an error
    given = f"pose {pose} and 2 points as rows {fmt(rows)}"
    want = geom.world_to_ego(pose, rows)
    got = np.asarray(call(fn, pose, rows.copy(), given=given), dtype=float)
    if got.shape != want.shape or not np.allclose(got, want, rtol=0, atol=TOL):
        matrix_first = geom.rotation_matrix(pose[2]).T @ (rows - np.array(pose[:2]))
        what = ("this is matrix @ rows: with 2 points it runs and mixes the points' parts; points stored as rows go on the left, rows @ R"
                if got.shape == want.shape and np.allclose(got, matrix_first, rtol=0, atol=NAME_TOL) else "wrong for 2 points stored as rows")
        raise Failure(f'{what}. See the section "Every point at once"', given=given, expected=want, got=got)
    rng = np.random.default_rng(1103)
    cloud = rng.uniform(-200, 200, (146, 4, 2))
    ego = np.asarray(call(fn, pose, cloud.copy(), given="146 quads of 4 corners at one pose"), dtype=float)
    if ego.shape != cloud.shape:
        raise Failure(f"returned shape {ego.shape}, expected {cloud.shape}: one answer per point, whatever the leading shape",
                      given="a (146, 4, 2) array of points")
    back = float(np.abs(geom.ego_to_world(pose, ego) - cloud).max())
    if back > TOL:
        raise Failure(f"going to the car's frame and back moves points by up to {back:.3g} m", expected=f"at most {TOL:g} m",
                      given=f"pose {pose} and a (146, 4, 2) array of points")
    worst = check_against_renderer(fn)
    proved = "; ".join(f"yaw {label} {proof}" for label, (_, proof) in NAMED_YAWS.items())
    return f"all four yaws pass ({proved}), and so do the shape and the round trip; largest difference from the renderer's transform {worst:.2g} m over {N_POSES} poses"


def check_against_renderer(fn) -> float:
    """Fail unless `fn` matches the renderer's transform entry by entry; return the largest difference (m)."""
    worst = renderer_gap(fn)
    if worst > TOL:
        raise Failure(f"differs from the renderer's transform by up to {worst:.3g} m", expected=f"at most {TOL:g} m",
                      given=f"{N_POSES} seeded poses, yaws up to ±4π")
    return worst


@exercise(
    "1.1.2.b",
    checked=f"your world_to_ego against the renderer's transform at {N_POSES} seeded poses, entry by entry, then inside the simulator's camera: the provided car must complete `{GATE_SCENARIO}` on the pictures it draws",
    section="Swap your transform into the renderer",
)
def _renderer_gate(fn, show, on_step):
    try:
        check_against_renderer(fn)  # the entries first: a drive can pass on a transform that is slightly wrong
        return gate(Car(), show, on_step, camera=Camera(world_to_ego=fn))
    except Failure as failure:
        failure.headline = f"{failure.headline}. Run Exercise A first: it names the mistake"
        raise
