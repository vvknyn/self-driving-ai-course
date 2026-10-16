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
NAMED_YAWS = {  # yaw -> what a pass at that yaw proves (and so what a failure there still leaves open)
    "0": (0.0, "at yaw 0 the rotation is the identity, so this yaw cannot tell R from its transpose, or from no rotation"),
    "pi/2": (math.pi / 2, "this yaw catches both R in place of its transpose and a missing rotation"),
    "pi": (math.pi, "this yaw catches a missing rotation, but not R in place of its transpose"),
    "-3pi/4": (-3 * math.pi / 4, "this yaw catches both R in place of its transpose and a missing rotation"),
}


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
             f"the round trip with ego_to_world; then {N_POSES} seeded poses against the renderer's own transform, to {TOL:g} m"),
    section="Every point at once",
)
def _world_to_ego(fn, show, on_step):
    car = np.array(CAR_AT)
    for label, (yaw, meaning) in NAMED_YAWS.items():
        point = car + AHEAD * geom.heading_vector(yaw) + LEFT * np.array([-math.sin(yaw), math.cos(yaw)])
        given = f"pose {(*CAR_AT, round(yaw, 4))}, world point {fmt(point)} (yaw {label})"
        got = np.asarray(call(fn, (*CAR_AT, yaw), point.copy(), given=given), dtype=float)
        if got.shape != (2,) or not np.allclose(got, (AHEAD, LEFT), atol=TOL):
            raise Failure(f"wrong at yaw {label} ({meaning})", given=given, expected=(AHEAD, LEFT), got=got)
    rng, pose = np.random.default_rng(1103), (31.0, -12.0, 2.2)
    cloud = rng.uniform(-200, 200, (146, 4, 2))
    ego = np.asarray(call(fn, pose, cloud.copy(), given="146 quads of 4 corners at one pose"), dtype=float)
    if ego.shape != cloud.shape:
        raise Failure(f"returned shape {ego.shape}, expected {cloud.shape}: one answer per point, whatever the leading shape",
                      given="a (146, 4, 2) array of points")
    back = float(np.abs(geom.ego_to_world(pose, ego) - cloud).max())
    if back > TOL:
        raise Failure(f"going to the car's frame and back moves points by up to {back:.3g} m", expected=f"at most {TOL:g} m",
                      given=f"pose {pose} and a (146, 4, 2) array of points")
    worst = renderer_gap(fn)
    if worst > TOL:
        raise Failure(f"differs from the renderer's transform by up to {worst:.3g} m", expected=f"at most {TOL:g} m",
                      given=f"{N_POSES} seeded poses, yaws up to ±4π")
    return f"all four yaws, the shape and the round trip pass; largest difference from the renderer's transform {worst:.2g} m over {N_POSES} poses"


@exercise(
    "1.1.2.b",
    checked=f"your world_to_ego inside the simulator's camera: the provided car must complete `{GATE_SCENARIO}` on the pictures it draws",
    section="Swap your transform into the renderer",
)
def _renderer_gate(fn, show, on_step):
    return gate(Car(), show, on_step, camera=Camera(world_to_ego=fn))
