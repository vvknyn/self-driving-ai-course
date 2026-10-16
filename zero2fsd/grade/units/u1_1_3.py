"""1.1.3 The road's own coordinates: to_road_frame and lane_errors, checked against the simulator's own `Road`."""
from __future__ import annotations

import numpy as np

from ...sim import Road, make_scenario
from ...sim.world import wrap_angle
from .._report import Failure, call, exercise, fmt, is_real

LESSON = ("1.1.3", "The road's own coordinates")  # (unit id, title): grade/__init__ files it in LESSON_TITLES
ROADS, N_PER_ROAD, TOL = ("gentle", "curvy"), 40, 1e-9
L_ROAD = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])  # two segments, east then north: the lecture's corner
NAMED_POINTS = {"beyond the open end": (12.0, 13.0), "inside the corner": (8.0, 3.0), "right of the line": (5.0, -1.0), "on the line": (5.0, 0.0)}
TIE_POSE = (12.0, -2.0, 0.3)  # outside the corner: its nearest point is the shared vertex (10, 0), the foot of both segments


def centerline_of(road: Road) -> np.ndarray:
    """The (N, 2) vertices a learner's function is given: a closed road's N starts, an open road's starts and its last end."""
    return road._a if road.closed else np.vstack([road._a, road._b[-1]])


def tie_headings(road: Road, x: float, y: float) -> np.ndarray:
    """The headings of every segment whose foot is as near to (x, y) as the nearest one (within TOL): two, at a vertex tie."""
    rel = np.array([x, y]) - road._a
    foot = rel - np.clip(np.einsum("ij,ij->i", rel, road._u), 0.0, road._seg_len)[:, None] * road._u
    dist = np.hypot(foot[:, 0], foot[:, 1])
    return road._heading[dist <= dist.min() + TOL]


def psi_misfit(road: Road, pose, psi: float) -> float:
    """How far `psi` is from the truth in rad; at a vertex tie the truth is any of the tied segments' headings."""
    x, y, yaw = pose
    return min(abs(psi - float(wrap_angle(yaw - h))) for h in tie_headings(road, x, y))


def mistakes_a(road: Road, p) -> dict[str, tuple[float, float]]:
    """The (s, d) each mistake the lesson names would answer at `p`; a mistake that happens to be right here is not told apart."""
    rel = np.array(p) - road._a
    along, cross = np.einsum("ij,ij->i", rel, road._u), road._u[:, 0] * rel[:, 1] - road._u[:, 1] * rel[:, 0]
    i = int(np.argmin(np.abs(cross)))  # no clamp: the nearest infinite line, whose foot may lie past the segment's end
    s_line = road._s0[i] + along[i]
    s, d = road.project(*p)
    verts = centerline_of(road)
    j = int(np.argmin(np.hypot(*(verts - p).T)))  # the nearest vertex, not the nearest segment
    s_vert = float(np.append(road._s0, road.length)[j])
    return {
        "forgetting to clamp t to the segment (the foot lands on its extension)": (float(s_line % road.length if road.closed else s_line), float(cross[i])),
        "the sign of d the wrong way round (left should be positive)": (s, -d),
        "the nearest vertex instead of the nearest point of a segment": (s_vert, float(np.copysign(np.hypot(*(verts[j] - p)), d))),
        "d as y minus the y of the nearest vertex": (s_vert, float(p[1] - verts[j][1])),
    }


def mistakes_b(road: Road, pose, e: float, psi: float) -> list[str]:
    """Every mistake the lesson names whose answer is this (e, psi) at `pose`."""
    x, y, yaw = pose
    _, want_e, want_psi = road.ego_truth(x, y, yaw)
    found = []
    if abs(e - (want_e - 1.8)) <= TOL:
        found.append("e = d, without adding the 1.8 m from the yellow line to the lane centre")
    if abs(e + want_e) <= TOL and want_e:
        found.append("the sign of e the wrong way round (left should be positive)")
    if any(abs(psi - (yaw - h)) <= TOL for h in tie_headings(road, x, y)) and abs(psi - want_psi) > TOL:
        found.append("psi = theta - theta_road with no wrap, so a whole turn of yaw is left in")
    if any(abs(psi + float(wrap_angle(yaw - h))) <= TOL for h in tie_headings(road, x, y)) and want_psi:
        found.append("psi the wrong way round (theta_road - theta)")
    return found


def _pair(result, given: str):
    """(first, second) of a function's answer as floats; anything else is a Failure."""
    try:
        first, second = result
    except (TypeError, ValueError):
        raise Failure(f"returned {fmt(result)}, expected two numbers", given=given) from None
    if not (is_real(first) and is_real(second)):
        raise Failure(f"returned {fmt(result)}, expected two finite numbers", given=given)
    return float(first), float(second)


def _seeded(road: Road, rng) -> tuple[np.ndarray, np.ndarray]:
    """(arc lengths, world points): N_PER_ROAD points scattered over the road's width, from the white line to past the yellow."""
    s = rng.uniform(0.0, road.length, N_PER_ROAD)
    return s, road.offset(s, rng.uniform(-4.0, 1.0, N_PER_ROAD))


def _seeded_poses(road: Road, rng) -> list[tuple[float, float, float]]:
    """N_PER_ROAD (x, y, yaw) on the road, each yaw off the road's direction by up to 0.3 rad plus 0 or a whole turn either way."""
    s, points = _seeded(road, rng)
    yaw = road.pose_at(s)[2] + rng.uniform(-0.3, 0.3, N_PER_ROAD) + 2 * np.pi * (np.arange(N_PER_ROAD) % 3 - 1)
    return [(float(x), float(y), float(h)) for (x, y), h in zip(points, yaw)]


@exercise(
    "1.1.3.a",
    checked=f"s and d against the simulator's `Road.project`, within {TOL:g}: four named cases on a small open L-shaped road, "
            f"then {N_PER_ROAD} seeded points on each of `{ROADS[0]}` and `{ROADS[1]}`",
    section="Which segment, and which side",
)
def _to_road_frame(fn, show, on_step):
    cases = [(f"the L-shaped road, {name}", L_ROAD, False, Road(L_ROAD, False), p) for name, p in NAMED_POINTS.items()]
    for name in ROADS:
        road, rng = make_scenario(name).road, np.random.default_rng(113)
        cases += [(f"`{name}`", centerline_of(road), road.closed, road, tuple(map(float, p))) for p in _seeded(road, rng)[1]]
    for what, line, closed, road, p in cases:
        given = f"{what}: the road's vertices, closed={closed}, p = {fmt(p)}"
        s, d = _pair(call(fn, line, closed, p, given=given), given)
        want_s, want_d = road.project(*p)
        if abs(s - want_s) > TOL or abs(d - want_d) > TOL:
            named = next((m for m, (ms, md) in mistakes_a(road, p).items() if abs(s - ms) <= TOL and abs(d - md) <= TOL), None)
            raise Failure("s or d does not match the simulator" + (f"; this is the answer for {named}" if named else
                          ": find the nearest point of every segment, clamped to the segment, and keep the closest"), given=given,
                          expected=f"s {want_s:.6f} m, d {want_d:+.6f} m", got=f"s {s:.6f} m, d {d:+.6f} m")
    return f"{len(cases)} points match the simulator's own s and d within {TOL:g}"


@exercise(
    "1.1.3.b",
    checked=f"e and psi against the simulator's `Road.ego_truth`, within {TOL:g}: a pose at a vertex tie of the L-shaped road, "
            f"then {N_PER_ROAD} seeded poses on each of `{ROADS[0]}` and `{ROADS[1]}`, with yaw up to a whole turn out",
    section="The two errors the controller needs",
)
def _lane_errors(fn, show, on_step):
    cases = [("the L-shaped road at a vertex tie", L_ROAD, False, Road(L_ROAD, False), TIE_POSE)]
    for name in ROADS:
        road, rng = make_scenario(name).road, np.random.default_rng(114)
        cases += [(f"`{name}`", centerline_of(road), road.closed, road, pose) for pose in _seeded_poses(road, rng)]
    for what, line, closed, road, pose in cases:
        given = f"{what}: the road's vertices, closed={closed}, pose (x, y, theta) = {fmt(pose)}"
        e, psi = _pair(call(fn, line, closed, pose, given=given), given)
        _, want_e, want_psi = road.ego_truth(*pose)
        e_bad, psi_bad = abs(e - want_e) > TOL, psi_misfit(road, pose, psi) > TOL
        if e_bad or psi_bad:
            named = mistakes_b(road, pose, e, psi)
            hint = f"; this is the answer for {' and '.join(named)}" if named else (
                ": e is d + 1.8, d from the nearest segment" if e_bad else ": psi is theta minus the nearest segment's direction, wrapped")
            raise Failure(("e" if e_bad else "psi") + " does not match the simulator" + hint, given=given,
                          expected=f"e {want_e:+.6f} m, psi {want_psi:+.6f} rad", got=f"e {e:+.6f} m, psi {psi:+.6f} rad")
    return f"{len(cases)} poses match the simulator's own e and psi within {TOL:g}"
