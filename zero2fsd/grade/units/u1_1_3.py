"""1.1.3 The road's own coordinates: to_road_frame and lane_errors, checked against the simulator's own `Road`."""
from __future__ import annotations

import numpy as np

from ...sim import Road, make_scenario
from ...sim.world import wrap_angle
from .._report import Failure, call, exercise, fmt, is_real

LESSON = ("1.1.3", "The road's own coordinates")  # (unit id, title): grade/__init__ files it in LESSON_TITLES
ROADS, N_PER_ROAD, TOL = ("gentle", "curvy"), 40, 1e-9
L_ROAD = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])  # two segments, east then north: the lecture's corner
NAMED_POINTS = {"past the open end": (12.0, 13.0), "on the line past the open end": (10.0, 13.0), "inside the corner": (8.0, 3.0),
                "right of the line": (5.0, -1.0), "on the line": (5.0, 0.0)}
NAMED_POSES = {"at a vertex tie": (12.0, -2.0, 0.3), "facing back along the road (yaw a half turn out)": (5.0, -1.0, np.pi)}
START_LINE = "gentle"  # the closed road whose start line the extra cases cross
START_LINE_LATERAL = (-1.8, 0.0)  # the lane centre and the yellow line, at each arc length in START_S_OFFSETS from the start line
START_S_OFFSETS = (-0.5, 0.0, 0.5)
SPARSE_SPACING = 5.0  # m: gentle re-sampled this coarsely has a long closing segment, which an open-road reading would miss


class _Segments:
    """What the checks need of a road, worked out from the vertices alone: starts a, directions u, lengths, arc length s0 at each start."""

    def __init__(self, line, closed: bool):
        pts = np.asarray(line, dtype=float)
        self.a, b = (pts, np.roll(pts, -1, axis=0)) if closed else (pts[:-1], pts[1:])
        vec = b - self.a
        self.length = np.hypot(vec[:, 0], vec[:, 1])
        self.u, self.heading = vec / self.length[:, None], np.arctan2(vec[:, 1], vec[:, 0])
        self.s0 = np.concatenate([[0.0], np.cumsum(self.length)])  # one longer than a: the last entry is the road's length
        self.closed = closed

    def feet(self, p):
        """(forward part t clamped to each segment, distance to its foot, left part): the three numbers per segment the lesson builds."""
        rel = np.asarray(p, dtype=float) - self.a
        t = np.clip(np.einsum("ij,ij->i", rel, self.u), 0.0, self.length)
        foot = rel - t[:, None] * self.u
        return t, np.hypot(foot[:, 0], foot[:, 1]), self.u[:, 0] * rel[:, 1] - self.u[:, 1] * rel[:, 0]

    def left_at_nearest(self, p) -> float:
        _, dist, left = self.feet(p)
        return float(left[int(np.argmin(dist))])

    def tie_headings(self, p) -> np.ndarray:
        dist = self.feet(p)[1]
        return self.heading[dist <= dist.min() + TOL]


def tie_headings(road: Road, x: float, y: float) -> np.ndarray:
    """The headings of every segment whose foot is as near to (x, y) as the nearest one (within TOL): two, at a vertex tie."""
    return _Segments(road.centerline, road.closed).tie_headings((x, y))


def _wrapped_truths(road: Road, pose) -> list[float]:
    """Every heading error the pose's nearest segments allow, in [-pi, pi); a truth of exactly -pi also allows +pi (`arctan2` gives that one)."""
    x, y, yaw = pose
    truths = [float(wrap_angle(yaw - h)) for h in tie_headings(road, x, y)]
    return truths + [np.pi for w in truths if abs(w + np.pi) <= TOL]


def psi_misfit(road: Road, pose, psi: float) -> float:
    """How far `psi` is from the truth in rad; at a vertex tie the truth is any of the tied segments' headings, and a turn out is a whole 2*pi off."""
    return min(abs(psi - w) for w in _wrapped_truths(road, pose))


def s_misfit(road: Road, s: float, want_s: float) -> float:
    """|s - want_s| in m; on a closed road the start line is s = 0 and s = length, and either is the same place."""
    gap = abs(s - want_s)
    return min(gap, road.length - gap) if road.closed else gap


def d_misfit(d: float, want_d: float, left: float) -> float:
    """|d - want_d| in m; a point exactly on its segment's line (left part 0) is neither left nor right, so either sign of the same distance is right."""
    return abs(abs(d) - abs(want_d)) if abs(left) <= TOL else abs(d - want_d)


def mistakes_a(road: Road, p) -> dict[str, tuple[float, float]]:
    """The (s, d) each mistake the lesson names would answer at `p`; a mistake that happens to be right here is not told apart."""
    seg = _Segments(road.centerline, road.closed)
    t, dist, left = seg.feet(p)
    s, d = road.project(*p)
    i = int(np.argmin(dist))
    rel = np.asarray(p, dtype=float) - seg.a
    along, cross = np.einsum("ij,ij->i", rel, seg.u), left
    k = int(np.argmin(np.abs(cross)))  # no clamp: the nearest infinite line, whose foot may lie past the segment's end
    s_line = float(seg.s0[k] + along[k])
    verts = road.centerline
    j = int(np.argmin(np.hypot(*(verts - p).T)))  # the nearest vertex, not the nearest segment
    s_vert = float(seg.s0[j])
    signed = np.where(left >= 0, dist, -dist)
    m = int(np.argmin(signed))  # the most negative d, not the smallest |d|
    open_s, open_d = Road(verts, False).project(*p) if road.closed else (s, d)
    return {
        "forgetting to clamp t to the segment (the foot lands on its extension)": (s_line % road.length if road.closed else s_line, float(cross[k])),
        "the sign of d the wrong way round (left should be positive)": (s, -d),
        "a closed road read as an open one, so the segment back to the first vertex is missing": (open_s, open_d),
        "the nearest vertex instead of the nearest point of a segment": (s_vert, float(np.copysign(np.hypot(*(verts[j] - p)), d))),
        "d as y minus the y of the nearest vertex": (s_vert, float(p[1] - verts[j][1])),
        "np.sign(left) * distance, which is 0 where the point is on the segment's line": (s, float(np.sign(left[i]) * dist[i])),
        "an s that starts the running total of lengths at the first segment's length, not at 0 (np.cumsum without a leading 0)": (float(seg.s0[i + 1] + t[i]), d),
        "the answer's two numbers the wrong way round (d first, then s)": (d, s),
        "the smallest signed d, where it should be the smallest |d|": (float(seg.s0[m] + t[m]), float(signed[m])),
    }


def mistakes_b(road: Road, pose, e: float, psi: float) -> list[str]:
    """Every mistake the lesson names whose answer is this (e, psi) at `pose`."""
    x, y, yaw = pose
    _, want_e, want_psi = road.ego_truth(x, y, yaw)
    raw = [yaw - h for h in tie_headings(road, x, y)]  # theta - theta_road before any wrap
    found = []
    if abs(e - (want_e - 1.8)) <= TOL:
        found.append("e = d, without adding the 1.8 m from the yellow line to the lane centre")
    if abs(e + want_e) <= TOL and want_e:
        found.append("the sign of e the wrong way round (left should be positive)")
    if any(abs(psi - r) <= TOL for r in raw) and psi_misfit(road, pose, psi) > TOL:
        found.append("psi = theta - theta_road with no wrap, so a whole turn of yaw is left in")
    if any(abs(psi - float(r % (2 * np.pi))) <= TOL for r in raw) and psi_misfit(road, pose, psi) > TOL:
        found.append("psi wrapped into [0, 2*pi), so a small turn to the right comes out near 2*pi")
    if any(abs(psi - float(np.degrees(w))) <= TOL for w in _wrapped_truths(road, pose)) and want_psi:
        found.append("psi in degrees, not radians")
    if any(abs(psi + w) <= TOL for w in _wrapped_truths(road, pose)) and want_psi:
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


def _scenes_a() -> list[tuple[str, Road, tuple[float, float]]]:
    """(what, road, p): the named cases on the L, the start line and a sparse copy of the closed road, then seeded points on each road."""
    gentle = make_scenario(START_LINE).road
    sparse = Road(gentle.sample_points(SPARSE_SPACING), True)
    scenes = [(f"the L-shaped road, {name}", Road(L_ROAD.copy(), False), p) for name, p in NAMED_POINTS.items()]
    scenes += [(f"`{START_LINE}` at {off:+.1f} m from the start line", gentle, tuple(map(float, gentle.offset(off, lat))))
               for off in START_S_OFFSETS for lat in START_LINE_LATERAL]
    scenes += [(f"`{START_LINE}` sampled every {SPARSE_SPACING:g} m", sparse, tuple(map(float, sparse.offset(s, -1.8))))
               for s in [*np.linspace(1.7, sparse.length - 1.0, 8), sparse.length - 0.3, 0.5]]
    for name in ROADS:
        road, rng = make_scenario(name).road, np.random.default_rng(113)
        scenes += [(f"`{name}`", road, tuple(map(float, p))) for p in _seeded(road, rng)[1]]
    return scenes


@exercise(
    "1.1.3.a",
    checked=f"s and d against the simulator's `Road.project`, within {TOL:g}: five named cases on a small open L-shaped road, "
            f"points either side of the start line and on a coarsely sampled closed road, then {N_PER_ROAD} seeded points on each of `{ROADS[0]}` and `{ROADS[1]}`",
    section="Which segment, and which side",
)
def _to_road_frame(fn, show, on_step):
    scenes = _scenes_a()
    for what, road, p in scenes:
        given = f"{what}: the road's vertices, closed={road.closed}, p = {fmt(p)}"
        s, d = _pair(call(fn, road.centerline, road.closed, p, given=given), given)  # a fresh copy each call: the learner may change it
        want_s, want_d = road.project(*p)
        left = _Segments(road.centerline, road.closed).left_at_nearest(p)
        if s_misfit(road, s, want_s) > TOL or d_misfit(d, want_d, left) > TOL:
            named = next((m for m, (ms, md) in mistakes_a(road, p).items() if s_misfit(road, s, ms) <= TOL and abs(d - md) <= TOL), None)
            s_gap, d_gap = s_misfit(road, s, want_s), d_misfit(d, want_d, left)
            hint = (f"; this is the answer for {named}" if named else
                    f": d matches but s is {s_gap:.6g} m out" if d_gap <= TOL else
                    f": s is {s_gap:.6g} m out and d is {d_gap:.6g} m out")
            raise Failure("s or d does not match the simulator" + hint, given=given,
                          expected=f"s {want_s:.6f} m, d {want_d:+.6f} m", got=f"s {s:.6f} m, d {d:+.6f} m")
    return f"{len(scenes)} points match the simulator's own s and d within {TOL:g}"


def _scenes_b() -> list[tuple[str, Road, tuple[float, float, float]]]:
    """(what, road, pose): the named poses on the L, then seeded poses on each road."""
    scenes = [(f"the L-shaped road {name}", Road(L_ROAD.copy(), False), pose) for name, pose in NAMED_POSES.items()]
    for name in ROADS:
        road, rng = make_scenario(name).road, np.random.default_rng(114)
        scenes += [(f"`{name}`", road, pose) for pose in _seeded_poses(road, rng)]
    return scenes


@exercise(
    "1.1.3.b",
    checked=f"e and psi against the simulator's `Road.ego_truth`, within {TOL:g}: two poses on the L-shaped road (one at a vertex tie, one with psi exactly a half turn), "
            f"then {N_PER_ROAD} seeded poses on each of `{ROADS[0]}` and `{ROADS[1]}`, with yaw up to a whole turn out",
    section="The two errors the controller needs",
)
def _lane_errors(fn, show, on_step):
    scenes = _scenes_b()
    for what, road, pose in scenes:
        given = f"{what}: the road's vertices, closed={road.closed}, pose (x, y, theta) = {fmt(pose)}"
        e, psi = _pair(call(fn, road.centerline, road.closed, pose, given=given), given)
        _, want_e, want_psi = road.ego_truth(*pose)
        e_bad, psi_bad = abs(e - want_e) > TOL, psi_misfit(road, pose, psi) > TOL
        if e_bad or psi_bad:
            named = mistakes_b(road, pose, e, psi)
            hint = f"; this is the answer for {' and '.join(named)}" if named else (
                f": e is {abs(e - want_e):.6g} m out" if not psi_bad else f": psi is {psi_misfit(road, pose, psi):.6g} rad out" if not e_bad else
                f": e is {abs(e - want_e):.6g} m out and psi is {psi_misfit(road, pose, psi):.6g} rad out")
            raise Failure(("e" if e_bad and not psi_bad else "psi" if psi_bad and not e_bad else "e and psi") + " does not match the simulator" + hint, given=given,
                          expected=f"e {want_e:+.6f} m, psi {want_psi:+.6f} rad", got=f"e {e:+.6f} m, psi {psi:+.6f} rad")
    return f"{len(scenes)} poses match the simulator's own e and psi within {TOL:g}"
