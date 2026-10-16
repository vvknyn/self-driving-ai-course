"""Every number in the lecture "The road's own coordinates" (1.1.3) comes from a line this script prints.

Run (from anywhere):
  PYTHONPATH=<course tree> PYTHONPYCACHEPREFIX=<scratch> MPLBACKEND=Agg <venv python> measure.py > measure.out
It also writes measured.json next to itself.  The functions this unit measures (wrap_angle, world_to_ego,
project_to_segment, to_road_frame, lane_errors) are the course's own, imported from zero2fsd.geom.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

import zero2fsd
from zero2fsd.car import Car, run
from zero2fsd.car.runner import OFF_ROAD_OFFSET
from zero2fsd.geom import lane_errors, project_to_segment, to_road_frame, world_to_ego, wrap_angle
from zero2fsd.grade._poses import sample_poses
from zero2fsd.sim import Road, make_scenario
from zero2fsd.sim.dynamics import DT
from zero2fsd.sim.scenarios import CENTERLINE_STEP, GENTLE_RADIUS, TARGET_SPEED
from zero2fsd.sim.world import EGO_LANE_CENTER, LANE_WIDTH, WHITE_LINE_LATERAL
from zero2fsd.sim.world import wrap_angle as sim_wrap

HERE = Path(__file__).parent
LANE_OFFSET = -EGO_LANE_CENTER  # 1.8 m: yellow line to lane centre, e = d + LANE_OFFSET
_GENTLE, _CURVY = make_scenario("gentle").road, make_scenario("curvy").road
FIRST_STRAIGHT = float(_GENTLE._s0[np.argmax(np.abs(_GENTLE._heading) > 1e-9)])  # 100 m: where gentle's first bend starts
_TURNS = np.abs(np.diff(_CURVY._heading))
CURVY_CORNER = float(np.median(_TURNS[_TURNS > 1e-9]))  # rad: the full-length segments' turn, 0.1 m / the curvy bend radius
CURVY_R = CENTERLINE_STEP / CURVY_CORNER
MEASURED: dict = {}


def out(tag: str, text: str, **values) -> None:
    print(f"[{tag}] {text}")
    MEASURED.update({f"{tag.lower()}.{k}": v for k, v in values.items()})


# ---------------------------------------------------------------- local helpers (not geom seams)
def _segments(centerline, closed):
    pts = np.asarray(centerline, dtype=float)
    a = pts if closed else pts[:-1]
    b = np.roll(pts, -1, axis=0) if closed else pts[1:]
    seg = np.hypot(*(b - a).T)
    return a, b, seg, np.concatenate([[0.0], np.cumsum(seg)[:-1]])


def _nearest_index(centerline, closed, p):
    """Index of the nearest segment to p (first one on an exact tie, as world.py:77), by geom.project_to_segment."""
    a, b, _, _ = _segments(centerline, closed)
    _, d = project_to_segment(a, b, p)
    return int(np.argmin(d * d))


# ---- two other correct-looking learner versions, to measure the vertex-tie risk
def lane_errors_footpoint(centerline, closed, pose):
    """Foot as a point a + t u, distance hypot(p - foot), argmin of the distance (not its square)."""
    a, b, seg, _ = _segments(centerline, closed)
    u = (b - a) / seg[:, None]
    p = np.asarray(pose[:2], dtype=float)
    t = np.clip(((p - a) * u).sum(axis=1), 0.0, seg)
    foot = a + t[:, None] * u
    dist = np.hypot(*(p - foot).T)
    i = int(np.argmin(dist))
    side = u[i, 0] * (p - a)[i, 1] - u[i, 1] * (p - a)[i, 0]
    return math.copysign(dist[i], side) + LANE_OFFSET, float(wrap_angle(pose[2] - math.atan2(u[i, 1], u[i, 0])))


def lane_errors_ego(centerline, closed, pose):
    """Each segment seen from its own start, facing its end: forward and left parts (the 1.1.1 idea)."""
    a, b, seg, _ = _segments(centerline, closed)
    th = np.arctan2(b[:, 1] - a[:, 1], b[:, 0] - a[:, 0])
    rel = np.asarray(pose[:2], dtype=float) - a
    fwd = rel[:, 0] * np.cos(th) + rel[:, 1] * np.sin(th)
    left = -rel[:, 0] * np.sin(th) + rel[:, 1] * np.cos(th)
    t = np.clip(fwd, 0.0, seg)
    dist = np.hypot(fwd - t, left)
    i = int(np.argmin(dist))
    return math.copysign(dist[i], left[i]) + LANE_OFFSET, float(wrap_angle(pose[2] - th[i]))


def check_file():
    out("ENV", f"zero2fsd from {zero2fsd.__file__}")
    out("ENV", f"LANE_WIDTH {LANE_WIDTH} m, EGO_LANE_CENTER {EGO_LANE_CENTER} m, WHITE_LINE_LATERAL {WHITE_LINE_LATERAL} m, "
               f"OFF_ROAD_OFFSET {OFF_ROAD_OFFSET} m, CENTERLINE_STEP {CENTERLINE_STEP} m, GENTLE_RADIUS {GENTLE_RADIUS} m, DT {DT} s")
    out("ENV", f"geom wrap_angle bit-identical to sim on 10001 angles in [-20, 20]: "
               f"{bool(np.array_equal(wrap_angle(np.linspace(-20, 20, 10001)), sim_wrap(np.linspace(-20, 20, 10001))))}")


def road_facts():
    for name in ("gentle", "curvy"):
        sc = make_scenario(name)
        road = sc.road
        n = len(road._a)
        out("ROAD", f"{name}: closed {road.closed}, {n + (0 if road.closed else 1)} vertices, {n} segments, "
                    f"length {road.length:.2f} m, segment length min {road._seg_len.min():.6f} max {road._seg_len.max():.6f} m, "
                    f"route_length {sc.route_length:.2f} m",
            **{f"{name}_segments": n, f"{name}_length": road.length})
    road = make_scenario("gentle").road
    r = GENTLE_RADIUS
    edges = np.cumsum([0, FIRST_STRAIGHT, math.pi * r, FIRST_STRAIGHT, math.pi * r])
    out("ROAD", "gentle pieces (s ranges, m): first straight " + f"{edges[0]:.2f}-{edges[1]:.2f}, first bend {edges[1]:.2f}-{edges[2]:.2f}, "
                f"far straight {edges[2]:.2f}-{edges[3]:.2f}, second bend {edges[3]:.2f}-{edges[4]:.2f}")
    for s in (0.0, FIRST_STRAIGHT, edges[1] + math.pi * r / 2, edges[2], edges[3]):
        x, y, h = road.pose_at(s)
        out("ROAD", f"gentle centreline at s {s:.2f}: ({x:+.2f}, {y:+.2f}) heading {h:+.4f} rad")
    out("ROAD", f"gentle corner angle between neighbouring segments in a bend: {CENTERLINE_STEP / r:.5f} rad (0.1/80); "
                f"curvy {CURVY_CORNER:.5f} rad (0.1/{CURVY_R:.0f})")
    out("ROAD", f"gentle lane-centre bend radius {r + LANE_OFFSET:.1f} m, yellow line {r:.0f} m, white line {r + LANE_WIDTH:.1f} m")


def provided_run():
    tel = run(Car(), "gentle", seed=0)
    k = len(tel.t)
    out("RUN", f"provided car, gentle, seed 0, 8 m/s: {k} steps, t 0-{tel.t[-1]:.2f} s, completed {tel.completed}, "
               f"mean|e| {np.abs(tel.lat).mean():.3f} m, max e {tel.lat.max():+.3f} m, min e {tel.lat.min():+.3f} m, "
               f"max|head_err| {np.abs(tel.head_err).max():.4f} rad, final progress {tel.progress_m[-1]:.2f} m of {tel.route_length:.2f} m",
        steps=k, mean_abs_e=float(np.abs(tel.lat).mean()), max_e=float(tel.lat.max()))
    out("RUN", f"start pose ({tel.x[0]:+.3f}, {tel.y[0]:+.3f}, {tel.yaw[0]:+.3f}); final yaw {tel.yaw[-1]:.3f} rad (never wrapped); "
               f"wrap(final yaw) {sim_wrap(tel.yaw[-1]):+.3f} rad", final_yaw=float(tel.yaw[-1]))
    return tel


def naive_lateral(tel, road):
    e = tel.lat
    naive = tel.y + LANE_OFFSET
    diff = np.abs(naive - e)
    s = np.array([road.project(x, y)[0] for x, y in zip(tel.x, tel.y)])
    first = s < FIRST_STRAIGHT
    first &= np.arange(len(s)) < np.argmax(s > FIRST_STRAIGHT)
    out("NAIVE", f"e_naive = y + 1.8 vs true e on the first straight (s < 100 m, t 0-{tel.t[first][-1]:.2f} s): "
                 f"max |diff| {diff[first].max():.2e} m", first_max=float(diff[first].max()))
    for thr in (0.01, 0.1, 1.0):
        k = int(np.argmax(diff > thr))
        out("NAIVE", f"first step with |e_naive - e| > {thr} m: t {tel.t[k]:.2f} s, s {s[k]:.1f} m (bend starts at 100 m)")
    k = int(np.argmax(diff))
    out("NAIVE", f"lap max |e_naive - e| {diff.max():.1f} m at t {tel.t[k]:.2f} s, s {s[k]:.1f} m: e_naive {naive[k]:+.1f} m, e {e[k]:+.3f} m",
        lap_max=float(diff.max()))
    k = int(np.argmin(np.abs(tel.t - 45.0)))
    out("NAIVE", f"far straight sample t {tel.t[k]:.2f} s, s {s[k]:.1f} m: y {tel.y[k]:+.2f} m, e_naive {naive[k]:+.2f} m, e {e[k]:+.3f} m")
    # the planner's subtler naive: y minus the y of the nearest lane-centre point
    lane_y = np.array([road.offset(si, EGO_LANE_CENTER)[1] for si in s])
    naive2 = tel.y - lane_y
    diff2 = np.abs(naive2 - e)
    k2 = int(np.argmax(diff2))
    out("NAIVE", f"y - y(lane centre at the same s): lap max |diff| {diff2.max():.3f} m at t {tel.t[k2]:.2f} s, s {s[k2]:.1f} m, "
                 f"road heading {road._locate(tel.x[k2], tel.y[k2])[2]:+.4f} rad, e {e[k2]:+.3f} m, value {naive2[k2]:+.3f} m (2|e| = {2 * abs(e[k2]):.3f})",
        y_minus_lane_max=float(diff2.max()))
    return s


def hand_examples():
    a, b, p = np.array([2.0, 1.0]), np.array([10.0, 7.0]), np.array([5.0, 7.0])
    ab = b - a
    L = math.hypot(*ab)
    u = ab / L
    rel = p - a
    fwd = rel @ u
    left = u[0] * rel[1] - u[1] * rel[0]
    foot = a + fwd * u
    t, d = project_to_segment(a, b, p)
    out("HAND", f"a (2, 1), b (10, 7): b - a ({ab[0]:.0f}, {ab[1]:.0f}), |b - a| {L:.1f}, u ({u[0]:.1f}, {u[1]:.1f}); "
                f"p (5, 7): p - a ({rel[0]:.0f}, {rel[1]:.0f}); forward part {fwd:.1f}, left part {left:+.1f}; "
                f"foot a + t u = ({foot[0]:.1f}, {foot[1]:.1f}); |p - foot| {math.hypot(*(p - foot)):.1f}")
    out("HAND", f"project_to_segment(a, b, p) = (t {t:.4f}, d {d:+.4f}); world_to_ego((2, 1, atan2(6, 8)), p) = "
                f"{np.round(world_to_ego((2.0, 1.0, math.atan2(6, 8)), p), 4).tolist()}; atan2(6, 8) {math.atan2(6, 8):.4f} rad")
    q = np.array([14.0, 5.0])
    rel = q - a
    fwd = rel @ u
    left = u[0] * rel[1] - u[1] * rel[0]
    foot_raw = a + fwd * u
    t, d = project_to_segment(a, b, q)
    out("CLAMP", f"q (14, 5): q - a ({rel[0]:.0f}, {rel[1]:.0f}); forward part {fwd:.1f} > |b - a| {L:.1f}; "
                 f"unclamped foot ({foot_raw[0]:.1f}, {foot_raw[1]:.1f}) is off the segment, |q - unclamped foot| {math.hypot(*(q - foot_raw)):.1f}, left part {left:+.1f}")
    out("CLAMP", f"clamped t {t:.1f} -> foot = b (10, 7); |q - b| = sqrt({(q - b) @ (q - b):.0f}) = {abs(d):.3f}; "
                 f"project_to_segment(a, b, q) = (t {t:.4f}, d {d:+.4f})")


def l_corner():
    line = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    road = Road(line, closed=False)
    cases = [("inside the corner", (8.0, 3.0)), ("outside the corner", (12.0, -2.0)), ("right of the line", (5.0, -1.0)),
             ("on the line", (5.0, 0.0)), ("beyond the open end", (12.0, 13.0)), ("before the open start", (-3.0, 1.0))]
    for label, p in cases:
        t1, d1 = project_to_segment(line[0], line[1], p)
        t2, d2 = project_to_segment(line[1], line[2], p)
        s, d = to_road_frame(line, False, p)
        ts, td = road.project(*p)
        out("LCORNER", f"{label} p ({p[0]:g}, {p[1]:g}): segment 1 (t {t1:.3f}, d {d1:+.3f}), segment 2 (t {t2:.3f}, d {d2:+.3f}); "
                       f"to_road_frame (s {s:.3f}, d {d:+.3f}); Road.project (s {ts:.3f}, d {td:+.3f}); equal {s == ts and d == td}")
    p = (10.0, 13.0)
    out("LCORNER", f"trap: p (10, 13) sits on the line of segment 2 past its end: cross 0, so copysign gives d {to_road_frame(line, False, p)[1]:+.3f} "
                   f"(Road.project {road.project(*p)[1]:+.3f})")


def worked_step(tel, road, k):
    x, y, th = float(tel.x[k]), float(tel.y[k]), float(tel.yaw[k])
    i = _nearest_index(road_pts(road), True, (x, y))
    s, d = to_road_frame(road_pts(road), True, (x, y))
    a, b = road._a[i], road._b[i]
    t, _ = project_to_segment(a, b, (x, y))
    th_road = math.atan2(b[1] - a[1], b[0] - a[0])
    e, psi = lane_errors(road_pts(road), True, (x, y, th))
    foot = a + t * (b - a) / math.hypot(*(b - a))
    out("CAR", f"step {k}, t {tel.t[k]:.2f} s: car (x {x:.3f}, y {y:.3f}, theta {th:.4f} rad); nearest segment {i} of {len(road._a)}, "
               f"a ({a[0]:.3f}, {a[1]:.3f}), s0 {road._s0[i]:.3f} m, t {t:.4f} m; foot ({foot[0]:.3f}, {foot[1]:.3f})",
        k=k, s=s, d=d, e=e, psi=psi, theta=th, theta_road=th_road)
    out("CAR", f"step {k}: s {s:.3f} m, d {d:+.4f} m, e = d + 1.8 = {e:+.4f} m (telemetry lat {tel.lat[k]:+.4f}); "
               f"theta_road {th_road:.4f} rad, psi = wrap(theta - theta_road) = {psi:+.4f} rad (telemetry head_err {tel.head_err[k]:+.4f})")
    ego = world_to_ego((foot[0], foot[1], th_road), (x, y))
    out("CAR", f"step {k}: the car seen from the road's own pose at the foot (foot, theta_road): world_to_ego -> "
               f"(forward {ego[0]:+.2e}, left {ego[1]:+.4f}) = (0, d)")
    out("CAR", f"step {k}: progress_m {tel.progress_m[k]:.3f} m (s from start {s:.3f}); distance from bend centre (100, 80) "
               f"{math.hypot(x - 100, y - 80):.4f} m = 80 - d = {80 - d:.4f}")
    return k


_PTS: dict = {}


def road_pts(road):
    """The road's own vertices, rebuilt from Road internals (closed: a; open: a + last b)."""
    key = id(road)
    if key not in _PTS:
        _PTS[key] = road._a.copy() if road.closed else np.vstack([road._a, road._b[-1:]])
    return _PTS[key]


def reproduce(tel, road):
    pts = road_pts(road)
    t0 = time.perf_counter()
    mine = np.array([(*to_road_frame(pts, True, (x, y)), *lane_errors(pts, True, (x, y, th)))
                     for x, y, th in zip(tel.x, tel.y, tel.yaw)])
    dt = time.perf_counter() - t0
    truth = np.array([road.project(x, y) for x, y in zip(tel.x, tel.y)])
    ds, dd = np.abs(mine[:, 0] - truth[:, 0]).max(), np.abs(mine[:, 1] - truth[:, 1]).max()
    de, dpsi = np.abs(mine[:, 2] - tel.lat).max(), np.abs(mine[:, 3] - tel.head_err).max()
    out("REPRO", f"to_road_frame + lane_errors on all {len(tel.t)} steps: max |s - truth| {ds:.1e} m, max |d - truth| {dd:.1e} m, "
                 f"max |e - lat| {de:.1e} m, max |psi - head_err| {dpsi:.1e} rad; time {dt:.2f} s",
        max_ds=float(ds), max_dd=float(dd), max_de=float(de), max_dpsi=float(dpsi), lap_seconds=dt)
    # what a per-segment python loop would cost; best of 5 runs, because the machine may be busy
    a, b = road._a, road._b
    p400 = (tel.x[400], tel.y[400])
    loops, bcasts = [], []
    for _ in range(5):
        t0 = time.perf_counter()
        for j in range(len(a)):
            project_to_segment(a[j], b[j], p400)
        loops.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        for _ in range(100):
            to_road_frame(pts, True, p400)
        bcasts.append((time.perf_counter() - t0) / 100)
    one, bc = min(loops), min(bcasts)
    out("REPRO", f"one position with a python loop over {len(a)} segments (best of 5): {one:.3f} s; times {len(tel.t)} steps = {one * len(tel.t):.0f} s; "
                 f"to_road_frame broadcast over all segments (best of 5): {bc * 1e3:.2f} ms per position; loop / broadcast {one / bc:.0f}x",
        loop_one=one, broadcast_one=bc)
    return mine


def ties(tel, road):
    pts = road_pts(road)
    a, b, seg = road._a, road._b, road._seg_len
    u = road._u
    clamped = 0
    for x, y in zip(tel.x, tel.y):
        i = road._locate(x, y)
        rel = np.array([x, y]) - a
        tt = np.einsum("ij,ij->i", rel, u)
        j = int(np.argmin(((rel - np.clip(tt, 0, seg)[:, None] * u) ** 2).sum(1)))
        clamped += tt[j] <= 0.0 or tt[j] >= seg[j]
    out("TIE", f"lap steps whose nearest foot is a vertex (clamped): {clamped} of {len(tel.t)} ({clamped / len(tel.t):.1%})",
        lap_clamped=int(clamped))
    variants = {"same arithmetic as the sim": lane_errors, "foot as a point, argmin of distance": lane_errors_footpoint,
                "forward/left parts from atan2": lane_errors_ego}
    for name, fn in variants.items():
        got = np.array([fn(pts, True, (x, y, th)) for x, y, th in zip(tel.x, tel.y, tel.yaw)])
        de, dpsi = np.abs(got[:, 0] - tel.lat), np.abs(got[:, 1] - tel.head_err)
        out("TIE", f"lap, {name}: max |e - lat| {de.max():.1e} m; steps with |psi - head_err| > 1e-9: {(dpsi > 1e-9).sum()}, "
                   f"largest {dpsi.max():.5f} rad")
    rng = np.random.default_rng(0)
    poses = sample_poses(rng, 2000, ["gentle", "curvy"], straight_only=False)
    for name, fn in variants.items():
        bad, bad_e = 0, 0.0
        for p in poses:
            pts_p = road_pts(p.road)
            got = fn(pts_p, p.road.closed, (p.state.x, p.state.y, p.state.yaw))
            want = p.road.ego_truth(p.state.x, p.state.y, p.state.yaw)
            bad += abs(got[1] - want[2]) > 1e-9
            bad_e = max(bad_e, abs(got[0] - want[1]))
        out("TIE", f"2000 grader-style poses (sample_poses seed 0, gentle+curvy), {name}: |psi| misses > 1e-9: {bad} ({bad / 2000:.1%}); "
                   f"max |e - truth| {bad_e:.1e} m")
    corner_g, corner_c = CENTERLINE_STEP / GENTLE_RADIUS, CURVY_CORNER
    out("TIE", f"a tie can only shift psi by one corner angle: gentle {corner_g:.5f} rad, curvy {corner_c:.5f} rad")


def wrap_trap(tel, road):
    heads = np.array([road._locate(x, y)[2] for x, y in zip(tel.x, tel.y)])
    raw = tel.yaw - heads
    err = np.abs(raw - tel.head_err)
    k = int(np.argmax(err > 1.0))
    out("WRAP", f"psi = theta - theta_road without wrap: wrong by > 1 rad on {(err > 1).sum()} of {len(err)} steps ({(err > 1).mean():.1%}); "
                f"max |error| {err.max():.4f} rad (2 pi = {2 * math.pi:.4f}); first wrong step t {tel.t[k]:.2f} s, "
                f"theta {tel.yaw[k]:.4f}, theta_road {heads[k]:+.4f}, raw {raw[k]:+.4f}, wrapped {tel.head_err[k]:+.4f}",
        share=float((err > 1).mean()), first_t=float(tel.t[k]))
    j = int(np.argmax(tel.t >= 65.0))
    out("WRAP", f"sample t {tel.t[j]:.2f} s: theta {tel.yaw[j]:.4f}, theta_road {heads[j]:+.4f}, raw {raw[j]:+.4f}, wrapped {tel.head_err[j]:+.4f}")
    jump = np.where(np.abs(np.diff(heads)) > math.pi)[0]
    out("WRAP", f"theta_road (atan2) jumps by about 2 pi at t {', '.join(f'{tel.t[i + 1]:.2f}' for i in jump)} s; "
                f"wrong-by-2pi steps run from t {tel.t[err > 1][0]:.2f} to {tel.t[err > 1][-1]:.2f} s")


def spacing(tel, road):
    x, y = np.asarray(tel.x), np.asarray(tel.y)
    d_true = np.array([road.project(a, b)[1] for a, b in zip(x, y)])
    for c in (1.0, 5.0, 10.0):
        pts = road.sample_points(c)
        sub = Road(pts, True)
        d_vertex = np.empty_like(d_true)
        for k, (a, b) in enumerate(zip(x, y)):
            rel = pts - (a, b)
            d_vertex[k] = math.sqrt(float(np.min((rel ** 2).sum(1))))
        d_seg = np.array([sub.project(a, b)[1] for a, b in zip(x, y)])
        ev, es = np.abs(d_vertex - np.abs(d_true)), np.abs(d_seg - d_true)
        kv = int(np.argmax(ev))
        out("SPACING", f"vertices every {c:g} m ({len(pts)} points): nearest-vertex |d| error max {ev.max():.3f} m "
                       f"(at t {tel.t[kv]:.2f} s, d {d_true[kv]:+.3f} m); segment-projection d error max {es.max():.4f} m; "
                       f"c^2/(8R) = {c * c / (8 * GENTLE_RADIUS):.4f} m",
            **{f"vertex_{c:g}": float(ev.max()), f"segment_{c:g}": float(es.max())})
    c = CENTERLINE_STEP
    out("SPACING", f"the sim's own polyline: vertices every {c} m, chord inside the true circle by at most c^2/(8R) = {c * c / (8 * GENTLE_RADIUS):.3e} m")
    # where on the far straight vs bends: nearest-vertex error vs |d| at 5 m spacing for a point mid-chord
    for dd in (0.5, 1.0, LANE_OFFSET, LANE_WIDTH):
        h = 2.5
        out("SPACING", f"straight 5 m spacing, point mid-way between vertices at |d| {dd}: nearest vertex sqrt(d^2 + 2.5^2) = "
                       f"{math.hypot(dd, h):.3f} m, error {math.hypot(dd, h) - dd:.3f} m")
    # the lane at 5 m spacing: draw-data for the figure (first bend)
    pts5 = road.sample_points(5.0)
    out("SPACING", f"5 m polyline: first bend vertices at s 100, 105, ...; chord mid-point sits {5 * 5 / (8 * GENTLE_RADIUS):.4f} m inside the 80 m circle; "
                   f"distance of chord mid (s 102.5) from bend centre "
                   f"{math.hypot(*(0.5 * (pts5[20] + pts5[21]) - (100, 80))):.4f} m")


def errors_number_line(tel):
    for d in (WHITE_LINE_LATERAL, EGO_LANE_CENTER, 0.0):
        out("ED", f"d {d:+.1f} m -> e = d + 1.8 = {d + LANE_OFFSET:+.1f} m")
    out("ED", f"off-road end of a run: |e| > {OFF_ROAD_OFFSET} m, i.e. d > 0 (over the yellow line) or d < -3.6 (over the white line)")


def arc_length(tel, road, s):
    i = int(np.searchsorted(road._s0, FIRST_STRAIGHT, side="right") - 1)
    out("ARC", f"segment {i} starts at s0 {road._s0[i]:.4f} m; s0 is the running total of segment lengths (0.1 m each); "
               f"s0 of segment 1000 = {road._s0[1000]:.4f} m")
    k = len(s) - 1
    out("ARC", f"last two steps: t {tel.t[k - 1]:.2f} s s {s[k - 1]:.3f} m, t {tel.t[k]:.2f} s s {s[k]:.3f} m; plain difference {s[k] - s[k - 1]:+.3f} m; "
               f"wrapped like an angle into [-L/2, L/2): {(s[k] - s[k - 1] + road.length / 2) % road.length - road.length / 2:+.4f} m; "
               f"one step at 8 m/s is {TARGET_SPEED * DT:.2f} m", end_s=float(s[k]))
    out("ARC", f"wrapping breaks only if one step moved more than L/2 = {road.length / 2:.1f} m ({road.length / 2 / DT:.0f} m/s)")
    out("ARC", f"final progress_m {tel.progress_m[-1]:.3f} m vs route length {tel.route_length:.3f} m")
    driven = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(tel.x), np.diff(tel.y)))])
    out("ARC", f"by step 400 (t {tel.t[400]:.2f} s) the car has driven {driven[400]:.3f} m; s {s[400]:.3f} m; whole lap driven {driven[-1]:.2f} m, "
               f"s runs 0-{road.length:.2f} m")
    lane = Road(road.sample_points(CENTERLINE_STEP, EGO_LANE_CENTER), True)
    out("ARC", f"the lane-centre loop is {lane.length:.2f} m long (200 + 2 pi 81.8 = {200 + 2 * math.pi * 81.8:.2f}); the centreline loop {road.length:.2f} m")


def challenge(road):
    s0, e, psi = 260.0, 0.3, 0.04
    pose = road.ego_pose(s0, e, psi)
    R = GENTLE_RADIUS + LANE_OFFSET
    ss = np.arange(s0 - 5, s0 + 60, 0.001)
    lane = road.offset(ss, EGO_LANE_CENTER)
    ego = world_to_ego(pose, lane)
    xs, ys = ego[:, 0], ego[:, 1]
    out("CHAL", f"pose at s {s0:.0f} m on gentle (first bend), e {e:+.1f} m, psi {psi:+.2f} rad: world ({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.4f}); "
                f"lane-centre radius R {R:.1f} m")
    for x in (0.0, 5.0, 10.0, 20.0, 30.0):
        true = float(np.interp(x, xs, ys))
        quad = -e - psi * x + x * x / (2 * R)
        lin = -e - psi * x
        out("CHAL", f"x {x:4.0f} m: true y {true:+.4f} m, quadratic {quad:+.4f} m (out by {abs(quad - true):.4f}), "
                    f"line {lin:+.4f} m (out by {abs(lin - true):.4f})", **{f"true_{x:g}": true, f"quad_{x:g}": quad})
    xg = np.arange(0.0, 50.0, 0.01)
    gap = np.abs(-e - psi * xg + xg ** 2 / (2 * R) - np.interp(xg, xs, ys))
    xq = float(xg[np.argmax(gap > 0.1)])
    out("CHAL", f"the quadratic is first 0.1 m out at x {xq:.2f} m", x_01=xq)
    gap1 = np.abs(-e - psi * xg - np.interp(xg, xs, ys))
    out("CHAL", f"the straight line is first 0.1 m out at x {float(xg[np.argmax(gap1 > 0.1)]):.2f} m")


def presets(tel, road):
    pts = road_pts(road)
    x, y, th = np.asarray(tel.x), np.asarray(tel.y), np.asarray(tel.yaw)
    truth = np.array([road.project(a, b) for a, b in zip(x, y)])
    heads = np.array([road._locate(a, b)[2] for a, b in zip(x, y)])
    # y_minus_lane: d = y - y_foot (world y difference instead of signed distance)
    feet = np.array([road.offset(s_, 0.0) for s_ in truth[:, 0]])
    d_y = y - feet[:, 1]
    out("PRESET", f"y_minus_lane (d = y - y of the nearest centreline point): largest |d - truth| over the lap {np.abs(d_y - truth[:, 1]).max():.3f} m, "
                  f"largest |s - truth| 0.000 m")
    # nearest_vertex: s of the nearest vertex, d = +-distance to it
    s_v, d_v = np.empty(len(x)), np.empty(len(x))
    s_vert = road._s0
    for k, (a, b) in enumerate(zip(x, y)):
        rel = pts - (a, b)
        j = int(np.argmin((rel ** 2).sum(1)))
        s_v[k] = s_vert[j]
        side = road._u[j, 0] * (-rel[j, 1]) - road._u[j, 1] * (-rel[j, 0])
        d_v[k] = math.copysign(math.hypot(*rel[j]), side)
    out("PRESET", f"nearest_vertex on the sim's 0.1 m centreline: largest |d - truth| {np.abs(d_v - truth[:, 1]).max():.2e} m, "
                  f"largest |s - truth| {np.abs(s_v - truth[:, 0]).max():.4f} m")
    line = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    for p in ((8.0, 3.0), (5.0, -1.0), (5.0, 0.0), (12.0, 13.0)):
        rel = line - p
        j = int(np.argmin((rel ** 2).sum(1)))
        out("PRESET", f"nearest_vertex on the L-corner, p ({p[0]:g}, {p[1]:g}): nearest vertex ({line[j][0]:g}, {line[j][1]:g}) at distance "
                      f"{math.hypot(*rel[j]):.3f} m; to_road_frame |d| {abs(to_road_frame(line, False, p)[1]):.3f} m")
    raw = th - heads
    e = truth[:, 1] + LANE_OFFSET
    out("PRESET", f"no_wrap (psi = theta - theta_road): largest |e - lat| {np.abs(e - tel.lat).max():.1e} m, "
                  f"largest |psi - head_err| {np.abs(raw - tel.head_err).max():.4f} rad")


def extras(tel, road):
    """Facts added after the first draft; appended so earlier measure.out lines keep their numbers."""
    from zero2fsd.car.provided import FAR_ROWS, NEAR_ROWS, lane_point, paint_masks
    from zero2fsd.sim.camera import Camera
    from zero2fsd.sim.dynamics import BicycleState
    pts = road_pts(road)
    # the cleverer guess, measured literally: nearest point of the lane-centre polyline (not "same s")
    lane = road.offset(road._s0, EGO_LANE_CENTER)
    s_lane = np.array([to_road_frame(lane, True, (x, y))[0] for x, y in zip(tel.x, tel.y)])
    _, _, _, s0_lane = _segments(lane, True)
    near = np.array([lane[_nearest_index(lane, True, (x, y))] for x, y in zip(tel.x, tel.y)])
    foot_y = []
    for x, y in zip(tel.x, tel.y):
        i = _nearest_index(lane, True, (x, y))
        a, b, seg, _ = _segments(lane, True)
        t, _ = project_to_segment(a[i:i + 1], b[i:i + 1], np.array([x, y]))
        foot_y.append(a[i, 1] + t[0] * (b[i, 1] - a[i, 1]) / seg[i])
    guess = tel.y - np.array(foot_y)
    diff = np.abs(guess - tel.lat)
    first = np.arange(len(tel.t)) < int(np.argmax(tel.t > 12.5))
    k = int(np.argmax(diff))
    out("EXTRA", f"cleverer guess y - y(nearest point of the lane-centre polyline): first straight (t <= 12.50 s) max |diff| {diff[first].max():.2e} m; "
                 f"lap max |diff| {diff.max():.3f} m at t {tel.t[k]:.2f} s, guess {guess[k]:+.3f} m, e {tel.lat[k]:+.3f} m")
    # the box's two forward distances d1, d2 at the start of gentle (Your first perception box)
    sc = make_scenario("gentle")
    state = BicycleState(*road.ego_pose(sc.start_s), sc.target_speed, 0.0)
    masks = paint_masks(Camera().render(road, state))
    (f1, c1), (f2, c2) = (lane_point(masks, rows) for rows in (NEAR_ROWS, FAR_ROWS))
    out("EXTRA", f"provided box at the start of gentle (step 0): near lane-centre point forward d1 {f1:.2f} m (left {c1:+.3f}), "
                 f"far point forward d2 {f2:.2f} m (left {c2:+.3f})")
    # one vertex tie on the lap, for the figure
    a, b, seg, u = road._a, road._b, road._seg_len, road._u
    for k, (x, y) in enumerate(zip(tel.x, tel.y)):
        rel = np.array([x, y]) - a
        tt = np.einsum("ij,ij->i", rel, u)
        j = int(np.argmin(((rel - np.clip(tt, 0, seg)[:, None] * u) ** 2).sum(1)))
        if (tt[j] <= 0.0 or tt[j] >= seg[j]) and FIRST_STRAIGHT < road._s0[j] < FIRST_STRAIGHT + math.pi * GENTLE_RADIUS:
            other = (j - 1) % len(seg) if tt[j] <= 0.0 else (j + 1) % len(seg)
            h = road._heading
            out("EXTRA", f"first lap vertex tie on the first bend: step {k}, t {tel.t[k]:.2f} s, segments {min(j, other)} and {max(j, other)} share the vertex; "
                         f"sim keeps segment {j}; headings {h[min(j, other)]:.5f} and {h[max(j, other)]:.5f} rad, "
                         f"difference {abs(h[j] - h[other]):.5f} rad; d {tel.lat[k] + EGO_LANE_CENTER:+.4f} m")
            break
    out("EXTRA", "nearest-vertex error / projection error: " + ", ".join(
        f"{c} m {MEASURED[f'spacing.vertex_{c}'] / MEASURED[f'spacing.segment_{c}']:.1f}x" for c in ("1", "5", "10")))
    out("EXTRA", f"step 400 psi -0.0339 rad = {math.degrees(-0.0339):.2f} deg; one step at 8 m/s = {TARGET_SPEED * DT:.2f} m")


def main():
    check_file()
    road_facts()
    tel = provided_run()
    road = make_scenario("gentle").road
    s = naive_lateral(tel, road)
    hand_examples()
    l_corner()
    worked_step(tel, road, 400)
    reproduce(tel, road)
    ties(tel, road)
    wrap_trap(tel, road)
    spacing(tel, road)
    errors_number_line(tel)
    arc_length(tel, road, s)
    challenge(road)
    presets(tel, road)
    extras(tel, road)
    (HERE / "measured.json").write_text(json.dumps(MEASURED, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
