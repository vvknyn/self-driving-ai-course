"""Measurements behind the v2 lecture of *Positions and headings*.  Every number in lecture-v2.md traces to a [TAG] line.

Run: PYTHONPATH=<course tree> PYTHONPYCACHEPREFIX=... MPLBACKEND=Agg <venv>/python measure.py > measure.out
Set ZERO2FSD_REPO to the tree being built; the default is the read-only tree the lecture was written against.
Once zero2fsd/geom.py exists, its four 1.1.1 functions are measured instead of the reference copies below,
so a diff of the two measure.out files (all but the [geom] line) checks the shared code against the lecture.
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

import zero2fsd
from zero2fsd.car import Car, run
from zero2fsd.sim import BicycleState, make_scenario
from zero2fsd.sim import world
from zero2fsd.sim.camera import to_vehicle
from zero2fsd.sim.dynamics import DT, MAX_STEER, MAX_STEER_RATE, WHEELBASE
from zero2fsd.sim.scenarios import GENTLE_RADIUS, TARGET_SPEED
from zero2fsd.sim.world import EGO_LANE_CENTER, LANE_WIDTH, ROAD_HALF_WIDTH

REPO = Path(os.environ.get("ZERO2FSD_REPO", "/Users/vivek/Antigravity/self-driving-ai-course-0.1.2")).resolve()
assert Path(zero2fsd.__file__).resolve().is_relative_to(REPO), f"zero2fsd resolves to {zero2fsd.__file__}, not {REPO}"
HERE = Path(__file__).resolve().parent
J = {}


def out(tag, msg, **values):
    print(f"[{tag}] {msg}", flush=True)
    for k, v in values.items():
        J[f"{tag}.{k}"] = v


# ---- plan §3.1 contracts (reference copies; zero2fsd.geom replaces them once F0 lands) ----
def heading_vector(theta):
    """(..., 2): the 1 m arrow pointing at angle theta, (cos θ, sin θ)."""
    theta = np.asarray(theta, dtype=float)
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1)


def direction_angle(v):
    """atan2(v_y, v_x) in [-π, π]; nan for a zero vector, which has no direction."""
    v = np.asarray(v, dtype=float)
    a = np.arctan2(v[..., 1], v[..., 0])
    a = np.where((v[..., 0] == 0) & (v[..., 1] == 0), np.nan, a)
    return a[()] if a.ndim == 0 else a


def wrap_angle(a):
    """Into [-π, π); the same arithmetic as sim.world.wrap_angle."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def forward_left(theta, v):
    """(..., 2): (v·ĥ(θ), ĥ(θ)×v), the forward and left parts of the world vector v."""
    h = heading_vector(theta)
    v = np.asarray(v, dtype=float)
    return np.stack([v[..., 0] * h[..., 0] + v[..., 1] * h[..., 1], h[..., 0] * v[..., 1] - h[..., 1] * v[..., 0]], axis=-1)


try:
    from zero2fsd.geom import direction_angle, forward_left, heading_vector, wrap_angle  # noqa: F811
    GEOM = "zero2fsd.geom"
except ImportError:
    GEOM = "reference copies in measure.py (zero2fsd.geom not present)"


def naive_arctan(v):
    """The preset: the angle from rise over run, as in *Your first perception box*."""
    v = np.asarray(v, dtype=float)
    return np.arctan(v[..., 1] / v[..., 0])


def atan2_by_hand(v):
    """Challenge 2: atan2 from arctan and np.where."""
    v = np.asarray(v, dtype=float)
    x, y = v[..., 0], v[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        base = np.arctan(y / x)
    a = np.where(x > 0, base, np.where(y >= 0, base + math.pi, base - math.pi))
    a = np.where(x == 0, np.where(y > 0, math.pi / 2, np.where(y < 0, -math.pi / 2, np.nan)), a)
    return a


def f(x, n=4):
    return f"{x:+.{n}f}"


# ---------------------------------------------------------------------------------------------------------------
out("env", f"zero2fsd {zero2fsd.__file__}; numpy {np.__version__}; python {'.'.join(map(str, sys.version_info[:3]))}")
out("geom", f"functions measured: {GEOM}")
out("const", f"WHEELBASE {WHEELBASE} m; DT {DT} s; TARGET_SPEED {TARGET_SPEED} m/s; one step {TARGET_SPEED * DT:.2f} m; "
             f"MAX_STEER {MAX_STEER} rad; MAX_STEER_RATE {MAX_STEER_RATE} rad/s; GENTLE_RADIUS {GENTLE_RADIUS} m; "
             f"EGO_LANE_CENTER {EGO_LANE_CENTER} m; LANE_WIDTH {LANE_WIDTH} m; ROAD_HALF_WIDTH {ROAD_HALF_WIDTH} m; "
             f"2pi {2 * math.pi:.4f}; pi {math.pi:.5f}; pi/2 {math.pi / 2:.4f}")

# ---- the lap ----
sc = make_scenario("gentle")
road = sc.road
tel = run(Car(), "gentle")
N = len(tel.t)
P = np.stack([tel.x, tel.y], axis=-1)
D = np.diff(P, axis=0)                      # step k -> k+1 arrows, k = 0 .. N-2
yaw = tel.yaw
out("lap", f"gentle, provided car: {N} steps, last t = {tel.t[-1]:.2f} s, completed {tel.completed}; road length {road.length:.1f} m; "
           f"x from {tel.x.min():.3f} to {tel.x.max():.3f} m; y from {tel.y.min():.4f} to {tel.y.max():.4f} m; "
           f"steps (arrows) {len(D)}", steps=N, T=float(tel.t[-1]))
out("start", f"start pose x {tel.x[0]:.1f} y {tel.y[0]:.1f} yaw {yaw[0]:.1f}; road.ego_pose(0) = {road.ego_pose(0.0)}; "
             f"lane centre {EGO_LANE_CENTER} m = right of the yellow centre line")
# where the lap goes: first point the car reaches each extreme, and its yaw there
for name, k in (("east-most", int(tel.x.argmax())), ("north-most", int(tel.y.argmax())), ("west-most", int(tel.x.argmin()))):
    out("extreme", f"{name}: step {k}, t {tel.t[k]:.2f} s, x {tel.x[k]:.1f} y {tel.y[k]:.1f}, yaw {yaw[k]:.4f}, progress {tel.progress_m[k]:.1f} m")
L = np.hypot(D[:, 0], D[:, 1])
out("step", f"arrow length over the lap: min {L.min():.8f} max {L.max():.8f} mean {L.mean():.6f} m; v*DT = {TARGET_SPEED * DT:.2f} m; "
            f"speed min {tel.speed.min():.3f} max {tel.speed.max():.3f}")

# ---- step 700: the opening scene ----
K = 700
th = float(yaw[K])
dx, dy = float(D[K, 0]), float(D[K, 1])
out("s700", f"step {K}: t {tel.t[K]:.2f} s, x {tel.x[K]:.3f} y {tel.y[K]:.3f}; next x {tel.x[K + 1]:.3f} y {tel.y[K + 1]:.3f}; "
            f"progress {tel.progress_m[K]:.1f} m; e {tel.lat[K]:+.3f} m; psi {tel.head_err[K]:+.4f}; steer applied {tel.steer[K + 1]:+.4f}")
out("s700arrow", f"dx {dx:+.4f} dy {dy:+.4f} m; length {math.hypot(dx, dy):.4f} m; dx^2 {dx * dx:.4f} dy^2 {dy * dy:.4f} sum {dx * dx + dy * dy:.4f}")
out("s700yaw", f"yaw {th:.4f} rad = {math.degrees(th):.1f} deg; cos {math.cos(th):+.4f} sin {math.sin(th):+.4f}; "
               f"cos^2+sin^2 {math.cos(th) ** 2 + math.sin(th) ** 2:.12f}; heading_vector {np.round(heading_vector(th), 4)}")
slope = dy / dx
nv = float(naive_arctan([dx, dy]))
out("s700naive", f"slope dy/dx {slope:+.4f}; arctan(slope) {nv:+.4f} rad = {math.degrees(nv):+.1f} deg; arrow at that angle "
                 f"(cos {math.cos(nv):+.4f}, sin {math.sin(nv):+.4f}); miss vs yaw {abs(float(wrap_angle(nv - th))):.4f} rad")
a700 = float(direction_angle([dx, dy]))
out("s700atan2", f"direction_angle {a700:+.4f} rad = {math.degrees(a700):.1f} deg; minus yaw {a700 - th:+.4f} rad; "
                 f"wrap(direction - yaw) {float(wrap_angle(a700 - th)):+.4f}")

# ---- hand cases of the heading arrow ----
for name, t in (("0", 0.0), ("pi/2", math.pi / 2), ("pi", math.pi), ("-pi/2", -math.pi / 2)):
    hv = heading_vector(t)
    out("hand", f"theta {name}: heading_vector ({hv[0]:.17g}, {hv[1]:.17g}) -> rounded ({hv[0]:+.4f}, {hv[1]:+.4f})")

# ---- the naive heading over the lap ----
naive = naive_arctan(D)
miss_naive = np.abs(wrap_angle(naive - yaw[:-1]))
bad = miss_naive > 1.0
out("naive", f"arctan(dy/dx) vs yaw: max miss {miss_naive.max():.5f} rad; steps missing > 1 rad {int(bad.sum())} of {len(D)} = {100 * bad.mean():.2f}%; "
             f"smallest miss among them {miss_naive[bad].min():.4f}; steps with dx < 0 {int((D[:, 0] < 0).sum())}; "
             f"misses > 1 rad that are not dx < 0: {int((bad & (D[:, 0] >= 0)).sum())}",
    share=float(100 * bad.mean()), count=int(bad.sum()))
good_naive = ~bad
out("naive_ok", f"max miss on the {int(good_naive.sum())} steps with dx > 0: {miss_naive[good_naive].max():.4f} rad")
kx = int(np.abs(D[:, 0]).argmin())
out("naive_dx0", f"smallest |dx| on the lap: step {kx}, dx {D[kx, 0]:+.2e} dy {D[kx, 1]:+.4f}; slope {D[kx, 1] / D[kx, 0]:+.1f}")
try:
    1.0 / 0.0
except ZeroDivisionError as exc:
    out("zerodiv", f"python 1.0/0.0 raises ZeroDivisionError: {exc}")
with np.errstate(divide="ignore"):
    out("zerodiv_np", f"numpy: arctan(1/0) {float(np.arctan(np.float64(1.0) / np.float64(0.0))):.4f}; arctan(-1/0) "
                      f"{float(np.arctan(np.float64(-1.0) / np.float64(0.0))):.4f}")

import warnings
with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")
    q = np.float64(1.0) / np.float64(0.0)
out("zerodiv_np2", f"numpy float64 1.0/0.0 = {q}; warnings: {[str(w.category.__name__) + ': ' + str(w.message) for w in wlist]}")

# ---- opposite arrows ----
for v in ((1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)):
    out("opp", f"v {v}: slope {v[1] / v[0]:+.1f}; arctan {float(naive_arctan(v)):+.4f}; atan2 {float(direction_angle(v)):+.4f}")
for v in ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (-1.0, 1e-12), (-1.0, -1e-12)):
    out("quad", f"direction_angle{v} = {float(direction_angle(v)):+.6f}")
out("zero", f"np.arctan2(0, 0) = {float(np.arctan2(0.0, 0.0))}; direction_angle((0, 0)) = {float(direction_angle((0.0, 0.0)))}; "
            f"rows [[3, 4], [0, 0]] -> {direction_angle(np.array([[3.0, 4.0], [0.0, 0.0]]))}")
zero_steps = int((L == 0).sum())
out("zero_lap", f"zero-length steps on the gentle lap: {zero_steps}")

# ---- atan2 over the lap ----
A = direction_angle(D)
gap = wrap_angle(A - yaw[:-1])
kmax = int(np.abs(gap).argmax())
out("atan2", f"direction_angle vs yaw: max |wrap(direction - yaw)| {np.abs(gap).max():.6f} rad at step {kmax} "
             f"(progress {tel.progress_m[kmax]:.1f} m); steps > 0.1 rad {int((np.abs(gap) > 0.1).sum())}", max=float(np.abs(gap).max()))
out("atan2_range", f"direction_angle over the lap: min {A.min():+.5f} max {A.max():+.5f}")

# ---- angles go round ----
raw = A - yaw[:-1]
out("raw", f"unwrapped direction - yaw: max |.| {np.abs(raw).max():.4f}; steps > 1 rad {int((np.abs(raw) > 1).sum())} = "
           f"{100 * (np.abs(raw) > 1).mean():.2f}%; first such step {int(np.argmax(np.abs(raw) > 1))} "
           f"(t {tel.t[int(np.argmax(np.abs(raw) > 1))]:.2f} s, progress {tel.progress_m[int(np.argmax(np.abs(raw) > 1))]:.1f} m)")
jumps = np.flatnonzero(np.abs(np.diff(A)) > math.pi)
out("jumps", f"steps where direction_angle jumps by more than pi between neighbours: {len(jumps)}; at steps {jumps.tolist()[:12]}"
             f"{' ...' if len(jumps) > 12 else ''}; first at progress {tel.progress_m[jumps[0]]:.1f} m, last at {tel.progress_m[jumps[-1]]:.1f} m; "
             f"inside the back straight (361.3-441.3 m): {int(((tel.progress_m[jumps] > 361.3) & (tel.progress_m[jumps] < 441.3)).sum())}")
back = (tel.progress_m[:-1] > 361.3) & (tel.progress_m[:-1] < 441.3)
out("back", f"back straight (progress 361.3-441.3 m), {int(back.sum())} steps: direction_angle > 0 on {int((A[back] > 0).sum())}, "
            f"< 0 on {int((A[back] < 0).sum())}; yaw there {yaw[:-1][back].min():.4f} to {yaw[:-1][back].max():.4f}")
out("yawrun", f"yaw over the lap: min {yaw.min():+.6f} max {yaw.max():.6f}; final yaw {yaw[-1]:.6f}, wrapped {float(wrap_angle(yaw[-1])):+.6f}; "
              f"2pi - final {2 * math.pi - yaw[-1]:.6f}; first step with yaw > pi: {int(np.argmax(yaw > math.pi))} "
              f"(progress {tel.progress_m[int(np.argmax(yaw > math.pi))]:.1f} m)", final=float(yaw[-1]), final_wrapped=float(wrap_angle(yaw[-1])))
cross_pi = np.flatnonzero(np.diff(np.sign(yaw - math.pi)) != 0)
out("yawcross", f"yaw crosses pi {len(cross_pi)} time(s); at progress {[round(float(tel.progress_m[i]), 1) for i in cross_pi]} m; "
                f"wrapped yaw drops by more than pi between neighbours {int((np.abs(np.diff(wrap_angle(yaw))) > math.pi).sum())} time(s)")
after = tel.progress_m[:-1] > 442.3
out("raw_after", f"steps after the last flip (progress > 442.3 m): {int(after.sum())}; with |raw + 2pi| < 0.05: "
                 f"{int((np.abs(raw[after] + 2 * math.pi) < 0.05).sum())}; raw range {raw[after].min():+.4f} to {raw[after].max():+.4f}")
out("yaw1100", f"step 1100: yaw {yaw[1100]:.4f}, wrapped {float(wrap_angle(yaw[1100])):+.4f}, direction_angle {A[1100]:+.4f}, "
               f"raw difference {A[1100] - yaw[1100]:+.4f}, wrapped difference {float(wrap_angle(A[1100] - yaw[1100])):+.4f}")
kl = len(D) - 1
out("lastrow", f"last arrow (step {kl}): direction_angle {A[kl]:+.4f}, yaw {yaw[kl]:.4f}, raw difference {A[kl] - yaw[kl]:+.4f}, "
               f"wrapped {float(wrap_angle(A[kl] - yaw[kl])):+.4f}")
a0 = float(yaw[-1])
out("wrapsteps", f"wrap of the final yaw by steps: {a0:.6f} + pi = {a0 + math.pi:.4f}; % 2pi = {(a0 + math.pi) % (2 * math.pi):.4f}; "
                 f"- pi = {(a0 + math.pi) % (2 * math.pi) - math.pi:+.4f}")
out("turn700", f"step 700: yaw change over the step {yaw[K + 1] - yaw[K]:.5f} rad, half {0.5 * (yaw[K + 1] - yaw[K]):.5f}")
for a in (6.226408, math.pi, -math.pi, 3.5, -3.5, 0.0, 2 * math.pi, 7.0):
    out("wrap", f"wrap({a:+.6f}) = {float(wrap_angle(a)):+.6f}; (a + pi) % 2pi = {(a + math.pi) % (2 * math.pi):.4f}")
out("mod", f"python -0.3584 % 6.2832 = {-0.3584 % 6.2832:.4f}; math.fmod(-0.3584, 6.2832) = {math.fmod(-0.3584, 6.2832):.4f}; "
           f"fmod wrap of -3.5: {math.fmod(-3.5 + math.pi, 2 * math.pi) - math.pi:+.4f}")
rng = np.random.default_rng(1111)
probe = np.concatenate([rng.uniform(-40, 40, 100_000), np.arange(-8, 9) * math.pi, [math.pi, -math.pi, 0.0, 1e-300]])
same = np.array_equal(wrap_angle(probe), world.wrap_angle(probe))
w = wrap_angle(probe)
out("wrap_bits", f"wrap_angle vs sim.world.wrap_angle on {len(probe)} angles (seed 1111): bit-identical {same}; "
                 f"range [{w.min():+.6f}, {w.max():+.6f}]; any == +pi {bool((w == math.pi).any())}")
raw_miss = np.abs(A - wrap_angle(yaw[:-1]))
out("gradertrap", f"|direction_angle - wrap_angle(yaw)| WITHOUT wrapping the difference: max {raw_miss.max():.4f} on "
                  f"{int((raw_miss > 0.05).sum())} steps; with the difference wrapped: max {np.abs(wrap_angle(A - wrap_angle(yaw[:-1]))).max():.6f}")

# ---- ahead and to the left: hand case facing north ----
fl = forward_left(math.pi / 2, (2.0, 3.0))
out("north", f"theta pi/2, v = (2 east, 3 north): forward {fl[0]:+.4f} left {fl[1]:+.4f}; heading arrow "
             f"({math.cos(math.pi / 2):+.4f}, {math.sin(math.pi / 2):+.4f}); left arrow ({-math.sin(math.pi / 2):+.4f}, {math.cos(math.pi / 2):+.4f})")
fl0 = forward_left(0.0, (2.0, 3.0))
out("east", f"theta 0, v = (2, 3): forward {fl0[0]:+.4f} left {fl0[1]:+.4f}")

# ---- the cone at step 700 ----
s_car, d_car = road.project(tel.x[K], tel.y[K])
s_cone, d_cone = s_car + 12.0, -4.3
cone = road.offset(s_cone, d_cone)
v = cone - P[K]
fwd, left = forward_left(th, v)
st = BicycleState(float(tel.x[K]), float(tel.y[K]), th, float(tel.speed[K]), float(tel.steer[K]))
tv = to_vehicle(st, cone[0], cone[1])
out("cone", f"car at s {s_car:.2f} m, d {d_car:+.3f} m; cone at s {s_cone:.2f} m, d {d_cone} m (grass: tarmac ends at -{ROAD_HALF_WIDTH}); "
            f"cone x {cone[0]:.3f} y {cone[1]:.3f}")
out("cone_v", f"v = cone - car = ({v[0]:+.4f}, {v[1]:+.4f}); length {math.hypot(*v):.4f} m; direction_angle {float(direction_angle(v)):+.4f}; "
              f"minus yaw {float(wrap_angle(direction_angle(v) - th)):+.4f} rad")
hx, hy = math.cos(th), math.sin(th)
out("cone_dot", f"forward = v.h = {v[0]:+.4f} x {hx:+.4f} + {v[1]:+.4f} x {hy:+.4f} = {v[0] * hx:+.4f} + {v[1] * hy:+.4f} = {fwd:+.4f} m")
out("cone_cross", f"left = h x v = {hx:+.4f} x {v[1]:+.4f} - {hy:+.4f} x {v[0]:+.4f} = {hx * v[1]:+.4f} - {hy * v[0]:+.4f} = {left:+.4f} m")
out("cone_check", f"forward^2 + left^2 = {fwd ** 2 + left ** 2:.4f}; |v|^2 = {v @ v:.4f}; to_vehicle {tv[0]:+.4f}, {tv[1]:+.4f}; "
                  f"max diff vs forward_left {max(abs(tv[0] - fwd), abs(tv[1] - left)):.1e}",
    forward=float(fwd), left=float(left))
nhat = np.array([-hy, hx])
out("cone_n", f"left arrow n-hat ({nhat[0]:+.4f}, {nhat[1]:+.4f}); v.n-hat {v @ nhat:+.4f}; h.n-hat {hx * nhat[0] + hy * nhat[1]:+.1e}")
# a naive guess: forward = distance along the road, left = d difference
out("cone_road", f"along-road gap 12.00 m, across-road gap {d_cone - d_car:+.3f} m (not the car's forward/left)")

# ---- moving is not pointing ----
steer_applied = tel.steer[1:]                     # the wheel angle used during step k -> k+1 (runner records state.steer)
half_turn = 0.5 * (yaw[1:] - yaw[:-1])
pred = yaw[:-1] + steer_applied + half_turn
resid = np.abs(wrap_angle(A - pred))
out("decomp", f"direction - (yaw + steer + half the step's turn): max residual {resid.max():.1e} rad over {len(A)} steps", resid=float(resid.max()))
out("decomp_wrong", f"with tel.steer[k] (the wheel angle BEFORE the step's command): max residual {np.abs(wrap_angle(A - (yaw[:-1] + tel.steer[:-1] + half_turn))).max():.1e}")
out("gapmax", f"largest gap step {kmax}: gap {gap[kmax]:+.6f} = steer {steer_applied[kmax]:+.6f} + half-turn {half_turn[kmax]:+.6f}; "
              f"progress {tel.progress_m[kmax]:.1f} m; t {tel.t[kmax]:.2f} s")
prog = tel.progress_m[:-1]
straight = ((prog > 10) & (prog < 90)) | ((prog > 361.3) & (prog < 441.3))
bend = ((prog > 150) & (prog < 300)) | ((prog > 500) & (prog < 650))
out("gap_straight", f"straights (progress 10-90, 361.3-441.3 m), {int(straight.sum())} steps: max |gap| {np.abs(gap[straight]).max():.4f} rad; "
                    f"max |steer| {np.abs(steer_applied[straight]).max():.4f}")
out("gap_bend", f"bend middles (progress 150-300, 500-650 m), {int(bend.sum())} steps: median gap {np.median(gap[bend]):+.4f} "
                f"= median steer {np.median(steer_applied[bend]):+.4f} + median half-turn {np.median(half_turn[bend]):+.5f}; "
                f"gap range {gap[bend].min():+.4f} to {gap[bend].max():+.4f}; e median {np.median(tel.lat[:-1][bend]):+.3f}")
dyaw_hold = TARGET_SPEED * DT * math.sin(0.0330) / WHEELBASE
out("halfturn_hand", f"one step at steer 0.0330: turn 0.4 x sin(0.0330) / 2.7 = {dyaw_hold:.5f} rad; half {dyaw_hold / 2:.5f}")
out("gap_s700", f"step 700: gap {gap[K]:+.4f} = steer {steer_applied[K]:+.4f} + half-turn {half_turn[K]:+.5f}")
fl_steps = forward_left(yaw[:-1], D)
out("steps_fl", f"each step's arrow in forward/left parts: mean forward {fl_steps[:, 0].mean():.6f} m; min forward {fl_steps[:, 0].min():.6f}; "
                f"largest left {fl_steps[:, 1].max():+.5f} m; smallest left {fl_steps[:, 1].min():+.5f} m; median left in bends "
                f"{np.median(fl_steps[bend, 1]):+.5f}")

# ---- lab previews on the canned gentle drive (the same telemetry) ----
miss = np.abs(wrap_angle(A - yaw[:-1]))
out("preview_a", f"provided direction_angle: steps where your direction misses the yaw by > 0.1 rad: {int((miss > 0.1).sum())}; worst miss (rad): {miss.max():.3f}")
out("preview_a_naive", f"preset naive_arctan.py: steps > 0.1 rad: {int((miss_naive > 0.1).sum())}; worst miss (rad): {miss_naive.max():.3f}")
out("preview_b", f"provided forward_left: mean forward per step (m): {fl_steps[:, 0].mean():.4f}; largest left per step (m): {fl_steps[:, 1].max():.4f}")
swapped = -fl_steps[:, 1]
out("preview_b_break", f"cross product in the wrong order (v x h): largest left per step (m): {swapped.max():.4f}; mean forward unchanged")
dot_both = fl_steps[:, 0]
out("preview_b_break2", f"left part taken as v.(sin, cos) (a sign slip, (sin, cos) instead of (-sin, cos)): largest left per step "
                        f"{(D[:, 0] * np.sin(yaw[:-1]) + D[:, 1] * np.cos(yaw[:-1])).max():.4f} m")

# ---- grader cases ----
named = [((1, 0), 0.0), ((0, 1), math.pi / 2), ((-1, 0), math.pi), ((0, -1), -math.pi / 2), ((1, 1), math.pi / 4),
         ((-1, 1), 3 * math.pi / 4), ((-1, -1), -3 * math.pi / 4), ((1, -1), -math.pi / 4), ((-1, 1e-12), math.pi - 1e-12),
         ((-1, -1e-12), -math.pi + 1e-12)]
worst = max(abs(float(direction_angle(v)) - want) for v, want in named)
out("grade_a_named", f"{len(named)} named cases: max |direction_angle - want| {worst:.1e}; zero vector -> {float(direction_angle((0, 0)))}")
rng = np.random.default_rng(1111)
vs = rng.normal(0.0, 5.0, (30, 2))
out("grade_a_seeded", f"30 vectors, seed 1111, normal(0, 5): max |direction_angle - np.arctan2| {np.abs(direction_angle(vs) - np.arctan2(vs[:, 1], vs[:, 0])).max():.1e}; "
                      f"naive preset max miss {np.abs(wrap_angle(naive_arctan(vs) - np.arctan2(vs[:, 1], vs[:, 0]))).max():.4f}")
out("grade_a_gentle", f"gentle displacements: max |wrap(direction_angle - wrap_angle(yaw))| {np.abs(wrap_angle(A - wrap_angle(yaw[:-1]))).max():.4f} "
                      f"against tolerance 0.05 rad (margin {0.05 - np.abs(gap).max():.4f})")
rng = np.random.default_rng(1112)
poses = np.column_stack([rng.uniform(-200, 200, 40), rng.uniform(-200, 200, 40), rng.uniform(-10, 10, 40)])
pts = rng.uniform(-50, 50, (40, 25, 2))
diff = 0.0
for (x0, y0, t0), p in zip(poses, pts):
    fwd_t, left_t = to_vehicle(BicycleState(float(x0), float(y0), float(t0), TARGET_SPEED, 0.0), p[:, 0], p[:, 1])
    got = forward_left(t0, p - (x0, y0))
    diff = max(diff, float(np.abs(got[:, 0] - fwd_t).max()), float(np.abs(got[:, 1] - left_t).max()))
out("grade_b", f"40 poses x 25 points (seed 1112; x, y uniform +-200 m, theta uniform +-10 rad, points +-50 m): "
               f"max |forward_left - to_vehicle| {diff:.1e} (tolerance 1e-9)")
hv_b = forward_left(np.array([0.3, 1.2]), np.array([[1.0, 2.0], [3.0, -1.0]]))
out("broadcast", f"forward_left(theta shape (2,), v shape (2, 2)) shape {hv_b.shape}; heading_vector(shape (3,)) shape {heading_vector(np.zeros(3)).shape}")
ab = atan2_by_hand(vs)
ab_named = max(abs(float(atan2_by_hand(np.array(v, float))) - want) for v, want in named)
out("byhand", f"challenge 2, atan2 from arctan + np.where: max diff vs np.arctan2 on the 30 seeded vectors {np.abs(ab - np.arctan2(vs[:, 1], vs[:, 0])).max():.1e}; "
              f"on the named cases {ab_named:.1e}; zero vector -> {float(atan2_by_hand(np.zeros(2)))}")

# ---- challenge: heading from noisy GPS ----
SIGMA, SEEDS, KS = 0.05, 500, np.arange(1, 41)
tel_s = run(Car(), "straight")
Ps = np.stack([tel_s.x, tel_s.y], axis=-1)
out("straight", f"straight, provided car: {len(tel_s.t)} steps, completed {tel_s.completed}; max |yaw| {np.abs(tel_s.yaw).max():.1e}; "
                f"max |y - lane centre| {np.abs(tel_s.y - EGO_LANE_CENTER).max():.1e}")
errs = {k: [] for k in KS}
for seed in range(SEEDS):
    noisy = Ps + np.random.default_rng(seed).normal(0.0, SIGMA, Ps.shape)
    for k in KS:
        d = noisy[k:] - noisy[:-k]
        errs[k].append(wrap_angle(np.arctan2(d[:, 1], d[:, 0]) - tel_s.yaw[:-k]))
law = math.sqrt(2) * SIGMA / (KS * TARGET_SPEED * DT)
spread = np.array([np.concatenate(errs[k]).std() for k in KS])
mean = np.array([np.concatenate(errs[k]).mean() for k in KS])
ratio = spread / law
out("gps", f"sigma {SIGMA} m per axis, seeds 0-{SEEDS - 1}, every start step, k = 1..40; spread = standard deviation of "
           f"wrap(fix-to-fix direction - yaw); law sqrt(2) sigma / (0.4 k)")
for k in (1, 2, 5, 10, 17, 18, 20, 40):
    out("gps_k", f"k {k}: law {law[k - 1]:.5f} rad, simulated {spread[k - 1]:.5f} rad, ratio {ratio[k - 1]:.3f}, mean {mean[k - 1]:+.1e}, "
                 f"samples {sum(len(e) for e in errs[k])}")
out("gps_fit", f"ratio simulated / law over k = 1..40: min {ratio.min():.3f} (k {KS[ratio.argmin()]}) max {ratio.max():.3f} (k {KS[ratio.argmax()]}); "
               f"k >= 2: within {100 * np.abs(ratio[1:] - 1).max():.1f}%", fit_max_dev=float(np.abs(ratio - 1).max()))
kneed = math.sqrt(2) * SIGMA / (0.01 * TARGET_SPEED * DT)
first_ok = int(KS[np.argmax(spread <= 0.01)])
out("gps_k_needed", f"law: k >= sqrt(2) x 0.05 / (0.01 x 0.4) = {kneed:.2f}, so k = {math.ceil(kneed)}; sqrt(2) x 0.05 = {math.sqrt(2) * SIGMA:.4f}; "
                    f"simulated: first k with spread <= 0.01 is {first_ok}", k_needed=math.ceil(kneed))
KC = math.ceil(kneed)
chord = direction_angle(P[KC:] - P[:-KC])
i_bend = np.flatnonzero(((prog[: N - KC] > 150) & (prog[KC - 1: N - 1] < 300)) | ((prog[: N - KC] > 500) & (prog[KC - 1: N - 1] < 650)))
c_err = wrap_angle(chord - yaw[:-KC])[i_bend]
turn_half = 0.5 * (yaw[KC:] - yaw[:-KC])[i_bend]
st_mean = np.array([steer_applied[i: i + KC].mean() for i in i_bend])
mid = wrap_angle(chord - yaw[KC // 2: N - KC + KC // 2])[i_bend]
c_str = wrap_angle(chord - yaw[:-KC])[np.flatnonzero((prog[: N - KC] > 10) & (prog[KC - 1: N - 1] < 90))]
out("bend_chord", f"gentle, no noise, k {KC}, chord from fix i to fix i+{KC} vs yaw at fix i, {len(i_bend)} starts inside the bend middles: "
                  f"median {np.median(c_err):+.4f} rad = median steer {np.median(st_mean):+.4f} + median half the {KC} steps' turn "
                  f"{np.median(turn_half):+.4f}; median residual {np.median(c_err - st_mean - turn_half):+.1e}; first straight max |.| {np.abs(c_str).max():.4f}")
out("bend_mid", f"same chords vs the yaw at the middle fix i+{KC // 2}: median {np.median(mid):+.4f} rad (the steering angle is left); "
                f"ratio to the 0.01 target {np.median(c_err) / 0.01:.1f}x at fix i, {np.median(mid) / 0.01:.1f}x at the middle")
out("bend_turn", f"one step's turn in the bends {np.median(2 * half_turn[bend]):.5f} rad; {KC} steps {KC * np.median(2 * half_turn[bend]):.4f}; half {KC * np.median(half_turn[bend]):.4f}")

# ---- next lesson thumbnail scene ----
s_bend = 100.0
pose = road.ego_pose(s_bend)
out("next", f"car at the start of the first bend: ego_pose(100) = ({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.4f}); "
            f"yellow line and white line drawn from s 100 to 130 m")

(HERE / "measured.json").write_text(json.dumps(J, indent=1, sort_keys=True))
out("done", f"{len(J)} values written to measured.json")
