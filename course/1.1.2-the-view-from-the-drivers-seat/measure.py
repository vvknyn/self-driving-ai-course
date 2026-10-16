"""Measurements for "The view from the driver's seat" (1.1.2). Every number in lecture-v2.md comes from this output.

Run with PYTHONPATH=<course repo>. It uses the real F0 geom functions and the Camera(world_to_ego=) and
run(..., camera=) seams. Lines tagged (timing) vary run to run.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

import zero2fsd

REPO = Path(os.environ.get("PYTHONPATH", ".").split(os.pathsep)[0]).resolve()
assert Path(zero2fsd.__file__).resolve().is_relative_to(REPO), (zero2fsd.__file__, REPO)

from zero2fsd.car import Car, run, runner  # noqa: E402
from zero2fsd.car import provided  # noqa: E402
from zero2fsd.score import metrics  # noqa: E402
from zero2fsd.sim import BicycleState, make_scenario  # noqa: E402
from zero2fsd.sim import camera as cam_mod  # noqa: E402
from zero2fsd.sim import dynamics, scenarios, world  # noqa: E402
from zero2fsd.sim.camera import Camera  # noqa: E402

HERE = Path(__file__).parent
OUT: dict = {}


def say(key: str, text: str, value=None) -> None:
    """Print one tagged measurement line and keep its value for measured.json."""
    OUT[key] = text if value is None else value
    print(f"{key} {text}")


from zero2fsd.geom import ego_to_world, forward_left, heading_vector, rotation_matrix, world_to_ego, wrap_angle  # noqa: E402


def drive(scenario: str, camera: Camera | None = None):
    """run(Car(), scenario, camera=camera): the provided car; only the renderer's camera changes."""
    return run(Car(), scenario, camera=camera)


# ---- learner-style transforms: the right one in matrix form, and the lab's two break-its ----
def matrix_form(pose, p_world):
    """(p − t) @ R(θ): rows times R is Rᵀ applied to every point."""
    x, y, theta = pose
    return (np.asarray(p_world, dtype=float) - np.array([x, y])) @ rotation_matrix(theta)


def r_not_transposed(pose, p_world):
    """Break-it: R(θ)(p − t), the rotation the wrong way round."""
    x, y, theta = pose
    return (np.asarray(p_world, dtype=float) - np.array([x, y])) @ rotation_matrix(theta).T


def no_rotation(pose, p_world):
    """Break-it: p − t, no rotation at all."""
    x, y, theta = pose
    return np.asarray(p_world, dtype=float) - np.array([x, y])


VARIANTS = {"provided": world_to_ego, "matrix form": matrix_form, "R not transposed": r_not_transposed, "no rotation": no_rotation}
BROKEN = {"R not transposed": r_not_transposed, "no rotation": no_rotation}


def f3(v) -> str:
    return f"{v:+.3f}"


def summary(tel) -> str:
    m = metrics(tel)
    return (f"completed {bool(tel.completed)}, end {tel.progress_m[-1]:.1f} m at t {tel.t[-1]:.2f} s, "
            f"mean|lat| {m['mean_abs_lat']:.3f} m, max|lat| {np.abs(tel.lat).max():.3f} m, steps {len(tel.t)}")


def state_at(tel, k) -> BicycleState:
    return BicycleState(float(tel.x[k]), float(tel.y[k]), float(tel.yaw[k]), float(tel.speed[k]), 0.0)


# ---- M0 constants the lecture quotes ----
say("M00", f"constants: WHEELBASE {dynamics.WHEELBASE} m, DT {dynamics.DT} s, MAX_STEER {dynamics.MAX_STEER} rad, "
    f"MAX_STEER_RATE {dynamics.MAX_STEER_RATE} rad/s, K_OFF {provided.K_OFF}, K_HEAD {provided.K_HEAD}, "
    f"LANE_WIDTH {world.LANE_WIDTH} m, EGO_LANE_CENTER {world.EGO_LANE_CENTER} m, OFF_ROAD_OFFSET {runner.OFF_ROAD_OFFSET} m, TARGET_SPEED {scenarios.TARGET_SPEED} m/s, "
    f"camera {Camera().width}x{Camera().height}, BEHIND {cam_mod.BEHIND} m, AHEAD {cam_mod.AHEAD} m, "
    f"GRID_STEP {cam_mod.GRID_STEP} m, NEAR_PLANE {cam_mod.NEAR_PLANE} m, WHITE_LINE_LATERAL {cam_mod.WHITE_LINE_LATERAL} m")
road = make_scenario("gentle").road
x0, y0, h0 = road.pose_at(0.0)
xb, yb, hb = road.pose_at(100.0)
say("M01", f"gentle road: length {road.length:.1f} m, start pose ({x0:.1f}, {y0:.1f}) heading {h0:.3f} rad, "
    f"first bend starts at s 100.0 m heading {hb:.3f} rad, back straight heading at s 400 {road.pose_at(400.0)[2]:.4f} rad")

# ---- M02 the reference drive ----
ref = drive("gentle")
say("M02", f"provided car on gentle, provided renderer: {summary(ref)}")

# ---- M03 the scene: the driver's seat, 50 m into the first bend (and the bend start for contrast) ----
for key, s_target in (("M03", 100.0), ("M04", 150.0)):
    k = int(np.argmax(ref.progress_m >= s_target))
    st = state_at(ref, k)
    pose = (st.x, st.y, st.yaw)
    s_car, lat_car = road.project(st.x, st.y)
    road_dir = float(road.pose_at(s_car)[2])
    e, psi = float(ref.lat[k]), float(ref.head_err[k])
    foot_w = road.offset(s_car, world.EGO_LANE_CENTER)
    foot_e = world_to_ego(pose, foot_w)
    say(key, f"scene at progress {s_target:.0f} m: step {k}, t {ref.t[k]:.2f} s, car x {st.x:.3f} y {st.y:.3f}, "
        f"theta {st.yaw:.4f} rad ({math.degrees(st.yaw):.1f} deg), road direction {road_dir:.4f} rad, "
        f"e {e:+.4f} m, psi {psi:+.4f} rad, theta - road direction {wrap_angle(st.yaw - road_dir):+.4f}; "
        f"lane centre beside the car: world ({foot_w[0]:.3f}, {foot_w[1]:.3f}), ego ({foot_e[0]:+.4f}, {foot_e[1]:+.4f})")
    for ahead in (10.0, 20.0, 30.0):
        pw = road.offset(s_car + ahead, world.EGO_LANE_CENTER)
        v = pw - np.array([st.x, st.y])
        pe = world_to_ego(pose, pw)
        say(f"{key}.{int(ahead)}", f"lane-centre point {ahead:.0f} m further along the lane: world ({pw[0]:.3f}, {pw[1]:.3f}), "
            f"v = p - t ({v[0]:+.3f}, {v[1]:+.3f}), ego (forward {pe[0]:+.3f}, left {pe[1]:+.3f})")
    if key == "M04":
        scene_k, scene_pose, scene_e, scene_psi = k, pose, e, psi
        pw10 = road.offset(s_car + 10.0, world.EGO_LANE_CENTER)
        v10 = pw10 - np.array([st.x, st.y])

# ---- M05 the worked example, step by step, at the M04 pose ----
th = scene_pose[2]
c, s = math.cos(th), math.sin(th)
h = heading_vector(th)
n = np.array([-math.sin(th), math.cos(th)])
fl = forward_left(th, v10)
Rm = rotation_matrix(th)
say("M05", f"worked example theta {th:.4f}: cos {c:.4f}, sin {s:.4f}; h = ({h[0]:.4f}, {h[1]:.4f}), n = ({n[0]:+.4f}, {n[1]:+.4f}); "
    f"v ({v10[0]:+.3f}, {v10[1]:+.3f}); v.h = {v10 @ h:+.3f}, v.n = {v10 @ n:+.3f}, h x v = {h[0]*v10[1]-h[1]*v10[0]:+.3f}; "
    f"forward_left ({fl[0]:+.3f}, {fl[1]:+.3f}); R^T v ({(Rm.T @ v10)[0]:+.3f}, {(Rm.T @ v10)[1]:+.3f})")
say("M05.R", f"R(theta) rows [[{Rm[0,0]:+.4f}, {Rm[0,1]:+.4f}], [{Rm[1,0]:+.4f}, {Rm[1,1]:+.4f}]]; "
    f"column 1 = h: {np.allclose(Rm[:, 0], h, atol=0, rtol=0)}, column 2 = n: {np.allclose(Rm[:, 1], n, atol=0, rtol=0)}")
q_ego = world_to_ego(scene_pose, pw10)
back = Rm @ q_ego + np.array(scene_pose[:2])
say("M05.walk", f"walk {q_ego[0]:.3f} m along h then {q_ego[1]:+.3f} m along n from t: ({back[0]:.3f}, {back[1]:.3f}) vs p ({pw10[0]:.3f}, {pw10[1]:.3f})")
# subtract-then-rotate vs rotate-then-subtract (the order trap)
wrong_order = Rm.T @ pw10 - np.array(scene_pose[:2])
say("M05.order", f"rotate first, then subtract t: ({wrong_order[0]:+.3f}, {wrong_order[1]:+.3f}) instead of ({q_ego[0]:+.3f}, {q_ego[1]:+.3f})")

# ---- M06 R^T R = I ----
rng = np.random.default_rng(112)
thetas = rng.uniform(-math.pi, math.pi, 40)
err_I = max(np.abs(rotation_matrix(t).T @ rotation_matrix(t) - np.eye(2)).max() for t in thetas)
say("M06", f"R^T R - I over 40 random theta: max |entry| {err_I:.1e}, below 1e-12 {err_I < 1e-12}; "
    f"at theta {th:.4f}: R^T R = [[{(Rm.T@Rm)[0,0]:.4f}, {(Rm.T@Rm)[0,1]:.4f}], [{(Rm.T@Rm)[1,0]:.4f}, {(Rm.T@Rm)[1,1]:.4f}]]")

# ---- M07 round trip on a point cloud ----
cloud = np.concatenate([road.sample_points(1.0, d) for d in (0.0, world.EGO_LANE_CENTER, cam_mod.WHITE_LINE_LATERAL)])
poses = np.column_stack([rng.uniform(-150, 150, 40), rng.uniform(-150, 150, 40), rng.uniform(-math.pi, math.pi, 40)])
rt1 = max(np.abs(ego_to_world(p, world_to_ego(p, cloud)) - cloud).max() for p in [scene_pose, *poses])
rt2 = max(np.abs(world_to_ego(p, ego_to_world(p, cloud)) - cloud).max() for p in [scene_pose, *poses])
say("M07", f"round trip on {len(cloud)} road points at 41 poses: world->ego->world max error {rt1:.1e} m, "
    f"ego->world->ego {rt2:.1e} m, both below 1e-12 m: {max(rt1, rt2) < 1e-12}")

# ---- M08 shapes: (K, 4, 2) quads in one call, and the shape trap ----
cam0 = Camera()
seen = []


def logging_w2e(pose, quads):
    seen.append(np.shape(quads))
    return world_to_ego(pose, quads)


st150 = BicycleState(*road.ego_pose(150.0), 8.0, 0.0)
Camera(world_to_ego=logging_w2e).render(road, st150)
say("M08", f"one picture at s 150 calls world_to_ego {len(seen)} times with shapes {seen}; "
    f"total quads {sum(sh[0] for sh in seen)}, corners {sum(sh[0] * sh[1] for sh in seen)}; output shape {world_to_ego(scene_pose, np.zeros(seen[0])).shape}")
pts5 = road.offset(np.array([160.0, 170.0, 180.0, 190.0, 200.0]), world.EGO_LANE_CENTER)
try:
    Rm @ (pts5 - np.array(scene_pose[:2]))
    trap5 = "no error"
except ValueError as exc:
    trap5 = f"ValueError: {exc}"
pts2 = pts5[:2] - np.array(scene_pose[:2])
right2 = pts2 @ Rm
wrong2 = Rm @ pts2
say("M08.trap", f"R @ points for 5 points stored as rows: {trap5}")
say("M08.trap2", f"2 points as rows (lane centre at road s 160 and 170): rows @ R -> [[{right2[0,0]:+.3f}, {right2[0,1]:+.3f}], [{right2[1,0]:+.3f}, {right2[1,1]:+.3f}]]; "
    f"R @ rows -> [[{wrong2[0,0]:+.3f}, {wrong2[0,1]:+.3f}], [{wrong2[1,0]:+.3f}, {wrong2[1,1]:+.3f}]] with no error")

# ---- M09 40 seeded poses against the renderer's own to_vehicle (lab 1.1.2.a, tolerance 1e-9) ----
rng_lab = np.random.default_rng(1102)
lab_poses = np.column_stack([rng_lab.uniform(-200, 200, 40), rng_lab.uniform(-200, 200, 40), rng_lab.uniform(-4 * math.pi, 4 * math.pi, 40)])
lab_pts = rng_lab.uniform(-200, 200, (40, 25, 2))
for name, fn in VARIANTS.items():
    worst = 0.0
    for (px, py, pth), pts in zip(lab_poses, lab_pts):
        F, L = cam_mod.to_vehicle(BicycleState(px, py, pth, 0.0, 0.0), pts[:, 0], pts[:, 1])
        worst = max(worst, float(np.abs(fn((px, py, pth), pts) - np.stack([F, L], axis=-1)).max()))
    say(f"M09.{name}", f"40 seeded poses vs to_vehicle, {name}: largest difference {worst:.3g} m, within 1e-9 {worst <= 1e-9}")

# ---- M10 named poses: a point 4 m ahead and 1 m left of the car; the answer is (4, 1) at every yaw ----
for label, yaw in (("0", 0.0), ("pi/2", math.pi / 2), ("pi", math.pi), ("-3pi/4", -3 * math.pi / 4), ("2.5", 2.5)):
    t = np.array([10.0, 5.0])
    p = t + 4.0 * heading_vector(yaw) + 1.0 * np.array([-math.sin(yaw), math.cos(yaw)])
    parts = []
    for name, fn in VARIANTS.items():
        q = fn((10.0, 5.0, yaw), p)
        parts.append(f"{name} ({q[0]:+.3f}, {q[1]:+.3f}) err {np.abs(q - [4.0, 1.0]).max():.1e}")
    say(f"M10.{label}", f"named pose yaw {label} = {yaw:.4f}, point 4 ahead 1 left at world ({p[0]:.3f}, {p[1]:.3f}): " + "; ".join(parts))

# ---- M11 renderer swap: byte-identical pictures over the reference drive ----
every10 = range(0, len(ref.t), 10)
idx18 = np.linspace(0, len(ref.t) - 1, 18).astype(int)
for name, fn in VARIANTS.items():
    cam_l = Camera(world_to_ego=fn)
    for tag, idx in (("every 10th step", every10), ("18 sampled", idx18)):
        frames = px = 0
        for k in idx:
            st = state_at(ref, k)
            d = (cam0.render(road, st) != cam_l.render(road, st)).any(axis=-1).sum()
            frames += int(d > 0)
            px += int(d)
        say(f"M11.{name}.{len(idx)}", f"{name}: {frames} of {len(idx)} pictures differ ({tag} of the gentle drive), {px} pixels")

# ---- M12 the swap and the dropped rotation at parked poses along gentle ----
for s_ in (0.0, 50.0, 100.0, 150.0, 260.0, 300.0, 400.0):
    st = BicycleState(*road.ego_pose(s_), 8.0, 0.0)
    good = cam0.render(road, st, return_labels=True)[1]
    parts = []
    for name, fn in BROKEN.items():
        bad = Camera(world_to_ego=fn).render(road, st, return_labels=True)[1]
        parts.append(f"{name} {(good != bad).mean():.1%}")
    say(f"M12.{int(s_)}", f"parked on the lane centre at s {s_:.0f}: yaw {st.yaw:.4f} rad, label pixels that differ: " + ", ".join(parts))

# ---- M13 the three renderers driving straight and gentle ----
tels = {}
for name, fn in (("provided", None), *BROKEN.items()):
    for scen in ("straight", "gentle"):
        tel = drive(scen, None if fn is None else Camera(world_to_ego=fn))
        tels[name, scen] = tel
        extra = f", {tel.progress_m[-1] - 100.0:.1f} m into the first bend" if scen == "gentle" and not tel.completed and tel.progress_m[-1] > 100 else ""
        say(f"M13.{name}.{scen}", f"{name} renderer, {scen}: {summary(tel)}{extra}")
mine = drive("gentle", Camera(world_to_ego=matrix_form))
same = all(np.array_equal(getattr(mine, c), getattr(ref, c)) for c in runner.COLUMNS)
say("M13.matrix", f"matrix-form renderer, gentle: {summary(mine)}; telemetry identical to the provided renderer's {same}; "
    f"gate mean|lat| < 0.5 passes {bool(mine.completed) and metrics(mine)['mean_abs_lat'] < 0.5}")

# ---- M14 challenge: the swap flips the heading the box reports ----
from zero2fsd.car.types import Observation  # noqa: E402

sw = tels["R not transposed", "straight"]
swap_cam = Camera(world_to_ego=r_not_transposed)


def box_at(camera, st):
    est = provided.estimate_lane(Observation(camera.render(road_straight, st), 8.0, 0.0))
    return est.offset_m, est.heading_rad


road_straight = make_scenario("straight").road
ok = np.isfinite(sw.est_heading) & (np.abs(sw.head_err) > 1e-3)
right_h = np.array([box_at(cam0, state_at(sw, k))[1] for k in np.nonzero(ok)[0]])
say("M14.flip", f"swap run on straight, the {ok.sum()} steps with |psi| > 0.001: box heading / true psi, median "
    f"{np.median(sw.est_heading[ok] / sw.head_err[ok]):+.3f} through the swapped renderer, "
    f"{np.nanmedian(right_h / sw.head_err[ok]):+.3f} through the right renderer at the same states "
    f"({np.isfinite(right_h).sum()} of them with a valid box reading)")
for yaw_err in (0.02, -0.05):
    st = BicycleState(*road_straight.ego_pose(50.0, 0.3, yaw_err), 8.0, 0.0)
    (o1, h1), (o2, h2) = box_at(cam0, st), box_at(swap_cam, st)
    say(f"M14.park{yaw_err:+}", f"parked on straight at s 50, e +0.3 m, psi {yaw_err:+.2f} rad: box (offset, heading) right "
        f"({o1:+.4f}, {h1:+.4f}), swapped ({o2:+.4f}, {h2:+.4f})")
v, L, ko, kh, dt = scenarios.TARGET_SPEED, dynamics.WHEELBASE, provided.K_OFF, provided.K_HEAD, dynamics.DT
say("M14.v", f"model inputs: v {v} m/s (TARGET_SPEED; median speed on the swap run {np.median(sw.speed):.3f}), "
    f"L {L} m, k_off {ko}, k_head {kh}, step {dt} s")
heading_only = v * kh / L
A = {sgn: np.array([[-v * ko, v * (1 - sgn * kh)], [-v * ko / L, -sgn * v * kh / L]]) for sgn in (1, -1)}


def step_map(sgn):
    """One simulator step, small angles, wheels at the commanded angle: x_{k+1} = Phi x_k, x = (e, psi)."""
    a = v * dt * (1 + v * dt / (2 * L))
    return np.array([[1 - a * ko, v * dt - a * sgn * kh], [-(v * dt / L) * ko, 1 - (v * dt / L) * sgn * kh]])


def rates(eig_map):
    lam = np.log(eig_map.astype(complex)) / dt
    return lam[np.argmax(lam.real)]


eg, es = np.linalg.eigvals(A[1]), np.linalg.eigvals(A[-1])
ls = es[np.argmax(es.imag)]
say("M14.model", f"heading-only model: psi changes at {-heading_only:+.3f} per second times psi with the right renderer, "
    f"{heading_only:+.3f} with the swap; two-variable model: right renderer rates {np.sort(eg.real)[0]:+.3f} and {np.sort(eg.real)[1]:+.3f} per second; "
    f"swap {ls.real:+.3f} +/- {ls.imag:.3f}i, growth {ls.real:.3f} per second, e-fold {1/ls.real:.3f} s, swing period {2*math.pi/ls.imag:.2f} s, "
    f"extreme to zero crossing {math.atan2(ls.imag, ls.real)/ls.imag:.3f} s")
ld_good, ld_swap = rates(np.linalg.eigvals(step_map(1))), rates(np.linalg.eigvals(step_map(-1)))
say("M14.stepmodel", f"step-by-step model: right renderer slowest rate {sorted(np.log(np.linalg.eigvals(step_map(1)).astype(complex)).real / dt)[-1]:+.3f} per second; "
    f"swap {ld_swap.real:+.3f} +/- {abs(ld_swap.imag):.3f}i, swing period {2*math.pi/abs(ld_swap.imag):.2f} s")
first = int(np.argmax(np.abs(sw.lat) >= 0.005))
last = int(np.nonzero(np.abs(sw.steer) < 0.1)[0][-1]) - 1
X = np.column_stack([sw.lat[first:last], sw.head_err[first:last]])
Y = np.column_stack([sw.lat[first + 1:last + 1], sw.head_err[first + 1:last + 1]])
Phi_fit = np.linalg.lstsq(X, Y, rcond=None)[0].T
lf = rates(np.linalg.eigvals(Phi_fit))
resid = Y - X @ Phi_fit.T
say("M14.fit", f"telemetry fit x_(k+1) = Phi x_k over t {sw.t[first]:.2f}-{sw.t[last]:.2f} s ({last - first} steps, |e| >= 0.005 m, |steer| < 0.1 rad): "
    f"{lf.real:+.3f} +/- {abs(lf.imag):.3f}i, growth {lf.real:.3f} per second, swing period {2*math.pi/abs(lf.imag):.2f} s, "
    f"rms one-step residual e {np.sqrt(np.mean(resid[:, 0]**2)):.4f} m, psi {np.sqrt(np.mean(resid[:, 1]**2)):.4f} rad")
xm = np.array([sw.lat[first], sw.head_err[first]])
pred = [xm]
for _ in range(first, last):
    pred.append(step_map(-1) @ pred[-1])
pred = np.array(pred)
for tt in (1.0, 1.5, 2.0):
    k = int(np.argmin(np.abs(sw.t - tt)))
    say(f"M14.pred{tt}", f"step model started from the telemetry at t {sw.t[first]:.2f} s predicts e {pred[k - first, 0]:+.4f} m at t {sw.t[k]:.2f} s; measured {sw.lat[k]:+.4f} m")
kmin = int(np.argmin(sw.lat))
kz = kmin + int(np.argmax(sw.lat[kmin:] > 0))
tz = sw.t[kz - 1] + (sw.t[kz] - sw.t[kz - 1]) * (-sw.lat[kz - 1]) / (sw.lat[kz] - sw.lat[kz - 1])
say("M14.turn", f"swap on straight: e most negative {sw.lat[kmin]:+.4f} m at t {sw.t[kmin]:.2f} s, back through zero at t {tz:.2f} s, "
    f"{tz - sw.t[kmin]:.2f} s later; heading psi crosses zero at t "
    f"{sw.t[kmin + int(np.argmax(sw.head_err[kmin:] > 0))]:.2f} s")
for tt in (0.5, 1.0, 1.5, 2.0, 2.5):
    k = int(np.argmin(np.abs(sw.t - tt)))
    say(f"M14.t{tt}", f"swap on straight at t {sw.t[k]:.2f} s: e {sw.lat[k]:+.4f} m, psi {sw.head_err[k]:+.4f} rad, "
        f"box heading {sw.est_heading[k]:+.4f} rad, steer {sw.steer[k]:+.4f} rad")
dsteer = np.abs(np.diff(sw.steer))
say("M14.rate", f"swap on straight: steps with the wheels at the 0.05 rad per step limit {int((dsteer > 0.0499).sum())} of {len(dsteer)}; "
    f"first at t {sw.t[1:][dsteer > 0.0499][0] if (dsteer > 0.0499).any() else float('nan'):.2f} s; sign changes of e {int((np.diff(np.sign(sw.lat[np.abs(sw.lat) > 1e-4])) != 0).sum())}")
say("M14.start", f"swap on straight, first steps: e {sw.lat[1]:+.2e} m and psi {sw.head_err[1]:+.2e} rad at t {sw.t[1]:.2f} s; box at t 0: offset {sw.est_offset[0]:+.4f}, heading {sw.est_heading[0]:+.4f}")
# extension: R(a)R(b) = R(a+b), and the car's own heading arrow in the ego frame
a_, b_ = rng.uniform(-math.pi, math.pi, 40), rng.uniform(-math.pi, math.pi, 40)
comp = max(np.abs(rotation_matrix(a) @ rotation_matrix(b) - rotation_matrix(a + b)).max() for a, b in zip(a_, b_))
arrow = max(np.abs(world_to_ego((x_, y_, t_), np.array([x_, y_]) + heading_vector(t_)) - [1.0, 0.0]).max() for x_, y_, t_ in poses)
say("M14.ext", f"R(a)R(b) - R(a+b) over 40 random pairs: max {comp:.1e}; car's heading arrow in ego at 40 poses: max |(.) - (1, 0)| {arrow:.1e}")

# ---- M15 (timing) the 18-picture preview ----
cam_l = Camera(world_to_ego=matrix_form)
t0 = time.perf_counter()
for k in idx18:
    cam_l.render(road, state_at(ref, k))
t_learner = time.perf_counter() - t0
t0 = time.perf_counter()
for k in idx18:
    cam0.render(road, state_at(ref, k))
t_prov = time.perf_counter() - t0
t0 = time.perf_counter()
n_steps = len(run(Car(), "straight", max_steps=100).t)
sps = n_steps / (time.perf_counter() - t0)
say("M15", f"(timing) CPython here: 18 pictures through the learner's transform {t_learner:.3f} s, 18 provided {t_prov:.3f} s, "
    f"probe_steps_per_s-style rate {sps:.0f} steps/s; the 36 renders cost {(t_learner + t_prov) * sps:.0f} simulator steps' worth",
    {"t_learner": t_learner, "t_provided": t_prov, "steps_per_s": sps, "renders_in_steps": (t_learner + t_prov) * sps})

# ---- M16 hand arithmetic the lecture shows (products of the M05 values, full precision, printed at 3-4 dp) ----
say("M16.fwd", f"forward by hand: v_x cos {v10[0]*c:.3f} + v_y sin {v10[1]*s:.3f} = {v10[0]*c + v10[1]*s:+.3f}")
say("M16.left", f"left by hand: cos v_y {c*v10[1]:.3f} - sin v_x {s*v10[0]:.3f} = {c*v10[1] - s*v10[0]:+.3f}")
say("M16.walk", f"walk by hand: {q_ego[0]:.3f} h = ({q_ego[0]*c:.3f}, {q_ego[0]*s:.3f}); {q_ego[1]:.3f} n = ({-q_ego[1]*s:+.3f}, {q_ego[1]*c:.3f}); "
    f"sum ({q_ego[0]*c - q_ego[1]*s:+.3f}, {q_ego[0]*s + q_ego[1]*c:+.3f}); plus t ({scene_pose[0]:.3f}, {scene_pose[1]:.3f}) = ({back[0]:.3f}, {back[1]:.3f})")
say("M16.unit", f"cos^2 {c*c:.4f} + sin^2 {s*s:.4f} = {c*c + s*s:.4f}; cos sin - sin cos = {c*s - s*c:.4f}")
say("M16.small", f"small angle 0.02 rad: cos {math.cos(0.02):.4f}, sin {math.sin(0.02):.4f}")
Ar, As = A[1], A[-1]
say("M16.coef", f"two-variable model entries: right [[{Ar[0,0]:+.3f}, {Ar[0,1]:+.3f}], [{Ar[1,0]:+.4f}, {Ar[1,1]:+.3f}]], trace {np.trace(Ar):+.3f}; "
    f"swap [[{As[0,0]:+.3f}, {As[0,1]:+.3f}], [{As[1,0]:+.4f}, {As[1,1]:+.3f}]], trace {np.trace(As):+.3f}; det both {np.linalg.det(Ar):.3f} {np.linalg.det(As):.3f}; "
    f"half trace squared {(np.trace(As)/2)**2:.3f}; det - that {np.linalg.det(As) - (np.trace(As)/2)**2:.3f}, root {math.sqrt(np.linalg.det(As) - (np.trace(As)/2)**2):.3f}; "
    f"heading-only e-fold {1/heading_only:.3f} s")

# ---- M17 the picture the swap_same figure shows: one of the 178 (a multiple of 10), nearest the opening's step ----
k17 = 380
a17, b17 = cam0.render(road, state_at(ref, k17)), cam_l.render(road, state_at(ref, k17))
say("M17", f"swap_same picture: step {k17}, t {ref.t[k17]:.2f} s, progress {ref.progress_m[k17]:.1f} m, yaw {ref.yaw[k17]:.4f} rad; "
    f"matrix-form picture differs from the provided one in {int((a17 != b17).any(axis=-1).sum())} of {a17.shape[0] * a17.shape[1]} pixels")

if os.environ.get("MEASURED_JSON"):  # the values behind each line, for tools that check the lecture's numbers
    Path(os.environ["MEASURED_JSON"]).write_text(json.dumps(OUT, indent=1, default=float))
