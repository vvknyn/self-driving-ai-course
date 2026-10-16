"""Measure every number the 1.1.4 "Poses you can chain" lecture states.

Run (from anywhere):
  PYTHONPATH=<course checkout> PYTHONPYCACHEPREFIX=<scratch> MPLBACKEND=Agg <venv python> measure.py > measure.out
Writes measured.json next to this file.  Fixed seeds: drift seeds 0..N_SEEDS-1, each np.random.default_rng(seed),
the same draws for every sigma; the 40 lab pairs use default_rng(114).
The geom functions are the real ones from zero2fsd.geom.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from zero2fsd.car import Car, run
from zero2fsd.geom import (T_EGO_REAR, dead_reckon, invert_pose, matrix_to_pose, pose_to_matrix, relative_pose,
                           rotation_matrix, world_to_ego, wrap_angle)
from zero2fsd.sim import BicycleState, make_scenario
from zero2fsd.sim.camera import Camera, to_vehicle
from zero2fsd.sim.dynamics import DT, WHEELBASE
from zero2fsd.sim.scenarios import GENTLE_RADIUS
from zero2fsd.sim.world import EGO_LANE_CENTER
from zero2fsd.sim.world import wrap_angle as sim_wrap

OUT = {}


def say(key, value, fmt="{}"):
    OUT[key] = value if not isinstance(value, np.generic) else value.item()
    print(f"{key}: {fmt.format(value)}")


# ---------------- the drive ----------------
tel = run(Car(), "gentle")
road = make_scenario("gentle").road
X, Y, TH = np.asarray(tel.x), np.asarray(tel.y), np.asarray(tel.yaw)
N = len(X) - 1
say("rows", len(X))
say("steps", N)
say("lap_time_s", N * DT, "{:.2f}")
say("dt_s", DT)
say("wheelbase_m", WHEELBASE)
say("gentle_bend_radius_m", GENTLE_RADIUS)
LANE_R = GENTLE_RADIUS - EGO_LANE_CENTER
say("ego_lane_centre_bend_radius_m", LANE_R)
_cam = __import__("inspect").signature(Camera.__init__).parameters
say("camera_mount_height_m", _cam["height_m"].default)
say("camera_pitch_deg", _cam["pitch_deg"].default)
say("start_pose", [float(X[0]), float(Y[0]), float(TH[0])])
say("end_yaw_rad_unwrapped", float(TH[-1]), "{:.4f}")
say("end_yaw_wrapped_rad", float(wrap_angle(TH[-1])), "{:.4f}")
say("speed_mean_mps", float(np.mean(tel.speed)), "{:.4f}")
Tw = np.stack([pose_to_matrix(p) for p in zip(X, Y, TH)])

# ---------------- A pose is a frame ----------------
print("\n== A pose is a frame ==")
K_EX = 400
p_ex = (float(X[K_EX]), float(Y[K_EX]), float(TH[K_EX]))
T_ex = Tw[K_EX]
say("example_step", K_EX)
say("example_pose", [round(v, 4) for v in p_ex])
say("example_theta_deg", math.degrees(p_ex[2]), "{:.2f}")
say("example_matrix", np.round(T_ex, 4).tolist())
# a point 10 m ahead and 2 m left in the ego frame, to the world, two ways
p_ego = np.array([10.0, 2.0])
via_matrix = (T_ex @ np.array([*p_ego, 1.0]))[:2]
via_rt = rotation_matrix(p_ex[2]) @ p_ego + np.array(p_ex[:2])
say("example_point_ego", p_ego.tolist())
say("example_point_world", np.round(via_matrix, 4).tolist())
say("matrix_vs_rotate_then_add_max_diff_m", float(np.abs(via_matrix - via_rt).max()), "{:.1e}")
say("example_third_row", T_ex[2].tolist())
say("example_point_last_entry", float((T_ex @ np.array([*p_ego, 1.0]))[2]))
rt = max(abs(wrap_angle(matrix_to_pose(pose_to_matrix(p))[2] - p[2])) + abs(matrix_to_pose(pose_to_matrix(p))[0] - p[0])
         + abs(matrix_to_pose(pose_to_matrix(p))[1] - p[1]) for p in zip(X, Y, TH))
say("pose_matrix_pose_roundtrip_max_err", float(rt), "{:.1e}")
th_back = np.array([matrix_to_pose(T)[2] for T in Tw])
say("matrix_to_pose_theta_range", [float(th_back.min()), float(th_back.max())], "{}")
say("matrix_to_pose_vs_unwrapped_yaw_max_raw_diff_rad", float(np.abs(th_back - TH).max()), "{:.4f}")
say("matrix_to_pose_vs_unwrapped_yaw_max_wrapped_diff_rad", float(np.abs(wrap_angle(th_back - TH)).max()), "{:.1e}")

# ---------------- Subscripts that cancel ----------------
print("\n== Subscripts that cancel ==")
T_world_rear = T_ex @ T_EGO_REAR
rear_by_hand = np.array(p_ex[:2]) - WHEELBASE * np.array([math.cos(p_ex[2]), math.sin(p_ex[2])])
say("T_EGO_REAR", T_EGO_REAR.tolist())
say("example_rear_pose", [round(v, 4) for v in matrix_to_pose(T_world_rear)])
say("rear_chain_vs_by_hand_max_diff_m", float(np.abs(T_world_rear[:2, 2] - rear_by_hand).max()), "{:.1e}")

# ---------------- Order matters (by hand) ----------------
print("\n== Order matters ==")
A = pose_to_matrix((1.0, 0.0, math.pi / 2))  # drive 1 m east and turn left a quarter turn
B = pose_to_matrix((1.0, 0.0, 0.0))  # drive 1 m forward, no turn
AB, BA = A @ B, B @ A
say("AB_pose", [round(v, 6) for v in matrix_to_pose(AB)])
say("BA_pose", [round(v, 6) for v in matrix_to_pose(BA)])
say("AB_BA_distance_m", float(np.hypot(*(AB[:2, 2] - BA[:2, 2]))), "{:.4f}")
say("naive_add_pose", [2.0, 0.0, round(math.pi / 2, 6)])
say("AB_matrix", np.round(AB, 6).tolist())
say("BA_matrix", np.round(BA, 6).tolist())

# ---------------- Undoing a pose ----------------
print("\n== Undoing a pose ==")
I3 = np.eye(3)
rt_err = max(float(np.abs(invert_pose(T) @ T - I3).max()) for T in Tw)
rt_err2 = max(float(np.abs(T @ invert_pose(T) - I3).max()) for T in Tw)
say("inverse_roundtrip_max_abs_err", rt_err, "{:.1e}")
say("inverse_roundtrip_other_side_max_abs_err", rt_err2, "{:.1e}")
say("inverse_vs_linalg_inv_max_abs_diff", max(float(np.abs(invert_pose(T) - np.linalg.inv(T)).max()) for T in Tw), "{:.1e}")
say("example_inverse", np.round(invert_pose(T_ex), 4).tolist())
# a world point taken to the ego frame by the inverse equals world_to_ego and the renderer's to_vehicle
rng_pts = np.random.default_rng(7)
diff_w2e = diff_tov = 0.0
for k in range(0, len(X), 50):
    st = BicycleState(X[k], Y[k], TH[k], 8.0, 0.0)
    for q in rng_pts.uniform(-200, 200, size=(4, 2)):
        a = (invert_pose(Tw[k]) @ np.array([*q, 1.0]))[:2]
        b = world_to_ego((X[k], Y[k], TH[k]), q)
        c = np.array(to_vehicle(st, q[0], q[1]))
        diff_w2e = max(diff_w2e, float(np.abs(a - b).max()))
        diff_tov = max(diff_tov, float(np.abs(b - c).max()))
say("inverse_point_vs_world_to_ego_max_diff_m", diff_w2e, "{:.1e}")
say("world_to_ego_vs_to_vehicle_max_diff_m", diff_tov, "{:.1e}")

# ---------------- What the car felt between two steps ----------------
print("\n== What the car felt between two steps ==")
steps = np.stack([relative_pose(Tw[k], Tw[k + 1]) for k in range(N)])
fwd, left, dth = steps[:, 0, 2], steps[:, 1, 2], np.arctan2(steps[:, 1, 0], steps[:, 0, 0])
K_STEP = 100
say("step100_forward_m", float(fwd[K_STEP]), "{:.4f}")
say("step100_left_m", float(left[K_STEP]), "{:.5f}")
say("step100_turn_rad", float(dth[K_STEP]), "{:.5f}")
say("step100_pose_k", [round(float(X[K_STEP]), 4), round(float(Y[K_STEP]), 4), round(float(TH[K_STEP]), 5)])
say("step100_pose_k1", [round(float(X[K_STEP + 1]), 4), round(float(Y[K_STEP + 1]), 4), round(float(TH[K_STEP + 1]), 5)])
say("step100_world_shift_m", [float(X[K_STEP + 1] - X[K_STEP]), float(Y[K_STEP + 1] - Y[K_STEP])], "{}")
say("mean_step_forward_m", float(fwd.mean()), "{:.4f}")
say("mean_step_turn_rad", float(dth.mean()), "{:.5f}")
say("max_abs_step_left_m", float(np.abs(left).max()), "{:.5f}")
say("max_abs_step_turn_rad", float(np.abs(dth).max()), "{:.5f}")
say("sum_step_turn_rad", float(dth.sum()), "{:.4f}")
say("sum_step_forward_m", float(fwd.sum()), "{:.2f}")
# the car's own speed and steering-angle readings predict each step (the bicycle model of the Week-2 module)
sp, st = np.asarray(tel.speed), np.asarray(tel.steer)
dist = 0.5 * (sp[:-1] + sp[1:]) * DT
dyaw = dist * np.sin(st[1:]) / WHEELBASE
half = 0.5 * dyaw
chord = dist * np.where(half != 0, np.sin(half) / np.where(half != 0, half, 1.0), 1.0)
odo_f, odo_l = chord * np.cos(half + st[1:]), chord * np.sin(half + st[1:])
say("odometry_model_vs_relative_pose_max_diff_m", float(max(np.abs(odo_f - fwd).max(), np.abs(odo_l - left).max())), "{:.1e}")
say("odometry_model_vs_relative_pose_max_turn_diff_rad", float(np.abs(dyaw - dth).max()), "{:.1e}")
say("step100_speed_mps", float(sp[K_STEP]), "{:.4f}")
say("step100_steer_rad", float(st[K_STEP + 1]), "{:.5f}")


# ---------------- Dead reckoning ----------------
print("\n== Dead reckoning ==")


def pos_err(Ts):
    return np.hypot(Ts[:, 0, 2] - X, Ts[:, 1, 2] - Y)


def head_err(Ts):
    return np.abs(wrap_angle(np.arctan2(Ts[:, 1, 0], Ts[:, 0, 0]) - TH))


good = dead_reckon(Tw[0], steps)
e_good = pos_err(good)
say("chain_max_pos_err_m", float(e_good.max()), "{:.1e}")
say("chain_end_pos_err_m", float(e_good[-1]), "{:.1e}")
say("chain_max_head_err_rad", float(head_err(good).max()), "{:.1e}")
say("chain_shape", list(good.shape))

# naive: add the step's forward, left and turn straight onto world x, y, theta
nx, ny = X[0] + np.concatenate([[0.0], np.cumsum(fwd)]), Y[0] + np.concatenate([[0.0], np.cumsum(left)])
e_naive = np.hypot(nx - X, ny - Y)
say("naive_end_pos", [round(float(nx[-1]), 2), round(float(ny[-1]), 2)])
say("naive_end_err_m", float(e_naive[-1]), "{:.2f}")
say("naive_max_err_m", float(e_naive.max()), "{:.2f}")
say("naive_err_at_end_of_first_straight_m", float(e_naive[250]), "{:.3f}")

# wrong order: T_world <- S @ T_world
W = [Tw[0]]
for S in steps:
    W.append(S @ W[-1])
W = np.stack(W)
e_wrong = pos_err(W)
k_max = int(np.argmax(e_wrong))
say("wrong_end_err_m", float(e_wrong[-1]), "{:.2f}")
say("wrong_max_err_m", float(e_wrong.max()), "{:.2f}")
say("wrong_max_step", k_max)
say("wrong_max_time_s", k_max * DT, "{:.2f}")
say("wrong_max_true_pos", [round(float(X[k_max]), 2), round(float(Y[k_max]), 2)])
say("wrong_max_wrong_pos", [round(float(W[k_max, 0, 2]), 2), round(float(W[k_max, 1, 2]), 2)])
say("wrong_max_true_yaw_rad", float(TH[k_max]), "{:.4f}")
say("wrong_max_head_err_rad", float(head_err(W).max()), "{:.1e}")
say("wrong_err_step250_m", float(e_wrong[250]), "{:.3f}")
say("wrong_err_step50_m", float(e_wrong[50]), "{:.4f}")
FIRST_WRONG = int(np.argmax(e_wrong > 1.0))
say("wrong_first_step_over_1m", FIRST_WRONG)
say("wrong_first_step_over_1m_time_s", FIRST_WRONG * DT, "{:.2f}")
say("wrong_steps_over_100m", int((e_wrong > 100).sum()))
# the wrong-order point at step m = the path's mirror image (y -> -y) spun about the world origin by the heading so far
Theta = TH - TH[0]
mir = np.zeros((N + 1, 2))
acc = np.array([X[0], Y[0]])
mir[0] = acc
for k in range(N):
    acc = acc + rotation_matrix(-Theta[k + 1]) @ np.array([fwd[k], left[k]])
    mir[k + 1] = acc
spun = np.einsum("kij,kj->ki", np.stack([rotation_matrix(t) for t in Theta]), mir)
say("wrong_vs_spun_mirror_identity_max_diff_m", float(np.abs(spun - W[:, :2, 2]).max()), "{:.1e}")
mirror_true = np.stack([X - X[0], -(Y - Y[0])], 1) + np.array([X[0], Y[0]])
say("mirror_sum_vs_true_mirror_end_diff_m", float(np.hypot(*(mir[-1] - mirror_true[-1]))), "{:.3f}")
say("true_far_side_max_y_m", float(Y.max()), "{:.2f}")
say("true_max_x_m", float(X.max()), "{:.2f}")
say("lap_closure_true_end_minus_start_m", float(np.hypot(X[-1] - X[0], Y[-1] - Y[0])), "{:.3f}")

# ---------------- Drift: heading noise on every step ----------------
print("\n== Drift ==")
N_SEEDS = 500
SIGMAS = (1e-4, 1e-3)
say("n_seeds", N_SEEDS)
noise = np.stack([np.random.default_rng(s).standard_normal(N) for s in range(N_SEEDS)])  # (seeds, N)


def reckon_noisy(f, l, dt, theta0, x0, y0, n):
    """Vectorised dead_reckon of S_k @ Rot(n_k) across seeds: returns x, y, theta of shape (seeds, N+1)."""
    S, Nk = n.shape
    x, y, th = np.full(S, x0), np.full(S, y0), np.full(S, theta0)
    xs, ys, ts = [x.copy()], [y.copy()], [th.copy()]
    for k in range(Nk):
        c, s = np.cos(th), np.sin(th)
        x = x + f[k] * c - l[k] * s
        y = y + f[k] * s + l[k] * c
        th = th + dt[k] + n[:, k]
        xs.append(x); ys.append(y); ts.append(th)
    return np.array(xs).T, np.array(ys).T, np.array(ts).T


def rot(a):
    return pose_to_matrix((0.0, 0.0, a))


drift = {}
for sg in SIGMAS:
    xs, ys, ts = reckon_noisy(fwd, left, dth, TH[0], X[0], Y[0], sg * noise)
    # check seed 0 against dead_reckon with the noisy steps
    T0 = dead_reckon(Tw[0], [steps[k] @ rot(sg * noise[0, k]) for k in range(N)])
    chk = max(float(np.abs(T0[:, 0, 2] - xs[0]).max()), float(np.abs(T0[:, 1, 2] - ys[0]).max()))
    err = np.hypot(xs - X, ys - Y)  # (seeds, N+1)
    herr = wrap_angle(ts - TH)
    rms_t = np.sqrt((err ** 2).mean(0))
    tag = f"sigma_{sg:g}"
    say(f"{tag}_vectorised_vs_dead_reckon_max_diff_m", chk, "{:.1e}")
    say(f"{tag}_seed0_end_err_m", float(err[0, -1]), "{:.3f}")
    say(f"{tag}_mean_end_err_m", float(err[:, -1].mean()), "{:.3f}")
    say(f"{tag}_rms_end_err_m", float(rms_t[-1]), "{:.3f}")
    say(f"{tag}_median_end_err_m", float(np.median(err[:, -1])), "{:.3f}")
    say(f"{tag}_p90_end_err_m", float(np.percentile(err[:, -1], 90)), "{:.3f}")
    say(f"{tag}_max_end_err_m", float(err[:, -1].max()), "{:.3f}")
    say(f"{tag}_rms_err_peak_m", float(rms_t.max()), "{:.3f}")
    say(f"{tag}_rms_err_peak_step", int(np.argmax(rms_t)))
    for q in (250, 500, 889, 1000, 1250, 1500):
        say(f"{tag}_rms_err_step{q}_m", float(rms_t[q]), "{:.3f}")
    say(f"{tag}_worst_seed_worst_step_err_m", float(err.max()), "{:.3f}")
    say(f"{tag}_end_head_err_std_rad", float(herr[:, -1].std()), "{:.5f}")
    say(f"{tag}_sigma_sqrt_N_rad", sg * math.sqrt(N), "{:.5f}")
    say(f"{tag}_seeds0to19_max_end_err_m", float(err[:20, -1].max()), "{:.3f}")
    say(f"{tag}_seeds0to19_max_err_m", float(err[:20].max()), "{:.3f}")
    drift[sg] = dict(err=err, rms=rms_t, xs=xs, ys=ys)
say("drift_ratio_rms_end_1e-3_over_1e-4", float(drift[1e-3]["rms"][-1] / drift[1e-4]["rms"][-1]), "{:.3f}")

# figure data (drift_end, drift_growth): the 20 first seeds' rebuilt laps and errors per sigma, every step
np.savez_compressed(Path(__file__).with_name("drift_fig7.npz"),
                    x_true=X, y_true=Y,
                    **{f"x_{sg:g}": drift[sg]["xs"][:20] for sg in SIGMAS},
                    **{f"y_{sg:g}": drift[sg]["ys"][:20] for sg in SIGMAS},
                    **{f"err_{sg:g}": drift[sg]["err"][:20] for sg in SIGMAS},
                    **{f"rms_{sg:g}": drift[sg]["rms"] for sg in SIGMAS})

# many laps: does the end-of-lap error keep growing?  Drive the same 1777 steps L times, sigma 1e-4.
print("\n-- repeated laps, sigma 1e-4, N_SEEDS seeds --")
L_MAX, S_L = 16, N_SEEDS
f_r, l_r, d_r = np.tile(fwd, L_MAX), np.tile(left, L_MAX), np.tile(dth, L_MAX)
n_r = 1e-4 * np.stack([np.random.default_rng(s).standard_normal(N * L_MAX) for s in range(S_L)])
xs, ys, ts = reckon_noisy(f_r, l_r, d_r, TH[0], X[0], Y[0], n_r)
xc, yc, tc = reckon_noisy(f_r, l_r, d_r, TH[0], X[0], Y[0], np.zeros((1, N * L_MAX)))
LAPS = (1, 2, 4, 8, 16)
laps_rms = []
for L in LAPS:
    j = L * N
    e = np.hypot(xs[:, j] - xc[0, j], ys[:, j] - yc[0, j])
    laps_rms.append(float(np.sqrt(np.mean(e ** 2))))
    say(f"laps_{L}_rms_end_err_m", laps_rms[-1], "{:.3f}")
    say(f"laps_{L}_head_err_std_rad", float(np.std(ts[:, j] - tc[0, j])), "{:.5f}")
np.savez_compressed(Path(__file__).with_name("drift_laps.npz"), laps=np.array(LAPS), rms=np.array(laps_rms))  # figure data: drift_laps

# ---------------- Challenge: the N^{3/2} law ----------------
print("\n== Challenge: how drift grows ==")
tel_s = run(Car(), "straight")
Xs, Ys, THs = np.asarray(tel_s.x), np.asarray(tel_s.y), np.asarray(tel_s.yaw)
Ts_s = np.stack([pose_to_matrix(p) for p in zip(Xs, Ys, THs)])
st_s = np.stack([relative_pose(Ts_s[k], Ts_s[k + 1]) for k in range(len(Xs) - 1)])
Ns = len(Xs) - 1
f_s, l_s, d_s = st_s[:, 0, 2], st_s[:, 1, 2], np.arctan2(st_s[:, 1, 0], st_s[:, 0, 0])
say("straight_steps", Ns)
say("straight_mean_step_m", float(f_s.mean()), "{:.4f}")
ell = float(f_s.mean())
sg = 1e-4
n_s = sg * noise[:, :Ns]
xs_s, ys_s, ts_s = reckon_noisy(f_s, l_s, d_s, THs[0], Xs[0], Ys[0], n_s)
err_s = np.hypot(xs_s - Xs, ys_s - Ys)
rms_s = np.sqrt((err_s ** 2).mean(0))
NS = [10, 20, 50, 100, 200, 400, Ns]
for n_ in NS:
    law = sg * ell * n_ ** 1.5 / math.sqrt(3)
    exact = sg * ell * math.sqrt((n_ - 1) * n_ * (2 * n_ - 1) / 6)
    say(f"straight_N{n_}_rms_m", float(rms_s[n_]), "{:.5f}")
    say(f"straight_N{n_}_law_m", law, "{:.5f}")
    say(f"straight_N{n_}_exact_sum_m", exact, "{:.5f}")
lg = np.log(np.array(NS[2:], float))
slope = np.polyfit(lg, np.log(rms_s[NS[2:]]), 1)[0]
say("straight_loglog_slope_N50_to_end", float(slope), "{:.3f}")
hstd = ts_s[:, -1] - THs[-1]
say("straight_end_head_err_std_rad", float(np.std(hstd)), "{:.6f}")
say("straight_sigma_sqrt_N_rad", sg * math.sqrt(Ns), "{:.6f}")
# the same law at the gentle lap's length, against the gentle lap's measured spread
law_gentle = sg * float(fwd.mean()) * N ** 1.5 / math.sqrt(3)
say("straight_law_at_gentle_N_m", law_gentle, "{:.3f}")
say("gentle_rms_end_err_sigma_1e-4_m", float(drift[1e-4]["rms"][-1]), "{:.3f}")
# loop law: each heading error spins the rest of the path about the point where it happened
pts = np.stack([X, Y], 1)
lever = np.hypot(*(pts[-1] - pts[1:]).T)  # noise after step j (index j=0..N-1) pivots about point j+1
say("gentle_lever_law_end_rms_m", sg * float(np.sqrt((lever ** 2).sum())), "{:.3f}")
say("gentle_lever_rms_m", float(np.sqrt((lever ** 2).mean())), "{:.2f}")
say("straight_lever_rms_at_gentle_N_m", float(fwd.mean()) * math.sqrt((N - 1) * (2 * N - 1) / 6), "{:.2f}")
for n_ in NS[:-1]:
    say(f"gentle_N{n_}_rms_m", float(drift[1e-4]["rms"][n_]), "{:.5f}")
lever_t = np.array([sg * math.sqrt(((pts[m] - pts[1:m + 1]) ** 2).sum()) for m in range(N + 1)])
say("gentle_lever_law_vs_measured_max_rel_diff_after_step50", float(np.max(np.abs(lever_t[50:] - drift[1e-4]["rms"][50:]) / drift[1e-4]["rms"][50:])), "{:.3f}")
say("gentle_lever_law_peak_m", float(lever_t.max()), "{:.3f}")
say("gentle_lever_law_peak_step", int(np.argmax(lever_t)))
np.savez_compressed(Path(__file__).with_name("drift_fig9.npz"), N=np.arange(Ns + 1), rms_straight=rms_s,
                    rms_gentle=drift[1e-4]["rms"], lever_gentle=lever_t, ell=ell, sigma=sg)

# ---------------- The rear axle is a frame too ----------------
print("\n== The rear axle is a frame too ==")
rear = np.einsum("kij,j->ki", Tw @ T_EGO_REAR, np.array([0.0, 0.0, 1.0]))[:, :2]
by_hand = np.stack([X - WHEELBASE * np.cos(TH), Y - WHEELBASE * np.sin(TH)], 1)
say("rear_chain_vs_by_hand_lap_max_diff_m", float(np.abs(rear - by_hand).max()), "{:.1e}")
lat_f = np.asarray(tel.lat)
lat_r = np.array([road.ego_truth(a, b, c)[1] for (a, b), c in zip(rear, TH)])
d = lat_r - lat_f
say("front_mean_abs_e_m", float(np.abs(lat_f).mean()), "{:.3f}")
say("front_max_abs_e_m", float(np.abs(lat_f).max()), "{:.3f}")
say("rear_mean_abs_e_m", float(np.abs(lat_r).mean()), "{:.3f}")
say("rear_max_abs_e_m", float(np.abs(lat_r).max()), "{:.3f}")
say("rear_minus_front_max_abs_m", float(np.abs(d).max()), "{:.3f}")
say("rear_minus_front_max_abs_step", int(np.argmax(np.abs(d))))
say("rear_minus_front_at_max", float(d[int(np.argmax(np.abs(d)))]), "{:.3f}")
s_arr = np.array([road.ego_truth(a, b, c)[0] for a, b, c in zip(X, Y, TH)])
kappa_lane = 1 / LANE_R
bend1 = (s_arr > 100 + 60) & (s_arr < 100 + math.pi * GENTLE_RADIUS - 60)
say("bend1_steps", int(bend1.sum()))
say("bend1_mean_rear_minus_front_m", float(d[bend1].mean()), "{:.4f}")
say("bend1_mean_front_e_m", float(lat_f[bend1].mean()), "{:.4f}")
say("bend1_mean_rear_e_m", float(lat_r[bend1].mean()), "{:.4f}")
say("offtracking_L2_over_2R_m", WHEELBASE ** 2 / (2 * LANE_R), "{:.4f}")
R_ego = LANE_R
say("offtracking_R_minus_sqrt_m", R_ego - math.sqrt(R_ego ** 2 - WHEELBASE ** 2), "{:.4f}")
say("bend1_mean_head_err_rad", float(np.asarray(tel.head_err)[bend1].mean()), "{:.4f}")
say("bend1_mean_steer_rad", float(st[bend1].mean()), "{:.4f}")
say("straight_mid_mean_abs_rear_minus_front_m", float(np.abs(d[(s_arr > 20) & (s_arr < 80)]).mean()), "{:.4f}")
np.savez_compressed(Path(__file__).with_name("rear_fig8.npz"), x=X, y=Y, xr=rear[:, 0], yr=rear[:, 1],
                    lat_f=lat_f, lat_r=lat_r, s=s_arr)

# ---------------- Derived numbers the prose states ----------------
print("\n== Derived for prose ==")
say("example_time_s", K_EX * DT, "{:.2f}")
say("example_cos_sin", [round(math.cos(p_ex[2]), 6), round(math.sin(p_ex[2]), 6)])
say("example_rotated_part", np.round(rotation_matrix(p_ex[2]) @ p_ego, 4).tolist())
say("A_inverse_pose", [round(v, 6) for v in matrix_to_pose(invert_pose(A))])
say("A_inverse_times_A_max_abs_err", float(np.abs(invert_pose(A) @ A - np.eye(3)).max()), "{:.1e}")
say("full_turn_rad", 2 * math.pi, "{:.4f}")
say("full_turn_minus_sum_step_turn_rad", 2 * math.pi - float(dth.sum()), "{:.4f}")
say("first_straight_end_step_time_s", 250 * DT, "{:.2f}")
say("sigma_1e-4_in_degrees", math.degrees(1e-4), "{:.4f}")
say("sigma_1e-4_over_mean_step_turn", 1e-4 / float(dth.mean()), "{:.3f}")
say("bend1_mean_step_turn_rad", float(dth[bend1[:-1]].mean()), "{:.5f}")
say("laps_ratio_4_over_1", OUT["laps_4_rms_end_err_m"] / OUT["laps_1_rms_end_err_m"], "{:.2f}")
say("laps_ratio_16_over_4", OUT["laps_16_rms_end_err_m"] / OUT["laps_4_rms_end_err_m"], "{:.2f}")
R_front = LANE_R - float(lat_f[bend1].mean())
say("bend1_front_axle_radius_m", R_front, "{:.2f}")
say("offtracking_at_front_axle_radius_m", R_front - math.sqrt(R_front ** 2 - WHEELBASE ** 2), "{:.4f}")
say("rear_minus_front_max_time_s", int(np.argmax(np.abs(d))) * DT, "{:.2f}")
I3 = np.broadcast_to(np.eye(3), (N, 3, 3))
still = dead_reckon(Tw[0], I3)
e_still = pos_err(still)
say("do_nothing_end_err_m", float(e_still[-1]), "{:.3f}")
say("do_nothing_max_err_m", float(e_still.max()), "{:.2f}")
say("wrong_max_time_fraction_of_lap", OUT["wrong_max_step"] / N, "{:.3f}")
say("straight_law_vs_rms_N501_rel", OUT["straight_N501_rms_m"] / OUT["straight_N501_law_m"] - 1, "{:.3f}")
say("example_axis_through_T", np.round(T_ex @ np.array([1.0, 0.0, 0.0]), 4).tolist())
say("example_point_through_T", np.round(T_ex @ np.array([1.0, 0.0, 1.0]), 4).tolist())
say("cone_world_minus_car", np.round(via_matrix - np.array(p_ex[:2]), 4).tolist())
say("cone_back_to_ego", np.round(world_to_ego(p_ex, via_matrix), 6).tolist())
K_PI = int(np.argmax(TH > math.pi))
say("yaw_first_above_pi_step", K_PI)
say("yaw_first_above_pi_time_s", K_PI * DT, "{:.2f}")
# the raw read-back comparison fails whenever the yaw is above pi (a full turn apart), not only after one crossing
_above = TH > math.pi
_cross = np.nonzero(np.diff(_above.astype(int)))[0] + 1
_late = np.nonzero(np.abs(th_back - TH) > 1)[0]
say("yaw_pi_crossings", len(_cross))
say("yaw_pi_first_crossing_step", int(_cross[0]))
say("yaw_pi_last_crossing_step", int(_cross[-1]))
say("yaw_pi_last_crossing_time_s", int(_cross[-1]) * DT, "{:.2f}")
say("raw_compare_fail_steps", len(_late))
say("raw_compare_fail_equals_yaw_above_pi", bool(np.array_equal(_late, np.nonzero(_above)[0])))
say("readback_theta_at_pi_crossing", [round(float(th_back[K_PI - 1]), 4), round(float(th_back[K_PI]), 4)])
say("step1000_time_s", 1000 * DT, "{:.2f}")
say("rms_peak_time_s", OUT["sigma_0.0001_rms_err_peak_step"] * DT, "{:.2f}")
t100 = matrix_to_pose(steps[K_STEP])[2]
say("step100_half_turn_rad", 0.5 * t100, "{:.6f}")
say("step100_heading_times_step_m", float(TH[K_STEP]) * float(steps[K_STEP][0, 2]), "{:.5f}")
say("step100_left_from_steer_and_half_turn_m", float(steps[K_STEP][0, 2]) * (OUT["step100_steer_rad"] + 0.5 * t100), "{:.5f}")

# ---------------- Lab a: 40 seeded pairs vs an inline closed form ----------------
print("\n== Lab checks ==")
rng = np.random.default_rng(114)
worst = 0.0
for _ in range(40):
    xa, ya, ta, xb, yb, tb = rng.uniform([-50, -50, -math.pi, -50, -50, -math.pi], [50, 50, math.pi, 50, 50, math.pi])
    ca, sa = math.cos(ta), math.sin(ta)
    dx, dy = xb - xa, yb - ya
    want = pose_to_matrix((ca * dx + sa * dy, -sa * dx + ca * dy, tb - ta))
    worst = max(worst, float(np.abs(relative_pose(pose_to_matrix((xa, ya, ta)), pose_to_matrix((xb, yb, tb))) - want).max()))
say("lab_a_40_pairs_max_diff", worst, "{:.1e}")
say("lab_a_identity_relative", relative_pose(T_ex, T_ex).round(12).tolist())
pt = relative_pose(pose_to_matrix((3.0, 4.0, 0.0)), pose_to_matrix((3.0, 4.0, 0.5)))
say("lab_a_pure_turn_pose", [round(v, 12) for v in matrix_to_pose(pt)])
say("lab_a_named_pair_AB_vs_BA_relative", [[round(v, 6) for v in matrix_to_pose(relative_pose(A, AB))],
                                           [round(v, 6) for v in matrix_to_pose(relative_pose(B, BA))]])
say("lab_b_wrong_order_fails_per_step_tol", bool(e_wrong.max() > 1e-6))
say("lab_b_wrong_order_passes_end_only_at_2m", bool(e_wrong[-1] < 2.0))
say("lab_b_naive_end_err_m", float(e_naive[-1]), "{:.2f}")
cz, sz, cy, sy = math.cos(0.7), math.sin(0.7), math.cos(0.1), math.sin(0.1)
T4 = np.eye(4)
T4[:3, :3] = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
T4[:3, 3] = [1.2, 0.3, 1.5]
say("lab_invert_pose_4x4_roundtrip_max_abs_err", float(np.abs(invert_pose(T4) @ T4 - np.eye(4)).max()), "{:.1e}")

Path(__file__).with_name("measured.json").write_text(json.dumps(OUT, indent=1))
print("\nwrote measured.json")
