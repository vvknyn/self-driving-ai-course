"""Measurements for "The camera's point of view" (1.1.5). Every number in the lecture comes from this output.

Run from the course worktree with PYTHONPATH=<repo>; geom and the camera mount come from zero2fsd itself.
"""
from __future__ import annotations

import inspect
import math
import os
from pathlib import Path

import numpy as np

import zero2fsd

REPO = Path(os.environ.get("PYTHONPATH", ".").split(os.pathsep)[0]).resolve()
assert Path(zero2fsd.__file__).resolve().is_relative_to(REPO), (zero2fsd.__file__, REPO)

from zero2fsd.car import Car, run  # noqa: E402
from zero2fsd.car.provided import FAR_ROWS, NEAR_ROWS, estimate_lane, paint_masks  # noqa: E402
from zero2fsd.car.types import Observation  # noqa: E402
from zero2fsd.score import metrics  # noqa: E402
from zero2fsd.sim import camera as cam_mod  # noqa: E402
from zero2fsd.sim import BicycleState, make_scenario  # noqa: E402
from zero2fsd.sim.scenarios import TARGET_SPEED  # noqa: E402
from zero2fsd.sim.camera import Camera  # noqa: E402

OUT: dict = {}


def plain(v):
    """numpy values to plain Python, so the printout reads as plain Python."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    return v


def say(key: str, value) -> None:
    """Print one tagged measurement."""
    value = plain(value)
    OUT[key] = value
    print(f"{key}: {value}")


# ---- the course's own geom and renderer seams (the author's stand-ins are gone: U2g) -------------------------
from zero2fsd.geom import (CAM_FROM_EGO_AXES, T_ego_cam, invert_pose, pixel_ray, ray_to_ground, rot_x, rot_y, rot_z,  # noqa: E402
                           rotation_matrix, transform_directions, transform_points, world_to_ego)
from zero2fsd.sim.dynamics import DT  # noqa: E402

def MountCamera(T, **kw):
    """A camera that draws through the geom world_to_ego and projects through the 4x4 mount T."""
    return Camera(world_to_ego=world_to_ego, T_ego_cam=T, **kw)


def drive(road_name, camera_factory):
    """The provided car on one road; only the renderer's camera changes."""
    return run(Car(), road_name, camera=camera_factory())


def summary(tel):
    m = metrics(tel)
    return {"completed": bool(tel.completed), "steps": len(tel.t), "time_s": round(len(tel.t) * DT, 2),
            "progress_m": round(float(tel.progress_m[-1]), 1), "mean_abs_lat": round(m["mean_abs_lat"], 3),
            "max_abs_lat": round(m["max_abs_lat"], 3), "final_lat": round(float(tel.lat[-1]), 3)}


cam = Camera()
H, PHI = cam.mount_height, math.radians(inspect.signature(Camera.__init__).parameters["pitch_deg"].default)  # the spec mount, read from the simulator
say("M00 zero2fsd file", zero2fsd.__file__)
say("M01 f, cx, cy, height, pitch_rad", [round(cam.f, 6), cam.cx, cam.cy, H, round(PHI, 4)])
say("M01b cos(pitch), sin(pitch)", [round(math.cos(PHI), 4), round(math.sin(PHI), 4)])

# ---- A third axis -----------------------------------------------------------------------------------------
X, Y, Z = np.eye(3)
say("M02 forward x left (np.cross)", np.cross(X, Y))
say("M02b 2D cross of forward and left (z part)", X[0] * Y[1] - X[1] * Y[0])
say("M02c left x forward", np.cross(Y, X))

# ---- Turning about each axis --------------------------------------------------------------------------------
say("M03 rot_z(pi/2) @ x", np.round(rot_z(math.pi / 2) @ X, 12))
say("M03b rot_x(pi/2) @ y", np.round(rot_x(math.pi / 2) @ Y, 12))
say("M03c rot_y(pi/2) @ z", np.round(rot_y(math.pi / 2) @ Z, 12))
say("M03d rot_y(pi/2) @ x", np.round(rot_y(math.pi / 2) @ X, 12))
say("M04 rot_y(pitch) @ x (nose)", np.round(rot_y(PHI) @ X, 4))
say("M04b rot_y(-pitch) @ x (nose, flipped sign)", np.round(rot_y(-PHI) @ X, 4))
tel_g = run(Car(), "gentle")
yaw150 = float(tel_g.yaw[np.argmin(abs(tel_g.progress_m - 150.0))])
say("M05 yaw at s=150 on gentle", round(yaw150, 3))
say("M05b max |rot_z(yaw)[:2,:2] - rotation_matrix(yaw)|", float(abs(rot_z(yaw150)[:2, :2] - rotation_matrix(yaw150)).max()))
say("M05c rot_z(yaw) @ z", np.round(rot_z(yaw150) @ Z, 12))
say("M06 rot_y(pitch) matrix", np.round(rot_y(PHI), 4))
say("M06b rot_x(pitch) @ x, @ y", [np.round(rot_x(PHI) @ X, 4), np.round(rot_x(PHI) @ Y, 4)])

# ---- The camera's own axes ----------------------------------------------------------------------------------
B = CAM_FROM_EGO_AXES
say("M07 B @ forward, B @ left, B @ up", [B @ X, B @ Y, B @ Z])
say("M07b B.T columns (cam x, y, z in ego, level camera)", [B.T[:, 0], B.T[:, 1], B.T[:, 2]])
say("M07c B.T @ B", B.T @ B)
R_ec = rot_y(PHI) @ B.T
say("M08 R_ego<-cam = rot_y(pitch) @ B.T", np.round(R_ec, 4))
say("M08b columns: cam x, cam y, cam z in ego", [np.round(R_ec[:, k], 4) for k in range(3)])
R_wrong = B.T @ rot_y(PHI)
say("M08c wrong order B.T @ rot_y(pitch): cam z in ego", np.round(R_wrong[:, 2], 4))
say("M08d wrong order: cam x in ego", np.round(R_wrong[:, 0], 4))
say("M08e wrong order: angle of cam z from forward (rad), its left part sign",
    [round(math.atan2(R_wrong[1, 2], R_wrong[0, 2]), 4), float(np.sign(R_wrong[1, 2]))])

# ---- One 4x4 for the mount ----------------------------------------------------------------------------------
T = T_ego_cam(H, PHI)
T_ce = invert_pose(T)
say("M09 T_ego<-cam", np.round(T, 4))
say("M09b T_cam<-ego (closed-form inverse)", np.round(T_ce, 4))
say("M09c max |closed form - np.linalg.inv|", float(abs(T_ce - np.linalg.inv(T)).max()))
say("M09d max |T_cam_ego @ T_ego_cam - I|", float(abs(T_ce @ T - np.eye(4)).max()))
say("M09e lens in ego = T @ (0,0,0,1)", transform_points(T, [0, 0, 0]))
rng = np.random.default_rng(0)
F = rng.uniform(2.0, 60.0, 200)
L = rng.uniform(-8.0, 8.0, 200)
c = transform_points(T_ce, np.stack([F, L, np.zeros(200)], -1))
u4, v4 = cam.cx + cam.f * c[:, 0] / c[:, 2], cam.cy + cam.f * c[:, 1] / c[:, 2]
u0, v0 = cam.ground_to_pixel(F, L)
say("M10 seed 0, 200 ground points forward U(2,60) left U(-8,8): max |du|, |dv| px",
    [float(abs(u4 - u0).max()), float(abs(v4 - v0).max())])
for name, p in (("10 m ahead", (10.0, 0.0, 0.0)), ("ego origin", (0.0, 0.0, 0.0))):
    pc = transform_points(T_ce, p)
    say(f"M11 worked {name}: p_cam", np.round(pc, 4))
    if pc[2] > 0:
        say(f"M11b worked {name}: (u, v) via 4x4, via ground_to_pixel",
            [round(cam.cx + cam.f * pc[0] / pc[2], 4), round(cam.cy + cam.f * pc[1] / pc[2], 4),
             [round(float(x), 4) for x in cam.ground_to_pixel(p[0], p[1])]])
# a pinhole step ingredient: row of a point from its cam y and z
p10 = transform_points(T_ce, (10, 0, 0))
say("M11c 10 m ahead: y/z, f*y/z (rows below the middle row)", [round(p10[1] / p10[2], 5), round(cam.f * p10[1] / p10[2], 4)])
inside = (u0 >= -0.5) & (u0 <= 319.5) & (v0 >= -0.5) & (v0 <= 179.5)
say("M10b of those 200 ground points, count inside the 320x180 picture", int(inside.sum()))

# ---- Points move, directions only turn ----------------------------------------------------------------------
axis_ego = transform_directions(T, [0, 0, 1])
say("M12 optical axis in ego (w=0)", np.round(axis_ego, 4))
bug = transform_points(T, [0, 0, 1])
say("M12b optical axis in ego with w=1 (bug)", np.round(bug, 4))
say("M12c lens + optical axis as a point: angle above level (rad) of the w=1 'direction'", round(math.atan2(bug[2], bug[0]), 4))
fwd_cam = transform_directions(T_ce, [1, 0, 0])
fwd_bug = transform_points(T_ce, [1, 0, 0])
say("M13 car's forward direction in cam (w=0)", np.round(fwd_cam, 4))
say("M13b its pixel row via pinhole (w=0)", round(cam.cy + cam.f * fwd_cam[1] / fwd_cam[2], 2))
say("M13c horizon row from ground_to_pixel(1e9, 0)", round(float(cam.ground_to_pixel(1e9, 0.0)[1]), 2))
say("M13d forward direction in cam with w=1 (bug)", np.round(fwd_bug, 4))
say("M13e its pixel row via pinhole (w=1 bug)", round(cam.cy + cam.f * fwd_bug[1] / fwd_bug[2], 2))

# ---- Where a pixel's ray meets the road ---------------------------------------------------------------------
frame0 = cam.render(make_scenario("gentle").road, BicycleState(*make_scenario("gentle").road.ego_pose(0.0), TARGET_SPEED, 0.0))
_, white = paint_masks(frame0)
cols150 = np.nonzero(white[150])[0]
say("M14 start frame of gentle: white-paint columns in row 150 (first, last, centre)",
    [int(cols150.min()), int(cols150.max()), float(cols150.mean())])
u_w = float(cols150.mean())
for name, (u, v) in (("plan pixel", (159.85, 150.0)), ("white line", (u_w, 150.0))):
    ray = pixel_ray(cam, u, v)
    d = transform_directions(T, ray)
    lam = -T[2, 3] / d[2]
    g = ray_to_ground(T, ray)
    say(f"M15 {name} ({u}, {v}): ray_cam", np.round(ray, 4))
    say(f"M15b {name}: d_ego", np.round(d, 4))
    say(f"M15c {name}: lambda", round(float(lam), 4))
    say(f"M15d {name}: ground point (ray_to_ground)", np.round(g, 4))
    say(f"M15e {name}: pixel_to_ground", [round(float(x), 4) for x in cam.pixel_to_ground(u, v)])
    say(f"M15f {name}: lens-to-ground distance |lambda d| (m)", round(float(lam * np.linalg.norm(d)), 4))
rng = np.random.default_rng(1)
us = rng.uniform(-0.5, 319.5, 200)
vs = rng.uniform(76.0, 179.5, 200)
g = ray_to_ground(T, pixel_ray(cam, us, vs))
gf, gl = cam.pixel_to_ground(us, vs)
say("M16 seed 1, 200 pixels u U(-0.5,319.5) v U(76,179.5): max |forward diff|, |left diff| (m)",
    [float(abs(g[:, 0] - gf).max()), float(abs(g[:, 1] - gl).max())])
say("M16b those 200: largest z of the ground point", float(abs(g[:, 2]).max()))
vs_up = rng.uniform(0.0, 75.0, 200)
g_up = ray_to_ground(T, pixel_ray(cam, us, vs_up))
say("M16c seed 1, 200 pixels with v U(0,75): count nan", int(np.isnan(g_up[:, 0]).sum()))
vv, uu = np.mgrid[0:180, 0:320]
rays = pixel_ray(cam, uu, vv)
ok = ~np.isnan(ray_to_ground(T, rays)[..., 0])
say("M16d all 57600 pixels: count that land on the road, first landing row", [int(ok.sum()), int(vv[ok].min())])
d_bug = transform_points(T, rays)  # direction with w=1
say("M16e w=1 bug: count of the 57600 pixel rays pointing down (d_z < 0)", int((d_bug[..., 2] < 0).sum()))
say("M16g w=1 bug: the row a pixel ray would need to point down, (h - sin)/cos * f + cy; the bottom row's b = (179.5-cy)/f",
    [round(cam.cy + cam.f * (H - math.sin(PHI)) / math.cos(PHI), 1), round((179.5 - cam.cy) / cam.f, 4)])
ray150 = pixel_ray(cam, u_w, 150.0)
say("M16f w=1 bug, white-line pixel: d_ego", np.round(transform_points(T, ray150), 4))

# middle-column formula: forward = h / tan(pitch + arctan((v - cy)/f))
for v in (150.0, 115.0, 95.0):
    say(f"M17 row {v}: h/tan(pitch + arctan((v-cy)/f)), ray_to_ground forward",
        [round(H / math.tan(PHI + math.atan((v - cam.cy) / cam.f)), 4),
         round(float(ray_to_ground(T, pixel_ray(cam, cam.cx, v))[0]), 4)])

for v in (179.0, 150.0, 125.0, 115.0, 95.0):
    g_row = ray_to_ground(T, pixel_ray(cam, np.array([-0.5, 159.5, 319.5]), np.full(3, v)))
    say(f"M17b row {v}: ground point at the left edge, middle, right edge of the picture [forward, left]",
        [[round(float(a), 3), round(float(b), 3)] for a, b, _ in g_row])
say("M17c the lens axis (middle pixel) meets the road at forward (m)", round(float(ray_to_ground(T, pixel_ray(cam, cam.cx, cam.cy))[0]), 3))

# ---- A camera mounted wrong: rows -> distance at +5 and -5 degrees ----------------------------------------
T_up = T_ego_cam(H, -PHI)
for v in (179.0, 150.0, 125.0, 115.0, 104.0, 95.0):
    say(f"M18 row {v}: forward at pitch +5, at pitch -5",
        [round(float(ray_to_ground(T, pixel_ray(cam, cam.cx, v))[0]), 3),
         round(float(ray_to_ground(T_up, pixel_ray(cam, cam.cx, v))[0]), 3)])
say("M18b horizon row at pitch -5", round(float(Camera(pitch_deg=-5.0).ground_to_pixel(1e9, 0.0)[1]), 2))
say("M18c row 150 at pitch -5: angle below level (rad)", round(-PHI + math.atan((150 - cam.cy) / cam.f), 4))
for band, (a, b) in (("near", NEAR_ROWS), ("far", FAR_ROWS)):
    say(f"M19 {band} band rows {a}-{b - 1}: ground the box assumes (+5) / truly sees (-5) (m)",
        [[round(float(ray_to_ground(TT, pixel_ray(cam, cam.cx, r))[0]), 2) for r in (b - 1, a)] for TT in (T, T_up)])

row70 = cam.cy + cam.f * math.tan(math.atan(H / cam_mod.AHEAD) - (-PHI))
say("M19b pitch -5: row that sees 70 m ahead (the end of the drawn road); the forward distance the box assumes for it",
    [round(row70, 2), round(float(ray_to_ground(T, pixel_ray(cam, cam.cx, row70))[0]), 2)])

# ---- Static poses: what the box reads when the renderer's mount is wrong ------------------------------------
road_s = make_scenario("straight").road
for label, kw in (("spec", {}), ("pitch -5", {"pitch_deg": -5.0}), ("pitch 0", {"pitch_deg": 0.0}),
                  ("height 1.2", {"height_m": 1.2}), ("height 1.6", {"height_m": 1.6})):
    rows = []
    for e, psi in ((0.4, 0.0), (0.0, 0.05), (0.4, 0.05)):
        st = BicycleState(*road_s.ego_pose(20.0, e, psi), TARGET_SPEED, 0.0)
        est = estimate_lane(Observation(Camera(**kw).render(road_s, st), TARGET_SPEED, 0.0))
        rows.append([e, psi, round(est.offset_m, 4), round(est.heading_rad, 4)])
    say(f"M20 straight s=20, renderer {label}: [true e, true psi, box offset, box heading]", rows)

rng = np.random.default_rng(3)
Fp, Lp = rng.uniform(2.0, 60.0, 200), rng.uniform(-8.0, 8.0, 200)
for hh in (1.2, 1.6):
    bf, bl = cam.pixel_to_ground(*Camera(height_m=hh).ground_to_pixel(Fp, Lp))
    say(f"M20b camera truly at {hh} m, box assumes 1.4: seed 3, 200 road points, box forward/true and box left/true (min, max), 1.4/h",
        [round(float((bf / Fp).min()), 6), round(float((bf / Fp).max()), 6), round(float((bl / Lp).min()), 6), round(float((bl / Lp).max()), 6), round(H / hh, 6)])

# ---- Drives: the provided car, only the renderer's mount changes (same road, seed 0, 8 m/s) ----------------
DRIVES = (("pitch +5 (spec), 1.4 m", {}), ("pitch 0", {"pitch_deg": 0.0}), ("pitch -5", {"pitch_deg": -5.0}),
          ("height 1.2 m", {"height_m": 1.2}), ("height 1.6 m", {"height_m": 1.6}))
tels = {}
for label, kw in DRIVES:
    for road in ("straight", "gentle"):
        tels[label, road] = tel = drive(road, lambda kw=kw: Camera(**kw))
        say(f"M21 {label:>22} {road:>8}", summary(tel))
gentle_road = make_scenario("gentle").road
tel_up = tels["pitch -5", "gentle"]
_s_grid = np.arange(0.0, 150.0, 0.05)  # the bend starts where the centreline's heading first leaves its starting value
BEND_S = round(float(_s_grid[np.nonzero(abs(gentle_road.pose_at(_s_grid)[2] - gentle_road.pose_at(0.0)[2]) > 1e-9)[0][0]]), 1)
say("M22 gentle: first bend starts at s (m)", BEND_S)
k_dep = len(tel_up.t) - 1
for s_mark in (40.0, 60.0, 70.0, 75.0, 80.0, 84.0, 86.0):
    k = int(np.argmin(abs(tel_up.progress_m - s_mark)))
    say(f"M22b pitch -5 gentle at s={s_mark}: [lat, est_offset, head_err, est_heading, steer]",
        [round(float(tel_up.lat[k]), 3), round(float(tel_up.est_offset[k]), 3), round(float(tel_up.head_err[k]), 4),
         round(float(tel_up.est_heading[k]), 4), round(float(tel_up.steer[k]), 4)])
spec = tels["pitch +5 (spec), 1.4 m", "gentle"]
for s_mark in (60.0, 80.0, 86.0):
    k = int(np.argmin(abs(spec.progress_m - s_mark)))
    say(f"M22c spec gentle at s={s_mark}: [lat, est_offset, head_err, est_heading, steer]",
        [round(float(spec.lat[k]), 3), round(float(spec.est_offset[k]), 3), round(float(spec.head_err[k]), 4),
         round(float(spec.est_heading[k]), 4), round(float(spec.steer[k]), 4)])
dev = np.nonzero(np.abs(tel_up.lat) > 0.1)[0]
say("M22d pitch -5 gentle: first s where |lat| > 0.1 m, lat sign there", [round(float(tel_up.progress_m[dev[0]]), 1), float(np.sign(tel_up.lat[dev[0]]))])
dev2 = np.nonzero(np.abs(tel_up.steer) > 0.02)[0]
say("M22e pitch -5 gentle: first s where |steer| > 0.02 rad, its sign", [round(float(tel_up.progress_m[dev2[0]]), 1), float(np.sign(tel_up.steer[dev2[0]]))])
say("M22f pitch -5 gentle: departure s, side (lat sign), steps", [round(float(tel_up.progress_m[-1]), 1), float(np.sign(tel_up.lat[-1])), len(tel_up.t)])
say("M22g pitch -5 gentle: departure s minus bend start (m)", round(float(tel_up.progress_m[-1]) - BEND_S, 1))
st = tels["pitch -5", "straight"]
gap_h = np.nonzero(np.abs(tel_up.est_heading - tel_up.head_err) > 0.01)[0]
say("M22k pitch -5 gentle: first s where |box heading - true heading| > 0.01 rad", round(float(tel_up.progress_m[gap_h[0]]), 1))
gap_s = np.nonzero(np.abs(tels["pitch -5", "straight"].est_heading - tels["pitch -5", "straight"].head_err) > 0.01)[0]
say("M22l pitch -5 straight: count of steps where |box heading - true heading| > 0.01 rad (of all)", [len(gap_s), len(tels["pitch -5", "straight"].t)])
say("M22h pitch -5 straight: max |lat| over the run", round(float(np.abs(st.lat).max()), 4))
# where the far band truly sees, at the step the car first steers: the distance to the bend
say("M22i at that first steer, distance from the car to the bend start (m)", round(BEND_S - float(tel_up.progress_m[dev2[0]]), 1))
# what the box would read with the spec camera at the same poses (the car's believed lane, same pose)
for s_mark in (70.0, 80.0):
    k = int(np.argmin(abs(tel_up.progress_m - s_mark)))
    pose = BicycleState(tel_up.x[k], tel_up.y[k], tel_up.yaw[k], TARGET_SPEED, 0.0)
    e_up = estimate_lane(Observation(Camera(pitch_deg=-5.0).render(gentle_road, pose), TARGET_SPEED, 0.0))
    e_ok = estimate_lane(Observation(Camera().render(gentle_road, pose), TARGET_SPEED, 0.0))
    say(f"M22j same pose (pitch -5 run, s={s_mark}): box offset/heading with -5 renderer, with spec renderer",
        [round(e_up.offset_m, 3), round(e_up.heading_rad, 4), round(e_ok.offset_m, 3), round(e_ok.heading_rad, 4)])

gentle_s = []
for s_mark in (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0):
    stt = BicycleState(*gentle_road.ego_pose(s_mark, 0.0, 0.0), TARGET_SPEED, 0.0)
    e_up = estimate_lane(Observation(Camera(pitch_deg=-5.0).render(gentle_road, stt), TARGET_SPEED, 0.0))
    e_ok = estimate_lane(Observation(Camera().render(gentle_road, stt), TARGET_SPEED, 0.0))
    gentle_s.append([s_mark, round(e_up.offset_m, 3), round(e_up.heading_rad, 4), round(e_ok.offset_m, 3), round(e_ok.heading_rad, 4)])
say("M22m gentle, car on the lane centre pointing along it: [s, box offset, heading with -5 renderer; with spec renderer]", gentle_s)

# ---- The gate: frames drawn through world_to_ego (matrix form) and the 4×4 mount -----------------------------
sample = np.linspace(0, len(tel_g.t) - 1, 18).astype(int)
diff_frames = diff_px = 0
for k in sample:
    stt = BicycleState(tel_g.x[k], tel_g.y[k], tel_g.yaw[k], tel_g.speed[k], 0.0)
    a = cam.render(gentle_road, stt)
    b = MountCamera(T).render(gentle_road, stt)
    nd = int((a != b).any(axis=-1).sum())
    diff_frames += nd > 0
    diff_px += nd
say("M23 18 sampled gentle frames: frames differing, pixels differing (your world_to_ego + your 4x4)", [int(diff_frames), diff_px])
tel_mine = drive("gentle", lambda: MountCamera(T))
say("M23b gentle drive through your world_to_ego + 4x4", summary(tel_mine))
say("M23c same as the provided renderer, step by step (max |lat diff|)", float(np.abs(tel_mine.lat - tel_g.lat).max()))
say("M23d provided renderer gentle", summary(tel_g))
tel_flip = drive("gentle", lambda: MountCamera(T_ego_cam(H, -PHI)))
say("M23e gentle through your world_to_ego + a 4x4 with the pitch sign flipped", summary(tel_flip))
diff_up = 0
for k in sample:
    stt = BicycleState(tel_g.x[k], tel_g.y[k], tel_g.yaw[k], tel_g.speed[k], 0.0)
    diff_up += int((cam.render(gentle_road, stt) != MountCamera(T_ego_cam(H, -PHI)).render(gentle_road, stt)).any(axis=-1).any())
say("M23f 18 sampled frames: frames differing with the pitch sign flipped", diff_up)
# rot_x in place of rot_y: a camera rolled 5 degrees
T_roll = np.eye(4)
T_roll[:3, :3], T_roll[:3, 3] = rot_x(PHI) @ B.T, (0, 0, H)
say("M24 rot_x in place of rot_y: optical axis in ego", np.round(transform_directions(T_roll, [0, 0, 1]), 4))
say("M24b rot_x in place of rot_y: cam x (right) in ego", np.round(transform_directions(T_roll, [1, 0, 0]), 4))
tel_roll = drive("gentle", lambda: MountCamera(T_roll))
say("M24c gentle with rot_x in place of rot_y (camera rolled 5 deg, level)", summary(tel_roll))
tel_roll_s = drive("straight", lambda: MountCamera(T_roll))
say("M24d straight with rot_x in place of rot_y", summary(tel_roll_s))
tel_wrong = drive("straight", lambda: MountCamera(np.block([[R_wrong, np.array([[0], [0], [H]])], [np.zeros((1, 3)), np.ones((1, 1))]])))
say("M24e straight with B.T @ rot_y (wrong order: camera turned right, level)", summary(tel_wrong))

say("M25 provided car on gentle: speed at the start, min, max over the lap (m/s)",
    [round(float(tel_g.speed[0]), 3), round(float(tel_g.speed.min()), 3), round(float(tel_g.speed.max()), 3)])

# ---- Challenge: calibrate the mount from four painted marks --------------------------------------------------
MARKS = np.array([[3.0, 1.0], [3.0, -1.0], [8.0, 1.0], [8.0, -1.0]])
mu, mv = cam.ground_to_pixel(MARKS[:, 0], MARKS[:, 1])
say("M30 marks (forward, left) and their pixels (u, v)", [[*m, round(float(a), 3), round(float(b), 3)] for m, a, b in zip(MARKS, mu, mv)])


def implied_heights(pitch, rows, forward):
    """The height each mark needs for its row to see its distance: forward * tan(pitch + arctan((v - cy)/f))."""
    return forward * np.tan(pitch + np.arctan((rows - cam.cy) / cam.f))


def calibrate(rows, forward, lo=-0.3, hi=0.3, iters=60):
    """Bisection on the pitch that makes the near and far marks imply the same height; then that height."""
    near, far = forward == forward.min(), forward == forward.max()
    gap = lambda p: implied_heights(p, rows[far], forward[far]).mean() - implied_heights(p, rows[near], forward[near]).mean()
    assert gap(lo) < 0 < gap(hi), (gap(lo), gap(hi))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gap(mid) < 0 else (lo, mid)
    p = 0.5 * (lo + hi)
    return p, float(implied_heights(p, rows, forward).mean())


p_hat, h_hat = calibrate(mv, MARKS[:, 0])
say("M31 recovered pitch (rad), height (m) from exact pixels", [round(p_hat, 6), round(h_hat, 6)])
for trial in (0.05, 0.0873, 0.15):
    say(f"M31b gap(pitch) at {trial}: far implied h - near implied h", round(float(implied_heights(trial, mv[2:], MARKS[2:, 0]).mean() - implied_heights(trial, mv[:2], MARKS[:2, 0]).mean()), 4))
say("M31c implied heights of near and far marks at pitch 0", [round(float(implied_heights(0.0, mv[:1], MARKS[:1, 0])[0]), 4), round(float(implied_heights(0.0, mv[2:3], MARKS[2:3, 0])[0]), 4)])
p1, h1 = calibrate(mv + 1.0, MARKS[:, 0])
say("M32 every row +1 px: recovered pitch, height; change in pitch (rad)", [round(p1, 5), round(h1, 4), round(p1 - p_hat, 5)])
for v in (mv[0], mv[2], 150.0, 89.5):
    say(f"M32b row {round(float(v), 2)}: angle between this row's ray and the next row's (rad)", round(math.atan((v + 1 - cam.cy) / cam.f) - math.atan((v - cam.cy) / cam.f), 5))
say("M32c 1/f (rad)", round(1 / cam.f, 5))
pf, hf = calibrate(mv[[0, 2]] + np.array([0.0, 1.0]), MARKS[[0, 2], 0])
say("M32d only the far mark's row +1 px: recovered pitch, height", [round(pf, 5), round(hf, 4)])
spread = []
rng = np.random.default_rng(2)
for _ in range(500):
    spread.append(calibrate(mv + rng.normal(0.0, 0.5, 4), MARKS[:, 0]))
spread = np.array(spread)
say("M33 seed 2, 500 calibrations, row noise sigma 0.5 px: pitch mean, std; height mean, std",
    [round(float(spread[:, 0].mean()), 5), round(float(spread[:, 0].std()), 5), round(float(spread[:, 1].mean()), 4), round(float(spread[:, 1].std()), 4)])
# what a 1-row pitch error does to the box's far band
for dp in (p1 - p_hat,):
    say("M34 pitch off by that change: row 115 forward with the true pitch, with pitch + change",
        [round(H / math.tan(PHI + math.atan((115 - cam.cy) / cam.f)), 3), round(H / math.tan(PHI + dp + math.atan((115 - cam.cy) / cam.f)), 3)])

# lecture-support probes added after drafting (height similar triangles, angle sum, calibration gap, blind zone)
low = Camera(height_m=1.2)
beta150 = math.atan((150 - cam.cy) / cam.f)
say("M35 row 150 middle column: true forward with the camera at 1.2 m, box forward assuming 1.4 m, ratio",
    [round(float(low.pixel_to_ground(cam.cx, 150.0)[0]), 4), round(float(cam.pixel_to_ground(cam.cx, 150.0)[0]), 4),
     round(float(cam.pixel_to_ground(cam.cx, 150.0)[0] / low.pixel_to_ground(cam.cx, 150.0)[0]), 6)])
say("M36 row 150: beta = arctan b, pitch + beta, h / tan(pitch + beta)",
    [round(beta150, 4), round(PHI + beta150, 4), round(H / math.tan(PHI + beta150), 4)])
g0 = implied_heights(0.0, mv[2:3], MARKS[2:3, 0])[0] - implied_heights(0.0, mv[:1], MARKS[:1, 0])[0]
say("M37 gap at pitch 0 (far - near implied h); pitch std over one row's angle at the middle row 1/f",
    [round(float(g0), 4), round(float(spread[:, 0].std() * cam.f), 3)])
say("M38 bottom row 179.5 edge: middle-column forward distance (m); ego origin's row",
    [round(H / math.tan(PHI + math.atan((179.5 - cam.cy) / cam.f)), 3), round(float(cam.cy + cam.f * H * math.cos(PHI) / (H * math.sin(PHI))), 1)])

# preset margins for the lab: pitch_up.py (pitch sign flipped in T_ego_cam) and direction_with_w1.py (ray moved as a point)
r0 = np.random.default_rng(0)
Fp, Lp = r0.uniform(2.0, 60.0, 200), r0.uniform(-8.0, 8.0, 200)
for name, Tm in (("spec", T_ego_cam(H, PHI)), ("pitch_up", T_ego_cam(H, -PHI))):
    cm = transform_points(invert_pose(Tm), np.stack([Fp, Lp, np.zeros(200)], -1))
    uu, vv = cam.cx + cam.f * cm[:, 0] / cm[:, 2], cam.cy + cam.f * cm[:, 1] / cm[:, 2]
    ug, vg = cam.ground_to_pixel(Fp, Lp)
    row150 = ray_to_ground(Tm, pixel_ray(cam, cam.cx, 150.0))
    say(f"M39 {name} mount: 200 seed-0 points, max |du|, |dv| px vs ground_to_pixel; row 150 middle-column forward via ray_to_ground (m)",
        [round(float(abs(uu - ug).max()), 4), round(float(abs(vv - vg).max()), 4), round(float(row150[0]), 4)])
r1 = np.random.default_rng(1)
up, vp = r1.uniform(-0.5, 319.5, 200), r1.uniform(76.0, 179.5, 200)
def ray_to_ground_w1(Tm, ray_cam):
    c, d = Tm[:3, 3], transform_points(Tm, ray_cam)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(d[..., 2] < 0, -c[2] / d[..., 2], np.nan)
    return c + lam[..., None] * d
gw = ray_to_ground_w1(T_ego_cam(H, PHI), pixel_ray(cam, up, vp))
say("M40 direction_with_w1: of the 200 seed-1 pixels below the horizon, count nan", int(np.isnan(gw[:, 0]).sum()))
