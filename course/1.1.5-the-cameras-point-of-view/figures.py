"""Figures for lecture 1.1.5 The camera's point of view.  `python scripts/make_figures.py course/1.1.5-the-cameras-point-of-view`.

Phone-legible figures: every figure is 5 in wide, saved at the figstyle dpi, with its smallest text at FS = 14 pt (>= 2.7 x 5 in), so a digit is
at least 9 px tall when the PNG is shown 330 px wide on a phone.  Panels are stacked in one column.  Every number drawn here is
recomputed from the course's own geom and renderer, never typed in.
"""
import functools
import inspect
import math

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon, Rectangle

from zero2fsd.car import Car, run
from zero2fsd.car.provided import estimate_lane
from zero2fsd.car.types import Observation
from zero2fsd.geom import (CAM_FROM_EGO_AXES, T_ego_cam, invert_pose, pixel_ray, ray_to_ground, rot_x, rot_y, transform_directions,
                           transform_points, world_to_ego)
from zero2fsd.palette import CHROME, QUANTITY, shades
from zero2fsd.sim import BicycleState, Camera, make_scenario
from zero2fsd.sim.scenarios import TARGET_SPEED
from zero2fsd.sim.world import EGO_LANE_CENTER, LINE_HALF_WIDTH, WHITE_LINE_LATERAL

TEXT, PANEL, BACKGROUND, BORDER = CHROME["text"], CHROME["panel"], CHROME["background"], CHROME["border"]
EGO, CAM, THETA = QUANTITY["ego"], QUANTITY["camera"], QUANTITY["theta"]
DIM = shades(BORDER, 5)[3]  # a quieter grey from the chrome family: the things that are wrong, or not the point
LABEL_BOX = {"boxstyle": "round,pad=0.25", "fc": PANEL, "ec": BORDER}  # text that sits on top of a picture
W = 5.0  # every figure: 1000 px wide at the figstyle dpi, shown at 500 px on a desktop and 330 px on a phone
FS, TITLE = 14, 16  # smallest text and titles in points: a digit stays at least 9 px tall at 330 px wide
STYLE = {"font.size": FS, "axes.labelsize": FS, "axes.titlesize": TITLE, "axes.titlepad": 8, "xtick.labelsize": FS, "ytick.labelsize": FS}

CAMERA = Camera()
H = CAMERA.mount_height
PHI = math.radians(inspect.signature(Camera.__init__).parameters["pitch_deg"].default)  # the spec tilt, read from the simulator
T_SPEC = T_ego_cam(H, PHI)
BAND = (95, 115, 125, 179)  # the far rows and the near rows the box reads
VIEW_S = 20.0  # the straight-road picture of figures 9, 12 and 20: car on the lane centre, 20 m along
FIRST_BEND_ARRAY = np.arange(0.0, 150.0, 0.05)


def _fig(height):
    return Figure(figsize=(W, height))


def _save(fig, path):
    fig.savefig(path)


# ---- drawing helpers ------------------------------------------------------------------------------------------
def _arrow(ax, a, b, colour, lw=2.6, ls="-", style="-|>"):
    ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": style, "color": colour, "lw": lw, "ls": ls, "shrinkA": 0, "shrinkB": 0, "mutation_scale": 18})


def _arc(ax, centre, r, a0, a1, colour, lw=2.6, proj=None):
    """A curved arrow from angle a0 to a1 (degrees, counterclockwise positive) about `centre`; with `proj`, the arc lies in the
    ego ground plane (centre is an ego point) and is drawn through that projection."""
    t = np.radians(np.linspace(a0, a1, 40))
    if proj is None:
        x, y = centre[0] + r * np.cos(t), centre[1] + r * np.sin(t)
    else:
        x, y = np.array([proj((centre[0] + r * np.cos(a), centre[1] + r * np.sin(a), centre[2])) for a in t]).T
    ax.plot(x[:-1], y[:-1], color=colour, lw=lw, solid_capstyle="butt")
    _arrow(ax, (x[-2], y[-2]), (x[-1], y[-1]), colour, lw=lw)


def _say(ax, xy, text, colour=TEXT, ha="center", va="center", **kwargs):
    return ax.text(*xy, text, color=colour, ha=ha, va=va, **kwargs)


def _badge(ax, xy, text, colour=TEXT, ha="center", va="center"):
    return ax.text(*xy, text, color=colour, ha=ha, va=va, bbox=LABEL_BOX)


def _sketch(ax, xlim, ylim):
    ax.set(xlim=xlim, ylim=ylim, aspect="equal", xticks=[], yticks=[])  # no ticks: a hidden axis still has tick labels as text
    ax.axis("off")


def _out_of_page(ax, xy, colour, r=0.14):
    ax.add_patch(Circle(xy, r, fill=False, edgecolor=colour, lw=2.4))
    ax.add_patch(Circle(xy, r / 4, color=colour))


def _into_page(ax, xy, colour, r=0.14):
    ax.add_patch(Circle(xy, r, fill=False, edgecolor=colour, lw=2.4))
    d = r * 0.7
    ax.plot([xy[0] - d, xy[0] + d], [xy[1] - d, xy[1] + d], color=colour, lw=2.4)
    ax.plot([xy[0] - d, xy[0] + d], [xy[1] + d, xy[1] - d], color=colour, lw=2.4)


def _oblique(p):
    """A screen position for an ego-frame point: left runs to the page's left, up to the top, forward away up and to the right."""
    x, y, z = p
    return np.array([-y + 0.55 * x, z + 0.42 * x])


def _show(ax, picture, title=None):
    ax.imshow(picture, interpolation="nearest", aspect="equal")
    if title:
        ax.set_title(title)


def _pixel_ticks(ax, rows=(0, 179), cols=(0, 319)):
    ax.set_xticks(cols)
    ax.set_yticks(rows)
    ax.set(xlabel="column index", ylabel="row index")


# ---- the drives and renders the figures read -------------------------------------------------------------------
def mount_camera(T):
    """The renderer drawing through the geom world_to_ego and projecting through the 4x4 mount `T`."""
    return Camera(world_to_ego=world_to_ego, T_ego_cam=T)


@functools.cache
def road(name):
    return make_scenario(name).road


@functools.cache
def drive(name, pitch_deg=None, mount=None):
    """The provided car on one road; only the renderer's camera changes (`mount` is a name in MOUNTS)."""
    if mount is not None:
        camera = mount_camera(MOUNTS[mount]())
    else:
        camera = Camera() if pitch_deg is None else Camera(pitch_deg=pitch_deg)
    return run(Car(), name, camera=camera)


MOUNTS = {
    "flipped": lambda: T_ego_cam(H, -PHI),
    "rot_x": lambda: np.block([[rot_x(PHI) @ CAM_FROM_EGO_AXES.T, np.array([[0.0], [0.0], [H]])], [np.zeros((1, 3)), np.ones((1, 1))]]),
}


def bend_start():
    gentle = road("gentle")
    return round(float(FIRST_BEND_ARRAY[np.nonzero(abs(gentle.pose_at(FIRST_BEND_ARRAY)[2] - gentle.pose_at(0.0)[2]) > 1e-9)[0][0]]), 1)


def straight_picture():
    straight = road("straight")
    return CAMERA.render(straight, BicycleState(*straight.ego_pose(VIEW_S, 0.0, 0.0), TARGET_SPEED, 0.0))


# ---- figures, in lecture order ---------------------------------------------------------------------------------
def _result_first(path):
    tel = drive("gentle")
    gentle = road("gentle")
    sample = np.linspace(0, len(tel.t) - 1, 18).astype(int)
    mine = mount_camera(T_SPEC)
    differ = 0
    matching = 0
    for k in sample:
        pose = BicycleState(tel.x[k], tel.y[k], tel.yaw[k], tel.speed[k], 0.0)
        n = int((CAMERA.render(gentle, pose) != mine.render(gentle, pose)).any(axis=-1).sum())
        differ += n
        matching += n == 0
    k = sample[int(np.argmin(abs(tel.progress_m[sample] - 105.0)))]
    pose = BicycleState(tel.x[k], tel.y[k], tel.yaw[k], tel.speed[k], 0.0)
    fig = _fig(6.5)
    for ax, camera, title in zip(fig.subplots(2, 1), (CAMERA, mine), ("Simulator's renderer", "Your world_to_ego + T_ego_cam")):
        _show(ax, camera.render(gentle, pose), title)
        ax.set_xticks([])
        ax.set_yticks([])
    pixels = CAMERA.render(gentle, pose).shape[0] * CAMERA.render(gentle, pose).shape[1]
    fig.text(0.5, 0.012, f"{differ} of {pixels:,} pixels differ\n{matching} of {len(sample)} sampled pictures match", ha="center", va="bottom")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, path)


def _ego_frame_3d(path):
    fig = _fig(3.7)
    ax = fig.subplots()
    corners = [(-1.5, -2.2, 0), (2.7, -2.2, 0), (2.7, 2.2, 0), (-1.5, 2.2, 0)]
    ax.add_patch(Polygon([_oblique(c) for c in corners], closed=True, facecolor=BORDER, alpha=0.45, linewidth=0))
    origin = _oblique((0, 0, 0))
    for tip, name, at, ha in (((2.3, 0, 0), "x forward", (2.5, 0, 0.2), "left"), ((0, 2.0, 0), "y left", (0, 2.0, 0.28), "center"),
                              ((0, 0, 2.3), "z up", (0, 0, 2.55), "center")):
        _arrow(ax, origin, _oblique(tip), EGO)
        _say(ax, _oblique(at), name, EGO, ha=ha)
    lens = _oblique((0, 0, H))
    ax.add_patch(Circle(lens, 0.09, color=CAM))
    _say(ax, lens + np.array([-0.2, 0.0]), f"lens (0, 0, {H})", CAM, ha="right")
    _arc(ax, (0, 0, 0), 1.0, 8, 80, TEXT, proj=_oblique)
    _say(ax, _oblique((0.5, 0.0, 0)) + np.array([-0.6, -0.95]), "curl x → y:\nthumb = z", TEXT)
    _sketch(ax, (-3.2, 3.9), (-1.4, 3.0))
    _save(fig, path)


def _top_view_axes(ax, colour_x=EGO, colour_y=EGO):
    _arrow(ax, (0, 0), (0, 2.0), colour_x)
    _arrow(ax, (0, 0), (-2.0, 0), colour_y)


def _right_hand_turn(path):
    fig = _fig(3.4)
    ax = fig.subplots()
    _top_view_axes(ax)
    _say(ax, (0, 2.3), "x forward", EGO)
    _say(ax, (-1.0, -0.38), "y left", EGO)
    _out_of_page(ax, (0, 0), EGO, 0.17)
    _say(ax, (0.3, -0.38), "z up, out of page", EGO, ha="left")
    _arc(ax, (0, 0), 1.25, 90, 178, THETA)
    _say(ax, (-0.5, 1.75), "+θ (yaw):\na left turn", THETA, ha="right")
    _sketch(ax, (-2.6, 2.9), (-0.8, 2.7))
    _save(fig, path)


def _panel(ax, title, first, second, arrow, tip_a, tip_b, a0, a1, out_label, a_pos, b_pos, turn=TEXT):
    ax.set_title(title)
    _arrow(ax, (0, 0), tip_a, EGO)
    _arrow(ax, (0, 0), tip_b, EGO)
    _say(ax, a_pos, first, EGO)
    _say(ax, b_pos, second, EGO)
    _out_of_page(ax, (0, 0), EGO, 0.17)
    _say(ax, (0.28, -0.4), out_label, EGO, ha="left")
    _arc(ax, (0, 0), 1.05, a0, a1, turn)
    _say(ax, arrow[0], arrow[1], turn, ha=arrow[2])
    _sketch(ax, (-3.2, 3.2), (-1.7, 2.6))


def _three_rotations(path):
    fig = _fig(9.8)
    roll, pitch, yaw = fig.subplots(3, 1)
    _panel(roll, "About x: roll", "left", "up", ((1.25, 1.4), "left → up", "left"), (2.0, 0), (0, 2.0), 0, 88, "x, out of page", (2.55, 0), (0, 2.3))
    _panel(pitch, "About y: pitch", "forward", "up", ((-0.8, -1.3), "forward → down", "right"), (-2.0, 0), (0, 2.0), 180, 268, "y, out of page", (-2.65, 0), (0, 2.3))
    _panel(yaw, "About z: yaw", "left", "forward", ((-1.25, 1.4), "forward → left", "right"), (-2.0, 0), (0, 2.0), 90, 178, "z, out of page", (-2.55, 0), (0, 2.3), turn=THETA)  # THETA marks yaw only: roll and pitch turns are TEXT
    fig.tight_layout()
    _save(fig, path)


def _pitch_columns(path):
    show = math.radians(5 * 3)  # drawn 3x larger than the real 5 degrees
    fig = _fig(4.4)
    ax = fig.subplots()
    _arrow(ax, (0, 0), (3.3, 0), DIM, lw=1.8)
    _arrow(ax, (0, 0), (0, 2.6), DIM, lw=1.8)
    _arrow(ax, (0, 0), (3.2 * math.cos(show), -3.2 * math.sin(show)), CAM)
    _arrow(ax, (0, 0), (2.5 * math.sin(show), 2.5 * math.cos(show)), CAM)
    _say(ax, (0.2, -1.65), "forward →\n(0.9962, 0, −0.0872)", CAM, ha="left")
    _say(ax, (0.55, 2.8), "up → (0.0872, 0, 0.9962)", CAM, ha="left")
    _into_page(ax, (0, 0), TEXT, 0.17)
    _say(ax, (-0.45, -0.6), "y: into page,\nunchanged", TEXT, ha="right")
    _arc(ax, (0, 0), 2.1, 0, -15, TEXT, lw=2.0)
    _say(ax, (2.4, 0.75), "5°, drawn\n3× larger", TEXT, ha="left")
    _sketch(ax, (-2.8, 5.0), (-2.4, 3.3))
    _save(fig, path)


def _optical_vs_ego(path):
    fig = _fig(8.0)
    top, bottom = fig.subplots(2, 1)
    origin = np.zeros(2)
    top.set_title("The ego frame")
    for tip, name, at in (((2.0, 0, 0), "x forward", (2.45, 0, 0.1)), ((0, 2.0, 0), "y left", (0, 2.1, 0.3)), ((0, 0, 2.0), "z up", (0, 0, 2.3))):
        _arrow(top, origin, _oblique(tip), EGO)
        _say(top, _oblique(at), name, EGO, ha="left" if name.startswith("x") else "center")
    _sketch(top, (-3.3, 3.0), (-0.5, 2.9))
    bottom.set_title("The optical frame")
    for tip, name, at, ha in (((0, -2.0, 0), "x right", (0.1, -2.1, 0), "left"), ((0, 0, -2.0), "y down", (0, 0, -2.3), "center"),
                              ((2.0, 0, 0), "z out of\nthe lens", (2.0, -0.45, 0.6), "left")):
        _arrow(bottom, origin, _oblique(tip), CAM)
        _say(bottom, _oblique(at), name, CAM, ha=ha)
    plate = [_oblique(c) for c in ((1.4, 0.6, 0.4), (1.4, -0.6, 0.4), (1.4, -0.6, -0.4), (1.4, 0.6, -0.4))]
    bottom.add_patch(Polygon(plate, closed=True, fill=False, edgecolor=TEXT, lw=1.8, ls="--"))
    _say(bottom, (plate[1] + plate[0])[0:2] / 2 + np.array([0.0, 0.3]), "u →", TEXT)
    _say(bottom, plate[0] + np.array([-0.3, -0.35]), "v ↓", TEXT, ha="right")
    _sketch(bottom, (-3.3, 3.9), (-3.0, 1.9))
    fig.text(0.5, 0.01, "cam x = −ego y\ncam y = −ego z\ncam z = ego x", ha="center", va="bottom")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(fig, path)


def _wrong_order(path):
    show = math.radians(15)  # 5 degrees drawn 3x larger
    fig = _fig(7.0)
    top, side = fig.subplots(2, 1)
    top.set_title("Seen from above")
    _arrow(top, (0, 0), (0, 2.6), CAM)
    _arrow(top, (0, 0), (2.6 * math.sin(show), 2.6 * math.cos(show)), DIM, ls="--")
    _say(top, (-0.2, 3.0), "$R_y(\\varphi)\\,B^T$", CAM, ha="right")
    _say(top, (0.9, 3.0), "$B^T R_y(\\varphi)$", DIM, ha="left")
    _say(top, (0.6, 1.2), "5° right", DIM, ha="left")
    _sketch(top, (-2.6, 3.6), (-0.3, 3.4))
    side.set_title("Seen from the car's right side")
    _arrow(side, (0, 0), (3.0 * math.cos(show), -3.0 * math.sin(show)), CAM)
    _arrow(side, (0, 0), (3.0, 0), DIM, ls="--")
    _say(side, (3.0, 0.45), "$B^T R_y(\\varphi)$: level", DIM, ha="right")
    _say(side, (1.3, -1.45), "$R_y(\\varphi)\\,B^T$:\n5° down", CAM, ha="left")
    _sketch(side, (-0.5, 3.8), (-2.4, 1.1))
    fig.text(0.5, 0.01, "angles drawn 3× larger", ha="center", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, path)


def _mount_side(path):
    drawn = math.radians(24)  # not to scale: the real tilt is far too small to see
    lens = np.array([0.0, 2.2])
    ground_x = lens[1] / math.tan(drawn)
    axis = np.array([math.cos(drawn), -math.sin(drawn)])
    down = np.array([-math.sin(drawn), -math.cos(drawn)])
    fig = _fig(3.1)
    ax = fig.subplots()
    ax.plot([-1.0, ground_x + 1.0], [0, 0], color=DIM, lw=2.5)
    ax.plot([0, 0], [0.05, lens[1] - 0.12], color=TEXT, lw=1.5, ls=":")
    _say(ax, (-0.15, 0.5), f"h = {H} m", TEXT, ha="right")
    ax.add_patch(Circle(lens, 0.1, color=CAM))
    mid = lens + 0.45 * ground_x * axis
    ax.plot(*zip(lens, mid - 0.25 * axis), color=CAM, lw=2.2, ls="--")
    ax.plot(*zip(mid + 0.25 * axis, (ground_x, 0)), color=CAM, lw=2.2, ls="--")
    for d in (-0.32, 0.32):  # the break mark
        ax.plot(*zip(mid + d * axis + 0.25 * down, mid + d * axis - 0.25 * down), color=CAM, lw=2.2)
    _arrow(ax, lens, lens + 1.5 * axis, CAM)
    _arrow(ax, lens, lens + 0.9 * down, CAM)
    _say(ax, (0.9, 2.6), "cam z", CAM, ha="left")
    _say(ax, lens + 0.9 * down + np.array([-0.12, 0.0]), "cam y", CAM, ha="right")
    ax.plot([0, 3.0], [lens[1], lens[1]], color=DIM, lw=1.6, ls=":")
    _arc(ax, lens, 2.6, 0, -math.degrees(drawn), TEXT, lw=2.0)
    _say(ax, lens + np.array([2.95, -0.5]), f"{round(math.degrees(PHI))}°", TEXT, ha="left")
    _arrow(ax, (0, 0), (1.3, 0), EGO)
    _arrow(ax, (0, 0), (0, 1.0), EGO)
    _say(ax, (1.45, 0.0), "ego x", EGO, ha="left")
    _say(ax, (0.12, 0.8), "ego z", EGO, ha="left")
    forward = float(ray_to_ground(T_SPEC, pixel_ray(CAMERA, CAMERA.cx, CAMERA.cy))[0])
    _say(ax, (ground_x, -0.38), f"{forward:.1f} m", TEXT, va="top")
    _say(ax, (ground_x, 2.9), "not to scale", DIM, ha="right")
    _sketch(ax, (-2.4, ground_x + 1.0), (-1.0, 3.2))
    _save(fig, path)


def _projected_points(path):
    rng = np.random.default_rng(0)
    forward, left = rng.uniform(2.0, 60.0, 200), rng.uniform(-8.0, 8.0, 200)
    pixel_u, pixel_v = CAMERA.ground_to_pixel(forward, left)
    cam = transform_points(invert_pose(T_SPEC), np.stack([forward, left, np.zeros(200)], -1))
    u, v = CAMERA.cx + CAMERA.f * cam[:, 0] / cam[:, 2], CAMERA.cy + CAMERA.f * cam[:, 1] / cam[:, 2]
    inside = (pixel_u >= -0.5) & (pixel_u <= 319.5) & (pixel_v >= -0.5) & (pixel_v <= 179.5)
    fig = _fig(3.9)
    ax = fig.subplots()
    _show(ax, straight_picture())
    ax.plot(pixel_u[inside], pixel_v[inside], "o", ms=9.5, fillstyle="none", mec=TEXT, mew=1.6, zorder=3)
    ax.scatter(u[inside], v[inside], s=14, color=CAM, zorder=4)
    ax.set(xlim=(-0.5, 319.5), ylim=(179.5, -0.5))
    ax.set_xticks([])
    ax.set_yticks([])
    biggest = max(abs(u - pixel_u).max(), abs(v - pixel_v).max())
    fig.text(0.5, 0.012, f"{int(inside.sum())} of {len(inside)} in view\nlargest difference {biggest:.1f} px", ha="center", va="bottom")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(fig, path)


def _blind_zone(path):
    near = float(CAMERA.pixel_to_ground(CAMERA.cx, 179)[0])
    middle = float(ray_to_ground(T_SPEC, pixel_ray(CAMERA, CAMERA.cx, CAMERA.cy))[0])
    origin_row = CAMERA.ground_to_pixel(0.0, 0.0)[1]
    lens = np.array([0.0, 2.4])
    near_x, mid_x = 3.2, 8.0  # not to scale
    fig = _fig(3.9)
    ax = fig.subplots()
    ax.add_patch(Rectangle((0, -0.12), near_x, 0.12, color=DIM))
    ax.plot([near_x, mid_x + 1.0], [0, 0], color=DIM, lw=2.5)
    ax.add_patch(Circle(lens, 0.1, color=CAM))
    _arrow(ax, lens, (near_x, 0), CAM)
    _arrow(ax, lens, (mid_x, 0), CAM, ls="--")
    _say(ax, (1.0, 0.4), "unseen", TEXT)
    _say(ax, (near_x, -0.45), f"bottom row\n→ {near:.2f} m", CAM, va="top")
    _say(ax, (mid_x, -0.45), f"lens axis →\n{middle:.1f} m", CAM, va="top")
    _say(ax, (-0.4, -2.6), f"ego origin: row {origin_row:.0f}\n(picture ends at 179)", TEXT, ha="left", va="top")
    _arrow(ax, (0, -2.45), (0, -0.2), TEXT, lw=1.8)
    _say(ax, (mid_x, 3.0), "not to scale", DIM, ha="right")
    _sketch(ax, (-0.6, mid_x + 1.2), (-3.9, 3.4))
    _save(fig, path)


def _point_vs_direction(path):
    axis = transform_directions(T_SPEC, [0, 0, 1])
    point = transform_points(T_SPEC, [0, 0, 1])  # one metre out along the lens axis, read as a place
    lens = np.array([0.0, H])
    wrong = np.array([axis[0], point[2]])
    angle = math.atan2(wrong[1], wrong[0])
    road_x = H / math.tan(PHI)
    fig = _fig(2.8)
    ax = fig.subplots()
    ax.plot([-1.0, road_x + 1.0], [0, 0], color=DIM, lw=2.5)
    ax.add_patch(Circle(lens, 0.17, color=CAM))
    _arrow(ax, lens, (road_x, 0), CAM)
    ax.add_patch(Circle((point[0], point[2]), 0.2, color=CAM, zorder=5))
    _arrow(ax, lens, lens + 3.7 * wrong / np.linalg.norm(wrong), DIM, ls="--")
    _say(ax, (9.0, -0.45), "direction, w = 0: 5° down", CAM, va="top")
    _badge(ax, (4.0, 2.9), "point, w = 1: 1 m out\nalong the axis", CAM, ha="left")
    ax.annotate("", xy=(point[0] + 0.2, point[2] + 0.12), xytext=(3.9, 2.4), arrowprops={"arrowstyle": "-", "color": CAM, "lw": 1.5, "shrinkA": 0, "shrinkB": 0})
    _say(ax, (2.5, 5.5), f"read as a direction:\n{angle:.2f} rad up", DIM, ha="left", va="center")
    _sketch(ax, (-1.0, road_x + 1.0), (-1.3, 6.2))
    _save(fig, path)


def _horizon_w(path):
    horizon = float(CAMERA.cy + CAMERA.f * math.tan(-PHI))  # row of the car's forward direction with w = 0
    cam = transform_directions(invert_pose(T_SPEC), [1.0, 0.0, 0.0])
    wrong = invert_pose(T_SPEC) @ np.array([1.0, 0.0, 0.0, 1.0])  # forward with w = 1: a place, not a direction
    bug_row = CAMERA.cy + CAMERA.f * wrong[1] / wrong[2]
    fig = _fig(4.2)
    ax = fig.subplots()
    _show(ax, straight_picture())
    ax.axhline(horizon, color=TEXT, ls="--", lw=2.0)
    _badge(ax, (160, horizon - 18), f"w = 0: horizon, row {horizon:.1f}")
    _arrow(ax, (160, 179), (160, 232), CAM, lw=3.0)
    _badge(ax, (175, 205), f"w = 1: row {bug_row:.1f}", CAM, ha="left")
    ax.set(xlim=(-0.5, 319.5), ylim=(238, -0.5))
    ax.set_xticks([])
    ax.set_yticks([])
    assert cam[2] > 0
    _save(fig, path)


def _pixel_ray(path):
    u_w, v_w = 254.5, 150.0
    ground_w = ray_to_ground(T_SPEC, pixel_ray(CAMERA, u_w, v_w))
    ground_c = ray_to_ground(T_SPEC, pixel_ray(CAMERA, 159.85, v_w))
    lam = H / -transform_directions(T_SPEC, pixel_ray(CAMERA, u_w, v_w))[2]
    fig = _fig(8.6)
    side, top = fig.subplots(2, 1)
    side.set_title("Seen from the car's right side")
    lens = np.array([0.0, H])
    side.plot([-0.4, 4.0], [0, 0], color=DIM, lw=2.5)
    side.add_patch(Circle(lens, 0.06, color=CAM))
    _arrow(side, lens, (ground_w[0], 0), CAM)
    _say(side, (-0.12, H / 2), f"{H} m", TEXT, ha="right")
    side.plot([0, 0], [0.04, H - 0.08], color=TEXT, lw=1.4, ls=":")
    _say(side, (1.55, 1.12), f"λ = {lam:.3f}", CAM, ha="left")
    _say(side, (ground_w[0], -0.12), f"{ground_w[0]:.3f} m", TEXT, va="top")
    _sketch(side, (-1.1, 4.2), (-0.7, 1.9))
    top.set_title("Seen from above")
    flip = lambda p: (-p[1], p[0])  # forward up the page, the car's left toward the page's left
    right = EGO_LANE_CENTER - WHITE_LINE_LATERAL  # the white line, this far to the car's right of the lane centre (m)
    top.add_patch(Rectangle((right - LINE_HALF_WIDTH, 0), 2 * LINE_HALF_WIDTH, 4.0, color=TEXT, alpha=0.85, lw=0))
    _arrow(top, (0, 0), flip(ground_w), CAM)
    _arrow(top, (0, 0), flip(ground_c), DIM)
    _say(top, (right, -0.3), "white line", TEXT, va="top")
    _say(top, (0.8, 4.3), f"({u_w}, {v_w:.0f}) →\n({ground_w[0]:.4f}, {ground_w[1]:.4f})".replace("-", "−"), CAM)
    _say(top, (-0.2, 2.0), f"(159.85, {v_w:.0f}) →\n({ground_c[0]:.4f}, {ground_c[1]:.4f})".replace("-", "−"), DIM, ha="right")
    _sketch(top, (-2.6, 2.9), (-0.8, 5.0))
    fig.tight_layout()
    _save(fig, path)


def _rows_across(path):
    rows = (179, 150, 125, 115, 95)
    fig = _fig(5.6)
    ax = fig.subplots()
    ax.add_patch(Rectangle((-0.9, -2.7), 1.8, 2.7, fill=False, edgecolor=EGO, lw=2.2))
    seen = []
    for row in rows:
        ahead, side = (float(CAMERA.pixel_to_ground(-0.5, row)[0]), float(CAMERA.pixel_to_ground(-0.5, row)[1]))
        seen.append(ahead)
        ax.plot([-side, side], [ahead, ahead], color=CAM, lw=3.0)
        if row == 95:  # the segment is wide enough to hold both labels
            _say(ax, (-side + 0.4, ahead - 0.9), f"±{side:.2f} m", CAM, ha="left")
            _say(ax, (side - 0.4, ahead - 0.9), f"row {row}", CAM, ha="right")
        else:
            _say(ax, (side + 0.5, ahead), f"row {row}", CAM, ha="left")
            _say(ax, (-side - 0.5, ahead), f"±{side:.2f} m", CAM, ha="right")
    _say(ax, (1.2, -1.35), "car", EGO, ha="left")
    ax.set(xlim=(-12.8, 12.8), ylim=(-3.2, 12.8), xlabel="metres to each side of the car", ylabel="forward distance seen (m)")
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_xticklabels(["10", "5", "0", "5", "10"])
    ax.set_yticks(seen)
    ax.set_yticklabels([f"{a:.2f}" for a in seen])
    fig.tight_layout()
    _save(fig, path)


def _forward_at(pitch_rad, rows):
    T = T_ego_cam(H, pitch_rad)
    return np.array([ray_to_ground(T, pixel_ray(CAMERA, CAMERA.cx, v))[0] for v in rows])


YMAX_ROWS = 75


def _rows_pitch(path):
    rows = np.linspace(95, 179, 400)
    spec, up = _forward_at(PHI, rows), _forward_at(-PHI, rows)
    horizon_up = CAMERA.cy + CAMERA.f * math.tan(PHI)  # where the tipped-up camera's horizon sits
    fig = _fig(5.2)
    ax = fig.subplots()
    ax.axvspan(95, 115, color=BORDER, alpha=0.55, lw=0)
    ax.axvspan(125, 179, color=BORDER, alpha=0.3, lw=0)
    ax.plot(rows, np.where(spec <= YMAX_ROWS, spec, np.nan), color=CAM, lw=2.8)  # the part past the top of the axes is not drawn
    ax.plot(rows, np.where(up <= YMAX_ROWS, up, np.nan), color=CAM, lw=2.8, ls="--")
    ax.axhline(70, color=TEXT, ls=":", lw=1.8)
    ax.axvline(horizon_up, color=TEXT, ls=":", lw=1.8)
    row = 150
    spec150, up150 = _forward_at(PHI, [row])[0], _forward_at(-PHI, [row])[0]
    ax.scatter([row, row], [spec150, up150], color=CAM, zorder=5)
    _badge(ax, (row + 3, 25), f"row {row}:\n{spec150:.3f} / {up150:.3f} m", CAM, ha="right")
    _say(ax, (176, 14), "tipped down 5° (spec)", CAM, ha="right", va="bottom")
    _say(ax, (118, 34), "tipped up 5°", CAM, ha="left")
    _say(ax, (178, 72), "drawn road ends: 70 m", TEXT, ha="right", va="bottom")
    _say(ax, (109.5, 52), f"horizon when\ntipped up:\nrow {horizon_up:.1f}", TEXT, ha="left")
    _badge(ax, (105, 72.5), "far band", TEXT, va="top")
    _say(ax, (152, 62), "near band", TEXT, ha="center")
    ax.set(xlim=(95, 179), ylim=(0, YMAX_ROWS), xlabel="row index (middle column)", ylabel="forward distance seen (m)")
    fig.tight_layout()
    _save(fig, path)


def _plot_to(ax, tel, xmax, **kwargs):
    """The run's lateral error against distance, only up to the right edge of the axes."""
    keep = tel.progress_m <= xmax
    ax.plot(tel.progress_m[keep], tel.lat[keep], **kwargs)


def _lateral_runs(ax, name, title, xmax):
    ax.set_title(title)
    ax.axhline(0, color=BORDER, lw=1.5)
    for pitch, colour, ls, label, at in ((None, CAM, "-", "spec", 0.95), (0.0, DIM, ":", "level", 0.6), (-5.0, CAM, "--", "tipped up", 0.3)):
        tel = drive(name, pitch)
        _plot_to(ax, tel, xmax, color=colour, ls=ls, lw=2.6)
        if not tel.completed:
            ax.scatter([tel.progress_m[-1]], [tel.lat[-1]], marker="x", s=110, color=colour, linewidths=3.0, zorder=5)
    return ax


GENTLE_XMAX = 260


def _lat_pitch(path):
    fig = _fig(7.8)
    straight, gentle = fig.subplots(2, 1)
    _lateral_runs(straight, "straight", "straight", 1e9)
    _lateral_runs(gentle, "gentle", "gentle", GENTLE_XMAX)
    flat_end = drive("straight").progress_m[-1]
    straight.set_ylim(-0.3, 0.3)
    _say(straight, (flat_end * 0.5, 0.17), "all three finish,\nwithin 0.03 m of the centre", TEXT)
    up, level = drive("gentle", -5.0), drive("gentle", 0.0)
    gentle.axvline(bend_start(), color=TEXT, ls=":", lw=1.8)
    _say(gentle, (bend_start() + 8, -0.9), f"first bend: {bend_start():.0f} m", TEXT, ha="left")
    _badge(gentle, (up.progress_m[-1] - 12, 2.1), f"tipped up:\n{up.progress_m[-1]:.1f} m", CAM, ha="right", va="top")
    _badge(gentle, (level.progress_m[-1] + 12, 2.1), f"level, leaves\nat {level.progress_m[-1]:.1f} m", DIM, ha="left", va="top")
    _say(gentle, (200, 0.85), "spec", CAM)
    gentle.set_ylim(-1.2, 2.2)
    gentle.set_xlim(0, GENTLE_XMAX)
    for ax in (straight, gentle):
        ax.set_ylabel("distance from lane centre (m)")
    gentle.set_xlabel("distance along the road (m)")
    straight.set_xlabel("distance along the road (m)")
    fig.tight_layout()
    _save(fig, path)


def _box_heading(pitch_deg, s):
    camera = Camera() if pitch_deg is None else Camera(pitch_deg=pitch_deg)
    gentle = road("gentle")
    state = BicycleState(*gentle.ego_pose(s, 0.0, 0.0), TARGET_SPEED, 0.0)
    return estimate_lane(Observation(camera.render(gentle, state), TARGET_SPEED, 0.0)).heading_rad


def _heading_pitch(path):
    s = np.arange(20.0, 90.1, 5.0)
    spec = np.array([_box_heading(None, k) for k in s])
    up = np.array([_box_heading(-5.0, k) for k in s])
    fig = _fig(4.4)
    ax = fig.subplots()
    ax.axhline(0, color=BORDER, lw=1.5)
    ax.axvline(30, color=TEXT, ls=":", lw=1.8)
    ax.plot(s, spec, color=CAM, lw=2.8)
    ax.plot(s, up, color=CAM, lw=2.8, ls="--")
    leader = {"arrowstyle": "-", "color": CAM, "lw": 1.4, "shrinkA": 0, "shrinkB": 4, "relpos": (1.0, 1.0)}
    for mark, at in ((60.0, (60, -0.11)), (70.0, (70, -0.19)), (90.0, (90, -0.33))):
        value = up[list(s).index(mark)]
        ax.scatter([mark], [value], color=CAM, zorder=5)
        ax.annotate(f"{value:.3f} ({mark:.0f} m)".replace("-", "−"), xy=(mark, value), xytext=at, color=CAM, ha="right", va="top", arrowprops=leader)
    _say(ax, (31, 0.06), "bend in view from 30 m", TEXT, ha="left")
    _say(ax, (89, 0.025), "spec", CAM, ha="right", va="bottom")
    _say(ax, (22, -0.045), "tipped up 5°", CAM, ha="left", va="top")
    ax.set(xlim=(20, 92), ylim=(-0.4, 0.1), xticks=range(30, 91, 10), xlabel="distance along the road (m)", ylabel="heading the box reports (rad)")
    fig.tight_layout()
    _save(fig, path)


GATE_XMAX = 330


def _gate_breaks(path):
    fig = _fig(5.2)
    ax = fig.subplots()
    ax.axhline(0, color=BORDER, lw=1.5)
    spec, flipped, rolled = drive("gentle"), drive("gentle", mount="flipped"), drive("gentle", mount="rot_x")
    _plot_to(ax, spec, GATE_XMAX, color=CAM, lw=2.6)
    for tel, colour, ls in ((flipped, CAM, "--"), (rolled, DIM, ":")):
        _plot_to(ax, tel, GATE_XMAX, color=colour, ls=ls, lw=2.6)
        ax.scatter([tel.progress_m[-1]], [tel.lat[-1]], marker="x", s=110, color=colour, linewidths=3.0, zorder=5)
    mean = float(np.abs(spec.lat).mean())
    _say(ax, (325, 1.0), f"your T_ego_cam:\nmean {mean:.3f} m", CAM, ha="right")
    _badge(ax, (flipped.progress_m[-1] + 3, 3.0), f"φ flipped:\nleaves at\n{flipped.progress_m[-1]:.1f} m", CAM, ha="right", va="top")
    _badge(ax, (rolled.progress_m[-1] + 8, 3.0), f"rot_x for rot_y:\nleaves at\n{rolled.progress_m[-1]:.1f} m", DIM, ha="left", va="top")
    ax.set(xlim=(0, GATE_XMAX), ylim=(-0.6, 3.0), xlabel="distance along the road (m)", ylabel="distance from lane centre (m)")
    fig.tight_layout()
    _save(fig, path)


def _height_scale(path):
    true_h = 1.2
    beta = math.atan((150 - CAMERA.cy) / CAMERA.f)
    below = PHI + beta  # the ray's angle below level, from the camera's tilt and the row
    near, far = true_h / math.tan(below), H / math.tan(below)
    fig = _fig(3.9)
    ax = fig.subplots()
    ax.plot([-0.5, far + 0.7], [0, 0], color=DIM, lw=2.5)
    ax.add_patch(Polygon([(0, 0), (far, 0), (0, H)], closed=True, facecolor=CAM, alpha=0.13, linewidth=0))
    ax.add_patch(Polygon([(0, 0), (near, 0), (0, true_h)], closed=True, facecolor=CAM, alpha=0.25, linewidth=0))
    _arrow(ax, (0, true_h), (near, 0), CAM)
    _arrow(ax, (0, H), (far, 0), CAM, ls="--")
    ax.add_patch(Circle((0, true_h), 0.06, color=CAM))
    ax.add_patch(Circle((0, H), 0.06, color=CAM, fill=False, lw=2.0))
    _say(ax, (-0.1, true_h - 0.18), f"true lens {true_h} m", CAM, ha="right", va="top")
    _say(ax, (-0.1, H + 0.12), f"assumed {H} m", CAM, ha="right", va="bottom")
    _say(ax, (near - 0.05, -0.2), f"{near:.3f} m", CAM, ha="right", va="top")
    _say(ax, (far + 0.05, -0.2), f"{far:.3f} m", CAM, ha="left", va="top")
    _arc(ax, (near, 0), 0.65, 180, 180 - math.degrees(below), TEXT, lw=2.0)
    _say(ax, (near - 0.75, 0.37), f"{below:.4f} rad", TEXT, ha="right")
    fig.text(0.5, 0.012, f"{far:.3f} / {near:.3f} = 7/6", ha="center", va="bottom")
    _sketch(ax, (-2.0, far + 1.5), (-0.8, 1.9))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, path)


MARKS = np.array([[3.0, 1.0], [3.0, -1.0], [8.0, 1.0], [8.0, -1.0]])  # (forward, left) of the four painted marks


def _calibration_marks(path):
    mu, mv = CAMERA.ground_to_pixel(MARKS[:, 0], MARKS[:, 1])
    fig = _fig(3.9)
    ax = fig.subplots()
    _show(ax, straight_picture())
    for row in sorted(set(np.round(mv, 3))):
        ax.axhline(row, color=TEXT, ls=":", lw=1.6)
    ax.plot(mu, mv, "o", ms=14.1, fillstyle="none", mec=CAM, mew=3.0, zorder=4)
    for (f, l), u, v in zip(MARKS, mu, mv):
        _say(ax, (u + (-8 if l > 0 else 8), v - 24), f"({f:.0f}, {l:.0f})".replace("-", "−"), CAM, ha="right" if l > 0 else "left")
    for row, side in ((mv.max(), "bottom"), (mv.min(), "top")):
        _badge(ax, (160, row + 18) if side == "bottom" else (30, row - 28), f"row {row:.3f}" if side == "bottom" else f"row\n{row:.3f}")
    ax.set(xlim=(-0.5, 319.5), ylim=(179.5, -0.5))
    ax.set_xticks([])
    ax.set_yticks([])
    _save(fig, path)


def _implied_heights(pitch, rows, forward):
    return forward * np.tan(pitch + np.arctan((rows - CAMERA.cy) / CAMERA.f))


def _calibrate(rows, forward, lo=-0.3, hi=0.3, iters=60):
    """The pitch that makes the near and far marks imply the same camera height, found by bisection."""
    near, far = forward == forward.min(), forward == forward.max()
    gap = lambda p: _implied_heights(p, rows[far], forward[far]).mean() - _implied_heights(p, rows[near], forward[near]).mean()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gap(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def _calibration_spread(path):
    _, mv = CAMERA.ground_to_pixel(MARKS[:, 0], MARKS[:, 1])
    rng = np.random.default_rng(2)
    pitches = np.array([_calibrate(mv + rng.normal(0.0, 0.5, 4), MARKS[:, 0]) for _ in range(500)])
    fig = _fig(3.9)
    ax = fig.subplots()
    ax.hist(pitches, bins=30, color=BORDER, edgecolor=TEXT, linewidth=0.8)
    top = ax.get_ylim()[1]
    ax.axvline(PHI, color=TEXT, lw=2.4)
    ax.axvline(pitches.mean(), color=CAM, lw=2.4, ls="--")
    _say(ax, (PHI - 0.0004, top * 1.22), f"true {PHI:.4f}", TEXT, ha="right", va="top")
    _say(ax, (pitches.mean() + 0.0004, top * 1.22), f"mean {pitches.mean():.4f}", CAM, ha="left", va="top")
    one_row = 1 / CAMERA.f
    ax.set(xlabel="recovered pitch (rad)", ylabel="runs", ylim=(0, top * 1.3))
    lo, hi = ax.get_xlim()
    ax.set_xticks([t for t in ax.get_xticks() if lo + 0.08 * (hi - lo) < t <= hi])  # no tick label in the corner, on top of the y axis' zero
    fig.text(0.5, 0.012, f"std {pitches.std():.4f} rad = {pitches.std() / one_row:.1f} of one row's angle", ha="center", va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, path)


def _next_lesson(path):
    tel = drive("gentle")
    fig = _fig(3.6)
    ax = fig.subplots()
    ax.plot(tel.t, tel.speed, color=CAM, lw=2.8)
    ax.set(xlim=(0, tel.t[-1] * 1.02), ylim=(0, 10), xlabel="time (s)", ylabel="speed (m/s)")
    _say(ax, (tel.t[-1] / 2, tel.speed.mean() - 1.6), f"{tel.speed.mean():.0f} m/s, start to finish", CAM)
    ax.set_xticks([0, 20, 40, 60, round(tel.t[-1], 1)])
    ax.set_xticklabels(["0", "20", "40", "60", f"{tel.t[-1]:.1f}"])
    _say(ax, (tel.t[-1], 9.0), f"{tel.t[-1]:.1f} s", TEXT, ha="right")
    fig.tight_layout()
    _save(fig, path)


def make(out_dir):
    with matplotlib.rc_context(STYLE):
        _result_first(out_dir / "result_first.png")
        _ego_frame_3d(out_dir / "ego_frame_3d.png")
        _right_hand_turn(out_dir / "right_hand_turn.png")
        _three_rotations(out_dir / "three_rotations.png")
        _pitch_columns(out_dir / "pitch_columns.png")
        _optical_vs_ego(out_dir / "optical_vs_ego.png")
        _wrong_order(out_dir / "wrong_order.png")
        _mount_side(out_dir / "mount_side.png")
        _projected_points(out_dir / "projected_points.png")
        _blind_zone(out_dir / "blind_zone.png")
        _point_vs_direction(out_dir / "point_vs_direction.png")
        _horizon_w(out_dir / "horizon_w.png")
        _pixel_ray(out_dir / "pixel_ray.png")
        _rows_across(out_dir / "rows_across.png")
        _rows_pitch(out_dir / "rows_pitch.png")
        _lat_pitch(out_dir / "lat_pitch.png")
        _heading_pitch(out_dir / "heading_pitch.png")
        _gate_breaks(out_dir / "gate_breaks.png")
        _height_scale(out_dir / "height_scale.png")
        _calibration_marks(out_dir / "calibration_marks.png")
        _calibration_spread(out_dir / "calibration_spread.png")
        _next_lesson(out_dir / "next_lesson.png")
