"""Figures for lecture 1.1.2 The view from the driver's seat.  `python scripts/make_figures.py course/1.1.2-the-view-from-the-drivers-seat`.

Ruling F: every figure is 5.5 in wide, saved at 200 dpi, smallest text 16 pt (titles 18 pt), panels stacked in one column.
Every number drawn comes from the simulator and zero2fsd.geom, never typed; measure.py prints the same values.
"""
import math

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Arc, Polygon

from zero2fsd.car import Car, run
from zero2fsd.car import provided
from zero2fsd.car.provided import estimate_lane
from zero2fsd.car.types import Observation
from zero2fsd.drawing import draw_frame
from zero2fsd.geom import ego_to_world, forward_left, heading_vector, rotation_matrix, world_to_ego
from zero2fsd.palette import CHROME, QUANTITY, SCENE, shades
from zero2fsd.sim import BicycleState, Camera, dynamics, make_scenario, scenarios
from zero2fsd.sim.world import EGO_LANE_CENTER, WHITE_LINE_LATERAL

W = 5.5  # every figure: 1100 px wide at the figstyle dpi, shown at 550 px on a desktop and 330 px on a phone
FS, TITLE = 16, 18  # smallest text, and titles, in points: a digit stays at least 9 px tall at 330 px wide
STYLE = {"font.size": FS, "axes.labelsize": FS, "axes.titlesize": TITLE, "axes.titlepad": 8, "xtick.labelsize": FS, "ytick.labelsize": FS}
TEXT, PANEL, BORDER = CHROME["text"], CHROME["panel"], CHROME["border"]
EGO, THETA, E_OFF, PSI, BOX = (QUANTITY[k] for k in ("ego", "theta", "e_off", "psi", "perception"))
YELLOW, WHITE = SCENE["yellow"], SCENE["white"]
GREY = shades(BORDER, 5)[4]  # the world frame and what names no quantity
FAINT = shades(BORDER, 5)[2]
RIGHT_RUN, SWAP_RUN, NO_ROT_RUN = shades(E_OFF, 3)  # the right transform, the swap, then no rotation
BADGE = {"boxstyle": "round,pad=0.2", "fc": PANEL, "ec": BORDER}  # text that sits on top of lines or a picture
VIEW_S = 150.0  # the scene: the provided car's lap at 150 m along gentle
PARKED_S = 150.0


# ---- the learner-style transforms the lab breaks: the right one in matrix form, then the swap and no rotation ----
def matrix_form(pose, p_world):
    x, y, theta = pose
    return (np.asarray(p_world, dtype=float) - np.array([x, y])) @ rotation_matrix(theta)


def swap(pose, p_world):
    x, y, theta = pose
    return (np.asarray(p_world, dtype=float) - np.array([x, y])) @ rotation_matrix(theta).T


def no_rotation(pose, p_world):
    x, y, theta = pose
    return np.asarray(p_world, dtype=float) - np.array([x, y])


class Data:
    """The runs and poses the figures draw: built once per make()."""

    def __init__(self):
        self.gentle = make_scenario("gentle").road
        self.straight = make_scenario("straight").road
        self.ref = run(Car(), "gentle")
        k = int(np.argmax(self.ref.progress_m >= VIEW_S))
        self.k = k
        self.pose = (float(self.ref.x[k]), float(self.ref.y[k]), float(self.ref.yaw[k]))
        self.e, self.psi = float(self.ref.lat[k]), float(self.ref.head_err[k])
        self.s_car = self.gentle.project(*self.pose[:2])[0]
        self.centre = lambda ahead: self.gentle.offset(self.s_car + ahead, EGO_LANE_CENTER)  # lane centre `ahead` m along the road
        self.v10 = self.centre(10.0) - np.array(self.pose[:2])
        self.parked = BicycleState(*self.gentle.ego_pose(PARKED_S), 8.0, 0.0)
        self.camera = Camera()
        self.parked_pose = (self.parked.x, self.parked.y, self.parked.yaw)
        self.tels = {(name, scen): (self.ref if fn is None and scen == "gentle" else run(Car(), scen, camera=None if fn is None else Camera(world_to_ego=fn)))
                     for name, fn in (("right", None), ("swap", swap), ("no rotation", no_rotation)) for scen in ("straight", "gentle")}

    def state(self, tel, k):
        return BicycleState(float(tel.x[k]), float(tel.y[k]), float(tel.yaw[k]), float(tel.speed[k]), 0.0)


def _fig(height, rows=1, ratios=None):
    fig = Figure(figsize=(W, height))
    return fig, fig.subplots(rows, 1, gridspec_kw={"height_ratios": ratios} if ratios else None)


def _panel(xlim, ylim):
    """Inches of height an equal-aspect panel needs at this figure's width, so a figure has no empty bands."""
    return (W - 1.35) * abs(ylim[1] - ylim[0]) / abs(xlim[1] - xlim[0])


def num(value, places=3, sign=False):
    """A number with a real minus sign, as the lecture prints it."""
    return f"{value:{'+' if sign else ''}.{places}f}".replace("-", "−")


def _save(fig, path):
    fig.tight_layout()
    fig.canvas.draw()  # an arrow patch reports its data-space path only once the axes are final
    fig.savefig(path)


def say(ax, xy, text, dx=0, dy=0, ha="center", va="center", colour=TEXT, badge=False, arrow=False, **kwargs):
    """Text at a data point, nudged by (dx, dy) points; `arrow` draws a leader to the point."""
    props = {"arrowstyle": "-", "color": colour, "lw": 1.5} if arrow else None
    return ax.annotate(text, xy, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, color=colour, arrowprops=props,
                       bbox=BADGE if badge else None, annotation_clip=False, **kwargs)


def arrow(ax, start, end, colour, lw=2.5, style="-|>"):
    """An arrow as an annotation: not an axes patch, so it carries no closed-path vertex outside the data."""
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": style, "color": colour, "lw": lw, "shrinkA": 0, "shrinkB": 0, "mutation_scale": FS * 1.2})


def arc(ax, centre, radius, start, end, colour, lw=2.5):
    ax.add_patch(Arc(centre, 2 * radius, 2 * radius, theta1=math.degrees(start), theta2=math.degrees(end), color=colour, lw=lw))


def lane_lines(road, s0, s1, pose=None, n=200):
    """{name: (n, 2)} the yellow line, the white line and the lane centre from s0 to s1, in the world or in `pose`'s ego frame."""
    s = np.linspace(s0, s1, n)
    lines = {"yellow": road.offset(s, 0.0), "white": road.offset(s, WHITE_LINE_LATERAL), "centre": road.offset(s, EGO_LANE_CENTER)}
    return {k: (v if pose is None else world_to_ego(pose, v)) for k, v in lines.items()}


def draw_lines(ax, lines, ego=False, box=None):
    """Plot lane lines; in the ego frame forward is up the page and left is to the left (x = left, inverted axis).
    `box` = ((x0, x1), (y0, y1)) in plotted coordinates: points outside it are dropped, never cut off by an axis limit."""
    for name, colour, style in (("yellow", YELLOW, (0, (7, 3))), ("white", WHITE, "-"), ("centre", GREY, "--")):
        pts = lines[name]
        xs, ys = (pts[:, 1], pts[:, 0]) if ego else (pts[:, 0], pts[:, 1])
        if box:
            out = (xs < min(box[0])) | (xs > max(box[0])) | (ys < min(box[1])) | (ys > max(box[1]))
            xs, ys = np.where(out, np.nan, xs), np.where(out, np.nan, ys)
        ax.plot(xs, ys, color=colour, ls=style, lw=2)


def ego_axes(ax, xlim, ylim, xlabel="left (m)", ylabel="forward (m)"):
    """An ego-frame panel: forward up the page, left to the left of the page, 1 m the same length both ways."""
    ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
    ax.set_aspect("equal", adjustable="box")


def bare(ax):
    """No axes at all: no ticks (an off axis still has tick labels the gates would measure), no frame."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def car_dot(ax, xy, markersize=10):
    ax.plot(*xy, marker="o", markersize=markersize, color=TEXT, zorder=4)


HOLLOW = (0.0, 0.0, 0.0, 0.0)  # fully transparent: a marker drawn as an outline only
ZOOM = "zoom-inset"  # the gid the data-inside-axes gate reads: an axes that shows only part of its data on purpose


def zoom_inset(ax, rect, xlim, ylim, parent=None):
    """A magnified window onto `xlim` x `ylim` of `ax`, drawn in `rect` (axes fractions) with its connectors; `parent` is the axes it zooms."""
    inset = ax.inset_axes(rect, gid=ZOOM)
    inset.set(xlim=xlim, ylim=ylim, xticks=[], yticks=[])
    inset.set_aspect("equal", adjustable="box")
    (parent or ax).indicate_inset_zoom(inset, edgecolor=GREY, linewidth=1.5)
    return inset


def bracket(ax, a, b, colour, off):
    """A measuring bracket from a to b, set `off` data units to the right of the direction a -> b; returns its middle."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    direction = (b - a) / np.linalg.norm(b - a)
    side = off * np.array([direction[1], -direction[0]])
    ax.plot(*zip(a, a + side, b + side, b), color=colour, lw=2)
    return (a + b) / 2 + side


# ---- 1. result_first ----
def _result_first(d, path):
    fig, (top, bottom) = _fig(9.8, 2, [3.4, 4.6])
    x, y, th = d.pose
    box = ((138, 170), (8, 35))
    world_lines = lane_lines(d.gentle, 135, 190, n=600)
    draw_lines(top, world_lines, box=box)
    c = world_lines["centre"]
    spot = c[int(np.argmin(np.abs(c[:, 0] - 163)))]
    say(top, tuple(spot), "lane centre\n(not painted)", dx=24, dy=-44, ha="left", colour=GREY, badge=True, arrow=True)
    top.plot([x, x + 12], [y, y], color=GREY, ls=":", lw=2)
    arc(top, (x, y), 9.0, 0, th, THETA)
    say(top, (x + 9.0, y), f"θ = {th:.4f} rad", dx=4, dy=-18, ha="left", colour=THETA)
    car_dot(top, (x, y))
    draw_frame(top, d.pose, EGO, "", length_m=5.0, fontsize=FS)
    top.set(xlim=box[0], ylim=box[1], xlabel="east (m)", ylabel="north (m)", title="world frame")
    top.set_aspect("equal", adjustable="box")
    ego_lines = lane_lines(d.gentle, 140, 190, d.pose, n=500)
    draw_lines(bottom, ego_lines, ego=True, box=((14, -14), (-8, 34)))
    c = ego_lines["centre"]
    spot = c[int(np.argmin(np.abs(c[:, 0] - 25)))]
    say(bottom, (spot[1], spot[0]), "lane centre\n(not painted)", dx=120, dy=0, ha="left", colour=GREY, badge=True, arrow=True)
    for ahead in (10, 20, 30):
        fl = world_to_ego(d.pose, d.centre(ahead))
        bottom.plot(fl[1], fl[0], "o", color=EGO, markersize=9, zorder=5)
        say(bottom, (fl[1], fl[0]), f"{fl[1]:.3f} m left", dx=12, ha="left", colour=EGO, badge=True)
    foot = world_to_ego(d.pose, d.gentle.offset(d.s_car, EGO_LANE_CENTER))
    bottom.plot(foot[1], foot[0], "o", color=EGO, markersize=9, zorder=5)
    say(bottom, (foot[1], foot[0]), f"{-foot[1]:.3f} m right", dx=30, dy=-26, ha="left", colour=EGO, badge=True, arrow=True)
    car_dot(bottom, (0, 0), markersize=14)
    ego_axes(bottom, (14, -14), (-8, 34))
    bottom.set_title("the car's own frame")
    _save(fig, path)


# ---- 2. two_dots ----
def _unit(theta):
    return heading_vector(theta), np.array([-math.sin(theta), math.cos(theta)])


def _two_dots(d, path):
    xlim, ylim = (-4, 13.5), (-2.5, 13.5)
    fig, ax = _fig(_panel(xlim, ylim) + 1.3)
    th = d.pose[2]
    h, n = _unit(th)
    v = d.v10
    fwd, left = forward_left(th, v)
    for hat, lo, hi in ((h, -1.5, 11.5), (n, -1.5, 7.0)):
        ax.plot(*zip(lo * hat, hi * hat), color=EGO, lw=1.5, alpha=0.6)
    arrow(ax, (0, 0), (11.5, 0), GREY, lw=1.5)
    arrow(ax, (0, 0), (0, 7.5), GREY, lw=1.5)
    for hat, name in ((h, r"$\hat{h}$"), (n, r"$\hat{n}$")):
        arrow(ax, (0, 0), tuple(1.5 * hat), EGO, lw=3.5)
        say(ax, tuple(1.5 * hat), name, dx=-16 if hat is n else 8, dy=8 if hat is n else -18, colour=EGO)
    for foot in (fwd * h, left * n):
        ax.plot(*zip(v, foot), color=EGO, ls="--", lw=1.8)
    arrow(ax, (0, 0), tuple(v), TEXT, lw=3.5)
    say(ax, tuple(v / 2), f"v =\n({num(v[0])}, {num(v[1])})", dx=-70, dy=130, arrow=True)
    mid = bracket(ax, (0, 0), fwd * h, EGO, 1.3)
    say(ax, tuple(mid), f"{num(fwd)} m along " + r"$\hat{h}$", dx=22 * math.sin(th), dy=-22 * math.cos(th), colour=EGO, rotation=math.degrees(th), rotation_mode="anchor")
    car_dot(ax, (0, 0))
    ax.set(xlim=xlim, ylim=ylim, xlabel="east (m), car at the origin", ylabel="north (m)")
    ax.set_aspect("equal", adjustable="box")
    foot = fwd * h
    centre, half = (v + foot) / 2, 0.7
    inset = zoom_inset(ax, [0.58, 0.55, 0.40, 0.44], (centre[0] - half, centre[0] + half), (centre[1] - half, centre[1] + half))
    inset.plot(*zip(foot - 1.2 * h, foot + 0.6 * h), color=EGO, lw=1.5, alpha=0.6)
    inset.plot(*zip(v, foot), color=EGO, ls="--", lw=2)
    arrow(inset, tuple(v - 0.5 * v / np.linalg.norm(v)), tuple(v), TEXT, lw=3.5)
    inset.plot(*zip(foot - 0.12 * h, foot - 0.12 * h + 0.12 * n, foot + 0.12 * n), color=EGO, lw=1.5)
    say(inset, tuple(v), f"{num(left)} m\nalong " + r"$\hat{n}$", dx=14, dy=20, ha="center", colour=EGO)
    _save(fig, path)


# ---- 3. stacked_dots ----
def _stacked_dots(d, path):
    fig, ax = _fig(2.8)
    th = d.pose[2]
    v = d.v10
    matrix = rotation_matrix(th).T  # rows: the heading arrow, then the left arrow
    out = matrix @ v
    light = shades(EGO, 2)[1]
    ys = (0.68, 0.30)
    half = 0.17

    def square_bracket(x_left, x_right):
        for x, tip in ((x_left, 0.012), (x_right, -0.012)):
            ax.plot([x + tip, x, x, x + tip], [ys[0] + half, ys[0] + half, ys[1] - half, ys[1] - half], color=TEXT, lw=2)

    for y, colour, word, entries, result in zip(ys, (EGO, light), ("forward", "left"), matrix, out):
        ax.axhspan(y - half, y + half, color=colour, alpha=0.18, lw=0)
        for x, value in zip((0.13, 0.36), entries):
            say(ax, (x, y), num(value, 4), colour=colour)
        say(ax, (0.855, y + 0.05), f"{result:.3f}", colour=colour, fontweight="bold")
        say(ax, (0.855, y - 0.07), word, colour=colour)
    for x, value, y in zip((0.59, 0.59), v, ys):
        say(ax, (x, y), f"{value:.3f}")
    square_bracket(0.02, 0.47)
    square_bracket(0.52, 0.66)
    square_bracket(0.74, 0.97)
    say(ax, (0.495, sum(ys) / 2), "·")
    say(ax, (0.70, sum(ys) / 2), "=")
    ax.set(xlim=(0, 1), ylim=(0.1, 0.88))
    bare(ax)
    _save(fig, path)


# ---- 4. columns_axes ----
def _columns_axes(d, path):
    top_lim, top_y, bot_x, bot_y = (-3.0, 3.3), (-0.75, 1.9), (144, 168), (9.5, 24.5)
    top_h, bot_h = _panel(top_lim, top_y), _panel(bot_x, bot_y)
    fig, (top, bottom) = _fig(top_h + bot_h + 2.4, 2, [top_h + 0.5, bot_h + 1.0])
    th = d.pose[2]
    h, n = _unit(th)
    arrow(top, (0, 0), (1.3, 0), GREY)
    arrow(top, (0, 0), (0, 1.3), GREY)
    say(top, (1.3, 0), "world x", dx=8, ha="left", colour=GREY)
    say(top, (0, 1.3), "world y", dy=16, colour=GREY)
    arrow(top, (0, 0), tuple(h), EGO, lw=3.5)
    arrow(top, (0, 0), tuple(n), EGO, lw=3.5)
    say(top, tuple(h), f"x (col 1)\n({num(h[0], 4)}, {num(h[1], 4)})", dx=12, dy=14, ha="left", colour=EGO)
    say(top, tuple(n), f"y (col 2)\n({num(n[0], 4)}, {num(n[1], 4)})", dx=-10, dy=4, ha="right", colour=EGO)
    car_dot(top, (0, 0))
    arc(top, (0, 0), 0.4, 0, th, THETA)
    say(top, (0.4, 0), f"θ = {th:.4f} rad", dx=10, dy=-28, ha="left", colour=THETA)
    top.set(xlim=top_lim, ylim=top_y)
    top.set_aspect("equal", adjustable="box")
    bare(top)
    x, y = d.pose[:2]
    fwd, left = world_to_ego(d.pose, d.centre(10.0))
    corner = np.array([x, y]) + fwd * h
    end = d.centre(10.0)
    arrow(bottom, (x, y), tuple(corner), EGO, lw=3.5)
    arrow(bottom, tuple(corner), tuple(end), EGO, lw=3.5)
    car_dot(bottom, (x, y))
    bottom.plot(*end, "o", color=TEXT, markersize=10, zorder=4)
    say(bottom, (x, y), f"({x:.3f}, {y:.3f})", dx=-4, dy=-24, ha="left")
    say(bottom, tuple(end), f"({end[0]:.3f}, {end[1]:.3f})", dx=22, dy=30, ha="right")
    say(bottom, tuple((np.array([x, y]) + corner) / 2), f"{fwd:.3f} m\nalong\ncolumn 1", dx=-14, dy=24, ha="right", colour=EGO, arrow=True)
    bottom.set(xlim=bot_x, ylim=bot_y, xlabel="east (m)", ylabel="north (m)", title="world coordinates")
    bottom.set_aspect("equal", adjustable="box")
    width = 0.30
    inset = zoom_inset(bottom, [1 - width - 0.01, 0.02, width, width * (bot_x[1] - bot_x[0]) / (bot_y[1] - bot_y[0])],
                       (corner[0] + (end[0] - corner[0]) / 2 - 1.4, corner[0] + (end[0] - corner[0]) / 2 + 1.4),
                       (corner[1] + (end[1] - corner[1]) / 2 - 1.4, corner[1] + (end[1] - corner[1]) / 2 + 1.4))
    inset.plot(*zip(corner - 2.0 * h, corner), color=EGO, lw=3.5)
    arrow(inset, tuple(corner), tuple(end), EGO, lw=3.5)
    inset.plot(*end, "o", color=TEXT, markersize=10, zorder=4)
    say(inset, tuple((corner + end) / 2), f"{left:.3f} m\nalong\ncolumn 2", dy=66, colour=EGO, arrow=True)
    _save(fig, path)


# ---- 5. undo ----
def _undo(d, path):
    fig, axes = _fig(2 * _panel((-1.5, 14.5), (-1.8, 8.5)) + 2.7, 2)
    th = d.pose[2]
    fl = np.array(forward_left(th, d.v10))
    wv = d.v10
    light = shades(EGO, 2)[1]
    ang_a, ang_b = math.atan2(fl[1], fl[0]), math.atan2(wv[1], wv[0])
    for ax, turned in zip(axes, (True, False)):
        arrow(ax, (0, 0), (14, 0), GREY, lw=1.5)
        arrow(ax, (0, 0), (0, 8.5), GREY, lw=1.5)
        arrow(ax, (0, 0), tuple(fl), light, lw=3.5)
        arrow(ax, (0, 0), tuple(wv), EGO, lw=3.5)
        say(ax, tuple(fl), f"({fl[0]:.3f}, {fl[1]:.3f})", dy=-22, colour=light)
        say(ax, tuple(wv), f"({wv[0]:.3f}, {wv[1]:.3f})", dx=-8, dy=18, ha="right", colour=EGO)
        r = 6.5
        start, end = (ang_a, ang_b) if turned else (ang_b, ang_a)
        arc(ax, (0, 0), r, min(start, end), max(start, end), THETA)
        tip, before = end, end - math.copysign(0.05, end - start)
        arrow(ax, (r * math.cos(before), r * math.sin(before)), (r * math.cos(tip), r * math.sin(tip)), THETA, lw=2.5)
        label = r"$R(\theta)$" + f"\nturn by {th:.4f} rad" if turned else r"$R(\theta)^{\mathsf{T}}$" + f"\nback by {th:.4f} rad"
        say(ax, (r * math.cos((ang_a + ang_b) / 2), r * math.sin((ang_a + ang_b) / 2)), label, dx=14, dy=-4, ha="left", colour=THETA)
        car_dot(ax, (0, 0))
        ax.set(xlim=(-1.5, 14.5), ylim=(-1.8, 8.5), xlabel="east (m), car at the origin")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("north (m)")
    axes[1].set_ylabel("north (m)")
    _save(fig, path)


# ---- 6. order_trap ----
def _order_trap(d, path):
    fig, ax = _fig(_panel((10, -90), (-15, 35)) + 1.6)
    right = np.array(forward_left(d.pose[2], d.v10))
    wrong = rotation_matrix(d.pose[2]).T @ d.centre(10.0) - np.array(d.pose[:2])  # turn first, then subtract the position
    draw_lines(ax, lane_lines(d.gentle, 120, 200, d.pose, n=500), ego=True, box=((10, -90), (-15, 35)))
    car_dot(ax, (0, 0))
    ax.plot(right[1], right[0], "o", color=EGO, markersize=11, zorder=5)
    say(ax, (right[1], right[0]), f"subtract, then turn\n({right[0]:.3f}, {right[1]:.3f})", dx=22, dy=-48, ha="left", colour=EGO, arrow=True)
    say(ax, (0, 0), "car", dx=-4, dy=-30, badge=True, arrow=True)
    ax.plot(wrong[1], wrong[0], "x", color=GREY, markersize=14, markeredgewidth=3, zorder=5)
    say(ax, (wrong[1], wrong[0]), f"turn, then subtract\n({num(wrong[0])}, {num(wrong[1])})", dx=-24, dy=64, ha="right", colour=GREY, arrow=True)
    ego_axes(ax, (10, -90), (-15, 35))
    ax.set_title("the car's own frame")
    _save(fig, path)


# ---- 7. round_trip ----
def _round_trip(d, path):
    cloud = np.concatenate([d.gentle.sample_points(1.0, lateral) for lateral in (0.0, EGO_LANE_CENTER, WHITE_LINE_LATERAL)])
    back = ego_to_world(d.pose, world_to_ego(d.pose, cloud))
    rng = np.random.default_rng(112)
    rng.uniform(-math.pi, math.pi, 40)  # measure.py draws 40 yaws first; the poses follow
    poses = [d.pose, *np.column_stack([rng.uniform(-150, 150, 40), rng.uniform(-150, 150, 40), rng.uniform(-math.pi, math.pi, 40)])]
    there = [np.abs(ego_to_world(p, world_to_ego(p, cloud)) - cloud).max() for p in poses]
    return_ = [np.abs(world_to_ego(p, ego_to_world(p, cloud)) - cloud).max() for p in poses]
    xlim, ylim = (cloud[:, 0].min() - 10, cloud[:, 0].max() + 10), (cloud[:, 1].min() - 10, cloud[:, 1].max() + 10)
    top_h = _panel(xlim, ylim)
    fig, (top, bottom) = _fig(top_h + 5.6, 2, [top_h + 0.8, 4.2])
    shown = slice(None, None, 15)  # every fifteenth point, so the dots stay dots
    top.scatter(cloud[shown, 0], cloud[shown, 1], s=30, color=GREY, zorder=1)
    top.scatter(back[shown, 0], back[shown, 1], s=110, facecolors=HOLLOW, edgecolors=EGO, linewidths=1.5, zorder=2)
    top.set(xlim=xlim, ylim=ylim, xlabel="east (m)", ylabel="north (m)", title=f"{len(cloud)} road points")
    top.set_aspect("equal", adjustable="box")
    left_end, right_end = cloud[int(np.argmin(cloud[:, 0]))], cloud[int(np.argmax(cloud[:, 0]))]
    say(top, tuple(left_end), "grey: the points", dx=40, dy=30, ha="left", colour=GREY, badge=True, arrow=True)
    say(top, tuple(right_end), "purple rings:\nafter the round trip", dx=-40, dy=-40, ha="right", colour=EGO, badge=True, arrow=True)
    k = np.arange(1, len(poses) + 1)
    bottom.semilogy(k, there, "o", color=GREY, markersize=5)
    bottom.semilogy(k, return_, "s", mfc=HOLLOW, mec=EGO, mew=2, markersize=9)
    bottom.axhline(1e-12, color=TEXT, ls="--", lw=2)
    low = 10.0 ** math.floor(math.log10(min(min(there), min(return_))))
    bottom.set(xlim=(0, len(poses) + 1), ylim=(low, 1e-11), xlabel="pose", ylabel="largest error (m)")
    say(bottom, (len(poses) / 2, 1e-12), r"$10^{-12}$ m", dy=16, colour=TEXT)
    say(bottom, (len(poses) / 2, max(there)), "world → ego → world", dy=34, colour=GREY, badge=True)
    say(bottom, (len(poses) / 2, min(return_)), "ego → world → ego", dy=-34, colour=EGO, badge=True)
    _save(fig, path)


# ---- 8. quads and 10. renderer_steps share the quads the camera draws ----
def _logged_quads(d):
    """The world quads of one picture at the parked pose, per layer (road, white line, yellow dashes), as the camera hands them to world_to_ego."""
    calls = []

    def logging(pose, quads):
        calls.append(np.array(quads))
        return world_to_ego(pose, quads)

    Camera(world_to_ego=logging).render(d.gentle, d.parked)
    return calls


def _outline(ax, quads, colour, lw=1.2):
    for quad in quads:
        ax.add_patch(Polygon(quad, closed=True, fill=False, edgecolor=colour, lw=lw))


def _quads(d, path):
    road, white, dashes = _logged_quads(d)
    px, py, _ = (d.parked.x, d.parked.y, d.parked.yaw)
    xlim, ylim = (px - 9, px + 12), (py - 4, py + 24)

    def inside(quads):
        return np.array([q for q in quads if (q[:, 0] > xlim[0]).all() and (q[:, 0] < xlim[1]).all() and (q[:, 1] > ylim[0]).all() and (q[:, 1] < ylim[1]).all()])

    fig, ax = _fig(_panel(xlim, ylim) + 1.3)
    _outline(ax, inside(road), FAINT)
    _outline(ax, inside(white), WHITE)
    _outline(ax, inside(dashes), YELLOW)
    ahead = np.array([q.mean(axis=0) @ heading_vector(d.parked.yaw) - (px, py) @ heading_vector(d.parked.yaw) for q in road])
    pick = road[int(np.argmin(np.abs(ahead - 8.0)))]
    ax.add_patch(Polygon(pick, closed=True, fc=EGO, ec=EGO, alpha=0.45))
    ax.plot(pick[:, 0], pick[:, 1], "o", color=EGO, markersize=6, zorder=5)
    car_dot(ax, (px, py))
    say(ax, (px, py), "car", dx=-12, dy=-24, ha="right", badge=True)
    ax.set(xlim=xlim, ylim=ylim, xlabel="east (m)", ylabel="north (m)")
    ax.set_aspect("equal", adjustable="box")
    ends = pick[np.argsort(pick[:, 1])[:2]]  # the two corners at one end of the strip, 0.5 m apart
    centre, half = ends.mean(axis=0), 0.4  # a 0.8 m window: over 10 times the main panel's scale
    inset = zoom_inset(ax, [0.02, 0.62, 0.42, 0.36], (centre[0] - half, centre[0] + half), (centre[1] - half, centre[1] + half))
    inset.add_patch(Polygon(pick, closed=True, fc=EGO, ec=EGO, alpha=0.45))
    inset.plot(pick[:, 0], pick[:, 1], "o", color=EGO, markersize=8)
    for corner, word, dx, dy in zip(ends, ("one corner:", "the next corner:"), (110, 110), (-8, 70)):
        say(inset, tuple(corner), f"{word}\n({corner[0]:.1f}, {corner[1]:.1f})", dx=dx, dy=dy, colour=EGO, badge=True, arrow=True)
    _save(fig, path)


# ---- 9. shape_trap ----
def _shape_trap(d, path):
    rows = np.stack([d.centre(10.0), d.centre(30.0)]) - np.array(d.pose[:2])
    rot = rotation_matrix(d.pose[2])
    right, wrong = rows @ rot, rot @ rows
    xlim, ylim = (26, -30), (-12, 36)
    fig, ax = _fig(_panel(xlim, ylim) + 2.2)
    draw_lines(ax, lane_lines(d.gentle, 140, 200, d.pose, n=500), ego=True, box=(xlim, ylim))
    car_dot(ax, (0, 0))
    for (fwd, left), (wf, wl) in zip(right, wrong):
        ax.plot(left, fwd, "o", color=EGO, markersize=11, zorder=5)
        ax.plot(wl, wf, "x", color=GREY, markersize=14, markeredgewidth=3, zorder=5)
    say(ax, (right[0, 1], right[0, 0]), f"rows @ rot:\n({num(right[0, 0])}, {num(right[0, 1])})", dx=-30, dy=-40, ha="right", colour=EGO, badge=True, arrow=True)
    say(ax, (right[1, 1], right[1, 0]), f"({num(right[1, 0])}, {num(right[1, 1])})", dx=26, dy=0, ha="left", colour=EGO, badge=True, arrow=True)
    say(ax, (wrong[0, 1], wrong[0, 0]), f"rot @ rows:\n({num(wrong[0, 0])}, {num(wrong[0, 1])})", dx=-14, dy=58, ha="left", colour=GREY, badge=True, arrow=True)
    say(ax, (wrong[1, 1], wrong[1, 0]), f"({num(wrong[1, 0])}, {num(wrong[1, 1])})", dy=-36, colour=GREY, badge=True, arrow=True)
    say(ax, (19.5, 32), "no error\nmessage", colour=TEXT, badge=True)
    ego_axes(ax, xlim, ylim)
    ax.set_title("the car's own frame")
    _save(fig, path)


# ---- pictures: the camera's 320 x 180 image, nearest-neighbour so a pixel stays a square ----
def _picture(ax, image, title):
    ax.imshow(image, interpolation="nearest")
    bare(ax)
    ax.set_title(title)


def _slice_ego(quads, pose, forward):
    ego = world_to_ego(pose, quads)
    keep = (ego[..., 0] > forward[0]).all(axis=1) & (ego[..., 0] < forward[1]).all(axis=1)
    return ego[keep]


# ---- 10. renderer_steps ----
def _renderer_steps(d, path):
    road, white, dashes = _logged_quads(d)
    every = np.concatenate([road, white, dashes])
    world = every.reshape(-1, 2)
    xlim, ylim = (world[:, 0].min() - 2, world[:, 0].max() + 2), (world[:, 1].min() - 2, world[:, 1].max() + 2)
    sliced = [(_slice_ego(quads, d.parked_pose, (-3.0, 40.0)), colour) for quads, colour in ((road, FAINT), (white, WHITE), (dashes, YELLOW))]
    lateral = np.concatenate([q[..., 1].ravel() for q, _ in sliced])
    mid_xlim, mid_ylim = (lateral.max() + 1, lateral.min() - 1), (-3, 40)
    top_h, mid_h = _panel(xlim, ylim), _panel(mid_xlim, mid_ylim)
    fig, (top, mid, bottom) = _fig(top_h + mid_h + 3.1 + 2.4, 3, [top_h + 0.5, mid_h + 0.5, 3.1])
    for quads, colour in ((road, FAINT), (white, WHITE), (dashes, YELLOW)):
        _outline(top, quads, colour, lw=0.8)
    car_dot(top, (d.parked.x, d.parked.y))
    top.set(xlim=xlim, ylim=ylim, xlabel="east (m)", ylabel="north (m)", title="world frame")
    top.set_aspect("equal", adjustable="box")
    for quads, colour in sliced:
        for quad in quads:
            mid.add_patch(Polygon(quad[:, ::-1], closed=True, fill=False, edgecolor=colour, lw=0.8))
    mid.plot(0, 0, "o", color=TEXT, markersize=10)
    mid.set(xlim=mid_xlim, ylim=mid_ylim, xlabel="left (m)", ylabel="forward (m)", title="ego frame")
    mid.set_aspect("equal", adjustable="box")
    _picture(bottom, d.camera.render(d.gentle, d.parked), "camera picture, 320 × 180")
    _save(fig, path)


# ---- 11. swap_same ----
def _swap_same(d, path):
    state = d.state(d.ref, 380)
    a = d.camera.render(d.gentle, state)
    b = Camera(world_to_ego=matrix_form).render(d.gentle, state)
    differ = int((a != b).any(axis=-1).sum())
    fig, (top, bottom) = _fig(7.4, 2)
    _picture(top, a, "simulator's transform")
    _picture(bottom, b, "rows times R(θ)")
    bottom.set_xlabel(f"{differ} of {a.shape[0] * a.shape[1]} pixels differ")
    _save(fig, path)


def _share(d, fn, state):
    good = d.camera.render(d.gentle, state, return_labels=True)[1]
    return float((good != Camera(world_to_ego=fn).render(d.gentle, state, return_labels=True)[1]).mean())


# ---- 12. pictures_150 ----
def _pictures_150(d, path):
    fig, axes = _fig(10.6, 3)
    _picture(axes[0], d.camera.render(d.gentle, d.parked), "right transform")
    for ax, fn, name in ((axes[1], swap, "the swap"), (axes[2], no_rotation, "no rotation")):
        _picture(ax, Camera(world_to_ego=fn).render(d.gentle, d.parked), f"{name}: {_share(d, fn, d.parked):.1%} changed")
    _save(fig, path)


# ---- 13. share_vs_s ----
SHARE_S = (0.0, 50.0, 100.0, 150.0, 260.0, 300.0, 400.0)


def _share_vs_s(d, path):
    s = np.linspace(0, 400, 401)
    yaw = np.array([d.gentle.ego_pose(v)[2] for v in s])
    dots = np.array(SHARE_S)
    dot_yaw = np.array([d.gentle.ego_pose(v)[2] for v in dots])
    states = [BicycleState(*d.gentle.ego_pose(v), 8.0, 0.0) for v in dots]
    shares = {name: 100 * np.array([_share(d, fn, st) for st in states]) for name, fn in (("the swap", swap), ("no rotation", no_rotation))}
    fig, (top, bottom) = _fig(8.0, 2)
    top.plot(s, yaw, color=THETA, lw=2.5)
    top.plot(dots, dot_yaw, "o", color=THETA, markersize=9)
    top.set(xlim=(-15, 450), ylim=(-0.3, 3.9), ylabel="yaw (rad)", title="the car's yaw")
    say(top, (400, dot_yaw[-1]), f"π = {math.pi:.4f}", dx=-14, dy=24, ha="right", colour=THETA)
    bottom.plot(dots, shares["the swap"], "o-", color=TEXT, lw=2.5, markersize=9)
    bottom.plot(dots, shares["no rotation"], "o--", color=TEXT, lw=2.5, markersize=9)
    bottom.set(xlim=(-15, 450), ylim=(-9, 48), xlabel="distance along gentle (m)", ylabel="pixels changed (%)")
    peak = int(np.argmax(shares["the swap"]))
    say(bottom, (dots[peak], shares["the swap"][peak]), "the swap", dx=-12, dy=16, ha="right")
    say(bottom, (260, shares["no rotation"][4]), "no rotation", dy=24)
    for x, y, text, dy, ha in ((0, shares["the swap"][0], "0.0%", 22, "left"), (50, shares["the swap"][1], "0.0%", -24, "center"),
                               (400, shares["no rotation"][-1], f"{shares['no rotation'][-1]:.1f}%", 0, "left"), (400, shares["the swap"][-1], f"{shares['the swap'][-1]:.1f}%", 0, "left")):
        say(bottom, (x, y), text, dx=12 if ha == "left" and x else 0, dy=dy, ha=ha)
    _save(fig, path)


# ---- 14. e_vs_s ----
def _e_vs_s(d, path):
    fig, (top, bottom) = _fig(8.2, 2)
    runs = (("right", RIGHT_RUN, "-"), ("swap", SWAP_RUN, "-."), ("no rotation", NO_ROT_RUN, "--"))
    half = abs(WHITE_LINE_LATERAL) / 2
    for ax, scen, upto in ((top, "straight", 160), (bottom, "gentle", 160)):
        for name, colour, style in runs:
            tel = d.tels[(name, scen)]
            keep = tel.progress_m <= upto
            ax.plot(tel.progress_m[keep], tel.lat[keep], style, color=colour, lw=2.5)
        ax.axhline(half, color=YELLOW, lw=2)
        ax.axhline(-half, color=WHITE, lw=2)
        ax.set(xlim=(-5, upto + 45), title=scen, ylabel="e (m)")
    say(top, (100, half), "yellow line", dy=14, ha="left", colour=YELLOW)
    say(top, (100, -half), "white line", dy=-14, ha="left", colour=WHITE)
    top.set_ylim(-2.4, 2.6)
    bottom.set_ylim(-2.4, 2.3)
    bottom.set_xlabel("distance along the road (m)")
    bottom.axvspan(100, 165, color=FAINT, alpha=0.35, lw=0)
    say(bottom, (132, -0.45), "first bend", colour=TEXT)
    for ax, scen in ((top, "straight"), (bottom, "gentle")):
        tel = d.tels[("right", scen)]
        k = int(np.argmin(np.abs(tel.progress_m - 160)))
        say(ax, (tel.progress_m[k], tel.lat[k]), "right\ntransform", dx=8, dy=-26, ha="left", colour=RIGHT_RUN)
    swap_s = d.tels[("swap", "straight")]
    top.plot(swap_s.progress_m[-1], swap_s.lat[-1], "x", color=SWAP_RUN, markersize=14, markeredgewidth=3)
    say(top, (swap_s.progress_m[-1], swap_s.lat[-1]), f"swap leaves\nat {swap_s.progress_m[-1]:.1f} m", dx=16, dy=-40, ha="left", colour=SWAP_RUN, arrow=True)
    nr = d.tels[("no rotation", "straight")]
    k = int(np.argmin(np.abs(nr.progress_m - 160)))
    say(top, (nr.progress_m[k], nr.lat[k]), "no\nrotation", dx=8, dy=44, ha="left", colour=NO_ROT_RUN)
    for name, colour, dy in (("swap", SWAP_RUN, 38), ("no rotation", NO_ROT_RUN, 36)):
        tel = d.tels[(name, "gentle")]
        bottom.plot(tel.progress_m[-1], tel.lat[-1], "x", color=colour, markersize=14, markeredgewidth=3)
        say(bottom, (tel.progress_m[-1], tel.lat[-1]), f"{'the swap' if name == 'swap' else name}\nleaves at {tel.progress_m[-1]:.1f} m", dx=16, dy=28 if name == 'swap' else -dy, ha="left", colour=colour, arrow=True)
    _save(fig, path)


# ---- 15. heading_flip ----
def _flip_data(d):
    sw = d.tels[("swap", "straight")]
    ok = np.isfinite(sw.est_heading) & (np.abs(sw.head_err) > 1e-3)
    idx = np.nonzero(ok)[0]
    right_h = np.array([estimate_lane(Observation(d.camera.render(d.straight, d.state(sw, k)), 8.0, 0.0)).heading_rad for k in idx])
    return sw.head_err[idx], sw.est_heading[idx], right_h


def _heading_flip(d, path):
    psi, swapped, right_h = _flip_data(d)
    valid = np.isfinite(right_h)
    fig, ax = _fig(5.9)
    span = 1.15 * max(np.abs(psi).max(), np.nanmax(np.abs(right_h)), np.abs(swapped).max())
    ax.plot([-span, span], [-span, span], "--", color=GREY, lw=1.5)
    ax.plot([-span, span], [span, -span], "--", color=GREY, lw=1.5)
    ax.plot(psi[valid], right_h[valid], "o", color=GREY, markersize=7)
    ax.plot(psi, swapped, "o", color=BOX, markersize=7)
    ax.set(xlim=(-span, span), ylim=(-span, span), xlabel="true heading ψ (rad)", ylabel="box heading (rad)")
    ax.set_aspect("equal", adjustable="box")
    say(ax, (span * 0.62, span * 0.62), "box = ψ", dx=-8, dy=18, ha="right", colour=GREY)
    say(ax, (-span * 0.62, span * 0.62), "box = −ψ", dx=8, dy=18, ha="left", colour=GREY)
    say(ax, (0, -span * 0.74), f"through the swap:\nmedian ratio {num(np.median(swapped / psi), sign=True)}", colour=BOX, badge=True)
    say(ax, (span * 0.66, 0), f"right transform:\n{num(np.nanmedian(right_h / psi), sign=True)}", colour=GREY, badge=True)
    _save(fig, path)


# ---- 16. park_box ----
def _park_box(d, path):
    pose = d.straight.ego_pose(50.0, 0.3, -0.05)
    state = BicycleState(*pose, 8.0, 0.0)
    road_yaw = d.straight.ego_pose(50.0)[2]
    far = np.array(pose[:2]) + 2000.0 * heading_vector(road_yaw)  # a point far down the road: its pixel is the road's vanishing point
    fig, axes = _fig(7.6, 2)
    for ax, camera, fn, name in ((axes[0], d.camera, world_to_ego, "right transform"), (axes[1], Camera(world_to_ego=swap), swap, "the swap")):
        image = camera.render(d.straight, state)
        box = estimate_lane(Observation(image, 8.0, 0.0))
        _picture(ax, image, f"{name}\nbox: {num(box.offset_m, 4)} m, {num(box.heading_rad, 4, True)} rad")
        ax.axvline(d.camera.cx, color=TEXT, lw=1.5)
        u, v = d.camera.ground_to_pixel(*fn(pose, far[None, :])[0])
        ax.plot(u, v, "o", mfc=PANEL, mec=TEXT, markersize=9, mew=2)
        say(ax, (u, v), "road's vanishing point", dx=14, dy=-26, ha="left", badge=True, arrow=True)
    _save(fig, path)


# ---- 17. next_lesson ----
def _next_lesson(d, path):
    s = np.linspace(0, 170, 400)
    pts = {name: d.gentle.offset(s, lateral) for name, lateral in (("yellow", 0.0), ("white", WHITE_LINE_LATERAL))}
    xlim, ylim = (-10, 165), (-12, 52)
    zoom_h = 3.4
    fig, (ax, near) = _fig(_panel(xlim, ylim) + zoom_h + 1.9, 2, [_panel(xlim, ylim) + 0.6, zoom_h])
    for name, colour in (("yellow", YELLOW), ("white", WHITE)):
        ax.plot(pts[name][:, 0], pts[name][:, 1], color=colour, ls=(0, (7, 3)) if name == "yellow" else "-", lw=2)
    x, y, _ = d.pose
    car_dot(ax, (x, y), markersize=6)
    path_s = np.linspace(0, d.s_car, 300)
    along = d.gentle.offset(path_s, 7.0)
    ax.plot(along[:, 0], along[:, 1], color=EGO, lw=2.5)
    for s_end in (0.0, d.s_car):
        a, b = d.gentle.offset(s_end, 4.0), d.gentle.offset(s_end, 7.0)
        ax.plot([a[0], b[0]], [a[1], b[1]], color=EGO, lw=2.5)
    mid = along[len(along) // 2]
    say(ax, tuple(mid), "?", dy=22, colour=EGO, badge=True)
    centre = d.gentle.offset(d.s_car, EGO_LANE_CENTER)
    ax.plot(*centre, "o", mfc=PANEL, color=EGO, markersize=14, markeredgewidth=2.5)
    say(ax, tuple(centre), "?", dx=34, dy=-34, colour=EGO, badge=True, arrow=True)
    ax.set(xlim=xlim, ylim=ylim, xlabel="east (m)", ylabel="north (m)")
    ax.set_aspect("equal", adjustable="box")
    near.set_gid(ZOOM)
    half = 3.0
    near_x, near_y = (x - half, x + half), (y - half, y + half)
    s_near = np.linspace(d.s_car - 8, d.s_car + 8, 100)
    for name, colour, style in (("yellow", YELLOW, (0, (7, 3))), ("white", WHITE, "-")):
        line = d.gentle.offset(s_near, 0.0 if name == "yellow" else WHITE_LINE_LATERAL)
        near.plot(line[:, 0], line[:, 1], color=colour, ls=style, lw=2.5)
    mid_line = d.gentle.offset(s_near, EGO_LANE_CENTER)
    near.plot(mid_line[:, 0], mid_line[:, 1], color=GREY, ls="--", lw=2)
    car_dot(near, (x, y))
    near.plot(*centre, "o", mfc=PANEL, color=EGO, markersize=10, markeredgewidth=2.5)
    gap = bracket(near, (x, y), tuple(centre), EGO, 0.25)
    say(near, tuple(gap), "?", dx=18, dy=-10, colour=EGO, badge=True)
    near.set(xlim=near_x, ylim=near_y, xticks=[], yticks=[], xlabel=f"the car and the lane centre, {2 * half:g} m across")
    near.set_aspect("equal", adjustable="box")
    ax.indicate_inset_zoom(near, edgecolor=GREY, linewidth=1.5)
    _save(fig, path)


# ---- 18. challenge_swing ----
def _turn_points(sw):
    kmin = int(np.argmin(sw.lat))
    zero_psi = kmin + int(np.argmax(sw.head_err[kmin:] > 0))
    kz = kmin + int(np.argmax(sw.lat[kmin:] > 0))
    tz = sw.t[kz - 1] + (sw.t[kz] - sw.t[kz - 1]) * (-sw.lat[kz - 1]) / (sw.lat[kz] - sw.lat[kz - 1])
    return kmin, float(sw.t[zero_psi]), float(tz)


def _challenge_swing(d, path):
    sw = d.tels[("swap", "straight")]
    kmin, t_psi, t_e = _turn_points(sw)
    keep = sw.t <= 2.5
    fig, (top, bottom) = _fig(8.0, 2)
    top.plot(sw.t[keep], sw.lat[keep], color=E_OFF, lw=2.5)
    top.plot(sw.t[kmin], sw.lat[kmin], "o", color=E_OFF, markersize=10)
    say(top, (sw.t[kmin], sw.lat[kmin]), f"{num(sw.lat[kmin], 4)} m at {sw.t[kmin]:.2f} s", dx=-30, dy=-4, ha="right", colour=E_OFF, badge=True, arrow=True)
    top.axhline(0, color=GREY, lw=1.2)
    top.plot(t_e, 0, "|", color=TEXT, markersize=18, markeredgewidth=3)
    say(top, (t_e, 0), f"{t_e:.2f} s", dx=-12, dy=-24, ha="right")
    top.set(xlim=(-0.05, 2.55), ylim=(sw.lat[kmin] - 0.03, 0.03), ylabel="e (m)", title="where the car is")
    bottom.plot(sw.t[keep], sw.head_err[keep], color=PSI, lw=2.5)
    bottom.axhline(0, color=GREY, lw=1.2)
    bottom.plot(t_psi, 0, "|", color=TEXT, markersize=18, markeredgewidth=3)
    say(bottom, (t_psi, 0), f"{t_psi:.2f} s", dy=22)
    bottom.set(xlim=(-0.05, 2.55), xlabel="time (s)", ylabel="ψ (rad)", title="which way it points")
    _save(fig, path)


# ---- 19. challenge_growth ----
def _step_map(sign):
    """One simulator step, small angles: x_(k+1) = Phi x_k with x = (e, psi). sign +1 is the right transform, -1 the swap."""
    v, L, ko, kh, dt = scenarios.TARGET_SPEED, dynamics.WHEELBASE, provided.K_OFF, provided.K_HEAD, dynamics.DT
    a = v * dt * (1 + v * dt / (2 * L))
    return np.array([[1 - a * ko, v * dt - a * sign * kh], [-(v * dt / L) * ko, 1 - (v * dt / L) * sign * kh]])


def _challenge_growth(d, path):
    sw, right = d.tels[("swap", "straight")], d.tels[("right", "straight")]
    first = int(np.argmax(np.abs(sw.lat) >= 0.005))
    last = int(np.nonzero(np.abs(sw.steer) < 0.1)[0][-1]) - 1
    pred = [np.array([sw.lat[first], sw.head_err[first]])]
    for _ in range(first, last):
        pred.append(_step_map(-1) @ pred[-1])
    pred = np.array(pred)
    keep = sw.t <= 2.5
    fig, ax = _fig(7.6)
    ax.plot(sw.t[keep], sw.lat[keep], color=E_OFF, lw=2.5)
    ax.plot(sw.t[first:last + 1], pred[:, 0], "--", color=TEXT, lw=2)
    ax.plot(right.t[right.t <= 2.5], right.lat[right.t <= 2.5], color=GREY, lw=2.5)
    ax.set(xlim=(-0.05, 2.55), ylim=(-0.41, 0.17), xlabel="time (s)", ylabel="e (m)")
    say(ax, (sw.t[last], pred[-1, 0]), f"model from\n{sw.t[first]:.2f} s", dx=-150, dy=-12, ha="right", colour=TEXT, arrow=True)
    say(ax, (sw.t[keep][-1], sw.lat[keep][-1]), "the swap", dx=-24, dy=-60, ha="right", colour=E_OFF)
    say(ax, (1.25, 0.12), f"right transform:\nwithin {math.ceil(1000 * np.abs(right.lat).max()) / 1000:.3f} m", colour=GREY)
    for tt, dx, dy, ha in ((1.0, 0, 78, "center"), (1.5, -6, -50, "right"), (2.0, 40, -62, "right")):
        k = int(np.argmin(np.abs(sw.t - tt)))
        ax.plot(sw.t[k], sw.lat[k], "o", color=E_OFF, markersize=9)
        say(ax, (sw.t[k], sw.lat[k]), f"model {num(pred[k - first, 0], 4)}\nmeasured {num(sw.lat[k], 4)}", dx=dx, dy=dy, ha=ha, colour=E_OFF, badge=True, arrow=True)
    _save(fig, path)


FIGURES = {"result_first": _result_first, "two_dots": _two_dots, "stacked_dots": _stacked_dots, "columns_axes": _columns_axes, "undo": _undo, "order_trap": _order_trap, "round_trip": _round_trip, "quads": _quads, "shape_trap": _shape_trap, "renderer_steps": _renderer_steps, "swap_same": _swap_same, "pictures_150": _pictures_150, "share_vs_s": _share_vs_s, "e_vs_s": _e_vs_s, "heading_flip": _heading_flip, "park_box": _park_box, "next_lesson": _next_lesson, "challenge_swing": _challenge_swing, "challenge_growth": _challenge_growth}


def make(out_dir):
    data = Data()
    with matplotlib.rc_context(STYLE):
        for name, draw in FIGURES.items():
            draw(data, out_dir / f"{name}.png")
