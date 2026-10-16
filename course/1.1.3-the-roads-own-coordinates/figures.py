"""Figures for lecture 1.1.3 The road's own coordinates.  `python scripts/make_figures.py course/1.1.3-the-roads-own-coordinates`."""
import math
from functools import lru_cache
from types import SimpleNamespace

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Arc, Rectangle
from matplotlib.text import Text
from matplotlib.textpath import TextPath
from matplotlib.ticker import FuncFormatter, MaxNLocator

from zero2fsd.car import Car, run
from zero2fsd.drawing import LABEL_GAP
from zero2fsd.geom import lane_errors, project_to_segment, to_road_frame, world_to_ego, wrap_angle
from zero2fsd.palette import CHROME, QUANTITY, SCENE, shades
from zero2fsd.sim import Road, make_scenario
from zero2fsd.sim.scenarios import GENTLE_RADIUS, TARGET_SPEED
from zero2fsd.sim.dynamics import DT
from zero2fsd.sim.world import EGO_LANE_CENTER, WHITE_LINE_LATERAL

TEXT, PANEL, BACKGROUND, BORDER = CHROME["text"], CHROME["panel"], CHROME["background"], CHROME["border"]
DIM = shades(BORDER, 5)[3]  # a quieter grey from the same chrome, for helper lines
BLUE, PINK, CORAL, PURPLE = QUANTITY["e_off"], QUANTITY["psi"], QUANTITY["perception"], QUANTITY["ego"]  # e and d, psi, the box's points, the ego frame
YELLOW, WHITE, ASPHALT = SCENE["yellow"], SCENE["white"], SCENE["road"]
LABEL_BOX = {"boxstyle": "round,pad=0.2", "fc": PANEL, "ec": BORDER}  # text that sits on top of a picture
CLEAR = (0, 0, 0, 0)
LANE = -EGO_LANE_CENTER  # 1.8 m: yellow line to lane centre, e = d + LANE
ZOOM_INSET = "zoom-inset"  # gid of an axes that deliberately shows only part of its data (the clipping gate skips it)
CENTRE = (100.0, GENTLE_RADIUS)  # the first bend's centre, (100, 80): the first straight is 100 m long
STEP = 400  # the worked step: 20.00 s into the provided car's lap

# Legibility (gate G8): every text is FS points on a PNG W_IN inches wide, shown 330 px wide on a phone.
FS, W_IN, SHOWN_PX, MIN_DIGIT_PX = 16, 5.5, 330, 9
RC = {"font.size": FS, "axes.titlesize": FS + 2, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
      "axes.titlepad": 8, "lines.linewidth": 2, "axes.linewidth": 1.2}
LEGIBILITY = {}


def _save(fig, path):
    """Save after checking, as gate G8 does, that no text is under MIN_DIGIT_PX tall when the PNG is shown SHOWN_PX wide."""
    dpi = matplotlib.rcParams["savefig.dpi"]
    scale = SHOWN_PX / (fig.get_figwidth() * dpi)
    px = min(TextPath((0, 0), "0", prop=t.get_fontproperties()).get_extents().height * dpi / 72 * scale
             for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip())
    assert px >= MIN_DIGIT_PX, f"{path.name}: smallest digit {px:.2f} px at {SHOWN_PX} px wide"
    LEGIBILITY[path.name] = round(px, 2)
    fig.savefig(path)


def _panels(specs, left=1.0, right=0.2, top=0.2, gap=1.0, bottom=0.9, width=W_IN):
    """A figure W_IN wide holding axes stacked top to bottom.  A spec is (xlim, ylim, height_in); a height of None gives
    the axes the height of equal scale (one data unit the same length across and up)."""
    panel_w = width - left - right
    heights = [h if h is not None else panel_w * abs(y[1] - y[0]) / abs(x[1] - x[0]) for x, y, h in specs]
    total = top + bottom + sum(heights) + gap * (len(specs) - 1)
    fig = Figure(figsize=(width, total))
    FigureCanvasAgg(fig)
    axes, y = [], total - top
    for (xlim, ylim, h), height in zip(specs, heights):
        ax = fig.add_axes([left / width, (y - height) / total, panel_w / width, height / total])
        ax.set(xlim=xlim, ylim=ylim)
        if h is None:
            ax.set_aspect("equal", adjustable="box")
        axes.append(ax)
        y -= height + gap
    return fig, axes


def _bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def _ticks(ax, nx=5, ny=5):
    ax.xaxis.set_major_locator(MaxNLocator(nx))
    ax.yaxis.set_major_locator(MaxNLocator(ny))


def _line(ax, x, y, **style):
    """Plot a line, leaving a gap wherever the data would leave the axes (so no trace is cut off by the limits)."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    x[(x < x0) | (x > x1)] = np.nan
    y[(y < y0) | (y > y1)] = np.nan
    x[np.isnan(y)] = np.nan
    y[np.isnan(x)] = np.nan
    return ax.plot(x, y, **style)


def _tag(ax, x, y, text, **kwargs):
    kwargs = {"ha": "center", "va": "center", **kwargs}
    return ax.text(x, y, text, bbox=LABEL_BOX, **kwargs)


def _dot(ax, x, y, colour, size=11):
    ax.plot(x, y, marker="o", ms=size, mfc=colour, mec=BACKGROUND, mew=2, ls="none", zorder=6)


def _pointer(ax, text, xy, xytext, colour=TEXT, **kwargs):
    kwargs = {"ha": "center", "va": "center", "bbox": LABEL_BOX, **kwargs}
    return ax.annotate(text, xy=xy, xytext=xytext, color=colour, arrowprops={"arrowstyle": "->", "color": colour, "lw": 2, "shrinkA": 2, "shrinkB": 3}, **kwargs)


def _brace(ax, p, q, colour, lw=3):
    """A dimension line from p to q with a tick at each end."""
    ax.annotate("", xy=q, xytext=p, arrowprops={"arrowstyle": "|-|", "color": colour, "lw": lw, "shrinkA": 0, "shrinkB": 0, "mutation_scale": 7})


def _right_angle(ax, corner, along, across, size=0.25, colour=TEXT):
    """A small square at `corner` between the unit vectors `along` and `across`."""
    a, b = np.array(along) * size, np.array(across) * size
    c = np.array(corner)
    ax.plot(*zip(c + a, c + a + b, c + b), color=colour, lw=1.5)


def _arrow(ax, tail, tip, colour=TEXT, lw=3):
    ax.annotate("", xy=tip, xytext=tail, arrowprops={"arrowstyle": "-|>", "color": colour, "lw": lw, "shrinkA": 0, "shrinkB": 0, "mutation_scale": 18}, zorder=7)


def _frame(ax, pose, colour, name, length=2.0):
    """A frame's two axes as arrows labelled x and y, and its name at the origin (drawing.draw_frame, with annotate arrows:
    its FancyArrowPatch patches trip the clipping gate before the figure is drawn)."""
    x, y, theta = pose
    x_hat = np.array([math.cos(theta), math.sin(theta)])
    y_hat = np.array([-x_hat[1], x_hat[0]])
    origin = np.array([x, y])
    for axis, label in ((x_hat, "x"), (y_hat, "y")):
        _arrow(ax, origin, origin + length * axis, colour, lw=2)
        ax.text(*(origin + length * (1 + LABEL_GAP) * axis), label, color=colour, ha="center", va="center")
    if name:
        ax.text(*(origin - LABEL_GAP * length * (x_hat + y_hat)), name, color=colour, ha="center", va="center")


def _car(ax, pose, colour=TEXT, length=1.0):
    x, y, th = pose
    h = np.array([math.cos(th), math.sin(th)])
    _arrow(ax, np.array([x, y]) - length * h, np.array([x, y]), colour)
    _dot(ax, x, y, colour, 9)


def _xy_axes(ax):
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    _ticks(ax, 4, 4)


# ---------------------------------------------------------------- the lap and the road, computed once


@lru_cache(maxsize=None)
def _lap():
    road = make_scenario("gentle").road
    tel = run(Car(), "gentle", seed=0)
    located = [road._locate(x, y) for x, y in zip(tel.x, tel.y)]
    s, lateral, theta_road = (np.array([row[i] for row in located]) for i in range(3))
    return SimpleNamespace(road=road, tel=tel, t=np.asarray(tel.t), s=s, d=lateral, e=np.asarray(tel.lat), psi=np.asarray(tel.head_err),
                           theta_road=theta_road, pts=road._a.copy())


def _band(ax, road, s0, s1, step=0.5):
    """The road surface from the white line to the yellow line, between arc lengths s0 and s1."""
    s = np.arange(s0, s1 + step, step)
    left, right = road.offset(s, 0.0), road.offset(s, WHITE_LINE_LATERAL)
    ax.fill(np.r_[left[:, 0], right[::-1, 0]], np.r_[left[:, 1], right[::-1, 1]], color=ASPHALT, lw=0)


def _paint(ax, road, s0, s1, step=0.2, lw=3):
    s = np.arange(s0, s1 + step, step)
    for lateral, colour, style in ((0.0, YELLOW, "-"), (WHITE_LINE_LATERAL, WHITE, "-"), (EGO_LANE_CENTER, TEXT, "--")):
        p = road.offset(s, lateral)
        _line(ax, p[:, 0], p[:, 1], color=colour, lw=lw if style == "-" else 1.5, ls=style, zorder=3)


def _foot(lap, k):
    """(index of the nearest segment, t along it, foot point) for step k."""
    p = np.array([lap.tel.x[k], lap.tel.y[k]])
    a, b = lap.road._a, lap.road._b
    t, d = project_to_segment(a, b, p)
    i = int(np.argmin(d * d))
    return i, float(t[i]) if np.ndim(t) else float(t), a[i] + float(t[i]) * lap.road._u[i]


# ---------------------------------------------------------------- 1 result_first


def result_first(path):
    lap = _lap()
    road, k = lap.road, STEP
    car = np.array([lap.tel.x[k], lap.tel.y[k]])
    yaw = float(lap.tel.yaw[k])
    foot = road.offset(lap.s[k], 0.0)
    lane_pt = road.offset(lap.s[k], EGO_LANE_CENTER)
    fig, (top, bot) = _panels([((-5, 185), (-10, 60), None), ((car[0] - 3, car[0] + 3), (car[1] - 3, car[1] + 3), None)], left=1.0, gap=1.0)
    bot.set_gid(ZOOM_INSET)  # the road surface is filled for 12 m, the limits show 6
    _band(top, road, 0, 200)
    _paint(top, road, 0, 200, lw=2)
    s = np.arange(0.0, lap.s[k], 0.2)
    p = road.offset(s, 0.0)
    top.plot(p[:, 0], p[:, 1], color=TEXT, lw=4, zorder=5)
    _dot(top, *car, TEXT, 11)
    _pointer(top, "159.011 m along the road", (60, 0.0), (85, 36))
    _xy_axes(top)
    _band(bot, road, lap.s[k] - 6, lap.s[k] + 6)
    _paint(bot, road, lap.s[k] - 6, lap.s[k] + 6)
    _car(bot, (*car, yaw))
    _brace(bot, lane_pt, car, BLUE)
    _pointer(bot, "0.528 m left of\nthe lane centre", 0.5 * (lane_pt + car), (car[0] + 0.9, car[1] - 1.9), colour=BLUE)
    _xy_axes(bot)
    assert round(lap.s[k], 3) == 159.011 and round(lap.e[k], 3) == 0.528 and round(lap.t[k], 2) == 20.0
    _save(fig, path)


# ---------------------------------------------------------------- 2 naive_y


def naive_y(path):
    lap = _lap()
    naive = lap.tel.y - EGO_LANE_CENTER
    diff = np.abs(naive - lap.e)
    k_max = int(np.argmax(diff))
    assert round(diff[k_max], 1) == 163.6 and round(lap.t[k_max], 2) == 53.8
    fig, (top, bot) = _panels([((0, 90), (-10, 180), 2.4), ((10, 15), (-0.4, 4.4), 2.6)], gap=1.0)
    top.plot(lap.t, naive, color=TEXT, lw=2)
    top.plot(lap.t, lap.e, color=BLUE, lw=2)
    _dot(top, lap.t[k_max], naive[k_max], TEXT)
    _tag(top, 36, 150, "163.6 m at 53.80 s")
    _tag(top, 58, 17, "e", color=BLUE)
    _tag(top, 25, 70, "y + 1.8", color=TEXT)
    top.set(xlabel="time (s)", ylabel="metres")
    _ticks(top, 5, 4)
    keep = (lap.t >= 10) & (lap.t <= 15)
    bot.plot(lap.t[keep], naive[keep], color=TEXT, lw=2)
    bot.plot(lap.t[keep], lap.e[keep], color=BLUE, lw=2)
    bot.axvline(12.5, color=DIM, ls="--", lw=2)
    _tag(bot, 11.5, 3.9, "first bend")
    for gap_m, text, where in ((0.01, "0.01 m at 12.70 s", (11.2, 1.1)), (0.1, "0.1 m at 13.05 s", (11.2, 2.0)), (1.0, "1 m at 14.10 s", (13.2, 3.0))):
        k = int(np.argmax(diff > gap_m))
        _dot(bot, lap.t[k], naive[k], TEXT, 9)
        _pointer(bot, text, (lap.t[k], naive[k]), where)
    bot.set(xlabel="time (s)", ylabel="metres")
    _ticks(bot, 5, 4)
    _save(fig, path)


# ---------------------------------------------------------------- 3 naive_nearest_y


def _cleverer_guess(lap):
    """The car's y minus the y of the nearest point of the lane-centre polyline."""
    lane = lap.road.offset(lap.road._s0, EGO_LANE_CENTER)
    a, b = lane, np.roll(lane, -1, axis=0)
    out = []
    for x, y in zip(lap.tel.x, lap.tel.y):
        t, d = project_to_segment(a, b, np.array([x, y]))
        i = int(np.argmin(d * d))
        seg = b[i] - a[i]
        out.append(y - (a[i, 1] + t[i] * seg[1] / math.hypot(*seg)))
    return np.array(out)


def naive_nearest_y(path):
    lap = _lap()
    guess = _cleverer_guess(lap)
    k = int(np.argmax(np.abs(guess - lap.e)))
    assert round(lap.t[k], 2) == 43.3 and round(guess[k], 3) == -0.528 and round(lap.e[k], 3) == 0.531
    fig, (ax,) = _panels([((0, 90), (-0.8, 0.8), 2.8)], bottom=0.9)
    ax.plot(lap.t, guess, color=TEXT, lw=2)
    ax.plot(lap.t, lap.e, color=BLUE, lw=2)
    ax.axhline(0, color=DIM, lw=1)
    _dot(ax, lap.t[k], guess[k], TEXT)
    _dot(ax, lap.t[k], lap.e[k], BLUE)
    _pointer(ax, "guess −0.528 m", (lap.t[k], guess[k]), (63, -0.62))
    _pointer(ax, "e +0.531 m", (lap.t[k], lap.e[k]), (63, 0.62), colour=BLUE)
    ax.set(xlabel="time (s)", ylabel="metres")
    _ticks(ax, 5, 5)
    _save(fig, path)


# ---------------------------------------------------------------- 4 polyline_zoom


def polyline_zoom(path):
    lap = _lap()
    road = lap.road
    i0 = int(np.searchsorted(road._s0, 200.0))  # the first vertex at or past s = 200 m: 11 vertices, 1 m of road
    pts = road._a[i0:i0 + 11]
    assert np.allclose(np.hypot(*np.diff(pts, axis=0).T), 0.1, atol=3e-5)
    centre = pts.mean(axis=0)
    fig, (top, bot) = _panels([((-90, 190), (-10, 170), None), ((centre[0] - 0.6, centre[0] + 0.6), (centre[1] - 0.6, centre[1] + 0.6), None)], gap=1.0)
    c = road._a
    top.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color=YELLOW, lw=3)
    top.add_patch(Rectangle((centre[0] - 4, centre[1] - 4), 8, 8, fc=CLEAR, ec=TEXT, lw=2, zorder=5))
    _pointer(top, "1 m of road,\nbox not to scale", (centre[0] - 4, centre[1]), (80, 55))
    _xy_axes(top)
    bot.plot(pts[:, 0], pts[:, 1], color=DIM, lw=3, zorder=2)
    bot.plot(pts[:, 0], pts[:, 1], marker="o", ms=9, mfc=TEXT, mec=BACKGROUND, mew=1.5, ls="none", zorder=4)
    along = (pts[5] - pts[4]) / 0.1
    side = np.array([-along[1], along[0]]) * 0.07
    _brace(bot, pts[4] + side, pts[5] + side, TEXT, lw=2)
    _tag(bot, *(0.5 * (pts[4] + pts[5]) + side / 0.07 * np.array([0.0, 0.0]) + np.array([-0.28, 0.0])), "0.1 m")
    _pointer(bot, "vertex", pts[8], (centre[0] + 0.33, centre[1] + 0.3))
    _pointer(bot, "segment", 0.5 * (pts[1] + pts[2]), (centre[0] + 0.33, centre[1] - 0.35))
    _xy_axes(bot)
    _save(fig, path)


# ---------------------------------------------------------------- the hand example, the clamp and the L


A, B, P, Q = np.array([2.0, 1.0]), np.array([10.0, 7.0]), np.array([5.0, 7.0]), np.array([14.0, 5.0])
U = (B - A) / np.hypot(*(B - A))
L_ROAD = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])


def segment_hand(path):
    t, d = project_to_segment(A, B, P)
    foot = A + float(t) * U
    assert np.allclose((t, d), (6.0, 3.0)) and np.allclose(foot, (6.8, 4.6))
    fig, (ax,) = _panels([((0, 12), (0, 8.5), None)], bottom=0.9)
    ax.plot(*zip(A, B), color=TEXT, lw=3)
    ax.grid(color=BORDER, lw=1)
    ax.set_axisbelow(True)
    ax.plot(*zip(foot, P), color=BLUE, lw=3)
    _right_angle(ax, foot, U, (-U[1], U[0]), 0.35)
    off = np.array([U[1], -U[0]]) * 0.45
    _brace(ax, A + off, foot + off, TEXT)
    _pointer(ax, "forward part 6.0", A + 3.0 * U + off, (8.2, 1.7))
    _tag(ax, 3.4, 5.8, "left part 3.0", color=BLUE)
    _tag(ax, 1.3, 1.9, "a (2, 1)")
    _tag(ax, 10.4, 7.7, "b (10, 7)")
    _tag(ax, 3.2, 3.8, "10 m")
    for pt in (A, B, P):
        _dot(ax, *pt, TEXT, 10)
    _tag(ax, 5.0, 7.8, "p (5, 7)")
    _dot(ax, *foot, BLUE, 10)
    _pointer(ax, "foot (6.8, 4.6)", foot, (9.3, 3.5))
    ax.set(xlabel="x (m)", ylabel="y (m)", xticks=np.arange(0, 13, 2), yticks=np.arange(0, 9, 2))
    _save(fig, path)


def clamp(path):
    t, d = project_to_segment(A, B, Q)
    assert np.allclose((t, d), (10.0, -4.4721), atol=1e-4)
    unclamped = A + 12.0 * U
    assert np.allclose(unclamped, (11.6, 8.2))
    fig, (top, bot) = _panels([((0, 16), (0, 11), None), ((0, 16), (0, 11), None)], top=0.5, gap=1.0)
    for ax in (top, bot):
        ax.plot(*zip(A, B), color=TEXT, lw=3)
        ax.grid(color=BORDER, lw=1)
        ax.set_axisbelow(True)
        for pt in (A, B, Q):
            _dot(ax, *pt, TEXT, 10)
        _tag(ax, 14.3, 2.9, "q (14, 5)")
        _tag(ax, 8.2, 8.3, "b (10, 7)")
        ax.set(xlabel="x (m)", ylabel="y (m)")
        _ticks(ax, 5, 5)
    top.set_title("Without the clamp")
    top.plot(*zip(B, unclamped), color=TEXT, lw=2, ls="--")
    top.plot(*zip(unclamped, Q), color=TEXT, lw=3)
    _dot(top, *unclamped, TEXT, 10)
    _tag(top, 12.0, 9.7, "foot (11.6, 8.2)")
    _tag(top, 14.4, 7.2, "4.0 m")
    bot.set_title("With the clamp")
    bot.plot(*zip(B, Q), color=BLUE, lw=3)
    _dot(bot, *B, BLUE, 10)
    _tag(bot, 3.6, 9.4, "forward part\ncut to 10", ha="center")
    _tag(bot, 11.2, 4.2, "4.472 m", color=BLUE)
    _save(fig, path)


def l_corner(path):
    p, tie = np.array([8.0, 3.0]), np.array([12.0, -2.0])
    a, b = L_ROAD[:-1], L_ROAD[1:]
    t, d = project_to_segment(a, b, p)
    assert np.allclose(d, (3.0, 2.0)) and np.isclose(math.hypot(*(tie - (10, 0))), 2.828, atol=5e-4)
    fig, (ax,) = _panels([((-1, 14), (-4, 12), None)], bottom=0.9)
    ax.add_patch(Rectangle((0, 0), 10, 10, fc=TEXT, alpha=0.10, ec=CLEAR, lw=0))
    ax.plot(*L_ROAD.T, color=TEXT, lw=4, zorder=3)
    ax.plot(*zip(p, (8.0, 0.0)), color=TEXT, lw=2, ls="--")
    ax.plot(*zip(p, (10.0, 3.0)), color=BLUE, lw=3)
    ax.plot(*zip(tie, (10.0, 0.0)), color=TEXT, lw=2, ls="--")
    for pt in (p, tie, (8.0, 0.0), (10.0, 3.0), (10.0, 0.0)):
        _dot(ax, *pt, TEXT, 9)
    _tag(ax, 5.5, 7.2, "left, +")
    _tag(ax, 12.2, 5.5, "right, −")
    _tag(ax, 6.2, 3.2, "p (8, 3)")
    _tag(ax, 6.8, 1.4, "3.0 m")
    _tag(ax, 11.6, 1.7, "(10, 3)\n2.0 m", color=BLUE)
    _tag(ax, 8.0, -1.0, "(8, 0)")
    _tag(ax, 12.0, -3.2, "(12, −2)")
    _tag(ax, 7.2, -2.9, "2.828 m, a tie")
    ax.set(xlabel="x (m)", ylabel="y (m)")
    _ticks(ax, 5, 5)
    _save(fig, path)


def d_sideways(path):
    f1, f2 = 2.64, 7.19
    fig, (ax,) = _panels([((-4, 9), (-4.6, 1.2), None)], bottom=0.9)
    ax.add_patch(Rectangle((-4, -3.6), 13, 3.6, fc=ASPHALT, ec=CLEAR, lw=0, zorder=0))
    ax.axhline(0, color=YELLOW, lw=4)
    ax.axhline(WHITE_LINE_LATERAL, color=WHITE, lw=4)
    ax.axhline(EGO_LANE_CENTER, color=DIM, lw=1.5, ls="--")
    _car(ax, (0.0, EGO_LANE_CENTER, 0.0), TEXT, 1.2)
    for f in (f1, f2):
        _dot(ax, f, EGO_LANE_CENTER, CORAL, 12)
    _brace(ax, (0.0, -2.35), (f1, -2.35), CORAL)
    _brace(ax, (0.0, -3.05), (f2, -3.05), CORAL)
    _tag(ax, 2.4, -0.8, "d₁ = 2.64 m", color=CORAL)
    _tag(ax, 7.0, -0.8, "d₂ = 7.19 m", color=CORAL)
    _tag(ax, 3.6, -4.2, "forward", color=CORAL)
    _brace(ax, (-0.9, 0.0), (-0.9, EGO_LANE_CENTER), BLUE)
    _tag(ax, -2.35, -0.9, "d = −1.8 m", color=BLUE)
    _tag(ax, -2.3, -2.7, "sideways", color=BLUE)
    ax.set(xlabel="x (m)", ylabel="y (m)")
    _ticks(ax, 5, 4)
    _save(fig, path)


def arc_length_l(path):
    p = np.array([8.0, 3.0])
    fig, (ax,) = _panels([((-1, 14), (-3, 12), None)], bottom=0.9)
    ax.plot(*L_ROAD.T, color=TEXT, lw=4, zorder=3)
    for pt, label, where in (((0, 0), "s = 0", (0.6, -1.4)), ((10, 0), "s = 10", (11.7, -0.9)), ((10, 10), "s = 20", (11.7, 10.6))):
        _dot(ax, *pt, TEXT, 10)
        _tag(ax, *where, label)
    _brace(ax, (10.8, 0.0), (10.8, 3.0), TEXT)
    _tag(ax, 12.4, 1.5, "t = 3 m")
    _tag(ax, 4.2, 6.5, "s = 10 + 3 = 13")
    ax.plot(*zip((10.0, 3.0), p), color=BLUE, lw=3)
    _dot(ax, *p, TEXT, 9)
    _tag(ax, 8.6, 4.2, "d = +2", color=BLUE)
    ax.set(xlabel="x (m)", ylabel="y (m)")
    _ticks(ax, 5, 5)
    _save(fig, path)


# ---------------------------------------------------------------- the road frame on gentle


def step400_foot(path):
    lap = _lap()
    k = STEP
    road = lap.road
    i, t, foot = _foot(lap, k)
    car = np.array([lap.tel.x[k], lap.tel.y[k]])
    assert i == 1590 and round(t, 4) == 0.0279 and round(float(road._s0[i]), 3) == 158.983
    mid = 0.5 * (car + foot)
    half = 2.0
    fig, (ax,) = _panels([((mid[0] - half, mid[0] + half), (mid[1] - half, mid[1] + half), None)], bottom=0.9)
    idx = (np.arange(i - 30, i + 31)) % len(road._a)
    pts = road._a[idx]
    _line(ax, pts[:, 0], pts[:, 1], color=YELLOW, lw=2, zorder=2)
    _line(ax, pts[:, 0], pts[:, 1], marker="o", ms=6, mfc=DIM, mec=DIM, ls="none", zorder=3)
    ax.plot(*zip(road._a[i], road._b[i]), color=TEXT, lw=5, zorder=4)
    ax.plot(*zip(foot, car), color=BLUE, lw=3, zorder=4)
    _dot(ax, *car, TEXT, 11)
    _dot(ax, *foot, BLUE, 10)
    n = np.array([-road._u[i][1], road._u[i][0]])
    _pointer(ax, "segment 1590\nof 7028", road._b[i] + 0.0 * n, (mid[0] + 1.1, mid[1] + 1.45))
    _pointer(ax, "a: s₀ = 158.983 m", road._a[i], (mid[0] - 1.1, mid[1] + 1.5))
    _pointer(ax, "t = 0.0279 m\ns = 159.011 m", foot, (mid[0] - 0.4, mid[1] - 1.45))
    _tag(ax, mid[0] + 1.1, mid[1] - 1.0, "d = −1.2717 m", color=BLUE)
    _xy_axes(ax)
    _save(fig, path)


def road_grid(path):
    lap = _lap()
    road = lap.road
    fig, (ax,) = _panels([((90, 190), (-10, 90), None)], bottom=0.9)
    _band(ax, road, 100, 225)
    for s in np.arange(100, 226, 10):
        r = road.offset(s, np.array([1.0, -4.6]))
        ax.plot(r[:, 0], r[:, 1], color=TEXT, lw=1.5, zorder=4)
        for end in r:
            _line(ax, [end[0], CENTRE[0]], [end[1], CENTRE[1]], color=DIM, lw=1, ls=":", zorder=1)
    ss = np.arange(100, 225.7, 0.5)
    for lateral, colour, style, lw in ((0.0, YELLOW, "-", 3), (EGO_LANE_CENTER, TEXT, "--", 2), (WHITE_LINE_LATERAL, WHITE, "-", 3)):
        p = road.offset(ss, lateral)
        ax.plot(p[:, 0], p[:, 1], color=colour, ls=style, lw=lw, zorder=3)
    _dot(ax, *CENTRE, TEXT, 9)
    _tag(ax, 115, 80, "(100, 80)")
    for s in (120, 160, 200):
        x, y = road.offset(s, -4.6)
        _tag(ax, x + 4.0 * math.sin((s - 100) / 80) , y - 4.0 * math.cos((s - 100) / 80) , f"s = {s}", ha="right" if s == 200 else "left")
    ax.set(xlabel="x (m)", ylabel="y (m)")
    _ticks(ax, 4, 4)
    _save(fig, path)


def e_from_d(path):
    fig, (ax,) = _panels([((-5.2, 1.4), (0, 5.0), 3.3)], left=0.2, right=0.2, bottom=0.2)
    _bare(ax)
    for spine in ax.spines.values():
        spine.set_visible(False)
    y_d, y_e = 4.0, 1.5
    for y, name in ((y_d, "d (m)"), (y_e, "e (m)")):
        ax.plot([-4.6, 0.8], [y, y], color=TEXT, lw=2)
        ax.text(-5.1, y + 0.5, name, ha="left", va="center")
    marks = ((-3.6, "white line", "−3.6", "−1.8"), (-1.8, "lane centre", "−1.8", "0"), (0.0, "yellow line", "0", "+1.8"))
    for x, name, d_label, e_label in marks:
        ax.plot([x, x], [y_d - 0.85, y_e], color=DIM, lw=1.5, zorder=1)
        ax.plot(x, y_d, marker="|", ms=14, color=TEXT, mew=2)
        ax.plot(x, y_e, marker="|", ms=14, color=TEXT, mew=2)
        ax.text(x, y_d - 0.38, d_label, ha="center", va="top")
        ax.text(x, y_e - 0.38, e_label, ha="center", va="top")
    for x0, x1 in ((0.0, 0.8), (-4.6, -3.6)):
        for y in (y_d, y_e):
            ax.add_patch(Rectangle((x0, y - 0.25), x1 - x0, 0.5, fc=CLEAR, ec=DIM, hatch="//", lw=0))
    _tag(ax, 0.4, y_d + 0.75, "off road", fontsize=FS, ha="center")
    _tag(ax, -2.7, (y_d + y_e) / 2, "e = d + 1.8", fontsize=18, ha="center")
    for y in (y_d, y_e):
        _dot(ax, -1.2717, y, BLUE, 12)
    ax.text(-2.4, 0.15, "car at step 400:\nd = −1.2717, e = +0.5283", color=BLUE, ha="center", va="center")
    _save(fig, path)


# ---------------------------------------------------------------- the frame at the foot, the wrap and the reproduction


def foot_frame(path):
    lap = _lap()
    d, psi = float(lap.d[STEP]), float(lap.psi[STEP])
    assert round(d, 4) == -1.2717 and round(psi, 4) == -0.0339
    fig, (ax,) = _panels([((-2, 4), (-2.5, 2.4), None)], bottom=0.2)
    _bare(ax)
    _frame(ax, (0.0, 0.0, 0.0), TEXT, "road frame at the foot", length=1.6)
    car = np.array([0.0, d])
    tip = car + 3.0 * np.array([math.cos(psi), math.sin(psi)])
    ax.plot(*zip(car, car + (3.0, 0.0)), color=DIM, lw=2, ls="--")
    _arrow(ax, car, tip, PINK)
    theta = np.linspace(psi, 0.0, 20)
    ax.plot(car[0] + 2.0 * np.cos(theta), car[1] + 2.0 * np.sin(theta), color=PINK, lw=3, zorder=5)
    _dot(ax, *car, TEXT, 11)
    _pointer(ax, "ψ = −0.0339 rad", (2.0, d), (2.7, -0.6), colour=PINK)
    _tag(ax, 1.0, -2.1, "(0, d) = (0, −1.2717)")
    _save(fig, path)


def wrap_trap(path):
    lap = _lap()
    raw = np.asarray(lap.tel.yaw) - lap.theta_road
    wrong = np.abs(raw - lap.psi) > 1e-9
    jump = int(np.argmax(np.abs(np.diff(raw)) > 3)) + 1
    assert round(float(np.max(np.abs(lap.psi))), 3) == 0.057 and round(lap.t[jump], 2) == 56.95 and round(raw[jump] - raw[jump - 1] - lap.psi[jump] + lap.psi[jump - 1], 4) == 6.2832
    assert (int(wrong.sum()), len(wrong)) == (639, 1778)
    fig, (ax,) = _panels([((0, 90), (-0.6, 7.2), 3.0)], left=1.5, bottom=0.9)
    ax.plot(lap.t, raw, color=TEXT, lw=2)
    ax.plot(lap.t, lap.psi, color=PINK, lw=3)
    ax.set_yticks([0.0, 2 * math.pi])
    ax.set_yticklabels(["0", "6.2832"])
    _tag(ax, 28, 0.8, "wrap(θ − θ_road)", color=PINK)
    _tag(ax, 28, 5.4, "θ − θ_road")
    _pointer(ax, "jumps by\n6.2832 rad\nat 56.95 s", (lap.t[jump], 3.2), (28, 3.2))
    ax.set(xlabel="time (s)", ylabel="angle (rad)")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    _save(fig, path)


def reproduced(path):
    lap = _lap()
    road, pts = lap.road, lap.pts
    s_mine, d_mine, e_mine, psi_mine = (np.empty(len(lap.t)) for _ in range(4))
    for k, (x, y, yaw) in enumerate(zip(lap.tel.x, lap.tel.y, lap.tel.yaw)):
        s_mine[k], d_mine[k] = to_road_frame(pts, True, (x, y))
        e_mine[k], psi_mine[k] = lane_errors(pts, True, (x, y, yaw))
    diffs = (np.abs(s_mine - lap.s), np.abs(d_mine - lap.d), np.abs(e_mine - lap.e), np.abs(psi_mine - lap.psi))
    assert max(float(v.max()) for v in diffs) == 0.0 and round(float(lap.e.max()), 3) == 0.533
    fig, (top, bot) = _panels([((0, 90), (-0.1, 0.7), 2.4), ((0, 90), (-1e-9, 1e-9), 2.2)], left=1.3, gap=1.0)
    top.plot(lap.t, lap.e, color=TEXT, lw=5)
    top.plot(lap.t, e_mine, color=BLUE, lw=2)
    k = int(np.argmax(lap.e))
    _dot(top, lap.t[k], lap.e[k], BLUE, 9)
    _pointer(top, "peak +0.533 m", (lap.t[k], lap.e[k]), (62, 0.22))
    top.set(xlabel="time (s)", ylabel="e (m)")
    _ticks(top, 5, 5)
    for v, colour, lw in zip(diffs, (TEXT, CORAL, BLUE, PINK), (8, 6, 4, 2)):
        bot.plot(lap.t, v, color=colour, lw=lw)
    bot.set_yticks([-1e-9, 0.0, 1e-9])
    bot.set_yticklabels(["−1e-9", "0", "1e-9"])
    _tag(bot, 45, 0.5e-9, "s, d, e and ψ minus the\nsimulator's: all exactly 0")
    bot.set(xlabel="time (s)", ylabel="difference")
    bot.xaxis.set_major_locator(MaxNLocator(5))
    _save(fig, path)


# ---------------------------------------------------------------- spacing of vertices


def _five_metre_foot(lap, k):
    """(nearest vertex, foot on the nearest 5 m segment, both of the 5 m polyline) for step k."""
    pts = lap.road.sample_points(5.0)
    a, b = pts, np.roll(pts, -1, axis=0)
    car = np.array([lap.tel.x[k], lap.tel.y[k]])
    vertex = pts[int(np.argmin(np.hypot(*(pts - car).T)))]
    t, d = project_to_segment(a, b, car)
    i = int(np.argmin(d * d))
    return pts, vertex, a[i] + float(t[i]) * (b[i] - a[i]) / math.hypot(*(b[i] - a[i])), car


def sparse_vertex(path):
    lap = _lap()
    k = int(np.argmin(np.abs(lap.t - 43.30)))
    pts, vertex, foot, car = _five_metre_foot(lap, k)
    to_vertex, to_foot = math.hypot(*(vertex - car)), math.hypot(*(foot - car))
    true_d = abs(float(lap.d[k]))
    assert round(to_vertex - true_d, 3) == 1.550 and round(abs(to_foot - true_d), 3) == 0.039
    half = 4.0
    fig, (ax,) = _panels([((car[0] - half, car[0] + half), (car[1] - half, car[1] + half), None)], bottom=0.9)
    arc = lap.pts
    _line(ax, arc[:, 0], arc[:, 1], color=YELLOW, lw=3, zorder=2)
    closed = np.vstack([pts, pts[:1]])
    _line(ax, closed[:, 0], closed[:, 1], color=DIM, lw=3, zorder=3)
    _line(ax, pts[:, 0], pts[:, 1], marker="o", ms=11, mfc=TEXT, mec=BACKGROUND, mew=2, ls="none", zorder=5)
    ax.plot(*zip(car, vertex), color=TEXT, lw=2.5, ls="--", zorder=4)
    ax.plot(*zip(car, foot), color=BLUE, lw=3, zorder=6)
    _dot(ax, *car, TEXT, 11)
    _pointer(ax, "nearest vertex:\n1.550 m out", 0.5 * (car + vertex), (car[0] + 1.0, car[1] - 3.0))
    _pointer(ax, "segment:\n0.039 m out", 0.5 * (car + foot), (car[0] - 1.8, car[1] + 1.6), colour=BLUE)
    _xy_axes(ax)
    _save(fig, path)


def sagitta(path):
    radius, chord = float(GENTLE_RADIUS), 5.0
    u = np.linspace(-chord / 2, chord / 2, 61)
    v = np.sqrt(radius**2 - u**2) - math.sqrt(radius**2 - (chord / 2) ** 2)
    sag = radius - math.sqrt(radius**2 - (chord / 2) ** 2)  # the arc's height above the chord at the middle
    assert round(sag, 4) == 0.0391 and round(radius - sag, 4) == 79.9609
    assert chord**2 / (8 * radius) == 25 / 640
    fig, (ax,) = _panels([((-3.4, 3.4), (-0.03, 0.17), 3.4)], bottom=0.2)
    _bare(ax)
    ax.plot(u, 0 * u, color=DIM, lw=3, zorder=2)
    ax.plot(u, v, color=YELLOW, lw=3, zorder=3)
    for x in (-chord / 2, chord / 2):
        _dot(ax, x, 0.0, TEXT, 11)
    _brace(ax, (0.0, 0.0), (0.0, sag), BLUE)
    _pointer(ax, "0.0391 m", (0.0, sag / 2), (1.5, 0.075), colour=BLUE)
    _tag(ax, 0.0, -0.014, "5 m apart")
    _tag(ax, -0.1, 0.143, "chord middle: 79.9609 m\nfrom the centre, not 80")
    _tag(ax, 0.0, 0.105, "c² / (8R) = 25 / 640")
    _tag(ax, -1.9, 0.07, "sideways scale\nstretched")
    _save(fig, path)


def _largest_d_error(lap, spacing):
    """(largest |nearest-vertex distance - |d||, largest |segment projection d - d|) over the lap with a vertex every `spacing` m."""
    pts = lap.road.sample_points(spacing)
    coarse = Road(pts, True)
    to_vertex = max(abs(float(np.min(np.hypot(*(pts - (x, y)).T))) - abs(d)) for x, y, d in zip(lap.tel.x, lap.tel.y, lap.d))
    to_segment = max(abs(coarse.project(x, y)[1] - d) for x, y, d in zip(lap.tel.x, lap.tel.y, lap.d))
    return to_vertex, to_segment


def spacing(path):
    lap = _lap()
    gaps = (1.0, 5.0, 10.0)
    errors = [_largest_d_error(lap, c) for c in gaps]
    assert [round(e[0], 3) for e in errors] == [0.096, 1.550, 3.924] and [round(e[1], 4) for e in errors] == [0.0016, 0.0391, 0.1562]
    fig, (ax,) = _panels([((0.0, 12.5), (3e-4, 14.0), 3.5)], left=1.15, bottom=0.9)
    ax.set_yscale("log")
    c = np.linspace(0.8, 11.0, 80)
    ax.plot(c, c**2 / (8 * GENTLE_RADIUS), color=DIM, lw=3, zorder=2)
    ax.plot(gaps, [e[0] for e in errors], marker="o", ms=11, mfc=TEXT, mec=BACKGROUND, mew=2, ls="none", zorder=5)
    ax.plot(gaps, [e[1] for e in errors], marker="o", ms=11, mfc=BLUE, mec=BACKGROUND, mew=2, ls="none", zorder=6)
    for g, (near, seg) in zip(gaps, errors):
        _tag(ax, 8.0 if g == 10.0 else g, near if g == 10.0 else near * 2.6, f"{near:.3f}")
        _tag(ax, g, seg / 2.8, f"{seg:.4f}", color=BLUE)
    _pointer(ax, "c² / (8 × 80)", (7.0, 49 / 640), (8.4, 0.006))
    ax.set_xticks(gaps)
    ax.set_xticklabels(["1", "5", "10"])
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1.0])
    ax.set_yticklabels(["0.001", "0.01", "0.1", "1"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(color=BORDER, lw=1)
    ax.set_axisbelow(True)
    ax.set(xlabel="spacing between vertices (m)", ylabel="largest error in d (m)")
    _save(fig, path)


def vertex_tie(path):
    lap = _lap()
    road = lap.road
    k = 372
    car = np.array([lap.tel.x[k], lap.tel.y[k]])
    vertex = road._b[1479]
    assert round(lap.t[k], 2) == 18.6 and np.allclose(vertex, road._a[1480]) and road.project(*car)[0] == lap.s[k]
    h1, h2 = float(road._heading[1479]), float(road._heading[1480])
    assert (round(h1, 5), round(h2, 5), round(h2 - h1, 5)) == (0.59920, 0.60045, 0.00125)
    assert round(float(np.hypot(*(car - vertex))), 2) == 1.27
    stretch, reach = 600.0, 2.3
    half = stretch * (h2 - h1) / 2  # each segment tilts this far from the bisector, drawn 200 times too large
    tail = np.array([-reach * math.cos(half), reach * math.sin(half)])
    head = np.array([reach * math.cos(half), reach * math.sin(half)])
    fig, (ax,) = _panels([((-2.7, 2.7), (-2.0, 2.5), None)], bottom=0.2)
    _bare(ax)
    ax.plot(*zip(tail, (0.0, 0.0)), color=TEXT, lw=4, zorder=3)
    ax.plot(*zip((0.0, 0.0), head), color=DIM, lw=4, zorder=3)
    ax.plot(*zip((0.0, 0.0), -tail), color=TEXT, lw=2, ls="--", zorder=2)
    theta = np.linspace(-half, half, 20)
    ax.plot(1.1 * np.cos(theta), 1.1 * np.sin(theta), color=PINK, lw=3, zorder=5)
    ax.plot(*zip((0.0, 0.0), (0.0, -1.27)), color=TEXT, lw=2, ls="--", zorder=2)
    _dot(ax, 0.0, 0.0, YELLOW, 12)
    _dot(ax, 0.0, -1.27, TEXT, 12)
    _tag(ax, -1.35, 1.75, "1479: 0.59920 rad\nthe simulator\nkeeps this one")
    _tag(ax, 1.3, 1.75, "1480:\n0.60045 rad")
    _pointer(ax, "one corner angle:\n0.00125 rad", (1.1, 0.0), (1.4, -1.5), colour=PINK)
    _tag(ax, -1.6, -0.45, "angle stretched")
    _tag(ax, -1.3, -1.3, "car, 1.27 m right")
    _save(fig, path)


# ---------------------------------------------------------------- the next lesson and the first challenge


def next_lesson(path):
    step = TARGET_SPEED * DT
    assert round(step, 2) == 0.40
    fig, (ax,) = _panels([((0, 10), (0, 4.6), None)], left=0.2, bottom=0.2)
    _bare(ax)
    ax.add_patch(Rectangle((0, 1.7), 10, 1.3, fc=ASPHALT, ec=CLEAR, lw=0, zorder=1))
    ax.add_patch(Rectangle((5.0, 1.3), 4.6, 2.1, fc=CLEAR, ec=TEXT, lw=3, zorder=2))
    for x in np.arange(0.6, 4.8, 0.9):
        _dot(ax, x, 2.35, TEXT, 11)
    for x in np.arange(5.4, 9.0, 0.7):
        _arrow(ax, (x, 2.35), (x + 0.7, 2.35), CORAL, lw=3)
    _tag(ax, 2.5, 3.9, "GPS fixes")
    _tag(ax, 7.3, 4.0, "tunnel")
    _pointer(ax, "one step as the car\nmeasured it: 0.40 m", (6.1, 2.35), (4.5, 0.55), colour=CORAL)
    _save(fig, path)


def challenge_setup(path):
    lap = _lap()
    road = lap.road
    x, y, yaw = road.ego_pose(260.0, 0.3, 0.04)
    centre = (174.108, 113.916)
    assert abs(x - centre[0]) < 0.6 and abs(y - centre[1]) < 0.6
    half = 6.0
    fig, (ax,) = _panels([((centre[0] - half, centre[0] + half), (centre[1] - half, centre[1] + half), None)], bottom=0.9)
    s = np.arange(240.0, 285.0, 0.1)
    for lateral, colour, style, lw in ((0.0, YELLOW, "-", 4), (EGO_LANE_CENTER, TEXT, "--", 2)):
        p = road.offset(s, lateral)
        _line(ax, p[:, 0], p[:, 1], color=colour, lw=lw, ls=style, zorder=2)
    car = np.array([x, y])
    lane = road.offset(260.0, EGO_LANE_CENTER)
    ahead = np.array([math.cos(yaw), math.sin(yaw)])
    road_dir = np.array([math.cos(yaw - 0.04), math.sin(yaw - 0.04)])
    ax.plot(*zip(car, car + 3.4 * road_dir), color=DIM, lw=2, ls="--", zorder=3)
    theta = np.linspace(yaw - 0.04, yaw, 12)
    ax.plot(car[0] + 3.0 * np.cos(theta), car[1] + 3.0 * np.sin(theta), color=PINK, lw=4, zorder=5)
    _brace(ax, lane, car, BLUE)
    _dot(ax, *car, TEXT, 11)
    _frame(ax, (x, y, yaw), QUANTITY["ego"], "", length=2.0)
    left_of = np.array([-ahead[1], ahead[0]])
    ax.text(*(car + 0.65 * (ahead + left_of)), "ego", color=QUANTITY["ego"], ha="center", va="center")
    _pointer(ax, "e = +0.3 m", 0.5 * (lane + car), (car[0] + 2.2, car[1] - 2.6), colour=BLUE)
    _pointer(ax, "ψ = +0.04 rad", car + 3.0 * np.array([math.cos(yaw - 0.02), math.sin(yaw - 0.02)]), (car[0] + 2.4, car[1] + 3.2), colour=PINK)
    _xy_axes(ax)
    _save(fig, path)


FIGURES = [(fn.__name__, fn) for fn in (
    result_first, naive_y, naive_nearest_y, polyline_zoom, segment_hand, clamp, l_corner, d_sideways, arc_length_l, step400_foot, road_grid,
    e_from_d, foot_frame, wrap_trap, reproduced, sparse_vertex, sagitta, spacing, vertex_tie, next_lesson, challenge_setup)]


def make(out_dir):
    with matplotlib.rc_context(RC):
        for name, fn in FIGURES:
            fn(out_dir / f"{name}.png")
