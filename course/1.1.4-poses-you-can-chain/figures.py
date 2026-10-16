"""Figures for lecture 1.1.4 Poses you can chain.  `python scripts/make_figures.py course/1.1.4-poses-you-can-chain`.

Ruling F: every figure is 5.5 in wide, saved at the figstyle dpi, with its smallest text at FS = 16 pt (at least 2.7 x the
width in inches), panels stacked in one column.  The lap is the simulator's own `gentle` drive; the drift figures read the
arrays `measure.py` wrote next to this file (drift_fig7.npz, drift_fig9.npz, drift_laps.npz, rear_fig8.npz).
"""
import functools
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Arc, Circle, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

from zero2fsd.car import Car, run
from zero2fsd.drawing import draw_frame
from zero2fsd.geom import T_EGO_REAR, dead_reckon, invert_pose, pose_to_matrix, relative_pose
from zero2fsd.palette import CHROME, QUANTITY, shades
from zero2fsd.sim import make_scenario
from zero2fsd.sim.camera import Camera
from zero2fsd.sim.dynamics import DT, WHEELBASE
from zero2fsd.sim.scenarios import GENTLE_RADIUS

HERE = Path(__file__).parent
W, FS, TITLE = 5.5, 16, 18
STYLE = {"font.size": FS, "axes.labelsize": FS, "axes.titlesize": TITLE, "axes.titlepad": 8, "xtick.labelsize": FS, "ytick.labelsize": FS}
TEXT, PANEL = CHROME["text"], CHROME["panel"]
GREY = shades(CHROME["border"], 3)[2]  # world axes, true paths and yardsticks
FAINT = shades(CHROME["border"], 3)[1]
EGO, CAMERA, THETA, CORAL, BLUE = (QUANTITY[k] for k in ("ego", "camera", "theta", "perception", "e_off"))
REAR, BLUE_REAR = shades(EGO, 2)[1], shades(BLUE, 2)[1]
BOX = {"boxstyle": "round,pad=0.15", "fc": PANEL, "ec": PANEL}  # text that sits on top of a line
K_EX, K_STEP = 400, 100  # the lecture's example step, and the zoomed one
SIGMAS = ("0.0001", "0.001")


def _fig(height):
    return Figure(figsize=(W, height))


def _save(fig, path):
    FigureCanvasAgg(fig).draw()  # lay the axes out first, so an arrow patch's path is in data coordinates when the figure is checked
    fig.savefig(path)


@functools.cache
def _lap():
    tel = run(Car(), "gentle")
    X, Y, TH = (np.asarray(v, dtype=float) for v in (tel.x, tel.y, tel.yaw))
    Tw = np.stack([pose_to_matrix(p) for p in zip(X, Y, TH)])
    steps = np.stack([relative_pose(Tw[k], Tw[k + 1]) for k in range(len(X) - 1)])
    return SimpleNamespace(X=X, Y=Y, TH=TH, Tw=Tw, steps=steps, t=np.arange(len(X)) * DT, road=make_scenario("gentle").road,
                           fwd=steps[:, 0, 2], left=steps[:, 1, 2], dth=np.arctan2(steps[:, 1, 0], steps[:, 0, 0]))


def _pose(k):
    L = _lap()
    return float(L.X[k]), float(L.Y[k]), float(L.TH[k])


def _data(name):
    return np.load(HERE / f"{name}.npz")




# ---- small drawing helpers ----------------------------------------------------------------------------------------

AXW = 5.0  # width in inches of a plain, tickless panel


def _m(v, fmt):
    """A number with a true minus sign."""
    return f"{v:{fmt}}".replace("-", "\u2212")


def _note(ax, xy, text, colour=TEXT, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    return ax.text(*xy, text, color=colour, bbox=BOX if ax.axison else None, **kw)


def _label(ax, point, xy, text, colour, **kw):
    """Text at `xy` joined by a thin line to `point`."""
    kw.setdefault("ha", "center")
    ax.annotate(text, xy=point, xytext=xy, color=colour, va="center", bbox=BOX if ax.axison else None,
                arrowprops={"arrowstyle": "-", "color": colour, "lw": 1.5, "shrinkA": 0, "shrinkB": 4}, **kw)


def _arrow(ax, a, b, colour, lw=2.5, style="-|>", scale=FS, **kw):
    ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": style, "color": colour, "lw": lw, "shrinkA": 0, "shrinkB": 0, "mutation_scale": scale, **kw})


def _dot(ax, xy, colour, size=9):
    ax.plot([xy[0]], [xy[1]], marker="o", ms=size, color=colour, ls="")


def _frame(ax, pose, colour, length, name=""):
    n, m = len(ax.texts), len(ax.patches)
    draw_frame(ax, pose, colour, name, length_m=length)
    for arrow in list(ax.patches)[m:]:  # redraw each arrow as an annotation: a patch arrow's path is not in data coordinates, so the data-in-axes gate misreads it
        (x0, y0), (x1, y1) = arrow._posA_posB
        arrow.remove()
        _arrow(ax, (x0, y0), (x1, y1), colour, lw=matplotlib.rcParams["patch.linewidth"], scale=matplotlib.rcParams["font.size"])
    for t in ax.texts[n:]:
        if t.get_text():  # the arrows above are empty annotations
            t.set_bbox(BOX if ax.axison else None)


def _panels(specs, footer=0.0, gap=0.2, title_h=0.55):
    """Stack equal-aspect panels with no ticks. Each spec is (xlim, ylim, titled, width in inches); the box of every panel
    matches its data, so nothing is letterboxed. `footer` leaves that many inches below for text."""
    heights = [w * (y1 - y0) / (x1 - x0) + (title_h if titled else 0.0) for (x0, x1), (y0, y1), titled, w in specs]
    total = sum(heights) + gap * (len(specs) - 1) + 0.2 + footer
    fig = Figure(figsize=(W, total))
    top, axes = total - 0.1, []
    for ((xlim, ylim, titled, w), h) in zip(specs, heights):
        top -= h
        ax = fig.add_axes(((W - w) / 2 / W, top / total, w / W, (h - (title_h if titled else 0.0)) / total))
        ax.set(xlim=xlim, ylim=ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        axes.append(ax)
        top -= gap
    return fig, axes


def _footer(fig, lines, colour=TEXT, inch=0.1):
    fig.text(0.04, (inch + 0.0) / fig.get_figheight() + (lines.count("\n") + 1) * 0.0, lines, color=colour, ha="left", va="bottom")


def _metres(ax, east="east (m)", north="north (m)", n=4):
    ax.set(xlabel=east, ylabel=north)
    ax.xaxis.set_major_locator(MaxNLocator(n))
    ax.yaxis.set_major_locator(MaxNLocator(n))


def _vec(theta):
    return np.array([math.cos(theta), math.sin(theta)])


def _lap_axes(ax, lw=3):
    L = _lap()
    ax.plot(L.X, L.Y, color=GREY, lw=lw)
    ax.set_aspect("equal")


# ---- 1  rebuilt_lap --------------------------------------------------------------------------------------------

def rebuilt_lap(path):
    L = _lap()
    good = dead_reckon(L.Tw[0], L.steps)
    gap = float(np.hypot(good[:, 0, 2] - L.X, good[:, 1, 2] - L.Y).max())
    fig = _fig(4.3)
    ax = fig.subplots()
    _lap_axes(ax)
    ax.plot(good[:, 0, 2], good[:, 1, 2], color=CORAL, lw=3)
    ax.set(xlim=(-95, 200), ylim=(-50, 175))
    _metres(ax)
    mantissa, exponent = f"{gap:.1e}".split("e")
    _note(ax, (50, 122), "true path (grey)", GREY)
    _note(ax, (50, 92), f"rebuilt from {len(L.steps)}\nsteps (coral)", CORAL)
    _note(ax, (50, 52), f"max gap ${mantissa}\\times10^{{{int(exponent)}}}$ m")
    _note(ax, (112, -28), "start (0, \u22121.8),\nfacing east")
    _arrow(ax, (50, -22), (4, -4), TEXT, lw=1.5)
    fig.tight_layout()
    _save(fig, path)


# ---- 2  pose_frame ---------------------------------------------------------------------------------------------

def pose_frame(path):
    L = _lap()
    x, y, th = _pose(K_EX)
    cone = (L.Tw[K_EX] @ np.array([10.0, 2.0, 1.0]))[:2]
    f, n = _vec(th), _vec(th + math.pi / 2)
    fig, (top, bottom) = _panels([((-45, 200), (-22, 47), False, AXW), ((-6.5, 13.5), (-5.5, 14.5), False, AXW)])
    _frame(top, (0, 0, 0), GREY, 22)
    _note(top, (-14, -13), "world", GREY)
    _arrow(top, (0, 0), (x - 3, y - 0.4), GREY, lw=2)
    _dot(top, (x, y), EGO)
    _note(top, (x - 12, y + 19), f"({x:.2f}, {y:.2f})")
    _frame(bottom, (0, 0, th), EGO, 4, "ego")
    mid, tip = 10 * f, 10 * f + 2 * n
    _arrow(bottom, (0, 0), tuple(mid), EGO, lw=3.5)
    _arrow(bottom, tuple(mid), tuple(tip), EGO, lw=3.5)
    _note(bottom, tuple(5.5 * f + 1.9 * n), "10 m forward", EGO, rotation=math.degrees(th))
    _note(bottom, tuple(mid + 2.9 * f), "2 m left", EGO)
    _dot(bottom, tuple(tip), TEXT, size=11)
    _note(bottom, (tip[0] - 3.2, tip[1] + 3.2), f"cone\n({cone[0]:.2f}, {cone[1]:.2f})")
    bottom.plot([0, 9], [0, 0], color=GREY, lw=1.5, ls="--")
    bottom.add_patch(Arc((0, 0), 10, 10, theta1=0, theta2=math.degrees(th), color=GREY, lw=1.5))
    _note(bottom, (7.6, 1.6), f"{math.degrees(th):.2f}\u00b0", GREY)
    _save(fig, path)


# ---- 3  pose_matrix --------------------------------------------------------------------------------------------

def _triple(ax, x, y, values, colour):
    """A boxed column of three numbers with its top-left corner at (x, y), 0.3 in per row; returns its right edge."""
    width = 0.17 * max(len(v) for v in values) + 0.22
    ax.add_patch(Rectangle((x, y - 0.95), width, 0.95, fill=False, edgecolor=colour, lw=2))
    for i, v in enumerate(values):
        ax.text(x + width / 2, y - 0.17 - 0.3 * i, v, color=colour, ha="center", va="center")
    return x + width


def _cell(v):
    return "0" if abs(v) < 1e-12 else _m(v, ".4f").rstrip("0").rstrip(".") if abs(v - round(v)) < 1e-12 else _m(v, ".4f")


def pose_matrix(path):
    T = _lap().Tw[K_EX]
    cols = [[_cell(float(v)) for v in T[:, j]] for j in range(3)]
    result = T @ np.array([10.0, 2.0, 1.0])
    height = 4.0
    fig = _fig(height)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, W), ylim=(1.0, 5.0))
    ax.axis("off")
    ax.text(0.15, 4.5, "10 \u00d7", ha="left", va="center")
    r = _triple(ax, 0.9, 4.35, cols[0], EGO)
    ax.text(0.9 + (r - 0.9) / 2, 4.55, "forward axis", color=EGO, ha="center", va="bottom")
    ax.text(r + 0.12, 3.9, "+ 2 \u00d7", ha="left", va="center")
    x2 = r + 0.95
    r2 = _triple(ax, x2, 4.35, cols[1], EGO)
    ax.text(x2 + (r2 - x2) / 2, 4.55, "left axis", color=EGO, ha="center", va="bottom")
    ax.text(0.15, 2.55, "+ 1 \u00d7", ha="left", va="center")
    r3 = _triple(ax, 0.9, 2.55, cols[2], EGO)
    ax.text(0.9 + (r3 - 0.9) / 2, 1.43, "origin", color=EGO, ha="center", va="top")
    ax.text(r3 + 0.15, 2.05, "=", ha="left", va="center")
    _triple(ax, r3 + 0.6, 2.55, [f"{result[0]:.3f}", f"{result[1]:.4f}", "1"], TEXT)
    _save(fig, path)


# ---- 4  arrow_vs_point -----------------------------------------------------------------------------------------

def arrow_vs_point(path):
    x, y, th = _pose(K_EX)
    T = _lap().Tw[K_EX]
    f = _vec(th)
    fig, (ax,) = _panels([((-3, 3), (-2.5, 3.5), False, AXW)], footer=1.35)
    _frame(ax, (0, 0, th), EGO, 1.4)
    _dot(ax, tuple(f), EGO, size=12)
    for start in ((-2.0, 1.6), (1.9, -1.2)):
        _arrow(ax, start, tuple(np.array(start) + f), EGO, lw=3)
    dot = (T @ np.array([1.0, 0.0, 1.0]))[:2]
    direction = (T @ np.array([1.0, 0.0, 0.0]))[:2]
    fig.text(0.04, 0.1 / fig.get_figheight(), f"(1, 0, 1) \u2192 ({dot[0]:.2f}, {dot[1]:.2f}, 1):\na place: the dot, 1 m ahead\n"
             f"(1, 0, 0) \u2192 ({direction[0]:.4f}, {direction[1]:.4f}, 0):\na direction: the arrows, 1 m long".replace("\n(1, 0, 0)", "\n\n(1, 0, 0)"),
             color=EGO, ha="left", va="bottom", linespacing=1.3)
    _save(fig, path)


# ---- 5  wrap_trap ----------------------------------------------------------------------------------------------

def wrap_trap(path):
    L = _lap()
    th_back = np.array([math.atan2(T[1, 0], T[0, 0]) for T in L.Tw])
    k = int(np.argmax(L.TH > math.pi))
    jumps = np.flatnonzero(np.abs(np.diff(th_back)) > math.pi) + 1
    drawn = th_back.copy()
    drawn[jumps] = np.nan  # break the line where it wraps, so only the real values are drawn
    fig = _fig(4.3)
    ax = fig.subplots()
    ax.plot(L.t, L.TH, color=GREY, lw=3)
    ax.plot(L.t, drawn, color=THETA, lw=3)
    ax.set(xlim=(0, 98), ylim=(-4.4, 7.3), xlabel="time (s)", ylabel="heading (rad)")
    ax.yaxis.set_major_locator(FixedLocator([-math.pi, 0, math.pi, 2 * math.pi]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: {-3: "\u2212\u03c0", 0: "0", 3: "\u03c0", 6: "2\u03c0"}[round(v)]))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    _note(ax, (40, 6.5), "yaw from the record", GREY)
    _note(ax, (24, -2.5), "\u03b8 from\nmatrix_to_pose", THETA)
    _label(ax, (k * DT, math.pi + 0.15), (26, 4.5), f"step {k}, {k * DT:.2f} s\n{th_back[k - 1]:.4f} \u2192 {_m(th_back[k], '.4f')}", TEXT, ha="center")
    t_arrow = 76
    i = int(t_arrow / DT)
    _arrow(ax, (t_arrow, th_back[i] + 0.1), (t_arrow, L.TH[i] - 0.1), TEXT, lw=2, style="<|-|>")
    _note(ax, (88.5, 2.6), "6.2832\nrad", TEXT)
    fig.tight_layout()
    _save(fig, path)


# ---- 6  rear_frame ---------------------------------------------------------------------------------------------

def rear_frame(path):
    x, y, th = _pose(K_EX)
    f, right = _vec(th), _vec(th - math.pi / 2)
    front, rear = 1.35 * f, -1.35 * f
    fig, (ax,) = _panels([((-4.4, 4.4), (-3.8, 3.2), False, AXW)])
    ax.plot([rear[0], front[0]], [rear[1], front[1]], color=GREY, lw=5, solid_capstyle="round", zorder=0.5)
    _frame(ax, (*front, th), EGO, 1.6)
    _frame(ax, (*rear, th), REAR, 1.6)
    _note(ax, tuple(front + 2.6 * right + 0.3 * f), "ego\n(front axle)", EGO)
    _note(ax, tuple(rear + np.array([-1.75, -0.6])), "rear", REAR)
    _note(ax, tuple(0.9 * right), "2.7 m", TEXT, rotation=math.degrees(th))
    _note(ax, (rear[0] - 0.4, rear[1] - 1.9), f"({x - WHEELBASE * f[0]:.2f}, {y - WHEELBASE * f[1]:.2f})", REAR)
    _save(fig, path)


# ---- 7  frame_chain --------------------------------------------------------------------------------------------

def frame_chain(path):
    fig = _fig(4.8)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 11), ylim=(0, 9.6))
    ax.set_aspect("equal")
    ax.axis("off")
    spots = ((1.8, 6.0, GREY, "world"), (5.5, 6.0, EGO, "ego"), (9.2, 6.0, REAR, "rear"))
    for sx, sy, colour, name in spots:
        _frame(ax, (sx, sy, 0), colour, 1.4)
        _note(ax, (sx + 0.1, sy - 1.1), name, colour)
    for a, b, text in ((spots[2], spots[1], "T ego\u2190rear"), (spots[1], spots[0], "T world\u2190ego")):
        ax.annotate("", xy=(b[0] + 0.3, b[1] + 0.7), xytext=(a[0] - 0.3, a[1] + 0.7),
                    arrowprops={"arrowstyle": "-|>", "color": TEXT, "lw": 2.2, "connectionstyle": "arc3,rad=0.45", "shrinkA": 4, "shrinkB": 4})
        _note(ax, ((a[0] + b[0]) / 2, a[1] + 2.9), text)
    ax.annotate("", xy=(spots[0][0], 4.1), xytext=(spots[2][0], 4.1),
                arrowprops={"arrowstyle": "-|>", "color": TEXT, "lw": 2.2, "connectionstyle": "arc3,rad=-0.25", "shrinkA": 4, "shrinkB": 4})
    fig.canvas.draw()
    _row(ax, fig._get_renderer(), ax.transData.inverted(), 2.1, ["T world\u2190rear = T world\u2190", ("ego", True), " \u00b7 T ", ("ego", True), "\u2190rear"], 0.2)
    ax.text(0.2, 0.7, "not to scale", ha="left", va="center", color=GREY)
    _save(fig, path)


def _row(ax, renderer, inv, y, pieces, x0):
    """Write `pieces` left to right on one baseline; a piece given as (text, True) is struck through."""
    x = x0
    for piece in pieces:
        text, struck = piece if isinstance(piece, tuple) else (piece, False)
        t = ax.text(x, y, text, ha="left", va="center", color=TEXT)
        box = t.get_window_extent(renderer)
        x1d = inv.transform((box.x1, box.y1))[0]
        if struck:
            ax.plot([x, x1d], [y, y], color=CORAL, lw=2.5)
        x = x1d + 0.12
    return x


# ---- 8  order_matters ------------------------------------------------------------------------------------------

def order_matters(path):
    cases = (("A then B", [(0, 0), (1, 0), (1, 1)], (1, 1), (2, 0), (1.62, 1.2)),
             ("B then A", [(0, 0), (1, 0), (2, 0)], (2, 0), (1, 1), (2.52, -0.3)))
    fig, axes = _panels([((-0.5, 3.0), (-0.55, 1.85), True, AXW)] * 2)
    for ax, (title, pts, end, other, where) in zip(axes, cases):
        ax.set_title(title)
        ax.plot([end[0], other[0]], [end[1], other[1]], color=FAINT, lw=1.5, ls=":")
        _dot(ax, other, FAINT, size=12)
        for a, b in zip(pts[:-1], pts[1:]):
            _arrow(ax, a, b, TEXT, lw=2, linestyle="--")
        _frame(ax, (0, 0, 0), GREY, 0.3)
        _frame(ax, (*end, math.pi / 2), EGO, 0.4)
        _note(ax, ((end[0] + other[0]) / 2 + 0.28, (end[1] + other[1]) / 2 + 0.2), "1.41 m", FAINT)
        _note(ax, where, f"({end[0]}, {end[1]})", EGO)
    _save(fig, path)


# ---- 9  undo_steps ---------------------------------------------------------------------------------------------

def undo_steps(path):
    L = _lap()
    x, y, th = _pose(K_EX)
    cone = (L.Tw[K_EX] @ np.array([10.0, 2.0, 1.0]))[:2]
    d = cone - np.array([x, y])
    fig, (top, bottom) = _panels([((-2.5, 12.5), (-2.2, 10.4), True, AXW), ((-2.5, 13.5), (-1.8, 4.6), True, AXW)])
    top.set_title(f"\u2212 position \u2192 ({d[0]:.2f}, {d[1]:.2f})")
    _frame(top, (0, 0, 0), GREY, 3)
    top.plot([0, d[0], d[0]], [0, 0, d[1]], color=GREY, lw=2, ls="--")
    _arrow(top, (0, 0), (d[0] - 0.12, d[1] - 0.12), GREY, lw=3)
    _dot(top, (0, 0), EGO)
    _dot(top, tuple(d), TEXT, size=12)
    _note(top, (-1.1, -1.2), "car", EGO)
    _note(top, (d[0] - 3.7, d[1] + 0.2), f"cone, was\n({cone[0]:.2f}, {cone[1]:.2f})")
    _note(top, (d[0] / 2 + 0.7, -1.2), f"{d[0]:.2f} east", GREY)
    _note(top, (d[0] + 0.5, d[1] / 2), f"{d[1]:.2f}\nnorth", GREY, ha="left")
    bottom.set_title(f"turn back {math.degrees(th):.2f}\u00b0 \u2192 (10, 2)")
    _frame(bottom, (0, 0, 0), EGO, 3)
    bottom.plot([0, 10, 10], [0, 0, 2], color=EGO, lw=2, ls="--")
    _arrow(bottom, (0, 0), (9.9, 1.95), EGO, lw=3)
    _dot(bottom, (0, 0), EGO)
    _dot(bottom, (10, 2), TEXT, size=12)
    _note(bottom, (-1.2, -1.2), "car", EGO)
    _note(bottom, (8.4, 3.5), "cone")
    _note(bottom, (5.0, -0.9), "10 m forward", EGO)
    _note(bottom, (10.5, 1.0), "2 m left", EGO, ha="left")
    _save(fig, path)


# ---- 10  inverse_A ---------------------------------------------------------------------------------------------

def inverse_A(path):
    fig, (ax,) = _panels([((-0.6, 1.7), (-0.55, 0.75), False, AXW)], footer=1.05)
    _frame(ax, (0, 0, 0), GREY, 0.3)
    _frame(ax, (1, 0, math.pi / 2), EGO, 0.3)
    _arrow(ax, (0.9, -0.14), (0.1, -0.14), TEXT, lw=2, linestyle="--")
    _note(ax, (0.0, -0.36), "start", GREY)
    _note(ax, (1.0, -0.36), "A", EGO)
    fig.text(0.04, 0.1 / fig.get_figheight(), "A = (1, 0, \u03c0/2)\nseen from A: (0, 1, \u2212\u03c0/2)\n0 m ahead, 1 m left", color=TEXT, ha="left", va="bottom")
    _save(fig, path)


# ---- 11  inverse_view ------------------------------------------------------------------------------------------

def inverse_view(path):
    L = _lap()
    x, y, th = _pose(K_EX)
    inv = invert_pose(L.Tw[K_EX])
    fwd_o, left_o = float(inv[0, 2]), float(inv[1, 2])
    po = (-left_o, fwd_o)
    fig, (top, bottom) = _panels([((-45, 205), (-26, 48), False, AXW), ((-135, 100), (-190, 32), False, AXW)])
    _frame(top, (0, 0, 0), GREY, 22)
    _dot(top, (0, 0), GREY)
    _note(top, (-3, -16), "world origin", GREY)
    _frame(top, (x, y, th), EGO, 22)
    _note(top, (x - 2, y - 22), f"car, {math.degrees(th):.2f}\u00b0", EGO)
    _frame(bottom, (0, 0, math.pi / 2), EGO, 18)
    _note(bottom, (24, -10), "car", EGO)
    bottom.plot([0, 0, po[0]], [0, po[1], po[1]], color=GREY, lw=2, ls="--")
    _frame(bottom, (po[0], po[1], math.pi / 2 - th), GREY, 20)
    _dot(bottom, po, GREY)
    _note(bottom, (22, po[1] / 2), f"{-fwd_o:.2f} m\nbehind", TEXT, ha="left")
    _note(bottom, (po[0] / 2, po[1] - 17), f"{left_o:.2f} m left", TEXT)
    _note(bottom, (-55, po[1] - 36), f"world origin\n({_m(fwd_o, '.2f')}, {left_o:.2f})", GREY)
    _save(fig, path)


# ---- 12  one_step ----------------------------------------------------------------------------------------------

def one_step(path):
    L = _lap()
    fwd, left, dth = float(L.fwd[K_STEP]), float(L.left[K_STEP]), float(L.dth[K_STEP])
    th0 = float(L.TH[K_STEP])
    dx, dy = float(L.X[K_STEP + 1] - L.X[K_STEP]), float(L.Y[K_STEP + 1] - L.Y[K_STEP])
    MAG = 200  # every angle is drawn this many times larger
    ang_world = MAG * math.atan2(dy, dx)
    a0, a1 = MAG * th0, MAG * (th0 + dth)
    tip = _vec(ang_world)
    along = math.cos(ang_world - a0) * _vec(a0)
    fig, (ax,) = _panels([((-0.75, 2.1), (-0.5, 1.0), False, AXW)])
    ax.axis("off")
    ax.plot([0, along[0], tip[0]], [0, along[1], tip[1]], color=EGO, lw=1.8, ls="--")
    ax.plot([0, 0, tip[0]], [0, tip[1], tip[1]], color=GREY, lw=1.8, ls="--")
    _arrow(ax, (0, 0), tuple(tip), TEXT, lw=3)
    _frame(ax, (0, 0, a0), EGO, 0.25)
    _frame(ax, (*tip, a1), REAR, 0.25)
    ax.add_patch(Arc(tuple(tip), 0.7, 0.7, theta1=math.degrees(a0), theta2=math.degrees(a1), color=THETA, lw=2.5))
    _label(ax, tuple(tip + 0.35 * _vec((a0 + a1) / 2)), (1.45, 0.75), f"{dth:.5f} rad", THETA)
    _note(ax, (0.46, -0.16), f"{fwd:.4f} m forward", EGO)
    _note(ax, (tip[0] + 0.07, tip[1] / 2 - 0.12), f"{left:.5f} m left", EGO, ha="left")
    _note(ax, (0.3, tip[1] + 0.2), f"{math.floor(dx * 1e5) / 1e5:.5f} m east", GREY)
    _note(ax, (-0.14, tip[1] / 2), f"{dy:.5f} m\nnorth", GREY, ha="right")
    _note(ax, (1.5, -0.27), f"angles \u00d7{MAG}", TEXT)
    _save(fig, path)


# ---- 13  step_turns --------------------------------------------------------------------------------------------

def step_turns(path):
    L = _lap()
    s = _data("rear_fig8")["s"]
    bend1 = ((s > 100 + 60) & (s < 100 + math.pi * GENTLE_RADIUS - 60))[:-1]  # the same mask measure.py averages over
    fig = _fig(4.3)
    ax = fig.subplots()
    ax.plot(L.t[:-1], L.dth, color=THETA, lw=2.5)
    ax.set(xlim=(0, 96), ylim=(-0.0009, 0.0088), xlabel="time (s)", ylabel="turn per step (rad)")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(FixedLocator([0, 0.002, 0.004, 0.006]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    _note(ax, (29, 0.0017), "\u2248 0 on\nthe straights", TEXT)
    _note(ax, (44, 0.0077), f"bend 1: {L.dth[bend1].mean():.5f} average\n{L.dth.max():.5f} max", THETA)
    _note(ax, (76, 0.0017), f"sum:\n{L.dth.sum():.4f} rad", TEXT)
    fig.tight_layout()
    _save(fig, path)


# ---- 14  naive_add ---------------------------------------------------------------------------------------------

def naive_add(path):
    L = _lap()
    nx = L.X[0] + np.concatenate([[0.0], np.cumsum(L.fwd)])
    ny = L.Y[0] + np.concatenate([[0.0], np.cumsum(L.left)])
    end_gap = float(np.hypot(nx[-1] - L.X[-1], ny[-1] - L.Y[-1]))
    fig = _fig(3.4)
    ax = fig.subplots()
    _lap_axes(ax)
    ax.plot(nx, ny, color=CORAL, lw=3)
    ax.set(xlim=(-100, 770), ylim=(-135, 185))
    ax.set_xticks([0, 250, 500, 750])
    ax.set_yticks([])
    ax.set_xlabel("east (m)")
    _note(ax, (50, 100), "true lap", GREY)
    _note(ax, (480, 75), "add the parts to x, y, \u03b8", CORAL)
    _note(ax, (590, -22), f"({nx[-1]:.2f}, {ny[-1]:.2f})", CORAL)
    _arrow(ax, (L.X[-1], -75), (nx[-1], -75), TEXT, lw=2, style="<|-|>")
    _note(ax, (360, -106), f"{end_gap:.2f} m", TEXT)
    fig.tight_layout()
    _save(fig, path)


# ---- 15  wrong_order, 16  wrong_order_error --------------------------------------------------------------------

def _wrong_order():
    L = _lap()
    W_ = [L.Tw[0]]
    for S in L.steps:
        W_.append(S @ W_[-1])
    W_ = np.stack(W_)
    return W_[:, 0, 2], W_[:, 1, 2], np.hypot(W_[:, 0, 2] - L.X, W_[:, 1, 2] - L.Y)


def wrong_order(path):
    L = _lap()
    wx, wy, err = _wrong_order()
    k = int(np.argmax(err))
    fig = _fig(5.0)
    ax = fig.subplots()
    _lap_axes(ax)
    ax.plot(wx, wy, color=CORAL, lw=2.5)
    ax.set(xlim=(-125, 215), ylim=(-30, 250))
    _metres(ax)
    ax.plot([L.X[k], wx[k]], [L.Y[k], wy[k]], color=TEXT, lw=2, ls="--")
    _dot(ax, (L.X[k], L.Y[k]), GREY, size=11)
    _dot(ax, (wx[k], wy[k]), CORAL, size=11)
    _label(ax, (wx[k], wy[k]), (-30, 237), f"({_m(wx[k], '.2f')}, {wy[k]:.2f})", CORAL)
    _label(ax, (L.X[k], L.Y[k]), (138, 128), f"({L.X[k]:.2f}, {L.Y[k]:.2f})", GREY)
    _note(ax, (40, 190), f"step {k}, {k * DT:.2f} s:\n{err[k]:.2f} m apart", TEXT)
    _label(ax, (L.X[-1], L.Y[-1]), (150, 25), f"end: {err[-1]:.2f} m off", CORAL)
    fig.tight_layout()
    _save(fig, path)


def wrong_order_error(path):
    L = _lap()
    _, _, err = _wrong_order()
    k = int(np.argmax(err))
    first = int(np.argmax(err > 1.0))
    fig = _fig(4.3)
    ax = fig.subplots()
    ax.plot(L.t, err, color=CORAL, lw=3)
    ax.axhline(2.0, color=GREY, lw=2, ls="--")
    ax.set(xlim=(0, 98), ylim=(-12, 232), xlabel="time (s)", ylabel="position error (m)")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    _label(ax, (first * DT, 2.0), (first * DT + 17, 75), f"{first * DT:.2f} s: 1 m", CORAL)
    _note(ax, (k * DT + 28, 212), f"{err[k]:.2f} m at {k * DT:.2f} s", CORAL)
    _label(ax, (L.t[-1], err[-1]), (80, 112), f"{err[-1]:.2f} m\nat the end", CORAL)
    _note(ax, (72, 30), "a 2 m end-only check", GREY)
    fig.tight_layout()
    _save(fig, path)


# ---- 17  drift_end, 18  drift_growth ---------------------------------------------------------------------------

def drift_end(path):
    d = _data("drift_fig7")
    ex, ey = float(d["x_true"][-1]), float(d["y_true"][-1])
    fig, axes = _panels([((ex - 15, ex + 15), (ey - 15, ey + 15), True, 4.7)] * 2, title_h=0.5)
    for ax, s, words in zip(axes, SIGMAS, ("within 0.944 m", "within 9.454 m")):
        xs, ys = d[f"x_{s}"], d[f"y_{s}"]
        ends = np.hypot(xs[:, -1] - ex, ys[:, -1] - ey)
        ax.set_gid("zoom-inset")
        for i in range(len(xs)):
            ax.plot(xs[i, -400:], ys[i, -400:], color=CORAL, lw=1.4, alpha=0.8)
            _dot(ax, (xs[i, -1], ys[i, -1]), CORAL, size=6)
        ax.plot(d["x_true"][-400:], d["y_true"][-400:], color=GREY, lw=2.5)
        _dot(ax, (ex, ey), TEXT, size=9)
        ax.add_patch(Circle((ex, ey), float(ends.max()), fill=False, edgecolor=TEXT, lw=1.5, ls="--"))
        ax.set_title(f"\u03c3 = {float(s):g} rad per step")
        _note(ax, (ex + 14, ey - 13), words, TEXT, ha="right")
    _save(fig, path)


def drift_growth(path):
    d = _data("drift_fig7")
    t = np.arange(d["rms_0.0001"].size) * DT
    fig = _fig(8.6)
    top, bottom = fig.subplots(2, 1)
    for ax, s in zip((top, bottom), SIGMAS):
        for e in d[f"err_{s}"]:
            ax.plot(t, e, color=GREY, lw=1, alpha=0.7)
        rms = d[f"rms_{s}"]
        ax.plot(t, rms, color=CORAL, lw=4)
        ax.set(xlim=(0, 98), xlabel="time (s)", ylabel="position error (m)")
        ax.set_title(f"\u03c3 = {float(s):g} rad per step")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_ylim(0, float(d[f"err_{s}"].max()) * 1.04)
    rms = d["rms_0.0001"]
    for k, text, where in ((250, "0.097 m, 12.50 s", (30, 0.62)), (1000, "0.447 m, 50 s", (72, 0.18)), (1485, "peak 0.712 m, 74.25 s", (40, 1.2))):
        _dot(top, (t[k], rms[k]), CORAL, size=9)
        _label(top, (t[k], rms[k]), where, text, CORAL)
    _label(top, (t[-1], rms[-1]), (86, 0.95), f"{rms[-1]:.3f} m\nat the end", CORAL)
    kb = int(np.argmax(d["rms_0.001"]))
    _dot(bottom, (t[kb], d["rms_0.001"][kb]), CORAL, size=9)
    _label(bottom, (t[kb], d["rms_0.001"][kb]), (40, 10.0), f"peak {d['rms_0.001'][kb]:.3f} m", CORAL)
    fig.tight_layout()
    _save(fig, path)


# ---- 19  drift_laps --------------------------------------------------------------------------------------------

def drift_laps(path):
    d = _data("drift_laps")
    laps, rms = d["laps"], d["rms"]
    fig = _fig(4.3)
    ax = fig.subplots()
    ax.plot(laps, rms, color=CORAL, lw=2, ls=":")
    ax.plot(laps, rms, color=CORAL, marker="o", ms=11, ls="")
    ax.set_xscale("log", base=2)
    ax.set_xticks(laps)
    ax.set_xticklabels([str(int(n)) for n in laps])
    ax.minorticks_off()
    ax.set(xlim=(0.5, 24), ylim=(0, 3.4), xlabel="laps driven", ylabel="RMS end error (m)")
    ax.yaxis.set_major_locator(MaxNLocator(4))
    for n, v in zip(laps, rms):
        _note(ax, (n * 1.05, v + 0.34), f"{v:.3f}", CORAL, ha="right")
    _note(ax, (2.6, 3.05), "\u03c3 = 0.0001, 200 seeds", TEXT)
    fig.tight_layout()
    _save(fig, path)


# ---- 20  rear_axle_paths, 21  rear_axle_error ------------------------------------------------------------------

def _rear():
    d = _data("rear_fig8")
    L = _lap()
    s_rear = np.array([L.road.ego_truth(a, b, c)[0] for a, b, c in zip(d["xr"], d["yr"], L.TH)])
    return d, s_rear


def rear_axle_paths(path):
    d, s_rear = _rear()
    MAG = 20  # the bend is straightened out and sideways distances are drawn this many times larger
    sel = slice(330, 471)
    k = K_EX
    fig = _fig(4.3)
    ax = fig.subplots()
    ax.axhline(0, color=GREY, lw=3)
    ax.plot(d["s"][sel], MAG * d["lat_f"][sel], color=EGO, lw=3)
    ax.plot(s_rear[sel], MAG * d["lat_r"][sel], color=REAR, lw=3)
    ax.plot([s_rear[k], d["s"][k]], [MAG * d["lat_r"][k], MAG * d["lat_f"][k]], color=TEXT, lw=6, solid_capstyle="round")
    x0 = float(s_rear[sel].min()) - 1
    ax.set(xlabel="distance along the lane (m)", ylabel=f"sideways \u00d7{MAG}", xlim=(x0, float(d["s"][sel].max()) + 1), ylim=(-3.5, 18))
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MaxNLocator(4))
    _note(ax, (x0 + 14, 1.5), "lane centre", GREY)
    _note(ax, (x0 + 14, 14.3), "rear axle", REAR)
    _note(ax, (x0 + 50, 7.6), "front axle", EGO)
    _note(ax, (float(d["s"][k]) - 13, 7.6), "car, 2.7 m", TEXT)
    _arrow(ax, (x0 + 3, 4), (x0 + 3, 9), TEXT, lw=1.5)
    _note(ax, (x0 + 24, 16.2), "toward the bend's centre", TEXT, fontsize=FS)
    fig.tight_layout()
    _save(fig, path)


def rear_axle_error(path):
    d, _ = _rear()
    t = np.arange(len(d["lat_f"])) * DT
    n = 1200
    gap = d["lat_r"] - d["lat_f"]
    kg = int(np.argmax(np.abs(gap[:n])))
    fig = _fig(4.3)
    ax = fig.subplots()
    ax.axhline(0, color=GREY, lw=1.5)
    ax.plot(t[:n], d["lat_f"][:n], color=BLUE, lw=3)
    ax.plot(t[:n], d["lat_r"][:n], color=BLUE_REAR, lw=3)
    ax.set(xlim=(0, 74), ylim=(-0.1, 0.85), xlabel="time (s)", ylabel="e (m)")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    _note(ax, (66.5, d["lat_r"][n - 1] + 0.1), "rear", BLUE_REAR)
    _note(ax, (66.5, d["lat_f"][n - 1] - 0.1), "front", BLUE)
    _arrow(ax, (t[kg], d["lat_f"][kg]), (t[kg], d["lat_r"][kg]), TEXT, lw=2, style="<|-|>")
    _label(ax, (t[kg], (d["lat_f"][kg] + d["lat_r"][kg]) / 2), (30, 0.74), f"{abs(gap[kg]):.3f} m at {t[kg]:.2f} s", TEXT)
    fig.tight_layout()
    _save(fig, path)


# ---- 22  levers, 23  drift_loglog, 24  next_lesson -------------------------------------------------------------

def levers(path):
    L = _lap()
    fig = _fig(4.2)
    ax = fig.subplots()
    _lap_axes(ax)
    for k in range(0, 1800, 100):
        ax.plot([L.X[k], L.X[-1]], [L.Y[k], L.Y[-1]], color=FAINT if k != 400 else TEXT, lw=1.2 if k != 400 else 2.2)
    ax.set(xlim=(-95, 320), ylim=(-45, 175))
    _metres(ax, n=5)
    _dot(ax, (L.X[-1], L.Y[-1]), TEXT, size=9)
    _label(ax, (L.X[-1], L.Y[-1]), (95, -28), "end of the lap", TEXT)
    _label(ax, (L.X[400], L.Y[400]), (250, 55), "lever from\nstep 400", TEXT)
    fig.tight_layout()
    _save(fig, path)


def drift_loglog(path):
    d = _data("drift_fig9")
    ell, sigma = float(d["ell"]), float(d["sigma"])
    Ns = np.array([10, 20, 50, 100, 200, 400, 501])
    rms = d["rms_straight"][Ns]
    grid = np.array([8, 700.0])
    fig = _fig(4.6)
    ax = fig.subplots()
    ax.plot(grid, sigma * ell * grid ** 1.5 / math.sqrt(3), color=GREY, lw=3)
    ax.plot(Ns, rms, color=CORAL, marker="o", ms=10, ls="")
    ax.set(xscale="log", yscale="log", xlim=(8, 700), ylim=(3e-4, 1.2), xlabel="steps N", ylabel="RMS sideways error (m)")
    _note(ax, (30, 0.13), "measured RMS,\n500 seeds", CORAL)
    _note(ax, (160, 0.0045), r"$\sigma\,\ell\,N^{3/2}/\sqrt{3}$", GREY)
    _note(ax, (160, 0.0014), "slope 3/2", GREY)
    _label(ax, (Ns[0], rms[0]), (40, 0.00045), f"{rms[0]:.5f} m", CORAL)
    _label(ax, (Ns[-1], rms[-1]), (110, 0.7), f"{rms[-1]:.5f} m", CORAL)
    fig.tight_layout()
    _save(fig, path)


def next_lesson(path):
    mount = inspect.signature(Camera.__init__).parameters  # the renderer's own defaults
    height, pitch_deg = mount["height_m"].default, mount["pitch_deg"].default
    pitch = math.radians(pitch_deg)
    reach = height / math.tan(pitch)  # where the ray meets flat ground
    xlim, ylim, aw, ah = (-6, 18.5), (-0.3, 3.7), 5.3, 2.4
    sx, sy = aw / (xlim[1] - xlim[0]), ah / (ylim[1] - ylim[0])  # inches per metre; heights are stretched
    fig = _fig(ah + 0.2)
    ax = fig.add_axes((0.1 / W, 0.1 / fig.get_figheight(), aw / W, ah / fig.get_figheight()))
    ax.set(xlim=xlim, ylim=ylim)
    ax.axis("off")

    def inches(vx, vy):  # a vector given in inches, as data units
        return vx / sx, vy / sy

    ax.plot(xlim, [0, 0], color=GREY, lw=2.5)
    ax.add_patch(Rectangle((-WHEELBASE - 0.7, 0.3), WHEELBASE + 1.4, 0.8, fill=False, edgecolor=GREY, lw=2))
    for cx in (-WHEELBASE, 0):
        ax.add_patch(Circle((cx, 0.3), 0.3, fill=False, edgecolor=GREY, lw=2))
    _arrow(ax, (0, 0), inches(0.6, 0), EGO, lw=2.5, scale=10)
    _arrow(ax, (0, 0), inches(0, 0.5), EGO, lw=2.5, scale=10)
    lens = np.array([0.0, height])
    ray = np.array([math.cos(pitch) * sx, -math.sin(pitch) * sy])
    ray /= np.hypot(*ray)
    _arrow(ax, tuple(lens), tuple(lens + np.array(inches(*(0.7 * ray)))), CAMERA, lw=2.5, scale=10)
    _arrow(ax, tuple(lens), tuple(lens + np.array(inches(*(0.4 * np.array([-ray[1], ray[0]]) * -1)))), CAMERA, lw=2.5, scale=10)
    _note(ax, (-3.6, 2.9), "car axes", EGO)
    _note(ax, (6.0, 2.95), "camera axes", CAMERA)
    ax.plot([0, 12], [height, height], color=GREY, lw=1.5, ls=":")
    ax.plot([0, reach], [height, 0], color=CORAL, lw=2.5, ls="--")
    _note(ax, (reach, 0.65), "?", CORAL, fontsize=TITLE)
    _note(ax, (9.0, 1.85), f"{pitch_deg:g}\u00b0 down", THETA)
    _arrow(ax, (2.2, 0), (2.2, height), TEXT, lw=1.8, style="<|-|>", scale=10)
    _note(ax, (3.9, 0.55), f"{height:g} m", TEXT)
    _save(fig, path)


FIGURES = {
    "rebuilt_lap": rebuilt_lap, "pose_frame": pose_frame, "pose_matrix": pose_matrix, "arrow_vs_point": arrow_vs_point,
    "wrap_trap": wrap_trap, "rear_frame": rear_frame, "frame_chain": frame_chain, "order_matters": order_matters,
    "undo_steps": undo_steps, "inverse_A": inverse_A, "inverse_view": inverse_view, "one_step": one_step,
    "step_turns": step_turns, "naive_add": naive_add, "wrong_order": wrong_order, "wrong_order_error": wrong_order_error,
    "drift_end": drift_end, "drift_growth": drift_growth, "drift_laps": drift_laps, "rear_axle_paths": rear_axle_paths,
    "rear_axle_error": rear_axle_error, "levers": levers, "drift_loglog": drift_loglog, "next_lesson": next_lesson,
}


def make(out_dir):
    with matplotlib.rc_context(STYLE):
        for name, draw in FIGURES.items():
            draw(Path(out_dir) / f"{name}.png")
