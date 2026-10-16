"""Figures for lecture 1.1.1 Positions and headings.  `python scripts/make_figures.py course/1.1.1-positions-and-headings`.

Every figure is 5.5 in wide (Ruling F: its smallest text is 15 pt, at least 2.7 x the width in pt).  Every number drawn
is read from the simulation or computed here from `tel`; the lecture's numbers come from measure.py (tags in comments).
"""
import functools
import math

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Arc, FancyArrowPatch, Polygon, Rectangle

from zero2fsd.car import Car, run
from zero2fsd.drawing import draw_frame
from zero2fsd.geom import direction_angle, heading_vector, wrap_angle
from zero2fsd.palette import CHROME, QUANTITY, SCENE, shades
from zero2fsd.sim import make_scenario
from zero2fsd.sim.dynamics import DT
from zero2fsd.sim.scenarios import TARGET_SPEED
from zero2fsd.sim.world import EGO_LANE_CENTER, LANE_WIDTH, ROAD_HALF_WIDTH

TEXT, PANEL, BORDER = CHROME["text"], CHROME["panel"], CHROME["border"]
EGO, THETA, PSI, DELTA = (QUANTITY[k] for k in ("ego", "theta", "psi", "delta"))
LABEL_BOX = {"boxstyle": "round,pad=0.2", "fc": PANEL, "ec": PANEL}  # keeps a painted line from running through a label
W = 5.5  # inches: every figure is this wide
FS, TITLE = 15, 16
STYLE = {"font.size": FS, "axes.labelsize": FS, "axes.titlesize": TITLE, "axes.titlepad": 8, "xtick.labelsize": FS, "ytick.labelsize": FS}
STEP = 700  # the lecture's opening step
BENDS = ((150.0, 300.0), (500.0, 650.0))  # m along the road: the middles of the two bends
BACK_STRAIGHT = (361.3, 441.3)  # m along the road
ZOOM_INSET = "zoom-inset"  # gid of an axes that deliberately shows only part of its data (the clipping gate skips it)
PI = math.pi
NORTH_EAST = "north ↑   east →"


def _minus(value, digits=1, sign=True):
    """A number as text with a true minus sign: +0.4, −0.3."""
    return f"{value:{'+' if sign else ''}.{digits}f}".replace("-", "−")


@functools.lru_cache(maxsize=None)
def _lap():
    """The gentle lap with the provided car, and everything the figures take from it."""
    road = make_scenario("gentle").road
    tel = run(Car(), "gentle")
    P = np.stack([tel.x, tel.y], axis=-1)
    D = np.diff(P, axis=0)
    yaw = tel.yaw
    A = direction_angle(D)
    return {"road": road, "tel": tel, "P": P, "D": D, "A": A, "t": tel.t[:-1], "prog": tel.progress_m[:-1], "yaw": yaw,
            "gap": wrap_angle(A - wrap_angle(yaw[:-1]))}


def _fig(height):
    return Figure(figsize=(W, height))


def _save(fig, path):
    fig.savefig(path)


def _arrow(ax, p, q, colour, lw=2.5, style="-|>", ls="-", alpha=1.0, scale=None, zorder=4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=scale or FS, color=colour, lw=lw, linestyle=ls, alpha=alpha,
                                 shrinkA=0, shrinkB=0, zorder=zorder))


def _span(ax, p, q, colour, lw=1.8):
    """A dimension line: a segment with a tick at each end."""
    _arrow(ax, p, q, colour, lw=lw, style="|-|", scale=5)


def _text(ax, x, y, s, colour=TEXT, ha="center", va="center", box=False, **kwargs):
    return ax.text(x, y, s, color=colour, ha=ha, va=va, bbox=LABEL_BOX if box else None, zorder=6, **kwargs)


def _canvas(height, xlim, ycentre):
    """A full-figure axes with no frame, x from `xlim`, y centred on `ycentre`, one data unit equal in both directions."""
    fig = _fig(height)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set(xticks=[], yticks=[])  # hidden axes still own tick labels, and the legibility gate counts them
    half = (xlim[1] - xlim[0]) / 2 * height / W
    ax.set(xlim=xlim, ylim=(ycentre - half, ycentre + half), aspect="equal")
    return fig, ax


def _scene(ax, road, s0, s1, ds=0.25):
    """Tarmac, yellow centre line and white edge lines of `road` between arc lengths s0 and s1, on grass."""
    ax.set_gid(ZOOM_INSET)  # a window onto the road: the road and its markings run on past the edges
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc=SCENE["grass"], ec=SCENE["grass"], lw=0, zorder=0))
    s = np.arange(s0, s1 + ds, ds)
    left, right = road.offset(s, ROAD_HALF_WIDTH), road.offset(s, -ROAD_HALF_WIDTH)
    ax.add_patch(Polygon(np.vstack([left, right[::-1]]), closed=True, fc=SCENE["road"], ec=SCENE["road"], lw=0, zorder=1))
    for d, colour in ((0.0, SCENE["yellow"]), (LANE_WIDTH, SCENE["white"]), (-LANE_WIDTH, SCENE["white"])):
        x, y = road.offset(s, d).T
        ax.plot(x, y, color=colour, lw=2.5, zorder=2)


def _car(ax, x, y, yaw, colour=EGO, alpha=0.45, scale=1.0):
    """The car's outline (front axle at x, y; 4 m long, 1.8 m wide, times `scale`) as a schematic."""
    body = scale * np.array([[-3.4, -0.9], [0.6, -0.9], [0.6, 0.9], [-3.4, 0.9]])
    c, s = math.cos(yaw), math.sin(yaw)
    pts = body @ np.array([[c, s], [-s, c]]) + [x, y]
    ax.add_patch(Polygon(pts, closed=True, fc=colour, ec=colour, alpha=alpha, lw=1.5, zorder=3))


def _arc(ax, centre, radius, a0, a1, colour, lw=2.5, arrow=True):
    """A circular arc from angle a0 to a1 (radians, anticlockwise if a1 > a0), ending in an arrowhead."""
    lo, hi = sorted((a0, a1))
    ax.add_patch(Arc(centre, 2 * radius, 2 * radius, theta1=math.degrees(lo), theta2=math.degrees(hi), color=colour, lw=lw, zorder=4))
    if arrow:
        t = np.array([math.cos(a1), math.sin(a1)])
        tangent = np.array([-t[1], t[0]]) * (1 if a1 > a0 else -1)
        tip = np.asarray(centre) + radius * t
        _arrow(ax, tip - 0.02 * radius * tangent, tip, colour, lw=lw)


def _right_angle(ax, origin, u, v, size, colour=TEXT):
    o, a, b = np.asarray(origin), np.asarray(u) * size, np.asarray(v) * size
    ax.plot(*np.array([o + a, o + a + b, o + b]).T, color=colour, lw=1.5, zorder=5)


def _axes(fig, rect=None):
    ax = fig.add_subplot() if rect is None else fig.add_axes(rect)
    return ax


# ---------------------------------------------------------------------------------------------------------------
def _result_first(path):  # [s700] [cone] [cone_dot] [cone_cross]
    d = _lap()
    road, tel = d["road"], d["tel"]
    x0, y0, yaw = tel.x[STEP], tel.y[STEP], tel.yaw[STEP]
    h = heading_vector(yaw)
    n = np.array([-h[1], h[0]])
    s_car, _ = road.project(x0, y0)
    cone = road.offset(s_car + 12.0, -4.3)
    v = cone - np.array([x0, y0])
    ahead, left = float(v @ h), float(h[0] * v[1] - h[1] * v[0])
    fig, ax = _canvas(6.0, (154.5, 171.5), 134.25)
    _scene(ax, road, 255.0, 305.0)
    _car(ax, x0, y0, yaw)
    p0, p1 = np.array([x0, y0]), np.array([tel.x[STEP + 1], tel.y[STEP + 1]])
    foot = p0 + ahead * h
    ax.plot(*np.array([p0, foot, cone]).T, color=TEXT, lw=1.4, ls=(0, (4, 3)), zorder=2)
    _arrow(ax, p0, p0 + 2.0 * h, EGO, lw=3)
    for p in (p0, p1):
        ax.plot(*p, marker="o", color=EGO, ms=7, zorder=5)
    _text(ax, p0[0] + 1.9, p0[1] - 1.3, "step 700", EGO, box=True)
    _text(ax, p1[0] + 0.6, p1[1] + 0.5, "701", EGO, ha="left", box=True)
    ax.plot(*cone, marker="^", color=SCENE["white"], mec=BORDER, mew=2, ms=11, zorder=6)
    _span(ax, p0 + 0.9 * n, foot + 0.9 * n, TEXT)
    _text(ax, *(p0 + foot) / 2 + 2.3 * n, "? m ahead", TEXT, box=True)
    _span(ax, foot + 0.9 * h, cone + 0.9 * h, TEXT)
    _text(ax, *(foot + cone) / 2 + 2.3 * h, "? m to the side", TEXT, box=True)
    _text(ax, 171.2, 142.3, NORTH_EAST, TEXT, ha="right")
    _save(fig, path)


def _world_lap(path):  # [lap] [start] [s700]
    d = _lap()
    road, tel = d["road"], d["tel"]
    fig = _fig(4.3)
    ax = fig.add_axes([0.16, 0.14, 0.82, 0.84])
    s = np.arange(0.0, road.length + 0.5, 0.5)
    rx, ry = road.offset(s, 0.0).T
    ax.plot(rx, ry, color=SCENE["road"], lw=11, solid_capstyle="round", zorder=1)
    ax.plot(rx, ry, color=SCENE["yellow"], lw=1.2, zorder=2)
    ax.plot(tel.x, tel.y, color=EGO, lw=2.5, zorder=3)
    ax.plot(tel.x[0], tel.y[0], marker="o", color=TEXT, ms=7, zorder=5)
    ax.plot(tel.x[STEP], tel.y[STEP], marker="o", color=EGO, mec=TEXT, mew=1.5, ms=9, zorder=5)
    mid = len(tel.x) // 4
    ax.annotate("", xy=(tel.x[mid + 3], tel.y[mid + 3]), xytext=(tel.x[mid], tel.y[mid]), arrowprops={"arrowstyle": "-|>", "color": EGO, "lw": 2.5, "mutation_scale": 22}, zorder=6)
    ax.set(aspect="equal", xlabel="east (m)", ylabel="north (m)", xlim=(-95, 195), ylim=(-30, 175))
    before = set(ax.get_children())
    draw_frame(ax, (0.0, 0.0, 0.0), TEXT, "", length_m=40.0, fontsize=FS)
    for artist in set(ax.get_children()) - before:
        artist.set_zorder(8)
    ax.texts[-1].set_text("")
    _text(ax, 62, 25, "world frame", TEXT)
    ax.annotate("start (0, −1.8)", xy=(0, -1.8), xytext=(80, -19), color=TEXT, arrowprops={"arrowstyle": "-", "color": TEXT, "lw": 1.2}, ha="center")
    ax.annotate("step 700", xy=(tel.x[STEP], tel.y[STEP]), xytext=(110, 105), color=EGO, arrowprops={"arrowstyle": "-", "color": EGO, "lw": 1.2}, ha="center")
    _save(fig, path)


def _step_arrow(path):  # [s700arrow]
    d = _lap()
    tel = d["tel"]
    dd = d["D"][STEP]
    fig, ax = _canvas(4.5, (-1.0, 0.8), 0.1)
    # coordinates relative to the step-700 point; the window is about 1.8 m wide
    q = dd
    corner = np.array([q[0], 0.0])
    for p in (np.zeros(2), q):
        ax.plot(*p, marker="o", color=EGO, ms=9, zorder=6)
    _arrow(ax, np.zeros(2), q, TEXT, lw=3, scale=22)
    ax.plot(*np.array([np.zeros(2), corner, q]).T, color=BORDER, lw=2, ls=(0, (4, 3)), zorder=2)
    for shift in (np.array([0.55, 0.35]), np.array([-0.55, -0.28])):
        _arrow(ax, shift, shift + q, TEXT, lw=1.6, scale=18, alpha=0.8)
    _text(ax, 0.04, 0.06, "step 700", EGO, ha="left", va="bottom", box=True)
    _text(ax, q[0], q[1] + 0.05, "701", EGO, va="bottom")
    _text(ax, q[0] / 2, -0.1, f"Δx = {_minus(q[0], 3)} m", TEXT, va="top", box=True)
    _text(ax, q[0] - 0.03, q[1] / 2, f"Δy = {_minus(q[1], 3)} m", TEXT, ha="right", box=True)
    mid = q / 2
    _text(ax, mid[0] + 0.07, mid[1] + 0.05, f"{np.hypot(*q):.3f} m", TEXT, ha="left", box=True)
    _text(ax, 0.78, -0.57, NORTH_EAST, TEXT, ha="right")
    _save(fig, path)


def _theta_psi(path):  # [s700] [s700yaw]
    d = _lap()
    yaw = d["tel"].yaw[STEP]
    psi = d["tel"].head_err[STEP]
    road_dir = yaw - 5 * psi  # the road's direction, with psi drawn 5 times larger
    fig, ax = _canvas(5.5, (-3.4, 3.2), 1.0)
    o = np.zeros(2)
    _car(ax, 0.0, 0.0, yaw, scale=0.6)
    _arrow(ax, o, 2.7 * np.array([1.0, 0.0]), TEXT, lw=1.6)
    _text(ax, 2.0, -0.28, "east", TEXT)
    _arrow(ax, o, 2.4 * heading_vector(yaw), EGO, lw=3.5, scale=22)
    _arc(ax, o, 1.0, 0.0, yaw, THETA)
    _text(ax, 0.62, 1.52, f"θ = {yaw:.4f} rad", THETA, ha="left", box=True)
    _arrow(ax, o, 3.0 * heading_vector(road_dir), TEXT, lw=1.8, ls=(0, (5, 3)), style="-")
    _text(ax, *3.0 * heading_vector(road_dir) + [0.0, 0.45], "road direction", TEXT)
    _arc(ax, o, 2.0, road_dir, yaw, PSI, arrow=False)
    ax.annotate(f"ψ = {_minus(psi, 4, sign=False)} rad\n(drawn 5× larger)", xy=2.0 * heading_vector((road_dir + yaw) / 2), xytext=(0.3, 3.6), color=PSI,
                ha="center", va="center", arrowprops={"arrowstyle": "-", "color": PSI, "lw": 1.2})
    _save(fig, path)


def _unit_circle(path):  # [hand] [s700yaw]
    d = _lap()
    yaw = d["tel"].yaw[STEP]
    h = heading_vector(yaw)
    fig, ax = _canvas(5.5, (-2.15, 2.15), 0.0)
    t = np.linspace(0, 2 * PI, 200)
    ax.plot(np.cos(t), np.sin(t), color=BORDER, lw=2, zorder=1)
    o = np.zeros(2)
    for theta, label, offset, ha, va in ((0.0, "θ = 0\n→ (1, 0)", (1.08, 0.0), "left", "center"),
                                         (PI / 2, "θ = π/2 → (0, 1)", (0.0, 1.12), "center", "bottom"),
                                         (PI, "θ = π\n→ (−1, 0)", (-1.08, 0.0), "right", "center")):
        _arrow(ax, o, heading_vector(theta), TEXT, lw=1.4, scale=16)
        _text(ax, *offset, label, TEXT, ha=ha, va=va)
    _arrow(ax, o, h, EGO, lw=3.5, scale=22)
    ax.plot(*np.array([[h[0], h[1]], [h[0], 0.0]]).T, color=TEXT, lw=1.8, ls=(0, (4, 3)), zorder=2)
    _arc(ax, o, 0.3, 0.0, yaw, THETA, lw=2.2)
    _text(ax, 0.42, 0.47, f"θ = {yaw:.4f}", THETA, ha="left", va="bottom", box=True)
    _span(ax, (h[0] - 0.08, 0.0), (h[0] - 0.08, h[1]), TEXT)
    _text(ax, h[0] - 0.15, h[1] * 0.85, f"north part\n{_minus(h[1], 4)}", TEXT, ha="right", box=True)
    _span(ax, (0.0, -0.08), (h[0], -0.08), TEXT)
    _text(ax, h[0] / 2, -0.17, f"east part\n{_minus(h[0], 4)}", TEXT, va="top", box=True)
    _text(ax, 0.62, -0.1, "1 m long", TEXT, va="top")
    _text(ax, 2.1, -1.95, NORTH_EAST, TEXT, ha="right")
    _save(fig, path)


def _naive_700(path):  # [s700arrow] [s700naive]
    d = _lap()
    dd = d["D"][STEP]
    naive = math.atan(dd[1] / dd[0])
    fig, ax = _canvas(5.0, (-1.1, 1.1), 0.0)
    a0 = np.array([-0.35, -0.15])
    _arrow(ax, a0, a0 + dd, TEXT, lw=3, scale=20)
    _text(ax, -1.05, 0.55, f"step 700:\n({_minus(dd[0], 3)}, {_minus(dd[1], 3)})", TEXT, ha="left")
    b0 = np.array([0.2, 0.45])
    tip = b0 + heading_vector(naive)
    _arrow(ax, b0, tip, TEXT, lw=3, ls=(0, (5, 3)), style="-")  # a dashed shaft, then a solid head
    _arrow(ax, tip - 0.02 * heading_vector(naive), tip, TEXT, lw=3, scale=22)
    _text(ax, 0.5, -0.7, f"arctan of the slope\narctan(Δy/Δx)\n= {_minus(naive, 4)} rad", TEXT)
    _text(ax, 1.05, 0.92, NORTH_EAST, TEXT, ha="right")
    _save(fig, path)


def _miss_series(d):
    """|wrap(angle - yaw)| for the arctan-of-slope angle, at every step."""
    D = d["D"]
    naive = np.arctan(D[:, 1] / D[:, 0])
    return np.abs(wrap_angle(naive - wrap_angle(d["yaw"][:-1]))), naive


def _naive_lap(path):  # [naive] [naive_ok]
    d = _lap()
    miss, _ = _miss_series(d)
    t = d["t"]
    fig = _fig(4.0)
    ax = fig.add_axes([0.13, 0.16, 0.84, 0.78])
    ax.scatter(t, miss, s=6, color=TEXT, alpha=0.6, linewidths=0, zorder=3)
    ax.axhline(1.0, color=BORDER, ls=(0, (5, 3)), lw=1.8, zorder=1)
    ax.set(xlabel="time into the lap (s)", ylabel="angle off the yaw (rad)", ylim=(-0.2, PI + 0.4))
    ax.set_yticks([0, 1, 2, PI], ["0", "1", "2", "π"])
    _text(ax, 3.0, 1.0, "1 rad", TEXT, ha="left", va="bottom")
    _text(ax, 44, 2.35, f"arctan of the slope, moving\nwest: {int((d['D'][:, 0] < 0).sum())} steps, about π off", TEXT, box=True)
    _text(ax, 44, 0.55, "moving east: near 0", TEXT, box=True)
    _save(fig, path)


def _opposite_arrows(path):  # [opp]
    fig, ax = _canvas(5.0, (-2.0, 2.0), 0.0)
    o = np.zeros(2)
    for v, label, pos, va in (((1.0, 1.0), "(1, 1)\nslope 1", (1.0, 1.3), "bottom"), ((-1.0, -1.0), "(−1, −1)\nslope 1", (-1.0, -1.3), "top")):
        _arrow(ax, o, np.array(v), TEXT, lw=3, scale=22)
        _text(ax, *pos, label, TEXT, va=va)
    _text(ax, -1.0, 1.0, f"arctan of the slope gives\n{math.atan(1.0):.4f} rad for both", TEXT)
    _text(ax, 1.95, -1.7, NORTH_EAST, TEXT, ha="right")
    ax.plot(0, 0, marker="o", color=TEXT, ms=6, zorder=5)
    _save(fig, path)


def _atan2_lap(path):  # [atan2] [gap_bend] [gapmax]
    d = _lap()
    miss, _ = _miss_series(d)
    gap = np.abs(d["gap"])
    t, prog = d["t"], d["prog"]
    fig = _fig(7.0)
    top, bottom = fig.subplots(2, 1, gridspec_kw={"height_ratios": [1, 1]})
    top.scatter(t, miss, s=5, color=TEXT, alpha=0.4, linewidths=0, zorder=2)
    top.scatter(t, gap, s=5, color=THETA, linewidths=0, zorder=3)
    top.set(ylabel="angle off (rad)", ylim=(-0.2, PI + 0.3))
    top.set_yticks([0, 1, 2, PI], ["0", "1", "2", "π"])
    _text(top, 44, 0.5, "atan2", THETA, box=True)
    _text(top, 44, 2.5, "arctan of the slope", TEXT, box=True)
    bottom.scatter(t, gap, s=5, color=THETA, linewidths=0, zorder=3)
    bottom.set(xlabel="time into the lap (s)", ylabel="angle off (rad)", ylim=(0, 0.052))
    bend = np.zeros(len(t), bool)
    first = True
    for a, b in BENDS:
        m = (prog >= a) & (prog <= b)
        bend |= m
        bottom.axvspan(t[m].min(), t[m].max(), fc=TEXT, alpha=0.12, lw=0)
        if first:
            _text(bottom, (t[m].min() + t[m].max()) / 2, 0.0135, f"bend\nmiddles:\n{gap[bend].min():.3f} to\n{gap[bend].max():.3f} rad", TEXT)
            first = False
    k = int(np.argmax(gap))
    bottom.annotate(f"max {gap[k]:.3f} rad, step {k}", xy=(t[k], gap[k]), xytext=(t[k] - 12, 0.047), color=THETA,
                    arrowprops={"arrowstyle": "-", "color": THETA, "lw": 1.2}, ha="left")
    fig.tight_layout()
    _save(fig, path)


def _raw_diff(path):  # [raw] [jumps] [raw_after]
    d = _lap()
    raw = d["A"] - d["yaw"][:-1]
    t, prog = d["t"], d["prog"]
    fig = _fig(4.5)
    ax = fig.add_axes([0.18, 0.17, 0.79, 0.77])
    back = (prog >= BACK_STRAIGHT[0]) & (prog <= BACK_STRAIGHT[1])
    ax.axvspan(t[back].min(), t[back].max(), fc=TEXT, alpha=0.12, lw=0)
    ax.plot(t, raw, color=THETA, lw=0.4, alpha=0.3, zorder=2)
    ax.scatter(t, raw, s=6, color=THETA, linewidths=0, zorder=3)
    for y in (0.0, -2 * PI):
        ax.axhline(y, color=BORDER, ls=(0, (5, 3)), lw=1.8, zorder=1)
    ax.set(xlabel="time into the lap (s)", ylabel="atan2 − yaw (rad)", ylim=(-2 * PI - 0.6, 1.7))
    ax.set_yticks([0, -2, -4, -2 * PI], ["0", "−2", "−4", "−2π"])
    _text(ax, 0.5 * (t[back].min() + t[back].max()), 0.95, "back\nstraight", TEXT)
    far = int((np.abs(raw) > 1).sum())
    _text(ax, 1.0, -2.8, f"{far} of {len(raw)} steps\nmore than 1 rad\nfrom 0", THETA, ha="left", box=True)
    lo, hi = 48.0, 50.0  # a 2 s window of the back straight, where the steps flip between the two lines
    ax.axvspan(lo, hi, fc=THETA, alpha=0.25, lw=0, zorder=0)
    zoom = fig.add_axes([0.70, 0.46, 0.26, 0.30])
    window = (t >= lo) & (t <= hi)
    zoom.plot(t[window], raw[window], color=THETA, lw=0.8, alpha=0.5)
    zoom.scatter(t[window], raw[window], s=14, color=THETA, linewidths=0)
    for y in (0.0, -2 * PI):
        zoom.axhline(y, color=BORDER, ls=(0, (5, 3)), lw=1.8, zorder=1)
    zoom.set(xlim=(lo, hi), ylim=(-2 * PI - 0.8, 0.8), xticks=[lo, hi])
    zoom.set_yticks([0, -2 * PI], ["0", "−2π"])
    zoom.set_title("zoom: 2 s", fontsize=FS, pad=4)
    _save(fig, path)


def _yaw_wrapped(path):  # [yawrun] [yawcross]
    d = _lap()
    tel = d["tel"]
    prog, yaw = tel.progress_m, tel.yaw
    fig = _fig(6.5)
    top, bottom = fig.subplots(2, 1)
    top.plot(prog, yaw, color=THETA, lw=2.5, zorder=3)
    top.axhline(PI, color=BORDER, ls=(0, (5, 3)), lw=1.8)
    top.set(ylabel="yaw (rad)", ylim=(-0.4, 7.6))
    top.set_yticks([0, PI, 2 * PI], ["0", "π", "2π"])
    top.annotate(f"{yaw[-1]:.4f} rad", xy=(prog[-1], yaw[-1]), xytext=(prog[-1] - 40, yaw[-1] + 0.9), color=THETA, ha="right",
                 arrowprops={"arrowstyle": "-", "color": THETA, "lw": 1.2})
    bottom.scatter(prog, wrap_angle(yaw), s=6, color=THETA, linewidths=0, zorder=3)
    for y in (PI, -PI):
        bottom.axhline(y, color=BORDER, ls=(0, (5, 3)), lw=1.8)
    lo, hi = BACK_STRAIGHT[0] + 42.6, BACK_STRAIGHT[1] + 1.4  # 403.9 to 442.7 m
    bottom.axvspan(lo, hi, fc=TEXT, alpha=0.12, lw=0)
    bottom.set(xlabel="distance along the road (m)", ylabel="wrapped yaw (rad)", ylim=(-4.6, 4.6))
    bottom.set_yticks([-PI, 0, PI], ["−π", "0", "π"])
    bottom.annotate("crosses π\n39 times", xy=(lo, -2.2), xytext=(190, -2.2), color=TEXT, ha="center", va="center",
                    arrowprops={"arrowstyle": "-", "color": TEXT, "lw": 1.2}, bbox=LABEL_BOX)
    bottom.annotate(f"{_minus(wrap_angle(yaw[-1]), 4, sign=False)} rad", xy=(prog[-1], wrap_angle(yaw[-1])), xytext=(620, -3.9), color=THETA,
                    ha="center", va="center", arrowprops={"arrowstyle": "-", "color": THETA, "lw": 1.2})
    fig.tight_layout()
    _save(fig, path)


def _wrap_saw(path):
    a_end = 10.0
    edges = [-a_end, -3 * PI, -PI, PI, 3 * PI, a_end]
    fig = _fig(4.5)
    ax = fig.add_axes([0.16, 0.16, 0.8, 0.8])
    for lo, hi in zip(edges[:-1], edges[1:]):
        a = np.linspace(lo, hi, 50, endpoint=False)
        ax.plot(a, wrap_angle(a), color=THETA, lw=2.5, zorder=3)
    for k in (-3, -1, 1):
        ax.plot(k * PI, -PI, marker="o", color=THETA, ms=8, zorder=5)
    for k in (-1, 1, 3):
        ax.plot(k * PI, PI, marker="o", color=THETA, mfc=PANEL, ms=8, mew=2, zorder=5)
    for y in (PI, -PI):
        ax.axhline(y, color=BORDER, ls=(0, (5, 3)), lw=1.8, zorder=1)
    ax.set(xlabel="a (rad)", ylabel="wrap(a) (rad)", xlim=(-a_end, a_end), ylim=(-4.2, 4.2))
    ax.set_yticks([-PI, 0, PI], ["−π", "0", "π"])
    end = _lap()["yaw"][-1]
    ax.plot(end, wrap_angle(end), marker="o", color=TEXT, ms=8, zorder=6)
    ax.annotate(f"{end:.4f}\nwraps to\n{_minus(wrap_angle(end), 4, sign=False)}", xy=(end, wrap_angle(end)), xytext=(2.7, -1.0), color=TEXT, ha="center",
                arrowprops={"arrowstyle": "-", "color": TEXT, "lw": 1.2})
    ax.annotate("wrap(π) = −π", xy=(PI, -PI), xytext=(5.5, -3.6), color=TEXT, ha="center", arrowprops={"arrowstyle": "-", "color": TEXT, "lw": 1.2})
    _save(fig, path)


def _north_hand(path):  # [east] [north]
    fig = _fig(9.0)
    axes = fig.subplots(2, 1)
    for ax, theta, ahead_label, left_label in ((axes[0], 0.0, "2 m ahead", "3 m left"), (axes[1], PI / 2, "3 m ahead", "2 m right")):
        h = heading_vector(theta)
        cone = np.array([2.0, 3.0])
        ahead = float(cone @ h)
        ax.set_axis_off()
        ax.set(xlim=(-2.6, 4.1), ylim=(-2.0, 4.5), aspect="equal")
        o = np.zeros(2)
        _car(ax, 0.0, 0.0, theta, scale=0.5)
        _arrow(ax, o, h, EGO, lw=3.5, scale=22)
        foot = ahead * h
        ax.plot(*np.array([o, foot, cone]).T, color=BORDER, lw=1.6, ls=(0, (4, 3)), zorder=1)
        ax.plot(*cone, marker="^", color=SCENE["white"], mec=BORDER, mew=2, ms=16, zorder=6)
        if theta == 0.0:
            _span(ax, (0.0, -0.7), (2.0, -0.7), TEXT)
            _text(ax, 1.0, -1.2, ahead_label, TEXT, box=True)
            _span(ax, (2.5, 0.0), (2.5, 3.0), TEXT)
            _text(ax, 3.3, 1.5, left_label, TEXT, box=True)
        else:
            _span(ax, (-0.7, 0.0), (-0.7, 3.0), TEXT)
            _text(ax, -1.25, 1.5, ahead_label, TEXT, ha="right", box=True)
            _span(ax, (0.0, 3.5), (2.0, 3.5), TEXT)
            _text(ax, 1.0, 4.05, left_label, TEXT, box=True)
        _text(ax, 4.0, -1.7, NORTH_EAST, TEXT, ha="right")
    fig.tight_layout()
    _save(fig, path)


def _left_arrow(path):  # [s700yaw] [cone_n]
    d = _lap()
    yaw = d["tel"].yaw[STEP]
    h = heading_vector(yaw)
    n = np.array([-h[1], h[0]])
    fig, ax = _canvas(4.5, (-4.2, 2.4), 0.1)
    o = np.zeros(2)
    _car(ax, 0.0, 0.0, yaw, scale=0.5)
    _arrow(ax, o, h, EGO, lw=3.5, scale=22)
    _arrow(ax, o, n, shades(EGO, 3)[2], lw=3.5, scale=22)
    _right_angle(ax, o, h, n, 0.18)
    _arc(ax, o, 0.62, yaw, yaw + PI / 2, TEXT, lw=2.0)
    _text(ax, -2.0, 0.35, "quarter turn\nanticlockwise", TEXT, box=True)
    _text(ax, h[0] - 0.1, h[1] + 0.45, f"ĥ = ({_minus(h[0], 4)}, {_minus(h[1], 4)})", EGO)
    _text(ax, 2.35, -2.45, NORTH_EAST, TEXT, ha="right")
    _text(ax, n[0] - 0.05, n[1] - 0.12, f"n̂ = ({_minus(n[0], 4)}, {_minus(n[1], 4)})", shades(EGO, 3)[2], ha="right", va="top")
    _save(fig, path)


def _cone_parts(path):  # [cone_v] [cone_dot] [cone_cross]
    d = _lap()
    road, tel = d["road"], d["tel"]
    x0, y0, yaw = tel.x[STEP], tel.y[STEP], tel.yaw[STEP]
    h = heading_vector(yaw)
    n = np.array([-h[1], h[0]])
    s_car, _ = road.project(x0, y0)
    v = road.offset(s_car + 12.0, -4.3) - np.array([x0, y0])
    ahead, left = float(v @ h), float(h[0] * v[1] - h[1] * v[0])
    fig, ax = _canvas(6.5, (-13.0, 5.0), 5.4)
    o = np.zeros(2)
    foot = ahead * h
    _car(ax, 0.0, 0.0, yaw)
    ax.plot(*np.array([-1.5 * h, 14.0 * h]).T, color=BORDER, lw=1.8, ls=(0, (5, 3)), zorder=1)
    _arrow(ax, o, h * 1.5, EGO, lw=3, scale=20)
    _arrow(ax, o, n * 1.5, shades(EGO, 3)[2], lw=3, scale=20)
    _arrow(ax, o, v, TEXT, lw=2, scale=18)
    ax.plot(*v, marker="^", color=SCENE["white"], mec=BORDER, mew=2, ms=16, zorder=6)
    _right_angle(ax, foot, -h, (v - foot) / np.hypot(*(v - foot)), 0.6)
    ax.plot(*np.array([o, foot]).T, color=EGO, lw=6, alpha=0.8, solid_capstyle="butt", zorder=2)
    ax.plot(*np.array([foot, v]).T, color=shades(EGO, 3)[2], lw=6, alpha=0.8, solid_capstyle="butt", zorder=2)
    _text(ax, 4.9, -4.6, NORTH_EAST, TEXT, ha="right")
    _text(ax, v[0] - 0.5, v[1] + 1.6, f"({_minus(v[0], 3)}, {_minus(v[1], 3)}), {np.hypot(*v):.2f} m", TEXT)
    _text(ax, *(foot / 2) + [-0.8, 0.0], f"forward {ahead:.2f} m", EGO, ha="right", box=True)
    _text(ax, v[0] + 1.4, v[1] + 0.1, f"left {_minus(left, 2, sign=False)} m\n({-left:.2f} m right)", shades(EGO, 3)[2], ha="left", box=True)
    _save(fig, path)


def _move_vs_point(path):  # [decomp] [gap_straight] [gap_bend] [gapmax]
    d = _lap()
    tel = d["tel"]
    t = d["t"]
    gap = d["gap"]
    decomp = tel.steer[1:] + 0.5 * np.diff(tel.yaw)
    fig = _fig(4.5)
    ax = fig.add_axes([0.21, 0.17, 0.76, 0.77])
    ax.plot(t, gap, color=TEXT, lw=5, alpha=0.5, zorder=2)
    ax.plot(t, decomp, color=DELTA, lw=1.8, ls=(0, (4, 2)), zorder=3)
    ax.set(xlabel="time into the lap (s)", ylabel="angle (rad)", ylim=(-0.07, 0.075))
    _text(ax, 44, 0.062, "thick: GPS step direction − yaw", TEXT, box=True)
    _text(ax, 44, 0.048, "dashed: δ + ½Δθ", DELTA, box=True)
    _text(ax, 44, -0.03, "straights: at most 0.0091 rad\nbend middles: about 0.036 rad\nmax 0.041 rad", TEXT, box=True)
    _save(fig, path)


def _move_point_diagram(path):
    fig, ax = _canvas(5.0, (-1.0, 5.4), 1.4)
    o = np.zeros(2)
    th, dl, hf = 0.28, 0.5, 0.2  # drawn angles, enlarged
    _arrow(ax, o, 3.0 * heading_vector(th), THETA, lw=3, scale=22)
    _arrow(ax, o, 3.0 * heading_vector(th + dl), DELTA, lw=3, scale=22)
    _arrow(ax, o, 3.0 * heading_vector(th + dl + hf), TEXT, lw=3, scale=22)
    _arrow(ax, o, 3.0 * np.array([1.0, 0.0]), BORDER, lw=1.6, ls=(0, (5, 3)), style="-")
    _arc(ax, o, 1.2, th, th + dl, DELTA, lw=2.2)
    _arc(ax, o, 1.7, th + dl, th + dl + hf, THETA, lw=2.2)
    _text(ax, 3.05, 0.83, "θ (yaw)", THETA, ha="left")
    _text(ax, 5.35, -1.35, NORTH_EAST, TEXT, ha="right")
    _text(ax, 2.25, 3.0 * math.sin(th + dl) + 0.1, "θ + δ (wheel)", TEXT, ha="left", va="bottom")
    _text(ax, -0.5, 3.0, "θ + δ + ½Δθ (chord)", TEXT, ha="left", va="bottom")
    _text(ax, 1.5, 0.9, "δ", DELTA, ha="left")
    _text(ax, 0.85, 1.75, "½Δθ", THETA, ha="right")
    _text(ax, 2.85, -0.25, "east", TEXT)
    _text(ax, 1.8, -0.85, "angles enlarged, not to scale", TEXT)
    _save(fig, path)


def _next_lesson(path):  # [next]
    road = _lap()["road"]
    s0 = 100.0
    x0, y0, yaw = road.ego_pose(s0)
    fig, ax = _canvas(4.5, (86.0, 126.0), 4.0)
    _scene(ax, road, 85.0, 135.0)
    _car(ax, x0, y0, yaw)
    _arrow(ax, np.array([x0, y0]), np.array([x0, y0]) + 4.0 * heading_vector(yaw), EGO, lw=3.5)
    for s in (108.0, 116.0, 124.0):
        for d, colour in ((0.0, SCENE["yellow"]), (-LANE_WIDTH, SCENE["white"])):
            ax.plot(*road.offset(s, d), marker="o", color=colour, mec=PANEL, mew=1.5, ms=7, zorder=5)
            _text(ax, *road.offset(s, d) + [0.0, 1.4 if d == 0.0 else -1.6], "?", TEXT)
    _text(ax, 108.0, 12.0, "? m ahead, ? m left", TEXT, box=True)
    _save(fig, path)


def _gps_law(path):  # the [gps] loop of measure.py
    SIGMA, SEEDS, KS = 0.05, 500, np.arange(1, 41)
    tel = run(Car(), "straight")
    Ps = np.stack([tel.x, tel.y], axis=-1)
    spread = []
    errs = {k: [] for k in KS}
    for seed in range(SEEDS):
        noisy = Ps + np.random.default_rng(seed).normal(0.0, SIGMA, Ps.shape)
        for k in KS:
            dk = noisy[k:] - noisy[:-k]
            errs[k].append(wrap_angle(direction_angle(dk) - tel.yaw[:-k]))
    spread = np.array([np.concatenate(errs[k]).std() for k in KS])
    law = math.sqrt(2) * SIGMA / (KS * TARGET_SPEED * DT)
    fig = _fig(4.5)
    ax = fig.add_axes([0.2, 0.17, 0.77, 0.77])
    ax.plot(KS, law, color=TEXT, lw=2, zorder=2)
    ax.scatter(KS, spread, s=22, color=THETA, linewidths=0, zorder=3)
    ax.axhline(0.01, color=BORDER, ls=(0, (5, 3)), lw=1.8, zorder=1)
    ax.set(yscale="log", xlabel="fix spacing k (steps)", ylabel="spread of yaw error (rad)", ylim=(0.0035, 0.25))
    ticks = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    ax.set_yticks(ticks, [f"{v:g}" for v in ticks])
    ax.minorticks_off()
    ax.set_xticks([1, 10, 18, 30, 40], ["1", "10", "18", "30", "40"])
    _text(ax, 21, 0.0125, "0.01 rad", TEXT, ha="left", va="bottom")
    _text(ax, 12, 0.11, "√2 σ / (0.4 k)", TEXT, ha="left", box=True)
    _text(ax, 12, 0.07, "dots: 500 seeds", THETA, ha="left", box=True)
    _save(fig, path)


def make(out_dir):
    with matplotlib.rc_context(STYLE):
        _result_first(out_dir / "result_first.png")
        _world_lap(out_dir / "world_lap.png")
        _step_arrow(out_dir / "step_arrow.png")
        _theta_psi(out_dir / "theta_psi.png")
        _unit_circle(out_dir / "unit_circle.png")
        _naive_700(out_dir / "naive_700.png")
        _naive_lap(out_dir / "naive_lap.png")
        _opposite_arrows(out_dir / "opposite_arrows.png")
        _atan2_lap(out_dir / "atan2_lap.png")
        _raw_diff(out_dir / "raw_diff.png")
        _yaw_wrapped(out_dir / "yaw_wrapped.png")
        _wrap_saw(out_dir / "wrap_saw.png")
        _north_hand(out_dir / "north_hand.png")
        _left_arrow(out_dir / "left_arrow.png")
        _cone_parts(out_dir / "cone_parts.png")
        _move_vs_point(out_dir / "move_vs_point.png")
        _move_point_diagram(out_dir / "move_point_diagram.png")
        _next_lesson(out_dir / "next_lesson.png")
        _gps_law(out_dir / "gps_law.png")
