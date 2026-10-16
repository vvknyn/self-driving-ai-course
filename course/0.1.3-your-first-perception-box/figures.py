"""Figures for lecture 0.1.3 Your first perception box.  `python scripts/make_figures.py course/0.1.3-your-first-perception-box`."""
import math
from functools import lru_cache

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator

from zero2fsd.car import Car, LaneEstimate, Observation, provided, run
from zero2fsd.car.provided import FAR_ROWS, NEAR_ROWS, estimate_lane, lane_point, paint_masks
from zero2fsd.grade.units.u0_1_3 import pose_trials
from zero2fsd.palette import CHROME, QUANTITY, SCENE, shades
from zero2fsd.sim import BicycleState, Camera, make_scenario
from zero2fsd.sim.camera import WHITE, YELLOW, to_vehicle
from zero2fsd.sim.dynamics import step as sim_step
from zero2fsd.sim.scenarios import TARGET_SPEED
from zero2fsd.sim.world import EGO_LANE_CENTER, WHITE_LINE_LATERAL

CAMERA = Camera()
HORIZON = CAMERA.ground_to_pixel(1e9, 0.0)[1]
WORKED = ("straight", 20.0, 0.4, 0.05)  # scenario, s, offset, heading: the lab renders this same frame
STRETCH_BANDS = ((125, 180), (106, 125), (95, 106))  # the curve box's three bands
TEXT, PANEL, BACKGROUND, BORDER = CHROME["text"], CHROME["panel"], CHROME["background"], CHROME["border"]
DIM = shades(BORDER, 5)[3]  # a quieter grey from the same chrome, for helper lines
COAL, BLUE, PINK = QUANTITY["perception"], QUANTITY["e_off"], QUANTITY["psi"]  # the box's answers, the true error e, the heading
CLEAR = (0, 0, 0, 0)  # transparent: an outline or a marker with no fill
LABEL_BOX = {"boxstyle": "round,pad=0.2", "fc": PANEL, "ec": BORDER}  # text that sits on top of a picture
IMG_X, IMG_Y = (-0.5, CAMERA.width - 0.5), (CAMERA.height - 0.5, -0.5)  # image coordinates: row 0 at the top

# Legibility: every text in every figure is FS points on a PNG W_IN inches wide, so a digit is about FS pixels tall at
# dpi 100 (dpi cancels) and about FS * 330 / (100 * W_IN) when the PNG is shown 330 px wide.  PNGs are saved at figstyle's dpi (savefig.dpi).  `_save` measures it and refuses under MIN_DIGIT_PX.
FS, W_IN, DPI, SHOWN_PX, MIN_DIGIT_PX = 18, 6.5, 100, 330, 9
RC = {"font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
      "axes.titlepad": 8, "lines.linewidth": 2, "axes.linewidth": 1.2}
LEGIBILITY = {}  # figure name -> (PNG width px, smallest text, its font size pt, effective digit px at 330 px wide)


def _digit_px(renderer, text):
    return renderer.get_text_width_height_descent("0", text.get_fontproperties(), False)[1]


def _save(fig, path):
    """Save after checking that no text is under MIN_DIGIT_PX tall when the PNG is shown SHOWN_PX wide."""
    renderer = fig.canvas.get_renderer()
    shown = [(_digit_px(renderer, t) * SHOWN_PX / (fig.get_figwidth() * DPI), t) for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    px, smallest = min(shown, key=lambda pair: pair[0])
    assert px >= MIN_DIGIT_PX, f"{path.name}: '{smallest.get_text()}' is {px:.2f} px tall at {SHOWN_PX} px wide"
    LEGIBILITY[path.name] = (round(fig.get_figwidth() * matplotlib.rcParams["savefig.dpi"]), smallest.get_text(), smallest.get_fontsize(), round(px, 2))
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
    return ax.plot(x, y, **style)


def _view(scenario, s, offset=0.0, heading=0.0):
    road = make_scenario(scenario).road
    state = BicycleState(*road.ego_pose(s, offset, heading), TARGET_SPEED, 0.0)
    return road, state, CAMERA.render(road, state)


def _show(ax, frame):
    ax.imshow(frame, interpolation="nearest", aspect="equal")
    ax.set(xlim=IMG_X, ylim=IMG_Y)
    _bare(ax)


def _tag(ax, x, y, text, **kwargs):
    """A label with a panel-coloured box, for text that sits on a picture."""
    kwargs = {"ha": "center", "va": "center", **kwargs}
    return ax.text(x, y, text, bbox=LABEL_BOX, **kwargs)


def _cross(ax, u, v, size=16):
    for colour, width in ((BACKGROUND, 7), (COAL, 3.5)):
        ax.plot(u, v, marker="x", ms=size, mew=width, color=colour, ls="none")


def _dot(ax, x, y, colour, size=11):
    ax.plot(x, y, marker="o", ms=size, mfc=colour, mec=BACKGROUND, mew=2, ls="none", zorder=6)


def _bands(ax, bands=(NEAR_ROWS, FAR_ROWS)):
    for a, b in bands:
        ax.axhspan(a - 0.5, b - 0.5, color=TEXT, alpha=0.22)
        for edge in (a - 0.5, b - 0.5):
            ax.axhline(edge, color=TEXT, lw=1.5, ls="--")


def _pointer(ax, text, xy, xytext, colour=TEXT, **kwargs):
    return ax.annotate(text, xy=xy, xytext=xytext, color=colour, arrowprops={"arrowstyle": "->", "color": colour, "lw": 2, "shrinkA": 2, "shrinkB": 3}, **kwargs)


def _top_down(ax, xlabel="metres ahead", ylabel="metres left"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _ticks(ax, 5, 5)
    ax.plot(0, 0, marker=">", ms=15, mfc=TEXT, mec=BACKGROUND, mew=2, ls="none", zorder=7)


def _paint_dot(ax, forward, left, paint):
    _dot(ax, forward, left, SCENE["yellow" if paint == YELLOW else "white"], 12)


def _ground_line(road, state, s, lateral, ahead):
    """A line of constant lateral position on the road, as (forward, left) in the car's frame."""
    forward, left = to_vehicle(state, *road.offset(np.linspace(s - 2.0, s + ahead + 4.0, 400), lateral).T)
    return forward, left


def _lane_points(masks, bands=(NEAR_ROWS, FAR_ROWS)):
    return [lane_point(masks, rows) for rows in bands]


def _centroid(mask, rows):
    v, u = np.nonzero(mask[slice(*rows)])
    return u.mean(), v.mean() + rows[0]


def _arc(ax, centre, radius, a0, a1, **style):
    angle = np.linspace(a0, a1, 60)
    return ax.plot(centre[0] + radius * np.cos(angle), centre[1] + radius * np.sin(angle), **style)


# ---------------------------------------------------------------- 1 box_result


def box_result(path):
    road, state, frame = _view(*WORKED)
    masks = paint_masks(frame)
    est = estimate_lane(Observation(frame, TARGET_SPEED, 0.0))
    fig, (img, top) = _panels([(IMG_X, IMG_Y, None), ((-0.6, 9.0), (-3.2, 3.6), None)], gap=0.95)
    _show(img, frame)
    _bands(img)
    _tag(img, 8, 105, "far band", ha="left")
    _tag(img, 160, 152, "near band")
    for rows in (NEAR_ROWS, FAR_ROWS):
        for mask in masks:
            _cross(img, *_centroid(mask, rows))
    near, far = _lane_points(masks)
    for rows in (NEAR_ROWS, FAR_ROWS):
        for mask, paint in zip(masks, (YELLOW, WHITE)):
            _paint_dot(top, *CAMERA.pixel_to_ground(*_centroid(mask, rows)), paint)
    ahead = np.array([0.0, 8.5])
    slope = (far[1] - near[1]) / (far[0] - near[0])
    _line(top, ahead, near[1] + slope * (ahead - near[0]), color=COAL, lw=3)
    for point in (near, far):
        _dot(top, *point, COAL, 13)
    _top_down(top)
    top.text(0.0, 3.1, f"offset {est.offset_m:+.2f} m, heading {est.heading_rad:+.3f} rad", color=COAL, va="center")
    top.text(0.0, 2.35, f"truth: e = {WORKED[2]:+.2f} m, ψ = {WORKED[3]:+.3f} rad", color=TEXT, va="center")
    _save(fig, path)


# ---------------------------------------------------------------- 2 radians


def radians(path):
    fig, (one, ten) = _panels([((-1.2, 3.4), (-1.2, 1.4), None), ((-0.6, 10.6), (-0.9, 1.5), None)], gap=0.5, bottom=0.2, left=0.2)
    for ax in (one, ten):
        _bare(ax)
    one.add_patch(Circle((0, 0), 1.0, fill=False, ec=DIM, lw=2))
    _line(one, [0, 1], [0, 0], color=TEXT, lw=2.5)
    _line(one, [0, math.cos(1)], [0, math.sin(1)], color=TEXT, lw=2.5)
    _arc(one, (0, 0), 1.0, 0, 1.0, color=TEXT, lw=7)
    _arc(one, (0, 0), 0.42, 0, 1.0, color=TEXT, lw=2.5)
    one.text(0.5, -0.12, "radius 1 m", ha="center", va="top")
    _pointer(one, "1 rad", (0.4, 0.22), (1.3, 0.2), va="center")
    _pointer(one, "arc 1 m", (0.88, 0.5), (1.5, 0.85), va="center")
    angle = 0.05
    _line(ten, [0, 10], [0, 0], color=TEXT, lw=2.5)
    _line(ten, [0, 10 * math.cos(angle)], [0, 10 * math.sin(angle)], color=TEXT, lw=2.5)
    _arc(ten, (0, 0), 10.0, 0, angle, color=TEXT, lw=7)
    _arc(ten, (0, 0), 3.0, 0, angle, color=TEXT, lw=2.5)
    ten.text(5, -0.15, "radius 10 m", ha="center", va="top")
    _pointer(ten, "0.05 rad", (3.0, 0.07), (1.2, 0.9), va="center")
    _pointer(ten, "arc 0.50 m", (10.0, 0.25), (7.2, 1.0), va="center")
    _save(fig, path)


# ---------------------------------------------------------------- 3 offset_heading


def offset_heading(path):
    e, psi, along = WORKED[2], WORKED[3], 10.0
    end = e + along * math.tan(psi)
    fig, (ax,) = _panels([((-0.8, 14.2), (-2.4, 2.4), 3.4)], bottom=0.2)
    _bare(ax)
    _line(ax, [-0.8, 14.2], [1.8, 1.8], color=SCENE["yellow"], lw=5)
    _line(ax, [-0.8, 14.2], [-1.8, -1.8], color=SCENE["white"], lw=5)
    _line(ax, [-0.8, 14.2], [0, 0], color=TEXT, lw=1.5, ls=":")
    _line(ax, [0, along + 0.3], [e, e], color=TEXT, lw=1.5, ls=":")
    _line(ax, [0, along], [e, end], color=PINK, lw=2.5, ls="--")
    ax.annotate("", xy=(1.3, e + 1.3 * math.tan(psi)), xytext=(0, e), arrowprops={"arrowstyle": "-|>", "color": TEXT, "lw": 4, "mutation_scale": 22})
    _line(ax, [0, 0], [0, e], color=BLUE, lw=7)
    _pointer(ax, f"e = {e:+.2f} m", (0.0, e / 2), (0.9, -0.75), BLUE, va="center")
    _arc(ax, (0, e), 3.5, 0, psi, color=PINK, lw=3)
    _pointer(ax, f"\u03c8 = {psi:+.3f} rad", (3.5, e + 0.09), (3.4, -1.3), PINK, va="center")
    _line(ax, [along + 0.3] * 2, [e, end], color=TEXT, lw=2.5)
    ax.text(along + 0.6, (e + end) / 2, f"+{along * math.tan(psi):.2f} m", va="center")
    ax.text(along, end + 0.3, f"{end:.2f} m after {along:.0f} m", ha="right", va="bottom")
    _save(fig, path)


# ---------------------------------------------------------------- 4 slide_shift


def slide_shift(path):
    fig, axes = _panels([(IMG_X, IMG_Y, None)] * 2, gap=0.75, top=0.55, bottom=0.2, left=0.2)
    marks = {}
    for ax, offset, title in zip(axes, (0.0, 0.5), ("car on the lane centre", "car 0.5 m left of the lane centre")):
        frame = _view("straight", 20.0, offset)[2]
        white = paint_masks(frame)[1]
        _show(ax, frame)
        ax.set_title(title)
        for row in (150, 100):
            ax.axhline(row, color=TEXT, lw=1.5, ls="--")
            marks[offset, row] = np.nonzero(white[row])[0].mean()
            _dot(ax, marks[offset, row], row, SCENE["white"], 13)
    for row, text_row in ((150, 55), (100, 25)):
        a, b = marks[0.0, row], marks[0.5, row]
        axes[1].plot(a, row, marker="o", ms=13, mfc=CLEAR, mec=TEXT, mew=2.5, ls="none", zorder=5)
        _tag(axes[1], 160, text_row, f"row {row}: {a:.1f} \u2192 {b:.1f}, {b - a:.1f} columns")
        _tag(axes[0], 36, row - 12, f"row {row}")
    _save(fig, path)


# ---------------------------------------------------------------- 5 perspective_grid


def perspective_grid(path):
    frame = _view("straight", WORKED[1])[2]
    fig, (ax,) = _panels([(IMG_X, IMG_Y, None)], bottom=0.2)
    _show(ax, frame)
    ahead = np.linspace(1.5, 70.0, 400)
    for left in range(-8, 9):
        _line(ax, *CAMERA.ground_to_pixel(ahead, np.full_like(ahead, left)), color=TEXT, lw=1.2, alpha=0.8)
    sideways = np.linspace(-9.0, 9.0, 200)
    for forward in range(2, 71):
        _line(ax, *CAMERA.ground_to_pixel(np.full_like(sideways, forward), sideways), color=TEXT, lw=1.2, alpha=0.8)
    ax.axhline(HORIZON, color=TEXT, lw=3.5, ls="--")
    _tag(ax, 160, HORIZON - 14, f"horizon, row {HORIZON:.1f}")
    _save(fig, path)


# ---------------------------------------------------------------- 6 tan_curve


def tan_curve(path):
    run_m, rise = 10.0, 0.5
    fig, (tri, curve) = _panels([((-0.6, 14.6), (-1.0, 2.5), None), ((-0.03, 1.1), (-0.05, 1.8), 3.6)], gap=0.9, bottom=0.9, left=0.9)
    _bare(tri)
    _line(tri, [0, run_m, run_m, 0], [0, 0, rise, 0], color=TEXT, lw=3)
    tri.text(run_m / 2, -0.15, f"run {run_m:.0f} m", ha="center", va="top")
    tri.text(run_m + 0.35, rise / 2, f"rise {rise} m", va="center")
    tri.text(0, 1.9, f"slope {rise / run_m:.2f}", va="center")
    tri.text(0, 1.2, f"angle arctan {rise / run_m:.2f} = {math.atan(rise / run_m):.5f} rad", va="center")
    _arc(tri, (0, 0), 2.5, 0, math.atan(0.05), color=TEXT, lw=3)
    theta = np.linspace(0, 1.0, 200)
    _line(curve, theta, np.tan(theta), color=TEXT, lw=3)
    _line(curve, theta, theta, color=DIM, lw=2.5, ls="--")
    for a in (0.05, 0.6, 1.0):
        _dot(curve, a, math.tan(a), TEXT, 10)
    curve.set(xlabel="angle \u03b8 (rad)")
    _ticks(curve, 4, 5)
    _pointer(curve, "tan \u03b8", (0.9, math.tan(0.9)), (0.45, 1.4), va="center")
    _pointer(curve, "slope = angle", (0.3, 0.3), (0.02, 0.85), va="center")
    _pointer(curve, f"tan 0.05 = {math.tan(0.05):.5f}", (0.05, 0.05), (0.2, 0.0), va="center")
    _pointer(curve, f"tan 0.6 = {math.tan(0.6):.3f}", (0.6, math.tan(0.6)), (0.62, 0.35), va="center")
    curve.text(0.97, math.tan(1.0) + 0.1, f"tan 1 = {math.tan(1.0):.3f}", ha="right", va="bottom")
    _save(fig, path)


# ---------------------------------------------------------------- 7 pinhole_screen


def pinhole_screen(path):
    f, cy, row = CAMERA.f, CAMERA.cy, 150
    drop, alpha = row - cy, math.atan((row - CAMERA.cy) / CAMERA.f)
    fig, (ax,) = _panels([((-12, 178), (-100, 100), None)], bottom=0.2, left=0.2)
    _bare(ax)
    _line(ax, [f, f], [cy, -(CAMERA.height - 1 - cy)], color=TEXT, lw=5)
    _line(ax, [0, f + 12], [0, 0], color=TEXT, lw=1.5, ls="--")
    _line(ax, [0, f], [0, -drop], color=TEXT, lw=3)
    _dot(ax, 0, 0, TEXT, 13)
    _dot(ax, f, -drop, TEXT, 11)
    ax.text(-6, -16, "pinhole", ha="left", va="top")
    ax.annotate("", xy=(f, 45), xytext=(0, 45), arrowprops={"arrowstyle": "<->", "color": TEXT, "lw": 2})
    ax.text(f / 2, 53, f"f = {f:.0f} px", ha="center", va="bottom")
    ax.annotate("", xy=(f - 8, -drop), xytext=(f - 8, 0), arrowprops={"arrowstyle": "<->", "color": TEXT, "lw": 2})
    ax.text(f - 16, -drop / 2, f"{drop:.1f} px", ha="right", va="center")
    ax.text(f - 6, 8, f"middle row {cy}", ha="right", va="bottom")
    ax.text(f - 6, -drop - 8, f"row {row}", ha="right", va="top")
    _arc(ax, (0, 0), 100, 0, -alpha, color=TEXT, lw=3)
    _pointer(ax, f"{alpha:.4f} rad", (100 * math.cos(alpha / 2), -100 * math.sin(alpha / 2)), (75, -80), va="center")
    ax.text(-6, -92, f"slope {drop:.1f} / {f:.0f} = {drop / f:.3f}", ha="left", va="center")
    _save(fig, path)


# ---------------------------------------------------------------- 8 ray_side_view

STRETCH = 3.0  # ray_side_view draws heights three times larger than distances


def ray_side_view(path):
    h, pitch = CAMERA.mount_height, math.radians(5.0)
    top = STRETCH * h
    fig, (ax,) = _panels([((-4.4, 12.6), (-7.2, 5.6), None)], bottom=0.2, left=0.2)
    _bare(ax)
    _line(ax, [-4.4, 12.6], [0, 0], color=TEXT, lw=3)
    _line(ax, [0, 12.5], [top, top], color=TEXT, lw=2, ls="--")
    _line(ax, [0, 12.5], [top, top - STRETCH * 12.5 * math.tan(pitch)], color=DIM, lw=2.5, ls=":")
    ax.text(12.4, top + 0.25, f"row {HORIZON:.1f}: horizon", ha="right", va="bottom")
    ax.text(7.0, top - STRETCH * 7.0 * math.tan(pitch) + 0.15, "axis, 5\u00b0 down", ha="left", va="bottom")
    ax.annotate("", xy=(-0.7, 0), xytext=(-0.7, top), arrowprops={"arrowstyle": "<->", "color": TEXT, "lw": 2})
    ax.text(-0.9, top / 2, f"{h} m", ha="right", va="center")
    _dot(ax, 0, top, TEXT, 13)
    rows = (179, 150, 125, 115, 95)
    for k, row in enumerate(rows):
        d = CAMERA.pixel_to_ground(CAMERA.cx, row)[0]
        _line(ax, [0, d], [top, 0], color=TEXT, lw=2.5)
        depth = -1.0 - 1.15 * (len(rows) - 1 - k)
        _line(ax, [d, d], [0, depth], color=DIM, lw=1.5)
        ax.text(d + (-0.1 if row == 95 else 0.1), depth, f"row {row}: {d:.2f} m", ha="right" if row == 95 else "left", va="center")
    _save(fig, path)


# ---------------------------------------------------------------- 9 row_150_ray


def row_150_ray(path):
    h, row = CAMERA.mount_height, 150
    pitch, alpha = math.radians(5.0), math.atan((row - CAMERA.cy) / CAMERA.f)
    d = CAMERA.pixel_to_ground(CAMERA.cx, row)[0]
    theta = pitch + alpha
    fig, (ax,) = _panels([((-0.5, 4.2), (-0.95, 2.3), None)], bottom=0.2, left=0.2)
    _bare(ax)
    _line(ax, [-0.5, 4.2], [0, 0], color=TEXT, lw=3)
    _line(ax, [0, 3.5], [h, h], color=TEXT, lw=2, ls="--")
    _line(ax, [0, 3.5], [h, h - 3.5 * math.tan(pitch)], color=DIM, lw=2.5, ls=":")
    _line(ax, [0, d], [h, 0], color=TEXT, lw=3.5)
    _line(ax, [d, d], [0, h], color=TEXT, lw=1.5)
    ax.text(3.55, h, "level", va="center")
    ax.text(3.55, h - 3.5 * math.tan(pitch), "axis", va="center")
    _arc(ax, (0, h), 1.2, -pitch, 0, color=TEXT, lw=3)
    _arc(ax, (0, h), 1.8, -theta, -pitch, color=TEXT, lw=3)
    _arc(ax, (0, h), 2.4, -theta, 0, color=DIM, lw=2.5)
    _pointer(ax, f"{pitch:.4f} rad", (1.2 * math.cos(pitch / 2), h - 1.2 * math.sin(pitch / 2)), (0.4, 2.05), va="center")
    _pointer(ax, f"{alpha:.4f} rad", (1.8 * math.cos(pitch + alpha / 2), h - 1.8 * math.sin(pitch + alpha / 2)), (2.0, 1.85), va="center", ha="left")
    _pointer(ax, f"\u03b8 = {theta:.4f} rad", (2.2 * math.cos(theta), h - 2.2 * math.sin(theta)), (-0.4, 0.45), va="center")
    ax.annotate("", xy=(d + 0.1, h), xytext=(d + 0.1, 0), arrowprops={"arrowstyle": "<->", "color": TEXT, "lw": 2})
    ax.text(d + 0.2, h / 2, f"{h} m", va="center")
    ax.text(d / 2, -0.12, f"{d:.3f} m", ha="center", va="top")
    ax.text(d / 2, -0.55, f"slope tan \u03b8 = {math.tan(theta):.4f}", ha="center", va="top")
    _save(fig, path)


# ---------------------------------------------------------------- 10 rows_to_distance

ROWS_ALL = np.arange(80, 180)


def _row_distance(row):
    return float(CAMERA.pixel_to_ground(CAMERA.cx, row)[0])


def _metres_per_column(row):
    """Metres of road one column covers in a row: the left-right gap between neighbouring columns' ground points."""
    return float(CAMERA.pixel_to_ground(0.0, row)[1] - CAMERA.pixel_to_ground(1.0, row)[1])


def rows_to_distance(path):
    fig, (ax,) = _panels([((78, 182), (0, 54), 6.4)], left=1.0, bottom=0.9, top=0.2)
    d = np.array([_row_distance(r) for r in ROWS_ALL])
    ax.plot(ROWS_ALL, d, color=TEXT, lw=3)
    ax.set(xlabel="row", ylabel="metres ahead")
    ax.set_xticks([80, 100, 120, 140, 160, 179])
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    for row in (179, 150, 125, 100, 95, 80):
        _dot(ax, row, _row_distance(row), TEXT, 12)
    labels = {179: (180.5, 19, "right"), 150: (150, 29, "center"), 125: (125, 19, "center"), 100: (104, 13.5, "left"), 95: (99, 27, "left"), 80: (82, 49.5, "left")}
    for row, (x, y, ha) in labels.items():
        text = f"row {row}\n{_row_distance(row):.2f} m"
        if row == 80:
            ax.text(x, y, text, ha=ha, va="center")
        else:
            _pointer(ax, text, (row, _row_distance(row) + 0.9), (x, y), ha=ha, va="center")
    _save(fig, path)


# ---------------------------------------------------------------- 11 column_width

COLUMN_ROWS = (80, 95, 100, 125, 150, 179)


def column_width(path):
    widths = np.array([_metres_per_column(r) for r in COLUMN_ROWS])
    lane = 3.6 / widths
    ratio = widths[0] / widths[-1]
    assert [round(w * 100, 1) for w in widths] == [31.2, 7.2, 5.7, 2.8, 1.9, 1.4] and round(ratio) == 23
    assert [round(x, 1) for x in lane] == [11.5, 49.9, 62.8, 126.8, 190.8, 265.1]
    fig, (top, bot) = _panels([((0, 44), (5.6, -0.6), 3.3), ((0, 330), (5.6, -0.6), 3.3)], left=1.35, right=0.3, gap=1.0, top=0.55, bottom=0.2)
    shown = ([f"{w * 100:.1f} cm" for w in widths], [f"{x:.1f}" if x < 20 else f"{x:.0f}" for x in lane])
    for ax, values, labels, title in zip((top, bot), (widths * 100, lane), shown, ("road one column covers", "columns across the 3.6 m lane")):
        ax.barh(range(len(COLUMN_ROWS)), values, color=DIM, edgecolor=TEXT, height=0.62)
        ax.set_yticks(range(len(COLUMN_ROWS)), [f"row {r}" for r in COLUMN_ROWS])
        ax.set_xticks([])
        ax.set_title(title)
        for k, (v, label) in enumerate(zip(values, labels)):
            ax.text(v + ax.get_xlim()[1] * 0.012, k, label, va="center", ha="left")
    top.text(0.8, 0, f"{round(ratio)}\u00d7 row 179", va="center", ha="left", color=BACKGROUND)
    _save(fig, path)


# ---------------------------------------------------------------- 12 ground_footprint

FOOT_ROWS = (179, 150, 125, 115, 95)


def ground_footprint(path):
    h, pitch = CAMERA.mount_height, math.radians(5.0)
    apex = h * math.tan(pitch)  # where every column's footprint line meets the ground: just ahead of the camera
    fig, (ax,) = _panels([((-0.5, 13.0), (-7.2, 7.2), None)], left=1.2, bottom=0.9, top=0.2)
    forward = np.linspace(apex, 13.0, 120)
    for u in range(0, 320, 20):
        (f1, l1), (f2, l2) = (CAMERA.pixel_to_ground(u, v) for v in (179, 100))
        _line(ax, forward, l1 + (forward - f1) * (l2 - l1) / (f2 - f1), color=DIM, lw=1.5)
    cols = np.arange(0, 320)
    for row in FOOT_ROWS:
        f, l = CAMERA.pixel_to_ground(cols, np.full(cols.shape, float(row)))
        _line(ax, f, l, color=TEXT, lw=2.5)
        ax.text(f[0] - 0.1, -7.1, f"{f[0]:.2f} m", rotation=90, ha="right", va="bottom")
    for row, text_xy, arrow_xy in ((179, (3.2, 6.3), None), (150, (8.4, 3.0), None), (95, (9.6, -2.8), None)):
        f, l = CAMERA.pixel_to_ground(np.array([150.0, 170.0]), np.full(2, float(row)))
        _line(ax, f, l, color=TEXT, lw=7)
        _pointer(ax, f"20 columns: {abs(l[0] - l[1]):.2f} m", (f.mean(), l.mean() + (0.3 if row == 150 else -0.3 if row == 179 else 0.5)), text_xy, va="center", ha="center", bbox=LABEL_BOX)
    ax.plot(apex, 0, marker=">", ms=15, mfc=TEXT, mec=BACKGROUND, mew=2, ls="none", zorder=7)
    ax.set(xlabel="metres ahead", ylabel="metres left")
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_yticks([-6, -3, 0, 3, 6])
    _save(fig, path)


# ---------------------------------------------------------------- 13 bands


def bands(path):
    frame = _view("straight", WORKED[1])[2]
    fig, (ax,) = _panels([(IMG_X, IMG_Y, None)], left=0.2, bottom=0.2)
    _show(ax, frame)
    _bands(ax)
    _tag(ax, 8, 150, "near band, rows 125-179", ha="left")
    _tag(ax, 8, 105, "far band, rows 95-115", ha="left")
    for row, text in ((174, "2.06 m"), (133, "4.44 m"), (106, "5.59 m"), (86, "11.45 m")):
        _tag(ax, 312, row, text, ha="right")
    assert [round(_row_distance(r), 2) for r in (179, 125, 115, 95)] == [2.06, 4.44, 5.59, 11.45]
    _save(fig, path)


# ---------------------------------------------------------------- 14 centroids


def centroids(path):
    frame = _view(*WORKED[:1], *WORKED[1:])[2]
    masks = paint_masks(frame)
    rows = slice(*NEAR_ROWS)
    facts = []
    for mask in masks:
        v, u = np.nonzero(mask[rows])
        facts.append((int(mask[rows].sum()), v.min() + NEAR_ROWS[0], v.max() + NEAR_ROWS[0], u.mean(), v.mean() + NEAR_ROWS[0]))
    assert [(n, a, b) for n, a, b, *_ in facts] == [(553, 137, 179), (543, 125, 177)] and round(facts[0][3], 2) == 83.02 and round(facts[1][4], 2) == 151.03
    fig, panels = _panels([((-0.5, 150.0), (180.5, 123.5), None), ((170.0, 319.5), (180.5, 123.5), None)], left=0.2, gap=0.5, bottom=0.2, top=0.2)
    for ax, (n, a, b, cu, cv), word in zip(panels, facts, ("yellow", "white")):
        ax.imshow(frame, interpolation="nearest", aspect="equal")
        ax.set_xlim(*ax.get_xlim()), ax.set_ylim(180.5, 123.5)
        _bare(ax)
        _cross(ax, cu, cv, 22)
    panels[0].set(xlim=(-0.5, 150.0), ylim=(180.5, 123.5))
    panels[1].set(xlim=(170.0, 319.5), ylim=(180.5, 123.5))
    (ny, ya, yb, yu, yv), (nw, wa, wb, wu, wv) = facts
    _tag(panels[0], 4, 131, f"{ny} yellow px, rows {ya}-{yb}", ha="left")
    _tag(panels[0], 146, 171, f"centroid ({yu:.2f}, {yv:.2f})", ha="right")
    _tag(panels[1], 174, 131, f"{nw} white px, rows {wa}-{wb}", ha="left")
    _tag(panels[1], 316, 171, f"centroid ({wu:.2f}, {wv:.2f})", ha="right")
    _save(fig, path)


# ---------------------------------------------------------------- 15 ground_points


def _true_lines(ax, road, state, s, ahead=9.6):
    for lateral in (0.0, EGO_LANE_CENTER, WHITE_LINE_LATERAL):
        f, l = to_vehicle(state, *road.offset(np.linspace(s - 1.0, s + ahead + 2.0, 400), lateral).T)
        _line(ax, f, l, color=DIM, lw=1.5)


def ground_points(path):
    road, state, frame = _view(*WORKED)
    masks = paint_masks(frame)
    fig, (ax,) = _panels([((-0.6, 9.8), (-3.6, 2.4), None)], bottom=0.9, top=0.2, left=1.25)
    _true_lines(ax, road, state, WORKED[1], 10.0)
    near, far = _lane_points(masks)
    positions = []
    for rows in (NEAR_ROWS, FAR_ROWS):
        for mask, paint in zip(masks, (YELLOW, WHITE)):
            f, l = CAMERA.pixel_to_ground(*_centroid(mask, rows))
            positions.append((f, l, paint))
            _paint_dot(ax, f, l, paint)
    for point in (near, far):
        _dot(ax, *point, COAL, 13)
    for (f, l, paint), dy in zip(positions, (0.55, -0.6, 0.55, -0.6)):
        ax.text(f, l + dy, f"({f:.3f}, {l:+.3f})".replace("-", "\u2212"), ha="center", va="center")
    for point, dx in ((near, 0.0), (far, 0.0)):
        ax.text(point[0] + dx, point[1] - 0.62, f"({point[0]:.3f}, {point[1]:+.3f})".replace("-", "\u2212"), ha="center", va="center", color=COAL)
    _top_down(ax)
    _save(fig, path)


# ---------------------------------------------------------------- 16 lane_estimate


def lane_estimate(path):
    frame = _view(*WORKED)[2]
    masks = paint_masks(frame)
    near, far = _lane_points(masks)
    est = estimate_lane(Observation(frame, TARGET_SPEED, 0.0))
    slope = (far[1] - near[1]) / (far[0] - near[0])
    c0 = near[1] - slope * near[0]
    assert round(-c0, 3) == 0.391 and round(est.heading_rad, 3) == 0.051
    fig, (ax,) = _panels([((-0.6, 9.4), (-1.9, 1.1), 3.6)], bottom=0.9, top=0.2, left=1.25)
    ahead = np.array([0.0, 8.6])
    _line(ax, ahead, c0 + slope * ahead, color=COAL, lw=3)
    _line(ax, [0, 4.6], [c0, c0], color=TEXT, lw=1.5, ls="--")
    _line(ax, [0, 0], [0, c0], color=COAL, lw=7)
    _arc(ax, (0, c0), 4.0, 0.0, math.atan(slope), color=COAL, lw=3)
    for point in (near, far):
        _dot(ax, *point, COAL, 13)
    _top_down(ax)
    for name, point in (("near", near), ("far", far)):
        _pointer(ax, f"{name}\n({point[0]:.3f}, {point[1]:+.3f})".replace("-", "\u2212"), (point[0], point[1] + 0.1), (point[0], 0.45), ha="center", va="center")
    _pointer(ax, f"offset {est.offset_m:+.3f} m", (0.0, c0 / 2), (0.3, -1.55), COAL, va="center")
    _pointer(ax, f"heading {est.heading_rad:+.3f} rad", (4.0, c0 + 4.0 * slope * 0.5), (4.6, -1.1), COAL, va="center")
    _save(fig, path)


# ---------------------------------------------------------------- shared by 17-24


def _bend_starts(road, ahead=12.0, n=40):
    """n start positions spread over the places where the lane turns one way for the whole window ahead."""
    def turn(start):
        return np.diff(np.unwrap(road.pose_at(np.arange(start - 1.0, start + ahead + 1))[2]))
    starts = [start for start in np.arange(1.0, road.length - ahead - 1) if np.all(turn(start) > 1e-6) or np.all(turn(start) < -1e-6)]
    return np.array(starts)[np.linspace(0, len(starts) - 1, n).astype(int)]


@lru_cache(maxsize=None)
def _centred(name):
    """The box's (offset, heading, turn direction) on frames of a car exactly on the lane centre, pointing along it."""
    road = make_scenario(name).road
    starts = np.linspace(0, 150, 40) if name == "straight" else _bend_starts(road)
    out = []
    for start in starts:
        est = estimate_lane(Observation(_view(name, float(start))[2], TARGET_SPEED, 0.0))
        turn = np.sign(np.diff(np.unwrap(road.pose_at(np.array([start, start + 1.0]))[2]))[0])
        out.append((est.offset_m, est.heading_rad, turn))
    return np.array(out)


def _bend_class(tel, road):
    """1 where the lane turns the same way from 1 m behind to 12 m ahead of the car, 0 where it is straight over that window, -1 between."""
    cls = []
    for x, y, yaw in zip(tel.x, tel.y, tel.yaw):
        along = road.ego_truth(x, y, yaw)[0]
        turn = np.diff(np.unwrap(road.pose_at(np.arange(along - 1.0, along + 13.0))[2]))
        cls.append(1 if np.all(np.abs(turn) > 1e-6) else (0 if np.all(np.abs(turn) < 1e-9) else -1))
    return np.array(cls)


@lru_cache(maxsize=None)
def _lap(name, perception=None):
    return run(Car(perception=perception), name)


def _perfect_lap(name):
    """The provided controller driven by the true offset and heading: only the perception box differs from `Car()`."""
    scenario = make_scenario(name)
    road = scenario.road
    state = BicycleState(*road.ego_pose(scenario.start_s), scenario.target_speed, 0.0)
    lateral, progress, previous = [], 0.0, scenario.start_s
    for _ in range(math.ceil(3 * scenario.route_length / (TARGET_SPEED * 0.05))):
        along, lat, head = road.ego_truth(state.x, state.y, state.yaw)
        step = along - previous
        progress += (step + road.length / 2) % road.length - road.length / 2 if road.closed else step
        previous = along
        lateral.append(lat)
        if abs(lat) > 1.8 or progress >= scenario.route_length:
            break
        state = sim_step(state, provided.steer_p(LaneEstimate(lat, head, True)), provided.hold_speed(state.speed))
    return np.abs(lateral).mean(), progress >= scenario.route_length


def _flipped_offset(obs):
    est = estimate_lane(obs)
    return LaneEstimate(-est.offset_m, est.heading_rad, est.valid)


# ---------------------------------------------------------------- 17 grader_scatter


def grader_scatter(path):
    heading, correct, forgot = [], [], []
    for pose, _, est, _ in pose_trials(estimate_lane):
        masks = paint_masks(CAMERA.render(pose.road, pose.state))
        near = lane_point(masks, NEAR_ROWS)
        heading.append(pose.heading)
        correct.append(est.offset_m - pose.offset)
        forgot.append(-near[1] - pose.offset)  # the lane-centre point's own sideways position, never carried back to the car
    heading, correct, forgot = map(np.array, (heading, correct, forgot))
    assert len(heading) == 40 and int((np.abs(forgot) > 0.25).sum()) == 18 and np.abs(correct).max() < 0.0225
    assert abs(np.polyfit(heading, forgot, 1)[0] - 2.85) < 0.01
    fig, (ax,) = _panels([((-0.17, 0.18), (-0.75, 0.75), 4.6)], left=1.3, bottom=1.0)
    ax.axhspan(-0.25, 0.25, color=TEXT, alpha=0.16)
    ax.axhline(0.25, color=TEXT, lw=1.5, ls="--")
    ax.axhline(-0.25, color=TEXT, lw=1.5, ls="--")
    _line(ax, heading, forgot, color=DIM, marker="o", ms=9, ls="none", mec=BACKGROUND)
    _line(ax, heading, correct, color=COAL, marker="o", ms=9, ls="none", mec=BACKGROUND)
    ax.set(xlabel="true heading (rad)", ylabel="offset error (m)")
    ax.set_xticks([-0.1, 0, 0.1])
    ax.set_yticks([-0.5, 0, 0.5])
    ax.text(-0.16, 0.62, "forgot to carry the line back:\n18 of 40 outside", va="center", color=TEXT)
    ax.text(-0.16, 0.11, "tolerance \u00b10.25 m", va="center", color=TEXT)
    _pointer(ax, "correct box", (0.07, 0.0), (0.05, -0.12), COAL, va="center")
    _save(fig, path)


# ---------------------------------------------------------------- 18 blind_frame


def blind_frame(path):
    frame = _view("straight", 20.0, 0.0, 0.6)[2]
    masks = paint_masks(frame)
    counts = [[int(m[slice(*rows)].sum()) for m in masks] for rows in (NEAR_ROWS, FAR_ROWS)]
    assert counts == [[696, 0], [117, 78]]
    fig, (ax,) = _panels([(IMG_X, IMG_Y, None)], left=0.2, bottom=0.2)
    _show(ax, frame)
    _bands(ax)
    _tag(ax, 160, 150, f"near band: {counts[0][0]} yellow, {counts[0][1]} white")
    _tag(ax, 8, 105, f"far band: {counts[1][0]} yellow, {counts[1][1]} white", ha="left")
    _save(fig, path)


# ---------------------------------------------------------------- 19 centred_errors


def centred_errors(path):
    straight, gentle, curvy = _centred("straight"), _centred("gentle"), _centred("curvy")
    parts = {"straight": straight, "gentle, left bends": gentle[gentle[:, 2] > 0], "curvy, left bends": curvy[curvy[:, 2] > 0], "curvy, right bends": curvy[curvy[:, 2] < 0]}
    assert [len(v) for v in parts.values()] == [40, 40, 19, 21]
    allp = np.vstack(list(parts.values()))
    assert np.abs(allp[:, 0]).max() < 0.9 and np.abs(allp[:, 1]).max() < 0.42
    fig, (ax,) = _panels([((-0.95, 0.95), (-0.45, 0.45), 4.8)], left=1.3, bottom=1.0)
    ax.add_patch(Rectangle((-0.25, -0.05), 0.5, 0.1, fc=TEXT, alpha=0.16, ec=CLEAR))
    ax.add_patch(Rectangle((-0.25, -0.05), 0.5, 0.1, fill=False, ec=TEXT, lw=1.5, ls="--"))
    for name, pts in parts.items():
        _line(ax, pts[:, 0], pts[:, 1], color=COAL, marker="o", ms=8, ls="none", mec=BACKGROUND)
    ax.set(xlabel="box's offset (m)", ylabel="box's heading (rad)")
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_yticks([-0.3, 0, 0.3])
    _pointer(ax, "straight", (0.0, 0.0), (-0.55, -0.12), va="center", ha="center")
    _pointer(ax, "gentle,\nleft bends", (0.13, -0.065), (0.42, -0.015), va="center")
    _pointer(ax, "curvy,\nleft bends", (0.5, -0.25), (0.0, -0.33), va="center", ha="center")
    _pointer(ax, "curvy,\nright bends", (-0.65, 0.31), (-0.1, 0.33), va="center")
    _pointer(ax, "Exercise A\ntolerance", (-0.25, 0.05), (-0.8, 0.14), va="center")
    _save(fig, path)


# ---------------------------------------------------------------- 20 chord_bias


def _lane_arc_y(x, radius=21.8):
    return radius - np.sqrt(radius ** 2 - np.asarray(x, dtype=float) ** 2)


def chord_bias(path):
    radius, xs = 21.8, np.array([2.73, 7.36])
    ys = _lane_arc_y(xs, radius)
    slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
    c0 = ys[0] - slope * xs[0]
    assert round(-c0, 2) == 0.48 and round(math.atan(slope), 2) == 0.23
    fig, (ax,) = _panels([((-0.6, 9.4), (-1.9, 2.9), None)], left=0.2, bottom=0.2, top=0.2)
    _bare(ax)
    road = np.linspace(0, 9.2, 100)
    _line(ax, road, _lane_arc_y(road, radius), color=DIM, lw=5)
    ahead = np.array([0.0, 8.6])
    _line(ax, ahead, c0 + slope * ahead, color=COAL, lw=3)
    _line(ax, [0, 4.8], [c0, c0], color=TEXT, lw=1.5, ls="--")
    _line(ax, [0, 0], [0, c0], color=COAL, lw=7)
    _arc(ax, (0, c0), 3.6, 0.0, math.atan(slope), color=COAL, lw=3)
    for x, y in zip(xs, ys):
        _dot(ax, x, y, COAL, 13)
    ax.plot(0, 0, marker=">", ms=15, mfc=TEXT, mec=BACKGROUND, mew=2, ls="none", zorder=7)
    _pointer(ax, "R = 21.8 m", (8.9, _lane_arc_y(8.9, radius)), (6.5, 2.55), va="center", ha="center")
    _pointer(ax, "2.73 m", (xs[0], ys[0] + 0.12), (1.6, 1.0), va="center", ha="center")
    _pointer(ax, "7.36 m", (xs[1] - 0.05, ys[1] - 0.15), (6.2, 0.1), va="center", ha="center")
    _pointer(ax, "0.48 m to the right", (0.0, c0 / 2), (1.1, -1.35), COAL, va="center")
    _pointer(ax, "0.23 rad", (3.55, c0 + 3.6 * math.sin(math.atan(slope) / 2) * 0.9), (5.3, -1.0), COAL, va="center")
    _save(fig, path)


# ---------------------------------------------------------------- 21 chord_prediction


def chord_prediction(path):
    rows = (("gentle, left bends", 0.125, 0.129), ("curvy, left bends", 0.461, 0.504), ("curvy, right bends", 0.566, 0.650))
    measured = {"gentle, left bends": _centred("gentle")[_centred("gentle")[:, 2] > 0][:, 0], "curvy, left bends": _centred("curvy")[_centred("curvy")[:, 2] > 0][:, 0],
                "curvy, right bends": -_centred("curvy")[_centred("curvy")[:, 2] < 0][:, 0]}
    for name, _, value in rows:
        assert round(float(np.median(measured[name])), 3) == value, (name, np.median(measured[name]))
    fig, (ax,) = _panels([((0, 1.9), (8.2, -0.4), 4.1)], left=0.2, bottom=0.2)
    _bare(ax)
    for k, (name, predicted, value) in enumerate(rows):
        y = k * 3.0
        ax.text(0.02, y - 0.1, name, va="center")
        ax.barh(y + 0.75, predicted, color=DIM, edgecolor=TEXT, height=0.7)
        ax.barh(y + 1.65, value, color=COAL, edgecolor=TEXT, height=0.7)
        ax.text(predicted + 0.03, y + 0.75, f"predicted {predicted:.3f} m", va="center")
        ax.text(value + 0.03, y + 1.65, f"measured {value:.3f} m", va="center")
    _save(fig, path)


# ---------------------------------------------------------------- 22 gentle_trace


def gentle_trace(path):
    tel = _lap("gentle")
    road = make_scenario("gentle").road
    bend = _bend_class(tel, road) == 1
    x = tel.progress_m
    fig, (ax,) = _panels([((0, x.max()), (-0.75, 0.95), 4.0)], left=1.3, bottom=1.0)
    edges = np.flatnonzero(np.diff(np.r_[0, bend.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(x[a], x[b - 1], color=TEXT, alpha=0.16, lw=0)
    ax.plot(x, tel.lat, color=BLUE, lw=3)
    ax.plot(x, tel.est_offset, color=COAL, lw=3)
    ax.set(xlabel="distance along the road (m)", ylabel="lateral error (m)")
    ax.set_xticks([0, 200, 400, 600])
    ax.set_yticks([-0.5, 0, 0.5])
    first = np.flatnonzero(bend)[0]
    ax.text(x[first] + 10, -0.3, "bend", va="center")
    mid = x[edges[0]:edges[1]].mean()  # inside the first bend, clear of both traces
    ax.text(mid, 0.82, "box's offset \u2193", color=COAL, va="center", ha="center")
    ax.text(mid, 0.40, "true e \u2191", color=BLUE, va="center", ha="center")
    _save(fig, path)


# ---------------------------------------------------------------- 23 like_for_like


def like_for_like(path):
    gentle, curvy = _lap("gentle"), _lap("curvy")
    perfect_gentle, ok_gentle = _perfect_lap("gentle")
    perfect_curvy, ok_curvy = _perfect_lap("curvy")
    values = (np.abs(gentle.lat).mean(), perfect_gentle, perfect_curvy)
    assert [round(float(v), 3) for v in values] == [0.376, 0.113, 0.348] and ok_gentle and ok_curvy and not curvy.completed
    assert round(float(curvy.progress_m[-1]), 1) == 48.3 and round(curvy.route_length, 1) == 218.5
    fig, (ax,) = _panels([((0, 1.3), (6.3, -0.4), 4.4)], left=0.2, bottom=0.2)
    _bare(ax)
    ax.text(0.01, 0.0, "gentle", va="center")
    ax.barh(1.0, values[0], color=COAL, edgecolor=TEXT, height=0.8)
    ax.barh(2.0, values[1], color=DIM, edgecolor=TEXT, height=0.8)
    ax.text(values[0] + 0.02, 1.0, f"two-band box {values[0]:.3f} m", va="center")
    ax.text(values[1] + 0.02, 2.0, f"perfect box {values[1]:.3f} m", va="center")
    ax.text(0.01, 3.4, "curvy", va="center")
    ax.barh(4.4, 1.3, fc=CLEAR, ec=COAL, hatch="xx", lw=2, height=0.8)
    ax.barh(5.4, values[2], color=DIM, edgecolor=TEXT, height=0.8)
    _tag(ax, 0.65, 4.4, f"left the lane at {curvy.progress_m[-1]:.1f} of {curvy.route_length:.1f} m", color=COAL)
    ax.text(values[2] + 0.02, 5.4, f"perfect box {values[2]:.3f} m", va="center")
    _save(fig, path)


# ---------------------------------------------------------------- 24 flipped_sign


def flipped_sign(path):
    flipped, correct = _lap("gentle", _flipped_offset), _lap("gentle")
    keep = correct.t <= 5.0
    assert round(float(flipped.t[-1]), 2) == 5.0 and round(float(flipped.progress_m[-1]), 1) == 39.9 and np.abs(correct.lat[keep]).max() < 0.0025
    at = {sec: float(np.interp(sec, flipped.t, flipped.lat)) for sec in (2, 3, 4, 5)}
    assert [round(v, 2) for v in at.values()] == [-0.04, -0.15, -0.54, -1.81]
    fig, (ax,) = _panels([((0, 5.4), (-2.1, 0.4), 4.6)], left=1.3, bottom=1.0)
    ax.plot(correct.t[keep], correct.lat[keep], color=DIM, lw=3)
    ax.plot(flipped.t, flipped.lat, color=BLUE, lw=3)
    for sec, value in at.items():
        _dot(ax, sec, value, BLUE, 12)
    ax.set(xlabel="time (s)", ylabel="true lateral error e (m)")
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_yticks([-2, -1, 0])
    ax.text(0.2, 0.2, "correct box: within 0.002 m", color=TEXT, va="center")
    ax.text(0.15, -1.4, "flipped offset", color=BLUE, va="center")
    _pointer(ax, "\u22120.04 m\nat 2 s", (2.0, at[2] - 0.04), (0.5, -0.5), BLUE, va="center")
    _pointer(ax, "\u22120.15 m\nat 3 s", (3.0, at[3] - 0.05), (1.9, -0.9), BLUE, va="center")
    _pointer(ax, "\u22120.54 m\nat 4 s", (4.0, at[4] - 0.06), (3.0, -1.25), BLUE, va="center")
    _pointer(ax, "\u22121.81 m at 5 s:\nacross the white line,\n39.9 m", (4.94, at[5]), (1.3, -1.8), BLUE, va="center")
    _save(fig, path)


# ---------------------------------------------------------------- 25 next_lesson


def next_lesson(path):
    fig, (ax,) = _panels([((0, 10), (1.6, 11.5), None)], left=0.2, bottom=0.2, top=0.2)
    _bare(ax)
    ax.axis("off")
    boxes = (("camera", 10.3, "solid"), ("perception box", 7.8, "lit"), ("controller box", 5.3, "closed"), ("car", 2.8, "solid"))
    for text, y, kind in boxes:
        lit, closed = kind == "lit", kind == "closed"
        ax.add_patch(FancyBboxPatch((1.0, y - 0.75), 5.0, 1.5, boxstyle="round,pad=0.1", fc=COAL if lit else PANEL, ec=COAL if lit else (DIM if closed else TEXT), lw=3, ls="--" if closed else "-"))
        ax.text(3.5, y, text, ha="center", va="center", color=BACKGROUND if lit else (DIM if closed else TEXT))
    ax.text(5.1, 5.3, "?", ha="center", va="center", color=DIM)
    for y in (10.3, 7.8, 5.3):
        ax.annotate("", xy=(3.5, y - 1.75 + 0.2), xytext=(3.5, y - 0.95), arrowprops={"arrowstyle": "-|>", "color": TEXT, "lw": 3, "mutation_scale": 22})
    ax.text(3.9, 6.55, "offset, heading", color=COAL, va="center")
    _line(ax, [6.1, 8.6, 8.6], [2.8, 2.8, 10.3], color=TEXT, lw=3)
    ax.annotate("", xy=(6.15, 10.3), xytext=(8.6, 10.3), arrowprops={"arrowstyle": "-|>", "color": TEXT, "lw": 3, "mutation_scale": 22})
    _save(fig, path)


# ---------------------------------------------------------------- 26 curve_box


def curve_box(path):
    road = make_scenario("curvy").road
    centred = _centred("curvy")
    start = _bend_starts(road)[[k for k, row in enumerate(centred) if row[2] > 0][len(centred[centred[:, 2] > 0]) // 2]]
    _, state, frame = _view("curvy", float(start))
    masks = paint_masks(frame)
    points = np.array([lane_point(masks, rows) for rows in STRETCH_BANDS])
    quad = np.polyfit(points[:, 0], points[:, 1], 2)
    chord_near, chord_far = lane_point(masks, NEAR_ROWS), lane_point(masks, FAR_ROWS)
    chord_slope = (chord_far[1] - chord_near[1]) / (chord_far[0] - chord_near[0])
    chord_c0 = chord_near[1] - chord_slope * chord_near[0]
    lane_f, lane_l = _ground_line(road, state, float(start), EGO_LANE_CENTER, 10.0)
    ahead = np.linspace(0, 8.8, 80)
    assert abs(round(quad[2], 1)) == 0.1 and round(chord_c0, 1) == -0.6
    fig, (ax,) = _panels([((-0.6, 9.4), (-1.9, 2.9), None)], left=0.2, bottom=0.2, top=0.2)
    _bare(ax)
    _line(ax, lane_f, lane_l, color=DIM, lw=5)
    _line(ax, ahead, chord_c0 + chord_slope * ahead, color=TEXT, lw=2, ls="--")
    _line(ax, ahead, np.polyval(quad, ahead), color=COAL, lw=3)
    for f, l in points:
        _dot(ax, f, l, COAL, 13)
    ax.plot(0, 0, marker=">", ms=15, mfc=TEXT, mec=BACKGROUND, mew=2, ls="none", zorder=7)
    _pointer(ax, "R = 21.8 m", (8.9, float(np.interp(8.9, lane_f, lane_l))), (6.5, 2.55), va="center", ha="center")
    _pointer(ax, "parabola", (points[1][0], points[1][1] - 0.1), (4.2, -0.6), COAL, va="center", ha="center")
    _pointer(ax, "chord", (6.6, chord_c0 + chord_slope * 6.6 - 0.05), (7.2, -1.2), va="center", ha="center")
    _save(fig, path)


# ---------------------------------------------------------------- 27 far_band_choices


def far_band_choices(path):
    frame = _view("straight", 20.0)[2]
    fig, (ax,) = _panels([(IMG_X, IMG_Y, None)], left=0.2, bottom=0.2)
    _show(ax, frame)
    ax.axhspan(NEAR_ROWS[0] - 0.5, NEAR_ROWS[1] - 0.5, color=TEXT, alpha=0.22)
    choices = []
    for metres, where in ((7.19, (240, 160)), (11.45, (78, 126)), (20.0, (72, 56))):
        u, v = CAMERA.ground_to_pixel(metres, 0.0)
        width = _metres_per_column(float(v))
        choices.append((metres, float(v), width))
        _dot(ax, float(u), float(v), COAL, 10)
        _pointer(ax, f"{metres:g} m ahead, row {v:.1f}\n{width:.4f} m per column", (float(u), float(v)), where, va="center", ha="center", bbox=LABEL_BOX)
    assert [(m, round(v, 1), round(w, 4)) for m, v, w in choices] == [(7.19, 106.4, 0.0455), (11.45, 95.0, 0.0721), (20.0, 86.7, 0.1253)]
    _save(fig, path)


def make(out_dir):
    with matplotlib.rc_context(RC):
        for name, fn in FIGURES:
            fn(out_dir / f"{name}.png")


FIGURES = [(fn.__name__, fn) for fn in (box_result, radians, offset_heading, slide_shift, perspective_grid, tan_curve, pinhole_screen, ray_side_view, row_150_ray,
    rows_to_distance, column_width, ground_footprint, bands, centroids, ground_points, lane_estimate,
    grader_scatter, blind_frame, centred_errors, chord_bias, chord_prediction, gentle_trace, like_for_like, flipped_sign, next_lesson, curve_box, far_band_choices)]
