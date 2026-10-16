"""Figures for lecture 0.1.1 Meet your car.  `python scripts/make_figures.py course/0.1.1-meet-your-car`."""
from collections import Counter

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path

from zero2fsd.car import Car, provided, run
from zero2fsd.car.types import Action
from zero2fsd.palette import CHROME, QUANTITY, SCENE, shades
from zero2fsd.score import OFF_LANE_LAT
from zero2fsd.sim import BicycleState, Camera, make_scenario, render_topdown
from zero2fsd.sim.camera import PALETTE, YELLOW
from zero2fsd.sim.scenarios import TARGET_SPEED
from zero2fsd.sim.world import LANE_WIDTH

ZOOM_X, ZOOM_Y = (85.0, 125.0), (-6.0, 2.0)  # m: the first-bend zoom inset of yardstick
ZOOM_S = (70.0, 110.0)  # m along `gentle` shown from above in car_and_view
VIEW_S = 80.0  # m along `gentle`: 20 m before the first bend, so the bend is in view
TRAP_CARS = (("2×", 0.7), ("4×", 1.4))  # cars that push back toward the lane centre 2 and 4 times as hard as the provided car (gains are unit 0.1.4)
ZOOM_INSET = "zoom-inset"  # gid of an axes that deliberately shows only part of its data (the clipping gate skips it)
PROGRESS_LABEL = "distance along the road (m)"
AVG_X = 82.0  # m along curvy: where signed_mean_trap points at each average line (a stretch both traces leave clear)
ROAD_ORDER = ("straight", "gentle", "curvy")  # test_roads layout, top first
FIG_W, MARGIN_L, MARGIN_R, MARGIN_B, MARGIN_T, GAP_Y, LABEL_ROOM = 5.5, 1.0, 0.15, 0.9, 0.5, 1.35, 0.6  # inches
ROAD_PAD, ARROW_LEN, LABEL_RISE = 10.0, 20.0, 7.0  # metres
HALF = LANE_WIDTH / 2  # m from lane centre to a painted line
FIVE = np.array([0.1, -0.3, 0.2, -0.4, 0.4])  # the lecture's worked example
PATCH = 6  # pixels per side of the magnified patch in next_lesson
TEXT, PANEL, BACKGROUND = CHROME["text"], CHROME["panel"], CHROME["background"]
LABEL_BOX = {"facecolor": PANEL, "linewidth": 0, "pad": 2}  # keeps a painted line from running through a label
W = 5.5  # inches: every figure is this wide, and its smallest text is 15 pt (gate G8: at least 2.7 x the width in pt)
PHONE_TEXT = {"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 15, "ytick.labelsize": 15}


def _canvas(height):
    """A W-inch-wide figure whose one axes fills it, with no axes drawn and data units of one inch: (fig, ax)."""
    fig = Figure(figsize=(W, height))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(0, height))
    ax.set_axis_off()
    return fig, ax


def _minus(value, digits=1, sign=True):
    """A number as text with a true minus sign: +0.4, −0.3."""
    return f"{value:{'+' if sign else ''}.{digits}f}".replace("-", "−")


def _camera_frame():
    road = make_scenario("gentle").road
    return Camera().render(road, BicycleState(*road.ego_pose(VIEW_S), TARGET_SPEED, 0.0))


def _bends(road):
    """(start, end) arc lengths of each stretch of constant-sign curvature, from the road's own heading."""
    s = np.arange(0.0, road.length, 0.1)
    curvature = np.gradient(np.unwrap(road.pose_at(s)[2]), s)
    sign = np.where(np.abs(curvature) > 1e-3, np.sign(curvature), 0.0)
    edges = np.flatnonzero(np.diff(sign) != 0) + 1
    return [(s[a], s[b - 1] + 0.1) for a, b in zip([0, *edges], [*edges, len(s)]) if sign[a] != 0]


def _mark_bends(ax, road):
    """Faint bands over the road's bends, each named at the top of the axes."""
    for i, (a, b) in enumerate(_bends(road), 1):
        ax.axvspan(a, b, facecolor=TEXT, alpha=0.12, linewidth=0)
        if ax.get_xlim()[0] < (a + b) / 2 < ax.get_xlim()[1]:  # a bend past the end of the run has no room for its name
            ax.text((a + b) / 2, 0.97, f"bend {i}", transform=ax.get_xaxis_transform(), ha="center", va="top")


def _brace(ax, x0, x1, y, height, **kwargs):
    """A curly brace from x0 to x1: its two feet stand on y, its tip points up to y + height."""
    xm, d, mid = (x0 + x1) / 2, (x1 - x0) / 8, y + height / 2
    verts = [(x0, y), (x0, mid), (x0 + d, mid), (xm - d, mid), (xm, mid), (xm, y + height),
             (xm, mid), (xm + d, mid), (x1 - d, mid), (x1, mid), (x1, y)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3]
    ax.add_patch(PathPatch(Path(verts, codes), fill=False, **kwargs))


def _lane_view(ax, ylim, yellow_text, white_text, label_y):
    """The lane seen from above, travelling up the page: left of centre is on the left, so e grows to the left."""
    ax.set(xlim=(3.7, -3.7), ylim=ylim, xlabel="lateral error e (m):\nleft of lane centre is +")
    ax.set_aspect("equal")
    ax.set_yticks([])
    ax.axvspan(-HALF, HALF, facecolor=SCENE["road"], alpha=0.5, linewidth=0)
    ax.axvline(HALF, color=SCENE["yellow"], linestyle=(0, (6, 3)), linewidth=3)
    ax.axvline(-HALF, color=SCENE["white"], linewidth=3)
    ax.axvline(0.0, color=TEXT, linestyle=":", linewidth=1)
    ax.text(HALF, label_y, yellow_text, ha="center", va="bottom", bbox=LABEL_BOX)
    ax.text(-HALF, label_y, white_text, ha="center", va="bottom", bbox=LABEL_BOX)
    ax.annotate("", xy=(-2.8, 2.2), xytext=(-2.8, 0.4), arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.text(-2.8, 2.45, "direction\nof travel", ha="center", va="bottom")


def _car_and_view(path):
    frame = _camera_frame()
    road = make_scenario("gentle").road
    tel = run(Car(), "gentle")
    fig = Figure(figsize=(W, 8.2))
    cam, top = fig.subplots(2, 1, gridspec_kw={"height_ratios": [1.25, 1]})
    cam.imshow(frame)
    cam.set_title(f"What the camera sees,\n{VIEW_S:.0f} m into 'gentle'")
    cam.set_xlabel("pixel column")
    cam.set_ylabel("pixel row")
    render_topdown(road, ax=top)
    top.set_gid(ZOOM_INSET)  # the whole loop is drawn, the limits below show 40 m of it
    near = (tel.progress_m >= ZOOM_S[0]) & (tel.progress_m <= ZOOM_S[1])
    top.plot(tel.x[near], tel.y[near], color=QUANTITY["e_off"], linewidth=2.5)
    here = int(np.argmin(np.abs(tel.progress_m - VIEW_S)))
    top.plot(tel.x[here], tel.y[here], "o", color=TEXT, markersize=9)
    top.annotate("car at 80 m:\nthe camera's view", xy=(tel.x[here], tel.y[here]), xytext=(tel.x[here] + 4, tel.y[here] + 8.0),
                 ha="center", va="bottom", arrowprops={"arrowstyle": "->", "color": TEXT, "lw": 1.5})
    top.text(ZOOM_S[1] - 1, tel.y[near].min() - 3.5, "path of the car", color=QUANTITY["e_off"], ha="right", va="top")
    x0, y0 = road.offset(ZOOM_S[0], 0.0)
    top.set(xlim=(x0, x0 + ZOOM_S[1] - ZOOM_S[0]), ylim=(-12.0, 14.0))
    top.set_title(f"From above, {ZOOM_S[0]:.0f} to {ZOOM_S[1]:.0f} m\ninto 'gentle'")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def _paint_box(road):
    """Corners ((x0, y0), (x1, y1)) of a box around the road's paint, with room for the markers."""
    s = np.linspace(0.0, road.length, max(2, int(road.length)))
    paint = np.vstack([road.offset(s, -LANE_WIDTH), road.offset(s, LANE_WIDTH)])
    return paint.min(axis=0) - ROAD_PAD, paint.max(axis=0) + ROAD_PAD


def _mark_route(ax, scenario, direction_label=None):
    """Start dot and direction-of-travel arrow in the ego lane, named on the picture; an open road also gets an end-of-route square."""
    road = scenario.road
    x, y, yaw = road.ego_pose(scenario.start_s)
    ax.plot(x, y, "o", color=TEXT, zorder=3)
    ax.annotate("", xy=(x + ARROW_LEN * np.cos(yaw), y + ARROW_LEN * np.sin(yaw)), xytext=(x, y),
                arrowprops={"arrowstyle": "-|>", "color": TEXT, "lw": 2})
    start_text = "start (arrow:\ndirection of travel)" if direction_label else "start"
    start_text = f"{start_text}, and the\nend of the loop" if road.closed else start_text
    if direction_label:  # the first label has two lines: it goes under the road so it cannot meet the end label
        ax.text(x, y - LABEL_RISE, start_text, ha="left", va="top")
    else:
        ax.text(x, y + LABEL_RISE, start_text, ha="left", va="bottom")
    if not road.closed:
        ex, ey = road.ego_pose(scenario.start_s + scenario.route_length)[:2]
        ax.plot(ex, ey, "s", color=TEXT, zorder=3)
        ax.text(ex, ey + LABEL_RISE, "end of route", ha="right", va="bottom")


def _test_roads(path):
    scenarios = {name: make_scenario(name) for name in ROAD_ORDER}
    boxes = {name: _paint_box(sc.road) for name, sc in scenarios.items()}
    # One scale for every panel: the widest road sets how many metres fit in an inch.
    m_per_in = max(hi[0] - lo[0] for lo, hi in boxes.values()) / (FIG_W - MARGIN_L - MARGIN_R)
    room = {name: LABEL_ROOM * m_per_in for name in ROAD_ORDER}  # metres of room for the labels above (and, on straight, under) the road
    boxes = {name: ((lo[0], lo[1] - (room[name] if name == "straight" else 0.0)), (hi[0], hi[1] + room[name])) for name, (lo, hi) in boxes.items()}
    heights = {name: (hi[1] - lo[1]) / m_per_in for name, (lo, hi) in boxes.items()}
    height = MARGIN_B + sum(heights.values()) + GAP_Y * (len(ROAD_ORDER) - 1) + MARGIN_T
    fig = Figure(figsize=(FIG_W, height))
    y_in = MARGIN_B
    for name in reversed(ROAD_ORDER):
        (x0, y0), (x1, y1) = boxes[name]
        ax = fig.add_axes([MARGIN_L / FIG_W, y_in / height, (x1 - x0) / m_per_in / FIG_W, heights[name] / height], xlim=(x0, x1), ylim=(y0, y1))
        render_topdown(scenarios[name].road, ax=ax)
        _mark_route(ax, scenarios[name], direction_label=name == "straight")
        ax.set_title(f"'{name}': {scenarios[name].route_length:.1f} m")
        y_in += heights[name] + GAP_Y
    fig.savefig(path, dpi=200)


def _sign_convention(path):
    fig = Figure(figsize=(W, 9.2))
    ax = fig.subplots()
    _lane_view(ax, (0.0, 11.0), f"yellow line\n(left edge of\nyour lane)\ne = +{HALF:.1f} m",
               f"white line\n(right edge of\nyour lane)\ne = {_minus(-HALF)} m", label_y=9.2)
    ax.text(0.0, 7.9, "lane centre,\ne = 0", ha="center", va="bottom", bbox=LABEL_BOX)
    for (y, e), color in zip(((3.8, 0.5), (7.6, -0.3)), shades(QUANTITY["e_off"], 2)):
        ax.add_patch(Rectangle((e - 0.8, y - 2.7), 1.6, 2.7, facecolor=color, alpha=0.35, edgecolor=color))  # 2.7 m: the wheelbase
        ax.annotate("", xy=(e, y), xytext=(0.0, y), arrowprops={"arrowstyle": "->", "color": color, "lw": 2})
        ax.plot(e, y, "o", color=color)
        ax.text(e, y - 2.9, f"e = {_minus(e)} m", color=color, ha="center", va="top", bbox=LABEL_BOX)
    ax.set_title("Lateral error e, measured at\nthe front axle (dot):\nleft of centre is positive")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def _report_card(path):
    tel = run(Car(), "gentle")
    size = np.abs(tel.lat)
    lines = (  # (value, label, x of the label's left edge, height of its row): staggered so no leader crosses another label
        (size.max(), "max |e| {:.2f} m", 20.0, 1.5),
        (np.sqrt((tel.lat ** 2).mean()), "RMS {:.2f} m", 190.0, 1.25),
        (size.mean(), "mean |e| {:.2f} m", 360.0, 1.0),
    )
    fig = Figure(figsize=(W, 5.0))
    ax = fig.subplots()
    ax.plot(tel.progress_m, size, color=QUANTITY["e_off"], linewidth=2, zorder=5)
    for (value, text, x, y), color in zip(lines, shades(QUANTITY["e_off"], 4)[1:][::-1]):
        ax.axhline(value, color=color, linewidth=1.5, linestyle=(0, (6, 3)), zorder=2)
        ax.annotate(text.format(value), xy=(x + 10, value), xytext=(x, y), color=color, ha="left", va="center",
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5, "shrinkA": 2, "shrinkB": 0})
    ax.axhline(HALF, color=TEXT, linewidth=2, zorder=2)
    ax.text(tel.progress_m[-1] / 2, HALF, "painted lines, 1.8 m", color=TEXT, ha="center", va="bottom")
    ax.set(ylim=(0.0, 2.0), xlim=(0.0, tel.progress_m[-1]), xlabel=PROGRESS_LABEL, ylabel="distance from\nlane centre |e| (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def _loop_diagram(path):
    fig, ax = _canvas(7.7)
    w, h, cx = 3.5, 0.9, 2.6
    stack = (("camera", 6.3, TEXT), ("perception", 4.7, QUANTITY["perception"]), ("controller", 3.1, TEXT), ("car moves", 1.5, TEXT))
    for name, y, edge in stack:
        ax.add_patch(FancyBboxPatch((cx - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    facecolor=BACKGROUND, edgecolor=edge, linewidth=2.5))
        closed = name in ("perception", "controller")
        ax.text(cx, y + (0.15 if closed else 0.0), name, ha="center", va="center")
        if closed:
            ax.text(cx, y - 0.2, "stays closed in this lesson", ha="center", va="center")

    def arrow(start, end, color):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.5, "shrinkA": 0, "shrinkB": 0})

    gap = (h / 2 + 0.05, 1.6 - h / 2 - 0.05)  # an arrow runs from just under one box to just above the next
    for (label, color), (name, y, _) in zip((("picture", TEXT), ("where the lane is", QUANTITY["perception"]), ("steering angle", QUANTITY["delta"])), stack):
        arrow((cx, y - gap[0]), (cx, y - gap[1]), color)
        ax.text(cx + 0.2, y - 0.8, label, color=color, ha="left", va="center")
    ax.plot([cx - w / 2, 0.4, 0.4], [1.5, 1.5, 6.3], color=TEXT, linewidth=2.5, solid_capstyle="butt")  # the return path, up the left side
    arrow((0.4, 6.3), (cx - w / 2, 6.3), TEXT)
    ax.text(0.68, 3.9, "the new view", rotation=90, ha="center", va="center")
    ax.plot(1.5, 1.05, "o", color=QUANTITY["e_off"], markersize=9)  # the tap: a measurement taken off the car
    ax.plot([1.5, 1.5, 1.8], [1.05, 0.5, 0.5], color=QUANTITY["e_off"], linewidth=2.5)
    ax.text(1.95, 0.5, "you measure here:\nlateral error e", color=QUANTITY["e_off"], ha="left", va="center")
    ax.text(cx, 7.3, "the loop runs every 0.05 s", ha="center", va="center")
    fig.savefig(path, dpi=200)


def _gentle_trace(path):
    tel = run(Car(), "gentle")
    fig = Figure(figsize=(W, 4.6))
    ax = fig.subplots()
    ax.set_xlim(0.0, tel.progress_m[-1])
    _mark_bends(ax, make_scenario("gentle").road)
    ax.plot(tel.progress_m, tel.lat, color=QUANTITY["e_off"], linewidth=1.5)
    ax.axhline(0.0, color=TEXT, linestyle=":", linewidth=1)
    ax.text(0.01, 0.0, "lane centre", transform=ax.get_yaxis_transform(), ha="left", va="top")
    ax.set(ylim=(-0.12, 0.75), xlabel=PROGRESS_LABEL, ylabel="lateral error e (m)\nleft of centre is +")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def _number_line(ax, limit, side_y):
    """A horizontal e axis across the lane, left of centre on the left like the lane figure; returns nothing."""
    ax.set_axis_off()
    ax.set(xlim=(limit, -limit))
    ax.plot([limit, -limit], [0, 0], color=TEXT, linewidth=1.5)
    ticks = np.arange(-0.4, 0.41, 0.1)
    ax.plot(ticks, np.zeros_like(ticks), "|", color=TEXT, markersize=10)
    ax.plot([0.0, 0.0], [0.0, ax.get_ylim()[1] - 0.12], color=TEXT, linestyle=":", linewidth=1)
    ax.text(0.0, ax.get_ylim()[1], "lane centre, e = 0", ha="center", va="top")
    ax.text(limit, side_y, "left of centre (+)", ha="left", va="center")
    ax.text(-limit, side_y, "right of centre (−)", ha="right", va="center")


def _cancel_numberline(path):
    fig = Figure(figsize=(W, 4.6))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96], ylim=(-1.05, 1.45))
    _number_line(ax, 0.55, -0.9)
    rank = {v: i for i, v in enumerate(sorted(FIVE))}
    for value in FIVE:  # neighbouring labels are closer than a label is wide, so every other one sits higher
        ax.plot(value, 0, "o", color=QUANTITY["e_off"], markersize=15)
        ax.text(value, 0.14 + 0.2 * (rank[value] % 2), _minus(value), color=QUANTITY["e_off"], ha="center", va="bottom")
    for lo, hi, text in ((0.07, 0.43, "+0.7"), (-0.43, -0.27, "−0.7")):
        _brace(ax, lo, hi, 0.62, 0.15, edgecolor=TEXT, linewidth=1.8)
        ax.text((lo + hi) / 2, 0.82, text, ha="center", va="bottom")
    ax.plot(0.0, -0.17, "^", color=TEXT, markersize=13)
    ax.text(0.0, -0.3, "average = 0", ha="center", va="top")
    fig.savefig(path, dpi=200)


def _signed_mean_trap(path):
    fig = Figure(figsize=(W, 7.4))
    for ax, (name, k_off) in zip(fig.subplots(2, 1, sharex=True, sharey=True), TRAP_CARS):
        tel = run(Car(controller=lambda est, obs: provided.steer_p(est, k_off, 2.7)), "curvy")
        signed = tel.lat.mean()
        ax.plot(tel.progress_m, tel.lat, color=QUANTITY["e_off"], linewidth=1.5)
        ax.axhline(0.0, color=TEXT, linestyle=":", linewidth=1)
        ax.axhline(signed, color=shades(QUANTITY["e_off"], 3)[2], linestyle="--", linewidth=2)
        shade = shades(QUANTITY["e_off"], 3)[2]
        ax.annotate(f"average {_minus(signed, 2)} m", xy=(AVG_X, signed), xytext=(AVG_X, 0.86), color=shade, ha="center", va="bottom",
                    arrowprops={"arrowstyle": "->", "color": shade, "lw": 1.5, "shrinkA": 2, "shrinkB": 2})
        ax.set(title=f"{name} car on 'curvy'", xlim=(0.0, tel.progress_m[-1]), ylim=(-1.0, 1.15), ylabel="lateral error e (m)\nleft of centre is +")
    fig.axes[1].set_xlabel(PROGRESS_LABEL)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def _fold_numberline(path):
    fig = Figure(figsize=(W, 5.0))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96], ylim=(-1.75, 1.45))
    _number_line(ax, 0.55, -1.55)
    solid, faded = QUANTITY["e_off"], shades(QUANTITY["e_off"], 2)[1]
    placed = Counter(FIVE[FIVE > 0])  # dots already sitting at each size
    slots = sorted({*FIVE, *np.abs(FIVE)})  # every position that gets a label, left to right

    def lower(x):  # neighbouring labels are closer than a label is wide, so every other one sits lower
        return -0.17 - 0.2 * (slots.index(x) % 2)

    for value in FIVE:
        size = abs(value)
        if value > 0:
            ax.plot(value, 0, "o", color=solid, markersize=15)
            ax.text(value, lower(value), f"{value:.1f}", color=solid, ha="center", va="top")
            continue
        ax.plot(value, 0, "o", color=faded, markersize=15)
        ax.text(value, lower(value), _minus(value), color=faded, ha="center", va="top")
        height = 0.32 * placed[size]  # a repeat goes on top of the dots already there
        placed[size] += 1
        ax.plot(size, height, "o", color=solid, markersize=15)
        if height == 0:
            ax.text(size, lower(size), f"{size:.1f}", color=solid, ha="center", va="top")
        ax.annotate("", xy=(size, height + 0.08), xytext=(value, 0.09), arrowprops={
            "arrowstyle": "->", "color": faded, "lw": 2, "connectionstyle": f"arc3,rad={0.45 if size < 0.35 else 0.35}",
            "shrinkA": 4, "shrinkB": 4})
    mean = np.abs(FIVE).mean()
    ax.plot(mean, -0.85, "^", color=TEXT, markersize=13)
    ax.text(mean, -0.97, f"average size = {mean:.2f} m", ha="center", va="top")
    fig.savefig(path, dpi=200)


def _squares_rms(path):
    sides = np.abs(FIVE)
    rms_side = np.sqrt((sides ** 2).mean())
    inch = 2.0  # inches per metre of side
    rows = [(side, f"side {side:.1f} m\n→ {side ** 2:.2f} m²") for side in sides]
    height = sum(max(side * inch, 0.6) + 0.15 for side, _ in rows) + 0.3 + rms_side * inch + 1.0
    fig, ax = _canvas(height)
    palette = dict(zip(sorted(set(sides)), shades(QUANTITY["e_off"], 4)[::-1]))  # the longer the side, the stronger the blue
    y = height - 0.1
    for side, text in rows:  # one row per error: the square, then what it says
        row = max(side * inch, 0.6)
        y -= row
        ax.add_patch(Rectangle((0.2, y + (row - side * inch) / 2), side * inch, side * inch, facecolor=palette[side], edgecolor=TEXT, linewidth=1))
        ax.text(1.4, y + row / 2, text, ha="left", va="center")
        y -= 0.15
    ax.plot([0.0, W], [y - 0.05, y - 0.05], color=TEXT, linestyle=":", linewidth=1)
    y -= 0.3 + rms_side * inch
    ax.add_patch(Rectangle((0.2, y), rms_side * inch, rms_side * inch, facecolor=QUANTITY["e_off"], edgecolor=TEXT, linewidth=2.5))
    ax.text(1.4, y + rms_side * inch / 2, f"average square:\n{(sides ** 2).mean():.3f} m²", ha="left", va="center")
    ax.text(0.2, y - 0.12, f"side {rms_side:.3f} m = RMS", ha="left", va="top")
    fig.savefig(path, dpi=200)


def _drive_a_b(path):
    step, side_b = 0.35, 1.0  # m between steps; side of Drive B's big square
    inch = 1.6  # inches per metre: the same in both drives
    fig, ax = _canvas(5.9)
    x0 = 0.2
    ax.text(x0, 5.55, "Drive A", ha="left", va="bottom")
    for k in range(5):  # drive A: five squares of side 0.2
        x = x0 + k * step * inch
        ax.add_patch(Rectangle((x, 4.6), 0.2 * inch, 0.2 * inch, facecolor=QUANTITY["e_off"], edgecolor=TEXT, linewidth=1))
        ax.text(x + 0.1 * inch, 4.97, f"{0.2:.1f}", ha="center", va="bottom")
    ax.text(x0, 4.45, f"mean |e| {0.2:.1f} m, RMS {0.2:.1f} m", ha="left", va="top")
    ax.text(x0, 3.4, "Drive B", ha="left", va="bottom")
    for k in range(4):  # drive B: four steps with nothing in them
        x = x0 + (k * step + 0.1) * inch
        ax.plot(x, 1.0, "o", color=TEXT, markersize=5)
        ax.text(x, 1.05, "0", ha="center", va="bottom")
    left = x0 + 4 * step * inch
    ax.add_patch(Rectangle((left, 1.0), side_b * inch, side_b * inch, facecolor=QUANTITY["e_off"], edgecolor=TEXT, linewidth=1))
    for k in range(1, 5):
        ax.plot([left + 0.2 * k * inch] * 2, [1.0, 1.0 + side_b * inch], color=TEXT, linewidth=0.8, alpha=0.6)
        ax.plot([left, left + side_b * inch], [1.0 + 0.2 * k * inch] * 2, color=TEXT, linewidth=0.8, alpha=0.6)
    ax.text(left + side_b * inch, 1.0 + side_b * inch + 0.1, "side 1.0 m, area = 25\nof A's squares", ha="right", va="bottom")
    ax.text(x0, 0.85, f"mean |e| {0.2:.1f} m, RMS {np.sqrt(0.2):.3f} m", ha="left", va="top")
    fig.savefig(path, dpi=200)


def _offlane_zone(path):
    fig = Figure(figsize=(W, 6.6))
    ax = fig.subplots()
    _lane_view(ax, (0.0, 7.2), f"yellow line\ne = +{HALF:.1f} m\nthe run\nends here", f"white line\ne = {_minus(-HALF)} m\nthe run\nends here", label_y=4.9)
    strip = shades(QUANTITY["e_off"], 2)[1]
    for lo, hi in ((OFF_LANE_LAT, HALF), (-HALF, -OFF_LANE_LAT)):
        ax.axvspan(lo, hi, facecolor=strip, alpha=0.45, linewidth=0)
        ax.text((lo + hi) / 2, 1.8, "off the lane:\n|e| > 0.9 m", rotation=90, ha="center", va="center")
    for edge in (-OFF_LANE_LAT, OFF_LANE_LAT):
        ax.axvline(edge, color=TEXT, linestyle="--", linewidth=1.5)
        ax.text(edge, 3.5, f"e =\n{_minus(edge)} m", ha="center", va="bottom", bbox=LABEL_BOX)
    ax.text(0.0, 1.8, "on the\nlane:\n|e| ≤\n0.9 m", ha="center", va="center")
    ax.set_title("The off-lane zone")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def _curvy_offlane(path):
    tel = run(Car(), "curvy")
    size, s = np.abs(tel.lat), tel.progress_m
    off = size > OFF_LANE_LAT
    fig = Figure(figsize=(W, 4.9))
    ax = fig.subplots()
    ax.set_xlim(0.0, 64.0)
    _mark_bends(ax, make_scenario("curvy").road)
    ax.plot(s, size, color=QUANTITY["e_off"], linewidth=2)
    ax.fill_between(s, OFF_LANE_LAT, size, where=off, step="mid", facecolor=shades(QUANTITY["e_off"], 3)[2], alpha=0.6, linewidth=0)
    ax.axhline(OFF_LANE_LAT, color=TEXT, linestyle="--", linewidth=1.5)
    ax.text(0.01, OFF_LANE_LAT, "off-lane limit, 0.9 m", transform=ax.get_yaxis_transform(), ha="left", va="bottom")
    ax.axhline(HALF, color=SCENE["yellow"], linewidth=3)
    ax.text(0.01, HALF, "yellow line, 1.8 m", transform=ax.get_yaxis_transform(), color=SCENE["yellow"], ha="left", va="bottom")
    ax.annotate(f"{off.sum()} steps = {off.sum() * (tel.t[1] - tel.t[0]):.1f} s", xy=(s[off].mean(), 1.2), xytext=(s[off].mean() - 18, 1.45),
                color=shades(QUANTITY["e_off"], 3)[2], ha="center",
                arrowprops={"arrowstyle": "->", "color": shades(QUANTITY["e_off"], 3)[2], "lw": 1.5})
    ax.plot(s[-1], size[-1], "X", color=TEXT, markersize=12)
    ax.annotate("run ends", xy=(s[-1], size[-1]), xytext=(s[-1] + 3, 1.05), va="center",
                arrowprops={"arrowstyle": "->", "color": TEXT, "lw": 1.5})
    ax.set(ylim=(0.0, 2.1), xlabel=PROGRESS_LABEL, ylabel="distance from\nlane centre |e| (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


class _DoNothing:
    """Steering 0 and the provided car's speed hold: the only thing that differs from the provided car is the steering."""

    @staticmethod
    def act(est, obs):
        return Action(0.0, provided.hold_speed(obs.speed))


def _zoom_on_bend(ax, road, provided_run, nothing_run):
    """An inset over the first bend, where the do-nothing path leaves the blue one: the grey path is drawn on top there."""
    inset = ax.inset_axes([0.2, 0.58, 0.6, 0.26], gid=ZOOM_INSET)
    render_topdown(road, ax=inset)
    for tel_run, color, width in ((provided_run, QUANTITY["e_off"], 3), (nothing_run, TEXT, 3)):
        inset.plot(tel_run.x, tel_run.y, color=color, linewidth=width)
    inset.plot(nothing_run.x[-1], nothing_run.y[-1], "X", color=TEXT, markersize=12)
    inset.set(xlim=ZOOM_X, ylim=ZOOM_Y, xlabel="", ylabel="")
    inset.set(xticks=[], yticks=[])
    ax.indicate_inset_zoom(inset, edgecolor=TEXT)


def _yardstick(path):
    names = ("straight", "gentle")
    grey = TEXT
    fig = Figure(figsize=(W, 6.2))
    top = fig.add_axes([0.2, 0.76, 0.77, 0.08])
    bottom = fig.add_axes([0.2, 0.0, 0.77, 0.7])
    for ax, name in zip((top, bottom), names):
        sc = make_scenario(name)
        provided_run, nothing_run = run(Car(), sc), run(Car(controller=_DoNothing()), sc)
        render_topdown(sc.road, ax=ax)
        ax.set_title(f"'{name}'" + (":\nthe grey path lies exactly under\nthe blue one" if name == "straight" else ""))
        ax.plot(nothing_run.x, nothing_run.y, color=grey, linewidth=9 if name == "straight" else 6, alpha=0.5, solid_capstyle="butt", zorder=2)
        ax.plot(provided_run.x, provided_run.y, color=QUANTITY["e_off"], linewidth=2.5, zorder=3)
        (x0, y0), (x1, y1) = _paint_box(sc.road)
        ax.set(xlim=(x0, x1), ylim=(y0, y1))
        ax.set_xlabel("")
        if name == "straight":
            ax.set(xticks=[0, 100, 200], yticks=[0])
        if name == "gentle":
            ax.plot(nothing_run.x[-1], nothing_run.y[-1], "X", color=TEXT, markersize=16, zorder=4)
            ax.annotate("do-nothing car:\nrun ends", xy=(nothing_run.x[-1], nothing_run.y[-1]), xytext=(30, 25),
                        color=grey, ha="left", va="bottom", arrowprops={"arrowstyle": "->", "color": grey, "lw": 1.5})
            ax.annotate("provided car", xy=(provided_run.x[len(provided_run.x) // 3], provided_run.y[len(provided_run.x) // 3]),
                        xytext=(40, 78), color=QUANTITY["e_off"], ha="center",
                        arrowprops={"arrowstyle": "->", "color": QUANTITY["e_off"], "lw": 1.5})
            _zoom_on_bend(ax, sc.road, provided_run, nothing_run)
    fig.savefig(path, dpi=200)


def _array_ops(path):
    cell, x0, height = 0.8, 0.9, 8.6
    fig, ax = _canvas(height)
    code = {"fontfamily": "monospace", "ha": "center", "va": "center"}
    mid = x0 + len(FIVE) * cell / 2

    def row(y, values, filled=()):
        for k, value in enumerate(values):
            hit = k in filled
            ax.add_patch(Rectangle((x0 + k * cell, y - cell / 2), cell, cell, facecolor=QUANTITY["e_off"] if hit else BACKGROUND,
                                   edgecolor=TEXT, linewidth=1.5))
            ax.text(x0 + (k + 0.5) * cell, y, value, color=BACKGROUND if hit else TEXT, **code)

    def link(y_from, y_to, text):
        ax.annotate("", xy=(mid, y_to), xytext=(mid, y_from), arrowprops={"arrowstyle": "-|>", "color": TEXT, "lw": 2})
        ax.text(mid + 0.2, (y_from + y_to) / 2, text, **{**code, "ha": "left"})

    for top, label, step, result, filled, out in (
        (8.1, "np.abs(e)", ".mean()", f"{np.abs(FIVE).mean():.2f}", (), [f"{abs(v):.1f}" for v in FIVE]),
        (3.9, "np.abs(e) > 0.35", ".sum()", f"{int((np.abs(FIVE) > 0.35).sum())}", (3, 4), ["T" if abs(v) > 0.35 else "F" for v in FIVE]),
    ):
        ax.text(0.1, top, "e =", **{**code, "ha": "left"})
        row(top, [_minus(v, sign=False) for v in FIVE])
        link(top - 0.5, top - 1.2, label)
        row(top - 1.6, out, filled)
        link(top - 2.1, top - 2.8, step)
        ax.add_patch(Rectangle((mid - cell / 2, top - 3.2 - cell / 2), cell, cell, facecolor=QUANTITY["e_off"] if filled else BACKGROUND,
                               edgecolor=TEXT, linewidth=1.5))
        ax.text(mid, top - 3.2, result, color=BACKGROUND if filled else TEXT, **code)
        ax.text(mid + cell, top - 3.2, "the answer", ha="left", va="center")
    fig.savefig(path, dpi=200)


def _next_lesson(path):
    frame = _camera_frame()
    yellow = np.all(frame == PALETTE[YELLOW], axis=-1)  # the paint colour of the yellow line
    best = None
    for r in range(110, frame.shape[0] - PATCH):  # a patch low in the picture where the line is wide and has an edge
        for c in range(frame.shape[1] - PATCH):
            count = yellow[r:r + PATCH, c:c + PATCH].sum()
            if 14 <= count <= 22 and (best is None or abs(count - 18) < abs(best[0] - 18)):
                best = (count, r, c)
    _, r, c = best
    patch = frame[r:r + PATCH, c:c + PATCH]
    fig = Figure(figsize=(W, 9.4))
    cam = fig.add_axes([0.04, 0.64, 0.92, 0.3])
    zoom = fig.add_axes([0.05, 0.09, 0.9, 0.5])
    cam.imshow(frame)
    cam.set(xticks=[], yticks=[])
    cam.set_title("The camera picture\n(320 by 180 pixels)")
    cam.add_patch(Rectangle((c - 0.5, r - 0.5), PATCH, PATCH, fill=False, edgecolor=TEXT, linewidth=2))
    zoom.imshow(patch, interpolation="nearest")
    zoom.set_xlabel(f"The patch magnified:\n{PATCH} by {PATCH} pixels")
    zoom.set_xticks(np.arange(-0.5, PATCH, 1), minor=True)
    zoom.set_yticks(np.arange(-0.5, PATCH, 1), minor=True)
    zoom.grid(which="minor", color=BACKGROUND, linewidth=2)
    zoom.set(xticks=[], yticks=[])
    zoom.tick_params(which="minor", length=0)
    for spine in zoom.spines.values():
        spine.set(edgecolor=TEXT, linewidth=2)
    for corner_cam, corner_zoom in (((c - 0.5, r + PATCH - 0.5), (0, 1)), ((c + PATCH - 0.5, r + PATCH - 0.5), (1, 1))):
        fig.add_artist(ConnectionPatch(xyA=corner_cam, coordsA=cam.transData, xyB=corner_zoom, coordsB=zoom.transAxes,
                                       color=TEXT, linewidth=1.5))
    flat = [(i, j) for i in range(PATCH) for j in range(PATCH)]
    on_yellow = [p for p in flat if yellow[r + p[0], c + p[1]]]
    off_yellow = [p for p in flat if not yellow[r + p[0], c + p[1]]]
    picks = [on_yellow[len(on_yellow) // 2], on_yellow[0], off_yellow[len(off_yellow) // 2]]
    for i, j in picks:
        red, green, blue = (int(v) for v in patch[i, j])
        light = 0.299 * red + 0.587 * green + 0.114 * blue > 128
        zoom.text(j, i, f"R {red}\nG {green}\nB {blue}", ha="center", va="center", color=BACKGROUND if light else TEXT)
    fig.savefig(path, dpi=200)


def make(out_dir):
    with matplotlib.rc_context(PHONE_TEXT):
        for name, draw in (("report_card", _report_card), ("loop_diagram", _loop_diagram), ("car_and_view", _car_and_view),
                           ("test_roads", _test_roads), ("sign_convention", _sign_convention), ("gentle_trace", _gentle_trace),
                           ("cancel_numberline", _cancel_numberline), ("signed_mean_trap", _signed_mean_trap),
                           ("fold_numberline", _fold_numberline), ("squares_rms", _squares_rms), ("drive_a_b", _drive_a_b),
                           ("offlane_zone", _offlane_zone), ("curvy_offlane", _curvy_offlane), ("yardstick", _yardstick),
                           ("array_ops", _array_ops), ("next_lesson", _next_lesson)):
            draw(out_dir / f"{name}.png")
