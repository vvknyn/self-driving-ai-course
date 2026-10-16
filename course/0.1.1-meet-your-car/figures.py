"""Figures for lecture 0.1.1 Meet your car.  `python scripts/make_figures.py course/0.1.1-meet-your-car`."""
from collections import Counter

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
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

VIEW_S = 80.0  # m along `gentle`: 20 m before the first bend, so the bend is in view
STIFF_GAINS = (0.7, 2.7)  # a car whose offset gain is twice the provided one (gains are unit 0.1.4)
ROAD_ROWS = (("straight",), ("gentle", "curvy"))  # test_roads layout, top row first
FIG_W, MARGIN_L, MARGIN_R, MARGIN_B, MARGIN_T, GAP_X, GAP_Y = 11.0, 0.75, 0.2, 0.65, 0.45, 0.85, 1.0  # inches
ROAD_PAD, ARROW_LEN = 10.0, 20.0  # metres
HALF = LANE_WIDTH / 2  # m from lane centre to a painted line
FIVE = np.array([0.1, -0.3, 0.2, -0.4, 0.4])  # the lecture's worked example
PATCH = 6  # pixels per side of the magnified patch in next_lesson
TEXT, PANEL, BACKGROUND = CHROME["text"], CHROME["panel"], CHROME["background"]


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
        ax.text((a + b) / 2, 0.97, f"bend {i}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11)


def _brace(ax, x0, x1, y, height, **kwargs):
    """A curly brace from x0 to x1: its two feet stand on y, its tip points up to y + height."""
    xm, d, mid = (x0 + x1) / 2, (x1 - x0) / 8, y + height / 2
    verts = [(x0, y), (x0, mid), (x0 + d, mid), (xm - d, mid), (xm, mid), (xm, y + height),
             (xm, mid), (xm + d, mid), (x1 - d, mid), (x1, mid), (x1, y)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3]
    ax.add_patch(PathPatch(Path(verts, codes), fill=False, **kwargs))


def _lane_view(ax, ylim, yellow_text, white_text):
    """The lane seen from above, travelling up the page: left of centre is on the left, so e grows to the left."""
    ax.set(xlim=(3.7, -3.7), ylim=ylim, xlabel="lateral error e (m): left of lane centre is +")
    ax.set_aspect("equal")
    ax.set_yticks([])
    ax.axvspan(-HALF, HALF, facecolor=SCENE["road"], alpha=0.5, linewidth=0)
    ax.axvline(HALF, color=SCENE["yellow"], linestyle=(0, (6, 3)), linewidth=3)
    ax.axvline(-HALF, color=SCENE["white"], linewidth=3)
    ax.axvline(0.0, color=TEXT, linestyle=":", linewidth=1)
    ax.text(HALF + 0.12, ylim[1] * 0.72, yellow_text, ha="right", va="center", fontsize=11)
    ax.text(-HALF - 0.12, ylim[1] * 0.72, white_text, ha="left", va="center", fontsize=11)
    ax.annotate("", xy=(-2.8, 2.2), xytext=(-2.8, 0.4), arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.text(-2.8, 2.45, "direction\nof travel", ha="center", va="bottom", fontsize=11)


def _car_and_view(path):
    frame = _camera_frame()
    road = make_scenario("gentle").road
    tel = run(Car(), "gentle")
    fig = Figure(figsize=(11, 4))
    cam, top = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.6, 1]})
    cam.imshow(frame)
    cam.set_title(f"What the camera sees, {VIEW_S:.0f} m into 'gentle'")
    cam.set_xlabel("pixel column")
    cam.set_ylabel("pixel row")
    render_topdown(road, [np.column_stack([tel.x, tel.y])], ["the provided car"], ax=top)
    top.plot(*road.offset(VIEW_S, -LANE_WIDTH / 2), "o", color=QUANTITY["perception"], label="camera position")
    top.legend(loc="center")
    top.set_title("Where the car drove (top-down)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def _paint_box(road):
    """Corners ((x0, y0), (x1, y1)) of a box around the road's paint, with room for the markers."""
    s = np.linspace(0.0, road.length, max(2, int(road.length)))
    paint = np.vstack([road.offset(s, -LANE_WIDTH), road.offset(s, LANE_WIDTH)])
    return paint.min(axis=0) - ROAD_PAD, paint.max(axis=0) + ROAD_PAD


def _mark_route(ax, scenario):
    """Start dot and direction-of-travel arrow in the ego lane; an open road also gets an end-of-route square."""
    road = scenario.road
    x, y, yaw = road.ego_pose(scenario.start_s)
    ax.plot(x, y, "o", color=QUANTITY["perception"], zorder=3)
    ax.annotate("", xy=(x + ARROW_LEN * np.cos(yaw), y + ARROW_LEN * np.sin(yaw)), xytext=(x, y),
                arrowprops={"arrowstyle": "-|>", "color": QUANTITY["perception"], "lw": 2})
    if not road.closed:
        ax.plot(*road.ego_pose(scenario.start_s + scenario.route_length)[:2], "s", color=TEXT, zorder=3)


def _test_roads(path):
    scenarios = {name: make_scenario(name) for row in ROAD_ROWS for name in row}
    boxes = {name: _paint_box(sc.road) for name, sc in scenarios.items()}
    size = {name: hi - lo for name, (lo, hi) in boxes.items()}  # metres, (x, y)
    # One scale for every panel: the tightest row sets how many metres fit in an inch.
    m_per_in = max(sum(size[n][0] for n in row) / (FIG_W - MARGIN_L - MARGIN_R - GAP_X * (len(row) - 1)) for row in ROAD_ROWS)
    row_h = [max(size[n][1] for n in row) / m_per_in for row in ROAD_ROWS]
    height = MARGIN_B + sum(row_h) + GAP_Y * (len(ROAD_ROWS) - 1) + MARGIN_T
    fig = Figure(figsize=(FIG_W, height))
    y_in = MARGIN_B
    for row, h in zip(reversed(ROAD_ROWS), reversed(row_h)):
        x_in = MARGIN_L
        for name in row:
            (x0, y0), (x1, y1) = boxes[name]
            ax = fig.add_axes([x_in / FIG_W, y_in / height, (x1 - x0) / m_per_in / FIG_W, (y1 - y0) / m_per_in / height],
                              xlim=(x0, x1), ylim=(y0, y1))
            render_topdown(scenarios[name].road, ax=ax)
            _mark_route(ax, scenarios[name])
            ax.set_title(f"'{name}': {scenarios[name].route_length:.1f} m")
            x_in += (x1 - x0) / m_per_in + GAP_X
        y_in += h + GAP_Y
    key = [Line2D([], [], marker="o", color=QUANTITY["perception"], linestyle="", label="start (arrow: direction of travel)"),
           Line2D([], [], marker="s", color=TEXT, linestyle="",
                  label="end of route (a loop ends at its start;\nthe paint of an open road runs on past it)")]
    fig.legend(handles=key, loc="upper right", bbox_to_anchor=(0.98, 1 - MARGIN_T / height), fontsize=9)
    fig.savefig(path, dpi=100)


def _sign_convention(path):
    fig = Figure(figsize=(8, 6.6))
    ax = fig.subplots()
    _lane_view(ax, (0.0, 9.6), f"yellow centre line\n(left edge of your lane)\ne = +{HALF:.1f} m",
               f"white edge line\n(right edge of your lane)\ne = {_minus(-HALF)} m")
    ax.text(0.0, 9.0, "lane centre, e = 0", ha="center", va="bottom", fontsize=11)
    for (y, e), color in zip(((3.8, 0.5), (7.6, -0.3)), shades(QUANTITY["e_off"], 2)):
        ax.add_patch(Rectangle((e - 0.8, y - 2.7), 1.6, 2.7, facecolor=color, alpha=0.35, edgecolor=color))  # 2.7 m: the wheelbase
        ax.annotate("", xy=(e, y), xytext=(0.0, y), arrowprops={"arrowstyle": "->", "color": color, "lw": 2})
        ax.plot(e, y, "o", color=color)
        ax.text(e, y - 2.9, f"e = {_minus(e)} m", color=color, ha="center", va="top", fontsize=11,
                bbox={"facecolor": PANEL, "linewidth": 0, "pad": 2})
    ax.set_title("Lateral error e, measured at the front axle (dot): left of centre is positive")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def _report_card(path):
    tel = run(Car(), "gentle")
    size = np.abs(tel.lat)
    lines = (  # (value, label, where the label sits on the y axis)
        (size.mean(), "typically off {:.2f} m (mean |e|)", 0.12),
        (np.sqrt((tel.lat ** 2).mean()), "big errors count extra {:.2f} m (RMS)", 0.62),
        (size.max(), "worst moment {:.2f} m (max |e|)", 1.12),
    )
    fig = Figure(figsize=(11, 4.4))
    ax = fig.subplots()
    fig.subplots_adjust(left=0.08, right=0.64, bottom=0.14, top=0.94)
    ax.plot(tel.progress_m, size, color=QUANTITY["e_off"], linewidth=1.2)
    for (value, text, y_label), color in zip(lines, shades(QUANTITY["e_off"], 3)):
        ax.axhline(value, color=color, linewidth=2, linestyle=(0, (6, 3)))
        ax.annotate(text.format(value), xy=(1.0, value), xytext=(1.03, y_label), xycoords=("axes fraction", "data"),
                    textcoords=("axes fraction", "data"), color=color, fontsize=11, va="center", annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 1.2, "shrinkA": 0, "shrinkB": 0})
    ax.axhline(HALF, color=SCENE["yellow"], linewidth=3)
    ax.text(1.03, HALF, "painted line, 1.8 m", transform=ax.get_yaxis_transform(), color=SCENE["yellow"], fontsize=11, va="center")
    ax.set(ylim=(0.0, 2.0), xlim=(0.0, tel.progress_m[-1]), xlabel="distance driven (m)", ylabel="distance from lane centre (m)")
    fig.savefig(path, dpi=100)


def _loop_diagram(path):
    fig = Figure(figsize=(11, 5.6))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 12), ylim=(0, 6.2))
    ax.set_axis_off()
    w, h = 3.2, 1.3
    boxes = {"camera": (2.4, 4.8, TEXT), "perception": (9.6, 4.8, QUANTITY["perception"]),
             "controller": (9.6, 1.9, TEXT), "car moves": (2.4, 1.9, TEXT)}
    for name, (x, y, edge) in boxes.items():
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    facecolor=BACKGROUND, edgecolor=edge, linewidth=2.5))
        closed = name in ("perception", "controller")
        ax.text(x, y + (0.2 if closed else 0.0), name, ha="center", va="center", fontsize=15)
        if closed:
            ax.text(x, y - 0.33, "stays closed in this lesson", ha="center", va="center", fontsize=10)

    def arrow(start, end, color):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.5, "shrinkA": 0, "shrinkB": 0})

    arrow((4.0, 4.8), (8.0, 4.8), TEXT)
    ax.text(6.0, 4.95, "picture", ha="center", va="bottom", fontsize=11)
    arrow((9.6, 4.15), (9.6, 2.55), QUANTITY["perception"])
    ax.text(9.8, 3.35, "where the lane is", color=QUANTITY["perception"], ha="left", va="center", fontsize=12)
    arrow((8.0, 1.9), (4.0, 1.9), QUANTITY["delta"])
    ax.text(6.0, 2.05, "steering angle", color=QUANTITY["delta"], ha="center", va="bottom", fontsize=12)
    arrow((2.4, 2.55), (2.4, 4.15), TEXT)
    ax.text(2.2, 3.35, "the new view", ha="right", va="center", fontsize=11)
    ax.plot(2.4, 1.25, "o", color=QUANTITY["e_off"], markersize=9)  # the tap: a measurement taken off the car
    ax.plot([2.4, 2.4], [1.25, 0.6], color=QUANTITY["e_off"], linewidth=2.5)
    ax.plot([2.4, 3.2], [0.6, 0.6], color=QUANTITY["e_off"], linewidth=2.5)
    ax.text(3.35, 0.6, "you measure here: lateral error e", color=QUANTITY["e_off"], ha="left", va="center", fontsize=13)
    ax.text(6.0, 5.9, "the loop runs every 0.05 s", ha="center", va="center", fontsize=12)
    fig.savefig(path, dpi=100)


def _gentle_trace(path):
    tel = run(Car(), "gentle")
    fig = Figure(figsize=(11, 4))
    ax = fig.subplots()
    _mark_bends(ax, make_scenario("gentle").road)
    ax.plot(tel.progress_m, tel.lat, color=QUANTITY["e_off"], linewidth=1.5)
    ax.axhline(0.0, color=TEXT, linestyle=":", linewidth=1)
    ax.text(0.01, 0.0, "lane centre", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=11)
    ax.set(xlim=(0.0, tel.progress_m[-1]), ylim=(-0.12, 0.75), xlabel="distance driven (m)", ylabel="lateral error e (m), left of centre is +")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def _number_line(ax, limit):
    """A horizontal e axis across the lane, left of centre on the left like the lane figure; returns nothing."""
    ax.set_axis_off()
    ax.set(xlim=(limit, -limit))
    ax.plot([limit, -limit], [0, 0], color=TEXT, linewidth=1.5)
    ticks = np.arange(-0.4, 0.41, 0.1)
    ax.plot(ticks, np.zeros_like(ticks), "|", color=TEXT, markersize=10)
    ax.plot([0.0, 0.0], [0.0, ax.get_ylim()[1] - 0.12], color=TEXT, linestyle=":", linewidth=1)
    ax.text(0.0, ax.get_ylim()[1], "lane centre, e = 0", ha="center", va="top", fontsize=11)
    ax.text(limit, -0.55, "left of centre (+)", ha="left", va="center", fontsize=11)
    ax.text(-limit, -0.55, "right of centre (−)", ha="right", va="center", fontsize=11)


def _cancel_numberline(path):
    fig = Figure(figsize=(11, 3.8))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96], ylim=(-0.75, 1.25))
    _number_line(ax, 0.6)
    for value in FIVE:
        ax.plot(value, 0, "o", color=QUANTITY["e_off"], markersize=15)
        ax.text(value, 0.17, _minus(value), color=QUANTITY["e_off"], ha="center", va="bottom", fontsize=12)
    for lo, hi, text in ((0.07, 0.43, "+0.7"), (-0.43, -0.27, "−0.7")):
        _brace(ax, lo, hi, 0.5, 0.25, edgecolor=TEXT, linewidth=1.8)
        ax.text((lo + hi) / 2, 0.82, text, ha="center", va="bottom", fontsize=14)
    ax.plot(0.0, -0.17, "^", color=TEXT, markersize=13)
    ax.text(0.0, -0.3, "average = 0", ha="center", va="top", fontsize=13)
    fig.savefig(path, dpi=100)


def _signed_mean_trap(path):
    runs = [
        (run(Car(), "gentle"), "provided car on 'gentle'"),
        (run(Car(controller=lambda est, obs: provided.steer_p(est, *STIFF_GAINS)), "curvy"),
         f"stiffer car (k_off={STIFF_GAINS[0]}) on 'curvy'"),
    ]
    fig = Figure(figsize=(11, 4))
    for ax, (tel, name) in zip(fig.subplots(1, 2, sharey=True), runs):
        signed = tel.lat.mean()
        ax.plot(tel.progress_m, tel.lat, color=QUANTITY["e_off"], linewidth=1)
        ax.axhline(signed, color=QUANTITY["perception"], linewidth=2)
        ax.text(0.5, 0.88, f"line = average {_minus(signed, 2)} m", transform=ax.transAxes, color=QUANTITY["perception"],
                ha="center", va="center", fontsize=12)
        ax.set_ylim(top=1.0)
        ax.set(title=name, xlabel="distance along the route (m)")
    fig.axes[0].set_ylabel("lateral error e (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def _fold_numberline(path):
    fig = Figure(figsize=(11, 4.4))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96], ylim=(-0.85, 1.45))
    _number_line(ax, 0.6)
    solid, faded = QUANTITY["e_off"], shades(QUANTITY["e_off"], 2)[1]
    placed = Counter(FIVE[FIVE > 0])  # dots already sitting at each size
    for value in FIVE:
        size = abs(value)
        if value > 0:
            ax.plot(value, 0, "o", color=solid, markersize=15)
            ax.text(value, -0.17, f"{value:.1f}", color=solid, ha="center", va="top", fontsize=12)
            continue
        ax.plot(value, 0, "o", color=faded, markersize=15)
        ax.text(value, -0.17, _minus(value), color=faded, ha="center", va="top", fontsize=12)
        height = 0.32 * placed[size]  # a repeat goes on top of the dots already there
        placed[size] += 1
        ax.plot(size, height, "o", color=solid, markersize=15)
        if height == 0:
            ax.text(size, -0.17, f"{size:.1f}", color=solid, ha="center", va="top", fontsize=12)
        ax.annotate("", xy=(size, height + 0.08), xytext=(value, 0.09), arrowprops={
            "arrowstyle": "->", "color": faded, "lw": 2, "connectionstyle": f"arc3,rad={0.45 if size < 0.35 else 0.35}",
            "shrinkA": 4, "shrinkB": 4})
    mean = np.abs(FIVE).mean()
    ax.plot(mean, -0.52, "^", color=TEXT, markersize=13)
    ax.text(mean, -0.64, f"average size = {mean:.2f} m", ha="center", va="top", fontsize=13)
    fig.savefig(path, dpi=100)


def _squares_rms(path):
    sides = np.abs(FIVE)
    rms_side = np.sqrt((sides ** 2).mean())
    fig = Figure(figsize=(11, 4.4))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_axis_off()
    ax.set_aspect("equal")
    palette = dict(zip(sorted(set(sides)), shades(QUANTITY["e_off"], 4)[::-1]))  # the longer the side, the stronger the blue
    x = 0.0
    for side in sides:
        ax.add_patch(Rectangle((x, 0), side, side, facecolor=palette[side], edgecolor=TEXT, linewidth=1))
        ax.text(x + side / 2, side + 0.03, f"{side:.1f} m\n→ {side ** 2:.2f} m²", ha="center", va="bottom", fontsize=11)
        x += side + 0.27
    x += 0.3
    ax.add_patch(Rectangle((x, 0), rms_side, rms_side, facecolor=QUANTITY["e_off"], edgecolor=TEXT, linewidth=2.5))
    ax.text(x + rms_side / 2, rms_side + 0.03, f"average square:\n{(sides ** 2).mean():.3f} m²", ha="center", va="bottom", fontsize=11)
    _brace(ax, x, x + rms_side, -0.2, -0.1, edgecolor=TEXT, linewidth=1.8)
    ax.text(x + rms_side / 2, -0.33, f"side {rms_side:.3f} m = RMS", ha="center", va="top", fontsize=13)
    ax.plot([x - 0.27] * 2, [0, 0.45], color=TEXT, linestyle=":", linewidth=1)
    ax.set(xlim=(-0.05, x + rms_side + 0.1), ylim=(-0.75, 0.75))
    fig.savefig(path, dpi=100)


def _drive_a_b(path):
    step = 0.35  # m between steps
    fig = Figure(figsize=(11, 4.9))
    spans = [4 * step + 0.2 + 0.2, 4 * step + 1.0 + 0.2]  # the x extent of each panel, in metres
    y0, y1 = -0.1, 1.3
    scale = (11 - 0.8) / sum(spans) * 0.97  # inches per metre: the same in both panels
    x_in = 0.4
    panels = []
    for span, name, text in zip(spans, "AB", (f"mean |e| {0.2:.1f} m, RMS {0.2:.1f} m", f"mean |e| {0.2:.1f} m, RMS {np.sqrt(0.2):.3f} m")):
        ax = fig.add_axes([x_in / 11, 0.9 / 4.9, span * scale / 11, (y1 - y0) * scale / 4.9], xlim=(0, span), ylim=(y0, y1))
        ax.set_axis_off()
        ax.text(span / 2, y0 - 0.1, text, ha="center", va="top", fontsize=13)
        ax.text(span / 2, y1 + 0.02, f"Drive {name}", ha="center", va="bottom", fontsize=13)
        panels.append(ax)
        x_in += span * scale + 0.4
    a, b = panels
    shade = shades(QUANTITY["e_off"], 3)
    for k in range(5):  # drive A: five squares of side 0.2
        a.add_patch(Rectangle((0.2 + k * step, 0), 0.2, 0.2, facecolor=QUANTITY["e_off"], edgecolor=TEXT, linewidth=1))
        a.text(0.3 + k * step, 0.24, f"{0.2:.1f}", ha="center", va="bottom", fontsize=11)
    for k in range(4):  # drive B: four steps with nothing in them
        b.plot(0.2 + k * step + 0.1, 0.0, "o", color=TEXT, markersize=5)
        b.text(0.3 + k * step, 0.04, "0", ha="center", va="bottom", fontsize=11)
    left = 4 * step + 0.2
    b.add_patch(Rectangle((left, 0), 1.0, 1.0, facecolor=shade[1], alpha=0.6, edgecolor=TEXT, linewidth=1.5))
    for k in range(1, 5):
        b.plot([left + 0.2 * k] * 2, [0, 1.0], color=TEXT, linewidth=0.8, alpha=0.6)
        b.plot([left, left + 1.0], [0.2 * k] * 2, color=TEXT, linewidth=0.8, alpha=0.6)
    b.text(left + 0.5, 1.04, "side 1.0 m = 25 of A's squares", ha="center", va="bottom", fontsize=11)
    fig.savefig(path, dpi=100)


def _offlane_zone(path):
    fig = Figure(figsize=(8, 5.6))
    ax = fig.subplots()
    _lane_view(ax, (0.0, 6.9), f"yellow line\ne = +{HALF:.1f} m\nthe run ends here", f"white line\ne = {_minus(-HALF)} m\nthe run ends here")
    strip = shades(QUANTITY["e_off"], 2)[1]
    for lo, hi in ((OFF_LANE_LAT, HALF), (-HALF, -OFF_LANE_LAT)):
        ax.axvspan(lo, hi, facecolor=strip, alpha=0.45, linewidth=0)
        ax.text((lo + hi) / 2, 3.1, "off the lane: |e| > 0.9 m", rotation=90, ha="center", va="center", fontsize=11)
    for edge in (-OFF_LANE_LAT, OFF_LANE_LAT):
        ax.axvline(edge, color=TEXT, linestyle="--", linewidth=1.5)
        ax.text(edge, 6.3, f"e = {_minus(edge)} m", ha="center", va="bottom", fontsize=11,
                bbox={"facecolor": PANEL, "linewidth": 0, "pad": 2})
    ax.text(0.0, 3.1, "on the lane:\n|e| ≤ 0.9 m", ha="center", va="center", fontsize=12)
    ax.set_title("The off-lane zone")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def _curvy_offlane(path):
    tel = run(Car(), "curvy")
    size, s = np.abs(tel.lat), tel.progress_m
    off = size > OFF_LANE_LAT
    fig = Figure(figsize=(11, 4.4))
    ax = fig.subplots()
    _mark_bends(ax, make_scenario("curvy").road)
    ax.plot(s, size, color=QUANTITY["e_off"], linewidth=2)
    ax.fill_between(s, OFF_LANE_LAT, size, where=off, step="mid", facecolor=QUANTITY["perception"], alpha=0.6, linewidth=0)
    ax.axhline(OFF_LANE_LAT, color=TEXT, linestyle="--", linewidth=1.5)
    ax.text(0.01, OFF_LANE_LAT, "off-lane limit, 0.9 m", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=11)
    ax.axhline(HALF, color=SCENE["yellow"], linewidth=3)
    ax.text(0.01, HALF, "painted line, 1.8 m", transform=ax.get_yaxis_transform(), color=SCENE["yellow"], ha="left", va="bottom", fontsize=11)
    ax.annotate(f"{off.sum()} steps = {off.sum() * (tel.t[1] - tel.t[0]):.1f} s", xy=(s[off].mean(), 1.2), xytext=(s[off].mean() - 18, 1.45),
                color=QUANTITY["perception"], fontsize=13, ha="center", arrowprops={"arrowstyle": "->", "color": QUANTITY["perception"], "lw": 1.5})
    ax.plot(s[-1], size[-1], "X", color=TEXT, markersize=12)
    ax.annotate("run ends", xy=(s[-1], size[-1]), xytext=(s[-1] + 3, 1.05), fontsize=12, va="center",
                arrowprops={"arrowstyle": "->", "color": TEXT, "lw": 1.5})
    ax.set(xlim=(0.0, 64.0), ylim=(0.0, 2.1), xlabel="distance driven (m)", ylabel="distance from lane centre |e| (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


class _DoNothing:
    """Steering 0 and the provided car's speed hold: the only thing that differs from the provided car is the steering."""

    @staticmethod
    def act(est, obs):
        return Action(0.0, provided.hold_speed(obs.speed))


def _yardstick(path):
    names = ("straight", "gentle")
    grey = TEXT
    fig = Figure(figsize=(10, 8.2))
    top = fig.add_axes([0.06, 0.82, 0.9, 0.12])
    bottom = fig.add_axes([0.06, 0.06, 0.9, 0.7])
    for ax, name in zip((top, bottom), names):
        sc = make_scenario(name)
        provided_run, nothing_run = run(Car(), sc), run(Car(controller=_DoNothing()), sc)
        render_topdown(sc.road, ax=ax)
        ax.set_title(f"'{name}'" + (": the grey path lies exactly under the blue one" if name == "straight" else ""))
        ax.plot(nothing_run.x, nothing_run.y, color=grey, linewidth=9 if name == "straight" else 6, alpha=0.5, solid_capstyle="butt", zorder=2)
        ax.plot(provided_run.x, provided_run.y, color=QUANTITY["e_off"], linewidth=2.5, zorder=3)
        (x0, y0), (x1, y1) = _paint_box(sc.road)
        ax.set(xlim=(x0, x1), ylim=(y0, y1))
        ax.set_xlabel("")
        if name == "gentle":
            ax.plot(nothing_run.x[-1], nothing_run.y[-1], "X", color=QUANTITY["perception"], markersize=16, zorder=4)
            ax.annotate("do-nothing car: run ends", xy=(nothing_run.x[-1], nothing_run.y[-1]), xytext=(20, 25), fontsize=12,
                        color=grey, ha="left", va="bottom", arrowprops={"arrowstyle": "->", "color": grey, "lw": 1.5})
            ax.annotate("provided car", xy=(provided_run.x[len(provided_run.x) // 3], provided_run.y[len(provided_run.x) // 3]),
                        xytext=(40, 60), fontsize=12, color=QUANTITY["e_off"], ha="center",
                        arrowprops={"arrowstyle": "->", "color": QUANTITY["e_off"], "lw": 1.5})
    fig.savefig(path, dpi=100)


def _array_ops(path):
    cell = 0.62
    fig = Figure(figsize=(12.5, 4.6))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 12.5), ylim=(0, 4.6))
    ax.set_axis_off()
    code = {"fontfamily": "monospace", "fontsize": 13, "ha": "center", "va": "center"}

    def row(x, y, values, filled=()):
        for k, value in enumerate(values):
            hit = k in filled
            ax.add_patch(Rectangle((x + k * cell, y - cell / 2), cell, cell, facecolor=QUANTITY["e_off"] if hit else BACKGROUND,
                                   edgecolor=TEXT, linewidth=1.5))
            ax.text(x + (k + 0.5) * cell, y, value, color=BACKGROUND if hit else TEXT, **{**code, "fontsize": 12})
        return x + len(values) * cell

    def link(x0, x1, y, text):
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops={"arrowstyle": "-|>", "color": TEXT, "lw": 2})
        ax.text((x0 + x1) / 2, y + 0.18, text, **{**code, "va": "bottom"})

    for y, label, step, result, filled, out in (
        (3.2, "np.abs(e)", ".mean()", f"{np.abs(FIVE).mean():.2f}", (), [f"{abs(v):.1f}" for v in FIVE]),
        (1.2, "np.abs(e) > 0.35", ".sum()", f"{int((np.abs(FIVE) > 0.35).sum())}", (3, 4), ["T" if abs(v) > 0.35 else "F" for v in FIVE]),
    ):
        ax.text(0.2, y, "e =", **{**code, "ha": "left"})
        end = row(0.9, y, [_minus(v, sign=False) for v in FIVE])
        link(end + 0.1, end + 2.2, y, label)
        end = row(end + 2.3, y, out, filled)
        link(end + 0.1, end + 1.3, y, step)
        row(end + 1.4, y, [result], filled=(0,) if filled else ())
        ax.text(end + 1.4 + cell / 2, y - 0.55, "the answer", **{**code, "fontfamily": "sans-serif", "fontsize": 11, "va": "top"})
    fig.savefig(path, dpi=100)


def _next_lesson(path):
    frame = _camera_frame()
    yellow = np.all(frame == PALETTE[YELLOW], axis=-1)  # the paint colour of the centre line
    best = None
    for r in range(110, frame.shape[0] - PATCH):  # a patch low in the picture where the line is wide and has an edge
        for c in range(frame.shape[1] - PATCH):
            count = yellow[r:r + PATCH, c:c + PATCH].sum()
            if 14 <= count <= 22 and (best is None or abs(count - 18) < abs(best[0] - 18)):
                best = (count, r, c)
    _, r, c = best
    patch = frame[r:r + PATCH, c:c + PATCH]
    fig = Figure(figsize=(12, 5.2))
    cam = fig.add_axes([0.04, 0.1, 0.5, 0.8])
    zoom = fig.add_axes([0.62, 0.1, 0.36, 0.8])
    cam.imshow(frame)
    cam.set_title("The camera picture (320 by 180 pixels)")
    cam.add_patch(Rectangle((c - 0.5, r - 0.5), PATCH, PATCH, fill=False, edgecolor=QUANTITY["perception"], linewidth=2))
    zoom.imshow(patch, interpolation="nearest")
    zoom.set_title(f"The patch magnified: {PATCH} by {PATCH} pixels")
    zoom.set_xticks(np.arange(-0.5, PATCH, 1), minor=True)
    zoom.set_yticks(np.arange(-0.5, PATCH, 1), minor=True)
    zoom.grid(which="minor", color=BACKGROUND, linewidth=2)
    zoom.set(xticks=[], yticks=[])
    zoom.tick_params(which="minor", length=0)
    for spine in zoom.spines.values():
        spine.set(edgecolor=QUANTITY["perception"], linewidth=2)
    for corner_cam, corner_zoom in (((c + PATCH - 0.5, r - 0.5), (0, 1)), ((c + PATCH - 0.5, r + PATCH - 0.5), (0, 0))):
        fig.add_artist(ConnectionPatch(xyA=corner_cam, coordsA=cam.transData, xyB=corner_zoom, coordsB=zoom.transAxes,
                                       color=QUANTITY["perception"], linewidth=1.5))
    flat = [(i, j) for i in range(PATCH) for j in range(PATCH)]
    on_yellow = [p for p in flat if yellow[r + p[0], c + p[1]]]
    off_yellow = [p for p in flat if not yellow[r + p[0], c + p[1]]]
    picks = [on_yellow[len(on_yellow) // 2], on_yellow[0], off_yellow[len(off_yellow) // 2]]
    for i, j in picks:
        red, green, blue = (int(v) for v in patch[i, j])
        light = 0.299 * red + 0.587 * green + 0.114 * blue > 128
        zoom.text(j, i, f"R {red}\nG {green}\nB {blue}", ha="center", va="center", fontsize=10, color=BACKGROUND if light else TEXT)
    fig.savefig(path, dpi=100)


def make(out_dir):
    for name, draw in (("report_card", _report_card), ("loop_diagram", _loop_diagram), ("car_and_view", _car_and_view),
                       ("test_roads", _test_roads), ("sign_convention", _sign_convention), ("gentle_trace", _gentle_trace),
                       ("cancel_numberline", _cancel_numberline), ("signed_mean_trap", _signed_mean_trap),
                       ("fold_numberline", _fold_numberline), ("squares_rms", _squares_rms), ("drive_a_b", _drive_a_b),
                       ("offlane_zone", _offlane_zone), ("curvy_offlane", _curvy_offlane), ("yardstick", _yardstick),
                       ("array_ops", _array_ops), ("next_lesson", _next_lesson)):
        draw(out_dir / f"{name}.png")
