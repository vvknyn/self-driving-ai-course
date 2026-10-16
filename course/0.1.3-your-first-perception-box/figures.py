"""Figures for lecture 0.1.3 Your first perception box.  `python scripts/make_figures.py course/0.1.3-your-first-perception-box`."""
import numpy as np
from matplotlib.figure import Figure

from zero2fsd.car import Observation
from zero2fsd.car.provided import FAR_ROWS, NEAR_ROWS, estimate_lane, lane_point, paint_masks
from zero2fsd.palette import CHROME, QUANTITY, SCENE
from zero2fsd.sim import BicycleState, Camera, make_scenario
from zero2fsd.sim.camera import WHITE, YELLOW, to_vehicle
from zero2fsd.sim.scenarios import TARGET_SPEED
from zero2fsd.sim.world import EGO_LANE_CENTER, WHITE_LINE_LATERAL

CAMERA = Camera()
HORIZON = CAMERA.ground_to_pixel(1e9, 0.0)[1]
RAY_ROWS = (95, 116, 125, 150, 179)  # the band edges, plus the lecture's worked row 150
WORKED = ("straight", 20.0, 0.4, 0.05)  # scenario, s, offset, heading: the lab renders this same frame
BENDS = (("gentle", 150.0), ("curvy", 40.0))  # centred and aligned, well inside a bend
BANDS = ((NEAR_ROWS, "near band"), (FAR_ROWS, "far band"))
PAINT = {YELLOW: SCENE["yellow"], WHITE: SCENE["white"]}  # paint drawn in the colours the camera sees


def _view(scenario, s, offset=0.0, heading=0.0):
    road = make_scenario(scenario).road
    state = BicycleState(*road.ego_pose(s, offset, heading), TARGET_SPEED, 0.0)
    return road, state, CAMERA.render(road, state)


def _road_line(ax, road, state, s, lateral, ahead, **style):
    """A line of constant lateral position on the road, drawn top-down in the car's frame (left to the left)."""
    forward, left = to_vehicle(state, *road.offset(np.linspace(s - 1, s + ahead, 200), lateral).T)
    ax.plot(left, forward, **style)


def _top_down(ax, title):
    ax.set_facecolor(SCENE["road"])
    ax.plot(0, 0, marker="^", ms=12, color=QUANTITY["perception"], zorder=5)
    ax.invert_xaxis()  # left is positive, so it goes on the left, as in the camera frame
    ax.set_xlabel("left of the car (m)")
    ax.set_ylabel("ahead of the car (m)")
    ax.set_title(title)


def rows_to_distance(path):
    fig = Figure(figsize=(12, 4))
    side, curve = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.3, 1]})
    h = CAMERA.mount_height
    side.axhline(0, color=CHROME["text"], lw=1)
    side.plot([0, 13], [h, h], ls="--", color=QUANTITY["e_off"])
    side.text(12.9, h + 0.05, f"row {HORIZON:.1f}: the horizon, parallel to the ground", ha="right", va="bottom", color=QUANTITY["e_off"])
    for k, v in enumerate(sorted(RAY_ROWS, reverse=True)):  # nearest first, labels on alternating levels
        d = CAMERA.pixel_to_ground(CAMERA.cx, v)[0]
        bold = v == 150
        side.plot([0, d], [h, 0], color=QUANTITY["perception"] if bold else CHROME["text"], lw=2 if bold else 1)
        side.text(d, -0.07 - 0.3 * (k % 2), f"row {v}\n{d:.2f} m", ha="center", va="top", fontsize=8,
                  color=QUANTITY["perception"] if bold else CHROME["text"])
    side.plot(0, h, marker="s", ms=9, color=CHROME["text"])
    side.annotate("", xy=(-0.3, 0), xytext=(-0.3, h), arrowprops={"arrowstyle": "<->"})
    side.text(-0.4, h / 2, f"h = {h} m", ha="right", va="center")
    side.set_xlim(-1.6, 13)
    side.set_ylim(-0.75, 1.75)
    side.set_xlabel("ahead of the car (m)")
    side.set_yticks([])
    side.set_title("side view: one ray per image row (vertical scale stretched)")

    rows = np.arange(76, 180)
    curve.semilogy(rows, CAMERA.pixel_to_ground(CAMERA.cx, rows)[0], color=CHROME["text"])
    curve.axvline(HORIZON, ls="--", color=QUANTITY["e_off"])
    curve.text(HORIZON + 1, 300, "horizon", color=QUANTITY["e_off"])
    for (a, b), name in BANDS:
        curve.axvspan(a, b, alpha=0.25, color=QUANTITY["psi"])
        curve.text((a + b) / 2, 150, name, ha="center", fontsize=9)
    curve.plot(150, CAMERA.pixel_to_ground(CAMERA.cx, 150)[0], "o", color=QUANTITY["perception"])
    curve.set_xlabel("image row v (0 is the top)")
    curve.set_ylabel("distance ahead (m), log scale")
    curve.set_title("distance ahead for every row below the horizon")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def perspective_grid(path):
    _, _, frame = _view("straight", WORKED[1])
    fig = Figure(figsize=(9, 5.2))
    ax = fig.subplots()
    ax.imshow(frame)
    ahead = np.linspace(2.0, 200.0, 400)
    for left in range(-6, 5):
        ax.plot(*CAMERA.ground_to_pixel(ahead, np.full_like(ahead, left)), color=CHROME["text"], lw=0.6)
    for forward in (3, 5, 10, 20, 40):
        v = CAMERA.ground_to_pixel(forward, 0.0)[1]
        ax.axhline(v, color=QUANTITY["perception"], lw=1.5)
        ax.text(4, v, f"{forward} m ahead", color=QUANTITY["perception"], fontsize=8, va="center",
                bbox={"facecolor": CHROME["panel"], "linewidth": 0, "pad": 1})
    ax.set_xlim(0, CAMERA.width - 1)
    ax.set_ylim(CAMERA.height - 1, 0)
    ax.set_title("grey lines: 1 m apart on the ground; red lines: fixed distances ahead")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def lane_estimate(path):
    scenario, s, offset, heading = WORKED
    road, state, frame = _view(*WORKED)
    masks = paint_masks(frame)
    fig = Figure(figsize=(13, 4.6))
    image, top = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.5, 1]})
    image.imshow(frame)
    for (a, b), name in BANDS:
        for edge in (a - 0.5, b - 0.5):
            image.axhline(edge, color=QUANTITY["psi"], lw=1.5, ls="--")
        image.text(3, a + 1, name, va="top", fontsize=9, bbox={"facecolor": CHROME["panel"], "linewidth": 0, "pad": 1})
    _top_down(top, "the same paint, back-projected onto the ground")
    points = []
    for rows, _ in BANDS:
        for mask, paint in zip(masks, (YELLOW, WHITE)):
            v, u = np.nonzero(mask[slice(*rows)])
            v = v + rows[0]
            forward, left = CAMERA.pixel_to_ground(u, v)
            top.scatter(left, forward, s=2, color=PAINT[paint])
            image.plot(u.mean(), v.mean(), "x", ms=10, mew=2, color=QUANTITY["perception"])
            top.plot(*CAMERA.pixel_to_ground(u.mean(), v.mean())[::-1], "x", ms=10, mew=2, color=QUANTITY["perception"])
        points.append(lane_point(masks, rows))
    (f1, c1), (f2, c2) = points
    slope = (c2 - c1) / (f2 - f1)
    ahead = np.array([0.0, 9.0])
    top.plot(c1 + slope * (ahead - f1), ahead, color=CHROME["text"], lw=1.5, label="line through the two lane centres")
    top.plot([c1, c2], [f1, f2], "o", ms=8, mfc=CHROME["text"], mec=CHROME["background"], label="lane centre in each band")
    _road_line(top, road, state, s, EGO_LANE_CENTER, 10, ls=":", color=QUANTITY["e_off"], lw=2, label="true lane centre")
    intercept = c1 - slope * f1
    top.annotate("", xy=(intercept, -0.6), xytext=(0, -0.6), arrowprops={"arrowstyle": "<->", "color": CHROME["text"]})
    top.text(intercept - 0.1, -0.6, f"offset = {-intercept:.2f} m", ha="left", va="center", color=CHROME["text"], fontsize=9)
    top.set_xlim(2.5, -3.5)
    top.set_ylim(-1.1, 9)
    top.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8)
    image.set_title(f"offset {offset} m, heading {heading} rad: red crosses mark each line's centroid in a band")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def chord_bias(path):
    fig = Figure(figsize=(12, 5.4))
    for ax, (scenario, s) in zip(fig.subplots(1, 2, sharey=True), BENDS):
        road, state, frame = _view(scenario, s)
        masks = paint_masks(frame)
        (f1, c1), (f2, c2) = (lane_point(masks, rows) for rows, _ in BANDS)
        est = estimate_lane(Observation(frame, TARGET_SPEED, 0.0))
        radius = 1 / abs(np.diff(road.pose_at(np.array([s, s + 1.0]))[2])[0])
        _top_down(ax, f"{scenario}, radius {radius:.0f} m: estimate {est.offset_m:+.2f} m, {est.heading_rad:+.3f} rad")
        for lateral, paint in ((0.0, YELLOW), (WHITE_LINE_LATERAL, WHITE)):
            _road_line(ax, road, state, s, lateral, 14, color=PAINT[paint], lw=3)
        _road_line(ax, road, state, s, EGO_LANE_CENTER, 14, ls=":", color=QUANTITY["e_off"], lw=2, label="true lane centre (a curve)")
        ax.plot([0, 0], [0, 14], ls="--", color=QUANTITY["perception"], lw=1, label="where the car points: along the lane")
        ahead = np.array([0.0, 14.0])
        ax.plot(c1 + (c2 - c1) / (f2 - f1) * (ahead - f1), ahead, color=CHROME["text"], lw=1.5, label="straight line through the two centres")
        ax.plot([c1, c2], [f1, f2], "o", ms=8, mfc=CHROME["text"], mec=CHROME["background"])
        ax.set_xlim(5, -5)
        ax.set_ylim(-0.3, 14)
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("the car is exactly on lane centre and aligned with it (truth: 0 m, 0 rad)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def make(out_dir):
    rows_to_distance(out_dir / "rows_to_distance.png")
    perspective_grid(out_dir / "perspective_grid.png")
    lane_estimate(out_dir / "lane_estimate.png")
    chord_bias(out_dir / "chord_bias.png")
