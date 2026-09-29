"""Figures for lecture 0.1.1 Meet your car.  `python scripts/make_figures.py course/0.1.1-meet-your-car`."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from zero2fsd.car import Car, provided, run
from zero2fsd.score import OFF_LANE_LAT
from zero2fsd.sim import BicycleState, Camera, make_scenario, render_topdown
from zero2fsd.sim.scenarios import TARGET_SPEED
from zero2fsd.sim.world import LANE_WIDTH

VIEW_S = 80.0  # m along `gentle`: 20 m before the first bend, so the bend is in view
STIFF_GAINS = (0.7, 2.7)  # a car whose offset gain is twice the provided one (gains are unit 0.1.4)


def _car_and_view(path):
    road = make_scenario("gentle").road
    frame = Camera().render(road, BicycleState(*road.ego_pose(VIEW_S), TARGET_SPEED, 0.0))
    tel = run(Car(), "gentle")
    fig = Figure(figsize=(11, 4))
    cam, top = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.6, 1]})
    cam.imshow(frame)
    cam.set_title(f"What the camera sees, {VIEW_S:.0f} m into 'gentle'")
    cam.set_xlabel("pixel column")
    cam.set_ylabel("pixel row")
    render_topdown(road, [np.column_stack([tel.x, tel.y])], ["the provided car"], ax=top)
    top.plot(*road.offset(VIEW_S, -LANE_WIDTH / 2), "o", color="C3", label="camera position")
    top.legend(loc="center")
    top.set_title("Where the car drove (top-down)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def _sign_convention(path):
    fig = Figure(figsize=(8, 3.2))
    ax = fig.subplots()
    half = LANE_WIDTH / 2
    ax.axhline(half, color="#c9a800", linestyle=(0, (6, 3)), linewidth=3)
    ax.axhline(-half, color="0.6", linewidth=3)
    ax.axhline(0.0, color="0.4", linestyle=":", linewidth=1)
    ax.text(0.2, half + 0.12, "yellow centre line (left edge of your lane)", fontsize=9)
    ax.text(0.2, -half - 0.3, "white edge line (right edge of your lane)", fontsize=9)
    ax.text(0.2, 0.08, "lane centre, e = 0", fontsize=9, color="0.3")
    for x, e, color in ((4.0, 0.5, "C0"), (8.5, -0.3, "C1")):
        ax.add_patch(Rectangle((x - 2.2, e - 0.8), 2.7, 1.6, facecolor=color, alpha=0.35, edgecolor=color))
        ax.annotate("", xy=(x, e), xytext=(x, 0.0), arrowprops={"arrowstyle": "->", "color": color, "lw": 2})
        ax.plot(x, e, "o", color=color)
        ax.text(x + 0.2, e / 2, f"e = {e:+.1f} m", color=color, fontsize=11, va="center")
    ax.annotate("", xy=(11.8, -1.25), xytext=(10.2, -1.25), arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.text(10.1, -1.25, "direction of travel", ha="right", va="center", fontsize=9)
    ax.set(xlim=(0, 12), ylim=(-2.3, 2.3), xlabel="metres along the road", ylabel="metres left of lane centre")
    ax.set_title("Lateral error e, measured at the front axle (dot): left of centre is positive")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def _signed_mean_trap(path):
    runs = [
        (run(Car(), "gentle"), "provided car on 'gentle'"),
        (run(Car(controller=lambda est, obs: provided.steer_p(est, *STIFF_GAINS)), "curvy"),
         f"stiffer car (k_off={STIFF_GAINS[0]}) on 'curvy'"),
    ]
    fig = Figure(figsize=(11, 4))
    for ax, (tel, name) in zip(fig.subplots(1, 2, sharey=True), runs):
        signed, mean_abs = tel.lat.mean(), np.abs(tel.lat).mean()
        ax.plot(tel.progress_m, tel.lat, color="0.3", linewidth=1)
        ax.axhline(signed, color="C3", linewidth=2, label=f"signed mean = {signed:+.2f} m")
        ax.axhline(mean_abs, color="C0", linewidth=2, linestyle="--", label=f"mean |e| = {mean_abs:.2f} m")
        for edge in (-OFF_LANE_LAT, OFF_LANE_LAT):
            ax.axhline(edge, color="0.6", linestyle=":", linewidth=1)
        ax.set(title=name, xlabel="distance along the route (m)")
        ax.legend(loc="lower left", fontsize=9)
    fig.axes[0].set_ylabel("lateral error e (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def make(out_dir):
    _car_and_view(out_dir / "car_and_view.png")
    _sign_convention(out_dir / "sign_convention.png")
    _signed_mean_trap(out_dir / "signed_mean_trap.png")
