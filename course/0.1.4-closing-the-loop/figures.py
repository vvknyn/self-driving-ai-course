"""Figures for lecture 0.1.4 Closing the loop.  `python scripts/make_figures.py course/0.1.4-closing-the-loop`."""
import math

import numpy as np
from matplotlib.figure import Figure

from zero2fsd.car import Car, LaneEstimate, provided, run
from zero2fsd.car.runner import OFF_ROAD_OFFSET
from zero2fsd.car.types import Action
from zero2fsd.score import metrics
from zero2fsd.sim import BicycleState, make_scenario
from zero2fsd.sim.dynamics import DT, WHEELBASE, step
from zero2fsd.sim.scenarios import GENTLE_RADIUS
from zero2fsd.sim.world import EGO_LANE_CENTER

K_OFF, K_HEAD = provided.K_OFF, provided.K_HEAD
BIAS = 0.002  # rad added to every steering command: the lecture's "wheel slightly off"
SWEEP = (0.05, 0.35, 1.5, 6.0)  # k_off values on `gentle`, with k_head = K_HEAD
CURVY_FIRST_ARC, CURVY_ARC = 30.0, math.pi / 2 * 20.0  # `curvy`: 30 m straight, then quarter circles of radius 20 m
CURVY_RADIUS = 20.0  # m, the centreline radius of every `curvy` bend
GENTLE_FIRST_BEND = 100.0  # m: `gentle` starts with a 100 m straight


def steady_offset(radius, turn, k_off=K_OFF, k_head=K_HEAD):
    """Predicted settled offset in a bend of ego-lane `radius`, `turn` +1 left or -1 right ("Why curves need error")."""
    return turn * math.atan(WHEELBASE / radius) * (k_head - 1) / k_off


def _lane_radius(radius, turn):
    """The ego lane sits EGO_LANE_CENTER (negative: right) of the centreline, so it is outside a left bend."""
    return radius - turn * EGO_LANE_CENTER


def _shade_curvy_bends(ax):
    for k in range(6):
        a = CURVY_FIRST_ARC + k * CURVY_ARC
        ax.axvspan(a, a + CURVY_ARC, color="tab:green" if k % 2 == 0 else "tab:purple", alpha=0.08)
    ax.text(CURVY_FIRST_ARC + 1, 1.45, "green: left bend\npurple: right bend", fontsize=8)


def _gains(k_off, k_head):
    return lambda est, obs: provided.steer_p(est, k_off, k_head)


class _PerfectEyes:
    """Perception that reads the true offset and heading, for the lecture's comparison only.

    A perception box is only given the camera frame, so this one keeps its own copy of the car's state: the runner's
    start (ego-lane centre at `start_s`, target speed, wheel straight), and every command replayed through the
    simulator's `step`.  `perfect_run` checks that the copy matched the runner's truth at every step.
    """

    def __init__(self, scenario, k_off, k_head):
        sc = make_scenario(scenario)
        self.road, self.gains = sc.road, (k_off, k_head)
        self.state = BicycleState(*sc.road.ego_pose(sc.start_s), sc.target_speed, 0.0)

    def estimate(self, obs):
        _, offset, heading = self.road.ego_truth(self.state.x, self.state.y, self.state.yaw)
        return LaneEstimate(offset, heading, True)

    def act(self, est, obs):
        action = Action(provided.steer_p(est, *self.gains), provided.hold_speed(obs.speed))
        self.state = step(self.state, action.steer_rad, action.accel)
        return action


def perfect_run(scenario, k_off=K_OFF, k_head=K_HEAD):
    eyes = _PerfectEyes(scenario, k_off, k_head)
    tel = run(Car(perception=eyes, controller=eyes), scenario)
    if not (np.array_equal(tel.est_offset, tel.lat) and np.array_equal(tel.est_heading, tel.head_err)):
        raise RuntimeError("the perfect-perception copy of the car's state drifted from the runner's truth")
    return tel


def _ends(tel):
    return "completes" if tel.completed else f"leaves the road at {tel.progress_m[-1]:.1f} m, t = {tel.t[-1]:.1f} s"


def _road_edges(ax):
    for edge in (-OFF_ROAD_OFFSET, OFF_ROAD_OFFSET):
        ax.axhline(edge, color="0.5", ls="--", lw=1)


def open_vs_closed(path):
    closed = run(Car(), "gentle")
    plan = closed.steer  # the wheel angle at each step; the command given at step k is the angle at step k + 1

    def replay(bias):
        return lambda est, obs: float(plan[min(round(obs.t / DT) + 1, len(plan) - 1)] + bias)

    if not np.array_equal(run(Car(controller=replay(0.0)), "gentle").lat, closed.lat):
        raise RuntimeError("replaying the recorded steering did not reproduce the closed-loop run")
    runs = (
        (closed, "closed loop, the provided car"),
        (run(Car(controller=lambda est, obs: provided.controller(est, obs) + BIAS), "gentle"), f"closed loop, wheel {BIAS} rad off"),
        (run(Car(controller=replay(BIAS)), "gentle"), f"open loop: the same steering replayed, {BIAS} rad off"),
    )
    fig = Figure(figsize=(10, 4.2))
    ax = fig.subplots()
    for (tel, label), style in zip(runs, ("-", ":", "-")):
        ax.plot(tel.t, tel.lat, ls=style, lw=2, label=f"{label}: {_ends(tel)}")
    _road_edges(ax)
    ax.set_xlim(0, 20)
    ax.set(xlabel="time (s)", ylabel="offset from lane centre (m), left +",
           title=f"gentle: a steering error of {BIAS} rad ({math.degrees(BIAS):.2f} degrees)")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def gain_sweep(path):
    runs = {k: run(Car(controller=_gains(k, K_HEAD)), "gentle") for k in SWEEP}
    fig = Figure(figsize=(12, 4.4))
    lat, steer = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.5, 1]})
    for i, (k, tel) in enumerate(runs.items()):
        lat.plot(tel.t, tel.lat, color=f"C{i}", label=f"k_off = {k}: {_ends(tel)}, mean |lat| {metrics(tel)['mean_abs_lat']:.3f} m")
    _road_edges(lat)
    bend = runs[K_OFF].t[np.argmax(runs[K_OFF].progress_m >= GENTLE_FIRST_BEND)]
    lat.axvline(bend, color="0.7", lw=1)
    lat.text(bend + 0.2, 1.6, "first bend", color="0.4", fontsize=9)
    lat.set_xlim(0, 20)
    lat.set(xlabel="time (s)", ylabel="offset from lane centre (m), left +", title=f"gentle, k_head = {K_HEAD}")
    lat.legend(loc="lower right", fontsize=8)
    for i, k in ((2, 1.5), (3, 6.0)):
        tel = runs[k]
        limited = np.mean(np.abs(np.diff(tel.steer)) / DT > 0.999)
        steer.plot(tel.t, tel.steer, color=f"C{i}", label=f"k_off = {k}: wheel turning at full speed in {limited:.0%} of steps")
    steer.set_xlim(0, 4.5)
    steer.set(xlabel="time (s)", ylabel="wheel angle (rad), left +", title="the wheel: at most 1 rad/s")
    steer.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def steady_state(path):
    gentle, curvy = perfect_run("gentle"), perfect_run("curvy")
    fig = Figure(figsize=(12, 4.2))
    left, right = fig.subplots(1, 2)
    left.plot(gentle.progress_m, gentle.lat, color="C0", label=f"perfect lane values: settles at {np.abs(gentle.lat).max():.3f} m")
    predicted = steady_offset(_lane_radius(GENTLE_RADIUS, +1), +1)
    left.axhline(predicted, color="black", ls=":", label=f"predicted: {predicted:.3f} m")
    left.set_ylim(-0.1, 0.3)
    left.set(xlabel="distance along the road (m)", ylabel="offset from lane centre (m), left +",
             title=f"gentle: two left bends of radius {_lane_radius(GENTLE_RADIUS, +1):.1f} m")
    left.legend(loc="upper left", fontsize=8)
    _shade_curvy_bends(right)
    right.plot(curvy.progress_m, curvy.lat, color="C0", label=f"perfect lane values: max |lat| {metrics(curvy)['max_abs_lat']:.2f} m")
    for turn, colour in ((+1, "tab:green"), (-1, "tab:purple")):
        radius = _lane_radius(CURVY_RADIUS, turn)
        right.axhline(steady_offset(radius, turn), color=colour, ls=":",
                      label=f"predicted for radius {radius:.1f} m: {steady_offset(radius, turn):+.3f} m")
    right.set_ylim(-1.9, 1.9)
    right.set(xlabel="distance along the road (m)", ylabel="offset from lane centre (m), left +", title="curvy: the bend changes direction every 31 m")
    right.legend(loc="lower left", fontsize=8)
    fig.suptitle(f"the P law with perfect lane values (k_off = {K_OFF}, k_head = {K_HEAD})")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def curvy_box(path):
    perfect, box = perfect_run("curvy"), run(Car(), "curvy")
    fig = Figure(figsize=(12, 4.4))
    lat, head = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.5, 1]})
    _shade_curvy_bends(lat)
    lat.plot(perfect.progress_m, perfect.lat, color="C0", label=f"perfect lane values: {_ends(perfect)}")
    lat.plot(box.progress_m, box.lat, color="C3", label=f"the two-point box: {_ends(box)}")
    _road_edges(lat)
    lat.set(xlabel="distance along the road (m)", ylabel="offset from lane centre (m), left +",
            title=f"curvy, the same P law (k_off = {K_OFF}, k_head = {K_HEAD})")
    lat.legend(loc="lower left", fontsize=8)
    head.axvspan(CURVY_FIRST_ARC, box.progress_m[-1], color="tab:green", alpha=0.08)
    head.plot(box.progress_m, box.head_err, color="black", label="true heading")
    head.plot(box.progress_m, box.est_heading, color="C3", label="what the box reported")
    head.set(xlabel="distance along the road (m)", ylabel="heading (rad), left +", title="the two-point box's run: heading")
    head.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def make(out_dir):
    open_vs_closed(out_dir / "open_vs_closed.png")
    gain_sweep(out_dir / "gain_sweep.png")
    steady_state(out_dir / "steady_state.png")
    curvy_box(out_dir / "curvy_box.png")
