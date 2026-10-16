"""Figures for lecture 0.1.4 Closing the loop.  `python scripts/make_figures.py course/0.1.4-closing-the-loop`.

Every figure is 5 inches wide and every text is 14 pt, so a digit is 9 px tall or more when the PNG is shown 330 px wide.
"""
import math
import sys
import types
from pathlib import Path

import numpy as np
import yaml
from figstyle import render_pool
from matplotlib import rc_context
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgb, to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Arc, Polygon, Rectangle

from zero2fsd.car import Car, LaneEstimate, provided, run
from zero2fsd.car.runner import OFF_ROAD_OFFSET
from zero2fsd.car.types import Action, Observation
from zero2fsd.palette import CHROME, QUANTITY, SCENE, shades
from zero2fsd.score import metrics
from zero2fsd.sim import BicycleState, Camera, make_scenario
from zero2fsd.sim.dynamics import DT, MAX_STEER, WHEELBASE, step
from zero2fsd.sim.scenarios import GENTLE_RADIUS, TARGET_SPEED
from zero2fsd.sim.world import EGO_LANE_CENTER

W, FS = 5.0, 14  # canvas width in inches, text size in points: 0.729 * FS * 330 / (72 * W) = 9.4 px per digit at 330 px wide
K_OFF, K_HEAD = provided.K_OFF, provided.K_HEAD
BIAS = 0.002  # rad added to every steering command: the lecture's "wheel slightly off"
PUSH_STEPS, PUSH_STEER = 20, 0.025  # the push test: the wheels held 0.025 rad left for 1 s, then the law takes over
GENTLE_BEND = math.pi * GENTLE_RADIUS  # gentle: 100 m straight, bend, 100 m straight, bend
GENTLE_BENDS = tuple((100.0 + k * (100.0 + GENTLE_BEND), 100.0 + k * (100.0 + GENTLE_BEND) + GENTLE_BEND) for k in range(2))
GENTLE_BEND_MIDDLES = ((150.0, 300.0), (500.0, 650.0))  # progress windows well inside gentle's two bends
CURVY_START, CURVY_ARC = 30.0, math.pi / 2 * 20.0  # curvy: 30 m straight, then six quarter circles of radius 20 m, left first
K_OFFS = (0.05, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)  # the tuning map: across
K_HEADS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.7, 3.5)  # and up
YELLOW, TEXT, PANEL, BORDER, BACKGROUND = SCENE["yellow"], CHROME["text"], CHROME["panel"], CHROME["border"], CHROME["background"]
BLUE, PINK, GREEN, RED = QUANTITY["e_off"], QUANTITY["psi"], QUANTITY["delta"], QUANTITY["perception"]
LABEL_BOX = {"boxstyle": "round,pad=0.2", "fc": PANEL, "ec": BORDER}  # text that sits on top of a picture


def gains(k_off, k_head):
    return lambda est, obs: provided.steer_p(est, k_off, k_head)


def box_run(scenario, law, **kw):
    """The provided perception box with `law` as the controller."""
    return run(Car(controller=law), scenario, **kw)


class Perfect:
    """Perception that reads the true offset and heading, by replaying every command through the simulator's `step`."""

    def __init__(self, scenario, law):
        sc = make_scenario(scenario)
        self.road, self.law = sc.road, law
        self.state = BicycleState(*sc.road.ego_pose(sc.start_s), sc.target_speed, 0.0)

    def estimate(self, obs):
        _, offset, heading = self.road.ego_truth(self.state.x, self.state.y, self.state.yaw)
        return LaneEstimate(offset, heading, True)

    def act(self, est, obs):
        action = Action(float(self.law(est, obs)), provided.hold_speed(obs.speed))
        self.state = step(self.state, action.steer_rad, action.accel)
        return action


def perfect_run(scenario, law):
    p = Perfect(scenario, law)
    tel = run(Car(perception=p, controller=p), scenario)
    if not (np.array_equal(tel.est_offset, tel.lat) and np.array_equal(tel.est_heading, tel.head_err)):
        raise RuntimeError("the perfect-perception copy of the car's state drifted from the runner's truth")
    return tel


def pushed(k_off, k_head):
    """The push test on gentle's first straight: 12 s, the wheels held at PUSH_STEER for the first PUSH_STEPS steps."""
    law = lambda est, obs: PUSH_STEER if obs.t < PUSH_STEPS * DT - 1e-9 else provided.steer_p(est, k_off, k_head)
    return box_run("gentle", law, max_steps=240)


def in_windows(tel, windows):
    keep = np.zeros(len(tel.t), bool)
    for a, b in windows:
        keep |= (tel.progress_m >= a) & (tel.progress_m <= b)
    return keep


def bend_radius(radius, turn):
    """The ego lane sits EGO_LANE_CENTER (negative: right) of the centreline, so it is outside a left bend."""
    return radius - turn * EGO_LANE_CENTER


def hold_angle(radius, turn=+1):
    """The steering angle that holds a bend of ego-lane `radius`: arcsin(L / R), left positive."""
    return turn * math.asin(WHEELBASE / radius)


def titles():
    """{id: title} for every loop, module and unit of the course map."""
    found = {}

    def walk(node):
        if isinstance(node, dict):
            if "id" in node and "title" in node:
                found[str(node["id"])] = node["title"]
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(yaml.safe_load((Path(__file__).resolve().parents[1] / "syllabus.yaml").read_text()))
    return found


# ---------------------------------------------------------------------------------------------------- drawing helpers

def canvas(height, *args, **kw):
    fig = Figure(figsize=(W, height), layout="constrained")
    return fig, fig.subplots(*args, **kw)


def save(fig, path):
    fig.savefig(path)


def minus(text):
    """Numbers formatted with a hyphen read as a dash; the axis tick labels already use the real minus sign."""
    return text.replace("-", "−")


def label(ax, x, y, text, **kw):
    """`ax.text` with real minus signs."""
    return ax.text(x, y, minus(text), **kw)


def note(ax, text, xy, off=(0, 0), colour=TEXT, line=False, **kw):
    """Text on the picture, `off` points from the data point `xy`, with a thin line back to it if `line`."""
    props = {"arrowstyle": "-", "color": TEXT, "lw": 1, "shrinkA": 0, "shrinkB": 4} if line else None
    return ax.annotate(minus(text), xy, off, textcoords="offset points", color=colour, arrowprops=props, **kw)


def corner(ax, text, x, y, colour=TEXT, **kw):
    return label(ax, x, y, text, transform=ax.transAxes, color=colour, **kw)


def dot(ax, x, y, colour=TEXT, size=9):
    ax.plot([x], [y], "o", color=colour, ms=size, mec=BACKGROUND, mew=1.5, zorder=5)


def cross(ax, x, y):
    ax.plot([x], [y], "x", color=TEXT, ms=13, mew=3, zorder=5)


def yellow_line(ax, y=OFF_ROAD_OFFSET):
    ax.axhline(y, color=YELLOW, lw=2.5, zorder=1)


def shade_bends(ax, spans, alpha=0.12):
    for a, b in spans:
        ax.axvspan(a, b, color=TEXT, alpha=alpha, lw=0)


def curvy_spans(turn):
    """Progress spans of curvy's left (+1) or right (-1) bends."""
    return [(CURVY_START + k * CURVY_ARC, CURVY_START + (k + 1) * CURVY_ARC) for k in range(6) if (-1) ** k == turn]


def finish(ax, xlabel=None, ylabel=None):
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.margins(x=0.02, y=0.08)


def blank(ax, xlim, ylim):
    """An axes for a drawing: no ticks, equal scale."""
    ax.set(xlim=xlim, ylim=ylim, aspect="equal")
    ax.axis("off")


def tag(ax, text, xy, colour=TEXT, **kw):
    """A boxed label on a schematic."""
    return ax.text(*xy, text, color=colour, ha="center", va="center", bbox={"boxstyle": "round,pad=0.35", "fc": PANEL, "ec": colour, "lw": 1.5}, **kw)


def link(ax, a, b, **kw):
    """An arrow from `a` to `b` on a schematic."""
    ax.annotate("", b, a, arrowprops={"arrowstyle": "-|>", "color": TEXT, "lw": 2, "shrinkA": 0, "shrinkB": 0, **kw})


# --------------------------------------------------------------------------------------------------------- the figures

def result_first(path):
    fig, axes = canvas(6.4, 2, 1)
    runs = [((0.35, 2.7), shades(BLUE, 2)[0]), ((0.5, 1.0), shades(BLUE, 2)[1])]
    for ax, scenario in zip(axes, ("gentle", "curvy")):
        tels = [box_run(scenario, gains(*g)) for g, _ in runs]
        for (g, colour), tel in zip(runs, tels):
            ax.plot(tel.progress_m, tel.lat, color=colour, lw=2.5)
        yellow_line(ax)
        corner(ax, scenario, 0.03, 0.9, weight="bold", va="top")
        finish(ax, "distance along the road (m)", "e (m)")
    gentle, curvy = (box_run(s, gains(*runs[0][0])) for s in ("gentle", "curvy"))
    note(axes[0], "0.35, 2.7", (560, gentle.lat.max()), (0, 6), runs[0][1], ha="center", va="bottom")
    note(axes[0], "0.5, 1.0", (20, 0.0), (0, 6), runs[1][1], ha="left", va="bottom")
    cross(axes[1], curvy.progress_m[-1], curvy.lat[-1])
    note(axes[1], f"leaves at\n{curvy.progress_m[-1]:.1f} m", (curvy.progress_m[-1], curvy.lat[-1]), (12, -6), ha="left", va="top")
    note(axes[1], "0.35, 2.7", (62, 0.85), colour=runs[0][1], ha="left", va="center")
    note(axes[1], "0.5, 1.0", (125, 0.25), (0, 4), runs[1][1], ha="center", va="bottom")
    save(fig, path)


def open_vs_closed(path):
    closed_plain = box_run("gentle", provided.controller)
    plan = closed_plain.steer  # the wheel angle at each step; the command given at step k is the angle at step k + 1

    def replay(bias):
        return lambda est, obs: float(plan[min(round(obs.t / DT) + 1, len(plan) - 1)] + bias)

    if not np.array_equal(box_run("gentle", replay(0.0)).lat, closed_plain.lat):
        raise RuntimeError("replaying the recorded steering did not reproduce the closed-loop run")
    open_loop = box_run("gentle", replay(BIAS))
    closed = box_run("gentle", lambda est, obs: provided.controller(est, obs) + BIAS)
    window = closed.t <= open_loop.t[-1]
    fig, ax = canvas(4.4)
    dark, light = shades(BLUE, 2)
    ax.plot(open_loop.t, open_loop.lat, color=dark, lw=2.5)
    ax.plot(closed.t[window], closed.lat[window], color=light, lw=2.5)
    yellow_line(ax)
    corner(ax, "yellow line, 1.8 m", 0.03, 0.89, colour=YELLOW, va="top")
    corner(ax, "open loop:\nreplays a recording", 0.03, 0.72, colour=dark, va="top")
    corner(ax, "closed loop:\nsteers from its camera", 0.97, 0.1, colour=light, ha="right", va="bottom")
    finish(ax, "time (s)", "e (m)")
    save(fig, path)


def loop_kinds(path):
    fig, ax = canvas(6.6)
    blank(ax, (0, 10), (0, 13.2))
    ax.text(5, 12.6, "Open loop", ha="center", va="center", weight="bold")
    tag(ax, "recorded\nsteering angles", (2.7, 10.9), GREEN)
    tag(ax, "car", (7.6, 10.9))
    link(ax, (4.8, 10.9), (7.0, 10.9))
    ax.text(5.9, 9.9, "nothing comes back", ha="center", va="top", color=TEXT)
    ax.text(5, 8.3, "Closed loop", ha="center", va="center", weight="bold")
    boxes = [("camera picture", TEXT), ("perception box", RED), ("offset, heading", RED), ("controller", TEXT), ("steering angle", GREEN), ("car", TEXT)]
    ys = np.linspace(7.1, 0.5, len(boxes))
    for (text, colour), y in zip(boxes, ys):
        tag(ax, text, (3.6, y), colour)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        link(ax, (3.6, y0 - 0.45), (3.6, y1 + 0.45))
    ax.plot([4.3, 8.7, 8.7], [ys[-1], ys[-1], ys[0]], color=TEXT, lw=2)
    link(ax, (8.7, ys[0]), (5.3, ys[0]))
    ax.text(9.1, ys.mean(), "new position", rotation=90, ha="center", va="center")
    save(fig, path)


def offset_only(path):
    tel = pushed(K_OFF, 0.0)
    fig, ax = canvas(4.8)
    ax.axvspan(0, 1.0, color=TEXT, alpha=0.12, lw=0)
    ax.plot(tel.t, tel.lat, color=BLUE, lw=2.5)
    after = tel.t >= PUSH_STEPS * DT - 1e-9
    cross_i = np.flatnonzero(after[:-1] & (np.sign(tel.lat[:-1]) != np.sign(tel.lat[1:])) & (np.abs(tel.lat[:-1]) > 1e-3))[0] + 1
    trough = int(np.argmin(tel.lat))
    dot(ax, tel.t[PUSH_STEPS], tel.lat[PUSH_STEPS])
    dot(ax, tel.t[cross_i], tel.lat[cross_i])
    dot(ax, tel.t[trough], tel.lat[trough])
    note(ax, f"handover:\n{tel.lat[PUSH_STEPS]:.2f} m left", (tel.t[PUSH_STEPS], tel.lat[PUSH_STEPS]), (20, 14), line=True, va="center")
    note(ax, f"first step past the centre,\n{tel.t[cross_i]:.2f} s, pointing\n{abs(tel.head_err[cross_i]):.3f} rad right", (tel.t[cross_i], tel.lat[cross_i]), (40, 70), line=True, va="center")
    note(ax, f"{abs(tel.lat[trough]):.3f} m past", (tel.t[trough], tel.lat[trough]), (30, -4), line=True, va="center")
    corner(ax, "push: 0.025 rad\nleft for 1 s", 0.97, 0.8, ha="right", va="top")
    finish(ax, "time (s)", "e (m)")
    save(fig, path)


def p_law(path):
    x = np.linspace(-2, 2, 401)
    fig, ax = canvas(4.6)
    colours = shades(GREEN, 2)
    for heading, colour in zip((0.0, 0.05), colours):
        ax.plot(x, [provided.steer_p(LaneEstimate(e, heading, True)) for e in x], color=colour, lw=2.5)
    for edge in (MAX_STEER, -MAX_STEER):
        ax.axhline(edge, color=TEXT, ls="--", lw=1.5)
    corner(ax, "+0.5 rad: the\nwheels' limit", 0.97, 0.93, ha="right", va="top")
    corner(ax, "−0.5 rad", 0.03, 0.11, va="bottom")
    flat, tilt = (provided.steer_p(LaneEstimate(0.4, h, True)) for h in (0.0, 0.05))
    dot(ax, 0.4, flat, colours[0])
    dot(ax, 0.4, tilt, colours[1])
    note(ax, f"{flat:.2f} rad", (0.4, flat), (30, 40), colours[0], line=True, va="center")
    note(ax, f"{tilt:.3f} rad", (0.4, tilt), (45, 0), colours[1], line=True, va="center")
    corner(ax, "heading 0", 0.02, 0.62, colour=colours[0], va="center")
    corner(ax, "heading\n0.05 rad left", 0.02, 0.42, colour=colours[1], va="center")
    corner(ax, f"slope {-K_OFF} rad\nper m", 0.62, 0.64, va="center")
    finish(ax, "offset e (m), left +", "steering (rad)")
    save(fig, path)


def push_family(path):
    heads = (0.0, 0.5, 1.0, 2.0, 2.7, 5.0)
    fig, ax = canvas(5.2)
    ax.axhspan(-0.05, 0.05, color=TEXT, alpha=0.12, lw=0)
    corner(ax, "grey band: within 5 cm", 0.97, 0.95, ha="right", va="top")
    for k, colour, label_y in zip(heads, shades(BLUE, len(heads))[::-1], (-0.125, -0.06, 0.13, 0.20, 0.27, 0.34)):
        tel = pushed(K_OFF, k)
        ax.plot(tel.t, tel.lat, color=colour, lw=2.5)
        i = round(2.0 / DT)  # every curve is told apart at 2 s, a second after the handover
        ax.plot([tel.t[i], 5.9], [tel.lat[i], label_y], color=colour, lw=1)
        ax.text(6.0, label_y, f"heading gain {k:g}", color=colour, va="center")
    finish(ax, "time (s)", "e (m)")
    save(fig, path)


def wheel_vs_lane(path):
    """The push test's handover seen from above: the car's offset, its heading psi, and the wheel angle delta the law then chooses."""
    big = 3  # angles are drawn 3 times larger
    tel = pushed(K_OFF, 1.0)
    e, psi = float(tel.lat[PUSH_STEPS]), float(tel.head_err[PUSH_STEPS])
    delta = provided.steer_p(LaneEstimate(e, psi, True), K_OFF, 1.0)
    fig, ax = canvas(5.8)
    blank(ax, (-3.6, 3.6), (-2.8, 6.6))
    ax.plot([0, 0], [-1.0, 6.2], color=TEXT, ls=(0, (6, 5)), lw=2)
    ax.text(0.12, 6.2, "lane centre", ha="left", va="top")
    unit = lambda th: np.array([math.cos(th), math.sin(th)])
    theta_b, theta_w = math.pi / 2 + big * psi, math.pi / 2 + big * (psi + delta)
    front = np.array([-e, WHEELBASE])  # e is the front axle's offset, so the front axle sits e left of the lane centre on the page
    rear = front - WHEELBASE * unit(theta_b)
    perp = np.array([-math.sin(theta_b), math.cos(theta_b)])
    corners = [rear - 0.5 * unit(theta_b) + 0.9 * perp, front + 0.6 * unit(theta_b) + 0.9 * perp, front + 0.6 * unit(theta_b) - 0.9 * perp, rear - 0.5 * unit(theta_b) - 0.9 * perp]
    ax.add_patch(Polygon(corners, closed=True, fc=PANEL, ec=TEXT, lw=2, zorder=2))
    for sign in (+1, -1):
        centre = front + sign * 0.9 * perp
        ax.plot(*np.array([centre - 0.45 * unit(theta_w), centre + 0.45 * unit(theta_w)]).T, color=TEXT, lw=6, solid_capstyle="butt", zorder=3)
    ax.plot(*np.array([rear, front]).T, color=TEXT, lw=1.5, zorder=3)
    ax.plot(*np.array([front, front + 3.0 * unit(theta_b)]).T, color=TEXT, ls=":", lw=1.5)
    ax.plot(*np.array([front, front + 3.0 * unit(math.pi / 2)]).T, color=TEXT, ls=":", lw=1.5)
    ax.annotate("", front + 1.9 * unit(theta_w), front, arrowprops={"arrowstyle": "-|>", "color": GREEN, "lw": 2.5}, zorder=4)
    deg = math.degrees
    ax.add_patch(Arc(front, 5.2, 5.2, theta1=deg(math.pi / 2), theta2=deg(theta_b), color=PINK, lw=3, zorder=4))
    ax.add_patch(Arc(front, 2.0, 2.0, theta1=deg(theta_w), theta2=deg(theta_b), color=GREEN, lw=3, zorder=4))
    ax.add_patch(Arc(front, 3.6, 3.6, theta1=deg(theta_w), theta2=deg(math.pi / 2), color=TEXT, lw=2, zorder=4))
    label(ax, front[0] - 0.6, front[1] + 2.9, f"ψ = {psi:.3f} rad", color=PINK, ha="right", va="center")
    label(ax, front[0] + 1.35, front[1] + 1.25, f"δ = {delta:.3f} rad", color=GREEN, ha="left", va="center")
    label(ax, front[0] + 1.35, front[1] + 2.45, f"ψ + δ = {psi + delta:.2f} rad", color=TEXT, ha="left", va="center")
    ax.plot([-e, -e], [front[1], -1.0], color=TEXT, ls=":", lw=1.5, zorder=3)  # from the front axle down to the arrow that measures e
    link(ax, (-e, -1.0), (0.0, -1.0))
    ax.text(-e / 2, -1.5, f"e = {e:.2f} m left", ha="center", va="top")
    ax.text(0, -2.5, "angles drawn 3× larger", ha="center", va="top")
    save(fig, path)


def circle(path):
    delta = 0.7
    r_rear, radius = WHEELBASE / math.tan(delta), WHEELBASE / math.sin(delta)
    centre = np.array([-r_rear, 0.0])
    front = np.array([0.0, WHEELBASE])
    wheel_dir = np.array([-math.sin(delta), math.cos(delta)])
    fig, ax = canvas(4.4)
    blank(ax, (-5.0, 2.6), (-1.0, 4.6))
    ax.plot([0, 0], [-0.6, 0.6], color=TEXT, lw=7, solid_capstyle="butt")
    ax.plot(*np.array([front - 0.6 * wheel_dir, front + 0.6 * wheel_dir]).T, color=TEXT, lw=7, solid_capstyle="butt")
    ax.plot([0, 0], [0, WHEELBASE], color=TEXT, lw=2)
    ax.plot([0, centre[0]], [0, 0], color=TEXT, ls="--", lw=1.5)
    ax.plot([front[0], centre[0]], [front[1], centre[1]], color=TEXT, ls="--", lw=1.5)
    ax.plot(*np.array([front, front + 1.4 * np.array([0, 1])]).T, color=TEXT, ls=":", lw=1.5)
    ax.add_patch(Arc(front, 2.4, 2.4, theta1=math.degrees(math.pi / 2), theta2=math.degrees(math.pi / 2 + delta), color=GREEN, lw=3))
    ax.add_patch(Arc(centre, 2.8, 2.8, theta1=0, theta2=math.degrees(delta), color=GREEN, lw=3))
    ax.text(front[0] + 0.3, front[1] + 1.1, "δ", color=GREEN, ha="left", va="center")
    ax.text(centre[0] + 1.7, 0.45, "δ", color=GREEN, ha="left", va="center")
    ax.text(0.25, WHEELBASE / 2, "L", ha="left", va="center")
    mid = (front + centre) / 2
    ax.text(mid[0] - 0.15, mid[1] + 0.35, "R", ha="right", va="bottom")
    ax.text(centre[0] - 0.2, 0.0, "turn\ncentre", ha="right", va="center")
    ax.text(0.9, -0.55, "sin δ = L / R", ha="left", va="center", weight="bold")
    ax.text(-4.9, 4.6, "angles drawn larger", ha="left", va="top")
    save(fig, path)


def steady_state(path):
    fig, ax = canvas(4.8)
    heads = (K_HEAD, 0.0, 1.0)
    shade_bends(ax, GENTLE_BENDS)
    for k, colour in zip(heads, shades(BLUE, 3)):
        tel = perfect_run("gentle", gains(K_OFF, k))
        ax.plot(tel.progress_m, tel.lat, color=colour, lw=2.5)
        e_bend = float(np.median(tel.lat[in_windows(tel, GENTLE_BEND_MIDDLES)]))
        if k == 1.0:
            note(ax, f"{k:g}: within {np.abs(tel.lat).max():.3f} m", (560, 0.0), (0, 8), colour, ha="center", va="bottom")
        else:
            note(ax, f"{k:g}: {e_bend:+.3f} m", (225, e_bend), (0, 8 if e_bend > 0 else -8), colour, ha="center", va="bottom" if e_bend > 0 else "top")
    finish(ax, "distance along the road (m)", "e (m)")
    ax.set_ylim(-0.17, 0.26)
    save(fig, path)


def steady_curvy(path):
    fig, ax = canvas(4.8)
    for turn, name, alpha in ((+1, "left\nbend", 0.2), (-1, "right\nbend", 0.07)):  # left bends darker, right bends paler, so neighbours stay apart
        shade_bends(ax, curvy_spans(turn), alpha)
        for a, b in curvy_spans(turn):
            ax.text((a + b) / 2, 0.02, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom")
    for k, colour, y in zip((K_HEAD, 1.0), shades(BLUE, 2), (0.96, 0.87)):
        tel = perfect_run("curvy", gains(K_OFF, k))
        ax.plot(tel.progress_m, tel.lat, color=colour, lw=2.5)
        reach = float(np.abs(tel.lat).max())
        corner(ax, f"heading gain {k:g}: " + (f"up to {reach:.3f} m" if k > 1 else f"within {reach:.3f} m"), 0.02, y, colour=colour, va="top")
    for turn in (+1, -1):
        predicted = (K_HEAD - 1) * hold_angle(bend_radius(20.0, turn), turn) / K_OFF
        for a, b in curvy_spans(turn):
            ax.plot([a, b], [predicted, predicted], color=TEXT, ls="--", lw=2)
        a, b = curvy_spans(turn)[0]
        note(ax, f"predicted {predicted:+.3f} m", ((a + b) / 2, predicted), (0, 6 if turn > 0 else -6), ha="center", va="bottom" if turn > 0 else "top")
    finish(ax, "distance along the road (m)", "e (m)")
    ax.set_ylim(-1.3, 1.3)
    save(fig, path)


def box_bias(tel, windows=GENTLE_BEND_MIDDLES):
    """(offset bias, heading bias) the provided box shows over `windows` of a run: median of what it said minus the truth."""
    keep = in_windows(tel, windows)
    return float(np.median((tel.est_offset - tel.lat)[keep])), float(np.median((tel.est_heading - tel.head_err)[keep]))


def bias_terms(path):
    provided_gentle = box_run("gentle", provided.controller)
    b_off, b_head = box_bias(provided_gentle)
    delta = hold_angle(bend_radius(GENTLE_RADIUS, +1))
    terms = [("P controller", (K_HEAD - 1) * delta / K_OFF), ("offset bias", -b_off), (f"heading bias × {K_HEAD / K_OFF:.2f} m", -K_HEAD * b_head / K_OFF)]
    total = sum(v for _, v in terms)
    measured = float(np.median(provided_gentle.lat[in_windows(provided_gentle, GENTLE_BEND_MIDDLES)]))
    fig, ax = canvas(4.6)
    start = 0.0
    rows = []
    for (name, value), colour in zip(terms, (BLUE, RED, RED)):
        rows.append((name, start, value, colour))
        start += value
    rows.append(("predicted", 0.0, total, BLUE))
    for i, (name, left, value, colour) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, value, left=left, height=0.42, color=colour)
        label(ax, min(left, left + value), y + 0.3, f"{name} {value:+.3f}", va="bottom", ha="left", color=TEXT)
    dot(ax, measured, 0.0, TEXT, 11)
    label(ax, measured, -0.35, f"measured {measured:+.3f}", ha="right", va="top")
    ax.axvline(0, color=TEXT, lw=1)
    ax.set_yticks([])
    ax.set_ylim(-0.9, len(rows) - 0.1)
    finish(ax, "bend offset e (m)")
    ax.margins(y=0)
    save(fig, path)


def curvy_leave(path):
    tel = box_run("curvy", provided.controller)
    fig, (top, bottom) = canvas(6.6, 2, 1, sharex=True)
    for ax in (top, bottom):
        shade_bends(ax, [(CURVY_START, tel.progress_m[-1])])
    top.plot(tel.progress_m, tel.lat, color=BLUE, lw=2.5)
    top.plot(tel.progress_m, tel.est_offset, color=RED, lw=2.5)
    yellow_line(top)
    i = int(np.argmin(np.abs(tel.progress_m - 30.3)))
    dot(top, tel.progress_m[i], tel.lat[i], BLUE)
    dot(top, tel.progress_m[i], tel.est_offset[i], RED)
    note(top, f"box {tel.est_offset[i]:.3f} m", (tel.progress_m[i], tel.est_offset[i]), (-14, 8), RED, ha="right", va="bottom")
    note(top, f"true {tel.lat[i]:.3f} m", (tel.progress_m[i], tel.lat[i]), (14, -8), BLUE, va="top")
    cross(top, tel.progress_m[-1], tel.lat[-1])
    bottom.plot(tel.progress_m, tel.head_err, color=PINK, lw=2.5)
    bottom.plot(tel.progress_m, tel.est_heading, color=RED, lw=2.5)
    j = int(np.argmin(np.abs(tel.progress_m - 44.3)))
    dot(bottom, tel.progress_m[j], tel.head_err[j], PINK)
    dot(bottom, tel.progress_m[j], tel.est_heading[j], RED)
    note(bottom, f"box {tel.est_heading[j]:.3f} rad", (tel.progress_m[j], tel.est_heading[j]), (-10, -8), RED, ha="right", va="top")
    note(bottom, f"true\n{tel.head_err[j]:.3f}\nrad", (tel.progress_m[j], tel.head_err[j]), (-10, -8), PINK, ha="right", va="top")
    finish(top, None, "e (m)")
    finish(bottom, "distance along the road (m)", "heading (rad)")
    save(fig, path)


# the tuning map: 182 box runs, spread over worker processes (forked, so the figures module needs no import path)

def _cell(spec):
    scenario, k_off, k_head = spec
    tel = box_run(scenario, gains(k_off, k_head))
    return spec, (tel.completed, metrics(tel)["mean_abs_lat"])


def tuning_cells():
    specs = [(s, a, b) for s in ("gentle", "curvy") for a in K_OFFS for b in K_HEADS]
    module = types.ModuleType(__name__)
    module.__dict__.update(globals())
    sys.modules[__name__] = module  # a worker unpickles `_cell` by this module name; this load's copy, not a stale earlier one
    with render_pool() as pool:
        return dict(pool.map(_cell, specs))


def zero_offset_line(delta, b_off, b_head):
    """(k_off, k_head) pairs on which the bias formula predicts no bend offset: k_head = (delta + k_off b_off) / (delta - b_head)."""
    pairs = [(k, (delta + k * b_off) / (delta - b_head)) for k in K_OFFS]
    return [(k, h) for k, h in pairs if K_HEADS[0] <= h <= K_HEADS[-1]]


def cell_position(value, grid):
    """Where `value` sits on a map axis whose cells are equal width: its index, interpolated between neighbouring grid values."""
    return float(np.interp(value, grid, np.arange(len(grid))))


def tuning_map(path):
    cells = tuning_cells()
    gentle = box_run("gentle", provided.controller)
    g_off, g_head = box_bias(gentle)
    road = make_scenario("curvy").road
    d = hold_angle(bend_radius(20.0, +1))
    est = provided.estimate_lane(Observation(Camera().render(road, BicycleState(*road.ego_pose(CURVY_START + CURVY_ARC / 2, 0.0, -d), TARGET_SPEED, 0.0)), TARGET_SPEED, 0.0))
    lines = {"gentle": zero_offset_line(hold_angle(bend_radius(GENTLE_RADIUS, +1)), g_off, g_head), "curvy": zero_offset_line(d, est.offset_m, est.heading_rad + d)}
    means = [m for (_, m) in cells.values() if m > 0]
    ramp = LinearSegmentedColormap.from_list("ramp", [shades(BLUE, 2)[1], BLUE, tuple(0.45 * np.array(to_rgb(BLUE)) + 0.55 * np.array(to_rgb(BACKGROUND)))])
    norm = LogNorm(min(means), max(means))
    fig, axes = canvas(8.2, 2, 1)
    for ax, scenario in zip(axes, ("gentle", "curvy")):
        grid = np.full((len(K_HEADS), len(K_OFFS)), np.nan)
        for (s, a, b), (done, mean) in cells.items():
            if s == scenario:
                grid[K_HEADS.index(b), K_OFFS.index(a)] = mean if done else np.nan
        image = ax.imshow(np.ma.masked_invalid(grid), cmap=ramp, norm=norm, origin="lower", aspect="auto", interpolation="nearest", extent=(-0.5, len(K_OFFS) - 0.5, -0.5, len(K_HEADS) - 0.5))
        for (s, a, b), (done, _) in cells.items():
            if s == scenario and not done:
                ax.add_patch(Rectangle((K_OFFS.index(a) - 0.5, K_HEADS.index(b) - 0.5), 1, 1, fill=False, hatch="////", ec=to_rgba(TEXT, 0.55), lw=0))
        xs, ys = zip(*[(cell_position(a, K_OFFS), cell_position(b, K_HEADS)) for a, b in lines[scenario]])
        ax.plot(xs, ys, color=TEXT, ls="--", lw=2)
        for a, b in ((K_OFF, K_HEAD), (0.7, K_HEAD), (1.4, K_HEAD), (0.5, 1.0)):
            ax.plot([K_OFFS.index(a)], [K_HEADS.index(b)], "o", ms=18, fillstyle="none", mec=TEXT, mew=2.5)
        ax.set_xticks(range(0, len(K_OFFS), 2), [f"{K_OFFS[i]:g}" for i in range(0, len(K_OFFS), 2)])
        ax.set_yticks(range(len(K_HEADS)), [f"{h:g}" for h in K_HEADS])
        ax.set(xlabel="offset gain k_off", ylabel="heading gain k_head")
        corner(ax, scenario, 1.0, 1.02, weight="bold", ha="right", va="bottom")
        if scenario == "gentle":
            for (a, b), text, shift in zip(((K_OFF, K_HEAD), (0.7, K_HEAD), (1.4, K_HEAD)), ("provided\n0.35, 2.7", "2×\n0.7, 2.7", "4×\n1.4, 2.7"), ((-34, 44), (0, 62), (34, 44))):
                ax.annotate(text, (K_OFFS.index(a), K_HEADS.index(b)), shift, textcoords="offset points", ha="center", va="bottom", color=TEXT, annotation_clip=False,
                            arrowprops={"arrowstyle": "-", "color": TEXT, "lw": 1, "shrinkA": 0, "shrinkB": 9})
            ax.text(K_OFFS.index(0.5) + 0.7, K_HEADS.index(1.0) - 0.75, "0.5, 1.0", color=TEXT, ha="left", va="center", bbox=LABEL_BOX)
    fig.colorbar(image, ax=axes, orientation="horizontal", label="mean |e| (m)", shrink=0.9, pad=0.04)
    save(fig, path)


def gain_sweep(path):
    fig, top = canvas(4.6)
    runs = {k: box_run("gentle", gains(k, K_HEAD)) for k in (0.05, 0.35, 1.4, 6.0)}
    for (k, tel), colour in zip(runs.items(), shades(BLUE, 4)):
        window = tel.t <= 20.0
        top.plot(tel.t[window], tel.lat[window], color=colour, lw=2.5)
        reach = float(np.abs(tel.lat[window]).max())
        if not tel.completed:
            cross(top, tel.t[-1], tel.lat[-1])
            note(top, f"{k}: leaves at {tel.t[-1]:.2f} s", (tel.t[-1], tel.lat[-1]), (-14, 4) if k < 1 else (14, 0), colour, ha="right" if k < 1 else "left", va="center")
        elif reach > 0.1:
            note(top, f"{k}: up to {reach:.3f} m", (15.0, reach), (-30, 16), colour, line=True, ha="right", va="center")
        else:
            note(top, f"{k}: within {reach:.3f} m", (19.5, 0.0), (0, -8), colour, ha="right", va="top")
    finish(top, "time (s)", "e (m)")
    save(fig, path)


def wheel_lag(path):
    """The steering the rule asks for against the angle the wheels reach, for offset gain 6.0 on gentle."""
    fig, bottom = canvas(4.4)
    bad = box_run("gentle", gains(6.0, K_HEAD))
    cmd = np.array([provided.steer_p(LaneEstimate(o, h, bool(v)), 6.0, K_HEAD) for o, h, v in zip(bad.est_offset, bad.est_heading, bad.est_valid)])
    light, dark = shades(GREEN, 2)[1], GREEN
    bottom.plot(bad.t[1:], cmd[:-1], color=light, lw=2.5)
    bottom.plot(bad.t[1:], bad.steer[1:], color=dark, lw=2.5)
    for t_mark in (1.0, 3.0):
        bottom.axvline(t_mark, color=TEXT, ls=":", lw=1.5)
    corner(bottom, "rule asks", 0.3, 0.96, colour=light, va="top")
    corner(bottom, "wheels reach", 0.3, 0.85, colour=dark, va="top")
    finish(bottom, "time (s)", "steering (rad)")
    bottom.set_ylim(-0.6, 0.85)
    save(fig, path)


def loop_lookback(path):
    names = titles()
    steps = [("camera picture", None),
             (f"paint_masks\n({names['0.1.2']})", None),
             (f"estimate_lane\n({names['0.1.3']})", f"chord bias on bends:\n{names['1.5']}"),
             (f"p_steer\n({names['0.1.4']})", f"steers on the error at\nthe car: {names['1.3']}"),
             ("car moves", "provided speed controller:\nLongitudinal control and\nvehicle motion")]
    if names["1.2"] != "Longitudinal control and vehicle motion":
        raise RuntimeError("the syllabus renamed lesson 1.2; update the car-moves note")
    fig, ax = canvas(8.4)
    ax.set(xlim=(0, 10.6), ylim=(0, 22))
    ax.axis("off")
    x, y, tops = 4.4, 21.5, []
    for name, weakness in steps:
        height = 0.9 * (name.count("\n") + 1) + 0.6
        tags = tag(ax, name, (x, y - height / 2), TEXT)
        tops.append((y, y - height))
        y -= height + 0.25
        if weakness:
            lines = weakness.count("\n") + 1
            ax.text(x, y, weakness, ha="center", va="top", color=TEXT)
            y -= 0.9 * lines + 0.2
        if name != steps[-1][0]:
            link(ax, (x, y), (x, y - 1.0))
            y -= 1.0
    ax.set_ylim(y - 0.3, 22)
    mid_first, mid_last = (tops[0][0] + tops[0][1]) / 2, (tops[-1][0] + tops[-1][1]) / 2
    ax.plot([x + 1.05, 9.0, 9.0], [mid_last, mid_last, mid_first], color=TEXT, lw=2)
    link(ax, (9.0, mid_first), (x + 1.7, mid_first))
    ax.text(9.8, (mid_first + mid_last) / 2, "every 0.05 s", rotation=90, ha="center", va="center")
    save(fig, path)


def alvinn(path):
    fig, ax = canvas(5.4)
    ax.set(xlim=(0, 10), ylim=(0, 12))
    ax.axis("off")
    ax.text(1.6, 11.4, "1217 inputs", ha="center", va="center", weight="bold")
    ax.text(8.5, 11.4, "46 outputs", ha="center", va="center", weight="bold")
    for y, text in zip((9, 6, 3), ("video 30 × 32\n= 960", "range 8 × 32\n= 256", "feedback\n1")):
        tag(ax, text, (1.6, y), TEXT)
        link(ax, (3.1, y), (4.2, 6.0 + (y - 6) * 0.25))
    tag(ax, "29 hidden\nunits", (5.3, 6), TEXT)
    for y, text in zip((9, 6, 3), ("sharp left", "straight", "sharp right")):
        tag(ax, text, (8.5, y), TEXT)
        link(ax, (6.5, 6.0 + (y - 6) * 0.25), (7.3, y))
    tag(ax, "brightness\nfeedback", (8.5, 0.9), TEXT)
    link(ax, (6.5, 6.0 - 5.1 * 0.25), (7.2, 1.2))
    ax.text(8.5, 7.5, "⋮", ha="center", va="center")
    ax.text(8.5, 4.5, "⋮", ha="center", va="center")
    save(fig, path)


AHEAD = 10.0  # the "?" target: a point on the lane centre further ahead than the box's two points


def next_lesson(path):
    road = make_scenario("gentle").road
    frame = Camera().render(road, BicycleState(*road.ego_pose(50.0), TARGET_SPEED, 0.0))
    points = [provided.lane_point(provided.paint_masks(frame), rows) for rows in (provided.NEAR_ROWS, provided.FAR_ROWS)]
    fig = Figure(figsize=(W, 3.6), layout="constrained")
    pic, top = fig.subplots(1, 2, width_ratios=(2.1, 1))
    pic.imshow(frame, interpolation="nearest")
    pic.set(xticks=[], yticks=[])
    pic.axis("off")
    for (forward, left), name, y in zip(points, ("near", "far"), (0.2, 0.8)):
        u, v = Camera().ground_to_pixel(forward, left)
        pic.plot([u], [v], "o", color=TEXT, ms=9, mec=BACKGROUND, mew=1.5)
        pic.annotate(f"{name}\n{forward:.2f} m", (u, v), (-0.04, y), textcoords="axes fraction", ha="right", va="center", color=TEXT, annotation_clip=False,
                     arrowprops={"arrowstyle": "-", "color": TEXT, "lw": 1, "shrinkA": 0, "shrinkB": 5})
    top.set(xlim=(3, -3), ylim=(-1, 12), aspect="equal", xlabel="left (m)", ylabel="forward (m)")
    top.plot([0], [0], "^", color=TEXT, ms=12)
    for forward, left in points:
        top.plot([left], [forward], "o", color=TEXT, ms=9, mec=BACKGROUND, mew=1.5)
    top.plot([0], [AHEAD], "o", fillstyle="none", mec=TEXT, mew=2.5, ms=22)
    top.text(-1.0, AHEAD, "?", ha="left", va="center", weight="bold", fontsize=24)
    save(fig, path)


FIGURES = (result_first, open_vs_closed, loop_kinds, offset_only, p_law, push_family, wheel_vs_lane, circle, steady_state, steady_curvy,
           bias_terms, curvy_leave, tuning_map, gain_sweep, wheel_lag, loop_lookback, alvinn, next_lesson)


def make(out_dir):
    with rc_context({"font.size": FS}):
        for figure in FIGURES:
            figure(out_dir / f"{figure.__name__}.png")
