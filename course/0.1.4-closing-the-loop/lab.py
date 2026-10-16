# %% [markdown]
# # Lab: Closing the loop
#
# You will see what feedback buys, write the P steering law `p_steer`, find out why a car holds a bend off the centre,
# tune the gains, and build a car in which every box is yours. Read the lecture first; this lab points back to its
# sections by title and follows its order.
#
# Run the cells from top to bottom (Shift+Enter). Each problem has a "Guess first" line: write your guess down before you
# run the cell, then compare. `practice` prints a green tick or tells you what it expected.
#
# ![](figures/result_first.png)

# %%
import sys
if "google.colab" in sys.modules:
    %pip install -q "git+https://github.com/vvknyn/self-driving-ai-course@main"
%matplotlib inline

# %% tags=["setup"]
import math

import matplotlib.pyplot as plt
import numpy as np

from zero2fsd.car import Car, LaneEstimate, provided, run
from zero2fsd.car.provided import FAR_ROWS, MIN_PIXELS, NEAR_ROWS, paint_masks
from zero2fsd.dashboard import dashboard
from zero2fsd.grade import check, practice
from zero2fsd.score import metrics
from zero2fsd.sim import Camera
from zero2fsd.sim.dynamics import DT, MAX_STEER, WHEELBASE

camera = Camera()


def ends(tel):
    """How a run ended, in words."""
    return "completes" if tel.completed else f"leaves the road at {tel.progress_m[-1]:.1f} m, t = {tel.t[-1]:.2f} s"

# %% [markdown]
# The next cell holds helpers for later problems: a perfect-box car, a way to lift the wheel's speed limit, and the median
# offset inside the bends. Run it; you do not need to read it.

# %%
from contextlib import contextmanager

from zero2fsd.car.types import Action
from zero2fsd.sim import BicycleState, dynamics, make_scenario
from zero2fsd.sim.dynamics import step

GENTLE_BENDS = ((150, 300), (500, 650))  # progress windows well inside gentle's two bends, in metres


def bend_median(tel, values, windows=GENTLE_BENDS):
    """The median of `values` (one number per step) while the car is inside the given stretches of the road."""
    keep = np.zeros(len(tel.t), bool)
    for a, b in windows:
        keep |= (tel.progress_m >= a) & (tel.progress_m <= b)
    return float(np.median(values[keep]))


class Perfect:
    """Perception that reads the true offset and heading from the simulator, and a controller that applies `law`."""

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
    """Drive `scenario` with a perfect box (the true offset and heading) and the steering law `law(est, obs)`."""
    p = Perfect(scenario, law)
    return run(Car(perception=p, controller=p), scenario)


@contextmanager
def without_wheel_limit():
    """Lift the steering wheel's speed limit inside a `with` block, then put it back."""
    saved = dynamics.MAX_STEER_RATE
    dynamics.MAX_STEER_RATE = 1e9
    try:
        yield
    finally:
        dynamics.MAX_STEER_RATE = saved

# %% [markdown]
# ## 1. Open loop or closed loop
#
# The next cell drives the provided car around `gentle` and records its steering at every step. Then it replays the
# recording with no camera at all: an **open-loop** car (lecture: "Open loop and closed loop"). Last, it sets the wheel
# 0.002 rad off, a tiny misalignment, in both cars.
#
# **Guess first:** with the wheel 0.002 rad off, does the closed-loop car leave the road? Does the open-loop replay? If it
# does, after how many seconds?

# %%
closed = run(Car(), "gentle")
plan = closed.steer  # the wheel angle at each step; the command given at step k is the angle recorded at step k + 1


def replay(bias):
    """An open-loop controller: ignores the camera and plays back `plan`, off by a constant `bias` radians."""
    return lambda est, obs: float(plan[min(round(obs.t / DT) + 1, len(plan) - 1)] + bias)


BIAS = 0.002  # rad
print(f"{BIAS} rad is {math.degrees(BIAS):.2f} degrees")
runs = {
    "closed loop": closed,
    "open loop, exact replay": run(Car(controller=replay(0.0)), "gentle"),
    f"closed loop, wheel {BIAS} rad off": run(Car(controller=lambda est, obs: provided.controller(est, obs) + BIAS), "gentle"),
    f"open loop, wheel {BIAS} rad off": run(Car(controller=replay(BIAS)), "gentle"),
}
plt.figure(figsize=(9, 3.5))
for name, tel in runs.items():
    print(f"{name:32}: {ends(tel)}, mean |lat| {metrics(tel)['mean_abs_lat']:.3f} m")
    plt.plot(tel.t, tel.lat, label=name)
plt.xlim(0, 20)
plt.xlabel("time (s)")
plt.ylabel("offset (m), left +")
plt.legend(fontsize=8)
plt.show()

# %% [markdown]
# **Problem 1.** Read the time the open-loop car with the wheel off left the road from the `runs` dictionary above (the
# run's last time, `tel.t[-1]`).

# %%
t_leave = None  # your code: the time, in seconds, at which that run ended

practice("open-loop leave time (s)", t_leave, 8.40, tol=0.05)

# %% [markdown]
# The exact replay drives the identical lap, so the recording is right. With 0.11 degrees of error, the replay leaves
# the road on the first straight, while the closed-loop car barely notices: 0.381 m instead of 0.376 m. Nothing in
# the replay measures the error, so it adds up; the closed loop measures it every 0.05 s and steers against it.
#
# ![](figures/open_vs_closed.png)
#
# ![](figures/loop_kinds.png)
#
# ## 2. The P controller
#
# The worked example of "The P controller": the steering is the negative of two terms, one for the offset and one for
# the heading. `provided.steer_p(est, k_off, k_head)` is the provided car's version; you write your own below.
#
# **Problem 2.** The car is 0.3 m left of centre and pointing 0.1 rad *right* (heading −0.1). What steering angle does the
# law give with `k_off = 0.35` and `k_head = 2.7`? Write the formula out; do not call `steer_p`.
#
# **Guess first:** does the car steer left or right?

# %%
steer_by_hand = None  # your code

practice("steering angle (rad)", steer_by_hand, 0.165, tol=0.001)

# %% [markdown]
# ### Graded exercise: `p_steer`
#
# Write the P law as a function of the estimate and the two gains. The check tries six named cases (signs, both clips, a
# missing estimate) and 30 random ones.

# %% tags=["exercise:0.1.4.a"]
def p_steer(est, k_off, k_head):
    """The P steering law.

    Inputs:  est, a LaneEstimate(offset_m, heading_rad, valid): offset_m > 0 when the car is LEFT of lane centre,
             heading_rad > 0 when it points LEFT of the lane.  k_off (rad per metre) and k_head (no units), both > 0.
    Output:  the steering angle in radians as a float, > 0 steers LEFT, steering against both errors and clipped to
             [-MAX_STEER, MAX_STEER] (MAX_STEER = 0.5).  0.0 when est.valid is False.
    Example: p_steer(LaneEstimate(0.4, 0.05, True), 0.35, 2.7)  ->  -0.275
    """
    raise NotImplementedError


# %%
check("0.1.4.a", p_steer)

# %% [markdown]
# Next, the push test of "The P controller": the wheels are held 0.025 rad left for the first second, which leaves the car
# 0.50 m left of the centre, and then the law takes over. The runs stop after 12 s, still on the first straight, where
# the provided box is almost exact. First with the offset term only, `k_head = 0`.
#
# **Guess first:** does the car stop at the centre, or go past it? If past, how far?

# %%
PUSH_STEPS, PUSH_STEER = 20, 0.025  # one second of 0.025 rad


def pushed(k_head, k_off=0.35):
    """The push test on gentle's first straight: wheels held at PUSH_STEER for PUSH_STEPS steps, then the P law."""
    law = lambda est, obs: PUSH_STEER if obs.t < PUSH_STEPS * DT - 1e-9 else provided.steer_p(est, k_off, k_head)
    return run(Car(controller=law), "gentle", max_steps=240)


tel = pushed(0.0)
print(f"handover at 1.0 s: {tel.lat[PUSH_STEPS]:.2f} m left of the centre, heading {tel.head_err[PUSH_STEPS]:+.3f} rad")
plt.figure(figsize=(9, 3.2))
plt.plot(tel.t, tel.lat)
plt.axhline(0, color="0.5", lw=1)
plt.xlabel("time (s)")
plt.ylabel("offset (m), left +")
plt.show()

# %% [markdown]
# **Problem 3.** How far past the centre does the car swing? Give a positive number.

# %%
swing = None  # your code: from `tel.lat`, how far the car goes to the far side of the centre, in metres

practice("swing past the centre (m)", swing, 0.135, tol=0.002)

# %% [markdown]
# The car reaches the centre pointing across the lane, so it overshoots: the offset term alone cannot see that the car
# is already turning back.
#
# ![](figures/offset_only.png)
#
# ![](figures/p_law.png)
#
# ## 3. How the heading gain shapes the return
#
# The next cell repeats the push test for six heading gains and measures how long each takes to get within 5 cm of the
# centre for good, counted from the handover (lecture: "How the heading gain shapes the return").
#
# **Guess first:** which gain comes back fastest, and which swings past the centre?

# %%
plt.figure(figsize=(9, 3.5))
settle = {}
for k_head in (0.0, 0.5, 1.0, 2.0, 2.7, 5.0):
    tel = pushed(k_head)
    outside = np.nonzero(np.abs(tel.lat) >= 0.05)[0]  # steps more than 5 cm from the centre
    settle[k_head] = tel.t[outside[-1] + 1] - PUSH_STEPS * DT
    print(f"k_head = {k_head:3}: furthest past the centre {-tel.lat[PUSH_STEPS:].min():+.3f} m, "
          f"within 5 cm for good {settle[k_head]:.2f} s after the handover")
    plt.plot(tel.t, tel.lat, label=f"k_head = {k_head}")
plt.axhline(0, color="0.5", lw=1)
plt.xlabel("time (s)")
plt.ylabel("offset (m), left +")
plt.legend(fontsize=8)
plt.show()

# %% [markdown]
# **Problem 4.** At the handover the car is 0.496 m left of the centre and points 0.074 rad left. With `k_head = 1`, the
# law's steering angle is δ, and the front wheels end up pointing ψ + δ from the lane's direction. Compute that sum.
# Compare it with `-k_off * offset`.

# %%
tel = pushed(1.0)
e, psi = tel.lat[PUSH_STEPS], tel.head_err[PUSH_STEPS]
delta = provided.steer_p(LaneEstimate(e, psi, True), 0.35, 1.0)
wheels_vs_lane = None  # your code: psi + delta

practice("psi + delta (rad)", wheels_vs_lane, -0.1735, tol=0.001)

# %% [markdown]
# With a heading gain of 1 the wheels point −0.35 × offset away from the lane's direction: the rule aims the wheels at
# the lane centre in proportion to the offset, and the car comes straight back without a swing.
#
# ![](figures/push_family.png)
#
# ![](figures/wheel_vs_lane.png)
#
# ## 4. Holding a bend
#
# On a bend the car must keep its wheels turned. **Problem 5.** On `curvy`'s right bends your lane has radius 18.2 m. What
# wheel angle holds that circle? Mind the sign: a right turn is negative. The wheelbase is `WHEELBASE`, 2.7 m.
#
# **Guess first:** is it more or less than 0.1 rad?

# %%
delta_right = None  # your code

practice("wheel angle to hold a right bend of radius 18.2 m (rad)", delta_right, -0.1489, tol=0.001)

# %% [markdown]
# The lecture's picture of the car turning about a point shows where the formula comes from.
#
# ![](figures/circle.png)
#
# Now let the car hold a bend. A perfect box reads the true offset and heading, so the only error left is the law's own.
# The next cell drives a lap of `gentle` with a perfect box and the P law at `k_off = 0.35` for three heading gains.
#
# **Guess first:** where does the car sit in the middle of a bend for `k_head` = 2.7, for 0 and for 1?

# %%
for k_head in (2.7, 0.0, 1.0):
    tel = perfect_run("gentle", lambda est, obs, k_head=k_head: provided.steer_p(est, 0.35, k_head))
    print(f"k_head = {k_head:3}: middle of the bends {bend_median(tel, tel.lat):+.3f} m, never more than {np.abs(tel.lat).max():.3f} m from the centre")

# %% [markdown]
# ![](figures/steady_state.png)
#
# **Problem 6.** With a perfect box the law settles at $e = (k_{\text{head}} - 1)\,\delta / k_{\text{off}}$ (lecture: "Holding a
# bend"). Use it for the right bends of `curvy` at gains 0.35 and 2.7, with the wheel angle you found in Problem 5.

# %%
offset_right = None  # your code

practice("offset in a right bend of radius 18.2 m (m)", offset_right, -0.723, tol=0.005)

# %%
for k_head in (2.7, 1.0):
    tel = perfect_run("curvy", lambda est, obs, k_head=k_head: provided.steer_p(est, 0.35, k_head))
    print(f"k_head = {k_head}: never more than {np.abs(tel.lat).max():.3f} m from the centre")

# %% [markdown]
# ![](figures/steady_curvy.png)
#
# ## 5. Where the box pulls the car
#
# The provided box is not perfect: on a bend it reports the chord of the curve, not the curve. The next cell drives
# `gentle` with the provided car and measures, in the middle of the bends, what the box said minus the truth.

# %%
tel = run(Car(), "gentle")
b_off = bend_median(tel, tel.est_offset - tel.lat)
b_head = bend_median(tel, tel.est_heading - tel.head_err)
measured = bend_median(tel, tel.lat)
delta_gentle = math.asin(WHEELBASE / 81.8)  # the wheel angle that holds gentle's bends, lane radius 81.8 m
print(f"box offset bias {b_off:+.3f} m, box heading bias {b_head:+.4f} rad, measured bend offset {measured:+.3f} m")

# %% [markdown]
# **Problem 7.** Lecture: "Where the box pulls the car". The bend offset is
# $(k_{\text{head}}-1)\,\delta/k_{\text{off}} - b_{\text{off}} - (k_{\text{head}}/k_{\text{off}})\,b_{\text{head}}$. Predict the
# provided car's bend offset from `b_off`, `b_head` and `delta_gentle`.
#
# **Guess first:** is it larger or smaller than the +0.160 m the perfect box gave?

# %%
predicted = None  # your code

practice("predicted bend offset (m)", predicted, 0.529, tol=0.005)

# %% [markdown]
# **Problem 8.** The two biases pull in the same direction through the gains. Their ratio $b_{\text{off}} / |b_{\text{head}}|$
# is the gain ratio $k_{\text{head}}/k_{\text{off}}$ that cancels them.

# %%
ratio = None  # your code

practice("bias ratio (m)", ratio, 1.995, tol=0.05)

# %% [markdown]
# ![](figures/bias_terms.png)
#
# Now the same car on `curvy` (lecture: "Why the provided car leaves curvy"). The next cell prints, at points along the
# first left bend, where the car truly was and what the box reported.

# %%
tel = run(Car(), "curvy")
print(f"the provided car on curvy: {ends(tel)}")
for s in (24, 30, 36, 44, 48):
    k = int(np.argmax(tel.progress_m >= s))
    print(f"{tel.progress_m[k]:5.1f} m | true {tel.lat[k]:+.3f} m, {tel.head_err[k]:+.3f} rad | box {tel.est_offset[k]:+.3f} m, {tel.est_heading[k]:+.3f} rad")

# %% [markdown]
# From 30 m on, the box says the car points much further right than it does, and the law multiplies that heading by 2.7.
# On the true values it would steer right, back to the centre; on the box's values it steers left, deeper inside.
#
# ![](figures/curvy_leave.png)
#
# ## 6. Tuning your car
#
# The next cell drives `gentle` with the provided box and your `p_steer` for a 3 by 3 grid of gains (lecture: "Tuning your
# car"). Run it after the exercise above passes.
#
# **Guess first:** which of the nine pairs has the smallest mean |lat|, and does any leave the road?

# %%
try:
    for k_off in (0.35, 1.4, 3.0):
        for k_head in (0.5, 1.0, 2.7):
            tel = run(Car(controller=lambda est, obs, a=k_off, b=k_head: p_steer(est, a, b)), "gentle")
            print(f"k_off {k_off:4}, k_head {k_head:3}: {ends(tel):38} mean |lat| {metrics(tel)['mean_abs_lat']:.3f} m")
except NotImplementedError:
    print("Write p_steer above first, then run this cell again.")

# %% [markdown]
# The lecture's map holds 91 such pairs for each road, run for you.
#
# ![](figures/tuning_map.png)
#
# ### Too much gain
#
# Keep `k_head = 2.7` and raise `k_off` to 6.0 on `gentle`. **Guess first:** does it leave the road, and when?

# %%
try:
    tel = run(Car(controller=lambda est, obs: p_steer(est, 6.0, 2.7)), "gentle")
    print(f"k_off 6.0: {ends(tel)}")
    t_crash = tel.t[-1]
except NotImplementedError:
    print("Write p_steer above first, then run this cell again.")
    t_crash = None

practice("leave time (s)", t_crash, 4.30, tol=0.05)

# %% [markdown]
# The wheels turn at most 0.05 rad per step. At this gain a few centimetres of offset ask for a big change of steering,
# the wheels turn as fast as they can and are always late, and each swing is wider than the last.
#
# ![](figures/gain_sweep.png)
#
# ## 7. Closing the loop
#
# Every box of the car becomes yours. First, paste your `estimate_lane` from the lab for *Your first perception box* into
# the next cell, replacing the skeleton. If it used your own `paint_masks`, paste that above it. The cell sees the same
# names as that lab: `math`, `np`, `camera`, `paint_masks`, `NEAR_ROWS`, `FAR_ROWS`, `MIN_PIXELS`, `LaneEstimate`.

# %% tags=["exercise:0.1.4.b"]
def estimate_lane(obs):
    """Your perception box from the lab for Your first perception box: paste it here.

    Input:   obs, an Observation: obs.frame is the camera frame, a uint8 array of shape (180, 320, 3).
    Output:  LaneEstimate(offset_m, heading_rad, valid). offset_m > 0 when the car is LEFT of lane centre,
             heading_rad > 0 when the car points LEFT of the lane, both measured at the car (the front axle).
             valid=False, with math.nan for both numbers, when a band has fewer than MIN_PIXELS pixels of either line.
    """
    raise NotImplementedError


# %% [markdown]
# Now the controller and the car (lecture: "Closing the loop"). `my_controller` returns your `p_steer` with the gains
# below; a controller that returns a number is a steering angle, and the `Car` holds the speed for you. Then build
# `my_car` from your two boxes. The check needs it to complete `gentle` with mean |lat| under 0.5 m.

# %% tags=["exercise:0.1.4.b"]
K_OFF, K_HEAD = 0.35, 2.7  # the provided gains; section 6 showed others


def my_controller(est, obs):
    """Your controller box.

    Inputs:  est, the LaneEstimate your estimate_lane returned; obs, the Observation (not needed here).
    Output:  the steering angle in radians, > 0 steers LEFT: your p_steer with K_OFF and K_HEAD.
    """
    raise NotImplementedError


my_car = None  # your code: a Car with your perception and your controller

# %%
check("0.1.4.b", my_car)

# %% [markdown]
# Run your car on both roads at the provided gains and at 0.5 and 1.0, the pair the lecture found for this box.
# **Guess first:** which of the four runs leave the road?

# %%
if isinstance(my_car, Car):
    try:
        for gains in ((0.35, 2.7), (0.5, 1.0)):
            for road in ("gentle", "curvy"):
                car = Car(perception=estimate_lane, controller=lambda est, obs, g=gains: p_steer(est, *g))
                tel = run(car, road)
                print(f"gains {gains}, {road:6}: {ends(tel):38} mean |lat| {metrics(tel)['mean_abs_lat']:.3f} m")
    except NotImplementedError:
        print("Finish estimate_lane and p_steer above, then run this cell again.")
else:
    print("Build my_car above, then run this cell again.")

# %% [markdown]
# At 0.5 and 1.0 the box's two errors cancel: a patch for this box, not a fix. The real fixes come later in the course.
#
# ![](figures/loop_lookback.png)
#
# ## 8. Break it, fix it
#
# A teammate rewrote the controller: "Positive heading means the road turns left, so steer left." Run it on `gentle`.
# Before you open the hint: why does the car leave the road on the first straight, where the box is nearly exact? Then
# fix the one wrong sign and rerun the cell until the car completes the lap.

# %%
def teammate_controller(est, obs):
    """The teammate's controller."""
    if not est.valid:
        return 0.0
    return float(np.clip(-0.35 * est.offset_m + 2.7 * est.heading_rad, -MAX_STEER, MAX_STEER))


tel = run(Car(controller=teammate_controller), "gentle")
print(f"{ends(tel)}, mean |lat| {metrics(tel)['mean_abs_lat']:.3f} m")
for k in range(0, len(tel.t), 10):
    print(f"t = {tel.t[k]:3.1f} s: offset {tel.lat[k]:+.3f} m, heading {tel.head_err[k]:+.3f} rad, wheel {tel.steer[k]:+.3f} rad")
dashboard(tel)

# %% [markdown]
# <details><summary>Hint (open after you have a guess)</summary>
#
# `est.heading_rad` is the car's heading relative to the lane, not the road's turn. A car pointing right
# (negative heading) needs to steer left, so the heading term must be `-2.7 * est.heading_rad`. With the sign flipped,
# pointing right makes the teammate steer right, which points the car further right: that is positive feedback on the
# heading, and the error grows instead of shrinking. The fix is `-(0.35 * est.offset_m + 2.7 * est.heading_rad)`.
#
# </details>

# %%
def fixed_controller(est, obs):
    """Your fix of the teammate's controller."""
    raise NotImplementedError


try:
    tel = run(Car(controller=fixed_controller), "gentle")
    print(f"{ends(tel)}")
    fixed_mean = metrics(tel)["mean_abs_lat"]
except NotImplementedError:
    print("Write fixed_controller above, then run this cell again.")
    fixed_mean = None

practice("mean |lat| of the fixed car (m)", fixed_mean, 0.376, tol=0.002)

# %% [markdown]
# ## 9. A 1988 car that learned to steer
#
# ALVINN replaced both boxes with one network that read pixels and distances and gave a steering direction (lecture:
# "A 1988 car that learned to steer"). Read the paper's Figure 1 and "Network Architecture", then answer the lecture's
# three questions in your own words.
#
# ![](figures/alvinn.png)
#
# ## 10. Stretch: challenge problems
#
# Optional and hard, the lecture's "Challenge problems". The answers are in the lecture.
#
# **Predict the crash.** The open-loop car's front axle moves sideways at $v\,b$ and its heading turns at $v\,b/L$, so
# $e(t) = v\,b\,t + v^2 b\,t^2 / (2L)$. Solve $e(t) = 1.8$ for $t$ with $v = 8$ m/s, $b = 0.002$ rad and $L$ = `WHEELBASE`,
# and compare with the 8.40 s you measured. `np.roots` solves a quadratic from its three coefficients.

# %%
v, b, L = 8.0, 0.002, WHEELBASE
t_formula = None  # your code: the positive root, in seconds

practice("time to the yellow line (s)", t_formula, 8.38, tol=0.02)

# %% [markdown]
# **The fastest stable gain.** With a perfect box, a heading gain of 1 and the wheels' speed limit lifted, a gain of 5 or
# more should stop the offset shrinking. The helper `without_wheel_limit()` lifts the limit for a `with` block. Run `gentle`
# with a perfect box at `k_off` 4.5 and at 5.0.

# %%
mean_e = {}
for k_off in (4.5, 5.0):
    with without_wheel_limit():
        tel = perfect_run("gentle", lambda est, obs, k_off=k_off: provided.steer_p(est, k_off, 1.0))
    mean_e[k_off] = None  # your code: the mean |lat| of this run, from `metrics(tel)`
    print(f"k_off {k_off}: {ends(tel)}")

practice("mean |lat| at k_off 4.5 (m)", mean_e[4.5], 0.000, tol=0.003)
practice("mean |lat| at k_off 5.0 (m)", mean_e[5.0], 0.078, tol=0.003)

# %% [markdown]
# **Two predictions.** Use the bias formula of Problem 7 with the `gentle` biases to predict the bend offsets of the 2× car,
# gains 0.7 and 2.7, and of gains 1.5 and 2.7. Then drive both and compare.

# %%
pred_2x = None  # your code: gains 0.7 and 2.7
pred_15 = None  # your code: gains 1.5 and 2.7

for gains, pred in (((0.7, 2.7), pred_2x), ((1.5, 2.7), pred_15)):
    tel = run(Car(controller=lambda est, obs, g=gains: provided.steer_p(est, *g)), "gentle")
    print(f"gains {gains}: measured {bend_median(tel, tel.lat):+.3f} m")
practice("predicted offset at 0.7 and 2.7 (m)", pred_2x, 0.200, tol=0.003)
practice("predicted offset at 1.5 and 2.7 (m)", pred_15, 0.025, tol=0.003)

# %% [markdown]
# ## What's next
#
# ALVINN aimed at the road centre 7 m ahead. To aim at a point ahead, a controller must say where that point is, and the
# answer depends on who measures it: pixels, metres forward and left of the car, or metres along the road. The next
# lesson, *Frames and transforms*, moves a point from one frame to another.
#
# ![](figures/next_lesson.png)
