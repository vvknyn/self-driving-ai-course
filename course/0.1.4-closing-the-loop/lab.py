# %% [markdown]
# # Lab 0.1.4: Closing the loop
#
# You will see what feedback buys, write the P steering law `p_steer`, tune its gains, and build a car in which every
# box is yours. Then you take it to `curvy`. Read the lecture first; this lab points back to its sections by title.
#
# Run the cells from top to bottom (Shift+Enter).

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
    return "completes" if tel.completed else f"leaves the road at {tel.progress_m[-1]:.1f} m, t = {tel.t[-1]:.1f} s"

# %% [markdown]
# ## 1. Predict: open loop or closed loop?
#
# The next cell drives the provided car around `gentle` and records its steering at every step. Then it replays the
# recording with no camera at all: an **open-loop** car (lecture: "Open loop and closed loop"). Last, it sets the wheel
# 0.002 rad off, a tiny misalignment, in both cars.
#
# **Before you run the next cell**, double-click this one and write your prediction.
#
# **My prediction:** with the wheel 0.002 rad off, the closed-loop car ______ and the open-loop replay ______,
# because ______.

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
# The exact replay drives the identical lap, so the recording is right. With 0.11 degrees of error, the replay leaves
# the road on the first straight, while the closed-loop car barely notices: 0.381 m instead of 0.376 m. Nothing in
# the replay measures the error, so it adds up; the closed loop measures it every 0.05 s and steers against it.
#
# ## 2. Two errors, two gains
#
# The worked example of "Two errors, two gains": the car is 0.4 m left of centre and pointing 0.05 rad left.
# `provided.steer_p(est, k_off, k_head)` is the provided car's P law; you write your own in section 3.

# %%
est = LaneEstimate(offset_m=0.4, heading_rad=0.05, valid=True)
k_off, k_head = 0.35, 2.7
print(f"offset term   -k_off * offset    = {-k_off * est.offset_m:+.3f} rad")
print(f"heading term  -k_head * heading  = {-k_head * est.heading_rad:+.3f} rad")
print(f"steer = {-(k_off * est.offset_m + k_head * est.heading_rad):+.3f} rad; provided.steer_p says {provided.steer_p(est, k_off, k_head):+.3f}")
back = LaneEstimate(offset_m=0.4, heading_rad=-0.05, valid=True)
print(f"same offset, pointing 0.05 rad right, back towards the centre: {provided.steer_p(back, k_off, k_head):+.3f} rad")

# %% [markdown]
# What does the heading gain do? The next cell pushes the car: it steers 0.05 rad left for the first second, then hands
# over to the P law with `k_off = 0.35` and three values of `k_head`. The runs stop after 12 s, still on the first
# straight, where the provided box is almost exact.

# %%
PUSH, PUSH_T = 0.05, 1.0  # rad, s


def pushed(k_head):
    def controller(est, obs):
        return PUSH if obs.t < PUSH_T else provided.steer_p(est, 0.35, k_head)
    return run(Car(controller=controller), "gentle", max_steps=240)


plt.figure(figsize=(9, 3.5))
for k_head in (0.0, 1.0, 2.7):
    tel = pushed(k_head)
    outside = np.nonzero(np.abs(tel.lat) >= 0.05)[0]  # steps more than 5 cm from the centre
    print(f"k_head = {k_head}: pushed to {tel.lat.max():.2f} m, furthest right {tel.lat[tel.t >= PUSH_T].min():+.3f} m, "
          f"within 5 cm for good {tel.t[outside[-1] + 1] - PUSH_T:.2f} s after the push")
    plt.plot(tel.t, tel.lat, label=f"k_head = {k_head}")
plt.axhline(0, color="0.5", lw=1)
plt.xlabel("time (s)")
plt.ylabel("offset (m), left +")
plt.legend()
plt.show()

# %% [markdown]
# Without the heading term the car arrives at the centre pointing across the lane and swings 0.30 m past it. With
# `k_head = 1` it comes straight back; with 2.7 it creeps. The heading gain brakes the swing.
#
# ### Practice (not graded)
#
# Each problem gives you less than the one before. Replace each `None` with your code and rerun the cell.
#
# **Problem 1.** The car is 0.3 m *right* of centre and pointing 0.02 rad left. What steering angle does the P law give
# with `k_off = 0.35` and `k_head = 2.7`? Write the formula out; do not call `steer_p`.

# %%
steer_1 = None  # your code

practice("steering angle (rad)", steer_1, 0.051, tol=0.001)

# %% [markdown]
# **Problem 2.** On `curvy`'s right bends your lane has radius 18.2 m. What wheel angle holds that circle? Mind the
# sign: a right turn is negative. The wheelbase is `WHEELBASE`.

# %%
delta_right = None  # your code

practice("wheel angle to hold a right bend of radius 18.2 m (rad)", delta_right, -0.1473, tol=0.001)

# %% [markdown]
# **Problem 3.** With perfect lane values and gains 0.35 and 2.7, where would the car settle in that bend if it lasted
# long enough? (Lecture: "Why curves need error".)

# %%
offset_right = None  # your code

practice("settled offset in a right bend of radius 18.2 m (m)", offset_right, -0.715, tol=0.001)

# %% [markdown]
# ## 3. Graded exercise 0.1.4.a `p_steer`
#
# Write the P law of "The P controller" and "Two errors, two gains" as a function of the estimate and the two gains.
# The check tries six named cases (signs, both clips, a missing estimate) and 30 random ones.

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
# ## 4. Predict, then tune
#
# The next cell drives `gentle` with the provided box and your `p_steer`, keeping `k_head = 2.7` and sweeping `k_off`
# (lecture: "Tuning the gains"). It also counts the steps where the wheel is turning at its full 1 rad/s.
#
# **My prediction:** the smallest mean |lat| comes from `k_off` = ______, and the car leaves the road for
# `k_off` = ______, because ______.

# %%
try:
    for k_off in (0.05, 0.35, 1.5, 3.0, 6.0):
        tel = run(Car(controller=lambda est, obs, k_off=k_off: p_steer(est, k_off, 2.7)), "gentle")
        full_speed = np.mean(np.abs(np.diff(tel.steer)) / DT > 0.999)
        print(f"k_off = {k_off:4}: {ends(tel)}, mean |lat| {metrics(tel)['mean_abs_lat']:.3f} m, "
              f"wheel at full speed in {full_speed:.0%} of steps")
except NotImplementedError:
    print("Write p_steer above first, then run this cell again.")

# %% [markdown]
# More gain helps up to about 1.5, then hurts: 3.0 is worse, and at 6.0 the wheel, turning as fast as it can, is always
# late, so each swing is wider than the last. "More gain, tighter tracking" is the misconception.
#
# ## 5. Graded exercise 0.1.4.b `my_car`
#
# Every box of the car becomes yours. First, paste your `estimate_lane` from lab 0.1.3 into the next cell, replacing
# the skeleton. If it used your own `paint_masks`, paste that above it. The cell sees the same names as lab 0.1.3:
# `math`, `np`, `camera`, `paint_masks`, `NEAR_ROWS`, `FAR_ROWS`, `MIN_PIXELS`, `LaneEstimate`.

# %% tags=["exercise:0.1.4.b"]
def estimate_lane(obs):
    """Your perception box from lab 0.1.3: paste it here.

    Input:   obs, an Observation: obs.frame is the camera frame, a uint8 array of shape (180, 320, 3).
    Output:  LaneEstimate(offset_m, heading_rad, valid). offset_m > 0 when the car is LEFT of lane centre,
             heading_rad > 0 when the car points LEFT of the lane, both measured at the car (the front axle).
             valid=False, with math.nan for both numbers, when a band has fewer than MIN_PIXELS pixels of either line.
    """
    raise NotImplementedError


# %% [markdown]
# Now the controller and the car (lecture: "Closing the loop"). `my_controller` returns your `p_steer` with the gains
# below; a controller that returns a number is a steering angle, and the `Car` holds the speed for you. Then build
# `my_car` from your two boxes. Check `0.1.4.b` needs it to complete `gentle` with mean |lat| under 0.5 m, then drives
# it on `curvy` without grading.

# %% tags=["exercise:0.1.4.b"]
K_OFF, K_HEAD = 0.35, 2.7  # the provided gains; section 4 showed others


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
# ## 6. Inside the bend
#
# Why does the car leave `curvy`? The next cell prints, at points along the first left bend, where the car truly was,
# what the box reported, and the steering the P law (gains 0.35 and 2.7) gives on each (lecture: "Why your car leaves
# curvy"). First the provided car, which the lecture uses; then yours.

# %%
def inside_the_bend(tel, at=(24, 30, 36, 44, 48)):
    for s in at:
        if s > tel.progress_m[-1]:
            break
        k = int(np.argmax(tel.progress_m >= s))
        true = LaneEstimate(tel.lat[k], tel.head_err[k], True)
        box = LaneEstimate(tel.est_offset[k], tel.est_heading[k], bool(tel.est_valid[k]))
        print(f"{tel.progress_m[k]:5.1f} m | true {true.offset_m:+.2f} m, {true.heading_rad:+.3f} rad -> steer "
              f"{provided.steer_p(true):+.3f} | box {box.offset_m:+.2f} m, {box.heading_rad:+.3f} rad -> steer {provided.steer_p(box):+.3f}")


tel = run(Car(), "curvy")
print(f"the provided car on curvy: {ends(tel)}")
inside_the_bend(tel)
if isinstance(my_car, Car):
    try:
        tel = run(my_car, "curvy")
        print(f"\nyour car on curvy: {ends(tel)}")
        inside_the_bend(tel)
    except NotImplementedError:
        print("\nFinish estimate_lane and my_controller in section 5, then run this cell again.")
else:
    print("\nBuild my_car in section 5, then run this cell again.")

# %% [markdown]
# From 30 m on, the box says the car points much further right than it does, and the P law multiplies that heading by
# 2.7. On the true values the law would steer right, back to the centre; on the box's values it steers left, deeper
# inside.
#
# Can gains save it? The next cell drives `curvy` with the provided box and your `p_steer` at four pairs of gains.

# %%
try:
    for gains in ((0.35, 2.7), (1.5, 2.7), (0.35, 1.0), (3.0, 2.7)):
        tel = run(Car(controller=lambda est, obs, gains=gains: p_steer(est, *gains)), "curvy")
        print(f"k_off, k_head = {gains}: {ends(tel)}, mean |lat| {metrics(tel)['mean_abs_lat']:.3f} m")
except NotImplementedError:
    print("Write p_steer above first, then run this cell again.")

# %% [markdown]
# Two pairs get round. That hides the bias; it does not remove it. The box still reports a wrong heading in every
# bend, and a tighter bend makes it worse. The fixes come in Loop 1: a lane model that bends (module 1.5) and a
# controller that looks ahead (module 1.3).
#
# ## 7. Break it, fix it
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
print(f"{ends(tel)}")
for k in range(0, len(tel.t), 10):
    print(f"t = {tel.t[k]:3.1f} s: offset {tel.lat[k]:+.3f} m, heading {tel.head_err[k]:+.3f} rad, wheel {tel.steer[k]:+.3f} rad")
dashboard(tel)

# %% [markdown]
# <details><summary>Hint (open after you have a guess)</summary>
#
# `est.heading_rad` is the car's heading relative to the lane, not the road's turn. A car pointing right
# (negative heading) needs to steer left, so the heading term must be `-2.7 * est.heading_rad`. With the sign flipped,
# pointing right makes the teammate steer right, which points the car further right: at 2.0 s the heading is
# −0.020 rad, half a second later −0.260 rad, and the car leaves the road at 21.1 m. That is positive feedback on the
# heading. The fix is `-(0.35 * est.offset_m + 2.7 * est.heading_rad)`.
#
# </details>
#
# ## What's next
#
# Every box of your car is now yours, and you have seen where each one falls short: a straight-line lane model and a
# controller that only reacts. Loop 1 opens each box properly, starting with the transforms the rest of the course
# builds on (module 1.1).
