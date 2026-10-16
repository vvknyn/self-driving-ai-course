# %% [markdown]
# # Lab: Positions and headings
#
# You will turn the car's logged positions into arrows, write the heading arrow, find an arrow's direction without
# getting it backwards half the time, wrap an angle, and split an arrow into the parts ahead of the car and to its left.
# Read the lecture first; this lab points back to its sections by title and follows its order.
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

import numpy as np

from zero2fsd.car import Car, run
from zero2fsd.grade import check, practice
from zero2fsd.sim import BicycleState, make_scenario
from zero2fsd.sim.camera import to_vehicle
from zero2fsd.sim.scenarios import TARGET_SPEED
from zero2fsd.sim.world import wrap_angle as sim_wrap

# %% [markdown]
# ## 1. Points and arrows
#
# The next cell drives the provided car once round `gentle` and keeps what the lecture used. `P` holds the front axle's
# position at every step, shape `(1778, 2)`: column 0 is $x$ (east), column 1 is $y$ (north). `D` holds the arrow from
# each position to the next, shape `(1777, 2)`. `yaw` is the car's yaw at every step, in radians. `road` is the road
# itself, which you use for the cone in section 5.

# %%
tel = run(Car(), "gentle")
road = make_scenario("gentle").road
P = np.stack([tel.x, tel.y], axis=-1)
D = np.diff(P, axis=0)
yaw = tel.yaw
print("positions:", P.shape, " step arrows:", D.shape)

# %% [markdown]
# **Problem 1.** Find the length of every step arrow with Pythagoras (`np.hypot` does it), as `L`, a vector of 1777
# lengths. Then read off the length of the step-700 arrow and of the shortest step on the lap.
#
# **Guess first:** the car drives 8 m/s and the simulator steps 0.05 s. How long is each arrow, and will the shortest
# differ from the others?

# %%
L = None  # your code: the length of every step arrow, from D[:, 0] and D[:, 1]

if L is not None:
    practice("length of the step-700 arrow (m)", L[700], 0.4000, tol=1e-4)
    practice("shortest step on the lap (m)", L.min(), 0.39999947, tol=1e-8)

# %% [markdown]
# ![](figures/world_lap.png)
#
# ![](figures/step_arrow.png)
#
# ## 2. The heading arrow
#
# **Problem 2.** Write `heading_vector(theta)`: the 1 m arrow $(\cos\theta, \sin\theta)$ at angle `theta`. It must take
# one angle or an array of angles, and put the two parts in the last axis, so an array of 3 angles gives shape `(3, 2)`.
#
# **Guess first:** the yaw at step 700 is 2.1812 rad, a little past a quarter turn. Is the east part positive or
# negative?

# %%
def heading_vector(theta):
    """The 1 m arrow (cos theta, sin theta); shape (..., 2) for angles of shape (...)."""
    return None  # your code: np.asarray(theta, float), then np.stack([...], axis=-1)

practice("heading arrow at step 700", heading_vector(yaw[700]), np.array([-0.5732, 0.8194]), tol=1e-4)
practice("shape for three angles", None if heading_vector(np.zeros(3)) is None else heading_vector(np.zeros(3)).shape, (3, 2))

# %% [markdown]
# ![](figures/theta_psi.png)
#
# ![](figures/unit_circle.png)
#
# ## 3. Which way is it pointing
#
# **Problem 3.** Compute the angle the lecture first tried, `np.arctan(dy / dx)`, for every step arrow, as `naive`.
# Then count the steps where it misses the car's yaw by more than 1 rad. Use `sim_wrap`, the simulator's own wrap, on the
# difference: you write your own in section 4. (Dividing by $\Delta x = 0$ is harmless here, so the cell silences the
# warning.)
#
# **Guess first:** what share of the 1777 steps will be more than 1 rad off: none, a few, or about half?

# %%
with np.errstate(divide="ignore", invalid="ignore"):
    naive = None  # your code: np.arctan of D[:, 1] / D[:, 0]

if naive is not None:
    n_off = int((np.abs(sim_wrap(naive - yaw[:-1])) > 1).sum())
    practice("steps more than 1 rad off", n_off, 888, tol=0)

# %% [markdown]
# ![](figures/naive_700.png)
#
# ![](figures/naive_lap.png)
#
# ![](figures/opposite_arrows.png)
#
# **Exercise A.** Write `direction_angle(v)`, the direction of an arrow `v` with `np.arctan2`. Two details matter: `v` can
# be one arrow of shape `(2,)` or many of shape `(N, 2)`, and the zero arrow has no direction, so it must give `nan`, not
# 0.

# %% tags=["exercise:1.1.1.a"]
def direction_angle(v):
    """The direction of an arrow, in radians anticlockwise from east, in [-pi, pi].

    Input:   v, an arrow (v_x, v_y) of shape (2,), or many arrows of shape (N, 2).
    Output:  the angle(s), a float for one arrow and shape (N,) for many; nan for the zero arrow (0, 0).
    Example: direction_angle(np.array([3.0, 4.0]))  ->  0.9272952180016122
    """
    raise NotImplementedError

# %% [markdown]
# **Guess first (Exercise A):** how far from the yaw is your direction in the worst step of the lap: about 0.04 rad, 0.4
# rad, or 4 rad?

# %%
check("1.1.1.a", direction_angle)
try:
    practice("direction of (3, 4)", direction_angle(np.array([3.0, 4.0])), 0.92729522, tol=1e-8)
    practice("the zero arrow gives nan", bool(np.isnan(direction_angle(np.zeros(2)))), True)
    practice("worst miss over the lap (rad)", np.abs(sim_wrap(direction_angle(D) - yaw[:-1])).max(), 0.040994, tol=1e-6)
except NotImplementedError:
    print("Write direction_angle above first, then run this cell again.")

# %% [markdown]
# ![](figures/atan2_lap.png)
#
# ## 4. Angles go round
#
# **Problem 4.** Subtract the yaw from your direction, `direction_angle(D) - yaw[:-1]`, with no wrap, and count the
# steps where the result is more than 1 rad from 0.
#
# **Guess first:** the directions are within 0.04 rad of the yaw (Exercise A). So how many steps differ by more than 1
# rad: 0, or a lot?

# %%
n_plain = None  # your code: the number of steps where |direction_angle(D) - yaw[:-1]| > 1

practice("steps more than 1 rad off, unwrapped", n_plain, 722, tol=0)

# %% [markdown]
# ![](figures/raw_diff.png)
#
# ![](figures/yaw_wrapped.png)
#
# **Problem 5.** Write `wrap_angle(a)`: the same angle shifted by whole turns into $[-\pi, \pi)$. Use
# $(a + \pi) \bmod 2\pi - \pi$ with Python's `%` (not `math.fmod`, which keeps the sign of its input).
#
# **Guess first:** the yaw at the end of the lap is 6.226408 rad, just under a full turn. Will its wrap be near 0, near
# $\pi$, or near $2\pi$?

# %%
def wrap_angle(a):
    """The angle a shifted by whole turns into [-pi, pi)."""
    return None  # your code: (a + math.pi) % (2 * math.pi) - math.pi

probe = np.concatenate([np.random.default_rng(1111).uniform(-40, 40, 100_000), np.arange(-8, 9) * math.pi, [math.pi, -math.pi, 0.0, 1e-300]])

practice("wrap of the final yaw", wrap_angle(yaw[-1]), -0.056778, tol=1e-6)
practice("wrap of pi", wrap_angle(np.pi), -np.pi, tol=1e-12)
practice("same as the simulator's wrap", None if wrap_angle(probe) is None else bool(np.array_equal(wrap_angle(probe), sim_wrap(probe))), True)

# %% [markdown]
# ![](figures/wrap_saw.png)
#
# ## 5. Ahead and to the left
#
# The cone is 12 m further along the road than the car at step 700 and 4.3 m to the right of the road's centre line. The
# next cell finds it from the road; the lecture rounds it to $(159.131, 140.083)$ but the lab keeps the exact values.

# %%
s_car, _ = road.project(tel.x[700], tel.y[700])
cone = road.offset(s_car + 12.0, -4.3)
print("cone:", cone)

# %% [markdown]
# **Guess first:** the car at step 700 faces north-west, and the cone is ahead and to its right. Facing exactly north,
# $\theta = \pi/2$, an arrow of 2 m east and 3 m north: how far ahead is it, and how far to the left?
#
# ![](figures/north_hand.png)
#
# ![](figures/left_arrow.png)
#
# **Exercise B.** Write `forward_left(theta, v)`: the forward part $\vec v \cdot \hat h$ and the left part
# $\hat h \times \vec v = \cos\theta\, v_y - \sin\theta\, v_x$ of the world arrow `v`, for a car heading `theta`. It must
# work for one heading and one arrow, and for arrays: `theta` of shape `(N,)` with `v` of shape `(N, 2)` gives `(N, 2)`.

# %% tags=["exercise:1.1.1.b"]
def forward_left(theta, v):
    """The forward and left parts of the world arrow v, for a car whose yaw is theta.

    Input:   theta, the yaw in radians (shape (...)); v, the arrow (v_x, v_y) in the world frame (shape (..., 2)).
    Output:  shape (..., 2): column 0 is how far ahead of the car, column 1 how far to its left (negative: to its right).
    Example: forward_left(np.pi / 2, np.array([2.0, 3.0]))  ->  array([ 3., -2.])
    """
    raise NotImplementedError

# %% [markdown]
# **Guess first (Exercise B):** the cone is 12 m ahead along the road but the car is turning left. Is the cone's left part
# positive or negative?

# %%
check("1.1.1.b", forward_left)
try:
    practice("facing north", forward_left(np.pi / 2, np.array([2.0, 3.0])), np.array([3.0, -2.0]), tol=1e-12)
    practice("the cone at step 700", forward_left(yaw[700], cone - P[700]), np.array([12.6579, -1.6678]), tol=1e-4)
    practice("same as the camera's to_vehicle", forward_left(yaw[700], cone - P[700]),
             np.array(to_vehicle(BicycleState(tel.x[700], tel.y[700], yaw[700], TARGET_SPEED, 0.0), *cone)), tol=1e-9)
except NotImplementedError:
    print("Write forward_left above first, then run this cell again.")

# %% [markdown]
# ![](figures/cone_parts.png)
#
# ## 6. Moving is not pointing
#
# **Problem 6.** The lecture's rule says a step's direction is $\theta + \delta + \tfrac{1}{2}\Delta\theta$: the yaw at the
# start, plus the wheel angle $\delta$ (`tel.steer[k + 1]` for step $k$), plus half the yaw change over the step. Build
# `pred` for every step and find its largest miss against `direction_angle(D)`, wrapping the difference.
#
# **Guess first:** will the rule match to within 0.01 rad, or to within a hair of rounding?

# %%
pred = None  # your code: yaw[:-1] + tel.steer[1:] + 0.5 * np.diff(yaw)

if pred is not None:
    practice("largest miss of the rule (rad)", np.abs(wrap_angle(direction_angle(D) - pred)).max(), 0.0, tol=1e-12)

# %% [markdown]
# The wheel angle is the one logged for the *end* of the step. Try `tel.steer[:-1]` in place of `tel.steer[1:]` and the
# rule misses by 0.011 rad.
#
# Now split every step arrow into its forward and left parts with your `forward_left`, and look at the two columns. The
# car moves 0.4 m a step, nearly all of it forward, and only a little to the side.
#
# **Guess first:** will the left part of a step be closer to 0.4 m, to 0.02 m, or to 0.0001 m?

# %%
try:
    fl = forward_left(yaw[:-1], D)
    practice("mean forward per step (m)", fl[:, 0].mean(), 0.3998, tol=1e-4)
    practice("largest left per step (m)", fl[:, 1].max(), 0.0164, tol=1e-4)
except NotImplementedError:
    print("Write forward_left above first, then run this cell again.")

# %% [markdown]
# ![](figures/move_vs_point.png)
#
# ![](figures/move_point_diagram.png)
#
# ## 7. Break it, fix it
#
# Each cell below holds a teammate's function with one bug. Run it, read what `practice` expected, find the bug and fix it.
#
# **Bug 1.** A teammate wrote `wrap` with `math.fmod`. The wrap of $-3.5$ should be $+2.783185$.

# %%
def teammate_wrap(a):
    return math.fmod(a + math.pi, 2 * math.pi) - math.pi

practice("teammate's wrap of -3.5", teammate_wrap(-3.5), 2.783185, tol=1e-6)

# %% [markdown]
# **Bug 2.** A teammate computed the left part for every step, but the largest should be 0.0164 m.

# %%
def teammate_left(theta, v):
    h = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    return v[..., 0] * h[..., 1] - v[..., 1] * h[..., 0]

practice("largest left per step (m)", teammate_left(yaw[:-1], D).max(), 0.0164, tol=1e-4)

# %% [markdown]
# **Bug 3.** A different teammate dotted each step with the left arrow, written as `(sin, cos)`. The largest left part
# should again be 0.0164 m.

# %%
def teammate_left_arrow(theta):
    return np.stack([np.sin(theta), np.cos(theta)], axis=-1)

practice("largest left per step (m)", (D * teammate_left_arrow(yaw[:-1])).sum(axis=-1).max(), 0.0164, tol=1e-4)

# %% [markdown]
# **Bug 4.** The playground offers a wrong answer to Exercise A under the preset name `naive_arctan`: the lecture's first
# try, `np.arctan(dy / dx)`. Run it there and read the verdict: it misses the yaw by more than 0.1 rad on 888 steps,
# and its worst miss is 3.142 rad. Then fix it.
#
# ![](figures/next_lesson.png)
#
# ## 8. Stretch: the challenge problems
#
# **Challenge 1.** Take the car's path on the `straight` road, add a random error of standard deviation
# $\sigma = 0.05$ m to each coordinate of each position, and take the direction from the fix at step $i$ to the fix at step
# $i + k$. The lecture shows the spread of the direction's error is about $\sqrt{2}\sigma / (0.4k)$. The cell simulates it
# for $k = 18$ over 500 random seeds.
#
# ![](figures/gps_law.png)

# %%
sigma, k = 0.05, 18
straight = run(Car(), "straight")
fixes = np.stack([straight.x, straight.y], axis=-1)
errors = []
for seed in range(500):
    noisy = fixes + np.random.default_rng(seed).normal(0.0, sigma, fixes.shape)
    errors.append(None)  # your code: wrap(direction_angle(noisy[k:] - noisy[:-k]) - straight.yaw[:-k])

if errors[0] is not None:
    practice("spread of the heading error at k = 18 (rad)", np.concatenate(errors).std(), 0.00980, tol=1e-4)

# %% [markdown]
# **Challenge 2.** Write `atan2_by_hand(v)` using only `np.arctan` of the slope and `np.where`. The cell compares it with
# `np.arctan2` on 30 seeded arrows.

# %%
def atan2_by_hand(v):
    """The direction of v from np.arctan and np.where only: all four quarters, +-pi/2 when v_x = 0, nan for (0, 0)."""
    return None  # your code

seeded = np.random.default_rng(1111).normal(0, 5, (30, 2))

if atan2_by_hand(seeded) is not None:
    practice("atan2_by_hand on 30 seeded arrows, largest difference", np.abs(atan2_by_hand(seeded) - np.arctan2(seeded[:, 1], seeded[:, 0])).max(), 0.0, tol=1e-12)

# %% [markdown]
# ## What's next
#
# You can now put any position and direction on the road's map, and say how far ahead of the car and how far to its left
# an object is. In the next lesson, *The view from the driver's seat*, you use these parts to write every lane line in the
# car's own coordinates, so the camera can draw it.
