# %% [markdown]
# # Lab: The view from the driver's seat
#
# You will write the three functions the lecture ends with: `rotation_matrix`, `world_to_ego` and `ego_to_world`. You will check
# `world_to_ego` against the simulator's own transform at 40 places and headings, put it into the camera so that every
# picture your car sees passes through your code, and find out why a test at yaw 0 proves nothing.
# Read the lecture first; this lab points back to its sections by title and follows its order.
#
# Run the cells from top to bottom (Shift+Enter). Each problem has a "Guess first" line: write your guess down before you
# run the cell, then compare. `practice` prints a green tick or tells you what it expected. A "Break it" cell makes a
# mistake on purpose, so a red cross there is what it is meant to show.
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
from zero2fsd.geom import forward_left, heading_vector, wrap_angle
from zero2fsd.grade import check, practice
from zero2fsd.preview.units.u1_1_2 import PICTURES, differing_pictures
from zero2fsd.sim import BicycleState, Camera, make_scenario
from zero2fsd.sim.camera import WHITE_LINE_LATERAL
from zero2fsd.sim.world import EGO_LANE_CENTER

# %% [markdown]
# ## The scene
#
# The lecture's scene is the provided car 150 m along `gentle`, at step 378 of its lap. The next cell finds that moment and
# the lane-centre points 10, 20 and 30 m further along the lane. `SCENE_POSE` is the car's pose `(x, y, theta)`;
# `lane_points` has one row per point, east part first.

# %%
road = make_scenario("gentle").road
lap = run(Car(), "gentle", max_steps=379)
SCENE_POSE = (float(lap.x[378]), float(lap.y[378]), float(lap.yaw[378]))
x, y, theta = SCENE_POSE
t = np.array([x, y])
s_car, _ = road.project(x, y)
lane_points = np.array([road.offset(s_car + ahead, EGO_LANE_CENTER) for ahead in (10.0, 20.0, 30.0)])
p = lane_points[0]
print(f"car at ({x:.3f}, {y:.3f}), theta {theta:.4f} rad; first lane-centre point at ({p[0]:.3f}, {p[1]:.3f})")

# %% [markdown]
# ## Two dot products make a frame change
#
# The arrow from the car to the first lane-centre point is $\mathbf{v} = \mathbf{p} - \mathbf{t}$. *Positions and headings*
# turned an arrow like it into "ahead and to the left" with `forward_left`.
#
# **Problem 1.** Compute `v`, then `forward_left(theta, v)`, and unpack the pair into `forward` and `left`.
#
# **Guess first:** how far ahead of the car is the point, and how far to its left? Compare with the lecture's 10.189 m and
# 0.453 m.

# %%
v = None  # your code: the arrow from the car to the point
forward = left = None  # your code: the two parts of v, from forward_left

practice("forward (m)", forward, 10.189, tol=0.001)
practice("left (m)", left, 0.453, tol=0.001)

# %% [markdown]
# **Break it.** Take the dot product with the left arrow first and the heading arrow second, and the pair comes out swapped.
# The cell makes that mistake; the red cross is what you should see.

# %%
h = heading_vector(theta)
n = np.array([-math.sin(theta), math.cos(theta)])
swapped = (float(v @ n), float(v @ h)) if v is not None else None
practice("forward, left with the arrows in the wrong order", swapped, (10.189, 0.453), tol=0.001)

# %% [markdown]
# ![](figures/two_dots.png)
#
# ![](figures/stacked_dots.png)

# %% [markdown]
# ## The rotation matrix
#
# **Problem 2.** Fill in `rotation_matrix(theta)`: the (2, 2) array whose first column is the heading arrow
# $(\cos\theta, \sin\theta)$ and whose second column is the left arrow $(-\sin\theta, \cos\theta)$.
#
# **Guess first:** what is `rotation_matrix(0)`?


# %%
def rotation_matrix(theta):
    """The (2, 2) matrix whose columns are the heading arrow and the left arrow.

    Input:   theta, the yaw in radians.
    Output:  a (2, 2) array.
    Example: rotation_matrix(0)  ->  [[1, 0], [0, 1]]
    """
    raise NotImplementedError


R = rotation_matrix(theta)
left_arrow = np.array([-math.sin(theta), math.cos(theta)])
practice("column 1 minus the heading arrow (largest entry)", float(np.abs(R[:, 0] - heading_vector(theta)).max()), 0, tol=1e-12)
practice("column 2 minus the left arrow (largest entry)", float(np.abs(R[:, 1] - left_arrow).max()), 0, tol=1e-12)

# %% [markdown]
# **Break it.** Write the sine in the wrong places, `[[c, s], [-s, c]]`. The columns stop being the two arrows.

# %%
c, s = math.cos(theta), math.sin(theta)
wrong = np.array([[c, s], [-s, c]])
practice("column 1 minus the heading arrow, wrong matrix", float(np.abs(wrong[:, 0] - heading_vector(theta)).max()), 0, tol=1e-12)

# %% [markdown]
# ![](figures/columns_axes.png)

# %% [markdown]
# ## Undo is the transpose
#
# **Problem 3.** Write `world_to_ego(pose, p_world)` and `ego_to_world(pose, p_ego)`. The lecture's formulas are
# $R(\theta)^{\mathsf T}(\mathbf{p} - \mathbf{t})$ and $R(\theta)\,\mathbf{q} + \mathbf{t}$. Use `rotation_matrix` and write
# the subtraction before the rotation. Keep the arrays as rows for now: one point of shape `(2,)` is enough, and the next
# section makes it work on every point at once.
#
# **Guess first:** what does `world_to_ego` give for the car's own position?
#
# The first function below is the one Exercise A and Exercise B check, so its cell carries the exercise tags.


# %% tags=["exercise:1.1.2.a", "exercise:1.1.2.b"]
def world_to_ego(pose, p_world):
    """World point(s) in the car's own frame: how far ahead of the car, and how far to its left.

    Input:   pose = (x, y, theta), the car's position and yaw; p_world, an array of shape (..., 2), one point per row,
             east part first.
    Output:  an array of the same shape (..., 2): forward, then left, in metres.
    Example: world_to_ego(SCENE_POSE, [156.018, 20.391])  ->  [10.189..., 0.453...]
    """
    raise NotImplementedError


# %%
def ego_to_world(pose, p_ego):
    """Points given in the car's own frame (forward, left) back in the world: R(theta) q + t, one point per row."""
    raise NotImplementedError


practice("scene point, forward then left", world_to_ego(SCENE_POSE, p), (10.189, 0.453), tol=0.001)
practice("largest entry of R transposed times R, minus the identity", float(np.abs(R.T @ R - np.eye(2)).max()), 0, tol=1e-12)

# %% [markdown]
# ![](figures/undo.png)
#
# **Break it.** Rotate first and subtract second: $R(\theta)^{\mathsf T}\mathbf{p} - \mathbf{t}$. The cell does it for the
# same point. The lecture's answer is (-7.260, -84.974).

# %%
rotate_first = R.T @ p - t
practice("rotate first, then subtract t", tuple(float(c) for c in rotate_first), (10.189, 0.453), tol=0.001)

# %% [markdown]
# ![](figures/order_trap.png)
#
# Now the lap. The next cell sends the 2109 road points of `gentle` into the car's frame and back out. Neither number is
# exactly 0, because every multiply-and-add rounds in the last digit.

# %%
cloud = np.concatenate([road.sample_points(1.0, d) for d in (0.0, EGO_LANE_CENTER, WHITE_LINE_LATERAL)])
round_trip = float(np.abs(ego_to_world(SCENE_POSE, world_to_ego(SCENE_POSE, cloud)) - cloud).max())
print(f"{len(cloud)} points")
practice("largest round-trip error (m)", round_trip, 0, tol=1e-12)

# %% [markdown]
# ![](figures/round_trip.png)

# %% [markdown]
# ## Every point at once
#
# The camera calls `world_to_ego` three times per picture, each time with a whole array: 146 road quads of 4 corners, the
# same for the white line, and 103 yellow dashes.
#
# **Problem 4.** Check that your `world_to_ego` returns the same shape it was given, for the road's array of shape
# `(146, 4, 2)`, and that its answers for the three lane-centre points match the lecture's table.
#
# **Guess first:** does `rot @ rows` work with 2 points? With 5?

# %%
quads = np.zeros((146, 4, 2)) + lane_points[0]
practice("output shape for the road quads", world_to_ego(SCENE_POSE, quads).shape, (146, 4, 2))

table = world_to_ego(SCENE_POSE, lane_points)
practice("the three lane-centre points: forward, then left", table, [[10.189, 0.453], [20.158, 2.694], [29.771, 6.160]], tol=0.001)

# %% [markdown]
# ![](figures/quads.png)
#
# **Break it.** The maths wrote arrows as columns, so `rot @ points` looks right. With rows it is not.

# %%
rot = np.array([[c, -s], [s, c]])
try:
    rot @ lane_points[:2].copy()
    print("2 points: no error, and the answer is wrong")
except ValueError as err:
    print("2 points:", err)
try:
    rot @ np.zeros((5, 2))
except ValueError as err:
    print("5 points: ValueError, the shapes do not line up")
print("rows @ rot works for both:", (lane_points[:2] - t) @ rot)

# %% [markdown]
# ![](figures/shape_trap.png)
#
# **Exercise A.** The check puts a point 4 m ahead and 1 m left of a car at yaws 0, π/2, π and −3π/4, where it must come out
# as (4, 1); then tries the `(K, 4, 2)` shape; then the round trip; then 40 seeded poses, with yaws up to ±4π, against the
# simulator's own transform. Expect a largest difference of about 5.7e-14 m for the matrix form
# `(p_world - t) @ rot`; the check allows 1e-9 m.

# %%
check("1.1.2.a", world_to_ego)

# %% [markdown]
# ## Swap your transform into the renderer
#
# The camera draws with `geom.world_to_ego` unless you give it another: `Camera(world_to_ego=your_function)`.
#
# **Problem 5.** Take 18 evenly spaced moments of the provided car's lap of `gentle`, draw each through the simulator's
# transform and through yours, and count the pictures that differ.
#
# **Guess first:** how many of the 18 pictures will differ, and by how many pixels?

# %%
lap_full = run(Car(), "gentle")
pictures, pixels = differing_pictures(world_to_ego, lap_full)
print(f"{pictures} of {PICTURES} pictures differ, {pixels} pixels")
practice("pictures that differ", pictures, 0, tol=0)

# %% [markdown]
# ![](figures/renderer_steps.png)

# %% [markdown]
# **Exercise B** puts your function in the camera and drives `gentle`. The provided car must complete it with a mean
# distance from the lane centre under 0.5 m. With the right transform the lap is identical to the provided one, near 0.376 m.

# %%
check("1.1.2.b", world_to_ego)

# %% [markdown]
# ![](figures/swap_same.png)

# %% [markdown]
# ## A test at yaw 0 proves nothing
#
# Two broken transforms: the swap, `(p - t) @ rot.T` (R where its transpose belongs), and no rotation, `p - t`. Each is
# right when the car points along east.
#
# **Problem 6.** Run both on the lecture's named poses, then count how much of the parked car's picture each changes at
# three places on the lap.
#
# **Guess first:** at which yaws will each one give (4, 1)?

# %%
def swap(pose, p_world):
    x, y, theta = pose
    return (np.asarray(p_world, dtype=float) - np.array([x, y])) @ np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]).T


def no_rotation(pose, p_world):
    x, y, theta = pose
    return np.asarray(p_world, dtype=float) - np.array([x, y])


for label, yaw in (("0", 0.0), ("pi/2", math.pi / 2), ("pi", math.pi), ("-3pi/4", -3 * math.pi / 4)):
    point = np.array([10.0, 5.0]) + 4.0 * heading_vector(yaw) + 1.0 * np.array([-math.sin(yaw), math.cos(yaw)])
    print(f"yaw {label:>7}: swap {np.round(swap((10.0, 5.0, yaw), point), 3)}, no rotation {np.round(no_rotation((10.0, 5.0, yaw), point), 3)}")


# %% [markdown]
# Parked on the lane centre, the car sees a different picture through a broken transform. `parked_share` is the percent of
# the picture's pixels, by what they truly are, that the broken transform changes.

# %%
def parked_share(fn, s):
    state = BicycleState(*road.ego_pose(s), 8.0, 0.0)
    good = Camera().render(road, state, return_labels=True)[1]
    bad = Camera(world_to_ego=fn).render(road, state, return_labels=True)[1]
    return 100.0 * float((good != bad).mean())


practice("swap at the start line (%)", parked_share(swap, 0.0), 0.0, tol=0.05)
practice("swap 150 m along (%)", parked_share(swap, 150.0), 33.9, tol=0.05)
practice("no rotation 150 m along (%)", parked_share(no_rotation, 150.0), 20.0, tol=0.05)

# %% [markdown]
# ![](figures/pictures_150.png)
#
# ![](figures/share_vs_s.png)
#
# The swap changes nothing at the start line, where the car points along east, yet it leaves `straight` after about three
# seconds. The next cell drives it.

# %%
swap_run = run(Car(), "straight", camera=Camera(world_to_ego=swap))
print(f"completed: {swap_run.completed}; left the road after {swap_run.t[-1]:.2f} s, {swap_run.progress_m[-1]:.1f} m along")
practice("seconds before the swap leaves straight", float(swap_run.t[-1]), 2.95, tol=0.05)

# %% [markdown]
# ![](figures/e_vs_s.png)
#
# ![](figures/heading_flip.png)
#
# ![](figures/park_box.png)
#
# **Fix it.** The way to catch both mistakes is the one Exercise A already uses: yaws where sin θ is well away from 0, and
# many random poses. Your `world_to_ego` passed it, and the picture test, so the camera now draws through your code.
# The lecture's last figure shows what the next lesson is about.
#
# ![](figures/next_lesson.png)
#
# The lecture's two challenges end with these figures: the swap seen through time, and how the error grows.
#
# ![](figures/challenge_swing.png)
#
# ![](figures/challenge_growth.png)
