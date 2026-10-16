# %% [markdown]
# # Lab: The view from the driver's seat
#
# You will write the three functions the lecture builds: `rotation_matrix`, `world_to_ego` and `ego_to_world`. You will check
# `world_to_ego` against the simulator's own transform at 40 places and yaws, put it into the camera so that every
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
from zero2fsd import geom
from zero2fsd.geom import forward_left, heading_vector
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
# **Guess first:** how far ahead of the car is the point, and how far to its left? Write down two numbers before you run the
# cell below; its ticks show what the right pair is.

# %%
v = None  # your code: the arrow from the car to the point
forward = left = None  # your code: the two parts of v, from forward_left

practice("forward (m)", forward, 10.189, tol=0.001)
practice("left (m)", left, 0.453, tol=0.001)

# %% [markdown]
# **Break it.** Take the dot product with the left arrow first and the heading arrow second, and the pair comes out swapped.
# The cell makes that mistake, so a red cross is what you should see here.

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
    Example: rotation_matrix(math.pi / 2)  ->  [[0, -1], [1, 0]]
    """
    raise NotImplementedError


R = rotation_matrix(theta)
left_arrow = np.array([-math.sin(theta), math.cos(theta)])
practice("column 1 minus the heading arrow (largest entry)", float(np.abs(R[:, 0] - heading_vector(theta)).max()), 0, tol=1e-12)
practice("column 2 minus the left arrow (largest entry)", float(np.abs(R[:, 1] - left_arrow).max()), 0, tol=1e-12)

# %% [markdown]
# **Break it.** Write the sine in the wrong places, `[[c, s], [-s, c]]`. The columns stop being the two arrows, so a red
# cross is what you should see here. This matrix is $R(\theta)^{\mathsf T}$, which is what the next section needs.

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
# $R(\theta)^{\mathsf T}(\mathbf{p} - \mathbf{t})$ and $R(\theta)\,\mathbf{q} + \mathbf{t}$. Use `rotation_matrix`, and write the
# subtraction before the rotation. Write `world_to_ego` for points stored as rows, as `(p_world - t) @ R`: the cell after the
# two functions sends 2109 points through them at once, and *Every point at once* explains why rows go on the left. You can
# also build it for one point from your own `forward_left` of *Positions and headings*; the camera needs the form that takes
# a whole array.
#
# **Guess first:** what does `world_to_ego` give for the car's own position? The cell prints it after the two functions.
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


print("the car's own position, in the car's frame:", world_to_ego(SCENE_POSE, t))
practice("scene point, forward then left", world_to_ego(SCENE_POSE, p), (10.189, 0.453), tol=0.001)
practice("largest entry of R transposed times R, minus the identity", float(np.abs(R.T @ R - np.eye(2)).max()), 0, tol=1e-12)

# %% [markdown]
# ![](figures/undo.png)
#
# **Break it.** Rotate first and subtract second: $R(\theta)^{\mathsf T}\mathbf{p} - \mathbf{t}$. The cell does it for the
# same point, so a red cross is what you should see here.

# %%
rotate_first = R.T @ p - t
practice("rotate first, then subtract t", tuple(float(c) for c in rotate_first), (10.189, 0.453), tol=0.001)
print("the lecture's answer for this mistake is (-7.260, -84.974)")

# %% [markdown]
# ![](figures/order_trap.png)
#
# Now the lap. The next cell sends the 2109 road points of `gentle` into the car's frame and back out. The error it prints
# is not exactly 0, because every multiply-and-add rounds in the last digit.

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
# The camera calls `world_to_ego` three times per picture, each time with a whole array: one of shape (146, 4, 2) for the
# road, one of the same shape for the white line, and one of shape (103, 4, 2) for the yellow centre dashes.
#
# **Problem 4.** Send the road's array of shape `(146, 4, 2)` and the three lane-centre points through your `world_to_ego`
# and check that the shape comes back unchanged and that the three answers match the lecture's table.

# %%
quads = np.zeros((146, 4, 2)) + lane_points[0]  # an array shaped like the road's (146, 4, 2), every corner at one point
ego_quads = None  # your code: the road's array through world_to_ego
ego_lane = None  # your code: the three lane-centre points through world_to_ego

practice("output shape for the road quads", None if ego_quads is None else ego_quads.shape, (146, 4, 2))
practice("the three lane-centre points: forward, then left", ego_lane, [[10.189, 0.453], [20.158, 2.694], [29.771, 6.160]], tol=0.001)

# %% [markdown]
# ![](figures/quads.png)
#
# **Break it.** The maths wrote arrows as columns, so `rot @ points` looks right. With rows it is not.
#
# **Guess first:** the cell tries 2, 3 and 146 points. Which of the three raises an error?

# %%
rot = np.array([[c, -s], [s, c]])
for k in (2, 3, 146):
    rows = np.tile(lane_points[0], (k, 1)) - t
    try:
        wrong_way = rot @ rows
    except ValueError:
        print(f"{k} points: ValueError, the shapes do not line up")
    else:
        print(f"{k} points: no error, and the answer is {'right' if np.allclose(wrong_way, rows @ rot) else 'wrong'}")
print("rows @ rot, for the first two points:", (lane_points[:2] - t) @ rot)

# %% [markdown]
# ![](figures/shape_trap.png)
#
# **Exercise A.** The check puts a point 4 m ahead and 1 m left of a car at yaws 0, π/2, π and −3π/4, where it must come out
# as (4, 1); then tries the `(K, 4, 2)` shape; then the round trip through the provided `ego_to_world`; then 40 seeded
# poses, with yaws up to ±4π, against the simulator's own transform. The matrix form `(p_world - t) @ rot` differs from it
# by about 5.7 × 10⁻¹⁴ m at most; the check allows 10⁻⁹ m.

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
# `differing_pictures(your_function, lap)` returns the count of differing pictures and the count of differing pixels.
#
# **Guess first:** how many of the 18 pictures will differ, and by how many pixels?

# %%
lap_full = run(Car(), "gentle")
pictures, pixels = None, None  # your code: differing_pictures(world_to_ego, lap_full)
if pictures is not None:
    print(f"{pictures} of {PICTURES} pictures differ, {pixels} pixels")
practice("pictures that differ", pictures, 0, tol=0)

# %% [markdown]
# ![](figures/renderer_steps.png)

# %% [markdown]
# **Exercise B** first compares your function with the renderer's transform entry by entry at 40 seeded poses, then puts it in the camera and drives `gentle`. The provided car must complete it with a mean
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
# **Problem 6.** Write both as functions with the signature of `world_to_ego`, run them on the named poses, then count how
# much of the parked car's picture each changes at three places on the lap.
#
# **Guess first:** the loop below also tries yaw −π/2 and yaw 2π. Which of the two do the swap and the dropped rotation
# get right?

# %%
def swap(pose, p_world):
    """The swap: the rows times the transpose of the matrix, `(p - t) @ rotation_matrix(theta).T`. Rows in, rows out."""
    raise NotImplementedError


def no_rotation(pose, p_world):
    """No rotation at all: `p - t`. Rows in, rows out."""
    raise NotImplementedError


def named_case(yaw):
    """A car at (10, 5) with this yaw, and the point 4 m ahead and 1 m left of it: the answer is (4, 1) at every yaw."""
    point = np.array([10.0, 5.0]) + 4.0 * heading_vector(yaw) + 1.0 * np.array([-math.sin(yaw), math.cos(yaw)])
    return (10.0, 5.0, yaw), point


pose_90, point_90 = named_case(math.pi / 2)
practice("swap at yaw pi/2", swap(pose_90, point_90), (-4.0, -1.0), tol=1e-6)
practice("no rotation at yaw pi/2", no_rotation(pose_90, point_90), (-1.0, 4.0), tol=1e-6)

for label, yaw in (("0", 0.0), ("pi/2", math.pi / 2), ("pi", math.pi), ("-3pi/4", -3 * math.pi / 4), ("-pi/2", -math.pi / 2), ("2pi", 2 * math.pi)):
    pose, point = named_case(yaw)
    print(f"yaw {label:>7}: swap {np.round(swap(pose, point), 3)}, no rotation {np.round(no_rotation(pose, point), 3)}")


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
# **Problem 7 (Fix it).** The way to catch both mistakes is the one Exercise A already uses: yaws where sin θ is well away
# from 0. Write the test: fill `test_yaws` with the yaws, in radians, your test would try. `fails_at` counts the yaws where a
# transform misses the point 4 m ahead and 1 m left.
#
# **Guess first:** does `[0.0]` catch the swap? Does `[math.pi]` catch the dropped rotation?

# %%
def fails_at(fn, yaws):
    """The yaws, out of the list, where fn misses (4, 1) on the named case."""
    return [yaw for yaw in yaws if np.abs(fn(*named_case(yaw)) - (4.0, 1.0)).max() > 1e-6]


test_yaws = []  # your code: the yaws (radians) of your test, for example [0.0] to start with, then more

caught = sum(bool(fails_at(fn, test_yaws)) for fn in (swap, no_rotation))
practice("broken transforms your yaws catch (of 2)", caught, 2, tol=0)
practice("yaws where your world_to_ego fails", len(fails_at(world_to_ego, test_yaws)), 0, tol=0)
practice("yaws where the simulator's transform fails", len(fails_at(geom.world_to_ego, test_yaws)), 0, tol=0)

# %% [markdown]
# While `test_yaws` is empty the first line shows a red cross: that is the cell waiting for you. A test that catches both
# mistakes and passes the right transform is what Exercise A is built from. Your `world_to_ego` passed Exercise A and the
# picture test, so the camera now draws through your code. The lecture's last figure shows what the next lesson is about.
#
# ![](figures/next_lesson.png)
#
# The two challenges of the lecture end with a figure each: the swap seen through time, then how fast its error grows.
#
# ![](figures/challenge_swing.png)
#
# ![](figures/challenge_growth.png)
