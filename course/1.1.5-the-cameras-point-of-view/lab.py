# %% [markdown]
# # Lab: The camera's point of view
#
# You will write the three turns about x, y and z, build the camera's mount as one 4×4 matrix, and send a pixel's ray
# down to the road. Then you swap your mount into the simulator's camera and drive with it.
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

import matplotlib.pyplot as plt
import numpy as np

from zero2fsd.car import Car, run
from zero2fsd.dashboard import dashboard
from zero2fsd.grade import check, practice
from zero2fsd.sim import BicycleState, Camera, make_scenario
from zero2fsd.sim.scenarios import TARGET_SPEED

# %% [markdown]
# The next cell brings in the helpers from the earlier lessons that this lab builds on. To use your own from
# *Poses you can chain*, replace the import with your function (it must keep the name).

# %%
from zero2fsd.geom import invert_pose, pixel_ray, pose_to_matrix, world_to_ego

camera = Camera()
print(f"f = {camera.f:.0f} px, centre ({camera.cx}, {camera.cy}), height {camera.mount_height} m, pitch 5 degrees = {math.radians(5):.4f} rad")

# %% [markdown]
# ## 1. A third axis, and the turn about each
#
# The ego frame gets a third axis, z, up (lecture: "A third axis"). A positive turn about an axis follows the right-hand
# rule: thumb along the axis, fingers curl the way the turn goes (lecture: "Turning about each axis").
#
# ![](figures/ego_frame_3d.png)
#
# ![](figures/right_hand_turn.png)
#
# **Problem 1.** Write the three turns. `rot_z` is the 2D rotation you know, with a 1 in the corner; `rot_x` and `rot_y`
# put the same cosines and sines in the other two planes. Mind where the minus sign goes in `rot_y`: its pattern is
# not the same as the other two.
#
# **Guess first:** `rot_y(0.0873) @ [1, 0, 0]` (a 5° turn about the car's left-pointing y axis, applied to forward):
# does the nose point a little up or a little down?

# %%
def rot_x(a):
    return np.eye(3)  # your code: [[1, 0, 0], [0, cos, -sin], [0, sin, cos]]


def rot_y(a):
    return np.eye(3)  # your code


def rot_z(a):
    return np.eye(3)  # your code


x, y, z = np.eye(3)
pitch = math.radians(5.0)
print("forward x left =", np.cross(x, y), "(up)")
practice("rot_z(pi/2) turns forward to left", rot_z(math.pi / 2) @ x, (0, 1, 0), tol=1e-12)
practice("rot_x(pi/2) turns left to up", rot_x(math.pi / 2) @ y, (0, 0, 1), tol=1e-12)
practice("rot_y(5 deg) turns forward to", rot_y(pitch) @ x, (0.9962, 0.0, -0.0872), tol=1e-4)

# %% [markdown]
# A positive turn about y tips the nose **down**: forward becomes (0.9962, 0, −0.0872). If you put the minus sign on the
# other `sin` you get (0.9962, 0, +0.0872): a camera tipped up. The first column of `rot_y` is where forward goes; read
# the columns of any rotation that way.
#
# ![](figures/three_rotations.png)
#
# ![](figures/pitch_columns.png)
#
# ## 2. The camera's own axes
#
# The camera's frame is not the ego frame: x right across the picture, y down it, z out through the lens (lecture:
# "The camera's own axes"). `B` takes ego numbers to camera numbers; `Bᵀ` takes them back.
#
# ![](figures/optical_vs_ego.png)
#
# **Guess first:** a level camera looks along ego forward. What does `Bᵀ` do to the camera's z axis (0, 0, 1)?

# %%
B = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])  # ego (forward, left, up) -> camera (right, down, forward)
print("B @ forward, left, up =", B @ x, B @ y, B @ z)
print("B.T @ camera z axis   =", B.T @ [0, 0, 1], "(ego forward: a level camera looks straight ahead)")

# %% [markdown]
# ## 3. One 4×4 for the mount
#
# The camera sits 1.4 m up and tipped down by φ = 5°. Its turn is `rot_y(φ) @ Bᵀ`, in that order: first `Bᵀ` lines
# the camera's axes up with the car's, then the pitch tips the whole camera (lecture: "One 4×4 for the mount"). The other order
# gives a camera looking level and 5° to the right.
#
# ![](figures/wrong_order.png)
#
# ![](figures/mount_side.png)
#
# **Problem 2.** Build the 4×4 `T_ego_cam(height_m, pitch_rad)` in the next section, then project 200 seeded road
# points through `invert_pose(T_ego_cam(...))` and the pinhole step, and compare with the simulator's own camera.
#
# **Guess first:** how far, in pixels, will the largest difference be for a correct mount: 1 px, 0.01 px, or exactly 0?
#
# ## 4. Graded exercises
#
# ### Exercise A and Exercise B: `T_ego_cam`
#
# `T_ego_cam(height_m, pitch_rad)` returns the camera's pose in the ego frame as a 4×4 array: the turn
# `rot_y(pitch_rad) @ Bᵀ` in the top-left 3×3 block, the lens position (0, 0, `height_m`) in the last column, and
# `0 0 0 1` below. A positive pitch tips the nose down.
#
# Exercise A checks that the lens axis points where it should and that 200 seeded road points land within $10^{-6}$ px of
# where the simulator draws them. Exercise B puts your mount into the simulator's camera and drives `gentle` (lecture:
# "A camera mounted wrong"): the car must complete it, and none of 18 sampled camera pictures may differ.
#
# ![](figures/projected_points.png)
#
# ![](figures/blind_zone.png)

# %% tags=["exercise:1.1.5.a", "exercise:1.1.5.b"]
from zero2fsd.geom import CAM_FROM_EGO_AXES as B, rot_y  # swap in your own rot_y from Problem 1 and the B above


def T_ego_cam(height_m, pitch_rad):
    """The camera's pose in the ego frame, as a 4 by 4 array.

    Input:   height_m, how high the lens is above the road (m); pitch_rad, how far the camera tips DOWN (rad).
    Output:  T, shape (4, 4): T[:3, :3] = rot_y(pitch_rad) @ B.T, T[:3, 3] = (0, 0, height_m), last row 0 0 0 1.
    Example: T_ego_cam(1.4, math.radians(5))
             -> [[0, -0.0872, 0.9962, 0], [-1, 0, 0, 0], [0, -0.9962, -0.0872, 1.4], [0, 0, 0, 1]]
    """
    raise NotImplementedError


# %% [markdown]
# **Guess first (Exercise A):** will your pixels match the simulator's to $10^{-6}$ px, or only to about 0.1 px?

# %%
check("1.1.5.a", T_ego_cam)

# %% [markdown]
# A correct mount matches to 0.0 px: it is the same arithmetic the simulator uses. A camera tipped **up** by mistake
# (`rot_y(-pitch_rad)`) puts the same points about 40 px away, and the check tells you the lens axis points up.
#
# The same projection by hand, for 200 seeded road points:

# %%
rng = np.random.default_rng(0)
forward, left = rng.uniform(2.0, 60.0, 200), rng.uniform(-8.0, 8.0, 200)
ground = np.stack([forward, left, np.zeros(200)], axis=1)
try:
    p_cam = (invert_pose(T_ego_cam(1.4, math.radians(5))) @ np.c_[ground, np.ones(200)].T).T[:, :3]
    u = camera.cx + camera.f * p_cam[:, 0] / p_cam[:, 2]
    v = camera.cy + camera.f * p_cam[:, 1] / p_cam[:, 2]
    spec = np.array(camera.ground_to_pixel(forward, left))
    print(f"largest pixel difference over 200 road points: {np.max(np.abs(np.stack([u, v]) - spec)):.2e} px")
except NotImplementedError:
    print("Write T_ego_cam above first, then run this cell again.")

# %% [markdown]
# ## 5. Points move, directions only turn
#
# A point takes w = 1: the 4×4 turns it and shifts it. A direction takes w = 0: it only turns (lecture: "Points move,
# directions only turn").
#
# ![](figures/point_vs_direction.png)
#
# **Problem 3.** Write `transform_points(T, p)` (w = 1) and `transform_directions(T, v)` (w = 0) for a 3-vector, using
# the 3×3 block `T[:3, :3]` and the last column `T[:3, 3]`.
#
# **Guess first:** with the mount `T`, where does the lens axis (0, 0, 1) go as a direction, and where as a point?

# %%
def transform_points(T, p):
    return np.asarray(p, dtype=float)  # your code: turn AND shift


def transform_directions(T, v):
    return np.asarray(v, dtype=float)  # your code: turn only


T = np.eye(4)
T[:3, :3] = rot_y(pitch) @ B.T
T[:3, 3] = (0.0, 0.0, 1.4)  # this lab's own T, so Problem 3 runs before your T_ego_cam is done
practice("the lens (camera origin) in ego numbers", transform_points(T, (0, 0, 0)), (0.0, 0.0, 1.4), tol=1e-12)
practice("the lens axis as a direction, in ego numbers", transform_directions(T, (0, 0, 1)), (0.9962, 0.0, -0.0872), tol=1e-4)
practice("the lens axis wrongly as a point (w = 1)", transform_points(T, (0, 0, 1)), (0.9962, 0.0, 1.3128), tol=1e-4)

# %% [markdown]
# As a direction the axis dips 5° below level; as a point it sits 1.3128 m up, which is nonsense for something that is
# only a heading. Forward as a direction lands on the horizon, row 75.5; treated as a point it lands in row 276.6, off
# the bottom of the picture.
#
# ![](figures/horizon_w.png)
#
# ## 6. Where a pixel's ray meets the road
#
# The ray through pixel (u, v) is ((u − cx) / f, (v − cy) / f, 1) in camera numbers. Turn it into ego numbers
# with `w = 0`, then walk λ steps from the lens until z = 0: `λ = −t_z / r_z` (lecture: "Where a pixel's ray meets the
# road"). If `r_z ≥ 0` the ray never meets the road: the answer is `nan`.
#
# ![](figures/pixel_ray.png)
#
# **Guess first:** the white line's pixel (254.5, 150): about how many steps λ down the ray to the road, and how far
# ahead and to the right is the road point?

# %%
ray = pixel_ray(camera, 254.5, 150)
print("ray in camera numbers:", np.round(ray, 4))
r = transform_directions(T, ray)
print("ray in ego numbers:   ", np.round(r, 4))
print("steps to the road:    ", round(float(-T[2, 3] / r[2]), 3))

# %% [markdown]
# **Problem 4 and Exercise C.** Write `ray_to_ground(T_ego_cam, ray_cam)` below; it returns the road point as a 3-vector
# (forward, left, up) and `nan` when the ray never meets the road. The check sends 200 seeded pixels below the horizon
# to the road and compares with `Camera.pixel_to_ground` within $10^{-9}$ m, then 200 pixels above the horizon, which must
# come back `nan`. The worked pixel gives (2.9073, −1.7921).
#
# ![](figures/rows_across.png)

# %% tags=["exercise:1.1.5.c"]
from zero2fsd.geom import transform_directions  # swap in your own from Problem 3


def ray_to_ground(T_ego_cam, ray_cam):
    """Where a ray from the camera's lens meets the road.

    Input:   T_ego_cam, the 4 by 4 mount; ray_cam, the ray in camera numbers, shape (3,).
    Output:  the road point in ego numbers, shape (3,): (forward, left, 0); math.nan for all three
             when the ray never meets the road (its ego z is not negative).
    Example: ray_to_ground(T_ego_cam(1.4, math.radians(5)), pixel_ray(Camera(), 254.5, 150))
             -> [2.9073, -1.7921, 0.0]
    """
    raise NotImplementedError


# %% [markdown]
# **Guess first:** what does a correct `ray_to_ground` return for a pixel above row 75.5?

# %%
check("1.1.5.c", ray_to_ground)

# %% [markdown]
# Every pixel below the horizon lands within $2\times10^{-16}$ m of the simulator's answer. Treat the ray as a point (`w = 1`) and
# all 200 come back `nan`, because the shifted ray no longer points down.
#
# ![](figures/rows_pitch.png)
#
# ## 7. A camera mounted wrong
#
# Now the payoff: put your mount into the simulator's camera and drive. Tip the camera 5° **up** instead of down and
# the picture still looks like a road; whether the car notices depends on the road (lecture: "A camera mounted
# wrong").
#
# **Guess first:** tip the camera up, and which road shows it: `straight`, `gentle`, both, or neither?

# %%
try:
    for mounted, label in ((T_ego_cam(1.4, pitch), "tipped down 5 deg (correct)"), (T_ego_cam(1.4, -pitch), "tipped UP 5 deg")):
        tel = run(Car(), "gentle", camera=Camera(T_ego_cam=mounted))
        print(f"{label}: completed {tel.completed}, ended after {tel.progress_m[-1]:.1f} m, mean |lateral error| {np.mean(np.abs(tel.lat)):.3f} m")
except NotImplementedError:
    print("Write T_ego_cam above first, then run this cell again.")

# %% [markdown]
# The correct mount completes `gentle` with a mean error of 0.376 m. Tipped up, the car leaves the lane at 86.3 m, before
# the first bend: the far band places road 20 to 70 m ahead only 5.6 to 7.1 m ahead, so the bend reads as a sharp turn.
# On `straight` the same mistake drives to the end (mean 0.004 m).
#
# ![](figures/lat_pitch.png)
#
# ![](figures/heading_pitch.png)
#
# Now the swap for grading: your `T_ego_cam` goes into the camera and the provided car drives `gentle`. Expect a mean
# distance from the lane centre near 0.376 m, and no camera pictures that differ.

# %%
check("1.1.5.b", T_ego_cam)

# %% [markdown]
# ![](figures/gate_breaks.png)
#
# ## 8. Break it, fix it
#
# A teammate says "a camera 0.2 m too low is the same error, just smaller". Mount the camera at 1.2 m and at 1.6 m,
# but keep the box's own 1.4 m, and drive `gentle`.
#
# **Guess first:** does the car leave the road, wobble, or complete with a bigger error? Which of heading and distance
# does a wrong height damage?

# %%
try:
    for height in (1.2, 1.6):
        tel = run(Car(), "gentle", camera=Camera(T_ego_cam=T_ego_cam(height, pitch)))
        print(f"camera at {height} m: completed {tel.completed}, mean |lateral error| {np.mean(np.abs(tel.lat)):.3f} m")
except NotImplementedError:
    print("Write T_ego_cam above first, then run this cell again.")

# %% [markdown]
# <details><summary>Hint (open after you have a guess)</summary>
#
# Both complete: 1.2 m gives 0.286 m and 1.6 m gives 0.472 m. A height error only scales every distance by 1.4 / h, so
# the box misjudges how far ahead things are and how wide the lane is, but angles survive. A pitch error shifts every
# ray's angle, and that is why it is the one that breaks the box.
#
# </details>
#
# ![](figures/height_scale.png)
#
# ## 9. Challenge problems
#
# Not graded; the answers are in the lecture (lecture: "Challenge problems"). Four marks painted on the road, at (3, ±1)
# and (8, ±1) m, let you recover the camera's pitch and height from one picture, with neither known.
#
# **Guess first:** the near marks sit in row 147.789 and the far marks in row 103.291. If a mark found one pixel too low in every row
# moves your pitch estimate, by about how much: 0.0001 rad, 0.007 rad, or 0.05 rad?
#
# ![](figures/calibration_marks.png)
#
# ![](figures/calibration_spread.png)
#
# Then try the others: why no pixel ray points down when you give it `w = 1`, and why a camera 1.2 m up read as 1.4 m
# scales every road point by the same factor.
#
# ## 10. What comes next
#
# Every lap so far ran at 8 m/s. Who chose it, and how would the car change speed? That is the next module,
# *Longitudinal control and vehicle motion*.
#
# ![](figures/next_lesson.png)
