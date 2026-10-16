# %% [markdown]
# # Lab: Poses you can chain
#
# You will write the four pieces of pose arithmetic: a pose as a 3×3 matrix, the pose back out of the matrix, the
# closed-form inverse, and the step between two poses. Then you chain the 1777 steps of the `gentle` lap back into the
# lap itself, break the chain on purpose, and use the same tools to find the car's rear axle.
# Read the lecture first; this lab points back to its sections by title and follows its order.
#
# Run the cells from top to bottom (Shift+Enter). Each problem has a "Guess first" line: write your guess down before you
# run the cell, then compare. `practice` prints a green tick or tells you what it expected.
#
# ![](figures/rebuilt_lap.png)

# %%
import sys
if "google.colab" in sys.modules:
    %pip install -q "git+https://github.com/vvknyn/self-driving-ai-course@main"
%matplotlib inline

# %% tags=["setup"]
import math

import numpy as np

from zero2fsd import geom
from zero2fsd.car import Car, run
from zero2fsd.geom import T_EGO_REAR, rotation_matrix, world_to_ego, wrap_angle
from zero2fsd.grade import check, practice
from zero2fsd.sim import make_scenario
from zero2fsd.sim.scenarios import GENTLE_RADIUS

# %% [markdown]
# The next cell drives the `gentle` lap once and keeps the car's pose at each of its 1778 steps: `poses[k]` is
# `(x, y, yaw)` in metres and radians. `Tw[k]` is the same pose as a 3×3 matrix, built with the provided
# `geom.pose_to_matrix` so that every problem below stands on its own. The lecture used `rotation_matrix` and
# `world_to_ego` from *The view from the driver's seat*; swap in your own from that lesson if you like.

# %%
tel = run(Car(), "gentle")
poses = np.column_stack([tel.x, tel.y, tel.yaw])
Tw = np.stack([geom.pose_to_matrix(p) for p in poses])
print(f"{len(poses)} poses; step 400 is x = {poses[400, 0]:.4f}, y = {poses[400, 1]:.4f}, yaw = {poses[400, 2]:.4f} rad")

# %% [markdown]
# ## 1. A pose is a matrix
#
# The lecture's section "A pose is a frame" puts a pose's axes and origin in the columns of a 3×3 matrix.
#
# ![](figures/pose_frame.png)
#
# ![](figures/pose_matrix.png)
#
# **Problem 1.** Write `pose_to_matrix`: from `(x, y, theta)` to the 3×3 matrix whose first two columns are the forward
# and left axes and whose last column is `(x, y, 1)`. Use `rotation_matrix(theta)` for the top-left 2×2 block.
#
# **Guess first:** the cone sits 10 m ahead and 2 m to the left of the car at step 400. Is its world position nearer
# the car's position, (154.7, 19.9), or further from the world origin than that?

# %%
def pose_to_matrix(pose):
    """The 3×3 matrix of a pose.

    Input:   pose, a tuple (x, y, theta): metres, metres, radians.
    Output:  a (3, 3) array [[cos, -sin, x], [sin, cos, y], [0, 0, 1]].
    Example: pose_to_matrix((1, 0, 0))  ->  [[1, 0, 1], [0, 1, 0], [0, 0, 1]]
    """
    raise NotImplementedError


try:
    T400 = pose_to_matrix(tuple(poses[400]))
    practice("the cone (10, 2, trailing 1) at step 400, in world metres", T400 @ [10, 2, 1], (160.987, 27.8514, 1), tol=0.001)
    practice("the arrow (1, 0, trailing 0) at step 400", T400 @ [1, 0, 0], (0.7623, 0.6473, 0), tol=0.0001)
except NotImplementedError:
    print("Write pose_to_matrix above first, then run this cell again.")

# %% [markdown]
# The cone is at (160.987, 27.8514): the car's position plus 10 forward axes plus 2 left axes. The arrow has no position
# part at all: a trailing 0 skips the third column and keeps only the turn (lecture: "A pose is a frame").
#
# ![](figures/arrow_vs_point.png)
#
# **Break it, fix it.** Send the cone as an arrow, with a trailing 0 instead of a trailing 1. **Guess first:** where does
# it land?

# %%
try:
    print("cone with a trailing 0:", np.round(T400 @ [10, 2, 0], 2))
except NameError:
    print("Finish Problem 1 first.")

# %% [markdown]
# It lands at (6.33, 8.00): the cone's offset from the car, turned, but with the car's position missing. A trailing 0
# turns the numbers and forgets where the car is. To place a point, the last number must be 1.
#
# **Problem 2.** Write `matrix_to_pose`, the way back: `(x, y, theta)` from the matrix. The position is the last column,
# and `theta = math.atan2(T[1, 0], T[0, 0])`.
#
# **Guess first:** the car's yaw climbs to 6.2264 rad over the lap. Will `theta` ever be more than π?

# %%
def matrix_to_pose(T):
    """The pose of a 3×3 matrix.

    Input:   T, a (3, 3) array.
    Output:  a tuple (x, y, theta), theta in [-pi, pi].
    Example: matrix_to_pose(pose_to_matrix((1, 2, 0.5)))  ->  (1.0, 2.0, 0.5)
    """
    raise NotImplementedError


try:
    back = np.array([matrix_to_pose(T) for T in Tw])
    practice("worst x or y difference after the round trip, over the lap (m)", np.abs(back[:, :2] - poses[:, :2]).max(), 0.0, tol=1e-12)
    practice("worst theta difference against the wrapped yaw, over the lap (rad)", np.abs(back[:, 2] - wrap_angle(poses[:, 2])).max(), 0.0, tol=1e-12)
except NotImplementedError:
    print("Write matrix_to_pose above first, then run this cell again.")

# %% [markdown]
# No: `atan2` returns an angle between −π and π, so the round trip gives back the yaw *wrapped* into that range
# (lecture: "A pose is a frame", the wrap trap).
#
# ![](figures/wrap_trap.png)
#
# **Break it, fix it.** Compare `theta` with the raw yaw instead of the wrapped one. **Guess first:** from which step
# on do they disagree, and by how much?

# %%
try:
    gap = back[:, 2] - poses[:, 2]
    late = np.nonzero(np.abs(gap) > 1)[0]
    print(f"the raw comparison fails from step {late[0]} on, by {abs(gap[late[0]]):.4f} rad (a full turn)")
except NameError:
    print("Finish Problem 2 first.")

# %% [markdown]
# From step 1021 on, the two differ by a full turn, 6.2832 rad, though both describe the same direction. Whenever you
# compare angles, wrap the difference. The graders in this lab do.
#
# ## 2. Chaining and undoing
#
# "Subscripts that cancel" and "Order matters" are paper-and-pencil sections: write down the product, cancel the inner
# names, and check that a swapped product ends somewhere else.
#
# ![](figures/rear_frame.png)
#
# ![](figures/frame_chain.png)
#
# ![](figures/order_matters.png)
#
# **Problem 3.** Write `invert_pose`, the closed-form inverse from "Undoing a pose": `[[Rᵀ, −Rᵀt], [0, 1]]`. It must
# work at any size, so take the rotation block as `T[:k, :k]` and the position as `T[:k, k]` with `k = T.shape[0] - 1`.
# Do not call `np.linalg.inv`: the formula is cheaper and exact, and it is the one the next lesson reuses on 4×4
# matrices.
#
# **Guess first:** seen from A = (1, 0, π/2), the frame 1 m east of the start and facing north, where is the start?

# %%
def invert_pose(T):
    """The inverse of a rigid transform, in closed form.

    Input:   T, a (k+1, k+1) array: a k×k rotation block, a position column and the bottom row (0, ..., 0, 1).
    Output:  a (k+1, k+1) array, the matrix that undoes T.
    Example: invert_pose(pose_to_matrix((1, 0, math.pi / 2)))  ->  [[0, 1, 0], [-1, 0, 1], [0, 0, 1]]
    """
    raise NotImplementedError


try:
    A = geom.pose_to_matrix((1, 0, math.pi / 2))
    practice("A seen from the start, inverted", invert_pose(A), [[0, 1, 0], [-1, 0, 1], [0, 0, 1]], tol=1e-12)
    practice("the world origin seen from the car at step 400", (invert_pose(Tw[400]) @ [0, 0, 1])[:2], world_to_ego(tuple(poses[400]), (0, 0)), tol=1e-9)
    worst = max(np.abs(invert_pose(T) @ T - np.eye(3)).max() for T in Tw)
    practice("worst entry of invert_pose(T) @ T minus the identity, over the lap", worst, 0.0, tol=1e-12)
    T4 = np.eye(4)
    T4[:3, :3] = geom.rot_z(0.7) @ geom.rot_y(0.1)
    T4[:3, 3] = (1.2, 0.3, 1.5)
    practice("a 4×4 pose: worst entry of invert_pose(T4) @ T4 minus the identity", np.abs(invert_pose(T4) @ T4 - np.eye(4)).max(), 0.0, tol=1e-12)
except NotImplementedError:
    print("Write invert_pose above first, then run this cell again.")

# %% [markdown]
# Seen from A, the start is 0 m ahead and 1 m to the left, and it points a quarter turn to A's right. From the car at
# step 400 the world origin is at (−130.7431, 84.9692): 130.74 m behind and 84.97 m to the left. The same function
# inverts the 4×4 pose, which is why it is written for any size.
#
# ![](figures/undo_steps.png)
#
# ![](figures/inverse_A.png)
#
# ![](figures/inverse_view.png)
#
# **Break it, fix it.** A tempting slip: use `R` instead of `Rᵀ` in both places. **Guess first:** will the round trip be
# slightly off or badly off?

# %%
def invert_pose_slip(T):
    inverse = np.eye(3)
    inverse[:2, :2] = T[:2, :2]
    inverse[:2, 2] = -T[:2, :2] @ T[:2, 2]
    return inverse


slip = [np.abs(invert_pose_slip(T) @ T - np.eye(3)).max() for T in Tw]
print(f"slip: worst entry of the round trip minus the identity is {slip[400]:.3f} at step 400 and {max(slip):.3f} over the lap")

# %% [markdown]
# It is off by almost 1 at step 400 and by 2.000 over the lap, because turning a position by the car's own angle
# turns it further away instead of back. Fix: transpose the rotation block, in both places.
#
# ## 3. Odometry: the step between two poses
#
# **Problem 4 (Exercise A).** Write `relative_pose(T_world_a, T_world_b)`, the pose of `b` seen from `a`. In the
# lecture's words, `T_a←b = inverse(T_world←a) · T_world←b`. Use your `invert_pose`.
#
# **Guess first:** is `relative_pose(A, A @ B)` equal to `B`? And is `relative_pose(B, B @ A)` equal to `A`?

# %% tags=["exercise:1.1.4.a"]
def relative_pose(T_world_a, T_world_b):
    """Frame b seen from frame a: the odometry step from pose a to pose b.

    Input:   T_world_a, T_world_b: two (3, 3) pose matrices, both written in world numbers.
    Output:  a (3, 3) array, T_a←b: the forward part, the left part and the turn of the move from a to b.
    Example: A = pose_to_matrix((1, 0, math.pi / 2)); B = pose_to_matrix((1, 0, 0))
             relative_pose(A, A @ B)  ->  the matrix of B, [[1, 0, 1], [0, 1, 0], [0, 0, 1]]
    """
    raise NotImplementedError


# %%
check("1.1.4.a", relative_pose)

# %% [markdown]
# A correct function passes all 44 cases (the four worked ones from the lecture and 40 seeded pairs of poses), every
# entry far inside the limit of $10^{-9}$. Now it can turn the car's own record into the steps it felt:

# %%
try:
    steps = np.stack([relative_pose(Tw[k], Tw[k + 1]) for k in range(len(Tw) - 1)])
    practice("mean step forward (m)", steps[:, 0, 2].mean(), 0.3998, tol=0.0001)
    practice("mean step turn (rad)", np.arctan2(steps[:, 1, 0], steps[:, 0, 0]).mean(), 0.00350, tol=0.00001)
except NotImplementedError:
    print("Write relative_pose above first, then run this cell again.")

# %% [markdown]
# Each step is about 0.4 m forward and 0.0035 rad of turn on average; the turn is near 0 on the straights.
#
# ![](figures/one_step.png)
#
# ![](figures/step_turns.png)
#
# **Break it, fix it.** Swap the arguments: `invert_pose(T_world_b) @ T_world_a`. **Guess first:** what does it say for
# `A` and `A @ B`, and does the check notice?

# %%
def relative_pose_swapped(T_world_a, T_world_b):
    return invert_pose(T_world_b) @ T_world_a


try:
    print("swapped, A then B:", np.round(geom.matrix_to_pose(relative_pose_swapped(A, A @ geom.pose_to_matrix((1, 0, 0)))), 4) + 0.0, "(B is (1, 0, 0))")
    check("1.1.4.a", relative_pose_swapped)
except NameError:
    print("Finish Problems 3 and 4 first.")
except NotImplementedError:
    print("Finish Problem 3 first.")

# %% [markdown]
# The swapped step comes out as (−1, 0, 0): the step run backwards. The check names the first case it gets wrong. Fix:
# invert the *first* pose, `invert_pose(T_world_a) @ T_world_b`.
#
# ## 4. Dead reckoning
#
# **Problem 5 (Exercise B).** Write `dead_reckon(T_world_start, steps)`. It takes the start pose and an array of steps,
# shape `(n, 3, 3)`, and returns the `n + 1` poses: the start, then each pose times the next step, with the step on the
# right (lecture: "Dead reckoning"). The check rebuilds the whole `gentle` lap from its 1777 steps and compares *every*
# pose with the simulator's record, not only the last.
#
# **Guess first:** with the step on the left instead, how far off will the end of the lap be?

# %% tags=["exercise:1.1.4.b"]
def dead_reckon(T_world_start, steps):
    """The poses a car passes through, rebuilt from the steps it felt.

    Input:   T_world_start: the (3, 3) start pose. steps: an (n, 3, 3) array, steps[k] the step from pose k to pose k + 1.
    Output:  an (n + 1, 3, 3) array: the start, then T_world_{k+1} = T_world_k @ steps[k] for each k.
    Example: dead_reckon(np.eye(3), steps)  with two steps of 1 m forward  ->  poses at x = 0, 1 and 2
    """
    raise NotImplementedError


# %%
check("1.1.4.b", dead_reckon)

# %% [markdown]
# A correct function rebuilds all 1778 poses far inside the limits, $10^{-6}$ m and $10^{-9}$ rad. Add the parts of each step to
# x, y and theta instead and the lap becomes a straight line 710.66 m from the truth; the lecture draws it.
#
# ![](figures/naive_add.png)
#
# **Break it, fix it.** Multiply on the other side: `step @ pose`. **Guess first:** will the end of the lap look right?
# Will the check pass?

# %%
def dead_reckon_wrong_order(T_world_start, steps):
    poses_out = [np.asarray(T_world_start, dtype=float)]
    for step in steps:
        poses_out.append(step @ poses_out[-1])
    return np.stack(poses_out)


try:
    wrong = dead_reckon_wrong_order(Tw[0], steps)
    error = np.hypot(wrong[:, 0, 2] - poses[:, 0], wrong[:, 1, 2] - poses[:, 1])
    print(f"wrong order: {error[-1]:.2f} m off at the end, but {error.max():.2f} m off at step {error.argmax()}")
    check("1.1.4.b", dead_reckon_wrong_order)
except NameError:
    print("Finish Problems 3, 4 and 5 first.")
except NotImplementedError:
    print("Finish Problems 3 and 4 first.")

# %% [markdown]
# The end is 1.11 m off, which a check of the end alone with a 2 m limit would accept, yet at step 866 the rebuilt car is
# 204.88 m from the truth. That is why the check looks at every pose. Fix: the step goes on the right,
# `pose @ step`.
#
# ![](figures/wrong_order.png)
#
# ![](figures/wrong_order_error.png)
#
# ## 5. Drift
#
# The lecture adds a tiny error to every step's turn and rebuilds the lap again. This part needs no code from you; the
# figures hold the measurements, and the challenge problems at the end of the lecture ask you to reproduce them.
#
# ![](figures/drift_end.png)
#
# ![](figures/drift_growth.png)
#
# ![](figures/drift_laps.png)
#
# ## 6. The rear axle is a frame too
#
# **Problem 6.** The rear frame is fixed to the car's body, so one product places the rear axle at every step:
# `Tw @ T_EGO_REAR`. Compute it, and compare step 400 with the by-hand answer, the front position minus 2.7 m along the
# heading.
#
# **Guess first:** on the first bend, where does the rear axle ride: on the front axle's line, nearer the lane centre,
# or further inside?

# %%
rear = Tw @ T_EGO_REAR
by_hand = np.column_stack([poses[:, 0] - 2.7 * np.cos(poses[:, 2]), poses[:, 1] - 2.7 * np.sin(poses[:, 2])])
practice("the rear axle at step 400 (x, y)", rear[400, :2, 2], (152.6006, 18.1067), tol=0.0001)
practice("worst difference from the by-hand answer over the lap (m)", np.abs(rear[:, :2, 2] - by_hand).max(), 0.0, tol=1e-9)

road = make_scenario("gentle").road
s = np.array([road.ego_truth(*p)[0] for p in poses])  # distance along the road
e_front = np.asarray(tel.lat)  # lateral error e at the front axle, from the simulator's record
e_rear = np.array([road.ego_truth(x, y, yaw)[1] for (x, y), yaw in zip(rear[:, :2, 2], poses[:, 2])])
bend = (s > 160) & (s < 100 + math.pi * GENTLE_RADIUS - 60)  # the middle of the first bend
practice("mean lateral error e in the first bend, front axle (m)", e_front[bend].mean(), 0.5298, tol=0.0001)
practice("mean lateral error e in the first bend, rear axle (m)", e_rear[bend].mean(), 0.5746, tol=0.0001)

# %% [markdown]
# The rear axle rides further inside the bend: 0.5746 m against 0.5298 m. Its path is shorter than the front's, which
# the triangle in the lecture explains.
#
# ![](figures/rear_axle_paths.png)
#
# ![](figures/rear_axle_error.png)
#
# ## Challenges
#
# The lecture ends with four paper-and-pencil challenge problems on how drift grows with the number of steps, and why
# the wrong order ends so near the truth. Their answers are measured with `dead_reckon` and the steps above.
#
# ![](figures/levers.png)
#
# ![](figures/drift_loglog.png)
#
# ## What's next
#
# You can now chain poses, undo them, and turn a record of motion back into a path, and you know what a small error in
# every step does to it. The next lesson, *The camera's point of view*, puts the camera in this arithmetic with a 4×4 pose and
# `invert_pose` at any size, so that a pixel can become a point on the road.
#
# ![](figures/next_lesson.png)
