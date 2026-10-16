# %% [markdown]
# # Lab: The road's own coordinates
#
# You will measure why `y + 1.8` is not a lateral error, project a point onto a segment by hand and in code, write
# `to_road_frame` (how far along the road, how far from its yellow line) and `lane_errors` (the two numbers the
# controller needs), and break each one on purpose. Read the lecture first; this lab points back to its sections by
# title and follows its order.
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
import numpy as np

from zero2fsd.car import Car, run
from zero2fsd.grade import check, practice
from zero2fsd.grade.units.u1_1_3 import psi_misfit  # how far a heading ψ is from the truth, counting either tied segment as right
from zero2fsd.sim import make_scenario

# %% [markdown]
# `wrap_angle` and `world_to_ego` are the provided ones. To build on your own from the lessons *Positions and headings*
# (`wrap_angle`) and *The view from the driver's seat* (`world_to_ego`), replace the import with your functions (they must
# keep their names).

# %%
from zero2fsd.geom import world_to_ego, wrap_angle

# %% [markdown]
# The next cell drives the provided car once round `gentle` (the lap of the lecture) and keeps its telemetry: `tel.x`,
# `tel.y`, `tel.yaw`, and the simulator's own answers `tel.lat` (the lateral error) and `tel.head_err` (the heading ψ).
# The road is a closed polyline; `centerline` is its 7028 vertices, one row each, and `L_ROAD` is the small
# open road of the lecture's corner examples: (0, 0), then (10, 0), then (10, 10).

# %%
road = make_scenario("gentle").road
tel = run(Car(), "gentle", seed=0)
centerline = road.centerline  # a copy of the vertices of the yellow line, shape (7028, 2); the road is closed: the last one joins the first
L_ROAD = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
print(f"{len(tel.t)} steps over {tel.t[-1]:.2f} s; {len(centerline)} vertices; the road is {road.length:.2f} m long")

# %% [markdown]
# ## 1. The lateral error you would guess
#
# The simplest guess for how far the car is from its lane centre is `y + 1.8`: the yellow line is the line `y = 0`
# on the first straight, and the lane centre is 1.8 m to its right (lecture: "The lateral error you would guess").
#
# **Guess first:** will `y + 1.8` match the simulator's `tel.lat` for the whole lap? If it breaks, where, and by how
# much?

# %%
naive = tel.y + 1.8
worst = None  # your code: the largest |naive - tel.lat| over the lap (m)
first_out = None  # your code: the first time (s) at which |naive - tel.lat| is more than 0.01 m
practice("largest |y + 1.8 − lat| (m)", worst, 163.6, tol=0.1)
practice("first time it is 0.01 m out (s)", first_out, 12.70, tol=0.051)

# %% [markdown]
# On the first straight the guess is exact. It leaves the true value at 12.70 s, in the first bend, and on the far
# straight it is 163.6 m wrong: the road there runs west at `y` of about 160, and `y + 1.8` measures from a line the
# car left long ago. The cleverer guess, `y` minus the `y` of the nearest lane-centre point, gets the sign wrong where the road runs west: −0.528 m
# against +0.531 m at 43.30 s.
#
# ![](figures/naive_y.png)
#
# ![](figures/naive_nearest_y.png)
#
# What the guess lacks is the road's own direction. The road is a polyline, the vertices of `centerline` 0.1 m apart
# joined by straight segments; the rest of the lab measures against those segments.
#
# ![](figures/polyline_zoom.png)
#
# ## 2. Projecting onto a segment
#
# For a segment from `a` to `b` and a point `p`: the forward part `t` is how far along the segment the point's
# shadow falls, clamped to the segment; the foot is `a + t u` with `u` the unit direction; `d` is the distance to the foot, with the
# sign of the left part (lecture: "Projecting onto a segment").
#
# **Guess first:** a = (2, 1), b = (10, 7), p = (5, 7). By hand, what are `t` and `d`? Is `p` on the left of the
# segment (`d` positive) or on its right?

# %%
t_hand = None  # your answer: the forward part (m)
d_hand = None  # your answer: the signed distance to the foot (m)
practice("t (m)", t_hand, 6.0, tol=1e-9)
practice("d (m)", d_hand, 3.0, tol=1e-9)

# %% [markdown]
# `t` is 6.0 and `d` is +3.0: `p` is on the left of the segment, where the left part is positive.
#
# Now the function, for any segment. It must work on many segments at once: `a` and `b` may be arrays of shape
# `(N, 2)` (one row per segment) with `p` a single point, and then `t` and `d` are arrays of length `N`. Plain numpy
# arithmetic on whole arrays does that without a loop. For the sign use `np.copysign(distance, left_part)`: it keeps
# the sign of the left part even when that part is exactly 0.

# %% tags=["exercise:1.1.3.a"]
def project_to_segment(a, b, p):
    """Where a point lands on one segment: how far along it, and how far to its side.
    Input:   a, b, the segment's two ends, shape (2,) or (N, 2) for N segments; p, a point, shape (2,).
    Output:  (t, d): t is the forward part clamped to [0, |b - a|] in metres; d is the distance from p to the foot
             a + t * u, with the sign of the left part: positive when p is to the left of a -> b.
    Example: project_to_segment((2, 1), (10, 7), (5, 7))  ->  (6.0, 3.0)
    """
    raise NotImplementedError

# %% [markdown]
# For `q = (14, 5)` and the same segment the forward part is 12, longer than the segment: without the clamp the foot
# lands past `b`, 4.0 m from `q`.
#
# **Guess first:** what does your function say for `q`? Is the distance 4.0, or is it more?

# %%
try:
    t, d = project_to_segment((2.0, 1.0), (10.0, 7.0), (5.0, 7.0))
    practice("t of the hand example (m)", t, 6.0, tol=1e-9)
    practice("d of the hand example (m)", d, 3.0, tol=1e-9)
    t, d = project_to_segment((2.0, 1.0), (10.0, 7.0), (14.0, 5.0))
    practice("clamped d of q (m)", d, -4.4721, tol=1e-4)
except NotImplementedError:
    print("Write project_to_segment above first, then run this cell again.")

# %% [markdown]
# The clamp cuts `t` to 10, so the foot is `b` itself and `q` is 4.472 m from it, on the right of the segment (`d` is
# negative). Without the clamp it would be 4.0 m: shorter than the true distance, and a point past the end would be
# treated as if the road went on.
#
# ![](figures/segment_hand.png)
#
# ![](figures/clamp.png)
#
# ## 3. Which segment, and which side (Exercise A)
#
# Every segment gives a candidate foot; the segment with the nearest foot wins (lecture: "Which segment, and which
# side"). The corner road is two segments: from (0, 0) east to (10, 0), then north to (10, 10).
#
# **Guess first:** for the point (8, 3), which segment wins, and what are `s` and `d`? (`s` is the arc length of the
# winning foot, measured from the first vertex.)

# %%
s_hand = None  # your answer: the arc length of the nearest foot (m)
d_hand = None  # your answer: the signed distance (m)
practice("s (m)", s_hand, 13.0, tol=1e-9)
practice("d (m)", d_hand, 2.0, tol=1e-9)

# %% [markdown]
# The second segment wins: its foot, (10, 3), is 2.0 m from (8, 3) against 3.0 m for the first segment's foot, (8, 0). So
# `s` = 10 + 3 = 13 m and `d` = +2.0 m.
#
# Now `to_road_frame`, which is Exercise A. `centerline` is an `(N, 2)` array of vertices. The segments start at
# `a = centerline` and end at `b = np.roll(centerline, -1, axis=0)` when the road is `closed` (the last vertex joins
# the first), or at `a = centerline[:-1]`, `b = centerline[1:]` when it is open. Call your `project_to_segment` once with
# all segments, pick the winner, and add the forward part to the length of the segments before it.
#
# Loop over segments in Python and a lap of `gentle` takes minutes; the whole-array version takes under a
# millisecond per position (0.2 to 1 ms on one machine), so broadcast.

# %% tags=["exercise:1.1.3.a"]
def to_road_frame(centerline, closed, p):
    """The road's own coordinates of a point: how far along the road, and how far from its yellow line.
    Input:   centerline, the vertices of the road, shape (N, 2); closed, True when the last vertex joins the first;
             p, a point in the world, shape (2,).
    Output:  (s, d): s is the arc length along the road of the nearest point on it, in metres; for a closed road it
             stays in [0, length). d is the signed distance to that point, positive to the left of the road's direction.
    Example: to_road_frame(L_ROAD, False, (8, 3))  ->  (13.0, 2.0)
    """
    raise NotImplementedError

# %% [markdown]
# **Guess first (Exercise A):** the grader's named cases on the corner road are past the open end, (12, 13); on the line
# past the open end, (10, 13); inside the corner, (8, 3); to the right of the line, (5, −1); and exactly on it, (5, 0). Which
# will be easiest to get wrong?

# %%
try:
    s, d = to_road_frame(L_ROAD, False, (8.0, 3.0))
    practice("s of (8, 3) (m)", s, 13.0, tol=1e-9)
    practice("d of (8, 3) (m)", d, 2.0, tol=1e-9)
except NotImplementedError:
    print("Write to_road_frame above first, then run this cell again.")

# %%
check("1.1.3.a", to_road_frame)

# %% [markdown]
# A correct function passes the five named cases on the corner road, with `d` of (12, 13), past the end, equal to
# -3.606 and `s` equal to 20, then points either side of `gentle`'s start line and on a coarse copy of it, and 40 seeded
# points each on `gentle` and `curvy`, against the simulator's own `Road.project`. The two cases past the end are the
# ones a missing clamp or `np.sign(left) * distance` gets wrong: on the line past the end the left part is exactly 0, so
# `np.sign` gives `d` = 0 where the true distance is 3.0 m. `np.copysign` keeps a sign even then.
#
# **Break it, fix it.** A teammate wrote the shortcut "the nearest point of the road is the nearest vertex". Run it.
#
# **Guess first:** will it pass? The vertices are only 0.1 m apart. Where will it first be wrong?

# %%
def to_road_frame_teammate(centerline, closed, p):
    """The teammate's version: the nearest vertex, not the nearest segment."""
    pts = np.asarray(centerline, dtype=float)
    a, b = (pts, np.roll(pts, -1, axis=0)) if closed else (pts[:-1], pts[1:])
    seg = np.hypot(*(b - a).T)
    j = int(np.argmin(np.hypot(*(pts - p).T)))  # the nearest VERTEX
    k = min(j, len(a) - 1)  # the segment that starts there (the last vertex of an open road ends one)
    u = (b[k] - a[k]) / seg[k]
    left = u[0] * (p[1] - pts[j][1]) - u[1] * (p[0] - pts[j][0])
    return float(np.concatenate([[0.0], np.cumsum(seg)])[j]), float(np.copysign(np.hypot(*(pts[j] - p)), left))

truth = np.array([road.project(x, y) for x, y in zip(tel.x, tel.y)])
mine = np.array([to_road_frame_teammate(centerline, True, (x, y)) for x, y in zip(tel.x, tel.y)])
print(f"on gentle, worst |d| error over the lap: {np.abs(np.abs(mine[:, 1]) - np.abs(truth[:, 1])).max():.2e} m")
print("on the corner road at (8, 3), the teammate says (s, d) =", to_road_frame_teammate(L_ROAD, False, (8.0, 3.0)))
check("1.1.3.a", to_road_frame_teammate)

# %% [markdown]
# On `gentle` the shortcut is off by under a millimetre ($9.82 \times 10^{-4}$ m), because the vertices are close. On the corner it
# says `d` = +3.606 where the truth is +2.000: the nearest vertex of (8, 3) is the corner (10, 0), 3.606 m away,
# while the second segment passes 2.000 m from it. A vertex is the nearest point of the road only when the
# vertices are very close together. The fix is the one you already wrote: project onto segments.
#
# ![](figures/l_corner.png)
#
# ![](figures/d_sideways.png)
#
# ## 4. How far along: arc length
#
# `s` is the length of the segments before the winning one, plus `t` (lecture: "How far along: arc length").
#
# **Guess first:** at step 400, 20.00 s into the lap at 8 m/s, the car has driven 160 m. Is its `s` more or less than
# 160?

# %%
s_400 = None  # your code: the s of the car at step 400, from to_road_frame(centerline, True, (tel.x[400], tel.y[400]))
practice("s at step 400 (m)", s_400, 159.011, tol=1e-3)

# %% [markdown]
# The road bends left and the car drives in the right lane, which is the outside of the bend, so its wheels cover more
# ground than the yellow line inside it: `s` is 159.011 m, a little less than 160. In the last step of the lap `s` goes from 702.306 m to 0.045 m, so the plain difference
# is -702.261 m, a whole lap backwards. Challenge problem 2, in the stretch section at the end, fixes that with the road's length.
#
# ![](figures/arc_length_l.png)
#
# ![](figures/step400_foot.png)
#
# ![](figures/road_grid.png)
#
# ## 5. The two errors the controller needs (Exercise B)
#
# The lateral error is `e = d + 1.8`, and the heading is `psi = wrap(theta - theta_road)`, where `theta_road` is
# the direction of the same nearest segment (lecture: "The two errors the controller needs").
#
# **Guess first:** the telemetry's `yaw` never wraps: past west it keeps growing past pi. What share of the lap does the plain
# difference `yaw - theta_road` get wrong by more than 1 rad? (`theta_road` from `atan2` stays in `[-pi, pi]`.)

# %%
theta_road = np.array([road.pose_at(road.project(x, y)[0])[2] for x, y in zip(tel.x, tel.y)])  # the simulator's own road direction, in [-pi, pi]
plain = tel.yaw - theta_road  # theta - theta_road, no wrap
pct = None  # your code: the share of the steps (%) where |plain - tel.head_err| is more than 1 rad
practice("share wrong by > 1 rad (%)", pct, 35.9, tol=0.1)

# %% [markdown]
# The plain difference is a whole turn out (6.2832 rad) on 35.9 % of the lap: from 56.95 s on, when the car crosses
# west and `yaw` passes pi while `theta_road` jumps from +pi to -pi.
#
# ![](figures/e_from_d.png)
#
# ![](figures/foot_frame.png)
#
# ![](figures/wrap_trap.png)
#
# Now `lane_errors`, which is Exercise B. Use the same nearest-segment search as in `to_road_frame`, and keep the winning segment: its `d` gives
# `e`, and its direction, `np.arctan2(dy, dx)` for the segment's arrow `(dx, dy)`, is `theta_road`. Use the one segment for both.

# %% tags=["exercise:1.1.3.b"]
def lane_errors(centerline, closed, pose):
    """The two errors the controller needs, from the road's own coordinates.
    Input:   centerline, shape (N, 2); closed, True when the last vertex joins the first; pose, (x, y, theta) of the
             car's front axle, theta in radians and not necessarily wrapped.
    Output:  (e, psi): e = d + 1.8, the sideways distance from the lane centre, positive to the left; psi =
             wrap(theta - theta_road), the heading ψ in [-pi, pi), positive when the car points left of the road.
    Example: lane_errors(L_ROAD, False, (5.0, -2.0, 0.1))  ->  (-0.2, 0.1)
    """
    raise NotImplementedError

# %% [markdown]
# **Guess first (Exercise B):** with the same simulator numbers in front of you, how far off will your `e` and `psi` be
# over the whole lap: about 0.01, about $10^{-6}$, or exactly 0.0?

# %%
try:
    mine = np.array([lane_errors(centerline, True, (x, y, th)) for x, y, th in zip(tel.x, tel.y, tel.yaw)])
    practice("max |e − lat| (m)", float(np.abs(mine[:, 0] - tel.lat).max()), 0.0, tol=1e-9)
    misfit = max(psi_misfit(road, pose, psi) for pose, psi in zip(zip(tel.x, tel.y, tel.yaw), mine[:, 1]))  # 0 for either tied segment's direction
    practice("largest ψ misfit (rad), either tied segment right", misfit, 0.0, tol=1e-9)
except NotImplementedError:
    print("Write lane_errors above first, then run this cell again.")

# %%
check("1.1.3.b", lane_errors)

# %% [markdown]
# Written this way the two functions reproduce the simulator's `lat` and `head_err` exactly, with a difference of 0.0 at
# all 1778 steps. If `psi` is 0.00125 rad out at a few steps, your search picked the other segment at a vertex tie; section 6
# explains it, and the grader accepts either. (The ψ line above counts either as right too.)
#
# ![](figures/reproduced.png)
#
# **Break it, fix it.** A teammate left out the wrap: `psi = theta - theta_road`. Run it.
#
# **Guess first:** where in the lap does it go wrong, and by how much?

# %%
def lane_errors_teammate(centerline, closed, pose):
    """The teammate's version: the heading ψ without the wrap."""
    pts = np.asarray(centerline, dtype=float)
    a, b = (pts, np.roll(pts, -1, axis=0)) if closed else (pts[:-1], pts[1:])
    t, d = project_to_segment(a, b, pose[:2])
    i = int(np.argmin(np.abs(d)))
    return d[i] + 1.8, pose[2] - np.arctan2(b[i, 1] - a[i, 1], b[i, 0] - a[i, 0])

try:
    mine = np.array([lane_errors_teammate(centerline, True, (x, y, th)) for x, y, th in zip(tel.x, tel.y, tel.yaw)])
    wrong = np.abs(mine[:, 1] - tel.head_err) > 1.0
    practice("max |ψ − head_err| of the teammate (rad)", float(np.abs(mine[:, 1] - tel.head_err).max()), 6.2832, tol=1e-4)
    print(f"wrong from {tel.t[wrong][0]:.2f} s to the end of the lap")
except NotImplementedError:
    print("Write project_to_segment above first, then run this cell again.")
check("1.1.3.b", lane_errors_teammate)

# %% [markdown]
# `e` is right all the way round, and `psi` is a whole turn (6.2832 rad) out from 56.95 s on. The fix is the wrap:
# `wrap_angle(theta - theta_road)`.
#
# ## 6. Corners in the polyline
#
# A road you are handed may not have a vertex every 0.1 m. The next cell keeps `gentle`'s yellow line but only a vertex every
# 5 m (lecture: "Corners in the polyline"), and measures each method against the dense road's truth along the lap.
#
# **Guess first:** which is worse on the 80 m bend, the nearest-vertex shortcut or projecting onto segments, and
# about how much worse?

# %%
sparse = road.sample_points(5.0)
true_d = np.abs(truth[:, 1])  # |d| of every step on the dense road
vertex_d = np.abs([to_road_frame_teammate(sparse, True, (x, y))[1] for x, y in zip(tel.x, tel.y)])
segment_d = None  # your code: |d| of every step from your to_road_frame(sparse, True, (x, y))
v = float(np.abs(vertex_d - true_d).max())
p = None if segment_d is None else float(np.abs(segment_d - true_d).max())
practice("nearest vertex, worst |d| error (m)", v, 1.550, tol=0.005)
practice("segments, worst |d| error (m)", p, 0.0391, tol=1e-4)

# %% [markdown]
# The shortcut is 1.550 m out and the segments 0.0391 m. A straight chord `c` long misses an arc of radius `R` by the
# sagitta `c² / (8R)`: 25 / 640 = 0.0391 m for 5 m on 80 m. The simulator's own 0.1 m gives $1.563 \times 10^{-5}$ m.
#
# ![](figures/sparse_vertex.png)
#
# ![](figures/sagitta.png)
#
# ![](figures/spacing.png)
#
# One more thing real roads do: at a vertex the winning foot can be shared by two segments, which are equally near.
# `theta_road` then depends on the segment picked, by one corner angle, 0.00125 rad on `gentle`. The simulator keeps
# the first, and so should you, but a function that keeps the other is not wrong.
#
# ![](figures/vertex_tie.png)
#
# ## What's next
#
# You can now turn a position and a yaw into `e` and `psi`, the two numbers the controller steers on, by way of the road's own `s` and `d`. In the
# next lesson, *Poses you can chain*, the car loses the outside world and has to find out where it is from how it moved.
#
# ![](figures/next_lesson.png)
#
# ## Stretch: the challenge problems
#
# Optional, not graded.
#
# The lecture's three challenge problems, each with a cell that checks your numbers.
#
# **Problem 1.** On `gentle`'s first bend, at s = 260 m, put the car 0.3 m left of its lane centre and 0.04 rad
# left of the road. How far left, in the car's own frame, is the lane centre `x` metres ahead? The cell computes
# the true values with `world_to_ego`; you compute the quadratic `-e - psi * x + x**2 / (2 * R)` and the line `-e - psi * x`, with `R` = 81.8 m.
#
# **Guess first:** about where is each first 0.1 m out: the quadratic, and the straight line?
#
# ![](figures/challenge_setup.png)

# %%
e0, psi0, R = 0.3, 0.04, 81.8
pose = road.ego_pose(260.0, e0, psi0)
ss = np.arange(255.0, 320.0, 0.001)
ego = world_to_ego(pose, road.offset(ss, -1.8))  # the lane centre, in the car's frame
xs, ys = ego[:, 0], ego[:, 1]
grid = np.arange(0.0, 50.0, 0.01)
true_y = np.interp([5.0, 10.0, 20.0, 30.0], xs, ys)
print("true lane centre at x = 5, 10, 20, 30 m:", np.round(true_y, 4))
quad = None  # your code: the quadratic at x = 5, 10, 20, 30 (an array of four numbers)
x_quad = None  # your code: the first x on `grid` where the quadratic is more than 0.1 m from np.interp(grid, xs, ys)
x_line = None  # your code: the same for the straight line
practice("true lane centre, x = 5, 10, 20, 30 (m)", true_y, [-0.3469, -0.0874, 1.3653, 4.1277], tol=1e-3)
practice("quadratic at x = 5, 10, 20, 30 (m)", quad, [-0.3472, -0.0888, 1.3450, 4.0012], tol=1e-3)
practice("quadratic first 0.1 m out at x (m)", x_quad, 28.49, tol=0.05)
practice("straight line first 0.1 m out at x (m)", x_line, 4.05, tol=0.05)

# %% [markdown]
# The line ignores the bend, whose `x² / (2R)` grows fast, so it is 0.1 m out at 4.05 m; the quadratic holds to 28.49 m.
#
# **Problem 2.** Progress across the start line, and when it breaks. In the last step of the lap `s` goes from
# 702.306 m to 0.045 m. Wrap the difference the way `wrap_angle` wraps an angle, but into `[-L/2, L/2)` with `L` the road's
# length: `(ds + L/2) % L - L/2`. The wrap puts a difference of `s` in that range, so the car can move at most half the
# road's length per step.
#
# **Guess first:** how far did the car move in that step, and how fast would it have to go for the wrap to give the wrong
# answer, at 20 steps a second?

# %%
s_last = np.array([road.project(x, y)[0] for x, y in zip(tel.x[-2:], tel.y[-2:])])  # the simulator's s at the last two steps: 702.306 m, then 0.045 m
progress = None  # your code: the wrapped difference of s over the last step of the lap (m), with L = road.length
practice("progress in the last step (m)", progress, 0.3936, tol=1e-4)
step_limit = None  # your code: the largest move per step (m) the wrapped difference can tell from a jump back
speed_limit = None  # your code: the speed (m/s) at which one 0.05 s step moves that far
practice("largest safe move per step (m)", step_limit, 351.3, tol=0.1)
practice("speed at which it breaks (m/s)", speed_limit, 7027, tol=1.0)

# %% [markdown]
# Wrapped, the move is +0.394 m, less than the 0.40 m of a step at 8 m/s because the car is on the outside of the bend and `s`
# is measured along the shorter yellow line. The wrap breaks only above half the road, 351.3 m per step, which is 7027 m/s: never on this road.
#
# **Problem 3.** Why the shortcut is worst near the line. With vertices every 5 m, a point midway between two vertices
# and `|d|` from the line is `sqrt(d² + 2.5²)` from either, so the shortcut is out by that minus `|d|`.
#
# **Guess first:** is a car near the line hurt more, or one far from it?

# %%
dist = np.array([3.6, 1.8, 1.0, 0.5])
shortcut_error = None  # your code: the shortcut's error at each |d| in `dist` (m)
practice("shortcut error at |d| = 3.6, 1.8, 1.0, 0.5 (m)", shortcut_error, [0.783, 1.281, 1.693, 2.050], tol=1e-3)

# %% [markdown]
# The closer to the line, the more the shortcut loses: the error tends to 2.5 m at `d = 0`, the largest it can be on
# this road. A car in its lane centre (`|d|` = 1.8 m) is out by 1.281 m.
