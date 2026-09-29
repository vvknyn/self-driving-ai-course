# %% [markdown]
# # Lab 0.1.3: Your first perception box
#
# You will turn image rows into metres, see why a pixel column has no fixed size, and write `estimate_lane`: the
# function that turns paint masks into the offset and heading the controller needs. Then you put it in the car.
# Read the lecture first; this lab points back to its sections by title.
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

from zero2fsd.car import Car, LaneEstimate, Observation, provided, run
from zero2fsd.car.provided import FAR_ROWS, MIN_PIXELS, NEAR_ROWS
from zero2fsd.dashboard import dashboard
from zero2fsd.grade import check, practice
from zero2fsd.sim import BicycleState, Camera, make_scenario
from zero2fsd.sim.scenarios import TARGET_SPEED

# %% [markdown]
# Your `estimate_lane` starts from paint masks. The next cell uses the provided `paint_masks`. To build on your own
# from unit 0.1.2, replace the import with your function (it must keep the name `paint_masks`).

# %%
from zero2fsd.car.provided import paint_masks

# %% [markdown]
# ## 1. Rows are distances
#
# The next cell puts the car on the `straight` road, 20 m along, 0.4 m left of lane centre and turned 0.05 rad left:
# the worked example of "From masks to a lane estimate". Then it does the worked example of "Rows are distances": how
# far ahead is the ground seen in row 150? Once by the formula, once with `Camera.pixel_to_ground`.

# %%
road = make_scenario("straight").road
camera = Camera()
state = BicycleState(*road.ego_pose(20.0, 0.4, 0.05), TARGET_SPEED, 0.0)
frame = camera.render(road, state)
obs = Observation(frame, TARGET_SPEED, 0.0)

PITCH = math.radians(5.0)  # the camera tips down 5 degrees (lecture: "The horizon")
print(f"f = {camera.f:.1f} px, centre column {camera.cx}, centre row {camera.cy}, height {camera.mount_height} m")
print(f"horizon row: {camera.cy - camera.f * math.tan(PITCH):.2f}")

theta = PITCH + math.atan((150 - camera.cy) / camera.f)  # how far below horizontal the ray of row 150 points
print(f"row 150 by the formula: theta = {math.degrees(theta):.2f} deg, d = {camera.mount_height / math.tan(theta):.3f} m")
for u in (0, 159.5, 319):
    forward, left = camera.pixel_to_ground(u, 150)
    print(f"pixel_to_ground(column {u:5}, row 150) -> forward {forward:.3f} m, left {left:+.3f} m")

plt.figure(figsize=(8, 4.5))
plt.imshow(frame)
for rows, colour in ((NEAR_ROWS, "tab:orange"), (FAR_ROWS, "tab:cyan")):
    for row in (rows[0], rows[1] - 1):
        plt.axhline(row, color=colour, ls="--", lw=1)
plt.title("the worked-example frame: near band (orange) and far band (cyan)")
plt.show()

# %% [markdown]
# Every column of row 150 gives the same forward distance, 2.907 m: a row is a distance. The `left` values are a
# different story; that is the next section.
#
# ## 2. Predict: how far does the paint move?
#
# Slide the car 0.5 m to the left. In the image, the white line (on the car's right) moves further right. In row 150
# the ground is 2.91 m ahead; in row 100 it is 9.09 m ahead (lecture: "Metres per pixel").
#
# **Before you run the next cell**, double-click this one and write your prediction.
#
# **My prediction:** in row 150 the white line moves about ______ columns, and in row 100 about ______ columns,
# because ______.

# %%
columns = {}
for offset in (0.0, 0.5):
    slid = BicycleState(*road.ego_pose(20.0, offset), TARGET_SPEED, 0.0)
    _, white = paint_masks(camera.render(road, slid))
    columns[offset] = [np.nonzero(white[row])[0].mean() for row in (150, 100)]  # mean column of the white paint
    print(f"car {offset} m left: white line at column {columns[offset][0]:.1f} in row 150, {columns[offset][1]:.1f} in row 100")
print(f"moved {columns[0.5][0] - columns[0.0][0]:.1f} columns in row 150, {columns[0.5][1] - columns[0.0][1]:.1f} in row 100")

# %% [markdown]
# The same 0.5 m of ground is 26.5 columns near the car and 9 columns at 9 m; the lecture's worked example predicted
# 26.5 and 8.7. If you guessed the same number for both rows, you assumed a fixed metres-per-column: the misconception of
# "Metres per pixel". A column only becomes metres once you know its row.
#
# ## 3. Practice (not graded)
#
# Each problem gives you less than the one before. Replace each `None` with your code and rerun the cell.
#
# **Problem 1.** How far ahead is the ground seen in row 116, the last row of the far band? Use the formula from
# section 1: `theta`, then `d`.

# %%
d_116 = None  # your code

practice("distance ahead seen in row 116 (m)", d_116, 5.451, tol=0.001)

# %% [markdown]
# **Problem 2.** How many metres of ground, sideways, does one column cover in row 125, the top of the near band? Use
# `camera.pixel_to_ground` on two neighbouring columns of that row. Give a positive number.

# %%
metres_per_column = None  # your code

practice("metres per column in row 125", metres_per_column, 0.02839, tol=0.00001)

# %% [markdown]
# **Problem 3.** Step 2 of the algorithm in "From masks to a lane estimate", for one line: where on the ground is the
# centroid of the yellow paint in the near band of the worked-example `frame`? Give `(forward, left)`. Mind that
# `mask[125:180]` numbers its rows from 0 again.

# %%
yellow_point = None  # your code

practice("near-band yellow centroid on the ground (forward, left)", yellow_point, (2.553, 1.274), tol=0.001)

# %% [markdown]
# <details><summary>If Problem 3 gives nan (open after trying)</summary>
#
# `np.nonzero(mask[125:180])` counts rows from the top of the slice, so its row 0 is image row 125. Without adding
# `125` to the mean row, the centroid lands near row 35, above the horizon, and `pixel_to_ground` returns nan there.
#
# </details>
#
# ## 4. Graded exercise
#
# ### 0.1.3.a and 0.1.3.b `estimate_lane`
#
# Write the whole box: the four steps of "From masks to a lane estimate", plus the rule of "What valid=False is for".
# The near band is image rows `NEAR_ROWS` = `(125, 180)` and the far band `FAR_ROWS` = `(95, 116)`, each as
# `[start, stop)`; a line counts as seen in a band only with at least `MIN_PIXELS` = 3 pixels there.
#
# Check `0.1.3.a` renders 40 frames on straight stretches, with the car up to 1 m off centre and up to 0.15 rad
# turned, and needs 90% of your estimates within 0.25 m and 0.05 rad. Check `0.1.3.b` puts your function in the car
# and drives `gentle` (lecture: "Swap your perception into the car").

# %% tags=["exercise:0.1.3.a", "exercise:0.1.3.b"]
def estimate_lane(obs):
    """Where the car is in its lane, from one camera frame.

    Input:   obs, an Observation: obs.frame is the camera frame, a uint8 array of shape (180, 320, 3).
    Output:  LaneEstimate(offset_m, heading_rad, valid). offset_m > 0 when the car is LEFT of lane centre,
             heading_rad > 0 when the car points LEFT of the lane, both measured at the car (the front axle).
             valid=False, with math.nan for both numbers, when a band has fewer than MIN_PIXELS pixels of either line.
    Example: estimate_lane(obs)  with the worked-example obs of this lab
             -> LaneEstimate(offset_m=0.39..., heading_rad=0.051..., valid=True)
    """
    raise NotImplementedError


# %%
check("0.1.3.a", estimate_lane)

# %% [markdown]
# Once 0.1.3.a passes, drive. The check runs your function in the car on `gentle` and shows the dashboard.

# %%
check("0.1.3.b", estimate_lane)

# %% [markdown]
# ## 5. When there is nothing to see
#
# Turn the car 0.6 rad left on the straight road (lecture: "What valid=False is for"). Count the paint in each band,
# and ask your box what it thinks.

# %%
turned = BicycleState(*road.ego_pose(20.0, 0.0, 0.6), TARGET_SPEED, 0.0)
turned_frame = camera.render(road, turned)
for name, mask in zip(("yellow", "white"), paint_masks(turned_frame)):
    print(f"{name}: {mask[slice(*NEAR_ROWS)].sum()} pixels in the near band, {mask[slice(*FAR_ROWS)].sum()} in the far band")
try:
    print("your box:", estimate_lane(Observation(turned_frame, TARGET_SPEED, 0.0)))
except NotImplementedError:
    print("Write estimate_lane above first, then run this cell again.")
plt.figure(figsize=(8, 4.5))
plt.imshow(turned_frame)
plt.title("turned 0.6 rad left: the white line has left the near band")
plt.show()

# %% [markdown]
# The white line is gone from the near band, so there is no near lane-centre point and no line. Your box should say
# `valid=False`, with `nan` for both numbers. It should not say `0.0, 0.0`: that claims the car is centred and aligned,
# and the controller would believe it.
#
# ## 6. Straight roads only
#
# How good is your box when the car sits exactly on lane centre, aligned with it (truth: 0 m and 0 rad)? The next
# cell renders 40 such frames on the straight road, and 40 inside the bends of `gentle` and `curvy` (lecture:
# "Straight roads only"). Before you run it: on which road do you expect the biggest error, and why?

# %%
def bend_starts(road, ahead=12.0, n=40):
    """n positions where the lane turns the same way from 1 m behind the car to `ahead` metres in front: one bend."""
    def turn(s):
        return np.diff(np.unwrap(road.pose_at(np.arange(s - 1.0, s + ahead + 1))[2]))
    starts = [s for s in np.arange(1.0, road.length - ahead - 1) if np.all(turn(s) > 1e-6) or np.all(turn(s) < -1e-6)]
    return np.array(starts)[np.linspace(0, len(starts) - 1, n).astype(int)]


def centred_errors(estimate, scenario, starts):
    """|offset| and |heading| of `estimate` at each start, with the car centred and aligned (truth 0 m, 0 rad)."""
    road = make_scenario(scenario).road
    errors = []
    for s in starts:
        centred = BicycleState(*road.ego_pose(s), TARGET_SPEED, 0.0)
        est = estimate(Observation(camera.render(road, centred), TARGET_SPEED, 0.0))
        errors.append((abs(est.offset_m), abs(est.heading_rad)))
    return np.array(errors)


try:
    for scenario in ("straight", "gentle", "curvy"):
        starts = np.linspace(0.0, 150.0, 40) if scenario == "straight" else bend_starts(make_scenario(scenario).road)
        errors = centred_errors(estimate_lane, scenario, starts)
        within = ((errors[:, 0] < 0.25) & (errors[:, 1] < 0.05)).sum()
        print(f"{scenario:8}: median error {np.median(errors[:, 0]):.3f} m, {np.median(errors[:, 1]):.3f} rad; "
              f"{within} of 40 within 0.25 m and 0.05 rad")
except NotImplementedError:
    print("Write estimate_lane above first, then run this cell again.")

# %% [markdown]
# On the straight road your box is almost exact. In the bends it is wrong every time, and wrong the same way: on
# `gentle` about 0.13 m and 0.065 rad, as the lecture's formula predicts, and on `curvy` about 0.6 m and 0.3 rad. A
# straight line cannot follow a curve. That is a limit of the model, not a bug in your code, and it is why the grader
# only samples straight stretches. Unit 0.1.4 shows what this bias does to the car.
#
# ## 7. Break it, fix it
#
# A teammate wrote their own box on top of the provided one: "Offset is where the lane centre is, so report that." Run
# it on `gentle`. Before you open the hint: why does the car leave the road on the first straight, where the box is
# nearly exact? Then fix the one wrong line and rerun the cell until the car completes the lap.

# %%
def estimate_lane_teammate(obs):
    """The teammate's box."""
    est = provided.estimate_lane(obs)
    lane_centre_left = -est.offset_m  # where the lane centre is, seen from the car: left positive
    return LaneEstimate(offset_m=lane_centre_left, heading_rad=est.heading_rad, valid=est.valid)


tel = run(Car(perception=estimate_lane_teammate), "gentle")
print("completed:", tel.completed, f" ended after {tel.t[-1]:.1f} s and {tel.progress_m[-1]:.1f} m")
for k in range(0, len(tel.t), 10):
    print(f"t = {tel.t[k]:3.1f} s: lateral error {tel.lat[k]:+.3f} m, the box said offset {tel.est_offset[k]:+.3f} m")
dashboard(tel)

# %% [markdown]
# <details><summary>Hint (open after you have a guess)</summary>
#
# `lane_centre_left` is the lane seen by the car. `offset_m` must be the car seen by the lane: the same number with
# the opposite sign ("From masks to a lane estimate": both minus signs switch between the two). With the sign flipped,
# a car 0.1 m right of centre is told it is 0.1 m left, so the controller steers right, away from the centre; the
# error grows, and a bigger error gives a bigger wrong correction. This is **positive feedback**: from 2 s on, the
# error doubles about every half second (0.043, 0.081, 0.155, 0.288, 0.544, 1.010 m) until the car is 1.8 m off after
# 5 s, still on the first straight. The fix is `offset_m=est.offset_m`. A wrong sign does not make a box a little
# worse; it turns correction into amplification.
#
# </details>
#
# ## What's next
#
# You have a whole car of your own design, except the controller. In unit 0.1.4 you write the steering controller,
# tune its two gains, and watch what the bend bias from section 6 does on `curvy`.
