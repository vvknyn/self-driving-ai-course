# %% [markdown]
# # Lab: Your first perception box
#
# You will turn image rows into metres, see why a pixel column has no fixed size, and write `estimate_lane`: the
# function that turns paint masks into the offset and heading the controller needs. Then you put it in the car.
# Read the lecture first; this lab points back to its sections by title and follows its order.
#
# Run the cells from top to bottom (Shift+Enter). Each problem has a "Guess first" line: write your guess down before you
# run the cell, then compare. `practice` prints a green tick or tells you what it expected.
#
# ![](figures/box_result.png)

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
from zero2fsd.car.provided import FAR_ROWS, MIN_PIXELS, NEAR_ROWS, lane_point
from zero2fsd.dashboard import dashboard
from zero2fsd.grade import check, practice
from zero2fsd.sim import BicycleState, Camera, make_scenario
from zero2fsd.palette import CHROME
from zero2fsd.sim.scenarios import TARGET_SPEED

# %% [markdown]
# Your `estimate_lane` starts from paint masks. The next cell uses the provided `paint_masks`. To build on your own
# from the lesson *Images are arrays*, replace the import with your function (it must keep the name `paint_masks`).

# %%
from zero2fsd.car.provided import paint_masks

# %% [markdown]
# ## The two numbers, and the worked-example frame
#
# The controller needs two numbers: how far the car is from the lane centre, and how far it is turned from the lane's
# direction (lecture: "Two numbers for the controller").
#
# ![](figures/radians.png)
#
# ![](figures/offset_heading.png)
#
# The next cell puts the car on the `straight` road, 20 m along, 0.4 m left of lane centre and turned 0.05 rad left: the
# worked example of the lecture. The near band and the far band, each between a pair of grey dashes, are the two strips
# of rows the box will look at.

# %%
road = make_scenario("straight").road
camera = Camera()
state = BicycleState(*road.ego_pose(20.0, 0.4, 0.05), TARGET_SPEED, 0.0)
frame = camera.render(road, state)
obs = Observation(frame, TARGET_SPEED, 0.0)

plt.figure(figsize=(8, 4.5))
plt.imshow(frame)
for rows in (NEAR_ROWS, FAR_ROWS):
    for row in (rows[0], rows[1] - 1):
        plt.axhline(row, color=CHROME["text"], ls="--", lw=1.5)
plt.title("the worked-example frame: near and far band (grey dashes)")
plt.show()

# %% [markdown]
# ## 1. Predict: how far does the paint move?
#
# Slide the car 0.5 m to the left. In the image, the white line (on the car's right) moves further right. In row 150
# the ground is 2.91 m ahead; row 100 is further up the picture (lecture: "Pixels are not metres").
#
# **Guess first:** in row 150 the white line moves about ______ columns, and in row 100 about ______ columns, because
# ______.

# %%
columns = {}
for offset in (0.0, 0.5):
    slid = BicycleState(*road.ego_pose(20.0, offset), TARGET_SPEED, 0.0)
    _, white = paint_masks(camera.render(road, slid))
    columns[offset] = [np.nonzero(white[row])[0].mean() for row in (150, 100)]  # mean column of the white paint
    print(f"car {offset} m left: white line at column {columns[offset][0]:.1f} in row 150, {columns[offset][1]:.1f} in row 100")
print(f"moved {columns[0.5][0] - columns[0.0][0]:.1f} columns in row 150, {columns[0.5][1] - columns[0.0][1]:.1f} in row 100")

# %% [markdown]
# The same 0.5 m of ground is 26.5 columns near the car and 9.0 columns at 9 m. Dividing 0.5 m by the width of one
# column (0.01886 m in row 150, 0.05737 m in row 100) predicts 26.5 and 8.7. If you guessed the same number for both
# rows, you assumed a fixed metres-per-column: the misconception of "Pixels are not metres". A column only becomes
# metres once you know its row.
#
# ![](figures/slide_shift.png)
#
# ![](figures/perspective_grid.png)
#
# ## 2. Rows are distances
#
# Rows turn into distances through the camera's geometry (lecture: "Slopes and angles", "The horizon").
#
# ![](figures/tan_curve.png)
#
# ![](figures/pinhole_screen.png)
#
# ![](figures/ray_side_view.png)
#
# ![](figures/row_150_ray.png)
#
# The worked example of "Rows are distances": how far ahead is the ground seen in row 150? Once by the formula, once
# with `Camera.pixel_to_ground`.

# %%
PITCH = math.radians(5.0)  # the camera tips down 5 degrees (lecture: "The horizon")
print(f"f = {camera.f:.1f} px, centre column {camera.cx}, centre row {camera.cy}, height {camera.mount_height} m")
print(f"horizon row: {camera.cy - camera.f * math.tan(PITCH):.2f}")

theta = PITCH + math.atan((150 - camera.cy) / camera.f)  # how far below horizontal the ray of row 150 points
print(f"row 150 by the formula: theta = {math.degrees(theta):.2f} deg, d = {camera.mount_height / math.tan(theta):.3f} m")
for u in (0, 159.5, 319):
    forward, left = camera.pixel_to_ground(u, 150)
    print(f"pixel_to_ground(column {u:5}, row 150) -> forward {forward:.3f} m, left {left:+.3f} m")

# %% [markdown]
# Every column of row 150 gives the same forward distance, 2.907 m: a row is a distance. The `left` values are a
# different story; that is the lesson of the previous section.
#
# **Guess first:** row 150 sees 2.91 m ahead. Does row 100, 50 rows further up, see about twice as far, three times, or
# more? Then draw the distance of every row from 179 to 80.

# %%
rows = np.arange(179, 79, -1)
distance = np.array([camera.pixel_to_ground(camera.cx, row)[0] for row in rows])
print(f"row 100 sees {distance[rows == 100][0]:.2f} m ahead: {distance[rows == 100][0] / distance[rows == 150][0]:.2f} times row 150")
plt.figure(figsize=(8, 3.5))
plt.plot(rows, distance, color=CHROME["border"], lw=2.5)
plt.xlabel("image row")
plt.ylabel("distance ahead (m)")
plt.gca().invert_xaxis()
plt.show()

# %% [markdown]
# Row 100 sees 9.09 m ahead, 3.13 times as far as row 150, although it is only 50 rows further up the picture: near
# the horizon each row covers more and more road.
#
# ![](figures/rows_to_distance.png)
#
# ## 3. Practice (not graded)
#
# Each problem gives you less than the one before. Replace each `None` with your code and rerun the cell.
#
# **Problem 1.** How far ahead is the ground seen in row 115, the last row of the far band? Use the formula from
# section 2: `theta`, then `d`.
#
# **Guess first:** about 5.6 m, or about 8 m? (The far band is rows 95 to 115; row 125 sees 4.44 m.)

# %%
d_115 = None  # your code

practice("distance ahead seen in row 115 (m)", d_115, 5.592, tol=0.001)

# %% [markdown]
# Row 115 sees 5.592 m: the far band starts there and reaches out to 11.45 m at row 95.
#
# **Problem 2.** How many metres of ground, sideways, does one column cover in row 125, the top of the near band? Use
# `camera.pixel_to_ground` on two neighbouring columns of that row. Give a positive number.
#
# **Guess first:** about 2.8 cm, from d / f = 4.44 / 160, or noticeably less?

# %%
metres_per_column = None  # your code

practice("metres per column in row 125", metres_per_column, 0.02839, tol=0.00001)

# %% [markdown]
# The shortcut d / f gives 0.0277 m; the exact width is 0.02839 m, a little more because the ground is seen at a slant
# (lecture: "Metres per pixel").
#
# ![](figures/column_width.png)
#
# ![](figures/ground_footprint.png)
#
# **Problem 3.** Step 2 of the algorithm in "From masks to a lane estimate", for one line: where on the ground is the
# centroid of the yellow paint in the near band of the worked-example `frame`? Give `(forward, left)`. Mind that
# `mask[125:180]` numbers its rows from 0 again.
#
# **Guess first:** is the point left or right of the car, and is it nearer than 2.06 m, between 2.06 m and 4.44 m, or
# further than 4.44 m? (Those are the ground distances of the near band's bottom and top rows.)
#
# ![](figures/bands.png)

# %%
yellow_point = None  # your code

practice("near-band yellow centroid on the ground (forward, left)", yellow_point, (2.553, 1.274), tol=0.001)

# %% [markdown]
# The yellow line is on the car's left, 1.274 m, and 2.553 m ahead, inside the near band's 2.06 to 4.44 m.
#
# <details><summary>If Problem 3 gives nan (open after trying)</summary>
#
# `np.nonzero(mask[125:180])` counts rows from the top of the slice, so its row 0 is image row 125. Without adding
# `125` to the mean row, the centroid lands near row 35, above the horizon, and `pixel_to_ground` returns nan there.
#
# </details>
#
# ![](figures/centroids.png)
#
# ![](figures/ground_points.png)
#
# ## 4. Graded exercises
#
# ### Exercise A and Exercise B: `estimate_lane`
#
# Write the whole box: the four steps of "From masks to a lane estimate", plus the rule of "What valid=False is for".
# The near band is image rows `NEAR_ROWS` = `(125, 180)` and the far band `FAR_ROWS` = `(95, 116)`, each as
# `[start, stop)`; a line counts as seen in a band only with at least `MIN_PIXELS` = 3 pixels there.
#
# Exercise A renders 40 frames on straight stretches, with the car up to 1 m off centre and up to 0.15 rad turned, and
# needs 90% of your estimates within 0.25 m and 0.05 rad. Exercise B puts your function in the car and drives `gentle`
# (lecture: "Swap your perception into the car"); the car must complete it with a mean distance from the lane centre
# under 0.5 m.
#
# ![](figures/lane_estimate.png)

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


# %% [markdown]
# **Guess first (Exercise A):** how many of the 40 frames will a correct box pass, and how far off will its worst frame
# be?

# %%
check("0.1.3.a", estimate_lane)

# %% [markdown]
# A correct box passes all 40, its worst frame 0.022 m and 0.0041 rad off, far inside the 0.25 m and 0.05 rad
# tolerance. A box that forgets to carry the line back to the car is wrong by about 2.85 m for each radian of heading.
#
# ![](figures/grader_scatter.png)
#
# Once Exercise A passes, drive. Exercise B runs your function in the car on `gentle` and shows the dashboard. Expect a
# mean distance from the lane centre near 0.376 m.

# %%
check("0.1.3.b", estimate_lane)

# %% [markdown]
# ## 5. When there is nothing to see
#
# Turn the car 0.6 rad left on the straight road (lecture: "What valid=False is for"). Count the paint in each band,
# and ask your box what it thinks.
#
# **Guess first:** which of the four paint counts (yellow or white, near or far) will be zero?

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
# The white line is gone from the near band (0 pixels), while the far band still sees 117 yellow and 78 white pixels. So
# there is no near lane-centre point and no line. Your box should say `valid=False`, with `nan` for both numbers. It
# should not say `0.0, 0.0`: that claims the car is centred and aligned, and the controller would believe it.
#
# ![](figures/blind_frame.png)
#
# How far can the car turn before that happens? The next cell turns a centred car in steps of 0.01 rad and asks the
# provided box for the first turn at which it says `valid=False`.

# %%
for sign in (+1, -1):
    for step in range(101):
        turn = sign * 0.01 * step
        if not provided.estimate_lane(Observation(camera.render(road, BicycleState(*road.ego_pose(20.0, 0.0, turn), TARGET_SPEED, 0.0)), TARGET_SPEED, 0.0)).valid:
            print(f"turned {'left' if sign > 0 else 'right'}: the first invalid frame is at {turn:+.2f} rad")
            break

# %% [markdown]
# Either way, the box goes blind at about 0.53 rad, 30 degrees.
#
# ## 6. Straight roads only
#
# How good is your box when the car sits exactly on lane centre, aligned with it (truth: 0 m and 0 rad)? The next
# cell renders 40 such frames on the straight road, and 40 inside the bends of `gentle` and `curvy` (lecture:
# "Straight roads only").
#
# **Guess first:** on a left bend, will the box say the car is left or right of the lane centre, and pointing left or
# right? On which road do you expect the biggest error?

# %%
def bend_starts(road, ahead=12.0, n=40):
    """n positions where the lane turns the same way from 1 m behind the car to `ahead` metres in front: one bend."""
    def turn(s):
        return np.diff(np.unwrap(road.pose_at(np.arange(s - 1.0, s + ahead + 1))[2]))
    starts = [s for s in np.arange(1.0, road.length - ahead - 1) if np.all(turn(s) > 1e-6) or np.all(turn(s) < -1e-6)]
    return np.array(starts)[np.linspace(0, len(starts) - 1, n).astype(int)]


def centred_errors(estimate, scenario, starts):
    """(offset, heading, turn) of `estimate` at each start, with the car centred and aligned (truth 0 m, 0 rad).
    `turn` is +1 on a left bend, -1 on a right bend and 0 on a straight."""
    road = make_scenario(scenario).road
    rows = []
    for s in starts:
        centred = BicycleState(*road.ego_pose(s), TARGET_SPEED, 0.0)
        est = estimate(Observation(camera.render(road, centred), TARGET_SPEED, 0.0))
        turn = np.diff(np.unwrap(road.pose_at(np.array([s, s + 1.0]))[2]))[0]
        rows.append((est.offset_m, est.heading_rad, np.sign(round(turn, 6))))
    return np.array(rows)


try:
    for scenario in ("straight", "gentle", "curvy"):
        starts = np.linspace(0.0, 150.0, 40) if scenario == "straight" else bend_starts(make_scenario(scenario).road)
        errors = centred_errors(estimate_lane, scenario, starts)
        within = ((np.abs(errors[:, 0]) < 0.25) & (np.abs(errors[:, 1]) < 0.05)).sum()
        print(f"{scenario}: {within} of 40 within 0.25 m and 0.05 rad")
        for turn, word in ((0, "straight"), (1, "left bends"), (-1, "right bends")):
            part = errors[errors[:, 2] == turn]
            if len(part):
                print(f"    {word:11} ({len(part)} frames): median offset {np.median(part[:, 0]):+.3f} m, median heading {np.median(part[:, 1]):+.4f} rad")
except NotImplementedError:
    print("Write estimate_lane above first, then run this cell again.")

# %% [markdown]
# On the straight road your box is almost exact. In the bends it is wrong every time, and wrong the same way: on the
# left bends of `gentle` it says the car is 0.129 m left of centre and pointing 0.0645 rad to the right, though the car
# is centred and aligned. The lecture's formula predicts 0.119 m and 0.0604 rad. On `curvy` the errors are four to five times
# bigger and flip sign with the bend: +0.504 m and -0.248 rad on its left bends, -0.650 m and +0.308 rad on its right
# ones. A straight line cannot follow a curve. That is a limit of the model, not a bug in your code, and it is why the
# grader only samples straight stretches.
#
# ![](figures/centred_errors.png)
#
# ![](figures/chord_bias.png)
#
# ![](figures/chord_prediction.png)
#
# The lecture's last section shows what this bias does to the car (lecture: "Swap your perception into the car"). On
# `gentle` the car with the two-band box rides about 0.5 m off the lane centre in every bend (mean 0.376 m over the lap);
# with a perfect box and the same controller it is 0.113 m. On `curvy` the two-band box leaves the lane 48.3 m into the
# 218.5 m route, while the perfect box completes it at 0.348 m: the chord bias, not the controller alone.
#
# ![](figures/gentle_trace.png)
#
# ![](figures/like_for_like.png)
#
# ## 7. Break it, fix it
#
# A teammate wrote their own box on top of the provided one: "Offset is where the lane centre is, so report that." Run
# it on `gentle`.
#
# **Guess first:** does the car leave the road at once, wobble, or drift slowly at first and then fast? Why does it
# happen on the first straight, where the box is nearly exact? Then fix the one wrong line and rerun the cell until the
# car completes the lap.

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
# error is 1.8 to 1.9 times what it was half a second earlier (-0.043, -0.081, -0.155, -0.288, -0.544, -1.010 m) until
# the car crosses the white line at 5.00 s, 39.9 m along the road, still on the first straight. The fix is
# `offset_m=est.offset_m`. A wrong sign does not make a box a little worse; it turns correction into amplification.
#
# </details>
#
# ![](figures/flipped_sign.png)
#
# **A second break.** Flip the heading's sign instead (`heading_rad=-est.heading_rad`). Guess first: does the car last
# longer or shorter than with the flipped offset?

# %%
def estimate_lane_flipped_heading(obs):
    est = provided.estimate_lane(obs)
    return LaneEstimate(offset_m=est.offset_m, heading_rad=-est.heading_rad, valid=est.valid)


tel2 = run(Car(perception=estimate_lane_flipped_heading), "gentle")
print("completed:", tel2.completed, f" ended after {tel2.t[-1]:.2f} s and {tel2.progress_m[-1]:.1f} m")

# %% [markdown]
# The car is off the road sooner, at 2.70 s and 21.1 m: the heading term is the stronger correction (its gain is 2.7 against 0.35 for the offset; you open it in
# *Closing the loop*), so flipping it is the more violent mistake.
#
# You now have a perception box of your own, between the camera and a controller that is still the provided one.
#
# ![](figures/next_lesson.png)
#
# ## 8. Stretch (optional, not graded): a box that sees the bend
#
# The lecture's challenge: draw a curve instead of a line. Use three bands, rows `(125, 180)`, `(106, 125)` and
# `(95, 106)`, find a lane-centre point in each with `lane_point`, and fit `np.polyfit(ds, cs, 2)`; it returns `(k, b, a)`,
# the `d` squared term first. Return `LaneEstimate(-a, -math.atan(b), True)`, or `valid=False` if a band has no point.
#
# ![](figures/curve_box.png)
#
# **Guess first:** on `gentle`, will the mean distance from the lane centre land nearer the perfect box's 0.113 m or the
# two-band box's 0.376 m?

# %%
BANDS = ((125, 180), (106, 125), (95, 106))


def estimate_lane_curve(obs):
    """Offset and heading from a parabola through three lane-centre points."""
    raise NotImplementedError


try:
    practice("curve box, offset on the worked-example frame", estimate_lane_curve(obs).offset_m, 0.3987, tol=0.002)
    practice("curve box, heading on the worked-example frame", estimate_lane_curve(obs).heading_rad, 0.0475, tol=0.002)
    curve_lap = run(Car(perception=estimate_lane_curve), "gentle")
    practice("curve box on gentle, mean distance from the lane centre (m)", float(np.abs(curve_lap.lat).mean()), 0.119, tol=0.01)
except NotImplementedError:
    print("Write estimate_lane_curve above first, then run this cell again.")

# %% [markdown]
# The lecture's second challenge, choosing where the far point sits, has the picture below: a point further out is
# less sensitive to the paint's position but adds more chord bias on a bend.
#
# ![](figures/far_band_choices.png)
#
# Optional: run the same curve box on `curvy`. The two-band box leaves the lane there; a working curve box finishes the
# route, with a mean distance of about 0.420 m.

# %%
try:
    curvy_lap = run(Car(perception=estimate_lane_curve), "curvy")
    print("completed:", curvy_lap.completed, f"mean distance from the lane centre {np.abs(curvy_lap.lat).mean():.3f} m")
except NotImplementedError:
    print("Write estimate_lane_curve above first, then run this cell again.")

# %% [markdown]
# ## What's next
#
# You can now find the paint, turn it into ground points and report the offset and heading the controller needs. In
# the next lesson, *Closing the loop*, you write the steering controller that uses them, tune its two gains, and watch
# what the bend bias from section 6 does to the car on `curvy`.
