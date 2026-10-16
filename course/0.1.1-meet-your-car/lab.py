# %% [markdown]
# # Lab: Meet your car
#
# You will drive the provided car, predict where it fails, and write the two functions that turn a drive into numbers.
# Read the lecture first; this lab points back to its sections by title.
#
# Run the cells from top to bottom (Shift+Enter).
#
# ![](figures/report_card.png)
#
# ![](figures/loop_diagram.png)

# %%
import sys
if "google.colab" in sys.modules:
    %pip install -q "git+https://github.com/vvknyn/self-driving-ai-course@main"
%matplotlib inline

# %% tags=["setup"]
import time

import numpy as np

from zero2fsd.car import Action, Car, provided, run
from zero2fsd.dashboard import dashboard
from zero2fsd.grade import check, practice
from zero2fsd.score import metrics_table

# %% [markdown]
# ## 1. Drive the car, but predict first
#
# The next code cell drives the provided car on two roads (see "The car you were handed"):
#
# - `gentle`: a 702.7 m loop with two bends of radius 80 m.
# - `curvy`: 30 m of straight, then six quarter-circle bends of radius 20 m, alternating left and right.
#
# ![](figures/car_and_view.png)
#
# ![](figures/test_roads.png)
#
# **Before you run it**, double-click the cell below and write your prediction.

# %% [markdown]
# **My prediction:** before it fails on `curvy`, the car's mean |e| will be bigger / smaller than its 0.38 m on `gentle`, and it will cross the ______ line (yellow or white).

# %%
tel_gentle = run(Car(), "gentle")
tel_curvy = run(Car(), "curvy")
print(metrics_table(tel_gentle, tel_curvy, labels=["gentle", "curvy"]))

# %%
dashboard(tel_gentle)

# %%
dashboard(tel_curvy)

# %% [markdown]
# Compare with your prediction. On `curvy` the run ended 48 m in, partway round the first bend, a left-hander.
# Over those 48 m its mean |e| was 0.55 m, bigger than 0.38 m. The lateral-error plot shows the error climbing past
# +0.9 m and on to +1.8 m, so the car crossed the **yellow** line, on the inside of that bend. The lesson *Closing the
# loop* finds out why. For now, you will measure it.
#
# About the table: `completion` is how much of the route the car drove. Ignore `steer rate` and `score` for now. RMS is
# not in this table: you compute it in Exercise A. The dashboard's third panel compares the perception box's estimate
# with the truth; it belongs to *Your first perception box*.
#
# ## 2. What a drive is made of
#
# Everything on the dashboard comes from arrays with one value per step. The lateral error is `tel.lat`
# (lecture: "Lateral error"). In the next cell, `shape` is how many values the array holds (a one-element tuple such
# as `(1778,)` means a flat list of 1778 numbers), and `dtype` is the kind of number stored (`float64` means decimals).
#
# ![](figures/sign_convention.png)
#
# ![](figures/gentle_trace.png)

# %%
lat = tel_gentle.lat
print("type:", type(lat).__name__, " shape:", lat.shape, " dtype:", lat.dtype)
print("first five values (m):", np.round(lat[:5], 4))
print("values 500 to 504 (m):  ", np.round(lat[500:505], 4), " (about 200 m in, inside the first bend)")
print("time between steps (s):", tel_gentle.t[1] - tel_gentle.t[0])

# %% [markdown]
# ### Worked example
#
# This is the five-number example from "The tempting shortcut", one NumPy tool per line.
#
# ![](figures/cancel_numberline.png)
#
# ![](figures/signed_mean_trap.png)
#
# ![](figures/fold_numberline.png)
#
# ![](figures/squares_rms.png)
#
# **Guess first:** `e.mean()` is the signed mean of these five errors. Will it print 0, 0.28, or something else? Then run
# the cell and match each printed value to the lecture. (You will see `0.27999999999999997` for 0.28: computers store most decimals approximately.)

# %%
e = np.array([0.1, -0.3, 0.2, -0.4, 0.4])  # array creation: a list of numbers becomes an array
print("np.abs(e):           ", np.abs(e))  # every element's size, signs dropped
print("mean |e|:            ", np.abs(e).mean())
print("max |e|:             ", np.abs(e).max())
print("e ** 2:              ", e ** 2)  # every element squared
print("RMS:                 ", np.sqrt((e ** 2).mean()))  # square, mean, root
print("np.abs(e) > 0.35:    ", np.abs(e) > 0.35)  # one True/False per element
print("(np.abs(e) > 0.35).sum():", (np.abs(e) > 0.35).sum())  # True counts as 1
print("signed mean (a trap):", e.mean())

# %% [markdown]
# ### Arrays, not loops
#
# The same mean $|e|$ on a million errors, as a Python loop and as one NumPy line (lecture: "Arrays, not loops").
#
# **Guess first:** how many times faster will the NumPy line be: 2 times, 20 times, or 200 times?

# %%
big = np.random.default_rng(0).normal(0.0, 0.3, 1_000_000)

start = time.perf_counter()
total = 0.0
for x in big:
    total += abs(x)
loop_answer, loop_seconds = total / len(big), time.perf_counter() - start

start = time.perf_counter()
numpy_answer, numpy_seconds = np.abs(big).mean(), time.perf_counter() - start

print(f"loop:  {loop_answer:.6f} m in {loop_seconds:.4f} s")
print(f"numpy: {numpy_answer:.6f} m in {numpy_seconds:.4f} s  ({loop_seconds / numpy_seconds:.0f} times faster here)")

# %% [markdown]
# ## 3. Practice (not graded)
#
# Each problem gives you less than the one before. Replace each `None` with your code and rerun the cell; `practice`
# prints a green tick or tells you what it expected.
#
# **Problem 1.** Drive B from "Why RMS punishes big errors". Mean $|e|$ is done for you; write the RMS.
#
# ![](figures/drive_a_b.png)
#
# **Guess first:** will the RMS of drive B be smaller than, equal to, or bigger than its mean $|e|$ of 0.2 m?

# %%
e_b = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
mean_abs_b = np.abs(e_b).mean()
rms_b = None  # your code: square, mean, root

practice("mean |e| of drive B", mean_abs_b, 0.2)
practice("RMS of drive B", rms_b, 0.4472136)

# %% [markdown]
# **Problem 2.** How many **seconds** did the provided car spend off the lane ($|e| > 0.9$ m) on `curvy`?
#
# ![](figures/offlane_zone.png)
#
# ![](figures/curvy_offlane.png)
#
# The lecture answered the guess: whole seconds. The loop below counts the steps the slow way. Write one line of NumPy that gives the seconds (lecture: "Counting
# steps off the lane" says how many steps make a second).

# %%
loop_count = 0
for x in tel_curvy.lat:
    if abs(x) > 0.9:
        loop_count += 1

seconds_off = None  # your code: one NumPy line, in seconds

practice("seconds off the lane on curvy", seconds_off, loop_count / 20)

# %% [markdown]
# **Problem 3.** Two more cars drive `curvy`. The 2× car steers back toward the lane centre twice as hard as the
# provided car when it is off centre, and the 4× car four times as hard. Both finish the road. Compute each car's signed
# mean and its mean $|e|$. Then decide: which car does the signed mean say is better? Which does mean $|e|$ say?
# (Lecture: "The tempting shortcut".)

# %%
car_2x = Car(controller=lambda est, obs: provided.steer_p(est, 0.7, 2.7))
car_4x = Car(controller=lambda est, obs: provided.steer_p(est, 1.4, 2.7))
tel_2x = run(car_2x, "curvy")
tel_4x = run(car_4x, "curvy")

signed_2x = None  # your code
mean_abs_2x = None  # your code
signed_4x = None  # your code
mean_abs_4x = None  # your code

practice("signed mean, 2x car on curvy (m)", signed_2x, 0.0234, tol=0.0005)
practice("mean |e|, 2x car on curvy (m)", mean_abs_2x, 0.5021, tol=0.0005)
practice("signed mean, 4x car on curvy (m)", signed_4x, 0.0239, tol=0.0005)
practice("mean |e|, 4x car on curvy (m)", mean_abs_4x, 0.0866, tol=0.0005)

# %% [markdown]
# The signed mean calls it a tie, +0.02 m each. Mean $|e|$ picks the 4× car, 0.09 m against 0.50 m.

# %% [markdown]
# ## 4. What does doing nothing score?
#
# A number means little until you have something to set it against, so score the laziest car there is: one that never steers.
#
# **Guess first:** on which of the three roads, if any, does this do-nothing car score a perfect 0.00?

# %%
do_nothing = Car(controller=lambda est, obs: Action(0.0, provided.hold_speed(obs.speed)))
runs = [run(car, road) for road in ("straight", "gentle") for car in (do_nothing, Car())]
print(metrics_table(*runs, labels=["do-nothing, straight", "provided, straight", "do-nothing, gentle", "provided, gentle"]))

# %% [markdown]
# ![](figures/yardstick.png)
#
# On `straight` the do-nothing car is as good as a perfect driver: all three numbers are 0.00 m. On `gentle` its run ends
# 117 m in, at the first bend. That is why one road is never enough, and why the provided car's 0.38 m now has something to
# be measured against.
#
# One more thing in the table. The do-nothing car's mean |e| on `gentle` is only 0.10 m, lower than the provided car's
# 0.38 m. That is not a better drive. It is an average over the 117 m the car managed, most of it straight road, and its
# `completion` is only about 17%. A number from an unfinished run cannot be compared with a full lap, so check
# completion first.

# %% [markdown]
# ## 5. Graded exercises
#
# ### Exercise A: `lane_error_stats`
#
# ![](figures/array_ops.png)
#
# Write the three summaries from "Three honest numbers" with NumPy. The checker compares your answers with a plain
# Python loop on six arrays, negative values included.

# %% tags=["exercise:0.1.1.a"]
def lane_error_stats(lat_err):
    """Summarise a drive's lateral errors.

    Input:   lat_err, a 1-D NumPy array of lateral errors in metres, one per step (left of lane centre is
             positive).  It has at least one element.
    Output:  a dict of three floats, all in metres:
               "mean_abs": the mean of |e|
               "max_abs":  the largest |e|
               "rms":      the square root of the mean of e squared
    Example: lane_error_stats(np.array([0.1, -0.3, 0.2, -0.4, 0.4]))
             -> {"mean_abs": 0.28, "max_abs": 0.4, "rms": 0.3033...}
    """
    raise NotImplementedError


# %%
check("0.1.1.a", lane_error_stats)

# %% [markdown]
# ### Exercise B: `steps_off_lane`
#
# Count the steps where the car was off its lane (lecture: "Counting steps off the lane"). Mind the strict `>`.

# %% tags=["exercise:0.1.1.b"]
def steps_off_lane(lat_err, threshold=0.9):
    """Count the steps where the car was off its lane.

    Input:   lat_err, a 1-D NumPy array of lateral errors in metres; threshold in metres (default 0.9).
    Output:  an int, the number of steps with |e| strictly greater than threshold.  A step exactly at the
             threshold does not count.
    Example: steps_off_lane(np.array([0.1, -0.3, 0.2, -0.4, 0.4]), threshold=0.35) -> 2
    """
    raise NotImplementedError


# %%
check("0.1.1.b", steps_off_lane)

# %% [markdown]
# ## 6. Break it, fix it
#
# A teammate wrote this metric to rank cars ("smaller is better"). Run it on the 2× and 4× cars from Problem 3. Before
# you open the hint: which line is wrong, and why? Then fix it and rerun the cell until the two cars get different scores.

# %%
def lane_keeping_error(lat_err):
    """How far the car strays from lane centre, in metres (smaller is better)."""
    return abs(np.mean(lat_err))


print(f"2x car on curvy: {lane_keeping_error(tel_2x.lat):.2f} m")
print(f"4x car on curvy: {lane_keeping_error(tel_4x.lat):.2f} m")

# %% [markdown]
# <details><summary>Hint (open after you have a guess)</summary>
#
# `abs(np.mean(lat_err))` averages first and takes the size afterwards, so left errors cancel right errors before the
# size is taken: that is the signed-mean trap. Take the size of every error first: `np.mean(np.abs(lat_err))`. The
# 2× car then scores 0.50 m and the 4× car 0.09 m.
#
# </details>
#
# Now print both metrics on `gentle` for all three cars:

# %%
for name, car in (("provided", Car()), ("2x", car_2x), ("4x", car_4x)):
    lat_gentle = run(car, "gentle").lat
    print(f"{name:>8} on gentle: signed mean {np.mean(lat_gentle):+.3f} m, mean |e| {np.mean(np.abs(lat_gentle)):.3f} m")

# %% [markdown]
# On `gentle` the fix changes almost nothing: these cars stay on one side of the lane centre, so nothing cancels. The
# plain average only fails when a car crosses the lane centre, which is exactly when you most need to know.
#
# ## What's next
#
# You can now say, in numbers, how well any car keeps its lane. In *Images are arrays* you open the camera picture itself
# and find the paint in it.
#
# ![](figures/next_lesson.png)
