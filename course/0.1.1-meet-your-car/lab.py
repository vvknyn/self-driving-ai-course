# %% [markdown]
# # Lab 0.1.1: Meet your car
#
# You will drive the provided car, predict where it fails, and write the two functions that turn a drive into numbers.
# Read the lecture first; this lab points back to its sections by title.
#
# Run the cells from top to bottom (Shift+Enter).

# %%
import sys
if "google.colab" in sys.modules:
    %pip install -q "git+https://github.com/vvknyn/self-driving-ai-course@main"
%matplotlib inline

# %% tags=["setup"]
import time

import numpy as np

from zero2fsd.car import Car, provided, run
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
# **Before you run it**, double-click the cell below and write your prediction.

# %% [markdown]
# **My prediction:** the car fails on the ______ road, about ______ m in, drifting to the ______ (left or right).

# %%
tel_gentle = run(Car(), "gentle")
tel_curvy = run(Car(), "curvy")
print(metrics_table(tel_gentle, tel_curvy, labels=["gentle", "curvy"]))

# %%
dashboard(tel_gentle)

# %%
dashboard(tel_curvy)

# %% [markdown]
# Compare with your prediction. On `curvy` the car left the road 48 m in, partway round the first bend, a left-hander.
# The lateral-error plot shows the error climbing past +0.9 m and on to +1.8 m, so the car left on the **left**,
# the inside of that bend. Unit 0.1.4 finds out why. For now, you will measure it.
#
# ## 2. What a drive is made of
#
# Everything on the dashboard comes from arrays with one value per step. The lateral error is `tel.lat`
# (lecture: "Lateral error").

# %%
lat = tel_gentle.lat
print("type:", type(lat).__name__, " shape:", lat.shape, " dtype:", lat.dtype)
print("first five values (m):", np.round(lat[:5], 4))
print("values 500 to 504 (m):  ", np.round(lat[500:505], 4), " (about 200 m in, inside the first bend)")
print("time between steps (s):", tel_gentle.t[1] - tel_gentle.t[0])

# %% [markdown]
# ### Worked example
#
# This is the five-number example from "Summarising lane error", one NumPy tool per line. Run it and match each printed
# value to the lecture. (You will see `0.27999999999999997` for 0.28: computers store most decimals approximately.)

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

# %%
e_b = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
mean_abs_b = np.abs(e_b).mean()
rms_b = None  # your code: square, mean, root

practice("mean |e| of drive B", mean_abs_b, 0.2)
practice("RMS of drive B", rms_b, 0.4472136)

# %% [markdown]
# **Problem 2.** How many **seconds** did the provided car spend off the lane ($|e| > 0.9$ m) on `curvy`?
# The loop below counts the steps the slow way. Write one line of NumPy that gives the seconds (lecture: "Counting
# steps off the lane" says how many steps make a second).

# %%
loop_count = 0
for x in tel_curvy.lat:
    if abs(x) > 0.9:
        loop_count += 1

seconds_off = None  # your code: one NumPy line, in seconds

practice("seconds off the lane on curvy", seconds_off, loop_count / 20)

# %% [markdown]
# **Problem 3.** For the same offset from centre, this car steers back twice as hard as the provided one. Compute its signed mean and its
# mean $|e|$ on `curvy`. Then decide: which of the two cars does each number say is better, and which number is telling
# the truth? (Lecture: "The signed-mean trap".)

# %%
tel_stiff = run(Car(controller=lambda est, obs: provided.steer_p(est, 0.7, 2.7)), "curvy")

signed_stiff = None  # your code
mean_abs_stiff = None  # your code

practice("signed mean, stiffer car on curvy (m)", signed_stiff, 0.0234, tol=0.0005)
practice("mean |e|, stiffer car on curvy (m)", mean_abs_stiff, 0.5021, tol=0.0005)

# %% [markdown]
# ## 4. Graded exercises
#
# ### 0.1.1.a `lane_error_stats`
#
# Write the three summaries from "Summarising lane error" with NumPy. The checker compares your answers with a plain
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
# ### 0.1.1.b `steps_off_lane`
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
# ## 5. Break it, fix it
#
# A teammate wrote this metric to rank cars ("smaller is better"). Run it. It ranks the stiffer car far ahead of the
# provided one. Before you open the hint: which line is wrong, and why? Then fix it and rerun the cell until the ranking
# flips.

# %%
def lane_keeping_error(lat_err):
    """How far the car strays from lane centre, in metres (smaller is better)."""
    return abs(np.mean(lat_err))


print(f"provided car on gentle: {lane_keeping_error(tel_gentle.lat):.2f} m")
print(f"stiffer car on curvy:   {lane_keeping_error(tel_stiff.lat):.2f} m")

# %% [markdown]
# <details><summary>Hint (open after you have a guess)</summary>
#
# `abs(np.mean(lat_err))` averages first and takes the size afterwards, so left errors cancel right errors before the
# size is taken: that is the signed-mean trap. Take the size of every error first: `np.mean(np.abs(lat_err))`. The
# provided car then scores 0.38 m and the stiffer car 0.50 m, and the ranking flips.
#
# </details>
#
# ## What's next
#
# You can now say, in numbers, how well any car keeps its lane. In unit 0.1.2 you open the camera picture itself and
# find the paint in it.
