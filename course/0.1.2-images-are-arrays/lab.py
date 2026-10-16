# %% [markdown]
# # Lab: Images are arrays
#
# You will open a camera frame, pick it apart with indexing and slicing, measure colours as distances, find out why exact
# matching breaks on noise, and write the function that finds the paint in a noisy frame: the first half of your
# perception box.
# Read the lecture first; this lab points back to its sections by title and follows its order.
#
# Run the cells from top to bottom (Shift+Enter). Each problem has a "Guess first" line: write your guess down before you
# run the cell, then compare. `practice` prints a green tick or tells you what it expected.
#
# ![](figures/camera_to_masks.png)

# %%
import sys
if "google.colab" in sys.modules:
    %pip install -q "git+https://github.com/vvknyn/self-driving-ai-course@main"
%matplotlib inline

# %% tags=["setup"]
import matplotlib.pyplot as plt
import numpy as np

from zero2fsd.grade import check, practice
from zero2fsd.sim import BicycleState, Camera, make_scenario
from zero2fsd.sim.camera import GRASS, PALETTE, ROAD, SKY, WHITE, YELLOW
from zero2fsd.sim.scenarios import TARGET_SPEED

# %% [markdown]
# ## 1. A frame is a grid of numbers
#
# The next cell puts the car 20 m along the `straight` road and takes two pictures from the same spot: a clean one and
# one with camera noise of $\sigma = 6$. These are the frames the lecture's figures show.
#
# It also asks the simulator for `labels`: an array of shape `(180, 320)` saying what every pixel truly is, using the
# numbers `SKY, GRASS, ROAD, YELLOW, WHITE` = 0, 1, 2, 3, 4. A real camera gives you no such answer key. Here you use it
# only to score your masks.

# %%
road = make_scenario("straight").road
state = BicycleState(*road.ego_pose(20.0), TARGET_SPEED, 0.0)
camera = Camera()
clean = camera.render(road, state)
noisy, labels = camera.render(road, state, noise_sigma=6.0, seed=0, return_labels=True)

plt.figure(figsize=(6, 3.4))
plt.imshow(clean)
plt.title("clean")
plt.show()

# %% [markdown]
# **Problem 1.** Print the shape, the dtype and the number of values (`size`) of `clean`.
#
# **Guess first:** how many numbers is that in total? Compare with the lecture's count, 172,800.

# %%
frame_size = None  # your code: the total number of values in the frame

print("shape:", clean.shape, " dtype:", clean.dtype)
practice("number of values in the frame", frame_size, 172800)

# %% [markdown]
# ![](figures/frame_grid.png)

# %% [markdown]
# ## 2. Finding one pixel: row, column, channel
#
# **Problem 2.** Read the pixel `clean[179, 160]` (row first!) and `clean[0, 0]`; print both. Then read the **blue**
# value, index 2, of one yellow pixel and of one road pixel, using the coordinates from the lecture's patch: yellow is at
# row 150, column 59, and road at row 150, column 58.
#
# **Guess first:** which of the two blue values is bigger, the yellow line's or the road's?

# %%
print("clean[179, 160]:", clean[179, 160], "  clean[0, 0]:", clean[0, 0])

blue_yellow = None  # your code: the blue value of the yellow pixel at row 150, column 59
blue_road = None  # your code: the blue value of the road pixel at row 150, column 58

practice("blue value of the yellow pixel", blue_yellow, 40)
practice("blue value of the road pixel", blue_road, 95)

# %% [markdown]
# Yellow paint has the smaller blue value, so in the blue channel the yellow line is *darker* than the road, as in the
# lecture's picture. ALVINN also used the blue channel alone, for a related reason: in its road scenes,
# blue gave the most contrast between road and non-road.
#
# ![](figures/row_col.png)
#
# ![](figures/channels.png)
#
# ## 3. Slicing: taking a block of rows
#
# **Problem 3.** Print the shape of `clean[100:180]`.
#
# **Guess first:** how many rows will it have, 80 or 81?

# %%
print("frame[100:180].shape:", clean[100:180].shape)

# %% [markdown]
# ![](figures/slices.png)
#
# ![](figures/line_width.png)
#
# The lecture says rows 0 to 75 of every frame are sky. Check that on all three roads, 20 m along each (this prints
# `True` three times). Then work out how much of the frame is paint (`|` and `.sum()` are in the
# lecture's section "Scoring a mask: intersection over union").
#
# **Guess first:** what share of all the pixels is yellow or white paint: about 0.3, 0.03 or 0.003?

# %%
for name in ("straight", "gentle", "curvy"):
    scenario_road = make_scenario(name).road
    scenario_frame = camera.render(scenario_road, BicycleState(*scenario_road.ego_pose(20.0), TARGET_SPEED, 0.0))
    print(name, "rows 0 to 75 all sky:", bool(np.all(scenario_frame[:76] == PALETTE[SKY])))

paint_share = None  # your code: the number of YELLOW or WHITE pixels in labels, divided by labels.size

practice("share of the frame that is paint", paint_share, 0.0274, tol=0.0001)

# %% [markdown]
# ## 4. Colour is a point in space
#
# ![](figures/rgb_cube.png)
#
# ![](figures/distance_build.png)
#
# **Problem 4.** Write `colour_distance(p, c)`: the straight-line distance between two colours, $\sqrt{\Delta_R^2 + \Delta_G^2
# + \Delta_B^2}$, where $\Delta_R = p_R - c_R$ ("delta R") is the red difference, and likewise for green and blue.
# Convert to float first: that is the whole point of the last section of the lecture.
#
# **Guess first:** the lecture computed road grey to yellow paint as 186.3. Will grass green to yellow paint be bigger or
# smaller?

# %%
def colour_distance(p, c):
    """Distance between two colours given as (R, G, B) arrays."""
    return None  # your code: convert both to float, subtract, square, add up, take the root


practice("road grey to yellow paint", colour_distance(PALETTE[ROAD], PALETTE[YELLOW]), 186.3, tol=0.05)
practice("grass green to yellow paint", colour_distance(PALETTE[GRASS], PALETTE[YELLOW]), 175.8, tol=0.05)
practice("sky blue to white paint", colour_distance(PALETTE[SKY], PALETTE[WHITE]), 116.4, tol=0.05)

# %% [markdown]
# Now find the colour nearest to yellow paint, and the colour nearest to white paint, among the other four of the five:
# loop over `range(5)`, skip the paint itself, and keep the index with the smallest distance. The indices are
# `SKY, GRASS, ROAD, YELLOW, WHITE` = 0 to 4.

# %%
nearest_to_yellow = None  # your code: the index (0 to 4) of the colour nearest to yellow paint
nearest_to_white = None  # your code: the index of the colour nearest to white paint

practice("colour nearest to yellow", nearest_to_yellow, GRASS)
practice("colour nearest to white", nearest_to_white, SKY)

# %% [markdown]
# ![](figures/nearest_bars.png)

# %% [markdown]
# ## 5. Noise: no two pixels quite alike
#
# **Problem 5.** The nudge noise added to every value is `noisy - clean`, computed in `int`, not `uint8`. What share of
# all the nudges lies between -6 and +6?
#
# **Guess first:** the lecture says nudges bunch around 0 with a spread of 6. Is the share about 1 in 2, 7 in 10, or
# 99 in 100?

# %%
nudges = noisy.astype(int) - clean.astype(int)
share_within_6 = None  # your code: the fraction of values in `nudges` with absolute value at most 6

practice("share of nudges within 6", share_within_6, 0.721, tol=0.0005)

# %% [markdown]
# ![](figures/noise_bell.png)
#
# ![](figures/noisy_patch.png)

# %% [markdown]
# ## 6. The tempting shortcut: exact equality
#
# **Problem 6.** Count the pixels exactly equal to yellow paint, with `np.all(frame == PALETTE[YELLOW], axis=-1)`, on the
# clean frame and on the noisy frame.
#
# **Guess first:** the clean frame has 709 yellow pixels. How many will the noisy frame have: about 700, about 350, or
# none?

# %%
exact_clean = None  # your code: how many pixels of `clean` equal yellow paint exactly
exact_noisy = None  # your code: the same count on `noisy`

practice("exact matches on the clean frame", exact_clean, 709)
practice("exact matches on the noisy frame", exact_noisy, 0)

# %% [markdown]
# ![](figures/exact_fails.png)
#
# ![](figures/exact_odds.png)

# %% [markdown]
# ## 7. Paint is a colour threshold
#
# ![](figures/broadcast_shapes.png)
#
# **Problem 7.** Build the distance map of the noisy frame to yellow paint, in one expression, using broadcasting (the
# three numbers of yellow are subtracted from every pixel). Print its shape, then make three masks: yellow at
# $\tau = 10$, yellow at $\tau = 60$, and white at $\tau = 120$, and count each.
#
# **Guess first:** will the strict threshold of 10 find all 709 yellow pixels, about half of them, or hardly any?

# %%
distance_yellow = None  # your code: float frame minus float yellow, then np.linalg.norm(..., axis=-1)
distance_white = None  # your code: the same for white paint

if distance_yellow is not None:
    print("distance map shape:", distance_yellow.shape)
yellow_at_10 = None  # your code: how many pixels of distance_yellow are below 10
yellow_at_60 = None  # your code: below 60
white_at_120 = None  # your code: how many pixels of distance_white are below 120

practice("yellow pixels at tau = 10", yellow_at_10, 393)
practice("yellow pixels at tau = 60", yellow_at_60, 709)
practice("white pixels at tau = 120", white_at_120, 18098)

# %% [markdown]
# ![](figures/colour_distance.png)
#
# ![](figures/tau_sweep.png)
#
# At 10 the mask loses almost half the paint; at 120 the white mask swallows most of the sky, 18,098 pixels instead of
# 872. Between the two extremes there is a wide gap with no pixels in it, and 60 sits inside it.
#
# ## 8. Scoring a mask: intersection over union
#
# **Problem 8.** Write `iou(M, T)`: the number of pixels in both masks divided by the number in either. Run it on the 8 by 8
# example of the lecture's figure, built in the next cell: the truth `T` outlines 10 cells, your mask `M` marks 12, and
# 8 are in both.
#
# **Guess first:** is the IoU of that example 0.8, 0.57, or 0.4?

# %%
def iou(M, T):
    """Intersection over union of two bool masks of the same shape."""
    return None  # your code: (M & T).sum() / (M | T).sum()


T = np.zeros((8, 8), bool)
T[3:5, 1:6] = True  # the 10 true paint cells
M = np.zeros((8, 8), bool)
M[3:5, 2:6] = True
M[5, 2:6] = True  # the 12 cells the mask marks

practice("IoU of the 8 by 8 example", iou(M, T), 0.5714, tol=0.0001)

# %% [markdown]
# Now the trap from the lecture. A mask that marks nothing at all gets almost every pixel right. Score the empty yellow
# mask on the noisy frame two ways: the share of pixels it gets right, and its IoU against `labels == YELLOW`.
#
# **Guess first:** which of the two scores is fooled?

# %%
true_yellow = labels == YELLOW
empty = np.zeros_like(true_yellow)

share_right = None  # your code: the fraction of pixels where `empty` equals `true_yellow`
iou_empty = iou(empty, true_yellow)

practice("share right of the empty mask", share_right, 0.9877, tol=0.0001)
practice("IoU of the empty mask", iou_empty, 0.0)

# %% [markdown]
# ![](figures/accuracy_trap.png)
#
# ![](figures/iou_grid.png)

# %% [markdown]
# ## 9. The uint8 wraparound trap: break it, fix it
#
# A teammate wrote a shorter version: "`PALETTE` already holds the colours, so skip the conversion." Run it. It is
# perfect on the clean frame and hopeless on the noisy one.
#
# **Guess first:** why does the clean frame hide the bug? Then score the noisy masks with your `iou` and fix the
# function until both scores reach 1.0.

# %%
def paint_masks_fast(frame):
    """The teammate's version: no conversion."""
    return (np.linalg.norm(frame - PALETTE[YELLOW], axis=-1) < 60,
            np.linalg.norm(frame - PALETTE[WHITE], axis=-1) < 60)


fast_yellow, fast_white = paint_masks_fast(noisy)
practice("IoU of the teammate's yellow mask on the noisy frame", iou(fast_yellow, labels == YELLOW), 0.1354, tol=0.001)
practice("IoU of the teammate's white mask on the noisy frame", iou(fast_white, labels == WHITE), 0.1388, tol=0.001)

# %% [markdown]
# ![](figures/uint8_wrap.png)

# %% [markdown]
# <details><summary>Hint (open after you have a guess)</summary>
#
# `frame` and `PALETTE` are both `uint8`, so `frame - PALETTE[YELLOW]` is computed in `uint8` and wraps around instead
# of going negative. For the noisy pixel at (150, 59), `(234, 199, 42) - (230, 200, 40)` gives `(4, 255, 2)`, a
# distance of 255 instead of 4.6. On the clean frame every paint pixel subtracts to exactly `(0, 0, 0)`, so nothing
# wraps and the bug stays hidden; noise pushes about half of all channels below the paint colour, and each of those
# pixels drops out. The fix is `frame.astype(float) - PALETTE[YELLOW]` (and the same for white). Test on noisy data,
# not only on clean data.
#
# </details>
#
# Now add the one `astype(float)`: copy the function, fix it, and run it on the noisy frame.

# %%
def paint_masks_fixed(frame):
    """The teammate's version, repaired."""
    return None  # your code: the same two masks, with the frame converted to float first


fixed = paint_masks_fixed(noisy)
if fixed is not None:
    practice("IoU of the repaired yellow mask", iou(fixed[0], labels == YELLOW), 1.0)
    practice("IoU of the repaired white mask", iou(fixed[1], labels == WHITE), 1.0)

# %% [markdown]
# Whenever you do arithmetic on image data, convert first.
#
# ## 10. Graded exercise
#
# ### Exercise A: `paint_masks`
#
# Now package the method as the first half of your perception box (lecture: "Paint is a colour threshold"). Mind the
# last section of the lecture, "The uint8 wraparound trap". The checker renders six noisy frames on all three roads,
# at spots it picks at random, and needs IoU of at least 0.85 for both masks on every frame.

# %% tags=["exercise:0.1.2.a"]
def paint_masks(frame):
    """Find the yellow and the white paint in a camera frame.

    Input:   frame, a NumPy uint8 array of shape (180, 320, 3): rows, columns, then red, green and blue.
             It may be noisy: every value nudged up or down by a few steps.
    Output:  a tuple (yellow, white) of two bool arrays, each of shape (180, 320), True where that pixel is
             yellow paint (or white paint).
    Example: yellow, white = paint_masks(clean)  with the clean frame of this lab
             -> yellow.sum() == 709 and white.sum() == 872
    """
    raise NotImplementedError


# %%
check("0.1.2.a", paint_masks)

# %% [markdown]
# Once it passes, look at what your function sees. The next cell draws your masks on the noisy frame.

# %%
try:
    yellow_found, white_found = paint_masks(noisy)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.2))
    axes[0].imshow(noisy)
    axes[0].set_title("noisy frame")
    axes[1].imshow(yellow_found, cmap="gray")
    axes[1].set_title(f"your yellow mask: {yellow_found.sum()} pixels")
    axes[2].imshow(white_found, cmap="gray")
    axes[2].set_title(f"your white mask: {white_found.sum()} pixels")
    plt.show()
except NotImplementedError:
    print("Write paint_masks above first, then run this cell again.")

# %% [markdown]
# ## What's next
#
# ![](figures/next_lesson.png)
#
# You can now find the paint in any frame the car sees. In the next lesson, *Your first perception box*, you turn these two masks into the numbers the
# controller needs: how far the car is from lane centre, and which way it is pointing.
