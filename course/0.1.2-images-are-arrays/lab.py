# %% [markdown]
# # Lab 0.1.2: Images are arrays
#
# You will open a camera frame, find out when exact colour matching works and when it breaks, and write the function
# that finds the paint in a noisy frame: the first half of your perception box.
# Read the lecture first; this lab points back to its sections by title.
#
# Run the cells from top to bottom (Shift+Enter).

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
# ## 1. Open a frame
#
# The next cell puts the car 20 m along the `straight` road and takes two pictures from the same spot: a clean one and
# one with camera noise of $\sigma = 6$ (lecture: "Paint is a colour threshold"). These are the frames the lecture's
# figures show.
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

print("shape:", clean.shape, " dtype:", clean.dtype)
print("clean[0, 0], the top-left pixel:", clean[0, 0])
print("clean[150, 56:62], six pixels of row 150:\n", clean[150, 56:62])
print("the same pixels, green channel only:", clean[150, 55:75, 1])
print("noisy[150, 55:75, 1], green after noise: ", noisy[150, 55:75, 1])
print("true yellow pixels:", (labels == YELLOW).sum(), " true white pixels:", (labels == WHITE).sum())

fig, axes = plt.subplots(1, 3, figsize=(15, 3.2))
axes[0].imshow(clean)
axes[0].set_title("clean")
axes[1].imshow(noisy)
axes[1].set_title("noise sigma = 6")
shown = axes[2].imshow(labels)
fig.colorbar(shown, ax=axes[2], ticks=range(5))
axes[2].set_title("labels: 0 sky, 1 grass, 2 road, 3 yellow, 4 white")
plt.show()

# %% [markdown]
# Match each printed line to the lecture ("A frame is a grid of numbers"). Row 150 crosses the yellow line: road grey up
# to column 58, yellow paint from column 59 to 70, then road again (the green values show all 20). The clean green
# values are exactly 90 and 200; after noise they scatter a few steps either side, from 77 to 102 and from 194 to 206.
# The noisy frame looks the same to your eye. It is not the same to `==`.
#
# ## 2. Predict: how much noise does exact matching survive?
#
# At $\sigma = 6$, `np.all(frame == PALETTE[YELLOW], axis=-1)` found none of the 709 yellow pixels (lecture: "Why exact
# equality fails"). Now turn the noise down to $\sigma = 1$: most nudges are now smaller than one step of brightness.
#
# **Before you run the next cell**, double-click this one and write your prediction. Use the lecture's reasoning: a
# channel survives when its nudge rounds to 0, and all three channels must survive.
#
# **My prediction:** at $\sigma = 1$, exact equality finds about ______ of the 709 yellow pixels (about 700, 350, 40,
# or 0?), because ______.

# %%
yellow_truth = labels == YELLOW
for sigma in (0.0, 0.3, 1.0, 3.0, 6.0):
    frame = camera.render(road, state, noise_sigma=sigma, seed=0)
    found = np.all(frame == PALETTE[YELLOW], axis=-1).sum()
    print(f"sigma = {sigma:3.1f}: exact equality finds {found:3d} of {yellow_truth.sum()} yellow pixels")

# %% [markdown]
# At $\sigma = 1$ a single nudge rounds to 0 with probability 0.38, so all three do with probability
# $0.38^3 = 0.056$, and $0.056 \times 709 \approx 40$. The frame found 36. Noise too small to see still hides 95% of
# the paint from `==`. Measured data is never exactly on its ideal value.
#
# ## 3. Worked example: from a colour to a mask
#
# Here is the whole method from "Paint is a colour threshold", one step per line, for yellow on the noisy frame. Run it
# and read the shapes: they are the point.

# %%
yellow = PALETTE[YELLOW].astype(float)  # (3,): the three numbers of yellow paint, as floats
pixels = noisy.astype(float)  # (180, 320, 3): floats can go negative, uint8 cannot
difference = pixels - yellow  # (180, 320, 3): broadcasting subtracts yellow from every pixel
distance = np.linalg.norm(difference, axis=-1)  # (180, 320): one distance per pixel, across the colour axis
yellow_mask = distance < 60  # (180, 320), bool: True where the pixel is close to yellow

print("difference:", difference.shape, " distance:", distance.shape, " mask:", yellow_mask.shape, yellow_mask.dtype)
print("pixel (150, 60): colour", noisy[150, 60], " difference", difference[150, 60], f" distance {distance[150, 60]:.2f}")
print("pixels marked:", yellow_mask.sum(), " true yellow pixels:", yellow_truth.sum())

both = (yellow_mask & yellow_truth).sum()  # the intersection: marked AND truly yellow
either = (yellow_mask | yellow_truth).sum()  # the union: marked OR truly yellow
print(f"IoU = {both} / {either} = {both / either:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
shown = axes[0].imshow(distance)
fig.colorbar(shown, ax=axes[0], label="distance to yellow")
axes[0].set_title("distance to yellow: dark is close")
axes[1].imshow(yellow_mask, cmap="gray")
axes[1].set_title("yellow_mask = distance < 60")
plt.show()

# %% [markdown]
# ## 4. Practice (not graded)
#
# Each problem gives you less than the one before. Replace each `None` with your code and rerun the cell; `practice`
# prints a green tick or tells you what it expected.
#
# **Problem 1.** The distance from the noisy pixel at row 150, column 60 to yellow, by the formula in "Colour is a point
# in space". Both colours are already floats. Write the formula out yourself, with `np.sqrt` and `.sum()`, instead of
# calling `np.linalg.norm`.

# %%
p = noisy[150, 60].astype(float)
c = PALETTE[YELLOW].astype(float)
distance_p = None  # your code: square the differences, add them up, take the root

practice("distance from pixel (150, 60) to yellow", distance_p, 8.246, tol=0.001)

# %% [markdown]
# **Problem 2.** How many pixels of `noisy` are within 60 of white paint? One distance array, one comparison, one
# count.

# %%
white_count = None  # your code

practice("pixels within 60 of white", white_count, (labels == WHITE).sum())

# %% [markdown]
# **Problem 3.** Could the same threshold find the road? Build a mask of the pixels of `noisy` within 60 of road grey,
# `PALETTE[ROAD]`, and score it with IoU against the true road, `labels == ROAD`. Then compute the distance between
# road grey and grass green, `PALETTE[GRASS]`, and use it to explain the score.

# %%
road_iou = None  # your code
road_to_grass = None  # your code

practice("IoU of the road mask at 60", road_iou, 0.708, tol=0.001)
practice("distance from road grey to grass green", road_to_grass, 56.79, tol=0.01)

# %% [markdown]
# <details><summary>What the road score means (open after Problem 3)</summary>
#
# Road grey and grass green are only 56.8 apart, less than the threshold, so a distance of 60 around road grey reaches
# past grass green itself: the mask marks thousands of grass pixels. A threshold only works inside the gap between a
# colour and its nearest neighbour ("Paint is a colour threshold"). For paint the gaps are 175.8 and 116.4, so 60 is
# safe. For road you would need a threshold of about 30.
#
# </details>
#
# ## 5. Graded exercise
#
# ### 0.1.2.a `paint_masks`
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
# ## 6. Break it, fix it
#
# A teammate wrote a shorter version: "`PALETTE` already holds the colours, so skip the conversion." Run it. It is
# perfect on the clean frame and hopeless on the noisy one. Before you open the hint: why does the clean frame hide the
# bug? Then fix the function and rerun the cell until both noisy scores reach 1.00.

# %%
def paint_masks_fast(frame):
    """The teammate's version: no conversion."""
    return (np.linalg.norm(frame - PALETTE[YELLOW], axis=-1) < 60,
            np.linalg.norm(frame - PALETTE[WHITE], axis=-1) < 60)


clean_labels = camera.render(road, state, return_labels=True)[1]
for name, frame, truth in (("clean", clean, clean_labels), ("noisy", noisy, labels)):
    scores = []
    for mask, paint in zip(paint_masks_fast(frame), (YELLOW, WHITE)):
        true_paint = truth == paint
        scores.append((mask & true_paint).sum() / (mask | true_paint).sum())
    print(f"{name} frame: yellow IoU {scores[0]:.2f}, white IoU {scores[1]:.2f}")

# %% [markdown]
# <details><summary>Hint (open after you have a guess)</summary>
#
# `frame` and `PALETTE` are both `uint8`, so `frame - PALETTE[YELLOW]` is computed in `uint8` and wraps around instead
# of going negative. For the noisy pixel at (150, 60), `(226, 196, 34) - (230, 200, 40)` gives `(252, 252, 250)`, a
# distance of 435 instead of 8.25. On the clean frame every paint pixel subtracts to exactly `(0, 0, 0)`, so nothing
# wraps and the bug stays hidden; noise pushes about half of all channels below the paint colour, and each of those
# pixels drops out. The fix is `frame.astype(float) - PALETTE[YELLOW]` (and the same for white). Test on noisy data,
# not only on clean data.
#
# </details>
#
# ## What's next
#
# You can now find the paint in any frame the car sees. In unit 0.1.3 you turn these two masks into the numbers the
# controller needs: how far the car is from lane centre, and which way it is pointing.
