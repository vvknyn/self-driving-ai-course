"""Figures for lecture 0.1.2 Images are arrays.  `python scripts/make_figures.py course/0.1.2-images-are-arrays`."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from zero2fsd.car.provided import PAINT_DISTANCE
from zero2fsd.sim import BicycleState, Camera, make_scenario
from zero2fsd.sim.camera import PALETTE, WHITE, YELLOW
from zero2fsd.sim.scenarios import TARGET_SPEED

VIEW_S, NOISE_SIGMA, SEED = 20.0, 6.0, 0  # the lab renders this same frame
PATCH_ROWS, PATCH_COLS = slice(148, 154), slice(55, 61)  # where the road meets the yellow line
CROP_ROWS = slice(100, 180)  # the part of the frame where the paint is
NAMES = ("sky", "grass", "road", "yellow", "white")


def _frames():
    road = make_scenario("straight").road
    state = BicycleState(*road.ego_pose(VIEW_S), TARGET_SPEED, 0.0)
    camera = Camera()
    clean = camera.render(road, state)
    noisy, labels = camera.render(road, state, noise_sigma=NOISE_SIGMA, seed=SEED, return_labels=True)
    return clean, noisy, labels


def _distance(frame, paint):
    return np.linalg.norm(frame.astype(float) - PALETTE[paint], axis=-1)


def _iou(a, b):
    return np.count_nonzero(a & b) / np.count_nonzero(a | b)


def _show_patch(ax, patch, title):
    """A few pixels drawn large, each labelled with its red, green and blue values."""
    ax.imshow(patch)
    for i in range(patch.shape[0]):
        for j in range(patch.shape[1]):
            ax.text(j, i, "\n".join(str(v) for v in patch[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if patch[i, j].mean() < 150 else "black")
    ax.set_xticks(range(patch.shape[1]), range(PATCH_COLS.start, PATCH_COLS.stop))
    ax.set_yticks(range(patch.shape[0]), range(PATCH_ROWS.start, PATCH_ROWS.stop))
    ax.set(title=title, xlabel="column index", ylabel="row index")


def _frame_grid(path, clean):
    fig = Figure(figsize=(11, 4.2))
    full, zoom = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.5, 1]})
    full.imshow(clean)
    r, c = PATCH_ROWS, PATCH_COLS
    full.add_patch(Rectangle((c.start - 0.5, r.start - 0.5), c.stop - c.start, r.stop - r.start, fill=False, color="red", lw=2))
    full.set(title="frame: 180 rows x 320 columns x 3 colours", xlabel="column index", ylabel="row index (row 0 is the top)")
    _show_patch(zoom, clean[r, c], f"frame[{r.start}:{r.stop}, {c.start}:{c.stop}]: red, green, blue")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def _noisy_patch(path, clean, noisy):
    fig = Figure(figsize=(9, 4))
    for ax, frame, name in zip(fig.subplots(1, 2), (clean, noisy), ("clean", f"with noise, sigma = {NOISE_SIGMA:g}")):
        _show_patch(ax, frame[PATCH_ROWS, PATCH_COLS], f"the same pixels, {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def _colour_distance(path, noisy, labels):
    fig = Figure(figsize=(11, 3.8))
    bins = np.arange(0, 300, 4)
    for ax, paint in zip(fig.subplots(1, 2, sharey=True), (YELLOW, WHITE)):
        d = _distance(noisy, paint)
        for k, name in enumerate(NAMES):
            ax.hist(d[labels == k], bins=bins, color=PALETTE[k] / 255, edgecolor="0.2", linewidth=0.4, label=f"true {name} pixels")
        ax.axvline(PAINT_DISTANCE, color="red", linewidth=2, label=f"threshold {PAINT_DISTANCE:g}")
        ax.set(yscale="log", xlabel=f"distance to {NAMES[paint]} paint colour {tuple(int(v) for v in PALETTE[paint])}",
               title=f"every pixel of the noisy frame: distance to {NAMES[paint]}")
        ax.legend(fontsize=8)
    fig.axes[0].set_ylabel("number of pixels (log scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def _exact_vs_threshold(path, noisy, labels):
    truth = (labels == YELLOW) | (labels == WHITE)
    exact = np.all(noisy == PALETTE[YELLOW], axis=-1) | np.all(noisy == PALETTE[WHITE], axis=-1)
    near = (_distance(noisy, YELLOW) < PAINT_DISTANCE) | (_distance(noisy, WHITE) < PAINT_DISTANCE)
    panels = [
        (noisy, f"noisy frame (sigma = {NOISE_SIGMA:g}), rows {CROP_ROWS.start}-{CROP_ROWS.stop - 1}"),
        (exact, f"frame == paint colour: {exact.sum()} pixels, IoU {_iou(exact, truth):.2f}"),
        (near, f"distance < {PAINT_DISTANCE:g}: {near.sum()} pixels, IoU {_iou(near, truth):.2f}"),
    ]
    fig = Figure(figsize=(12, 1.9))
    for ax, (img, title) in zip(fig.subplots(1, 3), panels):
        ax.imshow(img[CROP_ROWS], cmap="gray", vmin=0, vmax=1) if img.ndim == 2 else ax.imshow(img[CROP_ROWS])
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=100)


def make(out_dir):
    clean, noisy, labels = _frames()
    _frame_grid(out_dir / "frame_grid.png", clean)
    _noisy_patch(out_dir / "noisy_patch.png", clean, noisy)
    _colour_distance(out_dir / "colour_distance.png", noisy, labels)
    _exact_vs_threshold(out_dir / "exact_vs_threshold.png", noisy, labels)
