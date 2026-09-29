"""0.1.2 Images are arrays: finding paint pixels."""
from __future__ import annotations

import numpy as np

from ...sim import Camera
from ...sim.camera import WHITE, YELLOW
from .._poses import sample_poses
from .._report import Failure, call, exercise

NOISE_SIGMA, IOU_MIN, N_FRAMES = 6.0, 0.85, 6


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else 1.0  # both empty: nothing to find, nothing found


def _as_masks(got, shape, given):
    """The learner's answer as two bool masks of the frame's shape, or a Failure saying what is off."""
    masks = list(got) if isinstance(got, tuple | list) else None
    if masks is None or len(masks) != 2:
        raise Failure(f"returned {type(got).__name__}, expected (yellow, white)", given=given, got=got)
    for name, mask in zip(("yellow", "white"), masks):
        if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != shape:
            seen = f"{mask.dtype} array of shape {mask.shape}" if isinstance(mask, np.ndarray) else type(mask).__name__
            raise Failure(f"the {name} mask must be a bool array of shape {shape}", given=given, got=seen)
    return masks


@exercise(
    "0.1.2.a",
    checked=f"yellow and white masks on {N_FRAMES} noisy (sigma={NOISE_SIGMA:g}) frames against the renderer's true paint, IoU >= {IOU_MIN}",
    hint='lecture 0.1.2, section "Paint is a colour threshold"',
)
def _paint_masks(fn, show):
    camera, rng = Camera(), np.random.default_rng(201)
    poses = sample_poses(rng, N_FRAMES, ("straight", "gentle", "curvy"), straight_only=False)
    for k, pose in enumerate(poses):
        frame, labels = camera.render(pose.road, pose.state, noise_sigma=NOISE_SIGMA, seed=k, return_labels=True)
        given = f"a {frame.shape[1]}x{frame.shape[0]} frame rendered at {pose}"
        masks = _as_masks(call(fn, frame.copy(), given=given), frame.shape[:2], given)
        for name, mask, paint in zip(("yellow", "white"), masks, (YELLOW, WHITE)):
            truth = labels == paint
            iou = _iou(mask, truth)
            if iou < IOU_MIN:
                raise Failure(
                    f"the {name} mask overlaps the true {name} paint too little (IoU {iou:.2f}, need {IOU_MIN})", given=given,
                    expected=f"{np.count_nonzero(truth)} {name} pixels",
                    got=f"{np.count_nonzero(mask)} pixels marked, {np.count_nonzero(mask & truth)} of them on {name} paint",
                )
    return f"both masks reach IoU >= {IOU_MIN} on all {N_FRAMES} noisy frames"
