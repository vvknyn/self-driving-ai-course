"""Three-camera BEV stitching."""

from __future__ import annotations

import cv2
import numpy as np

from camera_model import PinholeCamera
from ipm import IPMTransformer


def stitch_three_cameras(
    frames: dict[str, np.ndarray],
    cameras: dict[str, PinholeCamera],
    x_range: tuple[float, float] = (4.0, 40.0),
    y_range: tuple[float, float] = (-10.0, 10.0),
    bev_resolution: float = 0.1,
) -> np.ndarray:
    """Blend front/left/right warps into one BEV canvas.

    Args:
        frames: Dict with keys ``front``, ``left``, ``right`` image arrays.
        cameras: Matching calibrated cameras.
        x_range: Forward metric extent in meters.
        y_range: Lateral metric extent in meters.
        bev_resolution: Meters per BEV pixel.

    Returns:
        Stitched BEV image (H, W, 3) uint8.
    """
    canvas: np.ndarray | None = None
    weight_sum: np.ndarray | None = None

    for name in ("front", "left", "right"):
        ipm = IPMTransformer(cameras[name], x_range, y_range, bev_resolution)
        bev = ipm.warp_to_bev(frames[name])
        mask = ipm.valid_mask.astype(np.float32)
        if canvas is None:
            canvas = bev.astype(np.float32) * mask[..., None]
            weight_sum = mask.copy()
        else:
            canvas += bev.astype(np.float32) * mask[..., None]
            weight_sum += mask

    assert canvas is not None and weight_sum is not None
    safe = np.maximum(weight_sum, 1e-6)
    out = (canvas / safe[..., None]).clip(0, 255).astype(np.uint8)
    return out
