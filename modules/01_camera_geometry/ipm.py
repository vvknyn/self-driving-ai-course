"""Inverse perspective mapping on the ground plane Z=0."""

from __future__ import annotations

import cv2
import numpy as np

from camera_model import PinholeCamera


def build_ground_homography(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Ground-plane homography — student fill.

    On Z=0 the third column of the extrinsic block drops out:
    H = K @ [r1 r2 t], mapping homogeneous ground (X, Y, 1) to pixels.

    Args:
        K: Intrinsic matrix (3, 3).
        R: Rotation ego→camera (3, 3).
        t: Translation (3,) or (3, 1) in the same convention as PinholeCamera.T.

    Returns:
        Homography H with shape (3, 3).
    """
    raise NotImplementedError("Implement H = K @ [r1 r2 t] for Z=0 ground plane")


class IPMTransformer:
    """Warp perspective images to a metric BEV grid via project_ego_to_pixel."""

    def __init__(
        self,
        camera: PinholeCamera,
        x_range: tuple[float, float] = (3.0, 45.0),
        y_range: tuple[float, float] = (-12.0, 12.0),
        bev_resolution: float = 0.1,
    ) -> None:
        self.camera = camera
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.res = bev_resolution
        self.bev_width = int(round((self.y_max - self.y_min) / self.res))
        self.bev_height = int(round((self.x_max - self.x_min) / self.res))
        self._build_lookup_map()

    def _build_lookup_map(self) -> None:
        u_idx = np.arange(self.bev_width)
        v_idx = np.arange(self.bev_height)
        u_grid, v_grid = np.meshgrid(u_idx, v_idx)
        y_ego = self.y_max - u_grid * self.res
        x_ego = self.x_max - v_grid * self.res
        z_ego = np.zeros_like(x_ego)
        points = np.stack([x_ego.flatten(), y_ego.flatten(), z_ego.flatten()], axis=-1)
        pixels, valid = self.camera.project_ego_to_pixel(points)
        self.map_x = pixels[:, 0].reshape(self.bev_height, self.bev_width).astype(np.float32)
        self.map_y = pixels[:, 1].reshape(self.bev_height, self.bev_width).astype(np.float32)
        self.valid_mask = valid.reshape(self.bev_height, self.bev_width)

    def warp_to_bev(self, image: np.ndarray) -> np.ndarray:
        """Warp image (H, W, 3) to BEV (bev_H, bev_W, 3)."""
        bev = cv2.remap(
            image,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        bev[~self.valid_mask] = 0
        return bev

    def metric_to_bev_pixel(self, x_ego: float, y_ego: float) -> tuple[int, int]:
        u_bev = int(round((self.y_max - y_ego) / self.res))
        v_bev = int(round((self.x_max - x_ego) / self.res))
        return u_bev, v_bev

    def bev_pixel_to_metric(self, u_bev: int, v_bev: int) -> tuple[float, float]:
        x_ego = self.x_max - v_bev * self.res
        y_ego = self.y_max - u_bev * self.res
        return x_ego, y_ego
