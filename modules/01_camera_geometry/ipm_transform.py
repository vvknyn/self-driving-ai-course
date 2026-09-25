"""
Inverse Perspective Mapping (IPM) & Ground Plane Homography
Transforms camera perspective images into a top-down metric Bird's-Eye View (BEV).
"""

import numpy as np
import cv2
from camera_model import PinholeCamera

class IPMTransformer:
    def __init__(
        self,
        camera: PinholeCamera,
        x_range: tuple = (3.0, 45.0), # forward distance in meters (X_ego)
        y_range: tuple = (-12.0, 12.0), # lateral distance in meters (Y_ego)
        bev_resolution: float = 0.1, # meters per pixel (10cm grid)
    ):
        self.camera = camera
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.res = bev_resolution

        self.bev_width = int(round((self.y_max - self.y_min) / self.res))
        self.bev_height = int(round((self.x_max - self.x_min) / self.res))
        
        # Build coordinate lookup map for fast warping
        self._build_lookup_map()

    def _build_lookup_map(self):
        """
        Creates pixel coordinate mappings (map_x, map_y) from BEV grid to camera image.
        For each pixel in the BEV image (u_bev, v_bev), maps to metric (X_ego, Y_ego, 0),
        then projects into camera (u_cam, v_cam).
        """
        # BEV pixel grid: v_bev=0 is x_max (farthest ahead), v_bev=H is x_min (closest)
        # u_bev=0 is y_max (leftmost), u_bev=W is y_min (rightmost)
        u_indices = np.arange(self.bev_width)
        v_indices = np.arange(self.bev_height)
        
        u_grid, v_grid = np.meshgrid(u_indices, v_indices)
        
        y_ego = self.y_max - u_grid * self.res
        x_ego = self.x_max - v_grid * self.res
        z_ego = np.zeros_like(x_ego)
        
        points_ego = np.stack([x_ego.flatten(), y_ego.flatten(), z_ego.flatten()], axis=-1)
        pixels_cam, valid_mask = self.camera.project_ego_to_pixel(points_ego)
        
        self.map_x = pixels_cam[:, 0].reshape(self.bev_height, self.bev_width).astype(np.float32)
        self.map_y = pixels_cam[:, 1].reshape(self.bev_height, self.bev_width).astype(np.float32)
        self.valid_mask = valid_mask.reshape(self.bev_height, self.bev_width)

    def warp_to_bev(self, image: np.ndarray) -> np.ndarray:
        """
        Warps perspective image (H, W, 3) to metric BEV image (bev_H, bev_W, 3).
        """
        bev_image = cv2.remap(
            image,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        # Mask out points outside camera frustum
        bev_image[~self.valid_mask] = 0
        return bev_image

    def metric_to_bev_pixel(self, x_ego: float, y_ego: float) -> tuple:
        """Converts metric coordinate (x_ego, y_ego) to BEV pixel (u_bev, v_bev)."""
        u_bev = int(round((self.y_max - y_ego) / self.res))
        v_bev = int(round((self.x_max - x_ego) / self.res))
        return u_bev, v_bev

    def bev_pixel_to_metric(self, u_bev: int, v_bev: int) -> tuple:
        """Converts BEV pixel (u_bev, v_bev) to metric coordinate (x_ego, y_ego)."""
        x_ego = self.x_max - v_bev * self.res
        y_ego = self.y_max - u_bev * self.res
        return x_ego, y_ego
