"""Pinhole camera model — projection and ground unprojection."""

from __future__ import annotations

import numpy as np

from extrinsics import create_euler_rotation  # re-export for backward compatibility

__all__ = ["PinholeCamera", "create_euler_rotation", "build_intrinsic_matrix"]


def build_intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Build pinhole intrinsic matrix K — student fill.

    K = [[fx, 0, cx],
         [0, fy, cy],
         [0,  0,  1]]

    Args:
        fx: Focal length in pixels (horizontal).
        fy: Focal length in pixels (vertical).
        cx: Principal point x.
        cy: Principal point y.

    Returns:
        Intrinsic matrix with shape (3, 3).
    """
    raise NotImplementedError("Derive K from fx, fy, cx, cy")


class PinholeCamera:
    """Pinhole camera with optional radial distortion."""

    def __init__(
        self,
        name: str,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        r_ego_to_cam: np.ndarray,
        t_ego_to_cam: np.ndarray,
        k1: float = 0.0,
        k2: float = 0.0,
    ) -> None:
        self.name = name
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2

        # Scaffold K inline — assignment is to re-derive in build_intrinsic_matrix.
        self.K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.K_inv = np.linalg.inv(self.K)

        self.R = np.array(r_ego_to_cam, dtype=np.float64)
        self.T = np.array(t_ego_to_cam, dtype=np.float64).reshape(3, 1)
        self.R_inv = self.R.T
        self.T_inv = -self.R_inv @ self.T

    def project_ego_to_pixel(self, points_ego: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project ego points (N, 3) to pixels (N, 2) and validity mask."""
        points_ego = np.asarray(points_ego, dtype=np.float64)
        if points_ego.ndim == 1:
            points_ego = points_ego.reshape(1, 3)

        points_cam = (self.R @ points_ego.T + self.T).T
        z = points_cam[:, 2]
        valid_z = z > 0.1
        safe_z = np.where(valid_z, z, 1.0)
        u = (self.fx * points_cam[:, 0] / safe_z) + self.cx
        v = (self.fy * points_cam[:, 1] / safe_z) + self.cy

        if abs(self.k1) > 1e-6 or abs(self.k2) > 1e-6:
            xn = (u - self.cx) / self.fx
            yn = (v - self.cy) / self.fy
            r2 = xn**2 + yn**2
            radial = 1.0 + self.k1 * r2 + self.k2 * (r2**2)
            u = self.fx * (xn * radial) + self.cx
            v = self.fy * (yn * radial) + self.cy

        pixels = np.stack([u, v], axis=-1)
        in_bounds = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        return pixels, valid_z & in_bounds

    def pixel_to_ray_ego(self, pixels: np.ndarray) -> np.ndarray:
        """Back-project pixels (N, 2) to unit ray directions in ego frame (N, 3)."""
        pixels = np.asarray(pixels, dtype=np.float64)
        if pixels.ndim == 1:
            pixels = pixels.reshape(1, 2)
        n = pixels.shape[0]
        homog = np.hstack([pixels, np.ones((n, 1))])
        rays_cam = (self.K_inv @ homog.T).T
        rays_cam /= np.maximum(np.linalg.norm(rays_cam, axis=1, keepdims=True), 1e-9)
        return (self.R_inv @ rays_cam.T).T

    def project_pixels_to_ground(
        self, pixels: np.ndarray, ground_z: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Intersect pixel rays with ground plane Z = ground_z."""
        rays_ego = self.pixel_to_ray_ego(pixels)
        cam_origin = self.T_inv.flatten()
        ray_dz = rays_ego[:, 2]
        valid_t = (ray_dz < -1e-4) & (cam_origin[2] > ground_z)
        safe_dz = np.where(valid_t, ray_dz, -1.0)
        t = (ground_z - cam_origin[2]) / safe_dz
        valid = valid_t & (t > 0)
        points = cam_origin[None, :] + t[:, None] * rays_ego
        return points, valid
