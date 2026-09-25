"""
Pinhole Camera Model & Multi-View Coordinate Transforms
Convention:
  Ego Vehicle Frame (ISO 8855):
    X: Forward (+X ahead of vehicle)
    Y: Left (+Y to the driver's left)
    Z: Up (+Z pointing up from ground)
    Origin: Center of vehicle rear axle at ground level.

  Camera Optical Frame:
    X: Right (+X to the right on sensor)
    Y: Down (+Y pointing down on sensor)
    Z: Forward (Optical optical axis pointing out into scene)
"""

import numpy as np
import torch
from typing import Tuple, Optional

class PinholeCamera:
    def __init__(
        self,
        name: str,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        r_ego_to_cam: np.ndarray, # 3x3 rotation matrix
        t_ego_to_cam: np.ndarray, # 3x1 translation vector
        k1: float = 0.0,
        k2: float = 0.0,
    ):
        self.name = name
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2

        # Intrinsic Matrix K (3x3)
        self.K = np.array([
            [fx,  0.0, cx],
            [0.0, fy,  cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)

        # Extrinsic transforms: Ego -> Camera
        self.R = np.array(r_ego_to_cam, dtype=np.float64)
        self.T = np.array(t_ego_to_cam, dtype=np.float64).reshape(3, 1)
        
        # Camera -> Ego inverse transforms
        self.R_inv = self.R.T
        self.T_inv = -self.R_inv @ self.T

    def project_ego_to_pixel(self, points_ego: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects 3D points in Ego Frame (N, 3) to 2D image coordinates (N, 2).
        Returns:
            pixels: (N, 2) in [u, v]
            valid_mask: (N,) boolean indicating if point is in front of camera and inside image bounds.
        """
        points_ego = np.asarray(points_ego, dtype=np.float64)
        if points_ego.ndim == 1:
            points_ego = points_ego.reshape(1, 3)
            
        N = points_ego.shape[0]
        # P_cam = R * P_ego + T
        points_cam = (self.R @ points_ego.T + self.T).T # (N, 3)
        
        z = points_cam[:, 2]
        valid_z = z > 0.1 # must be in front of lens
        
        # Avoid division by zero
        safe_z = np.where(valid_z, z, 1.0)
        u = (self.fx * points_cam[:, 0] / safe_z) + self.cx
        v = (self.fy * points_cam[:, 1] / safe_z) + self.cy
        
        # Optional radial distortion
        if abs(self.k1) > 1e-6 or abs(self.k2) > 1e-6:
            xn = (u - self.cx) / self.fx
            yn = (v - self.cy) / self.fy
            r2 = xn**2 + yn**2
            radial = 1.0 + self.k1 * r2 + self.k2 * (r2**2)
            u = self.fx * (xn * radial) + self.cx
            v = self.fy * (yn * radial) + self.cy

        pixels = np.stack([u, v], axis=-1)
        in_bounds = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        valid = valid_z & in_bounds
        return pixels, valid

    def pixel_to_ray_ego(self, pixels: np.ndarray) -> np.ndarray:
        """
        Backprojects 2D pixels (N, 2) into unit 3D direction vectors in the Ego Frame (N, 3).
        """
        pixels = np.asarray(pixels, dtype=np.float64)
        if pixels.ndim == 1:
            pixels = pixels.reshape(1, 2)
            
        N = pixels.shape[0]
        homog_pixels = np.hstack([pixels, np.ones((N, 1))]) # (N, 3)
        
        # Rays in Camera Frame: K_inv * p
        rays_cam = (self.K_inv @ homog_pixels.T).T # (N, 3)
        # Normalize in camera frame
        norm_cam = np.linalg.norm(rays_cam, axis=1, keepdims=True)
        rays_cam = rays_cam / np.maximum(norm_cam, 1e-9)
        
        # Transform direction vector to Ego frame: R_inv * rays_cam
        rays_ego = (self.R_inv @ rays_cam.T).T
        return rays_ego

    def project_pixels_to_ground(self, pixels: np.ndarray, ground_z: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Intersects camera rays from pixels (N, 2) with the flat ground plane Z_ego = ground_z.
        Returns:
            points_ground: (N, 3) 3D coordinates in Ego Frame.
            valid_mask: (N,) boolean indicating forward intersection with ground.
        """
        rays_ego = self.pixel_to_ray_ego(pixels)
        cam_origin_ego = self.T_inv.flatten() # (3,)
        
        # Ray equation: P(t) = CamOrigin + t * RayDir
        # Ground plane constraint: P_z(t) = ground_z
        # CamOrigin_z + t * RayDir_z = ground_z  ==> t = (ground_z - CamOrigin_z) / RayDir_z
        ray_dz = rays_ego[:, 2]
        # Valid if ray is pointing downward towards ground (ray_dz < -1e-4) and camera is above ground
        valid_t = (ray_dz < -1e-4) & (cam_origin_ego[2] > ground_z)
        
        safe_dz = np.where(valid_t, ray_dz, -1.0)
        t = (ground_z - cam_origin_ego[2]) / safe_dz
        valid = valid_t & (t > 0)
        
        points_ground = cam_origin_ego[None, :] + t[:, None] * rays_ego
        return points_ground, valid

def create_euler_rotation(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    """
    Constructs a 3x3 rotation matrix transforming from Ego Frame to Camera Optical Frame.
    Angles in degrees:
      pitch: rotation about vehicle Y (nose down/up)
      yaw: rotation about vehicle Z (facing forward/left/right)
      roll: rotation about vehicle X (body tilt)
    """
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)
    r = np.radians(roll_deg)
    
    # Ego to standard forward orientation
    # Base transformation from Ego (X forward, Y left, Z up) to Camera (X right, Y down, Z forward):
    # X_cam = -Y_ego, Y_cam = -Z_ego, Z_cam = X_ego
    R_base = np.array([
        [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0, -1.0],
        [ 1.0,  0.0,  0.0]
    ])
    
    # Vehicle attitude rotations in Ego frame
    # Ry(pitch)
    R_pitch = np.array([
        [ np.cos(p), 0.0, np.sin(p)],
        [ 0.0,       1.0, 0.0],
        [-np.sin(p), 0.0, np.cos(p)]
    ])
    # Rz(yaw)
    R_yaw = np.array([
        [np.cos(y), -np.sin(y), 0.0],
        [np.sin(y),  np.cos(y), 0.0],
        [0.0,        0.0,       1.0]
    ])
    # Rx(roll)
    R_roll = np.array([
        [1.0, 0.0,        0.0],
        [0.0, np.cos(r), -np.sin(r)],
        [0.0, np.sin(r),  np.cos(r)]
    ])
    
    R_ego_att = R_yaw @ R_pitch @ R_roll
    # Combined rotation: first apply attitude in ego, then convert to camera optical frame
    R_total = R_base @ R_ego_att.T
    return R_total
