"""Reference answer for 1.1.5 The camera's point of view.  Labs never import this; the grader never reads it."""
import math

import numpy as np

from zero2fsd.geom import world_to_ego

__all__ = ["T_ego_cam", "ray_to_ground", "camera_mount"]

CAM_FROM_EGO = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])  # B: ego (forward, left, up) -> optical (right, down, forward)


def rot_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def T_ego_cam(height_m, pitch_rad):
    T = np.eye(4)
    T[:3, :3] = rot_y(pitch_rad) @ CAM_FROM_EGO.T  # turn about y after the optical axes are laid onto the ego axes: the order matters
    T[:3, 3] = (0.0, 0.0, height_m)
    return T


def ray_to_ground(T_ego_cam, ray_cam):
    lens, r = T_ego_cam[:3, 3], T_ego_cam[:3, :3] @ np.asarray(ray_cam, dtype=float)  # a ray is a direction: w = 0, only the turn
    if r[2] >= 0:
        return np.full(3, np.nan)  # level or climbing: it never meets the road
    return lens + (-lens[2] / r[2]) * r


camera_mount = (world_to_ego, T_ego_cam)  # B: the pair the simulator draws a camera picture through
