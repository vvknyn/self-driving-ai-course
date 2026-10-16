import math

import numpy as np

from zero2fsd.geom import world_to_ego

B = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])  # ego (forward, left, up) -> optical (right, down, forward)


def rot_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def T_ego_cam(height_m, pitch_rad):
    T = np.eye(4)
    T[:3, :3] = rot_y(-pitch_rad) @ B.T  # the sign of the pitch is flipped: the camera tips up
    T[:3, 3] = (0.0, 0.0, height_m)
    return T


camera_mount = (world_to_ego, T_ego_cam)
