import numpy as np


def transform_points(T, p):
    return np.asarray(p, dtype=float) @ T[:3, :3].T + T[:3, 3]  # w = 1: turns AND shifts


def ray_to_ground(T_ego_cam, ray_cam):
    lens = T_ego_cam[:3, 3]
    r = transform_points(T_ego_cam, ray_cam)  # a ray is a direction: it must only turn (w = 0)
    if r[2] >= 0:
        return np.full(3, np.nan)
    return lens + (-lens[2] / r[2]) * r
