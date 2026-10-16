"""Reference answer for 1.1.2 The view from the driver's seat.  Labs never import this; the grader never reads it."""
import math

import numpy as np

__all__ = ["world_to_ego"]


def _rotation_matrix(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def world_to_ego(pose, p_world):
    x, y, theta = pose
    return (np.asarray(p_world, dtype=float) - np.array([x, y])) @ _rotation_matrix(theta)  # rows @ R is R-transpose on every row
