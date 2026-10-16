"""Reference answers for 1.1.1 Positions and headings.  Labs never import this; the grader never reads it."""
import numpy as np

__all__ = ["direction_angle", "forward_left"]


def direction_angle(v):
    v = np.asarray(v, dtype=float)
    angle = np.arctan2(v[..., 1], v[..., 0])
    return np.where((v[..., 0] == 0) & (v[..., 1] == 0), np.nan, angle)[()]


def forward_left(theta, v):
    theta, v = np.asarray(theta, dtype=float), np.asarray(v, dtype=float)
    h = np.stack([np.cos(theta), np.sin(theta)], axis=-1)  # the heading arrow
    return np.stack([(v * h).sum(axis=-1), h[..., 0] * v[..., 1] - h[..., 1] * v[..., 0]], axis=-1)
