"""Reference build_ground_homography."""

from __future__ import annotations

import numpy as np


def build_ground_homography(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    t_vec = np.asarray(t, dtype=np.float64).reshape(3, 1)
    Rt = np.hstack([R[:, 0:2], t_vec])
    return K @ Rt
