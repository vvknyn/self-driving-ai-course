"""Ego ↔ camera extrinsics helpers."""

from __future__ import annotations

import numpy as np


def create_euler_rotation(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    """Rotation from ego frame to camera optical frame.

    Ego ISO 8855: X forward, Y left, Z up.
    Camera optical: X right, Y down, Z forward.

    Args:
        pitch_deg: Rotation about ego Y (nose down positive).
        yaw_deg: Rotation about ego Z.
        roll_deg: Rotation about ego X.

    Returns:
        Rotation matrix R with shape (3, 3). P_cam = R @ P_ego + T.
    """
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)
    r = np.radians(roll_deg)

    R_base = np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    R_pitch = np.array(
        [
            [np.cos(p), 0.0, np.sin(p)],
            [0.0, 1.0, 0.0],
            [-np.sin(p), 0.0, np.cos(p)],
        ],
        dtype=np.float64,
    )
    R_yaw = np.array(
        [
            [np.cos(y), -np.sin(y), 0.0],
            [np.sin(y), np.cos(y), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    R_roll = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(r), -np.sin(r)],
            [0.0, np.sin(r), np.cos(r)],
        ],
        dtype=np.float64,
    )

    R_ego_att = R_yaw @ R_pitch @ R_roll
    return R_base @ R_ego_att.T


def camera_position_to_translation(R: np.ndarray, cam_position_ego: np.ndarray) -> np.ndarray:
    """Compute T = -R @ cam_position with shape (3, 1)."""
    pos = np.asarray(cam_position_ego, dtype=np.float64).reshape(3)
    return (-R @ pos).reshape(3, 1)
