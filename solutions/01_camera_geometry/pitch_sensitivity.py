"""Reference pitch_shift_meters."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_M01 = Path(__file__).resolve().parents[2] / "modules" / "01_camera_geometry"
if str(_M01) not in sys.path:
    sys.path.insert(0, str(_M01))

from camera_model import PinholeCamera
from extrinsics import camera_position_to_translation, create_euler_rotation


def pitch_shift_meters(calib: dict, delta_deg: float, range_m: float) -> float:
    w, h = int(calib["width"]), int(calib["height"])
    fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]
    pos = np.array(calib["cam_position_ego"], dtype=np.float64)

    pitch = calib["pitch_deg"]
    yaw = calib["yaw_deg"]
    roll = calib["roll_deg"]

    R_true = create_euler_rotation(pitch, yaw, roll)
    T_true = camera_position_to_translation(R_true, pos)
    true_cam = PinholeCamera("true", fx, fy, cx, cy, w, h, R_true, T_true)

    pt = np.array([[range_m, 0.0, 0.0]])
    pix, valid = true_cam.project_ego_to_pixel(pt)
    if not valid[0]:
        return 0.0

    R_bias = create_euler_rotation(pitch + delta_deg, yaw, roll)
    T_bias = camera_position_to_translation(R_bias, pos)
    bias_cam = PinholeCamera("bias", fx, fy, cx, cy, w, h, R_bias, T_bias)
    est, v2 = bias_cam.project_pixels_to_ground(pix)
    if not v2[0]:
        return 0.0
    return float(est[0, 0] - range_m)
