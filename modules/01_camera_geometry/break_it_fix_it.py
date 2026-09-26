"""Demo: pitch miscalibration grows range error with distance."""

from __future__ import annotations

import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

import numpy as np

from camera_model import PinholeCamera
from extrinsics import camera_position_to_translation, create_euler_rotation


def _range_error(nominal_pitch: float, true_pitch: float, range_m: float, cam_h: float = 1.4) -> float:
    img_w, img_h = 640, 360
    fx = (img_w / 2.0) / np.tan(np.radians(60.0))
    cx, cy = img_w / 2.0, img_h / 2.0

    R_true = create_euler_rotation(true_pitch, 0.0, 0.0)
    T_true = camera_position_to_translation(R_true, np.array([2.0, 0.0, cam_h]))
    true_cam = PinholeCamera("true", fx, fx, cx, cy, img_w, img_h, R_true, T_true)

    target = np.array([[range_m, 0.0, 0.0]])
    pix, _ = true_cam.project_ego_to_pixel(target)

    R_nom = create_euler_rotation(nominal_pitch, 0.0, 0.0)
    T_nom = camera_position_to_translation(R_nom, np.array([2.0, 0.0, cam_h]))
    naive_cam = PinholeCamera("naive", fx, fx, cx, cy, img_w, img_h, R_nom, T_nom)
    est, _ = naive_cam.project_pixels_to_ground(pix)
    return float(est[0, 0] - range_m)


def main() -> None:
    true_pitch = 6.5
    nominal_pitch = 4.0
    range_m = 25.0

    err = _range_error(nominal_pitch, true_pitch, range_m)

    print("=" * 70)
    print("Break-it: static pitch calibration vs braking pitch change")
    print("=" * 70)
    print(f"True pitch: {true_pitch}°  |  Nominal model pitch: {nominal_pitch}°")
    print(f"Ground point: {range_m} m ahead")
    print(f"Range error: {err:+.2f} m")
    print()
    err_10 = abs(_range_error(nominal_pitch, true_pitch, 10.0))
    err_40 = abs(_range_error(nominal_pitch, true_pitch, 40.0))
    print(f"|error| at 10 m: {err_10:.2f} m")
    print(f"|error| at 40 m: {err_40:.2f} m  (grows with range)")
    print()
    print("Principle 3 — Wrong extrinsics → wrong meters; error grows with range.")


if __name__ == "__main__":
    main()
