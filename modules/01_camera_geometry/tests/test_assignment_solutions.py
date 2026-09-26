"""Tests that reference solutions satisfy the assignment contract."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_M01 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_M01.parents[0]))
from _import_isolation import prepare_module_imports

prepare_module_imports(_M01)

import numpy as np

from calibrate_rig import build_tesla_style_rig
import importlib.util

from config import IPMConfig

_loader_path = Path(__file__).resolve().parent / "_solution_loader.py"
_spec = importlib.util.spec_from_file_location("m01_solution_loader", _loader_path)
assert _spec and _spec.loader
_loader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_loader)
load_solution_module = _loader.load_solution_module


class TestAssignmentSolutions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sol_cam = load_solution_module("sol_m01_camera_model", "camera_model.py")
        cls.sol_ipm = load_solution_module("sol_m01_ipm", "ipm.py")
        cls.sol_pitch = load_solution_module("sol_m01_pitch", "pitch_sensitivity.py")
        cls.data_dir = IPMConfig().data_dir
        with (cls.data_dir / "calib.json").open(encoding="utf-8") as f:
            cls.front_calib = json.load(f)

    def test_build_intrinsic_matrix(self) -> None:
        K = self.sol_cam.build_intrinsic_matrix(200.0, 210.0, 160.0, 90.0)
        self.assertEqual(K.shape, (3, 3))
        self.assertAlmostEqual(K[0, 0], 200.0)
        self.assertAlmostEqual(K[1, 1], 210.0)
        self.assertAlmostEqual(K[0, 2], 160.0)
        self.assertAlmostEqual(K[1, 2], 90.0)

    def test_homography_matches_projection(self) -> None:
        cam = build_tesla_style_rig(320, 180)["front"]
        H = self.sol_ipm.build_ground_homography(cam.K, cam.R, cam.T)
        for pt in [(10.0, 0.0), (20.0, 1.0), (30.0, -1.5)]:
            X, Y = pt
            homog = H @ np.array([X, Y, 1.0])
            u_h, v_h = homog[0] / homog[2], homog[1] / homog[2]
            pix, valid = cam.project_ego_to_pixel(np.array([X, Y, 0.0]))
            self.assertTrue(valid[0])
            self.assertAlmostEqual(u_h, pix[0, 0], delta=1e-3)
            self.assertAlmostEqual(v_h, pix[0, 1], delta=1e-3)

    def test_identity_R_wrong_for_pitched_cam(self) -> None:
        cam = build_tesla_style_rig(320, 180)["front"]
        H_wrong = self.sol_ipm.build_ground_homography(cam.K, np.eye(3), cam.T)
        X, Y = 20.0, 0.0
        homog = H_wrong @ np.array([X, Y, 1.0])
        u_w, v_w = homog[0] / homog[2], homog[1] / homog[2]
        pix, _ = cam.project_ego_to_pixel(np.array([X, Y, 0.0]))
        dist = np.hypot(u_w - pix[0, 0], v_w - pix[0, 1])
        self.assertGreater(dist, 5.0)

    def test_pitch_shift_zero_delta(self) -> None:
        shift = self.sol_pitch.pitch_shift_meters(self.front_calib, 0.0, 25.0)
        self.assertAlmostEqual(shift, 0.0, delta=1e-2)

    def test_pitch_shift_grows_with_range(self) -> None:
        s10 = abs(self.sol_pitch.pitch_shift_meters(self.front_calib, 2.0, 10.0))
        s40 = abs(self.sol_pitch.pitch_shift_meters(self.front_calib, 2.0, 40.0))
        self.assertGreater(s40, s10)

    def test_pitch_shift_sign_matches_independent(self) -> None:
        from camera_model import PinholeCamera
        from extrinsics import camera_position_to_translation, create_euler_rotation

        cal = self.front_calib
        delta = 2.0
        rng = 25.0
        sol = self.sol_pitch.pitch_shift_meters(cal, delta, rng)

        w, h = cal["width"], cal["height"]
        pos = np.array(cal["cam_position_ego"])
        R_t = create_euler_rotation(cal["pitch_deg"], cal["yaw_deg"], cal["roll_deg"])
        T_t = camera_position_to_translation(R_t, pos)
        true_cam = PinholeCamera("t", cal["fx"], cal["fy"], cal["cx"], cal["cy"], w, h, R_t, T_t)
        pix, _ = true_cam.project_ego_to_pixel(np.array([[rng, 0.0, 0.0]]))
        R_b = create_euler_rotation(cal["pitch_deg"] + delta, cal["yaw_deg"], cal["roll_deg"])
        T_b = camera_position_to_translation(R_b, pos)
        bias_cam = PinholeCamera("b", cal["fx"], cal["fy"], cal["cx"], cal["cy"], w, h, R_b, T_b)
        est, _ = bias_cam.project_pixels_to_ground(pix)
        expected = float(est[0, 0] - rng)
        self.assertAlmostEqual(sol, expected, delta=1e-2)


if __name__ == "__main__":
    unittest.main()
