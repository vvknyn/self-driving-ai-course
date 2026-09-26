"""Optional checks when student fills are implemented."""

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
from camera_model import build_intrinsic_matrix
from config import IPMConfig
from ipm import build_ground_homography
from pitch_sensitivity import pitch_shift_meters


class TestStudentFills(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (IPMConfig().data_dir / "calib.json").open(encoding="utf-8") as f:
            cls.front_calib = json.load(f)

    def test_intrinsic_if_implemented(self) -> None:
        try:
            K = build_intrinsic_matrix(100.0, 100.0, 50.0, 50.0)
        except NotImplementedError:
            self.skipTest("build_intrinsic_matrix not implemented")
        self.assertAlmostEqual(K[0, 0], 100.0)

    def test_homography_if_implemented(self) -> None:
        cam = build_tesla_style_rig(320, 180)["front"]
        try:
            H = build_ground_homography(cam.K, cam.R, cam.T)
        except NotImplementedError:
            self.skipTest("build_ground_homography not implemented")
        homog = H @ np.array([10.0, 0.0, 1.0])
        pix, _ = cam.project_ego_to_pixel(np.array([10.0, 0.0, 0.0]))
        self.assertAlmostEqual(homog[0] / homog[2], pix[0, 0], delta=1e-3)

    def test_pitch_if_implemented(self) -> None:
        try:
            s0 = pitch_shift_meters(self.front_calib, 0.0, 20.0)
        except NotImplementedError:
            self.skipTest("pitch_shift_meters not implemented")
        self.assertAlmostEqual(s0, 0.0, delta=1e-2)


if __name__ == "__main__":
    unittest.main()
