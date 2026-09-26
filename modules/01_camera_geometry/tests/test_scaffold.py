"""Scaffold tests for Module 01."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_M01 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_M01.parents[0]))
from _import_isolation import prepare_module_imports

prepare_module_imports(_M01)

import cv2
import numpy as np

from calibrate_rig import build_tesla_style_rig
from camera_model import PinholeCamera, build_intrinsic_matrix, create_euler_rotation
from config import IPMConfig
from ipm import IPMTransformer, build_ground_homography
from pitch_sensitivity import pitch_shift_meters
from stitch import stitch_three_cameras


class TestScaffold(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[3]
        cls.data_dir = IPMConfig().data_dir

    def test_euler_rotation_properties(self) -> None:
        R = create_euler_rotation(4.0, 0.0, 0.0)
        self.assertTrue(np.allclose(R @ R.T, np.eye(3), atol=1e-6))
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=5)

    def test_pinhole_projection_roundtrip(self) -> None:
        cam = build_tesla_style_rig(640, 360)["front"]
        pts = np.array([[15.0, 0.0, 0.0], [25.0, 1.8, 0.0], [35.0, -1.8, 0.0]])
        pix, valid = cam.project_ego_to_pixel(pts)
        self.assertTrue(np.all(valid))
        recovered, ok = cam.project_pixels_to_ground(pix)
        self.assertTrue(np.all(ok))
        np.testing.assert_allclose(recovered, pts, atol=1e-4)

    def test_ipm_dimensions(self) -> None:
        cam = build_tesla_style_rig(640, 360)["front"]
        ipm = IPMTransformer(cam, x_range=(5.0, 35.0), y_range=(-10.0, 10.0), bev_resolution=0.1)
        self.assertEqual(ipm.bev_width, 200)
        self.assertEqual(ipm.bev_height, 300)
        u, v = ipm.metric_to_bev_pixel(20.0, 0.0)
        x, y = ipm.bev_pixel_to_metric(u, v)
        self.assertAlmostEqual(x, 20.0, delta=0.1)
        self.assertAlmostEqual(y, 0.0, delta=0.1)

    def test_build_tesla_style_rig_shapes(self) -> None:
        cams = build_tesla_style_rig(256, 128)
        for k in ("front", "left", "right"):
            self.assertIn(k, cams)
            self.assertEqual(cams[k].K.shape, (3, 3))
            self.assertEqual(cams[k].R.shape, (3, 3))
            self.assertEqual(cams[k].T.shape, (3, 1))

    def test_points_json_reprojection(self) -> None:
        with (self.data_dir / "calib.json").open(encoding="utf-8") as f:
            calib = json.load(f)
        cams = build_tesla_style_rig(calib["width"], calib["height"])
        with (self.data_dir / "points.json").open(encoding="utf-8") as f:
            payload = json.load(f)
        for entry in payload["points"]:
            if not entry["valid"]:
                continue
            cam = cams[entry["camera"]]
            pix, valid = cam.project_ego_to_pixel(np.array(entry["ego_xyz"]))
            self.assertTrue(valid[0])
            err = np.linalg.norm(pix[0] - np.array(entry["pixel_uv"]))
            self.assertLess(err, 0.5)

    def test_sample_images_not_blank(self) -> None:
        for name in ("front", "left", "right"):
            img = cv2.imread(str(self.data_dir / f"{name}.png"))
            self.assertIsNotNone(img)
            self.assertGreater(float(img.std()), 5.0)

    def test_student_fills_raise(self) -> None:
        with self.assertRaises(NotImplementedError):
            build_intrinsic_matrix(100.0, 100.0, 160.0, 90.0)
        with self.assertRaises(NotImplementedError):
            build_ground_homography(np.eye(3), np.eye(3), np.zeros(3))
        with self.assertRaises(NotImplementedError):
            pitch_shift_meters({"pitch_deg": 4.0}, 0.0, 10.0)

    def test_stitch_nontrivial(self) -> None:
        cams = build_tesla_style_rig(320, 180)
        frames = {
            k: cv2.imread(str(self.data_dir / f"{k}.png")) for k in ("front", "left", "right")
        }
        out = stitch_three_cameras(frames, cams)
        self.assertGreater(int((out > 0).any(axis=-1).sum()), 100)

    def test_example_metrics_json(self) -> None:
        path = self.repo / "artifacts" / "m01" / "example" / "metrics.json"
        data = json.loads(path.read_text())
        for key in ("run_id", "note", "ipm_mean_reprojection_error_px", "pitch_shift_m", "outputs"):
            self.assertIn(key, data)


if __name__ == "__main__":
    unittest.main()
