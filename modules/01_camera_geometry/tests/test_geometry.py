"""
Unit tests for Module 01: Multi-Camera Rig & IPM Geometry
Verifies projection math, matrix invertibility, and ground intersection accuracy.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera_model import PinholeCamera, create_euler_rotation
from ipm_transform import IPMTransformer

class TestCameraGeometry(unittest.TestCase):
    def setUp(self):
        self.img_w, self.img_h = 640, 360
        self.fx = (self.img_w / 2.0) / np.tan(np.radians(60.0))
        self.fy = self.fx
        self.cx, self.cy = self.img_w / 2.0, self.img_h / 2.0
        
        self.R = create_euler_rotation(pitch_deg=4.0, yaw_deg=0.0, roll_deg=0.0)
        self.T = -self.R @ np.array([2.0, 0.0, 1.4])
        self.camera = PinholeCamera("front", self.fx, self.fy, self.cx, self.cy, self.img_w, self.img_h, self.R, self.T)

    def test_euler_rotation_properties(self):
        """Verifies rotation matrices are orthogonal and have determinant +1."""
        self.assertTrue(np.allclose(self.R @ self.R.T, np.eye(3), atol=1e-6))
        self.assertAlmostEqual(np.linalg.det(self.R), 1.0, places=5)

    def test_pinhole_projection_roundtrip(self):
        """Verifies that unprojecting a projected ground point returns the original 3D point."""
        original_points = np.array([
            [15.0, 0.0, 0.0],
            [25.0, 1.8, 0.0],
            [35.0, -1.8, 0.0]
        ])
        
        pixels, valid = self.camera.project_ego_to_pixel(original_points)
        self.assertTrue(np.all(valid), "All test points should be inside image view")
        
        recovered_points, valid_ground = self.camera.project_pixels_to_ground(pixels)
        self.assertTrue(np.all(valid_ground))
        
        np.testing.assert_allclose(recovered_points, original_points, atol=1e-4,
                                   err_msg="Recovered 3D ground points must match original coordinates")

    def test_ipm_dimensions(self):
        """Verifies IPM grid dimensions and coordinate conversion consistency."""
        ipm = IPMTransformer(self.camera, x_range=(5.0, 35.0), y_range=(-10.0, 10.0), bev_resolution=0.1)
        expected_w = int(20.0 / 0.1)
        expected_h = int(30.0 / 0.1)
        
        self.assertEqual(ipm.bev_width, expected_w)
        self.assertEqual(ipm.bev_height, expected_h)
        
        # Test metric <-> pixel conversion
        u, v = ipm.metric_to_bev_pixel(20.0, 0.0)
        x_rec, y_rec = ipm.bev_pixel_to_metric(u, v)
        self.assertAlmostEqual(x_rec, 20.0, delta=0.1)
        self.assertAlmostEqual(y_rec, 0.0, delta=0.1)

if __name__ == "__main__":
    unittest.main()
