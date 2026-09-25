"""
Unit tests for Module 03: Lift-Splat-Shoot BEV Transformation
Verifies depth probability distribution normalization, frustum generation, and BEV pooling dimensions.
"""

import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lift_splat_shoot import LiftSplatShoot, DepthFeatureLift

class TestBEVTransform(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.lss = LiftSplatShoot(
            in_channels=16,
            out_channels=32,
            d_min=2.0,
            d_max=20.0,
            num_depth_bins=10,
            x_bound=(0.0, 20.0, 1.0),
            y_bound=(-10.0, 10.0, 1.0)
        ).to(self.device)

    def test_depth_lift_normalization(self):
        """Verifies depth probabilities sum to 1.0 along depth bin axis (MITx Categorical Dist)."""
        lift_mod = DepthFeatureLift(in_channels=16, out_channels=32, num_depth_bins=10)
        dummy_feat = torch.randn(2, 16, 8, 16)
        probs, context = lift_mod(dummy_feat)
        
        self.assertEqual(probs.shape, torch.Size([2, 10, 8, 16]))
        self.assertEqual(context.shape, torch.Size([2, 32, 8, 16]))
        
        # Check sum to 1.0
        prob_sums = probs.sum(dim=1)
        torch.testing.assert_close(prob_sums, torch.ones_like(prob_sums), atol=1e-5, rtol=1e-5)

    def test_camera_frustum_dimensions(self):
        """Verifies (D, H, W, 3) frustum point grid coordinates."""
        frustum = self.lss.create_camera_frustum(8, 16, self.device)
        self.assertEqual(frustum.shape, torch.Size([10, 8, 16, 3]))
        
        # Min depth should be >= d_min
        self.assertGreaterEqual(frustum[..., 2].min().item(), 2.0)
        # Max depth should be <= d_max
        self.assertLessEqual(frustum[..., 2].max().item(), 20.0)

    def test_full_lss_forward_pass(self):
        """Verifies end-to-end multi-camera forward pass produces correct BEV dimensions."""
        B, N_cams = 1, 2
        feats = torch.randn(B, N_cams, 16, 8, 16)
        
        # Identity matrices for simplicity in unit test
        K = torch.eye(3).view(1, 1, 3, 3).repeat(B, N_cams, 1, 1)
        R = torch.eye(3).view(1, 1, 3, 3).repeat(B, N_cams, 1, 1)
        T = torch.zeros(3, 1).view(1, 1, 3, 1).repeat(B, N_cams, 1, 1)
        
        bev_out = self.lss(feats, K, R, T)
        expected_nx = 20 # 20m / 1.0m
        expected_ny = 20 # 20m / 1.0m
        self.assertEqual(bev_out.shape, torch.Size([B, 32, expected_ny, expected_nx]))
        self.assertFalse(torch.isnan(bev_out).any())

if __name__ == "__main__":
    unittest.main()
