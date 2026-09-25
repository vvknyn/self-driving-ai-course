"""
Unit tests for Module 04: 3D Occupancy Networks & Temporal Dynamics
Verifies 3D voxel indexing, ConvGRU recurrent state flow, Bernoulli probability bounds, and velocity tensors.
"""

import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from voxel_grid import VoxelGridConfig, generate_synthetic_3d_occupancy
from temporal_fusion import OccupancyNetwork, ConvGRUCell

class TestOccupancyNetwork(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.cfg = VoxelGridConfig(x_range=(0.0, 16.0), y_range=(-8.0, 8.0), z_range=(-1.0, 3.0), voxel_size=1.0)
        self.occ_net = OccupancyNetwork(in_channels=16, hidden_channels=16, nz=self.cfg.nz).to(self.device)

    def test_voxel_config_dimensions(self):
        """Verifies voxel counts match metric ranges."""
        self.assertEqual(self.cfg.nx, 16) # 16m / 1m
        self.assertEqual(self.cfg.ny, 16) # 16m / 1m
        self.assertEqual(self.cfg.nz, 4)  # 4m / 1m
        
        ix, iy, iz = self.cfg.point_to_voxel_index(5.0, 0.0, 1.0)
        self.assertEqual(ix, 5)
        self.assertEqual(iy, 8)
        self.assertEqual(iz, 2)

    def test_synthetic_occupancy_shapes(self):
        """Verifies synthetic 3D generator outputs valid 5D tensors."""
        data = generate_synthetic_3d_occupancy(self.cfg, lead_x=8.0, lead_y=0.0)
        occ = data["occupancy"]
        vel = data["velocity"]
        
        self.assertEqual(occ.shape, torch.Size([1, 1, 16, 16, 4]))
        self.assertEqual(vel.shape, torch.Size([1, 3, 16, 16, 4]))
        self.assertTrue((occ == 0.0).logical_or(occ == 1.0).all())

    def test_temporal_gru_state_flow(self):
        """Verifies recurrent state preservation and output tensor dimensions."""
        B, C, H, W = 1, 16, 16, 16
        x1 = torch.randn(B, C, H, W)
        x2 = torch.randn(B, C, H, W)
        
        # Step 1
        occ1, vel1, h1 = self.occ_net(x1, h_prev=None)
        self.assertEqual(occ1.shape, torch.Size([B, 1, 16, 16, 4]))
        self.assertEqual(vel1.shape, torch.Size([B, 3, 16, 16, 4]))
        self.assertEqual(h1.shape, torch.Size([B, 16, 16, 16]))
        
        # Step 2 with memory
        occ2, vel2, h2 = self.occ_net(x2, h_prev=h1)
        self.assertEqual(occ2.shape, torch.Size([B, 1, 16, 16, 4]))
        
        # Bernoulli probabilities must be strictly in [0, 1]
        self.assertTrue((occ2 >= 0.0).all() and (occ2 <= 1.0).all())

if __name__ == "__main__":
    unittest.main()
