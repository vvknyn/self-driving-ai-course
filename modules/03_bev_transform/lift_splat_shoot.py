"""
Lift-Splat-Shoot (LSS) Module for Multi-Camera Bird's-Eye View (BEV) Perception
Reference: Philion & Fidler (ECCV 2020)
"Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D"

Tensor Flow:
  1. Input: Multi-Camera Image Features (B, N_cams, C_in, H, W)
  2. Lift: Outer product of Depth Distribution P(D) and Feature C -> (B, N_cams, D, H, W, C_out)
  3. Splat: Unproject 3D optical frustum to vehicle Ego Coordinates (X, Y, Z)
  4. Shoot: Accumulate features into discrete BEV grid (B, C_out, BEV_X, BEV_Y)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class DepthFeatureLift(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_depth_bins: int):
        super().__init__()
        self.num_depth_bins = num_depth_bins
        self.out_channels = out_channels
        
        # Predicts both depth logits (D) and context features (C_out)
        self.conv = nn.Conv2d(in_channels, num_depth_bins + out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, C_in, H, W)
        Returns:
          depth_probs: (B, D, H, W) categorical probability distribution
          features: (B, C_out, H, W) context representation
        """
        out = self.conv(x)
        depth_logits = out[:, :self.num_depth_bins]
        context = out[:, self.num_depth_bins:]
        
        # Categorical depth distribution over bins (MITx Probability: P(D = d_k))
        depth_probs = F.softmax(depth_logits, dim=1)
        return depth_probs, context

class LiftSplatShoot(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 64,
        d_min: float = 2.0,
        d_max: float = 42.0,
        num_depth_bins: int = 20,
        x_bound: tuple = (0.0, 40.0, 0.5), # (min, max, step_size) in meters forward
        y_bound: tuple = (-15.0, 15.0, 0.5), # (min, max, step_size) in meters lateral
        z_bound: tuple = (-1.0, 3.0, 4.0) # single vertical pillar
    ):
        super().__init__()
        self.d_min = d_min
        self.d_max = d_max
        self.num_depth_bins = num_depth_bins
        self.depth_bins = torch.linspace(d_min, d_max, num_depth_bins)
        
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.z_bound = z_bound
        
        self.nx = int(round((x_bound[1] - x_bound[0]) / x_bound[2]))
        self.ny = int(round((y_bound[1] - y_bound[0]) / y_bound[2]))
        self.out_channels = out_channels
        
        self.lift = DepthFeatureLift(in_channels, out_channels, num_depth_bins)
        
        # BEV refinement backbone (conv layers operating directly in top-down metric space)
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def create_camera_frustum(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """
        Generates 3D coordinates for every (u, v, d) bin in the camera coordinate frame.
        Shape: (D, H, W, 3)
        """
        v_coords = torch.linspace(0, h - 1, h, device=device)
        u_coords = torch.linspace(0, w - 1, w, device=device)
        d_coords = self.depth_bins.to(device)
        
        # Grid mesh: (D, H, W)
        D, H, W = len(d_coords), len(v_coords), len(u_coords)
        grid_d, grid_v, grid_u = torch.meshgrid(d_coords, v_coords, u_coords, indexing="ij")
        
        # Stack into points [u * d, v * d, d]
        frustum = torch.stack([grid_u * grid_d, grid_v * grid_d, grid_d], dim=-1)
        return frustum # (D, H, W, 3)

    def unproject_to_ego(self, frustum: torch.Tensor, K: torch.Tensor, R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Unprojects camera frustum points into Vehicle Ego Frame coordinates (X_ego, Y_ego, Z_ego).
        K: (3, 3) intrinsics
        R: (3, 3) rotation ego -> cam
        T: (3, 1) translation ego -> cam
        """
        D, H, W, _ = frustum.shape
        flat_frustum = frustum.reshape(-1, 3).T # (3, N)
        
        # Invert intrinsics: P_cam = K_inv @ [u*d, v*d, d]
        K_inv = torch.linalg.inv(K)
        points_cam = K_inv @ flat_frustum # (3, N)
        
        # Invert extrinsics: P_ego = R_inv @ (P_cam - T)
        R_inv = R.T
        points_ego = R_inv @ (points_cam - T) # (3, N)
        return points_ego.T.reshape(D, H, W, 3)

    def forward(
        self,
        features: torch.Tensor, # (B, N_cams, C_in, H, W)
        intrinsics: torch.Tensor, # (B, N_cams, 3, 3)
        extrinsics_R: torch.Tensor, # (B, N_cams, 3, 3)
        extrinsics_T: torch.Tensor # (B, N_cams, 3, 1)
    ) -> torch.Tensor:
        """
        Returns:
          bev_feature_map: (B, C_out, NY, NX)
        """
        B, N_cams, C_in, H, W = features.shape
        device = features.device
        frustum = self.create_camera_frustum(H, W, device)
        
        # Initialize empty BEV accumulator: (B, C_out, NY, NX)
        bev_accum = torch.zeros((B, self.out_channels, self.ny, self.nx), device=device)
        
        for b in range(B):
            for cam_idx in range(N_cams):
                cam_feat = features[b, cam_idx].unsqueeze(0) # (1, C_in, H, W)
                depth_probs, context = self.lift(cam_feat) # (1, D, H, W), (1, C_out, H, W)
                
                # Outer product: Lift 2D features into 3D ray features
                # Shape: (D, H, W, C_out)
                lifted = (depth_probs.unsqueeze(-1) * context.permute(0, 2, 3, 1).unsqueeze(1)).squeeze(0)
                
                # Unproject geometry to Ego frame (D, H, W, 3)
                K = intrinsics[b, cam_idx]
                R = extrinsics_R[b, cam_idx]
                T = extrinsics_T[b, cam_idx]
                points_ego = self.unproject_to_ego(frustum, K, R, T)
                
                # Voxel pooling / splatting into discrete BEV indices
                x_ego = points_ego[..., 0].flatten()
                y_ego = points_ego[..., 1].flatten()
                feat_flat = lifted.reshape(-1, self.out_channels) # (N, C_out)
                
                # Discrete grid indices
                x_idx = ((x_ego - self.x_bound[0]) / self.x_bound[2]).long()
                y_idx = ((y_ego - self.y_bound[0]) / self.y_bound[2]).long()
                
                # Filter points inside BEV boundaries
                valid = (x_idx >= 0) & (x_idx < self.nx) & (y_idx >= 0) & (y_idx < self.ny)
                
                if valid.sum() > 0:
                    x_valid = x_idx[valid]
                    y_valid = y_idx[valid]
                    feat_valid = feat_flat[valid] # (M, C_out)
                    
                    # Accumulate (Splat) into BEV grid using flat indices
                    flat_indices = y_valid * self.nx + x_valid
                    flat_bev = bev_accum[b].view(self.out_channels, -1)
                    flat_bev.index_add_(1, flat_indices, feat_valid.T)
                    
        # Shoot: Refine pooled BEV map with top-down convolutions
        bev_refined = self.bev_encoder(bev_accum)
        return bev_refined
