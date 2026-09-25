"""
Spatiotemporal Recurrent Fusion & Occupancy Network
Fuses sequential BEV/3D feature frames using a 2D/3D ConvGRU to maintain memory
over time, handling occlusions and estimating dynamic object velocities.

Tensor Shapes:
  Input features X_t: (B, C_in, NX, NY)
  Hidden state H_t:   (B, C_hidden, NX, NY)
  Occupancy Output:   (B, NZ, NX, NY) -> unpooled to 3D voxels
  Velocity Output:    (B, 3, NX, NY)  -> (vx, vy, vz)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGRUCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        # Gate convolutions (reset and update gates)
        self.conv_gates = nn.Conv2d(
            in_channels + hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1
        )
        # Candidate state convolution
        self.conv_candidate = nn.Conv2d(
            in_channels + hidden_channels, hidden_channels, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor = None) -> torch.Tensor:
        B, _, H, W = x.shape
        if h_prev is None:
            h_prev = torch.zeros(B, self.hidden_channels, H, W, device=x.device)
            
        combined = torch.cat([x, h_prev], dim=1)
        gates = torch.sigmoid(self.conv_gates(combined))
        reset_gate, update_gate = torch.chunk(gates, 2, dim=1)
        
        combined_candidate = torch.cat([x, reset_gate * h_prev], dim=1)
        candidate_state = torch.tanh(self.conv_candidate(combined_candidate))
        
        h_next = (1.0 - update_gate) * h_prev + update_gate * candidate_state
        return h_next

class OccupancyNetwork(nn.Module):
    def __init__(self, in_channels: int = 64, hidden_channels: int = 64, nz: int = 8):
        super().__init__()
        self.nz = nz
        self.temporal_gru = ConvGRUCell(in_channels, hidden_channels)
        
        # 3D Occupancy head (predicts occupancy logits for each vertical Z voxel layer)
        self.occ_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, nz, kernel_size=1) # (B, nz, NX, NY)
        )
        
        # 3D Velocity flow head (predicts velocity vectors: vx, vy, vz)
        self.vel_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1) # (B, 3, NX, NY)
        )

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor = None):
        """
        x: (B, C_in, NX, NY)
        h_prev: (B, C_hidden, NX, NY)
        Returns:
          occ_probs: (B, 1, NX, NY, NZ) Bernoulli occupancy probabilities
          velocities: (B, 3, NX, NY, NZ) 3D velocity vectors
          h_next: (B, C_hidden, NX, NY) updated memory state
        """
        h_next = self.temporal_gru(x, h_prev)
        
        # Predict 2D maps with NZ vertical channels: (B, NZ, NX, NY)
        occ_logits = self.occ_head(h_next)
        vel_2d = self.vel_head(h_next) # (B, 3, NX, NY)
        
        # Bernoulli occupancy probability: sigma(logits) in [0, 1]
        occ_probs_2d = torch.sigmoid(occ_logits)
        
        # Reshape to true 5D 3D Voxel Tensor: (B, 1, NX, NY, NZ)
        occ_probs = occ_probs_2d.permute(0, 2, 3, 1).unsqueeze(1) # (B, 1, NX, NY, NZ)
        
        # Broadcast 2D planar velocity along vertical Z axis: (B, 3, NX, NY, NZ)
        vel_3d = vel_2d.unsqueeze(-1).repeat(1, 1, 1, 1, self.nz)
        
        return occ_probs, vel_3d, h_next
