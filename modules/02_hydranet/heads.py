"""
HydraNet Task-Specific Heads
1. LaneSegmentationHead: Upsamples features to predict lane line mask (B, 1, H, W).
2. FreespaceHead: Predicts drivable ground area mask (B, 1, H, W).
3. VehicleDetectionHead: Predicts vehicle objectness heatmap + bounding box offsets (B, 5, H/8, W/8).
4. TrafficLightHead: Classifies global scene traffic light state (B, 4) [Green, Yellow, Red, None].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import HydraNetBackbone

class LaneSegmentationHead(nn.Module):
    def __init__(self, in_channels: int = 128):
        super().__init__()
        # Upsampling decoder with skip connections
        self.up1 = nn.ConvTranspose2d(in_channels, 64, kernel_size=4, stride=2, padding=1) # -> H/4
        self.conv1 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) # -> H/2
        self.conv2 = nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1)
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1) # -> H
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.up1(p3))
        x = torch.cat([x, p2], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.up2(x))
        x = torch.cat([x, p1], dim=1)
        x = F.relu(self.conv2(x))
        x = F.relu(self.up3(x))
        return self.out_conv(x) # (B, 1, H, W)

class FreespaceHead(nn.Module):
    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(self, p3: torch.Tensor) -> torch.Tensor:
        return self.decoder(p3) # (B, 1, H, W)

class VehicleDetectionHead(nn.Module):
    def __init__(self, in_channels: int = 128):
        super().__init__()
        # Predicts: [confidence, center_x, center_y, width, height]
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 5, kernel_size=1)
        )

    def forward(self, p3: torch.Tensor) -> torch.Tensor:
        return self.conv(p3) # (B, 5, H/8, W/8)

class TrafficLightHead(nn.Module):
    def __init__(self, in_channels: int = 128, num_states: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_states)
        )

    def forward(self, p3: torch.Tensor) -> torch.Tensor:
        x = self.pool(p3)
        x = torch.flatten(x, 1)
        return self.fc(x) # (B, 4)

class HydraNet(nn.Module):
    """Full HydraNet multi-task perception model."""
    def __init__(self):
        super().__init__()
        self.backbone = HydraNetBackbone()
        self.lane_head = LaneSegmentationHead(128)
        self.freespace_head = FreespaceHead(128)
        self.vehicle_head = VehicleDetectionHead(128)
        self.traffic_light_head = TrafficLightHead(128)

    def forward(self, x: torch.Tensor) -> dict:
        feats = self.backbone(x)
        p1, p2, p3 = feats["p1"], feats["p2"], feats["p3"]
        
        return {
            "lane": self.lane_head(p1, p2, p3),
            "freespace": self.freespace_head(p3),
            "vehicles": self.vehicle_head(p3),
            "traffic_light": self.traffic_light_head(p3),
            "backbone_features": p3 # Caching for BEV lifting in Module 03!
        }
