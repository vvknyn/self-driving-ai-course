"""
HydraNet Shared Trunk Backbone
Takes raw camera frames (B, 3, H, W) and extracts shared multi-scale features:
  Stage 1 (stride 2): (B, 32, H/2, W/2)
  Stage 2 (stride 4): (B, 64, H/4, W/4)
  Stage 3 (stride 8): (B, 128, H/8, W/8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.skip = nn.Identity() if (stride == 1 and in_c == out_c) else nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x) + self.skip(x))

class HydraNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False), # H/2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Stage 2: Stride 4
        self.stage2 = ConvBlock(32, 64, stride=2) # H/4
        # Stage 3: Stride 8
        self.stage3 = ConvBlock(64, 128, stride=2) # H/8

    def forward(self, x: torch.Tensor) -> dict:
        """
        Returns dictionary of multi-scale feature pyramids.
        """
        f1 = self.stem(x)    # (B, 32, H/2, W/2)
        f2 = self.stage2(f1)  # (B, 64, H/4, W/4)
        f3 = self.stage3(f2)  # (B, 128, H/8, W/8)
        return {
            "p1": f1,
            "p2": f2,
            "p3": f3
        }

if __name__ == "__main__":
    backbone = HydraNetBackbone()
    dummy = torch.randn(2, 3, 256, 512)
    feats = backbone(dummy)
    print("HydraNet Backbone Feature Shapes:")
    for k, v in feats.items():
        print(f"  {k}: {v.shape}")
