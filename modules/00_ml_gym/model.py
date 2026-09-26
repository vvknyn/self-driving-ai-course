"""Tiny CNN for 64×64 driving patch classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DrivingClassifier(nn.Module):
    """Small conv net: (B, 3, 64, 64) -> logits (B, K)."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return embedding of shape (B, 64)."""
        x = self.stem(x)
        x = self.stage2(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (B, num_classes)."""
        return self.fc(self.extract_features(x))
