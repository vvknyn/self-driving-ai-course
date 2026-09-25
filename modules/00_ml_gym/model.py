"""
Driving Scene Classifier & Focal Loss
Implements:
1. DrivingClassifier: Modern Residual ConvNet backbone for driving patches.
2. FocalLoss: Multi-class focal loss for handling skewed driving object frequencies.

Tensor Shapes:
  Input: (B, 3, 64, 64)
  Features: (B, 64)
  Logits: (B, 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class DrivingClassifier(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        # Initial projection
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False), # -> (B, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32)
        )
        
        # Second downsampling stage
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # -> (B, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64)
        )
        
        # Global Average Pooling & Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # -> (B, 64, 1, 1)
        self.fc = nn.Linear(64, num_classes)
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts dense embedding vector (B, 64)."""
        x = self.stem(x)
        x = self.stage2(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits (B, num_classes)."""
        features = self.extract_features(x)
        logits = self.fc(features)
        return logits

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in autonomous driving perception.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross entropy loss without reduction: log(p_t)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        # Probability of the true class
        p_t = torch.exp(-ce_loss)
        # Modulating factor (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = modulating_factor * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
            
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

if __name__ == "__main__":
    model = DrivingClassifier(num_classes=4)
    dummy_input = torch.randn(8, 3, 64, 64)
    logits = model(dummy_input)
    print(f"✅ Model forward pass successful!")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Logits shape: {logits.shape} (Expected: 8, 4)")
    
    # Test focal loss
    targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    loss_fn = FocalLoss(gamma=2.0)
    loss = loss_fn(logits, targets)
    print(f"Computed Focal Loss: {loss.item():.4f}")
