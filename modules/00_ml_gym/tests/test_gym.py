"""
Unit tests for Module 00: Driving ML Gym
Verifies tensor dimensions, forward pass, focal loss calculation, and backpropagation.
"""

import unittest
import torch
import torch.nn.functional as F
import sys
import os

# Add parent dir to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import DrivingPatchDataset
from model import DrivingClassifier, FocalLoss

class TestDrivingMLGym(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = DrivingClassifier(num_classes=4).to(self.device)
        self.dataset = DrivingPatchDataset(num_samples=10, seed=123)

    def test_dataset_tensor_shapes(self):
        """Verifies dataset returns tensors with proper dimensions and normalization."""
        images, labels = self.dataset[0]
        self.assertEqual(images.shape, torch.Size([3, 64, 64]), "Image must be (3, 64, 64)")
        self.assertTrue(0 <= labels.item() < 4, "Label must be in [0, 3]")
        self.assertTrue(images.min() >= 0.0, "Pixels must be non-negative")
        self.assertTrue(images.max() <= 1.0, "Pixels must be <= 1.0")

    def test_model_forward_pass_dimensions(self):
        """Verifies (B, 3, 64, 64) -> (B, 4) logits."""
        batch = torch.randn(8, 3, 64, 64)
        logits = self.model(batch)
        self.assertEqual(logits.shape, torch.Size([8, 4]), "Output logits must be shape (8, 4)")
        
        # Test feature extractor
        features = self.model.extract_features(batch)
        self.assertEqual(features.shape, torch.Size([8, 64]), "Extracted features must be (8, 64)")

    def test_focal_loss_weighting(self):
        """Verifies that focal loss penalizes well-classified examples less than standard CE."""
        criterion_fl = FocalLoss(gamma=2.0)
        criterion_ce = torch.nn.CrossEntropyLoss()
        
        # High confidence correct prediction
        confident_logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
        target = torch.tensor([0])
        
        ce_val = criterion_ce(confident_logits, target).item()
        fl_val = criterion_fl(confident_logits, target).item()
        
        self.assertLess(fl_val, ce_val, "Focal loss must down-weight easy confident examples")
        self.assertGreater(fl_val, 0.0, "Loss must be positive")

    def test_gradient_backpropagation(self):
        """Verifies backward pass computes valid finite gradients for all parameters."""
        inputs = torch.randn(4, 3, 64, 64)
        targets = torch.tensor([0, 1, 2, 3])
        criterion = FocalLoss(gamma=2.0)
        
        self.model.zero_grad()
        logits = self.model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient for {name} must not be None")
            self.assertFalse(torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN")
            self.assertFalse(torch.isinf(param.grad).any(), f"Gradient for {name} contains Inf")

if __name__ == "__main__":
    unittest.main()
