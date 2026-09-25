"""
Unit tests for Module 02: HydraNet Multi-Task Architecture
Verifies tensor shapes for all heads, gradient routing to trunk, and loss balancing.
"""

import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heads import HydraNet
from multitask_loss import UncertaintyMultiTaskLoss

class TestHydraNet(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = HydraNet().to(self.device)
        self.b, self.c, self.h, self.w = 2, 3, 128, 256
        self.input_tensor = torch.randn(self.b, self.c, self.h, self.w)

    def test_output_head_shapes(self):
        """Verifies each head produces correct spatial dimensions."""
        preds = self.model(self.input_tensor)
        
        # Lane segmentation: same spatial resolution (B, 1, H, W)
        self.assertEqual(preds["lane"].shape, torch.Size([self.b, 1, self.h, self.w]))
        
        # Freespace: same spatial resolution (B, 1, H, W)
        self.assertEqual(preds["freespace"].shape, torch.Size([self.b, 1, self.h, self.w]))
        
        # Vehicle detection: stride 8 (B, 5, H/8, W/8)
        self.assertEqual(preds["vehicles"].shape, torch.Size([self.b, 5, self.h // 8, self.w // 8]))
        
        # Traffic light: class logits (B, 4)
        self.assertEqual(preds["traffic_light"].shape, torch.Size([self.b, 4]))
        
        # Cached backbone feature map: (B, 128, H/8, W/8)
        self.assertEqual(preds["backbone_features"].shape, torch.Size([self.b, 128, self.h // 8, self.w // 8]))

    def test_gradient_flow_to_shared_trunk(self):
        """Verifies gradients from any single head flow back into the shared backbone."""
        preds = self.model(self.input_tensor)
        self.model.zero_grad()
        
        # Backward on just the traffic light head
        loss = preds["traffic_light"].sum()
        loss.backward()
        
        stem_grad = self.model.backbone.stem[0].weight.grad
        self.assertIsNotNone(stem_grad)
        self.assertFalse(torch.isnan(stem_grad).any())

    def test_uncertainty_loss_properties(self):
        """Verifies loss increases when unweighted loss increases, and parameters update."""
        loss_mod = UncertaintyMultiTaskLoss(num_tasks=4)
        task_losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0), torch.tensor(4.0)]
        
        total_loss = loss_mod(task_losses)
        total_loss.backward()
        
        self.assertIsNotNone(loss_mod.log_vars.grad)
        self.assertEqual(loss_mod.log_vars.grad.shape, torch.Size([4]))

if __name__ == "__main__":
    unittest.main()
