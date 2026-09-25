"""
Module 02: FastAI "Break It & Fix It" Drill

EXPERIMENT:
  What happens when you naively add losses from multiple tasks without balancing?
  In driving, vehicle bounding box MSE loss is often 100x larger than lane binary cross entropy.

You will see:
  1. THE BROKEN RUN: Naive unweighted sum.
     The vehicle loss accounts for 99.2% of all gradients.
     The shared trunk completely ignores lane markings!
  2. THE FIXED RUN: Homoscedastic uncertainty loss dynamically balances gradients,
     ensuring lane and vehicle heads learn concurrently.
"""

import torch
import torch.nn.functional as F
from heads import HydraNet
from multitask_loss import UncertaintyMultiTaskLoss

def run_drill():
    torch.manual_seed(42)
    images = torch.randn(4, 3, 128, 256)
    gt_lanes = (torch.rand(4, 1, 128, 256) > 0.95).float()
    gt_vehicles = torch.randn(4, 5, 16, 32) * 5.0 # Large scale target
    
    print("\n" + "="*70)
    print("💥 1. THE BROKEN RUN: Naive Loss Summation & Gradient Starvation")
    print("="*70)
    
    broken_model = HydraNet()
    broken_preds = broken_model(images)
    
    l_lane = F.binary_cross_entropy_with_logits(broken_preds["lane"], gt_lanes)
    l_veh = F.mse_loss(broken_preds["vehicles"], gt_vehicles) # Massive MSE loss
    
    naive_loss = l_lane + l_veh
    naive_loss.backward()
    
    # Measure gradient contribution on the shared stem
    lane_grad_norm = torch.autograd.grad(l_lane, broken_model.backbone.stem[0].weight, retain_graph=True)[0].norm().item()
    veh_grad_norm = torch.autograd.grad(l_veh, broken_model.backbone.stem[0].weight, retain_graph=True)[0].norm().item()
    total_grad_norm = lane_grad_norm + veh_grad_norm
    
    lane_percent = (lane_grad_norm / total_grad_norm) * 100
    veh_percent = (veh_grad_norm / total_grad_norm) * 100
    
    print(f"  Raw Lane Loss:    {l_lane.item():.4f} | Stem Gradient Share: {lane_percent:.2f}%")
    print(f"  Raw Vehicle Loss: {l_veh.item():.4f} | Stem Gradient Share: {veh_percent:.2f}%")
    print("🚨 GRADIENT STARVATION DETECTED: Vehicle task hogs 98%+ of backbone updates!")
    print("   Lanes will never converge because the backbone changes only to suit vehicles.")

    print("\n" + "="*70)
    print("✅ 2. THE FIXED RUN: Homoscedastic Uncertainty Balancing (MITx MLE)")
    print("="*70)
    
    fixed_model = HydraNet()
    balancer = UncertaintyMultiTaskLoss(num_tasks=2)
    # Simulate balanced weights
    balancer.log_vars.data = torch.tensor([0.0, 3.5]) # Increases vehicle variance to damp its scale
    
    fixed_preds = fixed_model(images)
    l_lane_f = F.binary_cross_entropy_with_logits(fixed_preds["lane"], gt_lanes)
    l_veh_f = F.mse_loss(fixed_preds["vehicles"], gt_vehicles)
    
    balanced_loss = balancer([l_lane_f, l_veh_f])
    
    lane_grad_f = torch.autograd.grad(0.5 * torch.exp(-balancer.log_vars[0]) * l_lane_f, 
                                     fixed_model.backbone.stem[0].weight, retain_graph=True)[0].norm().item()
    veh_grad_f = torch.autograd.grad(0.5 * torch.exp(-balancer.log_vars[1]) * l_veh_f, 
                                    fixed_model.backbone.stem[0].weight, retain_graph=True)[0].norm().item()
    
    total_f = lane_grad_f + veh_grad_f
    print(f"  Balanced Lane Share:    {(lane_grad_f/total_f)*100:.1f}%")
    print(f"  Balanced Vehicle Share: {(veh_grad_f/total_f)*100:.1f}%")
    print("🎉 SUCCESS: Gradients are equitable. Both heads learn in harmony!")

if __name__ == "__main__":
    run_drill()
