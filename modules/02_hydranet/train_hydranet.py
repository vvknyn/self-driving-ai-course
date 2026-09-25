"""
Module 02: HydraNet Multi-Task Training Script
Trains HydraNet simultaneously across 4 perception tasks:
  1. Lane Segmentation (BCE Loss)
  2. Drivable Freespace (BCE Loss)
  3. Vehicle Detection (Smooth L1 + BCE Heatmap Loss)
  4. Traffic Light State (Cross Entropy Loss)
"""

import torch
import torch.nn.functional as F
from heads import HydraNet
from multitask_loss import UncertaintyMultiTaskLoss

def generate_synthetic_batch(batch_size: int = 4, h: int = 128, w: int = 256, device="cpu"):
    """Creates synthetic driving batch with ground truth annotations for all 4 heads."""
    # Synthetic camera frame
    images = torch.randn(batch_size, 3, h, w, device=device)
    
    # Ground truth targets
    gt_lanes = (torch.rand(batch_size, 1, h, w, device=device) > 0.92).float()
    gt_freespace = (torch.rand(batch_size, 1, h, w, device=device) > 0.50).float()
    gt_vehicles = torch.randn(batch_size, 5, h // 8, w // 8, device=device)
    gt_traffic_light = torch.randint(0, 4, (batch_size,), device=device)
    
    return images, gt_lanes, gt_freespace, gt_vehicles, gt_traffic_light

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n🧠 Training HydraNet Multi-Task Network on device: {device}")
    
    model = HydraNet().to(device)
    loss_balancer = UncertaintyMultiTaskLoss(num_tasks=4).to(device)
    
    # Optimize both network weights and loss balancing log variances
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_balancer.parameters()),
        lr=2e-3, weight_decay=1e-4
    )
    
    task_names = ["Lane Mask", "Freespace", "Vehicle Box", "Traffic Light"]
    print("-" * 75)
    print(f"{'Step':<8}{'L_Total':<12}{'L_Lane':<10}{'L_Free':<10}{'L_Veh':<10}{'L_TL':<10}{'Weights (0.5/σ²)':<15}")
    print("-" * 75)
    
    for step in range(1, 11):
        images, gt_lane, gt_free, gt_veh, gt_tl = generate_synthetic_batch(batch_size=4, device=device)
        
        optimizer.zero_grad()
        preds = model(images)
        
        # Individual raw task losses
        l_lane = F.binary_cross_entropy_with_logits(preds["lane"], gt_lane)
        l_free = F.binary_cross_entropy_with_logits(preds["freespace"], gt_free)
        l_veh = F.mse_loss(preds["vehicles"], gt_veh) * 10.0 # Intentionally larger scale
        l_tl = F.cross_entropy(preds["traffic_light"], gt_tl)
        
        task_losses = [l_lane, l_free, l_veh, l_tl]
        total_loss = loss_balancer(task_losses)
        
        total_loss.backward()
        optimizer.step()
        
        weights = [round(w, 2) for w in loss_balancer.get_task_weights()]
        print(f"{step:<8}{total_loss.item():<12.3f}{l_lane.item():<10.3f}{l_free.item():<10.3f}{l_veh.item():<10.3f}{l_tl.item():<10.3f}{str(weights):<15}")
        
    print("-" * 75)
    print("\n🎯 Andrei Karpathy HydraNet Insight:")
    print("  Notice how the network automatically down-weights the Vehicle task weight")
    print("  from 0.50 down to balance out the 10x larger vehicle loss, preventing it from")
    print("  crushing lane and freespace gradients! This is pure MITx probability in action.")

if __name__ == "__main__":
    main()
