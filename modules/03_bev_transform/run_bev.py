"""
Module 03: Multi-Camera to BEV Execution Script
Connects the 3-camera rig (Front, Left, Right) to the Lift-Splat-Shoot BEV pipeline.
Outputs a unified top-down 2D metric BEV feature map.
"""

import torch
import numpy as np
import sys
import os

# Add Module 01 to sys.path for camera rig
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_camera_geometry"))
from calibrate_rig import build_tesla_style_rig
from lift_splat_shoot import LiftSplatShoot

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("\n" + "="*70)
    print("🦅 MODULE 03: MULTI-CAMERA BIRD'S-EYE VIEW (BEV) LIFT-SPLAT-SHOOT")
    print("="*70)
    
    # 1. Load 3-camera rig
    cams = build_tesla_style_rig(img_w=256, img_h=128)
    cam_keys = ["front", "left", "right"]
    N_cams = len(cam_keys)
    
    # Pack camera calibration matrices into tensors (1, N_cams, 3, 3)
    K_list = [torch.tensor(cams[k].K, dtype=torch.float32) for k in cam_keys]
    R_list = [torch.tensor(cams[k].R, dtype=torch.float32) for k in cam_keys]
    T_list = [torch.tensor(cams[k].T, dtype=torch.float32) for k in cam_keys]
    
    K_tensor = torch.stack(K_list).unsqueeze(0).to(device)
    R_tensor = torch.stack(R_list).unsqueeze(0).to(device)
    T_tensor = torch.stack(T_list).unsqueeze(0).to(device)
    
    # 2. Instantiate Lift-Splat-Shoot
    # BEV grid: 0 to 40 meters forward (step 0.5m -> 80 bins)
    #           -15 to +15 meters lateral (step 0.5m -> 60 bins)
    lss = LiftSplatShoot(
        in_channels=32,
        out_channels=64,
        d_min=2.0,
        d_max=40.0,
        num_depth_bins=16,
        x_bound=(0.0, 40.0, 0.5),
        y_bound=(-15.0, 15.0, 0.5)
    ).to(device)
    
    # 3. Simulate multi-camera image feature maps (B=1, N_cams=3, C=32, H=16, W=32)
    # In full stack, these features come directly from HydraNet backbone!
    sim_cam_features = torch.randn(1, N_cams, 32, 16, 32, device=device)
    
    print("Executing Lift-Splat-Shoot transformation across 3 camera feeds...")
    print(f"  Input camera features shape: {sim_cam_features.shape}")
    
    with torch.no_grad():
        bev_map = lss(sim_cam_features, K_tensor, R_tensor, T_tensor)
        
    print(f"  Output BEV feature map shape: {bev_map.shape} (B, Channels, Y_bins, X_bins)")
    print(f"  Metric coverage: {lss.x_bound[0]}m to {lss.x_bound[1]}m forward, {lss.y_bound[0]}m to {lss.y_bound[1]}m lateral")
    print(f"  Resolution: {lss.x_bound[2]}m per grid cell")
    print("\n✅ BEV transformation completed successfully!")
    print("The 3 separate 2D camera views have been seamlessly fused into a top-down metric plane.")

if __name__ == "__main__":
    main()
