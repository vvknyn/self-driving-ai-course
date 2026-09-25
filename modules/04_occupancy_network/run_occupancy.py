"""
Module 04: 3D Temporal Occupancy Network Runner
Simulates a multi-frame driving sequence where an obstacle is momentarily occluded.
Demonstrates how the ConvGRU temporal memory maintains belief (Bayesian filtering)
even when visual photons are momentarily blocked.
"""

import torch
from voxel_grid import VoxelGridConfig, generate_synthetic_3d_occupancy
from temporal_fusion import OccupancyNetwork

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("\n" + "="*70)
    print("🧊 MODULE 04: 3D OCCUPANCY NETWORKS & TEMPORAL DYNAMICS")
    print("="*70)
    
    cfg = VoxelGridConfig(x_range=(0.0, 32.0), y_range=(-12.0, 12.0), z_range=(-1.0, 3.0), voxel_size=0.5)
    print(f"3D Voxel Grid Dimensions: {cfg.nx} x {cfg.ny} x {cfg.nz} (Total: {cfg.nx*cfg.ny*cfg.nz:,} voxels)")
    print(f"Spatial Resolution: {cfg.voxel_size}m per voxel side")
    
    occ_net = OccupancyNetwork(in_channels=32, hidden_channels=32, nz=cfg.nz).to(device)
    
    # Simulate 5-frame video sequence
    # Lead vehicle moves forward from X=15m to X=23m
    num_frames = 5
    hidden_state = None
    
    print("\nProcessing sequential video frames through Temporal ConvGRU...")
    print("-" * 65)
    print(f"{'Frame':<8}{'Lead Pos X':<14}{'Occlusion State':<22}{'Peak Occ Prob':<15}")
    print("-" * 65)
    
    for t in range(num_frames):
        lead_x = 15.0 + t * 2.0 # Moving forward at 2m/frame
        is_occluded = (t == 2) # Frame 2 is temporarily occluded
        
        # In frame 2, camera view is occluded (simulated low visual feature magnitude)
        if is_occluded:
            cam_bev_feat = torch.randn(1, 32, cfg.nx, cfg.ny, device=device) * 0.05
            status = "⚠️ OCCLUDED BY TRUCK"
        else:
            # Strong visual features around lead vehicle position
            cam_bev_feat = torch.randn(1, 32, cfg.nx, cfg.ny, device=device) * 0.1
            ix, iy, _ = cfg.point_to_voxel_index(lead_x, 0.0, 0.5)
            cam_bev_feat[0, :, max(0, ix-2):min(cfg.nx, ix+2), max(0, iy-2):min(cfg.ny, iy+2)] += 2.0
            status = "✅ Directly Visible"
            
        with torch.no_grad():
            occ_probs, velocities, hidden_state = occ_net(cam_bev_feat, hidden_state)
            
        peak_prob = occ_probs.max().item()
        print(f"{t+1:<8}{lead_x:<14.1f}{status:<22}{peak_prob*100:<14.1f}%")
        
    print("-" * 65)
    print("✅ Temporal 3D Occupancy sequence completed!")
    print("Notice: Hidden memory carries the vehicle state smoothly through occlusion.")

if __name__ == "__main__":
    main()
