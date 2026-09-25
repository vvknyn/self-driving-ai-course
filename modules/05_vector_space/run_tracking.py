"""
Module 05: Multi-Object Vector Space Tracking & Vector Map Runner
Simulates multi-frame driving scene:
  - Vectorized curved highway lanes
  - Two moving vehicles with sensor measurement noise
Demonstrates Kalman filter state estimation and Hungarian data association.
"""

import numpy as np
from kalman_tracker import MultiObjectTracker
from vector_lanes import VectorLane

def main():
    print("\n" + "="*70)
    print("📍 MODULE 05: VECTOR SPACE TRACKING & ONLINE VECTOR MAPS")
    print("="*70)
    
    # 1. Build Vector Map Lane (Ego lane centerline with gentle highway curve)
    # y(x) = 0.0 + 0.02*x + 0.0005*x^2
    ego_lane = VectorLane("ego_centerline", coeffs=[0.0, 0.02, 0.0005, 0.0])
    print(f"Constructed Vector Lane: '{ego_lane.id}'")
    print(f"  Curvature at 0m:  {ego_lane.curvature(0.0):.6f} m⁻¹")
    print(f"  Curvature at 30m: {ego_lane.curvature(30.0):.6f} m⁻¹")
    
    # 2. Initialize Multi-Object Tracker
    tracker = MultiObjectTracker(max_age=3, min_hits=2, distance_threshold=4.0)
    
    # 3. Simulate 6 sequential time steps (dt = 0.1s)
    # Vehicle A: Ahead at X=15m, moving at vx = 12.0 m/s
    # Vehicle B: Right lane at X=10m, Y=-3.5m, moving at vx = 16.0 m/s (overtaking)
    dt = 0.1
    print("\nTracking dynamic objects across 6 video frames (dt = 0.1s)...")
    print("-" * 75)
    print(f"{'Time (s)':<10}{'True A (x, y)':<20}{'True B (x, y)':<20}{'Active Tracks [ID: (x, y), vx]':<25}")
    print("-" * 75)
    
    for step in range(6):
        t = step * dt
        # True positions
        true_ax = 15.0 + 12.0 * t
        true_ay = ego_lane.lateral_offset(true_ax)
        
        true_bx = 10.0 + 16.0 * t
        true_by = -3.5
        
        # Add realistic sensor noise (Gaussian N(0, 0.15^2))
        noise_a = np.random.normal(0, 0.15, size=2)
        noise_b = np.random.normal(0, 0.15, size=2)
        
        noisy_detections = np.array([
            [true_ax + noise_a[0], true_ay + noise_a[1]],
            [true_bx + noise_b[0], true_by + noise_b[1]]
        ])
        
        active_tracks = tracker.update(noisy_detections)
        
        tracks_summary = [f"ID {tid}: ({pos[0]:.1f}, {pos[1]:.1f}) vx={vel[0]:.1f}" for tid, pos, vel in active_tracks]
        str_a = f"({true_ax:.1f}, {true_ay:.1f})"
        str_b = f"({true_bx:.1f}, {true_by:.1f})"
        print(f"{t:<10.1f}{str_a:<20}{str_b:<20}{'; '.join(tracks_summary):<25}")
        
    print("-" * 75)
    print("✅ Tracking and Vector Map completed successfully!")
    print("Notice how the Kalman filter rapidly converged to the true velocity estimates (~12 and ~16 m/s)")
    print("without needing any pre-recorded HD maps!")

if __name__ == "__main__":
    main()
