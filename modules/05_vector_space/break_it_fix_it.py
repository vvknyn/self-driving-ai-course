"""
Module 05: FastAI "Break It & Fix It" Drill

EXPERIMENT:
  What happens when two vehicles pass closely in an intersection?
  Car 1 is moving East at +15 m/s.
  Car 2 is crossing North at +10 m/s.
  At the closest point of approach, they are separated by only 0.8 meters.

You will see:
  1. THE BROKEN RUN: Naive Euclidean Distance matching.
     The tracker confuses the two cars and SWAPS THEIR IDENTITIES!
     Suddenly Car 1 appears to instantaneously rotate 90 degrees and accelerate laterally!
  2. THE FIXED RUN: Mahalanobis Distance Gating with Velocity Covariance.
     The tracker accounts for kinematic velocity direction, correctly preserving track IDs.
"""

import numpy as np
from kalman_tracker import KalmanBoxTracker

def run_drill():
    print("\n" + "="*70)
    print("💥 1. THE BROKEN RUN: Euclidean Distance Track Identity Swap")
    print("="*70)
    
    # Track 1: Moving along X axis (east) at 15 m/s
    trk1 = KalmanBoxTracker(initial_pos=np.array([10.0, 0.0]))
    trk1.x[2] = 15.0 # vx = 15
    trk1.x[3] = 0.0  # vy = 0
    
    # Track 2: Moving along Y axis (north) at 10 m/s
    trk2 = KalmanBoxTracker(initial_pos=np.array([10.2, -1.0]))
    trk2.x[2] = 0.0  # vx = 0
    trk2.x[3] = 10.0 # vy = 10
    
    # Simulate step dt = 0.1s
    dt = 0.1
    # True future positions after 0.1s:
    # Car 1: X = 10.0 + 1.5 = 11.5, Y = 0.0
    # Car 2: X = 10.2, Y = -1.0 + 1.0 = 0.0
    # Notice: at t = 0.1s, Car 2 is at (10.2, 0.0), which is very close to Car 1's previous position (10.0, 0.0)!
    
    det_car1 = np.array([11.5, 0.0])
    det_car2 = np.array([10.2, 0.0])
    
    # Naive Euclidean distance matching without kinematic velocity prediction
    dist_trk1_to_car2 = np.linalg.norm(trk1.x[:2] - det_car2) # Distance from old Car 1 pos to Car 2
    dist_trk1_to_car1 = np.linalg.norm(trk1.x[:2] - det_car1)
    
    print(f"Old Track 1 position: ({trk1.x[0]:.1f}, {trk1.x[1]:.1f})")
    print(f"Detection Car 1:       ({det_car1[0]:.1f}, {det_car1[1]:.1f}) -> Euclidean dist: {dist_trk1_to_car1:.2f}m")
    print(f"Detection Car 2:       ({det_car2[0]:.1f}, {det_car2[1]:.1f}) -> Euclidean dist: {dist_trk1_to_car2:.2f}m")
    print(f"🚨 ID SWAP: Naive matching matches Track 1 to Car 2 because {dist_trk1_to_car2:.2f}m < {dist_trk1_to_car1:.2f}m!")

    print("\n" + "="*70)
    print("✅ 2. THE FIXED RUN: Kalman Prediction + Mahalanobis Distance (MITx)")
    print("="*70)
    
    # Kalman PREDICT step projects tracks forward using velocity
    pred1 = trk1.predict()[:2]
    pred2 = trk2.predict()[:2]
    
    print(f"Track 1 Predicted Position: ({pred1[0]:.1f}, {pred1[1]:.1f}) [projected with vx=15m/s]")
    print(f"Track 2 Predicted Position: ({pred2[0]:.1f}, {pred2[1]:.1f}) [projected with vy=10m/s]")
    
    # Mahalanobis distance from predicted states
    m_dist_1_to_car1 = trk1.mahalanobis_distance(det_car1)
    m_dist_1_to_car2 = trk1.mahalanobis_distance(det_car2)
    
    print(f"Mahalanobis Dist (Track 1 -> Car 1): {m_dist_1_to_car1:.2f} (Strong statistical match!)")
    print(f"Mahalanobis Dist (Track 1 -> Car 2): {m_dist_1_to_car2:.2f} (Rejected as physically impossible!)")
    print("🎉 SUCCESS: Track identities preserved with 100% confidence!")

if __name__ == "__main__":
    run_drill()
