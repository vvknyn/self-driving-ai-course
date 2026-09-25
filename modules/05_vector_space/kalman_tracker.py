"""
Multi-Object Vector Space Tracker via Kalman Filtering & Hungarian Matching
Implements:
1. KalmanBoxTracker: Gaussian State Estimation (x, y, vx, vy) with covariance P.
2. MultiObjectTracker: Temporal track lifecycle management & Mahalanobis data association.

MITx Probability Connection:
  - Predict step computes prior via Chapman-Kolmogorov / Total Probability.
  - Update step conditions Gaussian likelihood via Bayes' Theorem.
  - Mahalanobis distance computes Chi-squared likelihood for outlier rejection.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple

class KalmanBoxTracker:
    count = 0

    def __init__(self, initial_pos: np.ndarray, dt: float = 0.1):
        """
        initial_pos: [x, y] in vehicle ego frame (meters).
        dt: time step in seconds between perception frames (e.g. 10Hz -> 0.1s).
        """
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        self.dt = dt
        # State vector: [x, y, vx, vy]^T
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=np.float64)
        
        # State covariance P (4x4): initial uncertainty
        self.P = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float64)
        
        # State Transition Matrix F: Constant velocity motion model
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        # Process Noise Covariance Q: uncertainty in vehicle acceleration
        q_pos = 0.1
        q_vel = 1.0
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float64)
        
        # Measurement Matrix H: sensor directly observes position [x, y]
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)
        
        # Measurement Noise Covariance R: sensor error (~0.3m standard deviation)
        self.R = np.diag([0.09, 0.09]).astype(np.float64)
        
        self.age = 0
        self.hits = 1
        self.time_since_update = 0

    def predict(self) -> np.ndarray:
        """Prior state update: x^- = F * x, P^- = F * P * F^T + Q"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.x

    def update(self, measurement: np.ndarray):
        """Bayesian conditioning: update prior belief with new sensor measurement z."""
        z = np.asarray(measurement, dtype=np.float64)[:2]
        y = z - (self.H @ self.x) # Innovation / residual
        
        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman Gain: K = P * H^T * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Posterior mean: x = x^- + K * y
        self.x = self.x + (K @ y)
        
        # Posterior covariance: P = (I - K * H) * P (contracting uncertainty)
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        
        self.hits += 1
        self.time_since_update = 0

    def mahalanobis_distance(self, measurement: np.ndarray) -> float:
        """
        Computes Mahalanobis distance between measurement and track's predicted position.
        d_M = sqrt(y^T * S^-1 * y)
        """
        z = np.asarray(measurement, dtype=np.float64)[:2]
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        d2 = y.T @ np.linalg.inv(S) @ y
        return np.sqrt(max(0.0, d2))

class MultiObjectTracker:
    def __init__(self, max_age: int = 5, min_hits: int = 2, distance_threshold: float = 3.5):
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        self.trackers: List[KalmanBoxTracker] = []

    def update(self, detections: np.ndarray) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        """
        Runs full track lifecycle:
          1. Predict all existing tracks
          2. Associate detections via Hungarian Algorithm
          3. Update matched tracks, create new tracks, delete dead tracks
        Returns:
          active_tracks: List of tuples (track_id, position [x, y], velocity [vx, vy])
        """
        # Step 1: Predict
        for trk in self.trackers:
            trk.predict()
            
        N_trk = len(self.trackers)
        N_det = len(detections)
        
        if N_trk > 0 and N_det > 0:
            # Build cost matrix using Mahalanobis distance
            cost_matrix = np.zeros((N_trk, N_det), dtype=np.float64)
            for i, trk in enumerate(self.trackers):
                for j, det in enumerate(detections):
                    cost_matrix[i, j] = trk.mahalanobis_distance(det)
                    
            # Hungarian bipartite optimal assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched_trks = set()
            matched_dets = set()
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < self.distance_threshold:
                    self.trackers[r].update(detections[c])
                    matched_trks.add(r)
                    matched_dets.add(c)
                    
            # Unmatched detections become new tracks
            for j in range(N_det):
                if j not in matched_dets:
                    self.trackers.append(KalmanBoxTracker(detections[j]))
        elif N_det > 0:
            # Initialize new tracks for all detections
            for det in detections:
                self.trackers.append(KalmanBoxTracker(det))
                
        # Remove dead tracks that haven't been updated recently
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        
        # Return confirmed active tracks (hits >= min_hits or new)
        active = []
        for t in self.trackers:
            if t.hits >= self.min_hits or t.age <= 2:
                active.append((t.id, t.x[:2], t.x[2:4]))
        return active
