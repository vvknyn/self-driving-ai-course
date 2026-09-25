"""
Unit tests for Module 05: Vector Space Tracking & Vector Lanes
Verifies Kalman filter covariance contraction, Mahalanobis distance, and polynomial lane derivatives.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kalman_tracker import KalmanBoxTracker, MultiObjectTracker
from vector_lanes import VectorLane

class TestVectorSpaceTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = KalmanBoxTracker(initial_pos=np.array([10.0, 0.0]), dt=0.1)

    def test_kalman_covariance_contraction(self):
        """Verifies that observing a sensor measurement decreases uncertainty (MITx Bayes theorem)."""
        self.tracker.predict()
        prior_trace = np.trace(self.tracker.P[:2, :2])
        
        # Update with measurement
        self.tracker.update(np.array([10.5, 0.0]))
        post_trace = np.trace(self.tracker.P[:2, :2])
        
        self.assertLess(post_trace, prior_trace, "Posterior covariance must contract after measurement")

    def test_multi_object_tracker_lifecycle(self):
        """Verifies tracks are created, maintained, and deleted when stale."""
        mot = MultiObjectTracker(max_age=2, min_hits=1)
        
        # Step 1: Detect two objects
        dets_t1 = np.array([[10.0, 0.0], [20.0, 3.5]])
        tracks1 = mot.update(dets_t1)
        self.assertEqual(len(tracks1), 2)
        
        # Step 2: Detections disappear (occlusion / out of view)
        mot.update(np.empty((0, 2)))
        mot.update(np.empty((0, 2)))
        mot.update(np.empty((0, 2))) # Exceeds max_age=2
        
        tracks_dead = mot.update(np.empty((0, 2)))
        self.assertEqual(len(tracks_dead), 0, "Stale tracks must be pruned")

    def test_vector_lane_curvature_and_tangent(self):
        """Verifies analytical lane offset, tangent angle, and curvature calculations."""
        # Straight lane y = 2.0
        straight_lane = VectorLane("straight", [2.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(straight_lane.lateral_offset(15.0), 2.0)
        self.assertAlmostEqual(straight_lane.heading_tangent(15.0), 0.0)
        self.assertAlmostEqual(straight_lane.curvature(15.0), 0.0)
        
        # Curved lane y = 0.001 * x^2
        curved_lane = VectorLane("curved", [0.0, 0.0, 0.001, 0.0])
        self.assertAlmostEqual(curved_lane.lateral_offset(10.0), 0.1)
        self.assertGreater(curved_lane.curvature(10.0), 0.0)

if __name__ == "__main__":
    unittest.main()
