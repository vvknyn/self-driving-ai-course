"""
Unit tests for Module 08: End-to-End Capstone Pipeline
Verifies complete perception -> prediction -> planning -> control loop execution.
"""

import unittest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import FullFSDPipeline
from scenario_generator import DrivingScenario

class TestIntegrationPipeline(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.pipeline = FullFSDPipeline(device=self.device)
        self.scenario = DrivingScenario(num_frames=5, dt=0.1)

    def test_pipeline_step_execution(self):
        """Verifies pipeline runs a step and returns complete valid telemetry dictionary."""
        cam_batch, obstacles = self.scenario.step()
        telemetry = self.pipeline.step(cam_batch, obstacles, dt=0.1)
        
        required_keys = [
            "ego_x", "ego_y", "ego_v", "ego_psi",
            "steer_angle", "throttle_accel", "cross_track_error",
            "active_tracks_count", "planned_waypoints_x", "planned_waypoints_y"
        ]
        
        for k in required_keys:
            self.assertIn(k, telemetry, f"Key '{k}' missing from pipeline telemetry output")
            
        # Numerical validity checks
        self.assertFalse(np.isnan(telemetry["steer_angle"]))
        self.assertFalse(np.isnan(telemetry["throttle_accel"]))
        self.assertFalse(np.isnan(telemetry["cross_track_error"]))
        self.assertGreater(telemetry["ego_v"], 0.0)

    def test_multi_step_state_advancement(self):
        """Verifies vehicle position advances forward monotonically over multiple steps."""
        initial_x = self.pipeline.state.x
        
        for _ in range(3):
            cam_batch, obstacles = self.scenario.step()
            telemetry = self.pipeline.step(cam_batch, obstacles, dt=0.1)
            
        self.assertGreater(telemetry["ego_x"], initial_x, "Vehicle must make forward progress")

if __name__ == "__main__":
    unittest.main()
