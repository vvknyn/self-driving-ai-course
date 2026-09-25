"""
Unit tests for Module 07: Closed-Loop Control & Kinematic Bicycle Model
Verifies non-holonomic kinematic updates, steering limit clamping, and Stanley convergence.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bicycle_model import KinematicBicycleModel, VehicleState
from controllers import StanleyController, PurePursuitController

class TestControlSimulation(unittest.TestCase):
    def setUp(self):
        self.car = KinematicBicycleModel(wheelbase=2.8, max_steer_deg=35.0)
        self.stanley = StanleyController(k_gain=0.8)

    def test_bicycle_kinematics_straight_motion(self):
        """Verifies straight driving moves vehicle along heading vector."""
        state = VehicleState(x=0.0, y=0.0, psi=0.0, v=10.0)
        new_state = self.car.step(state, throttle_accel=0.0, steer_delta=0.0, dt=1.0)
        
        self.assertAlmostEqual(new_state.x, 10.0, places=3)
        self.assertAlmostEqual(new_state.y, 0.0, places=3)
        self.assertAlmostEqual(new_state.psi, 0.0, places=3)

    def test_steering_angle_saturation(self):
        """Verifies steering actuator commands are strictly clamped within mechanical limits."""
        state = VehicleState(x=0.0, y=0.0, psi=0.0, v=5.0)
        excessive_steer = np.radians(80.0) # > 35 degrees
        
        # Step with excessive steer
        new_state = self.car.step(state, throttle_accel=0.0, steer_delta=excessive_steer, dt=0.1)
        expected_dpsi = (5.0 / 2.8) * np.tan(self.car.max_steer) * 0.1
        self.assertAlmostEqual(new_state.psi, expected_dpsi, places=3)

    def test_stanley_error_convergence(self):
        """Verifies Stanley controller generates corrective steering toward centerline."""
        path_x = np.linspace(0, 50, 100)
        path_y = np.zeros_like(path_x)
        path_psi = np.zeros_like(path_x)
        
        # Vehicle is shifted to the left (y = +2.0m)
        front_x, front_y = 5.0, 2.0
        steer, cte, heading_err = self.stanley.compute_steering(
            front_x, front_y, vehicle_psi=0.0, vehicle_v=10.0,
            path_x=path_x, path_y=path_y, path_psi=path_psi
        )
        
        # Cross track error should be positive, steering should steer RIGHT (negative angle)
        self.assertGreater(cte, 0.0)
        self.assertLess(steer, 0.0, "Steering must command right turn to correct left offset")

if __name__ == "__main__":
    unittest.main()
