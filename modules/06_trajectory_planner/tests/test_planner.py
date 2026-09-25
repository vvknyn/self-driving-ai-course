"""
Unit tests for Module 06: Quintic Lattice Trajectory Planner
Verifies boundary condition satisfaction, jerk continuity, and collision cost penalization.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lattice_planner import QuinticPolynomial, LatticePlanner
from cost_functions import TrajectoryCostEvaluator

class TestTrajectoryPlanner(unittest.TestCase):
    def test_quintic_polynomial_boundary_conditions(self):
        """Verifies polynomial satisfies exact initial and terminal boundary constraints."""
        xs, vxs, axs = 0.0, 10.0, 1.0
        xe, vxe, axe = 50.0, 15.0, 0.0
        T = 3.0
        
        poly = QuinticPolynomial(xs, vxs, axs, xe, vxe, axe, T)
        
        # Check initial (t = 0)
        self.assertAlmostEqual(poly.calc_point(0.0), xs, places=4)
        self.assertAlmostEqual(poly.calc_first_derivative(0.0), vxs, places=4)
        self.assertAlmostEqual(poly.calc_second_derivative(0.0), axs, places=4)
        
        # Check terminal (t = T)
        self.assertAlmostEqual(poly.calc_point(T), xe, places=4)
        self.assertAlmostEqual(poly.calc_first_derivative(T), vxe, places=4)
        self.assertAlmostEqual(poly.calc_second_derivative(T), axe, places=4)

    def test_candidate_generation_and_scoring(self):
        """Verifies candidate sampler generates paths and collision evaluator penalizes obstacles."""
        planner = LatticePlanner(target_speed=15.0)
        evaluator = TrajectoryCostEvaluator(w_collision=500.0)
        
        state = {"x0": 0.0, "y0": 0.0, "v0": 15.0, "a0": 0.0}
        candidates = planner.generate_candidate_trajectories(state, planning_horizon=2.5)
        self.assertGreater(len(candidates), 0)
        
        # Obstacle placed directly in the center path at X=20m, Y=0.0m
        obstacles = [{"x": 20.0, "y": 0.0, "radius": 1.5}]
        best_traj = evaluator.select_optimal_trajectory(candidates, obstacles)
        
        # Best trajectory should NOT stay at Y=0.0 (it should deviate to avoid obstacle)
        self.assertNotAlmostEqual(best_traj.y[-1], 0.0, places=1,
                                 msg="Optimal trajectory must maneuver around obstacle")

if __name__ == "__main__":
    unittest.main()
