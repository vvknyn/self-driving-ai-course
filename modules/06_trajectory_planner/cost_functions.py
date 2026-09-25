"""
Multi-Objective Trajectory Cost Evaluator
Evaluates candidate trajectories against safety, legal, and passenger comfort objectives:
  1. Collision Risk (Exponential safety bubble over tracked obstacles & occupancy)
  2. Lane Centering (Distance from target lane centerline)
  3. Passenger Comfort / Jerk (Integral of squared jerk)
  4. Speed Progress (Deviation from desired target speed)
"""

import numpy as np
from typing import List, Dict
from lattice_planner import TrajectoryCandidate

class TrajectoryCostEvaluator:
    def __init__(
        self,
        w_collision: float = 500.0,
        w_lane_center: float = 2.0,
        w_jerk: float = 0.5,
        w_speed_progress: float = 1.0,
        safety_radius: float = 2.2
    ):
        self.w_coll = w_collision
        self.w_lane = w_lane_center
        self.w_jerk = w_jerk
        self.w_speed = w_speed_progress
        self.safety_radius = safety_radius

    def score_trajectory(
        self,
        traj: TrajectoryCandidate,
        obstacles: List[Dict], # list of dicts: {"x": float, "y": float, "radius": float}
        target_lane_y: float = 0.0,
        target_speed: float = 18.0
    ) -> float:
        """
        Computes decomposed cost for a single candidate trajectory.
        """
        cost_coll = 0.0
        cost_lane = 0.0
        cost_jerk = 0.0
        cost_speed = 0.0
        
        N = len(traj.t)
        for i in range(N):
            px = traj.x[i]
            py = traj.y[i]
            pv = traj.v[i]
            pj = traj.jerk[i]
            
            # 1. Collision Cost: Gaussian repulsive potential around obstacles
            for obs in obstacles:
                ox = obs["x"]
                oy = obs["y"]
                r_eff = obs.get("radius", 1.8) + self.safety_radius
                dist = np.hypot(px - ox, py - oy)
                if dist < r_eff:
                    # Exponential penalty inside safety bubble
                    cost_coll += np.exp(3.0 * (r_eff - dist))
                    
            # 2. Lane Centering Cost: squared lateral deviation
            cost_lane += (py - target_lane_y)**2
            
            # 3. Comfort / Jerk Cost
            cost_jerk += pj**2
            
            # 4. Speed Progress Cost
            cost_speed += (pv - target_speed)**2
            
        # Normalize by trajectory length
        cost_coll = (cost_coll / N) * self.w_coll
        cost_lane = (cost_lane / N) * self.w_lane
        cost_jerk = (cost_jerk / N) * self.w_jerk
        cost_speed = (cost_speed / N) * self.w_speed
        
        traj.costs = {
            "collision": float(cost_coll),
            "lane_centering": float(cost_lane),
            "jerk_comfort": float(cost_jerk),
            "speed_progress": float(cost_speed)
        }
        traj.total_cost = float(cost_coll + cost_lane + cost_jerk + cost_speed)
        return traj.total_cost

    def select_optimal_trajectory(
        self,
        candidates: List[TrajectoryCandidate],
        obstacles: List[Dict],
        target_lane_y: float = 0.0,
        target_speed: float = 18.0
    ) -> TrajectoryCandidate:
        """Scores all candidates and returns the minimum-cost trajectory."""
        if not candidates:
            raise ValueError("No valid trajectory candidates provided to evaluator.")
            
        for traj in candidates:
            self.score_trajectory(traj, obstacles, target_lane_y, target_speed)
            
        # Sort by total cost ascending
        candidates.sort(key=lambda t: t.total_cost)
        return candidates[0]
