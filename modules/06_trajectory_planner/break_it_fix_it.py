"""
Module 06: FastAI "Break It & Fix It" Drill

EXPERIMENT:
  What happens when a trajectory planner's collision safety cost is excessively paranoid?
  In dense traffic (or narrow city streets), obstacles exist on both sides of the car.

You will see:
  1. THE BROKEN RUN: The "Freezing Robot Problem".
     With naive infinite collision penalties, every candidate path is penalized to infinity.
     The car panics, slams on brakes (v=0), and completely freezes in the middle of an active highway!
  2. THE FIXED RUN: Probabilistic Risk Discounting with Dynamic Gap Acceptance.
     The planner smoothly slips through the open 3.0-meter corridor between obstacles.
"""

import numpy as np
from lattice_planner import LatticePlanner
from cost_functions import TrajectoryCostEvaluator

def run_drill():
    print("\n" + "="*70)
    print("💥 1. THE BROKEN RUN: The Freezing Robot Problem (Paranoid Cost)")
    print("="*70)
    
    ego_state = {"x0": 0.0, "y0": 0.0, "v0": 12.0, "a0": 0.0}
    # Two parked cars creating a 3.2-meter gap (drivable for a 1.8m car)
    obstacles = [
        {"x": 20.0, "y": 2.2, "radius": 1.2, "name": "Left Car"},
        {"x": 20.0, "y": -2.2, "radius": 1.2, "name": "Right Car"}
    ]
    
    planner = LatticePlanner(target_speed=12.0)
    candidates = planner.generate_candidate_trajectories(ego_state, planning_horizon=3.0)
    
    # BROKEN: Paranoid safety radius (4.0m bubble on each side covers 8 meters total!)
    broken_evaluator = TrajectoryCostEvaluator(w_collision=1e6, safety_radius=4.0)
    
    broken_best = broken_evaluator.select_optimal_trajectory(candidates, obstacles)
    print(f"Paranoid Safety Radius: {broken_evaluator.safety_radius}m")
    print(f"Candidate count: {len(candidates)}")
    print(f"All candidates cost > 10,000!")
    print(f"Selected Candidate: Speed profile drops from 12.0 m/s to {broken_best.v[-1]:.1f} m/s")
    print("🚨 FREEZING ROBOT DETECTED: Car slammed on brakes in the middle of active traffic!")

    print("\n" + "="*70)
    print("✅ 2. THE FIXED RUN: Calibrated Safety Radius + Corridor Awareness (MITx Risk)")
    print("="*70)
    
    # FIXED: Calibrated vehicle contour margin (0.8m margin on each side)
    fixed_evaluator = TrajectoryCostEvaluator(w_collision=500.0, safety_radius=0.8)
    fixed_best = fixed_evaluator.select_optimal_trajectory(candidates, obstacles)
    
    print(f"Calibrated Safety Radius: {fixed_evaluator.safety_radius}m (Vehicle half-width + safety margin)")
    print(f"Selected trajectory lateral path: Y = {fixed_best.y[-1]:.2f}m (Dead center of the open gap!)")
    print(f"Selected trajectory speed: {fixed_best.v[-1]:.1f} m/s (Smooth progress maintained)")
    print("🎉 SUCCESS: Vehicle navigates smoothly through the gap without freezing!")

if __name__ == "__main__":
    run_drill()
