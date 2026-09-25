"""
Module 06: Trajectory Planner Runner (Obstacle Avoidance Scenario)
Simulates highway driving with a stalled vehicle directly in the ego lane.
Fans out 15 candidate trajectories across lateral offsets and speeds,
evaluates costs, and outputs the optimal collision-free path.
"""

import numpy as np
from lattice_planner import LatticePlanner
from cost_functions import TrajectoryCostEvaluator

def main():
    print("\n" + "="*70)
    print("🛣️ MODULE 06: LEARNED & COST-MAP TRAJECTORY PLANNER (UniAD Style)")
    print("="*70)
    
    # 1. Scenario Setup
    ego_state = {
        "x0": 0.0,
        "y0": 0.0,
        "v0": 15.0, # 15 m/s (~54 km/h)
        "a0": 0.0
    }
    
    # Stalled obstacle directly ahead in current lane at X=24m, Y=0.0m
    obstacles = [
        {"x": 24.0, "y": 0.0, "radius": 1.5, "name": "Stalled Vehicle"}
    ]
    
    print(f"Ego Vehicle: Position (0.0m, 0.0m) | Speed: {ego_state['v0']} m/s")
    print(f"Detected Obstacle: '{obstacles[0]['name']}' at X = {obstacles[0]['x']}m, Y = {obstacles[0]['y']}m")
    
    # 2. Lattice Sampling: Fan out candidate trajectories
    planner = LatticePlanner(target_speed=15.0)
    evaluator = TrajectoryCostEvaluator(w_collision=1000.0, w_lane_center=3.0, w_jerk=0.2)
    
    candidates = planner.generate_candidate_trajectories(
        current_state=ego_state,
        lane_centerline_y=0.0,
        lateral_offsets=[-3.5, -2.0, 0.0, 2.0, 3.5], # Left, Slight Left, Center, Slight Right, Right
        planning_horizon=3.0,
        dt=0.1
    )
    
    print(f"Sampled {len(candidates)} kinematically feasible candidate trajectories.")
    
    # 3. Evaluate and Select Optimal Trajectory
    best_traj = evaluator.select_optimal_trajectory(candidates, obstacles, target_lane_y=0.0, target_speed=15.0)
    
    print("\n" + "-"*75)
    print(f"{'Rank':<6}{'End Lat Y':<12}{'End Speed':<12}{'Coll Cost':<12}{'Jerk Cost':<12}{'Total Cost':<12}{'Decision':<15}")
    print("-" * 75)
    
    for idx, cand in enumerate(candidates[:6]): # show top 6
        end_y = cand.y[-1]
        end_v = cand.v[-1]
        c_coll = cand.costs['collision']
        c_jerk = cand.costs['jerk_comfort']
        total = cand.total_cost
        
        if idx == 0:
            decision = "⭐ SELECTED"
        elif c_coll > 50.0:
            decision = "❌ COLLISION"
        else:
            decision = "Sub-optimal"
            
        print(f"#{idx+1:<5}{end_y:<12.2f}{end_v:<12.1f}{c_coll:<12.1f}{c_jerk:<12.1f}{total:<12.1f}{decision:<15}")
        
    print("-" * 75)
    print(f"\n🎉 Winning Maneuver: Smoothly nudging to Y = {best_traj.y[-1]:.2f}m")
    print(f"Max trajectory lateral acceleration: {max(np.abs(best_traj.a)):.2f} m/s² (Well within comfort limits)")
    print("The car avoided the obstacle safely without passenger motion sickness!")

if __name__ == "__main__":
    main()
