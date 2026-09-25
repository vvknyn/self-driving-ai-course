"""
Master Mini-FSD End-to-End Pipeline
Wires Modules 01 through 07 into a unified autonomous driving stack:
  Multi-Camera Frames -> HydraNet -> LSS BEV -> 3D Occupancy -> Vector Tracking -> Trajectory Planner -> Closed-Loop Control
"""

import torch
import numpy as np
import os
import sys
from typing import Dict, List, Tuple

# Add module directories to sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(repo_root, "modules", "01_camera_geometry"))
sys.path.insert(0, os.path.join(repo_root, "modules", "02_hydranet"))
sys.path.insert(0, os.path.join(repo_root, "modules", "03_bev_transform"))
sys.path.insert(0, os.path.join(repo_root, "modules", "04_occupancy_network"))
sys.path.insert(0, os.path.join(repo_root, "modules", "05_vector_space"))
sys.path.insert(0, os.path.join(repo_root, "modules", "06_trajectory_planner"))
sys.path.insert(0, os.path.join(repo_root, "modules", "07_control_sim"))

from calibrate_rig import build_tesla_style_rig
from heads import HydraNet
from lift_splat_shoot import LiftSplatShoot
from temporal_fusion import OccupancyNetwork
from kalman_tracker import MultiObjectTracker
from vector_lanes import VectorLane
from lattice_planner import LatticePlanner
from cost_functions import TrajectoryCostEvaluator
from bicycle_model import KinematicBicycleModel, VehicleState
from controllers import StanleyController, PIDLongitudinalController

class FullFSDPipeline:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        
        # 1. Geometry & Cameras
        self.cameras = build_tesla_style_rig(img_w=256, img_h=128)
        self.cam_keys = ["front", "left", "right"]
        self.N_cams = len(self.cam_keys)
        
        K_list = [torch.tensor(self.cameras[k].K, dtype=torch.float32) for k in self.cam_keys]
        R_list = [torch.tensor(self.cameras[k].R, dtype=torch.float32) for k in self.cam_keys]
        T_list = [torch.tensor(self.cameras[k].T, dtype=torch.float32) for k in self.cam_keys]
        
        self.K_tensor = torch.stack(K_list).unsqueeze(0).to(self.device)
        self.R_tensor = torch.stack(R_list).unsqueeze(0).to(self.device)
        self.T_tensor = torch.stack(T_list).unsqueeze(0).to(self.device)
        
        # 2. Perception: HydraNet shared trunk
        self.hydranet = HydraNet().to(self.device)
        self.hydranet.eval()
        
        # 3. BEV Transformation: Lift-Splat-Shoot
        self.lss = LiftSplatShoot(
            in_channels=128, out_channels=32,
            d_min=2.0, d_max=40.0, num_depth_bins=12,
            x_bound=(0.0, 36.0, 0.6), y_bound=(-12.0, 12.0, 0.6)
        ).to(self.device)
        self.lss.eval()
        
        # 4. 3D Occupancy & Temporal Memory
        self.occ_net = OccupancyNetwork(in_channels=32, hidden_channels=32, nz=8).to(self.device)
        self.occ_net.eval()
        self.hidden_state = None
        
        # 5. Vector Space Tracking & Lanes
        self.tracker = MultiObjectTracker(max_age=3, min_hits=1, distance_threshold=4.0)
        self.vector_lane = VectorLane("ego_lane", [0.0, 0.01, 0.0002, 0.0])
        
        # 6. Trajectory Planning
        self.planner = LatticePlanner(target_speed=14.0)
        self.evaluator = TrajectoryCostEvaluator(w_collision=800.0, w_lane_center=2.5, w_jerk=0.3)
        
        # 7. Vehicle Dynamics & Control
        self.vehicle = KinematicBicycleModel(wheelbase=2.8)
        self.stanley = StanleyController(k_gain=0.8)
        self.pid_speed = PIDLongitudinalController(kp=1.5, ki=0.05, kd=0.1)
        self.state = VehicleState(x=0.0, y=0.0, psi=0.0, v=12.0)

    def step(self, camera_frames: torch.Tensor, ground_truth_obstacles: List[Dict], dt: float = 0.1) -> Dict:
        """
        Executes one full perception -> prediction -> planning -> control loop.
        camera_frames: (1, N_cams, 3, H, W)
        """
        with torch.no_grad():
            B, N_cams, C, H, W = camera_frames.shape
            
            # Step 1: HydraNet feature extraction across all cameras
            # Reshape to batch of camera frames: (B*N_cams, 3, H, W)
            flat_cams = camera_frames.view(B * N_cams, C, H, W)
            cam_feats = self.hydranet.backbone(flat_cams)["p3"] # (B*N_cams, 128, H/8, W/8)
            feat_h, feat_w = cam_feats.shape[-2], cam_feats.shape[-1]
            cam_feats = cam_feats.view(B, N_cams, 128, feat_h, feat_w)
            
            # Step 2: Lift-Splat-Shoot to BEV
            bev_features = self.lss(cam_feats, self.K_tensor, self.R_tensor, self.T_tensor)
            
            # Step 3: 3D Temporal Occupancy Network
            occ_probs, velocities, self.hidden_state = self.occ_net(bev_features, self.hidden_state)
            
        # Step 4: Vector Space Tracking (associate detections to persistent tracks)
        det_positions = np.array([[obs["x"], obs["y"]] for obs in ground_truth_obstacles]) if ground_truth_obstacles else np.empty((0, 2))
        active_tracks = self.tracker.update(det_positions)
        
        # Step 5: Trajectory Planning (Quintic Lattice candidate generation & scoring)
        ego_dict = {"x0": self.state.x, "y0": self.state.y, "v0": self.state.v, "a0": 0.0}
        candidates = self.planner.generate_candidate_trajectories(
            ego_dict, lane_centerline_y=0.0,
            lateral_offsets=[-3.5, -1.8, 0.0, 1.8, 3.5],
            planning_horizon=2.5, dt=dt
        )
        
        # Format obstacles for collision evaluator
        obs_for_planner = [{"x": p[0], "y": p[1], "radius": 1.6} for _, p, _ in active_tracks]
        optimal_traj = self.evaluator.select_optimal_trajectory(
            candidates, obs_for_planner, target_lane_y=0.0, target_speed=14.0
        )
        
        # Step 6: Closed-Loop Control (Stanley steering & PID throttle)
        xf, yf = self.vehicle.front_axle_position(self.state)
        path_x = np.array(optimal_traj.x)
        path_y = np.array(optimal_traj.y)
        path_psi = np.zeros_like(path_x)
        if len(path_x) > 1:
            dx = np.gradient(path_x)
            dy = np.gradient(path_y)
            path_psi = np.arctan2(dy, dx)
            
        steer_cmd, cte, heading_err = self.stanley.compute_steering(
            xf, yf, self.state.psi, self.state.v, path_x, path_y, path_psi
        )
        accel_cmd = self.pid_speed.compute_acceleration(self.state.v, target_v=optimal_traj.v[1], dt=dt)
        
        # Step 7: Integrate Vehicle Physics
        self.state = self.vehicle.step(self.state, accel_cmd, steer_cmd, dt)
        
        return {
            "ego_x": self.state.x,
            "ego_y": self.state.y,
            "ego_v": self.state.v,
            "ego_psi": self.state.psi,
            "steer_angle": steer_cmd,
            "throttle_accel": accel_cmd,
            "cross_track_error": cte,
            "active_tracks_count": len(active_tracks),
            "planned_waypoints_x": optimal_traj.x,
            "planned_waypoints_y": optimal_traj.y,
            "selected_traj_cost": optimal_traj.total_cost
        }
