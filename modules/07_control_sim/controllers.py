"""
Autonomous Vehicle Path Tracking Controllers
Implements:
1. StanleyController: Stanford DARPA Grand Challenge winner (Sebastian Thrun)
2. PurePursuitController: Geometric lookahead tracker (MIT Duckietown standard)
3. PIDLongitudinalController: Speed & acceleration regulator
"""

import numpy as np
from typing import Tuple
from bicycle_model import VehicleState

class StanleyController:
    def __init__(self, k_gain: float = 0.8, soft_factor: float = 1.0):
        """
        k_gain: cross-track error scaling gain
        soft_factor: prevents singularity when v -> 0
        """
        self.k = k_gain
        self.eps = soft_factor

    def compute_steering(
        self,
        front_x: float,
        front_y: float,
        vehicle_psi: float,
        vehicle_v: float,
        path_x: np.ndarray,
        path_y: np.ndarray,
        path_psi: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Returns:
          steer_angle: front wheel steering command (radians)
          cross_track_error: perpendicular distance to path (meters)
          heading_error: angle between vehicle and path tangent (radians)
        """
        # 1. Find nearest path waypoint to front axle
        dx = path_x - front_x
        dy = path_y - front_y
        dist_sq = dx**2 + dy**2
        nearest_idx = np.argmin(dist_sq)
        
        target_x = path_x[nearest_idx]
        target_y = path_y[nearest_idx]
        target_psi = path_psi[nearest_idx]
        
        # 2. Heading alignment error: theta_e = psi_path - psi_car
        heading_error = (target_psi - vehicle_psi + np.pi) % (2.0 * np.pi) - np.pi
        
        # 3. Signed cross-track error: determine whether vehicle is left or right of path
        # Vector from path point to front axle: [front_x - target_x, front_y - target_y]
        # Cross product with path tangent direction [-sin(target_psi), cos(target_psi)]:
        # e > 0 if car is to the right of path, e < 0 if left
        vec_path_to_axle = np.array([front_x - target_x, front_y - target_y])
        path_normal = np.array([-np.sin(target_psi), np.cos(target_psi)])
        cross_track_error = float(np.dot(vec_path_to_axle, path_normal))
        
        # 4. Stanley control law: delta = heading_error + arctan(k * e / (v + eps))
        cross_track_term = np.arctan2(self.k * -cross_track_error, max(0.1, vehicle_v) + self.eps)
        steer_angle = float(heading_error + cross_track_term)
        
        return steer_angle, cross_track_error, heading_error

class PurePursuitController:
    def __init__(self, wheelbase: float = 2.8, k_lookahead: float = 0.3, min_lookahead: float = 3.0):
        self.L = wheelbase
        self.k_look = k_lookahead
        self.min_look = min_lookahead

    def compute_steering(
        self,
        state: VehicleState,
        path_x: np.ndarray,
        path_y: np.ndarray
    ) -> float:
        # Dynamic lookahead distance
        ld = max(self.min_look, self.k_look * state.v)
        
        # Find path point at lookahead distance ld
        dx = path_x - state.x
        dy = path_y - state.y
        dists = np.hypot(dx, dy)
        
        # Pick point closest to lookahead distance ahead of car
        idx = np.argmin(np.abs(dists - ld))
        target_x, target_y = path_x[idx], path_y[idx]
        
        # Angle from car heading to target point: alpha
        angle_to_target = np.arctan2(target_y - state.y, target_x - state.x)
        alpha = (angle_to_target - state.psi + np.pi) % (2.0 * np.pi) - np.pi
        
        # Pure pursuit curvature: kappa = 2*sin(alpha) / ld
        steer_angle = np.arctan2(2.0 * self.L * np.sin(alpha), ld)
        return float(steer_angle)

class PIDLongitudinalController:
    def __init__(self, kp: float = 1.2, ki: float = 0.05, kd: float = 0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def compute_acceleration(self, current_v: float, target_v: float, dt: float) -> float:
        error = target_v - current_v
        self.integral += error * dt
        derivative = (error - self.prev_error) / max(1e-4, dt)
        self.prev_error = error
        
        accel = self.kp * error + self.ki * self.integral + self.kd * derivative
        return float(accel)
