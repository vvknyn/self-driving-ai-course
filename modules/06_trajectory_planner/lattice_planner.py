"""
Lattice Trajectory Planner via Quintic Polynomial Optimization
Solves continuous, jerk-minimal trajectory boundary value problems in Frenet / Cartesian coordinates:
  s(t): longitudinal distance along road
  d(t): lateral offset from lane centerline
"""

import numpy as np
from typing import List, Tuple

class QuinticPolynomial:
    def __init__(self, xs: float, vxs: float, axs: float, xe: float, vxe: float, axe: float, T: float):
        """
        Boundary conditions:
          Start (t=0): position xs, velocity vxs, acceleration axs
          End (t=T):   position xe, velocity vxe, acceleration axe
        """
        self.T = max(1e-3, T)
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0
        
        # 3x3 linear system for [a3, a4, a5]
        A = np.array([
            [self.T**3,       self.T**4,        self.T**5],
            [3.0 * self.T**2, 4.0 * self.T**3,  5.0 * self.T**4],
            [6.0 * self.T,    12.0 * self.T**2, 20.0 * self.T**3]
        ], dtype=np.float64)
        
        b = np.array([
            xe - self.a0 - self.a1 * self.T - self.a2 * (self.T**2),
            vxe - self.a1 - 2.0 * self.a2 * self.T,
            axe - 2.0 * self.a2
        ], dtype=np.float64)
        
        try:
            coeffs = np.linalg.solve(A, b)
            self.a3, self.a4, self.a5 = coeffs[0], coeffs[1], coeffs[2]
        except np.linalg.LinAlgError:
            self.a3, self.a4, self.a5 = 0.0, 0.0, 0.0

    def calc_point(self, t: float) -> float:
        return self.a0 + self.a1 * t + self.a2 * t**2 + self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5

    def calc_first_derivative(self, t: float) -> float:
        """Velocity"""
        return self.a1 + 2.0 * self.a2 * t + 3.0 * self.a3 * t**2 + 4.0 * self.a4 * t**3 + 5.0 * self.a5 * t**4

    def calc_second_derivative(self, t: float) -> float:
        """Acceleration"""
        return 2.0 * self.a2 + 6.0 * self.a3 * t + 12.0 * self.a4 * t**2 + 20.0 * self.a5 * t**3

    def calc_third_derivative(self, t: float) -> float:
        """Jerk (comfort metric)"""
        return 6.0 * self.a3 + 24.0 * self.a4 * t + 60.0 * self.a5 * t**2

class TrajectoryCandidate:
    def __init__(self):
        self.t = []      # time stamps
        self.x = []      # forward ego positions (m)
        self.y = []      # lateral ego positions (m)
        self.v = []      # forward speeds (m/s)
        self.a = []      # forward accelerations (m/s^2)
        self.jerk = []   # jerks (m/s^3)
        self.costs = {}  # cost decomposition
        self.total_cost = 0.0

class LatticePlanner:
    def __init__(
        self,
        target_speed: float = 18.0, # ~65 km/h cruise
        max_speed: float = 25.0,    # ~90 km/h
        max_accel: float = 3.5,     # comfort limit
        max_jerk: float = 4.0       # passenger nausea limit
    ):
        self.target_speed = target_speed
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_jerk = max_jerk

    def generate_candidate_trajectories(
        self,
        current_state: dict, # x0, y0, v0, a0
        lane_centerline_y: float = 0.0,
        lateral_offsets: List[float] = [-3.5, -1.75, 0.0, 1.75, 3.5],
        planning_horizon: float = 3.0,
        dt: float = 0.1
    ) -> List[TrajectoryCandidate]:
        """
        Samples candidate trajectories by varying target lateral offset d_T and terminal speed v_T.
        """
        x0 = current_state["x0"]
        y0 = current_state["y0"]
        v0 = current_state["v0"]
        a0 = current_state.get("a0", 0.0)
        
        candidates = []
        speed_targets = [self.target_speed - 4.0, self.target_speed, self.target_speed + 2.0]
        
        for d_target in lateral_offsets:
            for v_target in speed_targets:
                # Longitudinal boundary conditions: s0 -> sT = s0 + avg_v * T
                s_target = x0 + 0.5 * (v0 + v_target) * planning_horizon
                lon_poly = QuinticPolynomial(x0, v0, a0, s_target, v_target, 0.0, planning_horizon)
                
                # Lateral boundary conditions: y0 -> d_target (target lane offset)
                lat_poly = QuinticPolynomial(y0, 0.0, 0.0, d_target, 0.0, 0.0, planning_horizon)
                
                traj = TrajectoryCandidate()
                t_steps = np.arange(0.0, planning_horizon + dt, dt)
                
                valid = True
                for t in t_steps:
                    x_t = lon_poly.calc_point(t)
                    y_t = lat_poly.calc_point(t)
                    v_t = lon_poly.calc_first_derivative(t)
                    a_t = lon_poly.calc_second_derivative(t)
                    j_t = lon_poly.calc_third_derivative(t)
                    
                    # Kinematic feasibility constraints
                    if a_t > self.max_accel or a_t < -6.0 or v_t < 0.0:
                        valid = False
                        break
                        
                    traj.t.append(t)
                    traj.x.append(x_t)
                    traj.y.append(y_t)
                    traj.v.append(v_t)
                    traj.a.append(a_t)
                    traj.jerk.append(j_t)
                    
                if valid:
                    candidates.append(traj)
                    
        return candidates
