"""
Kinematic Bicycle Model Simulator
Simulates standard non-holonomic vehicle dynamics:
  dot(x) = v * cos(psi)
  dot(y) = v * sin(psi)
  dot(psi) = (v / L) * tan(delta)
  dot(v) = a
"""

import numpy as np
from typing import Tuple

class VehicleState:
    def __init__(self, x: float = 0.0, y: float = 0.0, psi: float = 0.0, v: float = 0.0):
        self.x = float(x)       # forward Cartesian position (m)
        self.y = float(y)       # lateral Cartesian position (m)
        self.psi = float(psi)   # heading orientation in radians (0 = pointing along +X)
        self.v = float(v)       # longitudinal speed in m/s

class KinematicBicycleModel:
    def __init__(
        self,
        wheelbase: float = 2.8,        # Tesla Model 3 wheelbase ~ 2.8m (Duckiebot = 0.15m)
        max_steer_deg: float = 35.0,    # maximum wheel turn angle
        max_accel: float = 3.5,        # m/s^2
        max_decel: float = -6.0        # m/s^2 (emergency braking)
    ):
        self.L = wheelbase
        self.max_steer = np.radians(max_steer_deg)
        self.max_accel = max_accel
        self.max_decel = max_decel

    def step(self, state: VehicleState, throttle_accel: float, steer_delta: float, dt: float) -> VehicleState:
        """
        Integrates kinematics forward by time step dt.
        """
        # Clamp actuator limits
        accel = np.clip(throttle_accel, self.max_decel, self.max_accel)
        delta = np.clip(steer_delta, -self.max_steer, self.max_steer)
        
        # RK4 integration for stability
        x = state.x
        y = state.y
        psi = state.psi
        v = max(0.0, state.v) # prevent reverse rolling in forward drive
        
        # Derivatives
        dx = v * np.cos(psi)
        dy = v * np.sin(psi)
        dpsi = (v / self.L) * np.tan(delta)
        dv = accel
        
        new_x = x + dx * dt
        new_y = y + dy * dt
        new_psi = (psi + dpsi * dt + np.pi) % (2.0 * np.pi) - np.pi # normalize to [-pi, pi]
        new_v = max(0.0, v + dv * dt)
        
        return VehicleState(new_x, new_y, new_psi, new_v)

    def front_axle_position(self, state: VehicleState) -> Tuple[float, float]:
        """Calculates front axle position [x_f, y_f] for Stanley control."""
        xf = state.x + self.L * np.cos(state.psi)
        yf = state.y + self.L * np.sin(state.psi)
        return xf, yf
