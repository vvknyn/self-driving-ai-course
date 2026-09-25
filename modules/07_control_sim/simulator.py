"""
Module 07: Closed-Loop Vehicle Simulation & Controller Benchmark
Runs closed-loop simulation on a challenging S-curve course:
  - Generates ground truth reference track (x, y, psi)
  - Simulates dynamic vehicle response with Stanley and PID controllers
  - Logs cross-track error, heading error, and steering commands
"""

import numpy as np
from bicycle_model import KinematicBicycleModel, VehicleState
from controllers import StanleyController, PIDLongitudinalController

def generate_curved_track(length_m: float = 120.0, step: float = 0.5):
    """Generates an S-curve reference path."""
    x = np.arange(0.0, length_m, step)
    # S-curve: y = 4.0 * sin(x / 15.0)
    y = 4.0 * np.sin(x / 15.0)
    
    # Path tangent angle psi = arctan(dy/dx)
    dy = 4.0 / 15.0 * np.cos(x / 15.0)
    psi = np.arctan(dy)
    return x, y, psi

def main():
    print("\n" + "="*70)
    print("🏎️ MODULE 07: CLOSED-LOOP CONTROL & KINEMATIC SIMULATION")
    print("="*70)
    
    # 1. Setup Vehicle & Track
    car = KinematicBicycleModel(wheelbase=2.8, max_steer_deg=35.0)
    stanley = StanleyController(k_gain=0.75, soft_factor=1.0)
    pid = PIDLongitudinalController(kp=1.5, ki=0.05, kd=0.1)
    
    ref_x, ref_y, ref_psi = generate_curved_track(length_m=100.0)
    target_speed = 12.0 # m/s (~43 km/h)
    
    # Initial state with intentional 1.5m cross-track error off the path
    state = VehicleState(x=0.0, y=1.5, psi=0.0, v=8.0)
    
    dt = 0.05 # 20 Hz control loop
    max_steps = 180
    
    errors_cte = []
    errors_heading = []
    
    print(f"Vehicle: Wheelbase {car.L}m | Target Speed: {target_speed} m/s")
    print(f"Initial Vehicle Offset: y = {state.y}m (1.5m off center)")
    print("\nRunning closed-loop simulation...")
    print("-" * 75)
    print(f"{'Time (s)':<10}{'Ego X (m)':<12}{'Ego Y (m)':<12}{'CTE (m)':<12}{'Heading Err (°)':<16}{'Steer (°)':<12}")
    print("-" * 75)
    
    for step in range(max_steps):
        t = step * dt
        xf, yf = car.front_axle_position(state)
        
        # Controller commands
        delta, cte, heading_err = stanley.compute_steering(xf, yf, state.psi, state.v, ref_x, ref_y, ref_psi)
        accel = pid.compute_acceleration(state.v, target_speed, dt)
        
        # Vehicle physics step
        state = car.step(state, accel, delta, dt)
        
        errors_cte.append(abs(cte))
        errors_heading.append(abs(heading_err))
        
        if step % 20 == 0:
            print(f"{t:<10.2f}{state.x:<12.1f}{state.y:<12.2f}{cte:<12.3f}{np.degrees(heading_err):<16.2f}{np.degrees(delta):<12.2f}")
            
        if state.x >= ref_x[-1] - 5.0:
            break
            
    print("-" * 75)
    mean_cte = np.mean(errors_cte)
    max_cte = np.max(errors_cte)
    print(f"\n📊 Performance Metrics (Sebastian Thrun Benchmark):")
    print(f"  Mean Absolute Cross-Track Error: {mean_cte:.3f} meters")
    print(f"  Max Absolute Cross-Track Error:  {max_cte:.3f} meters (initial pull-in)")
    print(f"  Steady-State Tracking Accuracy:  < {np.mean(errors_cte[-40:]):.3f} meters")
    print("\n✅ Closed-loop tracking simulation completed successfully!")

if __name__ == "__main__":
    main()
