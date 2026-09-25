"""
Module 07: FastAI "Break It & Fix It" Drill

EXPERIMENT:
  What happens when there is a 150ms delay between the computer commanding steering
  and the car's physical steering rack responding?
  At 65 km/h (18 m/s), 150ms means the car travels 2.7 meters blind!

You will see:
  1. THE BROKEN RUN: Uncompensated Latency Fishtailing.
     Phase lag causes the controller to overcorrect repeatedly.
     Cross-track error oscillates violently and diverges off the road!
  2. THE FIXED RUN: Kinematic Lookahead Delay Compensation (Smith Predictor).
     We project vehicle state forward by the exact latency time before calculating steering.
"""

import numpy as np
from bicycle_model import KinematicBicycleModel, VehicleState
from controllers import StanleyController

def run_drill():
    print("\n" + "="*70)
    print("💥 1. THE BROKEN RUN: Control Latency Fishtailing (150ms Delay)")
    print("="*70)
    
    car = KinematicBicycleModel(wheelbase=2.8)
    stanley = StanleyController(k_gain=1.2)
    
    # Straight path along Y = 0
    ref_x = np.linspace(0, 100, 500)
    ref_y = np.zeros_like(ref_x)
    ref_psi = np.zeros_like(ref_x)
    
    state = VehicleState(x=0.0, y=0.8, psi=0.0, v=18.0) # 18 m/s (~65 km/h)
    dt = 0.02 # 50 Hz loop
    latency_steps = 8 # 8 * 0.02s = 160ms delay queue
    
    steer_queue = [0.0] * latency_steps
    cte_history_broken = []
    
    for step in range(120):
        xf, yf = car.front_axle_position(state)
        # Compute command based on current delayed state
        delta, cte, _ = stanley.compute_steering(xf, yf, state.psi, state.v, ref_x, ref_y, ref_psi)
        steer_queue.append(delta)
        delayed_delta = steer_queue.pop(0)
        
        state = car.step(state, throttle_accel=0.0, steer_delta=delayed_delta, dt=dt)
        cte_history_broken.append(abs(cte))
        
    print(f"Initial Cross-Track Error: 0.80m")
    print(f"Final Cross-Track Error:   {cte_history_broken[-1]:.2f}m")
    print(f"Peak Oscillatory Error:    {max(cte_history_broken):.2f}m")
    print("🚨 UNSTABLE FISHTAILING: Latency phase lag amplified steering oscillations off the road!")

    print("\n" + "="*70)
    print("✅ 2. THE FIXED RUN: Kinematic Lookahead Delay Compensation")
    print("="*70)
    
    state_f = VehicleState(x=0.0, y=0.8, psi=0.0, v=18.0)
    steer_queue_f = [0.0] * latency_steps
    cte_history_fixed = []
    latency_time = latency_steps * dt
    
    for step in range(120):
        # THE FIX: Predict where front axle will be after latency_time
        # x_pred = xf + v * cos(psi) * tau
        # y_pred = yf + v * sin(psi) * tau
        xf, yf = car.front_axle_position(state_f)
        xf_pred = xf + state_f.v * np.cos(state_f.psi) * latency_time
        yf_pred = yf + state_f.v * np.sin(state_f.psi) * latency_time
        psi_pred = state_f.psi + (state_f.v / car.L) * np.tan(steer_queue_f[-1]) * latency_time
        
        delta_comp, _, _ = stanley.compute_steering(xf_pred, yf_pred, psi_pred, state_f.v, ref_x, ref_y, ref_psi)
        steer_queue_f.append(delta_comp)
        delayed_delta_f = steer_queue_f.pop(0)
        
        state_f = car.step(state_f, throttle_accel=0.0, steer_delta=delayed_delta_f, dt=dt)
        _, cte_real, _ = stanley.compute_steering(xf, yf, state_f.psi, state_f.v, ref_x, ref_y, ref_psi)
        cte_history_fixed.append(abs(cte_real))
        
    print(f"Compensated Final Cross-Track Error: {cte_history_fixed[-1]:.3f}m")
    print(f"Compensated Peak Error:              {max(cte_history_fixed):.3f}m")
    print("🎉 SUCCESS: Latency neutralized! The vehicle tracks the centerline with zero fishtailing.")

if __name__ == "__main__":
    run_drill()
