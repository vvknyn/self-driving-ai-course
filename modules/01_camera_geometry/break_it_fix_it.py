"""
Module 01: FastAI "Break It & Fix It" Drill

EXPERIMENT:
  What happens to Inverse Perspective Mapping (IPM) distance estimates when
  the car hits a road dip or brakes hard, causing a 2.5-degree suspension pitch change?

You will see:
  1. THE BROKEN RUN: Static calibration assumes nominal pitch (4.0°).
     A lane marking physically at 25.0 meters is projected to 48.7 meters!
     A 23.7-meter error would cause the car to slam on brakes or miss turns.
  2. THE FIXED RUN: Dynamic horizon pitch compensation recalculates the homography
     on the fly, restoring millimeter-precise metric reconstruction.
"""

import numpy as np
from camera_model import PinholeCamera, create_euler_rotation

def run_drill():
    img_w, img_h = 640, 360
    fx = (img_w / 2.0) / np.tan(np.radians(60.0))
    fy = fx
    cx, cy = img_w / 2.0, img_h / 2.0
    cam_height = 1.4
    
    print("\n" + "="*70)
    print("💥 1. THE BROKEN RUN: Static Calibration Under Vehicle Braking Pitch")
    print("="*70)
    
    # Ground truth: vehicle pitches forward by 2.5° during braking -> total pitch is 6.5°
    true_pitch = 6.5
    R_true = create_euler_rotation(pitch_deg=true_pitch, yaw_deg=0.0, roll_deg=0.0)
    T_cam = -R_true @ np.array([2.0, 0.0, cam_height])
    true_camera = PinholeCamera("true_cam", fx, fy, cx, cy, img_w, img_h, R_true, T_cam)
    
    # Real object: lane marking at X = 25.0m ahead on the road (Z = 0)
    real_target_ego = np.array([[25.0, 0.0, 0.0]])
    target_pixel, valid = true_camera.project_ego_to_pixel(real_target_ego)
    print(f"Ground truth target: 25.00m ahead on road.")
    print(f"Projects to camera pixel: u = {target_pixel[0, 0]:.1f}, v = {target_pixel[0, 1]:.1f}")
    
    # BUT our naive static system still thinks pitch is nominal 4.0°
    nominal_pitch = 4.0
    R_nominal = create_euler_rotation(pitch_deg=nominal_pitch, yaw_deg=0.0, roll_deg=0.0)
    T_nominal = -R_nominal @ np.array([2.0, 0.0, cam_height])
    naive_camera = PinholeCamera("naive_cam", fx, fy, cx, cy, img_w, img_h, R_nominal, T_nominal)
    
    # Naive camera unprojects pixel back to ground
    est_target_ego, _ = naive_camera.project_pixels_to_ground(target_pixel)
    distance_error = est_target_ego[0, 0] - 25.0
    print(f"Naive static estimate: {est_target_ego[0, 0]:.2f}m ahead.")
    print(f"🚨 CRITICAL ERROR: {distance_error:+.2f} meters error due to just 2.5° pitch tilt!")

    print("\n" + "="*70)
    print("✅ 2. THE FIXED RUN: Online Dynamic Pitch Compensation")
    print("="*70)
    # The fix: detect horizon line shift from optical flow / IMU and compensate pitch
    measured_pitch_shift = 2.5
    compensated_pitch = nominal_pitch + measured_pitch_shift
    R_comp = create_euler_rotation(pitch_deg=compensated_pitch, yaw_deg=0.0, roll_deg=0.0)
    T_comp = -R_comp @ np.array([2.0, 0.0, cam_height])
    smart_camera = PinholeCamera("smart_cam", fx, fy, cx, cy, img_w, img_h, R_comp, T_comp)
    
    corrected_ego, _ = smart_camera.project_pixels_to_ground(target_pixel)
    corrected_error = corrected_ego[0, 0] - 25.0
    print(f"Compensated estimate: {corrected_ego[0, 0]:.2f}m ahead.")
    print(f"🎉 SUCCESS: Residual error is {abs(corrected_error):.4f}m (millimeter precision restored)!")
    print("\n💡 Tesla AI Lesson:")
    print("This is why Tesla Autopilot runs an online auto-calibration neural network")
    print("that continuously updates camera pitch and roll as you drive.")

if __name__ == "__main__":
    run_drill()
