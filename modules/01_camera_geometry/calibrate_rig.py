"""
Module 01: Multi-Camera Rig Calibration & Ground Plane Projection
Constructs a calibrated 3-camera Tesla-style vision rig:
  1. Front Camera (120° wide FOV, center windshield)
  2. Left Forward Camera (90° FOV, B-pillar facing forward-left)
  3. Right Forward Camera (90° FOV, B-pillar facing forward-right)

Renders a synthetic road scene with lane markers and transforms the 3 cameras
into a unified, metric Bird's-Eye View (BEV) ground plane.
"""

import numpy as np
import cv2
import os
from camera_model import PinholeCamera, create_euler_rotation
from ipm_transform import IPMTransformer

def build_tesla_style_rig(img_w: int = 640, img_h: int = 360) -> dict:
    """
    Constructs calibrated 3-camera rig.
    Units:
      Translations: meters in Vehicle Ego Frame (X forward, Y left, Z up).
      Rotations: degrees (pitch nose-down positive, yaw counter-clockwise positive, roll right-down).
    """
    cameras = {}
    
    # 1. Front Wide Camera (Center windshield, 1.4m height, pitch 4° down)
    fx_f = (img_w / 2.0) / np.tan(np.radians(60.0)) # 120° total horizontal FOV
    fy_f = fx_f
    R_f = create_euler_rotation(pitch_deg=4.0, yaw_deg=0.0, roll_deg=0.0)
    T_f = np.array([2.0, 0.0, 1.4]) # 2.0m ahead of rear axle, 1.4m above road
    cameras["front"] = PinholeCamera(
        name="front_wide",
        fx=fx_f, fy=fy_f, cx=img_w/2.0, cy=img_h/2.0,
        width=img_w, height=img_h,
        r_ego_to_cam=R_f, t_ego_to_cam=-R_f @ T_f
    )
    
    # 2. Left Forward Camera (B-pillar, 1.1m height, yaw +40° left, pitch 3° down)
    fx_l = (img_w / 2.0) / np.tan(np.radians(45.0)) # 90° FOV
    fy_l = fx_l
    R_l = create_euler_rotation(pitch_deg=3.0, yaw_deg=40.0, roll_deg=0.0)
    T_l = np.array([1.2, 0.9, 1.1]) # 1.2m forward, 0.9m left
    cameras["left"] = PinholeCamera(
        name="left_forward",
        fx=fx_l, fy=fy_l, cx=img_w/2.0, cy=img_h/2.0,
        width=img_w, height=img_h,
        r_ego_to_cam=R_l, t_ego_to_cam=-R_l @ T_l
    )
    
    # 3. Right Forward Camera (B-pillar, 1.1m height, yaw -40° right, pitch 3° down)
    fx_r = (img_w / 2.0) / np.tan(np.radians(45.0)) # 90° FOV
    fy_r = fx_r
    R_r = create_euler_rotation(pitch_deg=3.0, yaw_deg=-40.0, roll_deg=0.0)
    T_r = np.array([1.2, -0.9, 1.1]) # 1.2m forward, 0.9m right
    cameras["right"] = PinholeCamera(
        name="right_forward",
        fx=fy_r, fy=fy_r, cx=img_w/2.0, cy=img_h/2.0,
        width=img_w, height=img_h,
        r_ego_to_cam=R_r, t_ego_to_cam=-R_r @ T_r
    )
    
    return cameras

def render_synthetic_driving_scene(cameras: dict) -> dict:
    """
    Renders 3 synthetic camera views of a 3-lane highway:
      - Left lane line (dashed white) at Y = +1.875m
      - Right lane line (dashed white) at Y = -1.875m
      - Outer road boundaries (solid yellow) at Y = +5.6m and Y = -5.6m
      - Lead vehicle ahead at X = 25m, Y = 0.0m
    """
    frames = {}
    
    # 3D points of road features in Ego frame (X, Y, Z=0)
    for cam_name, cam in cameras.items():
        # Dark asphalt canvas
        img = np.full((cam.height, cam.width, 3), 45, dtype=np.uint8)
        # Sky for top portion
        sky_height = int(cam.height * 0.38)
        img[:sky_height, :] = [180, 150, 100] # BGR sky tint
        
        # 1. Project Road Boundary Lines (solid yellow)
        for y_lane in [5.6, -5.6]:
            pts_x = np.linspace(3.0, 50.0, 100)
            pts_y = np.full_like(pts_x, y_lane)
            pts_z = np.zeros_like(pts_x)
            pts_ego = np.stack([pts_x, pts_y, pts_z], axis=-1)
            pix, valid = cam.project_ego_to_pixel(pts_ego)
            
            valid_idx = np.where(valid)[0]
            if len(valid_idx) > 1:
                poly = pix[valid_idx].astype(np.int32)
                cv2.polylines(img, [poly], isClosed=False, color=(0, 215, 255), thickness=3)

        # 2. Project Dashed Lane Markers (white)
        for y_lane in [1.875, -1.875]:
            for x_start in range(4, 50, 6):
                pts_x = np.linspace(x_start, x_start + 3.0, 10)
                pts_y = np.full_like(pts_x, y_lane)
                pts_z = np.zeros_like(pts_x)
                pts_ego = np.stack([pts_x, pts_y, pts_z], axis=-1)
                pix, valid = cam.project_ego_to_pixel(pts_ego)
                valid_idx = np.where(valid)[0]
                if len(valid_idx) > 1:
                    poly = pix[valid_idx].astype(np.int32)
                    cv2.polylines(img, [poly], isClosed=False, color=(240, 240, 240), thickness=2)

        # 3. Project Lead Vehicle Box (X=22m, Y=0.0m)
        lead_box_corners = np.array([
            [22.0, -0.9, 0.0], [22.0, 0.9, 0.0],
            [22.0, 0.9, 1.5],  [22.0, -0.9, 1.5],
            [26.0, -0.9, 0.0], [26.0, 0.9, 0.0],
            [26.0, 0.9, 1.5],  [26.0, -0.9, 1.5]
        ])
        box_pix, box_valid = cam.project_ego_to_pixel(lead_box_corners)
        if np.sum(box_valid) >= 4:
            # Draw rear face
            rear_poly = box_pix[:4].astype(np.int32)
            cv2.fillPoly(img, [rear_poly], color=(140, 60, 40))
            cv2.polylines(img, [rear_poly], isClosed=True, color=(220, 220, 220), thickness=2)

        frames[cam_name] = img
        
    return frames

def main():
    print("\n" + "="*70)
    print("📸 MODULE 01: MULTI-CAMERA RIG CALIBRATION & IPM STITCHING")
    print("="*70)
    
    cameras = build_tesla_style_rig()
    print(f"Constructed {len(cameras)} cameras:")
    for name, cam in cameras.items():
        print(f"  • {name:<14} | Resolution: {cam.width}x{cam.height} | fx={cam.fx:.1f}, fy={cam.fy:.1f}")

    print("\nRendering synthetic multi-camera perspective frames...")
    frames = render_synthetic_driving_scene(cameras)
    
    # Initialize IPM transformer for the front camera
    ipm = IPMTransformer(cameras["front"], x_range=(4.0, 40.0), y_range=(-10.0, 10.0), bev_resolution=0.1)
    print(f"IPM BEV Grid size: {ipm.bev_width}x{ipm.bev_height} pixels (Coverage: 36m forward x 20m lateral)")
    
    bev_front = ipm.warp_to_bev(frames["front"])
    
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(out_dir, "cam_front.jpg"), frames["front"])
    cv2.imwrite(os.path.join(out_dir, "cam_left.jpg"), frames["left"])
    cv2.imwrite(os.path.join(out_dir, "cam_right.jpg"), frames["right"])
    cv2.imwrite(os.path.join(out_dir, "bev_ipm_front.jpg"), bev_front)
    
    print(f"\n✅ Perspective images and warped BEV saved to: {out_dir}/")
    print("Notice how the converging perspective road lines become straight, parallel lines in BEV!")

if __name__ == "__main__":
    main()
