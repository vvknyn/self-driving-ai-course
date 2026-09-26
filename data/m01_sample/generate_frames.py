#!/usr/bin/env python3
"""Render synthetic 3-camera road frames and calibration JSON for Module 01."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "modules" / "01_camera_geometry"))

from calibrate_rig import build_tesla_style_rig  # noqa: E402

IMG_W, IMG_H = 320, 180
OUT_DIR = Path(__file__).resolve().parent


def _render_scene(cam) -> np.ndarray:
    img = np.full((cam.height, cam.width, 3), 45, dtype=np.uint8)
    sky_h = int(cam.height * 0.38)
    img[:sky_h, :] = [180, 150, 100]

    for y_lane, color, thick in [
        (5.6, (0, 215, 255), 3),
        (-5.6, (0, 215, 255), 3),
    ]:
        xs = np.linspace(3.0, 50.0, 80)
        pts = np.stack([xs, np.full_like(xs, y_lane), np.zeros_like(xs)], axis=-1)
        pix, valid = cam.project_ego_to_pixel(pts)
        idx = np.where(valid)[0]
        if len(idx) > 1:
            cv2.polylines(img, [pix[idx].astype(np.int32)], False, color, thick)

    for y_lane in (1.875, -1.875):
        for x0 in range(4, 50, 6):
            xs = np.linspace(x0, x0 + 3.0, 10)
            pts = np.stack([xs, np.full_like(xs, y_lane), np.zeros_like(xs)], axis=-1)
            pix, valid = cam.project_ego_to_pixel(pts)
            idx = np.where(valid)[0]
            if len(idx) > 1:
                cv2.polylines(img, [pix[idx].astype(np.int32)], False, (240, 240, 240), 2)

    box = np.array(
        [
            [22.0, -0.9, 0.0],
            [22.0, 0.9, 0.0],
            [22.0, 0.9, 1.5],
            [22.0, -0.9, 1.5],
        ]
    )
    bp, bv = cam.project_ego_to_pixel(box)
    if np.sum(bv) >= 4:
        cv2.fillPoly(img, [bp.astype(np.int32)], (140, 60, 40))
    return img


def main() -> None:
    cameras = build_tesla_style_rig(IMG_W, IMG_H)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, cam in cameras.items():
        cv2.imwrite(str(OUT_DIR / f"{name}.png"), _render_scene(cam))

    front = cameras["front"]
    calib = {
        "width": IMG_W,
        "height": IMG_H,
        "fx": front.fx,
        "fy": front.fy,
        "cx": front.cx,
        "cy": front.cy,
        "R": front.R.tolist(),
        "T": front.T.tolist(),
        "pitch_deg": 4.0,
        "yaw_deg": 0.0,
        "roll_deg": 0.0,
        "cam_position_ego": [2.0, 0.0, 1.4],
        "ego_frame": "ISO 8855: X forward, Y left, Z up; origin rear axle on ground",
        "license": "synthetic course data, CC0",
    }
    with (OUT_DIR / "calib.json").open("w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)

    ground_pts = [
        [10.0, 0.0, 0.0],
        [20.0, 1.0, 0.0],
        [30.0, -1.5, 0.0],
        [15.0, 1.875, 0.0],
        [25.0, -1.875, 0.0],
    ]
    points_payload = {"points": []}
    for ego in ground_pts:
        for cam_name, cam in cameras.items():
            pix, valid = cam.project_ego_to_pixel(np.array(ego))
            points_payload["points"].append(
                {
                    "camera": cam_name,
                    "ego_xyz": ego,
                    "pixel_uv": [float(pix[0, 0]), float(pix[0, 1])],
                    "valid": bool(valid[0]),
                }
            )

    with (OUT_DIR / "points.json").open("w", encoding="utf-8") as f:
        json.dump(points_payload, f, indent=2)

    print(f"Wrote sample data to {OUT_DIR}")


if __name__ == "__main__":
    main()
