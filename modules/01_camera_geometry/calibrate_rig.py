"""Tesla-style 3-camera rig, sample-data demo, artifact export."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from camera_model import PinholeCamera
from config import IPMConfig
from extrinsics import camera_position_to_translation, create_euler_rotation
from ipm import IPMTransformer
from stitch import stitch_three_cameras


def build_tesla_style_rig(img_w: int = 640, img_h: int = 360) -> dict[str, PinholeCamera]:
    """Build front/left/right cameras used by modules 03 and 08.

    Ego ISO 8855: X forward, Y left, Z up; origin at rear axle on ground.
    Translation stored as T = -R @ camera_position_ego, shape (3, 1).

    Args:
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        Dict with keys ``front``, ``left``, ``right`` mapping to PinholeCamera.
    """
    cameras: dict[str, PinholeCamera] = {}

    fx_f = (img_w / 2.0) / np.tan(np.radians(60.0))
    R_f = create_euler_rotation(pitch_deg=4.0, yaw_deg=0.0, roll_deg=0.0)
    T_f = camera_position_to_translation(R_f, np.array([2.0, 0.0, 1.4]))
    cameras["front"] = PinholeCamera(
        "front_wide", fx_f, fx_f, img_w / 2.0, img_h / 2.0, img_w, img_h, R_f, T_f
    )

    fx_l = (img_w / 2.0) / np.tan(np.radians(45.0))
    R_l = create_euler_rotation(pitch_deg=3.0, yaw_deg=40.0, roll_deg=0.0)
    T_l = camera_position_to_translation(R_l, np.array([1.2, 0.9, 1.1]))
    cameras["left"] = PinholeCamera(
        "left_forward", fx_l, fx_l, img_w / 2.0, img_h / 2.0, img_w, img_h, R_l, T_l
    )

    fx_r = (img_w / 2.0) / np.tan(np.radians(45.0))
    R_r = create_euler_rotation(pitch_deg=3.0, yaw_deg=-40.0, roll_deg=0.0)
    T_r = camera_position_to_translation(R_r, np.array([1.2, -0.9, 1.1]))
    cameras["right"] = PinholeCamera(
        "right_forward", fx_r, fx_r, img_w / 2.0, img_h / 2.0, img_w, img_h, R_r, T_r
    )

    return cameras


def _load_sample_frames(data_dir: Path) -> dict[str, np.ndarray]:
    frames = {}
    for name in ("front", "left", "right"):
        path = data_dir / f"{name}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run data/m01_sample/generate_frames.py")
        frames[name] = cv2.imread(str(path))
    return frames


def _reprojection_error(data_dir: Path, cameras: dict[str, PinholeCamera]) -> float:
    with (data_dir / "points.json").open(encoding="utf-8") as f:
        payload = json.load(f)
    errors: list[float] = []
    for entry in payload["points"]:
        cam = cameras[entry["camera"]]
        pt = np.array(entry["ego_xyz"], dtype=np.float64)
        pix, valid = cam.project_ego_to_pixel(pt)
        if not valid[0]:
            continue
        u, v = entry["pixel_uv"]
        errors.append(float(np.linalg.norm(pix[0] - np.array([u, v]))))
    return float(np.mean(errors)) if errors else 0.0


def main(cfg: IPMConfig | None = None) -> dict:
    cfg = cfg or IPMConfig()
    data_dir = Path(cfg.data_dir)
    with (data_dir / "calib.json").open(encoding="utf-8") as f:
        calib_meta = json.load(f)

    img_w = int(calib_meta["width"])
    img_h = int(calib_meta["height"])
    cameras = build_tesla_style_rig(img_w=img_w, img_h=img_h)
    frames = _load_sample_frames(data_dir)

    ipm = IPMTransformer(
        cameras["front"], cfg.x_range, cfg.y_range, cfg.bev_resolution
    )
    ground_bev = ipm.warp_to_bev(frames["front"])
    stitched = stitch_three_cameras(
        frames, cameras, cfg.x_range, cfg.y_range, cfg.bev_resolution
    )

    reproj = _reprojection_error(data_dir, cameras)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parents[2] / cfg.artifacts_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "ground_bev.png"), ground_bev)
    cv2.imwrite(str(out_dir / "stitched_bev.png"), stitched)

    calib_out = {
        "run_id": run_id,
        "cameras": {
            k: {"K": c.K.tolist(), "R": c.R.tolist(), "T": c.T.tolist()}
            for k, c in cameras.items()
        },
    }
    with (out_dir / "calib.json").open("w", encoding="utf-8") as f:
        json.dump(calib_out, f, indent=2)

    metrics = {
        "run_id": run_id,
        "note": "sample synthetic rig demo",
        "ipm_mean_reprojection_error_px": reproj,
        "pitch_shift_m": {"10": None, "20": None, "40": None},
        "outputs": {
            "ground_bev": str(out_dir / "ground_bev.png"),
            "stitched_bev": str(out_dir / "stitched_bev.png"),
            "calib_json": str(out_dir / "calib.json"),
        },
    }
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    _modules_root = Path(__file__).resolve().parents[1]
    if str(_modules_root) not in sys.path:
        sys.path.insert(0, str(_modules_root))
    from common.progress import record_event

    record_event(
        "m01",
        "artifact_exported",
        artifacts=[str(metrics_path)],
        next_session_minutes=25,
    )

    print(f"reprojection_error={reproj:.3f}px  artifacts -> {out_dir}")
    return metrics


if __name__ == "__main__":
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
