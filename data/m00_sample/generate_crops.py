#!/usr/bin/env python3
"""Generate synthetic 64×64 driving crop PNGs and labels.csv for Module 00.

Uses NumPy + Pillow only (no torch.randn). Seed is fixed for reproducibility.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 42
IMAGE_SIZE = 64

CLASS_NAMES = ["clear_road", "lead_vehicle", "pedestrian", "lane_marking"]

SPLIT_COUNTS = {
    "train": {0: 40, 1: 14, 2: 6, 3: 14},
    "val": {0: 10, 1: 4, 2: 3, 3: 4},
}


def _asphalt_base(rng: np.random.Generator) -> np.ndarray:
    """Return HWC float asphalt background in [0, 1]."""
    base = rng.normal(loc=0.28, scale=0.035, size=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base = np.clip(base, 0.12, 0.38)
    grad = np.linspace(0.92, 1.08, IMAGE_SIZE)[:, None, None]
    img = base * grad
    if rng.random() < 0.35:
        y0 = rng.integers(10, IMAGE_SIZE - 10)
        img[y0 : y0 + 2, :, :] *= 0.75
    return img


def _draw_clear_road(rng: np.random.Generator) -> np.ndarray:
    img = _asphalt_base(rng)
    if rng.random() < 0.4:
        x = rng.integers(8, IMAGE_SIZE - 8)
        for y in range(IMAGE_SIZE):
            xx = int(x + 0.12 * (y - IMAGE_SIZE // 2))
            if 0 <= xx < IMAGE_SIZE - 1:
                img[y, xx : xx + 2, :] = np.clip(img[y, xx : xx + 2, :] * 0.85, 0, 1)
    return np.clip(img, 0, 1)


def _draw_lead_vehicle(rng: np.random.Generator) -> np.ndarray:
    img = _asphalt_base(rng)
    x_off = rng.integers(-6, 7)
    body_color = rng.uniform(0.08, 0.35, size=3)
    body_color[2] = max(body_color[2], 0.15)
    x0, x1 = 14 + x_off, 50 + x_off
    y0, y1 = 22, 52
    img[y0:y1, x0:x1, :] = body_color
    img[y0 + 2 : y0 + 14, x0 + 4 : x1 - 4, :] = [0.05, 0.06, 0.08]
    tl_y = y0 + 16
    img[tl_y : tl_y + 6, x0 + 2 : x0 + 8, :] = [0.85, 0.08, 0.08]
    img[tl_y : tl_y + 6, x1 - 8 : x1 - 2, :] = [0.85, 0.08, 0.08]
    img[y1 : min(y1 + 4, IMAGE_SIZE), max(0, x0 - 2) : min(IMAGE_SIZE, x1 + 2), :] *= 0.55
    return np.clip(img, 0, 1)


def _draw_pedestrian(rng: np.random.Generator) -> np.ndarray:
    img = _asphalt_base(rng)
    x_center = rng.integers(22, IMAGE_SIZE - 14)
    clothing = rng.uniform(0.15, 0.75, size=3)
    img[14:20, x_center : x_center + 5, :] = [0.72, 0.58, 0.48]
    img[20:38, x_center - 1 : x_center + 6, :] = clothing
    img[38:56, x_center : x_center + 3, :] = [0.12, 0.12, 0.18]
    img[38:56, x_center + 4 : x_center + 7, :] = [0.12, 0.12, 0.18]
    return np.clip(img, 0, 1)


def _draw_lane_marking(rng: np.random.Generator) -> np.ndarray:
    img = _asphalt_base(rng)
    color = [0.95, 0.92, 0.15] if rng.random() < 0.45 else [0.95, 0.95, 0.95]
    x_start = rng.integers(20, 36)
    angle = rng.uniform(-0.18, 0.18)
    dashed = rng.random() < 0.5
    for y in range(IMAGE_SIZE):
        x = int(x_start + angle * (y - IMAGE_SIZE // 2))
        if dashed and (y // 8) % 2 == 1:
            continue
        if 0 <= x < IMAGE_SIZE - 3:
            img[y, x : x + 3, :] = color
    return np.clip(img, 0, 1)


DRAW_FNS = [_draw_clear_road, _draw_lead_vehicle, _draw_pedestrian, _draw_lane_marking]


def generate(out_dir: Path) -> None:
    """Write PNG crops and labels.csv under ``out_dir``."""
    rng = np.random.default_rng(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []

    for split, counts in SPLIT_COUNTS.items():
        for label, count in counts.items():
            for i in range(count):
                img = DRAW_FNS[label](rng)
                noise = rng.normal(0, 0.015, size=img.shape)
                img = np.clip(img + noise, 0.0, 1.0)
                filename = f"{split}_{CLASS_NAMES[label]}_{i:03d}.png"
                arr = (img * 255.0).astype(np.uint8)
                Image.fromarray(arr).save(out_dir / filename)
                rows.append(
                    {
                        "filename": filename,
                        "split": split,
                        "label": label,
                        "class_name": CLASS_NAMES[label],
                    }
                )

    csv_path = out_dir / "labels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "split", "label", "class_name"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} crops to {out_dir}")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent)
