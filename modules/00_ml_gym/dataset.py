"""Driving patch dataset — loads checked-in PNGs and labels.csv only."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CLASS_NAMES = ["clear_road", "lead_vehicle", "pedestrian", "lane_marking"]


class DrivingPatchDataset(Dataset):
    """Image classifier dataset backed by PNG files on disk.

    Args:
        data_dir: Directory containing ``labels.csv`` and PNG crops.
        split: ``"train"`` or ``"val"``.
        image_size: Expected square side length (default 64).
    """

    def __init__(
        self,
        data_dir: Path | str,
        split: str = "train",
        image_size: int = 64,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        csv_path = self.data_dir / "labels.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Missing labels.csv at {csv_path}. Run data/m00_sample/generate_crops.py."
            )
        self.rows: list[dict[str, str | int]] = []
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.rows.append(
                        {
                            "filename": row["filename"],
                            "label": int(row["label"]),
                            "class_name": row["class_name"],
                        }
                    )
        if not self.rows:
            raise FileNotFoundError(f"No rows for split={split!r} in {csv_path}")
        for row in self.rows:
            png = self.data_dir / str(row["filename"])
            if not png.is_file():
                raise FileNotFoundError(f"Missing image {png}. Regenerate sample data.")

    def class_counts(self) -> dict[int, int]:
        """Return mapping label id -> count for this split."""
        counts: dict[int, int] = {i: 0 for i in range(len(CLASS_NAMES))}
        for row in self.rows:
            counts[int(row["label"])] += 1
        return counts

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        path = self.data_dir / str(row["filename"])
        img = Image.open(path).convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = torch.from_numpy(np.array(img, dtype="float32") / 255.0)
        arr = arr.permute(2, 0, 1)
        assert arr.shape == (3, self.image_size, self.image_size), (
            f"Expected (3,{self.image_size},{self.image_size}), got {tuple(arr.shape)}"
        )
        label = torch.tensor(row["label"], dtype=torch.long)
        return arr, label


def get_dataloaders(
    data_dir: Path | str,
    batch_size: int = 16,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build seeded train/val dataloaders.

    Args:
        data_dir: Root directory with PNGs and labels.csv.
        batch_size: Mini-batch size.
        seed: RNG seed for shuffling.

    Returns:
        ``(train_loader, val_loader)``
    """
    train_ds = DrivingPatchDataset(data_dir, split="train")
    val_ds = DrivingPatchDataset(data_dir, split="val")
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
