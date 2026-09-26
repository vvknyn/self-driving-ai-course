"""Training configuration for Module 00 — Driving ML Gym."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _default_data_dir() -> Path:
    """Resolve default sample-crop directory relative to repo root."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "m00_sample"


@dataclass
class TrainConfig:
    """Hyperparameters and paths for the driving-patch classifier."""

    data_dir: Path | str = ""
    image_size: int = 64
    num_classes: int = 4
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 6
    seed: int = 0
    minority_class: int = 2
    artifacts_root: str = "artifacts/m00"
    gamma: float = 2.0
    loss_name: str = "cross_entropy"

    def __post_init__(self) -> None:
        if not self.data_dir:
            self.data_dir = _default_data_dir()
        self.data_dir = Path(self.data_dir)
