"""Configuration for Module 01 — cameras & IPM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class IPMConfig:
    """Paths and BEV grid settings."""

    data_dir: Path | str = ""
    image_width: int = 320
    image_height: int = 180
    x_range: tuple[float, float] = (4.0, 40.0)
    y_range: tuple[float, float] = (-10.0, 10.0)
    bev_resolution: float = 0.1
    artifacts_root: str = "artifacts/m01"

    def __post_init__(self) -> None:
        if not self.data_dir:
            self.data_dir = _repo_root() / "data" / "m01_sample"
        self.data_dir = Path(self.data_dir)
