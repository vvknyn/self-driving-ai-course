#!/usr/bin/env python3
"""Execute notebooks headlessly with nbclient."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def run_notebook(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    print(f"OK: {path.name}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for name in ("00_driving_ml_gym.ipynb", "01_cameras_and_ipm.ipynb"):
        run_notebook(root / "notebooks" / name)
