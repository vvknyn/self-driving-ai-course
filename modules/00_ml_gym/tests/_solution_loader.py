"""Load reference solutions under unique module names."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_solution_module(name: str, filename: str) -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "solutions" / "00_ml_gym" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load solution {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
