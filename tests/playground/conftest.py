import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # before any matplotlib import: tests never open a window
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))  # the course scripts are importable modules

import pytest


@pytest.fixture(scope="session")
def shim():
    """playground/shim.py loaded by path: it is a page file, not part of the zero2fsd package."""
    spec = importlib.util.spec_from_file_location("playground_shim", ROOT / "playground" / "shim.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
