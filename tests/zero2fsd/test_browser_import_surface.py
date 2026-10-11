"""The browser (Pyodide) build has numpy and pillow but neither matplotlib nor pyyaml until asked.

Each test starts a fresh interpreter in which the named package cannot be imported (`sys.modules[name] = None`
makes `import name` raise ImportError), imports every module the playground shim imports, and runs the
learner-facing paths: `grade.check(..., show=False)` and a full drive.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BROWSER_ABSENT = ("matplotlib", "yaml")
BROWSER_MODULES = (
    "zero2fsd", "zero2fsd.sim", "zero2fsd.car", "zero2fsd.grade", "zero2fsd.score", "zero2fsd.dashboard",
    "zero2fsd.export",
    "zero2fsd.palette",
)
SCRIPT = """
import importlib, sys

absent, *modules = sys.argv[1:]
sys.modules[absent] = None  # from here on `import <absent>` raises ImportError, as it does in the browser
for module in modules:
    importlib.import_module(module)

import numpy as np
from zero2fsd.car import Car, run
from zero2fsd.grade import check


def lane_error_stats(lat):
    a = np.abs(lat)
    return {"mean_abs": float(a.mean()), "max_abs": float(a.max()), "rms": float(np.sqrt(np.mean(lat ** 2)))}


result = check("0.1.1.a", lane_error_stats, show=False)
assert result.passed, result.details
assert run(Car(), "straight").completed
"""


@pytest.mark.parametrize("absent", BROWSER_ABSENT)
def test_the_package_imports_and_runs_without(absent):
    done = subprocess.run([sys.executable, "-c", SCRIPT, absent, *BROWSER_MODULES],
                          capture_output=True, text=True, cwd=REPO_ROOT, timeout=120)
    if done.returncode:
        pytest.fail(done.stderr[-2000:], pytrace=False)
