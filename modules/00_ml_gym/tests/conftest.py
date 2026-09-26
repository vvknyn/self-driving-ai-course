"""Pytest path setup for Module 00.

Loaded before this directory's tests. Isolates top-level names such as
``config`` from Module 01 when both suites run in one process.
"""

from __future__ import annotations

import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parents[1]
_MODULES_ROOT = _MODULE_DIR.parent
if str(_MODULES_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULES_ROOT))

from _import_isolation import prepare_module_imports

prepare_module_imports(_MODULE_DIR)
