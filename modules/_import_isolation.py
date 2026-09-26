"""Import isolation for numeric module directories (00_ml_gym, 01_camera_geometry, …).

Each module test suite calls ``prepare_module_imports`` at import time so that
duplicate top-level names (``config``, ``dataset``, ``losses``, …) do not leak
across module boundaries when tests run in one process.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Top-level module basenames that collide across LearnFSD modules.
_COLLIDING_NAMES = frozenset(
    {
        "config",
        "dataset",
        "model",
        "losses",
        "metrics",
        "train",
        "error_gallery",
        "break_it_fix_it",
        "camera_model",
        "extrinsics",
        "ipm",
        "stitch",
        "pitch_sensitivity",
        "calibrate_rig",
        "ipm_transform",
    }
)


def prepare_module_imports(module_dir: str | Path) -> Path:
    """Prepare ``sys.path`` and ``sys.modules`` for importing one module directory.

    Args:
        module_dir: Absolute or repo-relative path to a single ``modules/XX_*`` folder.

    Returns:
        Resolved absolute path to ``module_dir``.
    """
    resolved = Path(module_dir).resolve()
    modules_root = resolved.parent

    to_drop: list[str] = []
    for name, mod in list(sys.modules.items()):
        if name not in _COLLIDING_NAMES:
            continue
        mod_file = getattr(mod, "__file__", None)
        if mod_file is None:
            continue
        mod_path = Path(mod_file).resolve()
        if mod_path.is_relative_to(modules_root) and not mod_path.is_relative_to(resolved):
            to_drop.append(name)

    for name in to_drop:
        del sys.modules[name]

    module_str = str(resolved)
    if module_str in sys.path:
        sys.path.remove(module_str)
    sys.path.insert(0, module_str)

    return resolved


def repo_root() -> Path:
    """Return repository root (parent of ``modules/``)."""
    return Path(__file__).resolve().parent.parent
