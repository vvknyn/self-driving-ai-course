"""Regenerate lecture figures: every unit folder with a figures.py exposing make(out_dir) gets `figures/`.

    python scripts/make_figures.py [--root DIR] [UNIT_DIR ...]

Figures must be deterministic (seeded), so rerunning leaves the committed PNGs byte-identical.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # before any figures.py imports matplotlib

from _course import REPO  # noqa: E402


def make_unit_figures(unit_dir: Path) -> None:
    spec = importlib.util.spec_from_file_location(f"figures_{unit_dir.name.replace('.', '_').replace('-', '_')}", unit_dir / "figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out_dir = unit_dir / "figures"
    out_dir.mkdir(exist_ok=True)
    module.make(out_dir=out_dir)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("units", nargs="*", type=Path)
    args = parser.parse_args(argv)
    for unit_dir in args.units or sorted(p.parent for p in (args.root / "course").glob("*/figures.py")):
        make_unit_figures(unit_dir)
        print(f"figures for {unit_dir.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
