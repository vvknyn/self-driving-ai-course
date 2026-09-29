"""Compile each unit's percent-format lab.py into the committed lab.ipynb (deterministic).

    python scripts/build_labs.py [--root DIR] [--check] [UNIT_DIR ...]

`--check` writes nothing and exits 1 if any lab.ipynb is missing or stale.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbformat import v4

from _course import REPO, parse_lab

KERNEL = {"display_name": "Python 3", "language": "python", "name": "python3"}


def compile_lab(lab: Path) -> str:
    """The notebook JSON for a lab.py; cell ids are c000, c001, ... so rebuilding never churns them."""
    make = {"code": v4.new_code_cell, "markdown": v4.new_markdown_cell}
    cells = []
    for index, cell in enumerate(parse_lab(lab.read_text(encoding="utf-8"))):
        metadata = {"tags": cell.tags} if cell.tags else {}
        cells.append(make[cell.kind](cell.source, id=f"c{index:03d}", metadata=metadata))
    notebook = v4.new_notebook(cells=cells, metadata={"kernelspec": KERNEL, "language_info": {"name": "python"}})
    nbformat.validate(notebook)
    return nbformat.writes(notebook) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("units", nargs="*", type=Path)
    args = parser.parse_args(argv)
    labs = [d / "lab.py" for d in args.units] or sorted((args.root / "course").glob("*/lab.py"))
    stale = []
    for lab in labs:
        target, built = lab.with_suffix(".ipynb"), compile_lab(lab)
        if args.check:
            if not target.exists() or target.read_text(encoding="utf-8") != built:
                stale.append(target)
        else:
            target.write_text(built, encoding="utf-8")
            print(f"built {target.relative_to(args.root) if target.is_relative_to(args.root) else target}")
    for target in stale:
        print(f"stale: {target} (run scripts/build_labs.py)")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
