"""Meta-check of the grader: for every exercise of every ready unit, the untouched skeleton must FAIL and
the reference solution must PASS.  This is what stops a lab from showing a green tick it did not earn.

    python scripts/verify_units.py [--root DIR]

The skeleton of an exercise is built from the committed lab.ipynb: the code cells tagged `setup` or
`exercise:*`, executed in order in one namespace up to and including the LAST cell tagged with the
exercise's id (an exercise may span cells), then the exercise's declared `name` is taken from it.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import nbformat  # noqa: E402

from _course import REPO, load_solution, load_yaml, syllabus_units  # noqa: E402
from zero2fsd.grade import check  # noqa: E402
from zero2fsd.grade._report import EXERCISES  # noqa: E402


def _tags(cell) -> list[str]:
    return cell.metadata.get("tags", [])


def skeleton_for(notebook, exercise_id: str, name: str):
    """Execute the notebook's setup and exercise cells up to this exercise; return the object called `name`."""
    cells = [c for c in notebook.cells if c.cell_type == "code" and ("setup" in _tags(c) or any(t.startswith("exercise:") for t in _tags(c)))]
    last = max((i for i, c in enumerate(cells) if f"exercise:{exercise_id}" in _tags(c)), default=None)
    if last is None:
        raise LookupError(f"no notebook cell is tagged exercise:{exercise_id}")
    namespace: dict = {"__name__": "__lab__"}
    for cell in cells[: last + 1]:
        exec(compile(cell.source, f"<lab cell {cell.get('id', '?')}>", "exec"), namespace)
    if name not in namespace:
        raise LookupError(f"the cells up to exercise:{exercise_id} do not define {name!r}")
    return namespace[name]


def _graded(exercise_id, target) -> bool:
    with contextlib.redirect_stdout(io.StringIO()):
        return check(exercise_id, target, show=False).passed


def verify_unit(root: Path, unit: dict) -> list[str]:
    folder = root / "course" / unit["folder"]
    try:
        exercises = load_yaml(folder / "unit.yaml")["exercises"]
        notebook, solution = nbformat.read(folder / "lab.ipynb", as_version=4), load_solution(root, unit["id"])
    except Exception as err:  # a ready unit that cannot even be loaded is a violation, not a crash
        return [f"{unit['id']}: cannot load the unit: {type(err).__name__}: {err}"]
    violations = []
    for ex in exercises:
        where = f"{unit['id']} {ex['id']} ({ex['name']})"
        if ex["id"] not in EXERCISES:
            violations.append(f"{where}: the grader has no exercise with this id")
            continue
        try:
            skeleton = skeleton_for(notebook, ex["id"], ex["name"])
        except Exception as err:  # a lab whose cells do not even run is broken, whatever the reason
            violations.append(f"{where}: cannot build the skeleton: {type(err).__name__}: {err}")
            skeleton = None
        if skeleton is not None and _graded(ex["id"], skeleton):
            violations.append(f"{where}: the untouched skeleton PASSES the grader")
        if not hasattr(solution, ex["name"]):
            violations.append(f"{where}: solutions/course/{unit['id']}/solution.py does not define {ex['name']!r}")
        elif not _graded(ex["id"], getattr(solution, ex["name"])):
            violations.append(f"{where}: the reference solution FAILS the grader")
    return violations


def verify(root: Path) -> list[str]:
    return [v for unit in syllabus_units(root, status="ready") for v in verify_unit(root, unit)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO)
    violations = verify(parser.parse_args(argv).root)
    for v in violations:
        print(f"VIOLATION {v}")
    print(f"{len(violations)} violation(s)" if violations else "all exercises: skeleton fails, solution passes")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
