"""`check(exercise_id, fn_or_box)`: grade the learner's own code against independently computed answers."""
from __future__ import annotations

import importlib
import pkgutil

from . import units
from ._report import EXERCISES, Result

for _unit in pkgutil.iter_modules(units.__path__):  # importing a unit registers its exercises
    importlib.import_module(f"{units.__name__}.{_unit.name}")

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"


def check(exercise_id: str, fn_or_box, show: bool = True) -> Result:
    """Run one exercise's checks on the learner's function (or Car), print the verdict, return it."""
    if exercise_id not in EXERCISES:
        raise ValueError(f"unknown exercise {exercise_id!r}; known: {', '.join(sorted(EXERCISES))}")
    result = EXERCISES[exercise_id](fn_or_box, show)
    head, *rest = result.details
    print(f"{GREEN if result.passed else RED}{head}{RESET}", *rest, sep="\n")
    return result


__all__ = ["Result", "check"]
