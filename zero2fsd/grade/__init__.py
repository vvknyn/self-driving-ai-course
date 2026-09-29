"""`check(exercise_id, fn_or_box)`: grade the learner's own code against independently computed answers."""
from __future__ import annotations

import importlib
import pkgutil

import numpy as np

from . import units
from ._report import EXERCISES, Result, fmt

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


def practice(what: str, got, want, tol: float = 1e-6) -> None:
    """Ungraded self-check for a lab's practice problem: `got` stays None until you answer; prints the verdict."""
    if got is None:
        print(f"{what}: not answered yet \N{EM DASH} replace None with your code and run the cell again")
        return
    try:
        right = np.shape(got) == np.shape(want) and bool(np.allclose(got, want, rtol=0.0, atol=tol))
    except TypeError:  # not a number or array of numbers
        right = False
    print(f"{GREEN}\N{CHECK MARK} {what}: {fmt(got)}{RESET}" if right
          else f"{RED}\N{BALLOT X} {what}: expected {fmt(want)}, got {fmt(got)}{RESET}")


__all__ = ["Result", "check", "practice"]
