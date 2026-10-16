"""`check(exercise_id, fn_or_box)`: grade the learner's own code against independently computed answers."""
from __future__ import annotations

import importlib
import pkgutil

import numpy as np

from ..car.runner import StepHook
from . import units
from ._report import EXERCISES, LESSON_TITLES, REPORT, Result, fmt, fmt_expected_got, lesson_titles, zero_below

# importing a unit registers its exercises; its LESSON files the unit's title for the reports
LESSON_TITLES.update(lesson_titles(importlib.import_module(f"{units.__name__}.{unit.name}") for unit in pkgutil.iter_modules(units.__path__)))

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"


def check(exercise_id: str, fn_or_box, show: bool = True, on_step: StepHook | None = None,
          report: bool = True) -> Result:
    """Run one exercise's checks on the learner's function (or Car), print the report unless `report` is False, return it.

    `on_step` is called with each telemetry row while a graded drive runs; exercises that do not drive never call it."""
    if exercise_id not in EXERCISES:
        raise ValueError(f"unknown exercise {exercise_id!r}; known: {', '.join(sorted(EXERCISES))}")
    token = REPORT.set(report)  # the graded drives print their metrics tables too: one switch for everything the grader prints
    try:
        result = EXERCISES[exercise_id](fn_or_box, show, on_step)
    finally:
        REPORT.reset(token)
    if report:
        head, *rest = result.details
        print(f"{GREEN if result.passed else RED}{head}{RESET}", *rest, sep="\n")
    return result


def practice(what: str, got, want, tol: float = 1e-6) -> None:
    """Ungraded self-check for a lab's practice problem: `got` stays None until you answer; prints the verdict.

    A passing check shows an entry within `tol` of zero as 0 (float noise such as 6e-17 is not a value) and says so; a failing one shows the raw numbers."""
    if got is None:
        print(f"{what}: not answered yet \N{EM DASH} replace None with your code and run the cell again")
        return
    try:
        right = np.shape(got) == np.shape(want) and bool(np.allclose(got, want, rtol=0.0, atol=tol))
    except TypeError:  # not a number or array of numbers
        right = False
    if right:
        shown, raw = fmt(zero_below(got, tol)), fmt(got)
        print(f"{GREEN}\N{CHECK MARK} {what}: {shown}{f' (shown as 0: all within {tol:g})' if shown != raw else ''}{RESET}")
    else:
        expected, actual = fmt_expected_got(want, got)
        print(f"{RED}\N{BALLOT X} {what}: expected {expected}, got {actual}{RESET}")


__all__ = ["Result", "check", "practice"]
