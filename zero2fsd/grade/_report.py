"""What a graded exercise is: a checker that raises `Failure` on the first thing wrong.

`exercise(...)` registers a checker in `EXERCISES` and turns it into `(target, show) -> Result`.
`call` and `learner_code` are the one place learner code is run, so an exception in it becomes a
readable red result instead of a traceback and a missing implementation can never count as a pass.
"""
from __future__ import annotations

import functools
import math
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable

import numpy as np

NOT_IMPLEMENTED = "not implemented yet \N{EM DASH} write your code in the cell above"


@dataclass(frozen=True)
class Result:
    passed: bool
    details: list[str]  # details[0] is the one-line summary


class Failure(Exception):
    """The first thing wrong with a learner's answer.  `shown` may hold `given`, `expected` and `got` (any
    values, None included); a field that is left out is not printed."""

    def __init__(self, headline, *, checked=None, **shown):
        super().__init__(headline)
        self.headline, self.checked, self.shown = headline, checked, shown


EXERCISES: dict[str, Callable[[Any, bool], Result]] = {}


def fmt(value) -> str:
    if isinstance(value, np.generic):  # a NumPy scalar such as np.int64(872): show the plain number
        value = value.item()
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=4, threshold=8, edgeitems=3)
    if isinstance(value, float | np.floating):
        return f"{value:.6g}"
    return value if isinstance(value, str) else repr(value)


def is_real(value) -> bool:
    """A finite number that is not a bool."""
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


@contextmanager
def learner_code(given: str):
    """Run learner code: NotImplementedError and any other exception become a `Failure` naming the input."""
    try:
        yield
    except NotImplementedError:
        raise Failure(NOT_IMPLEMENTED) from None
    except Exception as err:
        where = traceback.extract_tb(err.__traceback__)[-1]
        raise Failure(f"your code raised {type(err).__name__}: {err} (line {where.lineno}, in {where.name})", given=given) from err


def call(fn, *args, given: str, **kwargs):
    with learner_code(given):
        return fn(*args, **kwargs)


_SHOWN_LABELS = {"given": "input", "expected": "expected", "got": "got"}


def _lines(failure: Failure, exercise_id, checked, hint) -> list[str]:
    fields = [("checked", failure.checked or checked),
              *((label, failure.shown[key]) for key, label in _SHOWN_LABELS.items() if key in failure.shown),
              ("hint", hint)]
    return [f"\N{BALLOT X} {exercise_id}: {failure.headline}", *(f"    {name + ':':<10}{fmt(value)}" for name, value in fields)]


def exercise(exercise_id: str, *, checked: str, hint: str):
    """Register a checker `(target, show) -> str` (what passed); it raises `Failure` for the first problem."""

    def register(checker):
        @functools.wraps(checker)
        def graded(target, show):
            try:
                message = checker(target, show)
            except Failure as failure:
                return Result(False, _lines(failure, exercise_id, checked, hint))
            return Result(True, [f"\N{CHECK MARK} {exercise_id}: {message}"])

        if exercise_id in EXERCISES:
            raise ValueError(f"duplicate exercise id {exercise_id!r}")
        EXERCISES[exercise_id] = graded
        return graded

    return register
