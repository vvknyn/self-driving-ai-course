"""What a graded exercise is: a checker that raises `Failure` on the first thing wrong.

`exercise(...)` registers a checker in `EXERCISES` and turns it into `(target, show, on_step) -> Result`.
`call` and `learner_code` are the one place learner code is run, so an exception in it becomes a
readable red result instead of a traceback and a missing implementation can never count as a pass.
"""
from __future__ import annotations

import functools
import math
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable, NamedTuple

import numpy as np

from ..car.runner import StepHook

REPORT: ContextVar[bool] = ContextVar("REPORT", default=True)  # False while `check(..., report=False)` runs: nothing the grader would print
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


EXERCISES: dict[str, Callable[[Any, bool, StepHook | None], Result]] = {}


SIGNIFICANT_DIGITS = 6
MAX_DIGITS = 17  # a float64 is told apart from its neighbour by 17 significant digits


def fmt(value, digits: int = SIGNIFICANT_DIGITS) -> str:
    if isinstance(value, np.generic):  # a NumPy scalar such as np.int64(872): show the plain number
        value = value.item()
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=digits - 2, threshold=8, edgeitems=3)
    if type(value) in (tuple, list):  # a plain container, such as a (forward, left) point: format each item
        items = ", ".join(fmt(v, digits) for v in value)
        return f"({items})" if isinstance(value, tuple) else f"[{items}]"
    if isinstance(value, float | np.floating):
        return f"{value:.{digits}g}"
    return value if isinstance(value, str) else repr(value)


def fmt_pair(expected, got) -> tuple[str, str]:
    """`fmt` of an expected/got pair of NUMBERS (or arrays or tuples of them; a str raises TypeError: format after, never before, this).
    When two different numbers read alike (same text, or the same 6 significant digits, whatever container each is in), print digits
    until they differ and give the gap with `got`."""
    if isinstance(expected, str) or isinstance(got, str):
        raise TypeError("fmt_pair takes numbers: a pre-formatted string cannot be told apart from its pair")
    try:
        want, have = np.asarray(expected, float), np.asarray(got, float)
        gap = float(np.max(np.abs(want - have))) if want.shape == have.shape else 0.0
    except (TypeError, ValueError):  # not numbers (None, a Car): nothing to tell apart
        return fmt(expected), fmt(got)
    if not gap > 0:  # equal, nan or a shape mismatch the grader names itself
        return fmt(expected), fmt(got)
    plain = lambda a: a if a.ndim else float(a)  # noqa: E731  the same text for the same numbers, whatever container each came in
    for digits in range(SIGNIFICANT_DIGITS, MAX_DIGITS + 1):
        alike = (fmt(expected, digits) == fmt(got, digits) or fmt(plain(want), digits) == fmt(plain(have), digits)
                 or [f"{x:.{digits}g}" for x in want.ravel()] == [f"{x:.{digits}g}" for x in have.ravel()])
        if not alike:
            break
    shown = fmt(expected, digits), fmt(got, digits)
    return shown if digits == SIGNIFICANT_DIGITS else (shown[0], f"{shown[1]} (off by {gap:.2g})")


def zero_below(value, tol: float):
    """`value` with every number of magnitude below `tol` replaced by 0, in the same container (array, tuple or list) it came in."""
    if isinstance(value, np.ndarray) and value.dtype.kind == "f":
        return np.where(np.abs(value) < tol, 0.0, value)
    if type(value) in (tuple, list):
        return type(value)(zero_below(v, tol) for v in value)
    if isinstance(value, float | np.floating) and abs(value) < tol:
        return 0.0
    return value


def fmt_gap(value, tol: float, spec: str | None = None) -> str:
    """A worst or largest error for a message, with float noise below the grader's tolerance `tol` shown as 0; `spec` is a format spec (default: `fmt`)."""
    floored = zero_below(float(value), tol)
    return fmt(floored) if spec is None else format(floored, spec)


def fmt_expected_got(expected, got) -> tuple[str, str]:
    """What a report prints for an expected/got pair: numbers go through `fmt_pair`; text is a description, printed as it is."""
    if isinstance(expected, str) or isinstance(got, str):
        return fmt(expected), fmt(got)
    return fmt_pair(expected, got)


def matching_mistakes(got, right, candidates, tol: float) -> list:
    """The labels of the known mistakes whose output `got` equals within `tol`.  A mistake whose own output equals `right` within `tol`
    (at yaw 0, R is its transpose) is skipped: it gives the right answer there, so naming it would accuse code that is not wrong.
    `candidates` is an iterable of (label, output)."""
    got, right = np.asarray(got, float), np.asarray(right, float)
    close = lambda a, b: a.shape == b.shape and bool(np.allclose(a, b, rtol=0.0, atol=tol))  # noqa: E731
    outputs = [(label, np.asarray(wrong, float)) for label, wrong in candidates]
    return [label for label, wrong in outputs if close(got, wrong) and not close(right, wrong)]


def is_real(value) -> bool:
    """A finite number that is not a bool."""
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


BORDERLINE_FRACTION = 0.02  # a verdict within this fraction of its limit can flip between computers
BORDERLINE_NOTE = "Borderline: tiny differences between computers can flip this. Aim for a clearer margin."


def margin(value: float, limit: float) -> float:
    """How far `value` is from `limit`, as a fraction of the limit."""
    return abs(value - limit) / abs(limit)


def borderline(value: float, threshold: float) -> str | None:
    """The borderline sentence when `value` is under BORDERLINE_FRACTION of `threshold` away from it, else None."""
    return BORDERLINE_NOTE if margin(value, threshold) < BORDERLINE_FRACTION else None


def vs_limit(value: float, limit: float, unit: str = "", decisive: bool = True) -> str:
    """`(limit 0.5 m, margin 25 %)`, then the borderline sentence when the margin is that thin and `value` decided the verdict."""
    percent = 100 * margin(value, limit)
    shown = f"{percent:.1f}" if percent < 10 else f"{percent:.0f}"
    text = f"(limit {limit:g}{' ' + unit if unit else ''}, margin {shown} %)"
    note = borderline(value, limit) if decisive else None
    return f"{text}. {note}" if note else text


class Threshold(NamedTuple):
    """A verdict that compares a simulation-derived number with a limit.  A grader module that calls `vs_limit` declares
    one per such verdict in its module-level `THRESHOLDS = {label: Threshold(...)}`; tests/zero2fsd/test_margins.py
    checks that the reference solution of `unit` clears each `limit` by a margin."""
    limit: float
    passes: Callable[[float, float], bool]  # operator.lt or operator.ge: the side of `limit` that passes
    unit: str  # whose reference solution is measured
    measure: Callable[[Any], float]  # the number a reference solution module (for `unit`) scores


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


def fresh(value):
    """`value` with every ndarray in it copied (a list, tuple or dict is rebuilt around the copies)."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if type(value) in (list, tuple):
        return type(value)(fresh(v) for v in value)
    if type(value) is dict:
        return {key: fresh(v) for key, v in value.items()}
    return value


def call(fn, *args, given: str, **kwargs):
    """Run a learner function on copies of its arrays (it may edit them in place; the grader's own stay as they were) and return its result.
    None means the learner forgot `return` (no grader's function may answer None)."""
    with learner_code(given):
        result = fn(*fresh(args), **fresh(kwargs))
    if result is None:
        raise Failure(f"your function {getattr(fn, '__name__', repr(fn))} returned None: it needs a `return` line giving its answer", given=given)
    return result[()] if isinstance(result, np.ndarray) and result.ndim == 0 else result  # np.where(..) on a scalar gives a 0-d array: graders get its NumPy scalar, which is_real and fmt already handle


LESSON_TITLES: dict[str, str] = {}  # unit id -> title, filled by grade/__init__ from each unit module's LESSON (a test keeps them equal to unit.yaml)


def lesson_titles(unit_modules) -> dict[str, str]:
    """{unit id: title} from each module's `LESSON = (id, title)`; two modules claiming one id is a bug."""
    titles: dict[str, str] = {}
    for module in unit_modules:
        unit_id, title = module.LESSON
        if unit_id in titles:
            raise ValueError(f"duplicate lesson id {unit_id!r}")
        titles[unit_id] = title
    return titles


def exercise_label(exercise_id: str) -> str:
    """`0.1.2.a` -> `Exercise A (Images are arrays)`: what the learner sees instead of the dotted id."""
    unit, _, suffix = exercise_id.rpartition(".")
    return f"Exercise {suffix.upper()} ({LESSON_TITLES[unit]})"


_SHOWN_LABELS = {"given": "input", "expected": "expected", "got": "got"}


def _lines(failure: Failure, label, checked, hint) -> list[str]:
    shown = {key: fmt(value) for key, value in failure.shown.items()}
    if "expected" in shown and "got" in shown:
        shown["expected"], shown["got"] = fmt_expected_got(failure.shown["expected"], failure.shown["got"])
    fields = [("checked", fmt(failure.checked or checked)),
              *((name, shown[key]) for key, name in _SHOWN_LABELS.items() if key in shown),
              ("hint", fmt(hint))]
    return [f"\N{BALLOT X} {label}: {failure.headline}", *(f"    {name + ':':<10}{text}" for name, text in fields)]


def exercise(exercise_id: str, *, checked: str, section: str):
    """Register a checker `(target, show, on_step) -> str` (what passed); it raises `Failure` for the first problem.

    Reports name the exercise by `exercise_label`, never the dotted id; `section` is the lecture section the hint points to."""

    def register(checker):
        @functools.wraps(checker)
        def graded(target, show, on_step):
            label = exercise_label(exercise_id)  # looked up per call: the unit's LESSON is filed only after its module has imported
            hint = f'lecture "{LESSON_TITLES[exercise_id.rpartition(".")[0]]}", section "{section}"'
            try:
                message = checker(target, show, on_step)
            except Failure as failure:
                return Result(False, _lines(failure, label, checked, hint))
            return Result(True, [f"\N{CHECK MARK} {label}: {message}"])

        if exercise_id in EXERCISES:
            raise ValueError(f"duplicate exercise id {exercise_id!r}")
        EXERCISES[exercise_id] = graded
        return graded

    return register
