"""The playground's preview registry: how each graded exercise runs the learner's code without grading it.

It mirrors `grade.EXERCISES`, one entry per exercise id.  `Preview.kind` says what the learner sees:
  drive  `build(learner_obj) -> Car`, which the playground drives on `road` and streams step by step
  stats  `build(learner_fn, tel) -> {label: number}`, the learner's function applied to a drive on `road`;
         a stat that is an error or a gap is returned as `Error(value, tol)`, and `register` shows it as 0 below `tol`
  image  `build(learner_fn, frame) -> uint8[H, W, 3]`, the learner's function drawn on a camera frame
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Any, Callable, Literal

from ..grade._report import zero_below

Kind = Literal["drive", "stats", "image"]


@dataclass(frozen=True)
class Preview:
    kind: Kind
    build: Callable[..., Any]
    road: str = ""  # drive and stats: the scenario the preview drives; image: empty (the caller picks the frame)


class Error(float):
    """A stats number that measures how far the learner's answer is from the truth; below `tol` it is float noise, not a value."""

    tol: float

    def __new__(cls, value: float, tol: float):
        self = super().__new__(cls, value)
        self.tol = tol
        return self


def settled(stats: dict[str, float]) -> dict[str, float]:
    """`stats` with every `Error` below its tolerance shown as 0; every other number (counts, means, plain values) as it is."""
    return {label: zero_below(float(v), v.tol) if isinstance(v, Error) else v for label, v in stats.items()}


PREVIEWS: dict[str, Preview] = {}


def register(exercise_id: str, kind: Kind, road: str = "") -> Callable:
    """Decorator: store the function as the preview of `exercise_id`; a second preview for one id is a bug."""

    def store(build):
        if exercise_id in PREVIEWS:
            raise ValueError(f"exercise {exercise_id!r} already has a preview")
        PREVIEWS[exercise_id] = Preview(kind, (lambda *args: settled(build(*args))) if kind == "stats" else build, road)
        return build

    return store


def get(exercise_id: str) -> Preview:
    if exercise_id not in PREVIEWS:
        raise ValueError(f"unknown exercise {exercise_id!r}; known: {', '.join(sorted(PREVIEWS))}")
    return PREVIEWS[exercise_id]


from . import units  # noqa: E402

for _unit in pkgutil.iter_modules(units.__path__):  # importing a unit registers its previews
    importlib.import_module(f"{units.__name__}.{_unit.name}")
