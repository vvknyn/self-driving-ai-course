"""The playground's preview registry: how each graded exercise runs the learner's code without grading it.

It mirrors `grade.EXERCISES`, one entry per exercise id.  `Preview.kind` says what the learner sees:
  drive  `build(learner_obj) -> Car`, which the playground drives on `road` and streams step by step
  stats  `build(learner_fn, tel) -> {label: number}`, the learner's function applied to a drive on `road`
  image  `build(learner_fn, frame) -> uint8[H, W, 3]`, the learner's function drawn on a camera frame
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

Kind = Literal["drive", "stats", "image"]


@dataclass(frozen=True)
class Preview:
    kind: Kind
    build: Callable[..., Any]
    road: str = ""  # drive and stats: the scenario the preview drives; image: empty (the caller picks the frame)


PREVIEWS: dict[str, Preview] = {}


def register(exercise_id: str, kind: Kind, road: str = "") -> Callable:
    """Decorator: store the function as the preview of `exercise_id`; a second preview for one id is a bug."""

    def store(build):
        if exercise_id in PREVIEWS:
            raise ValueError(f"exercise {exercise_id!r} already has a preview")
        PREVIEWS[exercise_id] = Preview(kind, build, road)
        return build

    return store


def get(exercise_id: str) -> Preview:
    if exercise_id not in PREVIEWS:
        raise ValueError(f"unknown exercise {exercise_id!r}; known: {', '.join(sorted(PREVIEWS))}")
    return PREVIEWS[exercise_id]


from . import units  # noqa: E402,F401  (importing it registers every exercise's preview)
