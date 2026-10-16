"""Previews of module 1.1's "The view from the driver's seat" (grade/units/u1_1_2.py): the learner's transform, run for a look."""
from __future__ import annotations

import numpy as np

from ...grade.units.u1_1_2 import renderer_gap
from ...sim import BicycleState, Camera, make_scenario
from .. import register

PICTURES = 18  # pictures sampled along the canned drive: enough to tell a transform from another, quick enough for the browser


def differing_pictures(fn, tel, count: int = PICTURES) -> tuple[int, int]:
    """(pictures, pixels) that differ between the simulator's transform and `fn`, over `count` evenly spaced steps of `tel`."""
    road, ours, theirs = make_scenario(tel.scenario).road, Camera(), Camera(world_to_ego=fn)
    pictures = pixels = 0
    for k in np.linspace(0, len(tel.t) - 1, count).astype(int):
        state = BicycleState(float(tel.x[k]), float(tel.y[k]), float(tel.yaw[k]), float(tel.speed[k]), 0.0)
        differ = int((ours.render(road, state) != theirs.render(road, state)).any(axis=-1).sum())
        pictures, pixels = pictures + (differ > 0), pixels + differ
    return pictures, pixels


@register("1.1.2.a", "stats", road="gentle")
def _world_to_ego(fn, tel):
    return {"largest difference from the renderer's transform (m)": renderer_gap(fn)}


@register("1.1.2.b", "stats", road="gentle")
def _renderer(fn, tel):
    pictures, pixels = differing_pictures(fn, tel)
    return {f"pictures that differ from the simulator's (of {PICTURES} sampled)": float(pictures), "pixels that differ": float(pixels)}
