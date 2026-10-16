"""One preview per graded exercise of lesson 1.1.5 (grade/units/u1_1_5.py): the learner's mount and ray, measured instead of graded."""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from ... import geom
from ...grade._report import Failure
from ...grade.units.u1_1_5 import N_PICTURES, ground_answers, ground_gaps, mount, mount_pair, pictures_that_differ, pixel_gap
from ...sim import Camera
from .. import register

ROAD = "gentle"  # the lab's own drive: a mount that is wrong shows on its bend


@contextmanager
def plain_errors():
    """The grader's reasons (a learner's exception, a wrong shape) as a TypeError whose message is the plain-language headline."""
    try:
        yield
    except Failure as failure:
        raise TypeError(failure.headline) from None


@register("1.1.5.a", "stats", road=ROAD)
def _mount_stats(fn, tel):
    with plain_errors():
        T = mount(fn)
    camera = Camera()
    seen = geom.ray_to_ground(T, geom.pixel_ray(camera, camera.cx, 150))
    return {"ground distance seen in row 150 (m)": float(seen[0]), "largest pixel difference from the renderer (px)": pixel_gap(T)}


@register("1.1.5.b", "stats", road=ROAD)
def _swapped_mount_stats(target, tel):
    with plain_errors():
        world_to_ego, fn = mount_pair(target)
        camera = Camera(world_to_ego=world_to_ego, T_ego_cam=mount(fn))
        differ = pictures_that_differ(camera, tel.scenario)
    return {f"pictures that differ (of {N_PICTURES})": float(differ)}


@register("1.1.5.c", "stats", road=ROAD)
def _ray_stats(fn, tel):
    with plain_errors():
        gaps = ground_gaps(ground_answers(fn))
    return {"largest ground difference (m)": float(np.max(gaps))}
