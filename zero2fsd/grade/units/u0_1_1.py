"""0.1.1 Meet your car: summarising lateral error."""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral

import numpy as np

from .._report import Failure, call, exercise, fmt, is_real

TOL = 1e-9


def _loop_stats(errors: list[float]) -> dict[str, float]:
    n = len(errors)
    return {
        "mean_abs": sum(abs(e) for e in errors) / n,
        "max_abs": max(abs(e) for e in errors),
        "rms": math.sqrt(sum(e * e for e in errors) / n),
    }


@exercise(
    "0.1.1.a",
    checked=f"mean_abs, max_abs and rms of seeded arrays (negatives included) against a python-loop computation, to {TOL}",
    hint='lecture 0.1.1, section "Summarising lane error"',
)
def _lane_error_stats(fn, show):
    rng = np.random.default_rng(101)
    arrays = [rng.normal(0.0, sd, n) for sd, n in ((0.3, 50), (0.6, 200), (1.0, 7), (0.05, 1000), (0.8, 2))]
    arrays.append(np.array([-0.4]))  # one negative sample: |e|, not e
    for lat in arrays:
        want, given = _loop_stats(lat.tolist()), f"lat_err = {fmt(lat)}"
        got = call(fn, lat.copy(), given=given)
        if not isinstance(got, Mapping):
            raise Failure(f"returned {type(got).__name__}, expected a dict", given=given, expected=f"keys {sorted(want)}", got=got)
        for key, value in want.items():
            if not is_real(got.get(key)) or not math.isclose(got[key], value, rel_tol=TOL, abs_tol=TOL):
                raise Failure(f"{key} is wrong", given=given, expected=f"{key} = {fmt(value)}", got=f"{key} = {fmt(got.get(key))}")
    return f"all {len(arrays)} arrays match the python-loop computation"


@exercise(
    "0.1.1.b",
    checked="the count of steps with |error| strictly greater than the threshold, at the default 0.9 and at custom thresholds",
    hint='lecture 0.1.1, section "Counting steps off the lane"',
)
def _steps_off_lane(fn, show):
    rng = np.random.default_rng(102)
    edge = np.array([0.9, -0.9, 0.9000001, -0.95, 0.2, 0.0])  # values exactly at the threshold are not off the lane
    cases = [(edge, None), (edge, 0.5)] + [(rng.normal(0.0, 0.7, n), thr) for n, thr in ((30, None), (200, 0.3), (1, None))]
    for lat, threshold in cases:
        limit = 0.9 if threshold is None else threshold
        want = sum(1 for e in lat.tolist() if abs(e) > limit)
        kwargs = {} if threshold is None else {"threshold": threshold}
        given = f"lat_err = {fmt(lat)}" + ("" if threshold is None else f", threshold={threshold}")
        got = call(fn, lat.copy(), given=given, **kwargs)
        if not isinstance(got, Integral) or isinstance(got, bool):
            raise Failure(f"returned {type(got).__name__}, expected an integer count", given=given, expected=want, got=got)
        if got != want:
            raise Failure(f"counted {got} steps off the lane, not {want}", given=given, expected=want, got=got)
    return f"all {len(cases)} cases counted correctly (exactly-at-threshold values are not off the lane)"
