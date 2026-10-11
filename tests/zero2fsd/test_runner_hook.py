import dataclasses

import numpy as np
import pytest

from zero2fsd.car import Car, Telemetry, run
from zero2fsd.car.runner import COLUMNS


def assert_same_telemetry(a: Telemetry, b: Telemetry) -> None:
    for field in dataclasses.fields(Telemetry):
        x, y = getattr(a, field.name), getattr(b, field.name)
        assert np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y, field.name


def test_columns_are_the_telemetry_arrays_in_order():
    names = [f.name for f in dataclasses.fields(Telemetry)]
    assert COLUMNS == tuple(names[: len(COLUMNS)])
    assert len(COLUMNS) == 14 and names[len(COLUMNS)] == "scenario"  # every array, and nothing else


@pytest.mark.parametrize("scenario", ["straight", "curvy"])  # curvy ends off the road, straight ends complete
def test_the_hook_streams_exactly_the_telemetry(scenario):
    seen = []
    hooked = run(Car(), scenario, on_step=seen.append)
    assert len(seen) == len(hooked.t)
    assert all(isinstance(row, tuple) and len(row) == len(COLUMNS) for row in seen)
    for i, name in enumerate(COLUMNS):
        assert np.array_equal(np.array([row[i] for row in seen]), getattr(hooked, name)), name
    assert_same_telemetry(hooked, run(Car(), scenario))  # the hook does not change the run


def test_the_hook_is_called_once_per_step_on_gentle():
    calls = []
    tel = run(Car(), "gentle", on_step=calls.append)
    assert tel.completed and len(calls) == len(tel.t) == 1778  # measured Oct 11 2026 on the provided car


def test_an_exception_in_the_hook_propagates():
    def boom(row):
        raise RuntimeError("hook failed")

    with pytest.raises(RuntimeError, match="hook failed"):
        run(Car(), "straight", on_step=boom)
