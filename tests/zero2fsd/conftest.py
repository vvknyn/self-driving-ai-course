import os

os.environ.setdefault("MPLBACKEND", "Agg")  # before any matplotlib import: tests never open a window

import numpy as np
import pytest

from zero2fsd.car.runner import Telemetry


@pytest.fixture
def make_tel():
    """Hand-built telemetry: `make_tel(lat=[...], progress=[...])`, every other array zero unless given."""

    def make(lat, progress, route_length=200.0, scenario="straight", **arrays):
        n = len(lat)
        zeros = {name: np.zeros(n) for name in ("x", "y", "yaw", "speed", "steer", "accel", "head_err", "est_offset", "est_heading")}
        fields = {
            "t": np.arange(n) * 0.05, **zeros, "lat": np.asarray(lat, float), "progress_m": np.asarray(progress, float),
            "est_valid": np.ones(n, bool), "off_road": np.zeros(n, bool),
        }
        fields.update({k: np.asarray(v) for k, v in arrays.items()})
        return Telemetry(**fields, scenario=scenario, completed=False, route_length=route_length)

    return make
