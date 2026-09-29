"""Metrics, the headline driving score, and the fixed-width table both are shown in."""
from __future__ import annotations

import numpy as np

from .car.runner import Telemetry

OFF_LANE_LAT = 0.9  # m: half a lane's slack; beyond it the car is counted off its lane


def metrics(tel: Telemetry) -> dict:
    """completion (0-1), mean/max |lat| (m), steps off lane, RMS steer rate (rad/s)."""
    abs_lat = np.abs(tel.lat)
    steer_rate = np.diff(tel.steer) / np.diff(tel.t)
    return {
        "completion": float(np.clip(tel.progress_m[-1] / tel.route_length, 0.0, 1.0)),
        "mean_abs_lat": float(abs_lat.mean()),
        "max_abs_lat": float(abs_lat.max()),
        "steps_off_lane": int((abs_lat > OFF_LANE_LAT).sum()),
        "rms_steer_rate": float(np.sqrt(np.mean(steer_rate**2))) if steer_rate.size else 0.0,  # no interval, no rate
    }


def driving_score(tel: Telemetry) -> int:
    """round(100 x completion x clip(1 - mean|lat| / 0.9, 0, 1)); gates use explicit metrics, not this alone."""
    m = metrics(tel)
    return round(100 * m["completion"] * float(np.clip(1 - m["mean_abs_lat"] / OFF_LANE_LAT, 0.0, 1.0)))


_COLUMNS = (
    ("completion", lambda m, s: f"{m['completion']:.0%}"),
    ("mean |lat| (m)", lambda m, s: f"{m['mean_abs_lat']:.2f}"),
    ("max |lat| (m)", lambda m, s: f"{m['max_abs_lat']:.2f}"),
    ("steps off lane", lambda m, s: str(m["steps_off_lane"])),
    ("steer rate (rad/s)", lambda m, s: f"{m['rms_steer_rate']:.2f}"),
    ("score", lambda m, s: str(s)),
)


def run_labels(tels, labels):
    """`labels` if given (one per run, else ValueError), otherwise each run's scenario name."""
    if labels is None:
        return [t.scenario for t in tels]
    return [lab for _, lab in zip(tels, labels, strict=True)]


def metrics_table(*tels: Telemetry, labels=None) -> str:
    """One row of metrics per run in a fixed-width text table."""
    rows = [
        [label, *(fmt(metrics(t), driving_score(t)) for _, fmt in _COLUMNS)]
        for t, label in zip(tels, run_labels(tels, labels))
    ]
    head = ["run", *(name for name, _ in _COLUMNS)]
    widths = [max(len(r[i]) for r in [head, *rows]) for i in range(len(head))]
    line = lambda r: "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths)))
    return "\n".join([line(head), "  ".join("-" * w for w in widths), *(line(r) for r in rows)])
