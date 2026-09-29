"""One figure that shows how a run (or several) went."""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from .car.runner import Telemetry
from .score import OFF_LANE_LAT, metrics_table, run_labels
from .sim import make_scenario, render_topdown


def dashboard(*tels: Telemetry, labels=None) -> Figure:
    """2x2: top-down trajectories, lateral error vs time, estimated vs true offset, metrics table.

    Runs must share a scenario (they are drawn on one road).  Returns the Figure; the caller shows it.
    """
    scenarios = {t.scenario for t in tels}
    if len(scenarios) != 1:
        raise ValueError(f"a dashboard compares runs on one scenario, got {sorted(scenarios) or 'no runs'}")
    names = run_labels(tels, labels)
    fig = Figure(figsize=(12, 8))
    top, lat, est, table = fig.subplots(2, 2).ravel()

    render_topdown(make_scenario(tels[0].scenario).road, [np.column_stack([t.x, t.y]) for t in tels], names, ax=top)
    top.set_title("Where the car drove")

    for i, (t, name) in enumerate(zip(tels, names)):
        lat.plot(t.t, t.lat, color=f"C{i}", label=name)
        est.scatter(t.lat, t.est_offset, s=6, color=f"C{i}", label=name)
    for edge in (-OFF_LANE_LAT, OFF_LANE_LAT):
        lat.axhline(edge, color="0.5", linestyle="--", linewidth=1)
    lat.set(title="Lateral error (dashed: off-lane limit)", xlabel="time (s)", ylabel="offset from lane centre (m)")

    reach = max(np.abs(np.concatenate([t.lat for t in tels])).max(), OFF_LANE_LAT)
    est.plot([-reach, reach], [-reach, reach], color="0.5", linestyle="--", linewidth=1)
    est.set(title="Estimated vs true offset (dashed: perfect)", xlabel="true offset (m)", ylabel="estimated offset (m)")
    if len(tels) > 1:
        lat.legend(loc="best")
        est.legend(loc="best")

    table.axis("off")
    table.text(0.0, 1.0, metrics_table(*tels, labels=names), family="monospace", fontsize=8, va="top")
    fig.tight_layout()
    return fig
