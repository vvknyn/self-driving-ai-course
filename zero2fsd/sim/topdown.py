"""Top-down plot of a road with optional trajectories, for dashboards and labs."""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from .world import LANE_WIDTH, Road

EDGE_COLOR, DASH_COLOR = "#666666", "#c9a800"


def render_topdown(road: Road, trajectories=(), labels=(), ax=None):
    """Draw `road` and each (N, 2) trajectory on `ax` (a new figure's axes if None); returns the Axes."""
    ax = ax if ax is not None else Figure(figsize=(6, 6)).subplots()
    s = np.linspace(0.0, road.length, max(2, int(road.length)))
    for lateral, style in ((-LANE_WIDTH, "-"), (LANE_WIDTH, "-"), (0.0, (0, (5, 3)))):
        x, y = road.offset(s, lateral).T
        ax.plot(x, y, color=DASH_COLOR if lateral == 0.0 else EDGE_COLOR, linestyle=style, linewidth=1.2)
    for i, traj in enumerate(trajectories):
        traj = np.asarray(traj)
        ax.plot(traj[:, 0], traj[:, 1], linewidth=2, label=labels[i] if i < len(labels) else None)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if labels:
        ax.legend(loc="best")
    return ax
