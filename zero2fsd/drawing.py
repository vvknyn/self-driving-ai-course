"""Matplotlib drawing helpers shared by the course's figures: one look for every frame and track.

Ruling F: nothing here sets a dpi, a colour or a font size. Colours come in as arguments (callers take them from
`zero2fsd.palette`); text inherits `rcParams["font.size"]` unless the caller passes `fontsize=`, because the legal
size depends on the figure's width.
"""
from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.patches import FancyArrowPatch

from .geom import Pose, heading_vector

LABEL_GAP = 0.18  # label distance beyond an arrow tip, as a fraction of the arrow length


def draw_frame(ax, pose: Pose, colour: str, name: str, length_m: float = 2.0, fontsize=None) -> None:
    """A frame's two axes as arrows labelled `x` and `y`, plus its `name` at the origin, all in `colour`."""
    x, y, theta = pose
    origin = np.array([x, y])
    x_hat = heading_vector(theta)
    y_hat = np.array([-x_hat[1], x_hat[0]])
    for axis, label in ((x_hat, "x"), (y_hat, "y")):
        tip = origin + length_m * axis
        ax.add_patch(FancyArrowPatch(origin, tip, arrowstyle="-|>", mutation_scale=mpl.rcParams["font.size"], color=colour))
        ax.text(*(origin + length_m * (1 + LABEL_GAP) * axis), label, color=colour, ha="center", va="center", fontsize=fontsize)
    ax.text(*(origin - LABEL_GAP * length_m * (x_hat + y_hat)), name, color=colour, ha="center", va="center", fontsize=fontsize)


def draw_track(ax, tel, colour: str) -> None:
    """The path a drive took, `tel.x` against `tel.y`, to scale."""
    ax.plot(tel.x, tel.y, color=colour)
    ax.set_aspect("equal", adjustable="datalim")
