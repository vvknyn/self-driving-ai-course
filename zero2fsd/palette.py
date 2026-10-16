# zero2fsd/palette.py
"""The course's colours in one place: `palette_dict()` is what the browser view reads as `palette.json`.

Paint colours come from the camera's own `PALETTE`, so the top-down view paints the road in the colours
the camera sees.  The quantity colours name the four things the lectures plot; the chrome colours are the
site's learnfsd page colours.
"""
from __future__ import annotations

from .sim.camera import GRASS, PALETTE, ROAD, SKY, WHITE, YELLOW

QUANTITY = {"e_off": "#58C4DD", "psi": "#D147BD", "delta": "#83C167", "perception": "#FC6255"}
SCENE = {
    name: "#{:02X}{:02X}{:02X}".format(*(int(c) for c in PALETTE[index]))
    for name, index in (("sky", SKY), ("grass", GRASS), ("road", ROAD), ("yellow", YELLOW), ("white", WHITE))
}
CHROME = {"background": "#090d16", "panel": "#0d1117", "border": "#30363d", "text": "#c9d1d9", "link": "#58a6ff"}


def palette_dict() -> dict[str, dict[str, str]]:
    return {"quantity": QUANTITY, "scene": SCENE, "chrome": CHROME}
