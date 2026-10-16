# zero2fsd/palette.py
"""The course's colours in one place: `palette_dict()` is what the browser view reads as `palette.json`.

Paint colours come from the camera's own `PALETTE`, so the top-down view paints the road in the colours
the camera sees.  The quantity colours name the four things the lectures plot; the chrome colours are the
site's learnfsd page colours.  `shades(hue, n)` is the one way to draw a quantity `n` times on one figure.
"""
from __future__ import annotations

from .sim.camera import GRASS, PALETTE, ROAD, SKY, WHITE, YELLOW

MIX = 0.65  # how far the lightest shade of a family goes toward the text colour: still clearly its hue


def _hex(channels) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(c) for c in channels))


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    return tuple(int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))


QUANTITY = {"e_off": "#58C4DD", "psi": "#D147BD", "delta": "#83C167", "perception": "#FC6255"}
SCENE = {name: _hex(PALETTE[index]) for name, index in (("sky", SKY), ("grass", GRASS), ("road", ROAD), ("yellow", YELLOW), ("white", WHITE))}
CHROME = {"background": "#090d16", "panel": "#0d1117", "border": "#30363d", "text": "#c9d1d9", "link": "#58a6ff"}


def shades(hex_colour: str, n: int) -> list[str]:
    """`n` shades of one hue for a quantity drawn `n` times: the hue itself, then even steps toward CHROME["text"]."""
    base, text = _rgb(hex_colour), _rgb(CHROME["text"])
    return [hex_colour] + [_hex(round(b + (t - b) * MIX * i / (n - 1)) for b, t in zip(base, text)) for i in range(1, n)]


def palette_dict() -> dict[str, dict[str, str]]:
    return {"quantity": QUANTITY, "scene": SCENE, "chrome": CHROME}
