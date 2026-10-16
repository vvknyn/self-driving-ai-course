# scripts/figstyle.py
"""The one figure style: dark page colours from the palette, applied by make_figures.py with `matplotlib.rc_context(RC)`.

A figures.py never sets a background, text colour, colour cycle or dpi; it only picks palette entries for what it draws.
"""
from cycler import cycler

from zero2fsd.palette import CHROME, QUANTITY

FIG_DENSITY = 2  # the site shows a figure PNG at its pixel width / this: a 2x srcset
FIG_DPI = 100 * FIG_DENSITY  # so one figure inch is 100 CSS px on the page

RC = {
    "savefig.dpi": FIG_DPI,
    "figure.facecolor": CHROME["background"],
    "axes.facecolor": CHROME["panel"],
    "axes.edgecolor": CHROME["border"],
    "axes.labelcolor": CHROME["text"],
    "xtick.color": CHROME["text"],
    "ytick.color": CHROME["text"],
    "text.color": CHROME["text"],
    "grid.color": CHROME["border"],
    "legend.edgecolor": CHROME["border"],
    "patch.edgecolor": CHROME["text"],  # an un-coloured annotate arrow is a patch: light on dark, not black
    "patch.facecolor": CHROME["text"],
    "axes.prop_cycle": cycler(color=list(QUANTITY.values())),
}
