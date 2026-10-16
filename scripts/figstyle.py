# scripts/figstyle.py
"""The one figure style: dark page colours from the palette, applied by make_figures.py with `matplotlib.rc_context(RC)`.

A figures.py never sets a background, text colour, colour cycle or dpi; it only picks palette entries for what it draws.
"""
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor

from cycler import cycler

from zero2fsd.palette import CHROME, QUANTITY

FIG_DENSITY = 2  # the site shows a figure PNG at its pixel width / this: a 2x srcset
FIG_DPI = 100 * FIG_DENSITY  # so one figure inch is 100 CSS px on the page

WORKERS_VAR = "ZERO2FSD_FIG_WORKERS"


def render_workers() -> int:
    """Size of every figure render pool: ZERO2FSD_FIG_WORKERS if set (a positive integer), else the CPU count."""
    raw = os.environ.get(WORKERS_VAR)
    if raw is None:
        return os.cpu_count()
    if not raw.isascii() or not raw.isdigit() or int(raw) < 1:
        raise ValueError(f"{WORKERS_VAR} must be a positive integer, got {raw!r}")
    return int(raw)


ORPHAN_POLL_SECONDS = 1


def _exit_with_parent(parent_pid: int) -> None:
    """Pool-worker initializer: a daemon thread that ends this process within a poll of its parent dying (a SIGKILLed pytest leaves workers at ppid 1)."""
    def watch():
        while os.getppid() == parent_pid:
            time.sleep(ORPHAN_POLL_SECONDS)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def render_pool() -> ProcessPoolExecutor:
    """The one figure render pool: render_workers() forked workers that exit when the process that built the pool does."""
    return ProcessPoolExecutor(render_workers(), mp_context=multiprocessing.get_context("fork"), initializer=_exit_with_parent, initargs=(os.getpid(),))


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
