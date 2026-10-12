"""Run the 0.1.4.a lap in the playground on an emulated iPhone 12 and print steps per second (spec S-5, section 7).

Usage: python scripts/measure_phone.py        Needs: python -m pip install 'playwright>=1.40' && python -m playwright install webkit
Emulation sets viewport, user agent and touch only. It does not slow the CPU, so the number is a ceiling:
read the real phone with a stopwatch (acceptance step C4).
"""
import re
import sys
import tempfile
import time
from pathlib import Path

from _course import REPO

sys.path.insert(0, str(REPO))  # `python scripts/<name>.py` puts scripts/ first, not the repo root

from zero2fsd.sim.dynamics import DT  # noqa: E402

DEVICE = "iPhone 12"
EXERCISE = "0.1.4.a"
COMPLETED = re.compile(r"Completed (?P<road>\w+) in (?P<seconds>[\d.]+) s")


def lap_report(status_text: str, scrub_max: int, wall_s: float) -> dict:
    """Steps per second and real-time factor of one finished lap, from what the page shows."""
    done = COMPLETED.search(status_text)
    if done is None:
        raise ValueError(f"the page did not report a finished lap: {status_text!r}")
    steps = scrub_max + 1  # the scrubber runs over step indices 0..steps-1
    lap_s = float(done["seconds"])
    if abs(steps * DT - lap_s) > 3 * DT:  # one step for the last row's index, one decimal of rounding
        raise ValueError(f"the scrubber has {steps} steps ({steps * DT:.2f} s) but the status says {lap_s} s")
    return {"road": done["road"], "steps": steps, "wall_s": wall_s, "steps_per_s": steps / wall_s, "real_time_x": lap_s / wall_s}


def main() -> int:
    # Imported here so lap_report stays importable (and unit-tested) without playwright installed.
    from build_playground import write_playground
    from playground_browser import P_STEER, open_ready, serve, set_code, wait_state
    from playwright.sync_api import sync_playwright

    with tempfile.TemporaryDirectory() as dist:
        write_playground(REPO, Path(dist))
        with serve(dist) as url, sync_playwright() as playwright:
            device = playwright.devices[DEVICE]
            browser = getattr(playwright, device["default_browser_type"]).launch()
            page = browser.new_context(**device).new_page()
            open_ready(page, url, EXERCISE)
            set_code(page, P_STEER)
            started = time.monotonic()
            page.click("#run")
            wait_state(page, "done", timeout_ms=600_000)
            wall_s = time.monotonic() - started
            report = lap_report(page.inner_text("#status"), int(page.get_attribute("#scrub", "max")), wall_s)
            browser.close()
    print(f"{DEVICE}: {report['road']} lap, {report['steps']} steps in {report['wall_s']:.1f} s of wall time")
    print(f"{report['steps_per_s']:.0f} steps/s, {report['real_time_x']:.1f}x real time (a ceiling: emulation does not slow the CPU)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
