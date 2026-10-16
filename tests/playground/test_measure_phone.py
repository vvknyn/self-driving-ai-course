import subprocess
import sys

import pytest

from _course import REPO
from measure_phone import lap_report, summary
from zero2fsd.sim.dynamics import DT

GENTLE_STEPS = 1778  # spec section 7: a lap of `gentle` is 88.9 s
STRAIGHT_STEPS = 502


def finished(road, steps):
    """The status line the page shows when a lap of `steps` rows is done."""
    return f"Completed {road} in {(steps - 1) * DT:.1f} s, mean |lat| 0.38 m"


def test_lap_report_gives_steps_per_second_and_real_time_factor():
    report = lap_report(finished("gentle", GENTLE_STEPS), GENTLE_STEPS - 1, wall_s=8.89)
    assert report["road"] == "gentle"
    assert report["steps"] == GENTLE_STEPS
    assert report["steps_per_s"] == pytest.approx(200.0, rel=1e-3)
    assert report["real_time_x"] == pytest.approx(10.0, rel=0.02)  # 88.9 s of driving in 8.89 s


def test_lap_report_on_a_second_road():
    report = lap_report(finished("straight", STRAIGHT_STEPS), STRAIGHT_STEPS - 1, wall_s=2.51)
    assert (report["road"], report["steps"]) == ("straight", STRAIGHT_STEPS)
    assert report["steps_per_s"] == pytest.approx(200.0, rel=1e-3)


@pytest.mark.parametrize("status", [
    "Ready.",
    "ZeroDivisionError: division by zero (your code, line 2)",  # the `error` state
    "Left the road on gentle after 12.3 s and 40 m, mean |lat| 1.20 m",  # the two ways a drive ends without finishing the lap
    "Ran out of time on gentle after 120.0 s and 5 m, mean |lat| 0.20 m",
])
def test_lap_report_rejects_a_status_that_is_not_a_finished_lap(status):
    with pytest.raises(ValueError, match="did not report a finished lap"):
        lap_report(status, GENTLE_STEPS - 1, wall_s=8.89)


def test_lap_report_rejects_a_scrubber_that_disagrees_with_the_status():
    with pytest.raises(ValueError, match="scrubber"):
        lap_report(finished("gentle", GENTLE_STEPS), STRAIGHT_STEPS - 1, wall_s=8.89)


@pytest.mark.parametrize("road, steps, seconds", [("gentle", GENTLE_STEPS, "88.8"), ("straight", STRAIGHT_STEPS, "25.0")])
def test_lap_report_accepts_a_status_one_rounding_step_below_the_scrubber(road, steps, seconds):
    """steps * DT is 88.9 s and the page printed 88.8: the page rounds its last row's time (steps - 1) * DT to a decimal, so this is legal."""
    assert lap_report(f"Completed {road} in {seconds} s, mean |lat| 0.38 m", steps - 1, wall_s=9.0)["steps"] == steps


@pytest.mark.parametrize("road, steps, short", [("gentle", GENTLE_STEPS, 3), ("gentle", GENTLE_STEPS, 5), ("straight", STRAIGHT_STEPS, 3)])
def test_lap_report_rejects_a_scrubber_a_few_steps_short_of_the_status(road, steps, short):
    with pytest.raises(ValueError, match="scrubber"):
        lap_report(finished(road, steps), steps - 1 - short, wall_s=9.0)


def lap_of(steps_per_s, wall_s=9.0):
    return {"road": "gentle", "steps": GENTLE_STEPS, "wall_s": wall_s, "steps_per_s": steps_per_s, "real_time_x": GENTLE_STEPS * DT / wall_s}


@pytest.mark.parametrize("rates, best", [([163.0, 200.0, 224.0], "224"), ([224.0, 163.0], "224"), ([190.0, 191.0, 150.0], "191")])
def test_summary_reports_the_fastest_lap_as_the_ceiling_and_every_rate(rates, best):
    first, second = summary([lap_of(rate) for rate in rates]).splitlines()
    assert f"best of {len(rates)} laps" in first
    assert second.startswith(f"{best} steps/s")
    assert ", ".join(f"{rate:.0f}" for rate in rates) in second  # the spread, in the order the laps ran


@pytest.mark.parametrize("modules", [["_course"], ["measure_phone"], ["measure_phone", "playground_browser"]], ids="+".join)
def test_a_script_entry_finds_the_repo_once_however_many_of_them_are_imported(modules, tmp_path):
    """`python scripts/<name>.py` puts scripts/ on the path, not the repo root: _course adds the root, once, for all of them."""
    probe = (
        f"import sys; from pathlib import Path; repo = Path({str(REPO)!r})\n"
        f"sys.path[:] = [{str(REPO / 'scripts')!r}] + [p for p in sys.path if Path(p or '.').resolve() != repo]\n"
        f"import {', '.join(modules)}, zero2fsd.sim.dynamics\n"
        "print(sum(Path(p or '.').resolve() == repo for p in sys.path))")
    done = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert (done.returncode, done.stdout.strip()) == (0, "1"), done.stderr
