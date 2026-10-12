import pytest

from measure_phone import lap_report
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


@pytest.mark.parametrize("status", ["Ready.", "ZeroDivisionError: division by zero (your code, line 2)"])
def test_lap_report_rejects_a_status_that_is_not_a_finished_lap(status):
    with pytest.raises(ValueError, match="did not report a finished lap"):
        lap_report(status, GENTLE_STEPS - 1, wall_s=8.89)


def test_lap_report_rejects_a_scrubber_that_disagrees_with_the_status():
    with pytest.raises(ValueError, match="scrubber"):
        lap_report(finished("gentle", GENTLE_STEPS), STRAIGHT_STEPS - 1, wall_s=8.89)
