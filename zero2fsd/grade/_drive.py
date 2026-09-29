"""Box labs: drive the learner's car, print what happened, and gate on the result."""
from __future__ import annotations

import io

from ..car import Car, Telemetry, run
from ..dashboard import dashboard
from ..score import driving_score, metrics, metrics_table
from ._report import Failure, learner_code

GATE_SCENARIO, GATE_MEAN_LAT = "gentle", 0.5  # the car must complete the scenario with mean |lat| below this (m)


def show_figure(fig) -> None:
    """Display a figure in a notebook; outside IPython (tests, scripts) there is nothing to display it in."""
    try:
        from IPython.display import Image, display
    except ImportError:
        return
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    display(Image(data=buf.getvalue()))


def outcome(tel: Telemetry) -> str:
    if tel.completed:
        return f"completed {tel.scenario}"
    how = "left the lane" if tel.off_road[-1] else "ran out of time"
    return f"{how} after {tel.progress_m[-1]:.0f} of {tel.route_length:.0f} m of {tel.scenario}"


def drive(car: Car, scenario: str, label: str, show: bool) -> Telemetry:
    """Run the car on a scenario (learner exceptions become a Failure), print its metrics and maybe the dashboard."""
    with learner_code(f"your car driving {scenario!r}"):
        tel = run(car, scenario)
    print(metrics_table(tel, labels=[label]))
    if show:
        show_figure(dashboard(tel, labels=[label]))
    return tel


def require_own(car: Car, *slots: str) -> None:
    """The car must not run the provided box in `slots` (every slot if none are named)."""
    still = [name for name in car.uses_provided() if not slots or name in slots]
    if still:
        raise Failure(f"your car still uses the provided {' and '.join(still)}; swap in your own",
                      given=f"Car with provided: {car.uses_provided()}")


def gate(car: Car, show: bool) -> str:
    """Fail unless the car completes the gate scenario with mean |lat| under the limit; else say how it did."""
    tel = drive(car, GATE_SCENARIO, f"your car on {GATE_SCENARIO}", show)
    mean_lat = metrics(tel)["mean_abs_lat"]
    if not tel.completed or mean_lat >= GATE_MEAN_LAT:
        raise Failure(f"your car did not pass the {GATE_SCENARIO} gate",
                      expected=f"completes {GATE_SCENARIO} with mean |lat| < {GATE_MEAN_LAT} m",
                      got=f"{outcome(tel)}, mean |lat| {mean_lat:.2f} m", given=f"your car driving {GATE_SCENARIO!r}")
    return f"{outcome(tel)}, mean |lat| {mean_lat:.2f} m (< {GATE_MEAN_LAT} m), score {driving_score(tel)}"
