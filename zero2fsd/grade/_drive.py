"""Box labs: drive the learner's car, print what happened, and gate on the result."""
from __future__ import annotations

import io
import operator

from ..car import Car, Telemetry, run
from ..car.runner import StepHook
from ..dashboard import dashboard
from ..score import driving_score, metrics, metrics_table
from ..sim import Camera
from ._report import REPORT, Failure, Threshold, learner_code, vs_limit

GATE_SCENARIO, GATE_MEAN_LAT = "gentle", 0.5  # the car must complete the scenario with mean |lat| below this (m)
THRESHOLDS = {"drive gate: mean |lat|": Threshold(GATE_MEAN_LAT, operator.lt, "0.1.4", lambda solution: metrics(run(solution.my_car, GATE_SCENARIO))["mean_abs_lat"])}


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


def drive(car: Car, scenario: str, label: str, show: bool, on_step: StepHook | None = None, camera: Camera | None = None) -> Telemetry:
    """Run the car on a scenario (learner exceptions become a Failure), print its metrics and maybe the dashboard.

    `on_step` receives each telemetry row as the car drives (see `run`)."""
    with learner_code(f"your car driving {scenario!r}"):
        tel = run(car, scenario, on_step=on_step, camera=camera)
    if REPORT.get():
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


def require_car(car) -> Car:
    """The all-yours exercise's answer must be a Car (it is `None` until the learner builds one); returns it."""
    if not isinstance(car, Car):
        raise Failure(f"my_car must be a Car(perception=..., controller=...), got {type(car).__name__}")
    return car


def gate(car: Car, show: bool, on_step: StepHook | None = None, camera: Camera | None = None) -> str:
    """Fail unless the car completes the gate scenario with mean |lat| under the limit; else say how it did."""
    tel = drive(car, GATE_SCENARIO, f"your car on {GATE_SCENARIO}", show, on_step, camera)
    mean_lat = metrics(tel)["mean_abs_lat"]
    # an unfinished run fails on completion, so a thin mean |lat| did not decide it and earns no borderline sentence
    seen = f"mean |lat| {mean_lat:.3f} m {vs_limit(mean_lat, GATE_MEAN_LAT, 'm', decisive=tel.completed)}"
    if not tel.completed or mean_lat >= GATE_MEAN_LAT:
        raise Failure(f"your car did not pass the {GATE_SCENARIO} gate",
                      expected=f"completes {GATE_SCENARIO} with mean |lat| < {GATE_MEAN_LAT} m",
                      got=f"{outcome(tel)}, {seen}", given=f"your car driving {GATE_SCENARIO!r}")
    return f"{outcome(tel)}, score {driving_score(tel)}, {seen}"
