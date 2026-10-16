"""The Python side of the playground: the worker calls these functions; nothing here touches the page.

Importable natively (the tests import it by path); in the browser it is fetched as its own file and
imported after bundle.zip has been unpacked.
"""
from __future__ import annotations

import functools
import io
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import numpy as np

from zero2fsd import geom
from zero2fsd.car import Car, Telemetry, run
from zero2fsd.car.runner import COLUMNS, StepHook
from zero2fsd.dashboard import dashboard
from zero2fsd.grade import check
from zero2fsd.grade._report import Failure, exercise_label
from zero2fsd.grade.units.u0_1_2 import NOISE_SIGMA
from zero2fsd.palette import SCENE
from zero2fsd.preview import get
from zero2fsd.preview.units.u0_1 import OVERLAY
from zero2fsd.score import metrics
from zero2fsd.sim import BicycleState, Camera, make_scenario

# What every exercise cell sees: the names the labs' setup cells bind, minus the notebook-only ones
# (plt, dashboard, check, practice, ends), plus the provided boxes a learner may build on.
PRELUDE = f"""\
import math
import time

import numpy as np

from zero2fsd.car import Action, Car, LaneEstimate, Observation, provided, run
from zero2fsd.car.provided import (FAR_ROWS, K_HEAD, K_OFF, MIN_PIXELS, NEAR_ROWS, estimate_lane, hold_speed,
                                   lane_point, paint_masks, steer_p)
from zero2fsd.geom import {', '.join(geom.__all__)}
from zero2fsd.palette import CHROME, QUANTITY, SCENE
from zero2fsd.score import metrics, metrics_table
from zero2fsd.sim import BicycleState, Camera, make_scenario
from zero2fsd.sim.camera import GRASS, PALETTE, ROAD, SKY, WHITE, YELLOW
from zero2fsd.sim.dynamics import DT, MAX_STEER, WHEELBASE
from zero2fsd.sim.scenarios import TARGET_SPEED

camera = Camera()
"""
LEARNER_FILENAME = "<learner>"
IMAGE_ROAD = "straight"  # the car's first view of this road is the picture; the registry gives image previews no road

LAST_RUN: Telemetry | None = None  # the telemetry of the last completed preview run, for the dashboard


def learner_object(code: str, name: str) -> Any:
    """Run the learner's code after the prelude and return what it bound to `name`."""
    ns: dict[str, Any] = {}
    exec(PRELUDE, ns)
    untouched = dict(ns)  # what the prelude bound, to tell a name the learner left alone from one they defined
    exec(compile(code, LEARNER_FILENAME, "exec"), ns)
    if name not in ns or (name in untouched and ns[name] is untouched[name]):  # a provided box is not an answer
        raise NameError(f"define `{name}` in the editor; the check looks for that exact name")
    return ns[name]


def _learner_line(exc: BaseException) -> int | None:
    """The line in the learner's code that raised `exc` (innermost learner frame, or a SyntaxError's own line)."""
    if isinstance(exc, SyntaxError) and exc.filename == LEARNER_FILENAME:
        return exc.lineno
    lines = [lineno for frame, lineno in traceback.walk_tb(exc.__traceback__) if frame.f_code.co_filename == LEARNER_FILENAME]
    return lines[-1] if lines else None


def format_error(exc: BaseException) -> dict[str, Any]:
    message = ": ".join(part for part in (type(exc).__name__, str(exc)) if part)  # an empty message leaves no dangling colon
    return {"message": message, "line": _learner_line(exc),
            "traceback": "".join(traceback.format_exception(exc))}


@contextmanager
def _at_boundary(exercise_id: str):
    """The one place a wrong answer type is reported. The registry's checks (a `Failure`: my_car is not a Car, the
    masks are not masks) and the course's own (a `TypeError` from Car.step) come from code that is not the
    learner's: say which exercise. The learner's own exceptions have a `<learner>` frame and pass through."""
    try:
        yield
    except Failure as exc:
        raise TypeError(f"{exercise_label(exercise_id)}: {exc.headline}") from exc
    except TypeError as exc:
        if _learner_line(exc) is not None:
            raise
        raise TypeError(f"{exercise_label(exercise_id)}: {exc}") from exc


def _preview(exercise_id: str, kind: str):
    preview = get(exercise_id)
    if preview.kind != kind:
        raise ValueError(f"{exercise_id} is a {preview.kind} exercise; this call is for {kind} exercises")
    return preview


def row_dict(row: tuple) -> dict[str, Any]:
    return {name: value.item() if isinstance(value, np.generic) else value for name, value in zip(COLUMNS, row, strict=True)}


class HostCallbackError(RuntimeError):
    """The playground's own per-row callback failed. Not the learner's code; the page is broken."""


def _host_hook(on_row: Callable[[dict], None] | None) -> tuple[StepHook | None, Callable[[], None]]:
    """The step hook for a host callback, and a check to run after the drive.

    The drive runs under the grader's learner_code / _at_boundary, which label every exception the learner's.
    A failing host callback is recorded once, the hook stays quiet for the rest of the run, and `after()`
    raises HostCallbackError outside those labels, so a broken page never reads as a learner mistake."""
    if on_row is None:
        return None, lambda: None
    failed: list[BaseException] = []

    def on_step(row):
        if failed:
            return
        try:
            on_row(row_dict(row))
        except Exception as exc:  # recorded here, raised by after(); never swallowed
            failed.append(exc)

    def after():
        if failed:
            raise HostCallbackError(f"the playground's row callback failed: {failed[0]!r}") from failed[0]

    return on_step, after


def summary(tel: Telemetry) -> dict[str, Any]:
    m = metrics(tel)
    return {"completed": tel.completed, "steps": len(tel.t), "mean_abs_lat": m["mean_abs_lat"],
            "max_abs_lat": m["max_abs_lat"], "progress_m": float(tel.progress_m[-1]), "off_road": bool(tel.off_road[-1])}


def preview_run(obj: Any, exercise_id: str, road: str, on_row: Callable[[dict], None]) -> dict[str, Any]:
    """Drive the learner's car on `road`, handing each step to `on_row`; return the run's summary. If `on_row` fails,
    the run still completes and `HostCallbackError` is raised after the drive (the page is broken, not the learner's code)."""
    global LAST_RUN
    preview = _preview(exercise_id, "drive")
    LAST_RUN = None  # a run that raises must not leave the previous run behind for the dashboard
    on_step, after = _host_hook(on_row)
    with _at_boundary(exercise_id):
        tel = run(preview.build(obj), road, on_step=on_step)
    after()  # outside _at_boundary: a failing host callback is not a wrong answer type
    LAST_RUN = tel
    return summary(tel)


@functools.cache
def canned_telemetry(road: str) -> Telemetry:
    """The provided car's run on `road`, the data a stats exercise summarises: one run per road, shared."""
    return run(Car(), road)


def preview_stats(obj: Any, exercise_id: str) -> dict[str, float]:
    """The numbers the learner's stats function produces from the canned drive on the exercise's road, by label."""
    preview = _preview(exercise_id, "stats")
    with _at_boundary(exercise_id):
        return preview.build(obj, canned_telemetry(preview.road))


def preview_image(obj: Any, exercise_id: str, seed: int = 0) -> dict[str, Any]:
    """A noisy camera frame from the start of `IMAGE_ROAD` and an overlay of the same size: the pixels the learner's
    masks mark are painted in the scene's yellow and white, the rest is black. The registry paints the masks in
    its OVERLAY colours (after checking them), and no camera pixel is anywhere near those, so they read back exactly."""
    preview = _preview(exercise_id, "image")
    scenario = make_scenario(IMAGE_ROAD)
    state = BicycleState(*scenario.road.ego_pose(scenario.start_s), scenario.target_speed, 0.0)
    frame = Camera().render(scenario.road, state, NOISE_SIGMA, seed)
    with _at_boundary(exercise_id):
        shown = preview.build(obj, frame)
    overlay = np.zeros_like(frame)
    for name, marker in OVERLAY.items():
        overlay[(shown == marker).all(axis=-1)] = tuple(bytes.fromhex(SCENE[name][1:]))
    height, width = frame.shape[:2]
    return {"width": width, "height": height, "frame": frame.tobytes(), "overlay": overlay.tobytes()}


def check_code(obj: Any, exercise_id: str, on_row: Callable[[dict], None] | None = None) -> dict[str, Any]:
    """The verdict Colab gives, plus the rows of any run the check makes when `on_row` is given. Colab prints the
    report; here the same lines are the verdict's `details`, so the check prints nothing. If `on_row` fails,
    `HostCallbackError` is raised after the drive instead of a verdict."""
    on_step, after = _host_hook(on_row)
    result = check(exercise_id, obj, show=False, on_step=on_step, report=False)
    after()
    return {"passed": bool(result.passed), "details": list(result.details)}


def frame_rgb(road: str, k: int, x: float, y: float, yaw: float, speed: float, seed: int, noise_sigma: float = 0.0) -> bytes:
    """The camera frame the car saw at step `k`, re-rendered from its recorded pose."""
    state = BicycleState(x, y, yaw, speed, 0.0)
    return Camera().render(make_scenario(road).road, state, noise_sigma, seed + k).tobytes()


def dashboard_png() -> bytes:
    if LAST_RUN is None:
        raise RuntimeError("run the car first: the dashboard shows the last run")
    buf = io.BytesIO()
    dashboard(LAST_RUN).savefig(buf, format="png", dpi=100)
    return buf.getvalue()


def probe_steps_per_s(n: int = 100) -> float:
    """Simulation speed of this runtime: steps per second of the provided car on `straight`."""
    start = time.perf_counter()
    steps = len(run(Car(), "straight", max_steps=n).t)
    return steps / (time.perf_counter() - start)
