# Zero2FSD Playground Implementation Plan (course repo)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the in-browser car playground: a learner edits one box of the car, watches it drive a course road, presses *Check my work*, and gets the verdict Colab gives, all from a static folder built by this repo.

**Architecture:** The Python package `zero2fsd/` gains four small seams (a per-step hook on `run`, lazy charting imports, a road export and a preview registry) and a margin test that keeps the graders honest across platforms. A new static folder `playground/` (ES modules, no framework) talks to a module Web Worker that loads Pyodide and the course bundle; the worker produces a full-speed run and streams rows, the page replays them with a player, strip charts and camera frames. `scripts/build_playground.py` writes `course/dist/playground/` (bundle.zip, exercises.json, roads.json, palette.json, manifest.json and the eight page files), which the site copies as-is.

**Tech Stack:** Python 3.12, numpy, Pillow, matplotlib (lazy), pytest; Pyodide 314.0.7 from cdn.jsdelivr.net in a module worker; CodeMirror 6 from cdn.jsdelivr.net with a textarea fallback; Canvas 2D; playwright-python for browser tests (optional `[test]` extra).

**Spec:** `docs/superpowers/specs/2026-10-11-playground-design.md` (branch `feat/playground-design`). The companion site plan is `docs/superpowers/plans/2026-10-11-playground-site-seams.md` in the `vvknyn/vivekn` repo; it runs after Part B of this plan is built and pushed.

## Global Constraints

Copied from the spec. Every task's requirements include this section.

- First load, wired: ≤ 3 s, ≤ 11 MB (measured 1.71 s, 10.36 MB without matplotlib). (§7)
- First load at 10 Mbit/s + 40 ms: ≤ 12 s (measured 9.84 s). (§7)
- Lap of `gentle` with camera: ≥ 5× real time on a Mac (measured 10.3×, 205 steps/s Chrome; 11.6× Safari). (§7)
- Same, CPU throttled 4×: ≥ 2× real time (measured 2.33×). (§7)
- Heap: ≤ 120 MB (measured 74.9 MB, worker, one lap). (§7)
- Phone: not predicted; measured in acceptance (S-5). (§7)
- Pyodide and CodeMirror come from cdn.jsdelivr.net at pinned versions; Pyodide 314.0.7 is the measured one. (§4.7)
- numpy and Pillow load at start; matplotlib loads only on the first *Dashboard* press. (S8)
- Learner code runs only in the learner's browser; nothing leaves the page. URL parameters are names looked up in `exercises.json`, never evaluated. (§8)
- The unit-page iframe is same-origin and carries no `sandbox` attribute (it would cut off `localStorage` and break S-3). (§8)
- Stop = `terminate()` and a fresh worker; no SharedArrayBuffer, no COOP/COEP headers. (S5)
- Camera frames are re-rendered on demand for the scrubbed step, never stored per step. (S6)
- Progress is additive: a new `exercises` field under each unit, playground passes only. (S7)
- Week 1 kinds are `drive`, `stats` and one `image` exercise (0.1.2.a); see deviation P14. (S10)
- Graders never import solutions; starters in `exercises.json` equal the lab cells byte for byte. (§4.2, §9)
- Every Mac-vs-browser threshold has a margin test and prints the borderline sentence verbatim: "Borderline: tiny differences between computers can flip this. Aim for a clearer margin." (§4.1 e, S3)
- Nothing from the spike harness is copied into the repo. (§12)
- Code follows `~/.claude/skills/writing-lean-code`: smallest complete diff, no lying fallbacks, never gate by enumerating names (read the registry or JSON instead).

## Review Focus

The spec's error table (§6) names the behaviours. These five are the ones most likely to bite a learner, most likely first. Each has a test in the task named.

1. **Learner code raises mid-run.** Expect: the console shows the traceback with the learner's line number, the run ends at that step, the car stays where it stopped, the verdict panel says nothing, and the next *Run* works without a reload. Tests: Task 7 `test_format_error_reports_the_learner_line_and_nothing_else` and `test_a_learner_that_raises_mid_run_stops_the_run_and_the_next_run_works`; Task 12 `test_raise_at_step_50`.
2. **Infinite loop in learner code.** Expect: *Stop* halts the page within about 3 s, status reads "Stopped.", the runtime reloads and a new run starts within 3 s. Test: Task 12 `test_stop_halts_while_true_and_page_stays_responsive` (S-4).
3. **Wrong return type from learner code** (a `steer` that returns a tuple, a stats function that returns a string). Expect: a plain-language error naming the exercise and its expected type, never a numpy traceback from deep in the simulator. Tests: Task 7 `test_a_wrong_return_type_is_a_plain_error_naming_the_exercise` (drive) and `test_a_stats_function_returning_the_wrong_type_is_a_plain_error` (stats); Task 12 `test_wrong_return_type[drive-tuple]` (the drive case: `#stdout` has "must return one number" and the exercise id) and `test_wrong_return_type[stats-text]` (the stats case: `#stdout` has "expected a number or a dict of numbers" and the exercise id).
4. **`localStorage` absent or corrupt** (private mode, storage blocked, a hand-edited value). Expect: `progress.js` never throws, the verdict still shows, and the site's `parseProgress` tolerates a unit with or without `exercises`. Tests: Task 11 `check_progress.mjs` (progress.js round trip under a throwing storage stub); Task 12 `test_storage_blocked`; site plan Task 3.
5. **Slow or blocked CDN.** Expect: the loading bar names the stage that is running, a blocked editor import falls back to the textarea, a failed Pyodide load turns into an error naming the file plus a *Retry* button, and a *Dashboard* press whose matplotlib download fails shows an error naming the wheel and leaves the page usable; never a silent blank page. Tests: Task 12 `test_editor_import_blocked_falls_back_to_textarea`, `test_route_abort_pyodide_js`, `test_route_abort_names_the_failed_file` and `test_route_abort_names_the_matplotlib_wheel`.

## Spec deviations (decided while writing this plan; each is reversible)

| # | Spec says | Plan does | Why |
|---|---|---|---|
| P1 | `Road.centerline_points(spacing_m)` (§4.1 c) | `Road.sample_points(spacing_m, lateral=0.0)` | One method serves the five exported lines (tarmac edges, yellow, white, ego centre) that the camera also draws |
| P2 | Export "the centreline" | Export five polylines in `roads.json` | The view must draw the same road the camera sees; one line cannot |
| P3 | Stats preview returns a string | `build(fn, tel) -> dict[str, float]` | The page formats numbers; the shim stays free of presentation |
| P4 | Shim inside bundle.zip under `playground/py/` | `playground/shim.py`, fetched as its own file | The shim is importable natively for its own tests and is not part of the course package |
| P5 | Presets per lecture link | Mechanism only in v1 (`starter` key); content presets arrive with the Week 1 content plan | Presets are content; this plan is structure |
| P6 | `palette.json` holds the quantity palette | Also carries `scene` (from the camera palette) and `chrome` | The view, the strips and the page chrome read one file |
| P7 | Sync copies the folder | `manifest.json` with sha256 per file; sync deletes stale files | A renamed or removed file must not linger on the site |
| P8 | §8 "the loader script carries an integrity hash" | Exact version pin only | A dynamic `import()` inside a module worker cannot carry an integrity attribute; see the `> NOTE:` under Task 9. Vivek's call whether to accept pin-only |
| P9 | One plan | Two plans (course, site) | The site repo is private and ships through a different door; each plan produces working software on its own |
| P10 | Stats preview telemetry | Runs the provided car once in the browser and caches it | No precomputed telemetry file to keep in sync in v1 |
| P11 | URL `/learnfsd/playground/?exercise=<id>` (§4.6) | `playground/index.html?exercise=<id>[&preset=<name>]`; lecture links are relative with no leading slash and the site renderer prefixes the base path | Next serves `public/` by exact path (the directory URL answers 308 then 404, verified by the site leg); a rewrite would break the page's relative module imports; the relative form survives the GitHub Pages move |
| P12 | Map lights "that box" (§4.6) | Each unit.yaml gains a short `car_box:` key (instruments, perception, steer); the build emits it per exercise record and fails naming a unit that lacks it | `course.json`'s `box` is a prose sentence; matching prose to a box name would be a guess |
| P13 | Grader constant names | Keep the existing `GATE_SCENARIO, GATE_MEAN_LAT` in `grade/_drive.py:11`; export rounds to `COORD_DECIMALS = 3` | Smallest diff; rounding makes the committed `roads.json` stable across machines (382 KB → 206 KB measured by the Tasks 1–3 leg) |
| P14 | S10: "Week 1 kinds are `drive` and `stats`; `image` has no Week 1 exercise" | Week 1 has one image exercise (0.1.2.a, paint masks, found when the registry test forced every exercise to carry a kind). v1 adds a minimal image preview: one camera frame beside the learner's mask overlay, no player | The alternative is a "Colab only" label on 0.1.2.a, which contradicts S-2 ("every Week 1 exercise"). **Vivek's call**: keep the minimal preview or strike it and label the exercise Colab-only |
| P15 | §2: each lab exercise cell links to its playground page | Deferred to the Week 1 content plan | The link must be absolute and the course's home URL is not fixed yet (vivekn.xyz/learnfsd now, zero2fsd.github.io after the open-source move); the content plan edits every lab cell anyway, and one `PLAYGROUND_BASE_URL` constant in the lab builder will carry it. Vivek's call to pull it forward |
| P16 | §7: a lap of `gentle` at ≥ 2× real time with the CPU throttled 4×; heap ≤ 120 MB | Both are read off by hand in step C9, not asserted by a test | Playwright's CDP session cannot read a dedicated worker's heap without flattened child sessions, which the Python API does not expose; the Mac ≥ 5× budget is asserted by Task 12 |

## Sequencing

Part A (Tasks 1–6b) is pure Python with tests and no site change. Part B (Tasks 7–12) builds the playground and its browser tests. Part C (Task 13) is acceptance. The site plan follows Part B. Weeks 2–4 then add their exercises and presets with no structural change (the `image` kind already ships with 0.1.2.a, deviation P14).

---

## Part A: Foundation

Baseline: `226 passed in 76.98s (0:01:16)` from `/Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest tests -q` (Python 3.13.5, pytest 9.1.1, tree == origin/main c3de7d4, `git status` clean).

> NOTE (suite command): running pytest at the repo root without the `tests` path does NOT run: it collects the legacy `modules/*/tests` and stops with "Interrupted: 11 errors during collection" (`No module named 'torch'`, `'cv2'`, `'scipy'`). Every "whole suite" step below therefore uses `python -m pytest tests -q`. Use the repo venv (`source .venv/bin/activate`); the miniconda `python` on PATH has no pytest.

> NOTE (worktree imports): in a git worktree, or any copy of the course next to the primary checkout, `python -m pytest tests` can import `zero2fsd` from the primary checkout. The editable install's finder maps `zero2fsd` to the primary path, and `tests/zero2fsd/__init__.py` makes the tests package register as `zero2fsd`, so a new submodule such as `zero2fsd.palette` is looked up through that mapping and not through `PYTHONPATH`. The symptom is `ModuleNotFoundError: No module named 'zero2fsd.palette'` after you created the file (the message is the same as the expected Step 2 failure, so the problem shows only at Step 4). Working in the primary checkout is not affected. In a worktree, point the finder at the worktree once per shell (verified; `pip install -e` would change the shared venv, so do not use it):
>
> ```bash
> mkdir -p .pyshim && cat > .pyshim/sitecustomize.py <<EOF
> import __editable___zero2fsd_0_1_0_finder as _f
> _f.MAPPING["zero2fsd"] = "$PWD/zero2fsd"
> EOF
> echo ".pyshim/" >> "$(git rev-parse --git-dir)/info/exclude"
> export PYTHONPATH="$PWD/.pyshim"
> ```

### Task 1: Step hook in the runner

**Files:**
- Modify: `zero2fsd/car/runner.py:4-5` (import), `:15-16` (constants), `:43-47` (signature and docstring), `:62-65` (the row)
- Test: `tests/zero2fsd/test_runner_hook.py`

**Interfaces:**
- Consumes: `run(car, scenario, max_steps=None, seed=0, noise_sigma=0.0) -> Telemetry` at `zero2fsd/car/runner.py:43`; the 14-tuple appended at `:62-63`; `Telemetry` fields at `:24-40` (the first 14 are the arrays, then `scenario`, `completed`, `route_length`).
- Produces: `COLUMNS: tuple[str, ...]` (14 names, equal to the first 14 `Telemetry` field names); `run(..., on_step: Callable[[tuple], None] | None = None)`. `on_step(row)` is called right after the row is appended and before the off-road/finished check, so the last row is delivered too. Nothing else about `run` changes.

- [ ] **Step 1: Write the failing test**

```python
# tests/zero2fsd/test_runner_hook.py
import dataclasses

import numpy as np
import pytest

from zero2fsd.car import Car, Telemetry, run
from zero2fsd.car.runner import COLUMNS


def assert_same_telemetry(a: Telemetry, b: Telemetry) -> None:
    for field in dataclasses.fields(Telemetry):
        x, y = getattr(a, field.name), getattr(b, field.name)
        assert np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y, field.name


def test_columns_are_the_telemetry_arrays_in_order():
    names = [f.name for f in dataclasses.fields(Telemetry)]
    assert COLUMNS == tuple(names[: len(COLUMNS)])
    assert len(COLUMNS) == 14 and names[len(COLUMNS)] == "scenario"  # every array, and nothing else


@pytest.mark.parametrize("scenario", ["straight", "curvy"])  # curvy ends off the road, straight ends complete
def test_the_hook_streams_exactly_the_telemetry(scenario):
    seen = []
    hooked = run(Car(), scenario, on_step=seen.append)
    assert len(seen) == len(hooked.t)
    assert all(isinstance(row, tuple) and len(row) == len(COLUMNS) for row in seen)
    for i, name in enumerate(COLUMNS):
        assert np.array_equal(np.array([row[i] for row in seen]), getattr(hooked, name)), name
    assert_same_telemetry(hooked, run(Car(), scenario))  # the hook does not change the run


def test_the_hook_is_called_once_per_step_on_gentle():
    calls = []
    tel = run(Car(), "gentle", on_step=calls.append)
    assert tel.completed and len(calls) == len(tel.t) == 1778  # measured Oct 11 2026 on the provided car


def test_an_exception_in_the_hook_propagates():
    def boom(row):
        raise RuntimeError("hook failed")

    with pytest.raises(RuntimeError, match="hook failed"):
        run(Car(), "straight", on_step=boom)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/zero2fsd/test_runner_hook.py -v`
Expected: collection error, `ImportError: cannot import name 'COLUMNS' from 'zero2fsd.car.runner'`.

- [ ] **Step 3: Write the minimal implementation**

All four edits are in `zero2fsd/car/runner.py`.

Lines 4-5, add the `Callable` import:

```diff
 import math
+from collections.abc import Callable
 from dataclasses import dataclass
```

Lines 15-16, add `COLUMNS` under the existing constants (`Telemetry` starts at line 19):

```diff
 OFF_ROAD_OFFSET = 1.8  # m of true offset from ego-lane centre that ends the episode
 MAX_STEP_FACTOR = 3  # default step budget: this many times the steps needed at target speed
+# The values of one step in the order `run` records them: the first 14 fields of `Telemetry`.
+# `run` hands one such tuple to `on_step`.
+COLUMNS = ("t", "x", "y", "yaw", "speed", "steer", "accel", "lat", "head_err",
+           "est_offset", "est_heading", "est_valid", "progress_m", "off_road")
```

Lines 43-47, the signature and docstring:

```diff
-def run(car: Car, scenario, max_steps: int | None = None, seed: int = 0, noise_sigma: float = 0.0) -> Telemetry:
+def run(car: Car, scenario, max_steps: int | None = None, seed: int = 0, noise_sigma: float = 0.0,
+        on_step: Callable[[tuple], None] | None = None) -> Telemetry:
     """Run until the route is complete, the car is off the road, or `max_steps` (default 3x the ideal).
 
-    Exceptions raised inside the car's boxes propagate.  `scenario` is a Scenario or its name.
+    Exceptions raised inside the car's boxes (or in `on_step`) propagate.  `scenario` is a Scenario or its name.
+    `on_step(row)` is called with each step's record (a tuple in `COLUMNS` order) as soon as it is appended, so a
+    caller can stream the telemetry; it sees exactly the values the returned `Telemetry` will hold.
     """
```

Lines 61-65, build `row` once, append it, hand the same object to the hook:

```diff
         off_road = abs(lat) > OFF_ROAD_OFFSET
-        rows.append((k * DT, state.x, state.y, state.yaw, state.speed, state.steer, action.accel, lat, head_err,
-                     est.offset_m, est.heading_rad, est.valid, progress, off_road))
+        row = (k * DT, state.x, state.y, state.yaw, state.speed, state.steer, action.accel, lat, head_err,
+               est.offset_m, est.heading_rad, est.valid, progress, off_road)
+        rows.append(row)
+        if on_step is not None:
+            on_step(row)
         if off_road or progress >= sc.route_length:
             break
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/zero2fsd/test_runner_hook.py -v`
Expected: PASS (5 tests; about 9 s, the `gentle` run is 6 s of it).

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests -q`
Expected: all green; number of tests = 226 + 5 = 231.

- [ ] **Step 6: Commit**

```bash
git add zero2fsd/car/runner.py tests/zero2fsd/test_runner_hook.py
git commit -m "feat(car): run() streams each step to an optional on_step hook

COLUMNS names the 14 per-step values; the hook gets the exact tuple that
becomes the telemetry row, so the playground never recomputes lat or progress.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Browser import surface (lazy matplotlib)

The grep, run from the repo root at c3de7d4 (`yaml` has no hit anywhere in `zero2fsd/`):

```
$ grep -rn "matplotlib\|^import yaml\|^from yaml" zero2fsd/
zero2fsd/dashboard.py:5:from matplotlib.figure import Figure
zero2fsd/sim/topdown.py:5:from matplotlib.figure import Figure
```

Both are module-level and both are reached by `import zero2fsd.sim` (`sim/__init__.py:5` imports `topdown`) and `import zero2fsd.grade` (`grade/_drive.py:7` imports `dashboard`). `course/*/figures.py` and `scripts/` also import matplotlib but are not in `bundle.zip` (only `zero2fsd/**/*.py` is), so they stay as they are.

**Files:**
- Modify: `zero2fsd/sim/topdown.py:5` (delete the import), `:14` (import where the figure is made)
- Modify: `zero2fsd/dashboard.py:4-5` (imports), `:9-11` (type-checking import), `:21` (import where the figure is made)
- Create: `tests/zero2fsd/test_browser_import_surface.py`

**Interfaces:**
- Consumes: `zero2fsd/sim/topdown.py:5,14` (`Figure` used once, only when `ax is None`); `zero2fsd/dashboard.py:5,12,21` (`Figure` in the return annotation and at `Figure(figsize=(12, 8))`); `zero2fsd.grade.check(exercise_id, fn_or_box, show=True) -> Result` at `grade/__init__.py:19`; `render_topdown(road, trajectories=(), labels=(), ax=None)` and `dashboard(*tels, labels=None)` keep their signatures and behaviour.
- Produces: `import zero2fsd.sim`, `zero2fsd.car`, `zero2fsd.grade`, `zero2fsd.score`, `zero2fsd.dashboard` work with `matplotlib` and `yaml` absent. matplotlib loads on the first call that draws (`dashboard(...)`, or `render_topdown(...)` without `ax`). `BROWSER_MODULES` in the new test is the list each later task extends: Task 3 adds `"zero2fsd.export"`, Task 4 adds `"zero2fsd.palette"`, Task 5 adds `"zero2fsd.preview"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/zero2fsd/test_browser_import_surface.py
"""The browser (Pyodide) build has numpy and pillow but neither matplotlib nor pyyaml until asked.

Each test starts a fresh interpreter in which the named package cannot be imported (`sys.modules[name] = None`
makes `import name` raise ImportError), imports every module the playground shim imports, and runs the
learner-facing paths: `grade.check(..., show=False)` and a full drive.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BROWSER_ABSENT = ("matplotlib", "yaml")
BROWSER_MODULES = (
    "zero2fsd", "zero2fsd.sim", "zero2fsd.car", "zero2fsd.grade", "zero2fsd.score", "zero2fsd.dashboard",
)
SCRIPT = """
import importlib, sys

absent, *modules = sys.argv[1:]
sys.modules[absent] = None  # from here on `import <absent>` raises ImportError, as it does in the browser
for module in modules:
    importlib.import_module(module)

import numpy as np
from zero2fsd.car import Car, run
from zero2fsd.grade import check


def lane_error_stats(lat):
    a = np.abs(lat)
    return {"mean_abs": float(a.mean()), "max_abs": float(a.max()), "rms": float(np.sqrt(np.mean(lat ** 2)))}


result = check("0.1.1.a", lane_error_stats, show=False)
assert result.passed, result.details
assert run(Car(), "straight").completed
"""


@pytest.mark.parametrize("absent", BROWSER_ABSENT)
def test_the_package_imports_and_runs_without(absent):
    done = subprocess.run([sys.executable, "-c", SCRIPT, absent, *BROWSER_MODULES],
                          capture_output=True, text=True, cwd=REPO_ROOT, timeout=120)
    if done.returncode:
        pytest.fail(done.stderr[-2000:], pytrace=False)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/zero2fsd/test_browser_import_surface.py -v`
Expected: `[matplotlib]` FAILED with `ModuleNotFoundError: No module named 'matplotlib.figure'; 'matplotlib' is not a package` (from `sim/topdown.py` line 5); `[yaml]` PASSED. That is "1 failed, 1 passed": no module in `zero2fsd/` imports yaml today, so the second case is a regression guard that is green from the start.

- [ ] **Step 3: Write the minimal implementation**

`zero2fsd/sim/topdown.py`, delete line 5 and import where the figure is made (hunks at lines 2-8 and 11-17 today):

```diff
@@ -2,7 +2,6 @@
 from __future__ import annotations
 
 import numpy as np
-from matplotlib.figure import Figure
 
 from .world import LANE_WIDTH, Road
 
@@ -11,7 +10,11 @@ EDGE_COLOR, DASH_COLOR = "#666666", "#c9a800"
 
 def render_topdown(road: Road, trajectories=(), labels=(), ax=None):
     """Draw `road` and each (N, 2) trajectory on `ax` (a new figure's axes if None); returns the Axes."""
-    ax = ax if ax is not None else Figure(figsize=(6, 6)).subplots()
+    if ax is None:
+        # imported here, not at module level: the browser build loads matplotlib only when a figure is drawn
+        from matplotlib.figure import Figure
+
+        ax = Figure(figsize=(6, 6)).subplots()
     s = np.linspace(0.0, road.length, max(2, int(road.length)))
     for lateral, style in ((-LANE_WIDTH, "-"), (LANE_WIDTH, "-"), (0.0, (0, (5, 3)))):
         x, y = road.offset(s, lateral).T
```

`zero2fsd/dashboard.py`, three hunks. Lines 1-12 (`Figure` stays importable for the return annotation, which `from __future__ import annotations` keeps unevaluated):

```diff
 """One figure that shows how a run (or several) went."""
 from __future__ import annotations
 
+from typing import TYPE_CHECKING
+
 import numpy as np
-from matplotlib.figure import Figure
 
 from .car.runner import Telemetry
 from .score import OFF_LANE_LAT, metrics_table, run_labels
 from .sim import make_scenario, render_topdown
 
+if TYPE_CHECKING:  # `dashboard` imports it when called: the browser build loads matplotlib only on demand
+    from matplotlib.figure import Figure
 
+
 def dashboard(*tels: Telemetry, labels=None) -> Figure:
```

Lines 20-21, import after the argument check so the `ValueError` paths never need matplotlib:

```diff
     names = run_labels(tels, labels)
+    from matplotlib.figure import Figure
+
     fig = Figure(figsize=(12, 8))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/zero2fsd/test_browser_import_surface.py tests/zero2fsd/test_score.py tests/zero2fsd/test_sim.py -v`
Expected: PASS. The two new tests pass; `test_score.py` and `test_sim.py` still draw real figures (`dashboard(...)` at `test_score.py:65`, `render_topdown(...)` at `test_sim.py:263`), which proves matplotlib still loads on demand.

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests -q`
Expected: all green; number of tests = 231 + 2 = 233.

- [ ] **Step 6: Commit**

```bash
git add zero2fsd/sim/topdown.py zero2fsd/dashboard.py tests/zero2fsd/test_browser_import_surface.py
git commit -m "refactor(zero2fsd): import matplotlib where a figure is drawn, not at module level

import zero2fsd.sim and zero2fsd.grade no longer load matplotlib, so the
browser build can skip it until a dashboard is asked for. A subprocess test
blocks matplotlib and yaml and still runs check() and a drive.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Road sampling and export

> NOTE: the spec (4.1 c) names the method `Road.centerline_points(spacing_m)`. The plan uses `Road.sample_points(spacing_m, lateral=0.0)` so one method serves all five exported lines (`lateral=0.0` is the centre line). The spec's "length within 0.1 %" test is kept and runs for every scenario.

> NOTE: `road_dict` rounds coordinates to `COORD_DECIMALS = 3` (millimetres). Reason: unrounded the three roads' five lines are 382 KB of JSON, rounded 206 KB (measured), and the committed `roads.json` no longer changes with the libm of whoever rebuilds it. Shape and keys are unchanged.

**Files:**
- Modify: `zero2fsd/sim/world.py:57-60` (add a method after `offset`)
- Create: `zero2fsd/export.py`
- Modify: `tests/zero2fsd/test_browser_import_surface.py:15-17` (add `"zero2fsd.export"` to `BROWSER_MODULES`)
- Test: `tests/zero2fsd/test_export.py`

**Interfaces:**
- Consumes: `Road.offset(s, d) -> (..., 2)` at `world.py:57-60`; `Road.closed` (`:32`), `Road.length` (`:40`), `Road.dash_period` (`:42`); constants `LANE_WIDTH`, `EGO_LANE_CENTER`, `WHITE_LINE_LATERAL`, `ROAD_HALF_WIDTH`, `DASH_FRACTION` at `world.py:13-19`; `OFF_ROAD_OFFSET = 1.8` at `car/runner.py:15`; `SCENARIOS` (`scenarios.py:73`), `make_scenario(name)` (`:76`, raises `ValueError("unknown scenario 'x'; valid names: ...")`), `Scenario.route_length` (`:26-28`), `Scenario.start_s` (`:21`).
- Produces: `Road.sample_points(spacing_m: float, lateral: float = 0.0) -> np.ndarray`; in `zero2fsd/export.py`: `ROAD_SPACING_M = 0.5`, `COORD_DECIMALS = 3`, `LINES: dict[str, float]` (keys `tarmac_left`, `tarmac_right`, `yellow`, `white`, `ego_center`), `road_dict(name) -> dict`, `roads_dict() -> dict[str, dict]`. `road_dict` keys: `name, closed, length, route_length, start_s, dash_period, dash_fraction, lane_width, ego_lane_center, road_half_width, off_road_offset, lines`. For B6, Task 8 writes `roads.json` from `roads_dict()` and Task 10's `view.js` draws `lines`.

- [ ] **Step 1: Write the failing test**

Extend the import-surface list from Task 2 (`tests/zero2fsd/test_browser_import_surface.py`, lines 15-17):

```diff
 BROWSER_MODULES = (
     "zero2fsd", "zero2fsd.sim", "zero2fsd.car", "zero2fsd.grade", "zero2fsd.score", "zero2fsd.dashboard",
+    "zero2fsd.export",
 )
```

Then the new test file:

```python
# tests/zero2fsd/test_export.py
"""Road sampling (`Road.sample_points`) and the JSON-ready road export the browser view draws."""
import json
import math

import numpy as np
import pytest

from zero2fsd.car.runner import OFF_ROAD_OFFSET
from zero2fsd.export import COORD_DECIMALS, LINES, ROAD_SPACING_M, road_dict, roads_dict
from zero2fsd.sim.scenarios import SCENARIOS, make_scenario
from zero2fsd.sim.world import (
    DASH_FRACTION, EGO_LANE_CENTER, LANE_WIDTH, ROAD_HALF_WIDTH, WHITE_LINE_LATERAL, Road,
)

NAMES = sorted(SCENARIOS)


def segment_lengths(points, closed):
    pts = np.asarray(points, float)
    if closed:
        pts = np.vstack([pts, pts[:1]])  # the segment back to the start line
    return np.hypot(*np.diff(pts, axis=0).T)


# --- Road.sample_points -------------------------------------------------------------------------------------------

def straight_10m():
    return Road([[0, 0], [5, 0], [10, 0]], closed=False)


def square_40m():  # counter-clockwise, so "left" is the inside
    return Road([[0, 0], [10, 0], [10, 10], [0, 10]], closed=True)


def test_an_open_road_ends_exactly_at_its_end():
    pts = straight_10m().sample_points(2.5)
    assert pts.shape == (5, 2)
    np.testing.assert_allclose(pts, [[0, 0], [2.5, 0], [5, 0], [7.5, 0], [10, 0]], atol=1e-12)


def test_an_open_road_whose_length_is_not_a_multiple_of_the_spacing_still_ends_at_its_end():
    pts = straight_10m().sample_points(4.0)
    np.testing.assert_allclose(pts, [[0, 0], [4, 0], [8, 0], [10, 0]], atol=1e-12)


def test_a_closed_road_stops_before_its_length_so_the_first_point_closes_the_loop():
    pts = square_40m().sample_points(15.0)
    np.testing.assert_allclose(pts, [[0, 0], [10, 5], [0, 10]], atol=1e-12)  # s = 0, 15, 30; not s = 40
    assert len(square_40m().sample_points(10.0)) == 4  # a multiple of the spacing: s = 40 is the first point again


def test_lateral_is_positive_to_the_left_of_travel_on_an_open_and_a_closed_road():
    np.testing.assert_allclose(straight_10m().sample_points(5.0, lateral=1.0), [[0, 1], [5, 1], [10, 1]], atol=1e-12)
    np.testing.assert_allclose(square_40m().sample_points(10.0, lateral=1.0), [[0, 1], [9, 0], [10, 9], [1, 10]], atol=1e-12)


# --- road_dict / roads_dict ---------------------------------------------------------------------------------------

def test_every_scenario_is_exported_and_the_result_is_plain_json():
    exported = roads_dict()
    assert sorted(exported) == NAMES
    assert json.loads(json.dumps(exported)) == exported  # floats, lists, bools only: nothing numpy


def test_an_unknown_road_is_a_value_error_naming_the_valid_ones():
    with pytest.raises(ValueError, match="unknown scenario 'nope'.*gentle"):
        road_dict("nope")


@pytest.mark.parametrize("name", NAMES)
def test_the_scalars_come_from_the_simulator_not_from_retyped_numbers(name):
    sc, d = make_scenario(name), road_dict(name)
    assert (d["name"], d["closed"], d["length"]) == (name, sc.road.closed, sc.road.length)
    assert (d["route_length"], d["start_s"], d["dash_period"]) == (sc.route_length, sc.start_s, sc.road.dash_period)
    assert (d["dash_fraction"], d["lane_width"], d["ego_lane_center"]) == (DASH_FRACTION, LANE_WIDTH, EGO_LANE_CENTER)
    assert (d["road_half_width"], d["off_road_offset"]) == (ROAD_HALF_WIDTH, OFF_ROAD_OFFSET)
    assert tuple(d["lines"]) == tuple(LINES)


def test_the_closed_and_the_open_scenarios_differ_in_the_fields_the_view_needs():
    assert road_dict("gentle")["closed"] is True and road_dict("gentle")["route_length"] == road_dict("gentle")["length"]
    assert road_dict("straight")["closed"] is False and road_dict("straight")["route_length"] < road_dict("straight")["length"]


def test_lines_sit_at_their_lateral_offsets_left_positive():
    d = road_dict("straight")  # a road along +x, so a line at lateral L has y == L
    for key, lateral in LINES.items():
        np.testing.assert_allclose(np.asarray(d["lines"][key])[:, 1], lateral, atol=10 ** -COORD_DECIMALS)
    assert LINES["tarmac_left"] > 0 > LINES["tarmac_right"] and LINES["ego_center"] == EGO_LANE_CENTER
    assert LINES["white"] == WHITE_LINE_LATERAL and LINES["yellow"] == 0.0


@pytest.mark.parametrize("name", ["gentle", "curvy"])  # on curves a line's lateral is what Road.project reports
def test_curved_lines_keep_their_lateral_offset(name):
    road, d = make_scenario(name).road, road_dict(name)
    for key, lateral in LINES.items():
        for x, y in d["lines"][key][::37]:
            assert road.project(x, y)[1] == pytest.approx(lateral, abs=5e-3), (key, x, y)


@pytest.mark.parametrize("name", NAMES)
def test_points_are_spaced_by_the_export_spacing(name):
    gaps = segment_lengths(road_dict(name)["lines"]["yellow"], closed=False)
    np.testing.assert_allclose(gaps[:-1], ROAD_SPACING_M, atol=2e-3)  # chords of the arc, to the rounding
    assert 0 < gaps[-1] <= ROAD_SPACING_M + 2e-3  # open roads: the remainder before the end; closed: one more full step


@pytest.mark.parametrize("name", NAMES)
def test_the_end_point_rule(name):
    road, d = make_scenario(name).road, road_dict(name)
    yellow = np.asarray(d["lines"]["yellow"])
    if d["closed"]:
        gap = float(np.hypot(*(yellow[-1] - yellow[0])))
        assert 0 < gap <= ROAD_SPACING_M + 2e-3  # the last point is short of the start line, never on it
    else:
        np.testing.assert_allclose(yellow[-1], road.offset(road.length, 0.0), atol=10 ** -COORD_DECIMALS)


@pytest.mark.parametrize("name", NAMES)
def test_the_exported_centre_line_is_as_long_as_the_road(name):
    d = road_dict(name)
    total = segment_lengths(d["lines"]["yellow"], d["closed"]).sum()
    assert math.isclose(total, d["length"], rel_tol=1e-3), (total, d["length"])


def test_coordinates_are_rounded_to_the_export_precision():
    for line in road_dict("gentle")["lines"].values():
        arr = np.asarray(line)
        assert np.array_equal(arr, np.round(arr, COORD_DECIMALS))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/zero2fsd/test_export.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'zero2fsd.export'`.

Run: `python -m pytest tests/zero2fsd/test_browser_import_surface.py -v`
Expected: both cases FAILED, stderr tail ends `ModuleNotFoundError: No module named 'zero2fsd.export'`.

- [ ] **Step 3: Write the minimal implementation**

`zero2fsd/sim/world.py`, a new method directly after `offset` (lines 57-60 today, `_locate` follows at 62):

```diff
     def offset(self, s, d):
         """World point(s) at arc length `s` and lateral `d` (left positive); shape (..., 2)."""
         x, y, h = self.pose_at(s)
         return np.stack([x - np.sin(h) * d, y + np.cos(h) * d], axis=-1)
 
+    def sample_points(self, spacing_m: float, lateral: float = 0.0) -> np.ndarray:
+        """(N, 2) world points every `spacing_m` of arc length at `lateral` (left positive).
+
+        Closed roads stop before `length` (the first point closes the loop); open roads end at `length`.
+        """
+        s = np.arange(0.0, self.length, spacing_m)
+        if not self.closed:
+            s = np.append(s, self.length)
+        return self.offset(s, lateral)
+
     def _locate(self, x, y):
```

New file `zero2fsd/export.py`:

```python
"""The road as plain data for the browser view: the polylines the top-down plot draws, ready for `json.dumps`."""
from __future__ import annotations

from .car.runner import OFF_ROAD_OFFSET
from .sim.scenarios import SCENARIOS, make_scenario
from .sim.world import DASH_FRACTION, EGO_LANE_CENTER, LANE_WIDTH, ROAD_HALF_WIDTH, WHITE_LINE_LATERAL

ROAD_SPACING_M = 0.5  # arc length between exported points
COORD_DECIMALS = 3  # millimetres: finer than a canvas shows, and the committed roads.json does not change with the libm
LINES = {  # lateral offsets (left positive) of the polylines the top-down view draws
    "tarmac_left": ROAD_HALF_WIDTH, "tarmac_right": -ROAD_HALF_WIDTH,
    "yellow": 0.0, "white": WHITE_LINE_LATERAL, "ego_center": EGO_LANE_CENTER,
}


def road_dict(name: str) -> dict:
    """One scenario's road as JSON-ready data; an unknown `name` is the ValueError `make_scenario` raises."""
    sc = make_scenario(name)
    road = sc.road
    return {
        "name": sc.name, "closed": road.closed, "length": float(road.length),
        "route_length": float(sc.route_length), "start_s": float(sc.start_s),
        "dash_period": float(road.dash_period), "dash_fraction": DASH_FRACTION,
        "lane_width": LANE_WIDTH, "ego_lane_center": EGO_LANE_CENTER,
        "road_half_width": ROAD_HALF_WIDTH, "off_road_offset": OFF_ROAD_OFFSET,
        "lines": {key: road.sample_points(ROAD_SPACING_M, lateral).round(COORD_DECIMALS).tolist()
                  for key, lateral in LINES.items()},
    }


def roads_dict() -> dict[str, dict]:
    """Every scenario in `SCENARIOS`, keyed by name."""
    return {name: road_dict(name) for name in SCENARIOS}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/zero2fsd/test_export.py tests/zero2fsd/test_browser_import_surface.py -v`
Expected: PASS (25 tests: 23 in `test_export.py`, 2 in the import-surface file; about 4 s).

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests -q`
Expected: all green; number of tests = 233 + 23 = 256 (verified: the same three tasks applied to a copy of c3de7d4 gave `256 passed`).

- [ ] **Step 6: Commit**

```bash
git add zero2fsd/sim/world.py zero2fsd/export.py tests/zero2fsd/test_export.py tests/zero2fsd/test_browser_import_surface.py
git commit -m "feat(zero2fsd): Road.sample_points and export.roads_dict for the browser road view

One sampling method serves the five drawn lines (tarmac edges, yellow, white,
ego centre). Closed roads stop before the start line, open roads end on their
end. Every scenario's exported centre line is within 0.1 % of road.length.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

**Tasks 4 to 6b: palette, preview registry, verdict margins, streamed gate**

Baseline for this part: `256 passed`, the whole suite after Task 3 (226 + 5 from Task 1 + 2 from Task 2 + 23 from Task 3). Every "whole suite" step uses `python -m pytest tests -q` (see the suite-command NOTE above Task 1). Expected totals:

| After task | Tests added | Whole suite (`python -m pytest tests -q`) |
|---|---|---|
| Task 3 | | 256 |
| Task 4: palette | 7 | 263 |
| Task 5: preview registry | 21 | 284 |
| Task 6: margins and borderline note | 30 | 314 |
| Task 6b: stream the graded drive | 8 | 322 |

These totals are verified: Tasks 4, 5, 6 and 6b applied in order to a copy of the repo gave 228, 235, 256, 286, 294 passed. The copy lacked Task 1's and Task 3's test files (28 tests), and 294 + 28 = 322. Every code block below is the code of that copy, and every `diff` on a grade file or on `test_grade.py` and `test_margins.py` is a real `diff -u` between the states. The two import-surface hunks are written against the file as it stands after Task 3.

Line numbers: the grade files and `tests/zero2fsd/test_grade.py` are untouched by Tasks 1 to 3, so their numbers are those of the primary checkout. `tests/zero2fsd/test_browser_import_surface.py` numbers are those after Task 3.

> NOTE (the gate constant): the repo already names the drive gate `GATE_SCENARIO, GATE_MEAN_LAT = "gentle", 0.5` at `zero2fsd/grade/_drive.py:11`, imported by tests and the playground tasks. The plan keeps those names and adds no second constant for the limit.

---

### Task 4: Palette

> NOTE: an early draft had `CHROME = {"background": "#0d1117", "panel": "#090d16", ...}`, with background and panel swapped against the spec (`docs/superpowers/specs/2026-10-11-playground-design.md:96`: page `#090d16`, surfaces `#0d1117`) and against the site task (`bg-[#090d16]`). The plan uses the spec values, and the test pins "the page is darker than its panels" so a swap cannot come back unnoticed.

**Files:**
- Create: `zero2fsd/palette.py`
- Modify: `tests/zero2fsd/test_browser_import_surface.py:15-18` (after Task 3 the tuple is lines 15-18; add `"zero2fsd.palette"` as new line 18, so it ends on line 19)
- Test: `tests/zero2fsd/test_palette.py`

**Interfaces:**
- Consumes: `SKY, GRASS, ROAD, YELLOW, WHITE = range(5)` at `zero2fsd/sim/camera.py:16` and `PALETTE` (one RGB row per index) at `:17`. `BROWSER_MODULES`, the tuple Task 2 defines in `tests/zero2fsd/test_browser_import_surface.py:15` (Task 2 Step 1); Task 3 appended `"zero2fsd.export"` as line 17. (Task 3's Files line cites `:17-19`; the tuple is lines 15-17 before Task 3 and 15-18 after it.)
- Produces: `QUANTITY`, `SCENE`, `CHROME` (each `dict[str, str]` of `#RRGGBB`) and `palette_dict() -> {"quantity": QUANTITY, "scene": SCENE, "chrome": CHROME}` in `zero2fsd/palette.py`. Task 8 writes `palette.json` from `palette_dict()`. Task 10's `view.js` reads `scene.*` for the road paint and `quantity.*` for the traces and gauges. `SCENE` is derived from the camera's own `PALETTE`, so the top-down view paints the road in the colours the camera sees.

- [ ] **Step 1: Write the failing test**

First extend the import-surface list (`tests/zero2fsd/test_browser_import_surface.py`, lines 15-18). This makes the browser guarantee (no matplotlib, no yaml) cover the new module:

```diff
--- a/tests/zero2fsd/test_browser_import_surface.py
+++ b/tests/zero2fsd/test_browser_import_surface.py
@@ -15,4 +15,5 @@
 BROWSER_MODULES = (
     "zero2fsd", "zero2fsd.sim", "zero2fsd.car", "zero2fsd.grade", "zero2fsd.score", "zero2fsd.dashboard",
     "zero2fsd.export",
+    "zero2fsd.palette",
 )
```

Then the new test file:

```python
# tests/zero2fsd/test_palette.py
"""The palette module: every colour is `#RRGGBB`, the paint colours are the camera's, and it is JSON-ready."""
import json
import re

import pytest

from zero2fsd.palette import CHROME, QUANTITY, SCENE, palette_dict
from zero2fsd.sim.camera import GRASS, PALETTE, ROAD, SKY, WHITE, YELLOW

HEX = re.compile(r"#[0-9A-Fa-f]{6}")
SCENE_INDEX = {"sky": SKY, "grass": GRASS, "road": ROAD, "yellow": YELLOW, "white": WHITE}


def rgb(hex_colour):
    return tuple(int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))


@pytest.mark.parametrize("group", [QUANTITY, SCENE, CHROME], ids=["quantity", "scene", "chrome"])
def test_every_colour_is_a_six_digit_hex_string(group):
    assert group and all(HEX.fullmatch(value) for value in group.values())


def test_scene_holds_one_colour_per_camera_palette_entry_and_equals_it():
    assert set(SCENE) == set(SCENE_INDEX)
    for name, index in SCENE_INDEX.items():
        assert rgb(SCENE[name]) == tuple(int(c) for c in PALETTE[index]), name


def test_the_four_quantities_are_told_apart_by_colour():
    assert set(QUANTITY) == {"e_off", "psi", "delta", "perception"}
    assert len(set(QUANTITY.values())) == len(QUANTITY)


def test_the_page_background_is_darker_than_its_panels():
    assert sum(rgb(CHROME["background"])) < sum(rgb(CHROME["panel"]))


def test_palette_dict_names_its_three_groups_and_survives_json():
    palette = palette_dict()
    assert list(palette) == ["quantity", "scene", "chrome"]
    assert json.loads(json.dumps(palette)) == palette
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/zero2fsd/test_palette.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'zero2fsd.palette'` ("Interrupted: 1 error during collection").

Run: `python -m pytest tests/zero2fsd/test_browser_import_surface.py -v`
Expected: both cases FAILED (2 failed), the subprocess stderr ends `ModuleNotFoundError: No module named 'zero2fsd.palette'`.

- [ ] **Step 3: Write the minimal implementation**

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/zero2fsd/test_palette.py tests/zero2fsd/test_browser_import_surface.py -v`
Expected: PASS (9 tests: 7 in `test_palette.py`, 2 in the import-surface file).

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests -q`
Expected: all green; number of tests = 256 + 7 = 263.

- [ ] **Step 6: Commit**

```bash
git add zero2fsd/palette.py tests/zero2fsd/test_palette.py tests/zero2fsd/test_browser_import_surface.py
git commit -m "feat(zero2fsd): palette module, the one place the browser view gets its colours

Paint colours come from the camera's PALETTE, so the top-down view paints
the road in the colours the camera sees. The browser import-surface test
now covers the module.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Preview registry

> NOTE (stats signature): the spec says a stats preview is `build(fn) -> str`. The plan uses `build(fn, tel) -> dict[str, float]` instead: the page prints the labelled numbers, and a registry that returns numbers can be tested without parsing text.

> NOTE (stats road): stats previews carry `road="curvy"` (`STATS_ROAD`). The 0.1.1 lab computes its stats on `curvy` (`course/0.1.1-meet-your-car/lab.py:44`, `:132`), and `curvy` is the only road where the provided car leaves the lane, so a count of off-lane steps is not 0 there. Measured Oct 11 2026 with the provided `Car()`: straight 502 steps, 0 off lane; gentle 1778 steps, 0 off lane, mean |lat| 0.376 m; curvy 123 steps, 40 off lane, mean |lat| 0.549 m, not completed. Task 7's `preview_stats` and `canned_telemetry` must run `get(exercise_id).road`, not a hard-coded `gentle`.

> NOTE (two helpers made public): the preview reuses two checks the graders already own, so they become public instead of being copied. `u0_1_2._as_masks` becomes `as_masks` (the image preview validates the learner's masks with it). A new `_drive.require_car(car)` holds the "my_car must be a Car" test that sat inline in `u0_1_4._my_car`; the 0.1.4.b grader and the 0.1.4.b preview now share one check and one message.

**Files:**
- Create: `zero2fsd/preview/__init__.py`
- Create: `zero2fsd/preview/units.py`
- Modify: `zero2fsd/grade/_drive.py:48-49` (insert `require_car` between `require_own` at `:42-47` and `gate` at `:50`; the new lines are 50-55)
- Modify: `zero2fsd/grade/units/u0_1_2.py:19` and `:42` (`_as_masks` to `as_masks`)
- Modify: `zero2fsd/grade/units/u0_1_4.py:8`, `:10`, `:59-60` (use `require_car`)
- Modify: `tests/zero2fsd/test_browser_import_surface.py:15-19` (add `"zero2fsd.preview"`; the tuple then ends on line 20)
- Test: `tests/zero2fsd/test_preview.py`

**Interfaces:**
- Consumes: `EXERCISES: dict[str, Callable]` at `zero2fsd/grade/_report.py:37`, `Failure` at `:28`; `Car` (`car/car.py:20`) and `Car.uses_provided() -> list[str]` (`:27`); `provided.K_OFF, K_HEAD = 0.35, 2.7` (`car/provided.py:23`), `paint_masks` (`:26`), `estimate_lane` (`:44`), `steer_p` (`:60`); `GATE_SCENARIO` (`grade/_drive.py:11`); `metrics(tel)` with keys `mean_abs_lat`, `max_abs_lat`, `steps_off_lane` (`score.py:11`); the "import the units at the bottom" pattern of `grade/__init__.py:12-13`.
- Produces, in `zero2fsd/preview/__init__.py`: `Kind = Literal["drive", "stats", "image"]`; frozen dataclass `Preview(kind, build, road="")`; `PREVIEWS: dict[str, Preview]`; decorator `register(exercise_id, kind, road="")` (returns `build` untouched, raises `ValueError` on a second preview for one id); `get(exercise_id) -> Preview` (`KeyError` listing the known ids). In `preview/units.py`: one preview for each of the seven ids:

| id | kind | road | `build` |
|---|---|---|---|
| `0.1.1.a` | stats | `curvy` | `build(fn, tel) -> {label: float}` of `fn(tel.lat.copy())` |
| `0.1.1.b` | stats | `curvy` | same, a bare number is labelled `steps_off_lane` |
| `0.1.2.a` | image | none | `build(fn, frame) -> uint8[H, W, 3]`, the learner's yellow and white masks painted over a copy |
| `0.1.3.a` | drive | `gentle` | `Car(perception=fn)` |
| `0.1.3.b` | drive | `gentle` | `Car(perception=fn)` |
| `0.1.4.a` | drive | `gentle` | `Car(controller=lambda est, obs: fn(est, K_OFF, K_HEAD))` |
| `0.1.4.b` | drive | `gentle` | `require_car(car)`, the learner's own `Car` |

  Also `STATS_ROAD = "curvy"` and `OVERLAY = {"yellow": (255, 0, 255), "white": (0, 255, 255)}` in `preview/units.py`, `as_masks` in `u0_1_2.py`, `require_car` in `_drive.py`. The class guard is `test_every_graded_exercise_has_exactly_one_preview`: a future exercise registered in `EXERCISES` without a preview fails it, so the registry cannot fall behind by construction.

- [ ] **Step 1: Write the failing test**

Extend the import-surface list again (`tests/zero2fsd/test_browser_import_surface.py`, lines 15-19 after Task 4):

```diff
--- a/tests/zero2fsd/test_browser_import_surface.py
+++ b/tests/zero2fsd/test_browser_import_surface.py
@@ -15,5 +15,6 @@
 BROWSER_MODULES = (
     "zero2fsd", "zero2fsd.sim", "zero2fsd.car", "zero2fsd.grade", "zero2fsd.score", "zero2fsd.dashboard",
     "zero2fsd.export",
     "zero2fsd.palette",
+    "zero2fsd.preview",
 )
```

Then the new test file. The drive previews are tested with wrapper functions, because `provided.estimate_lane` itself would count as the provided slot, not as the learner's:

```python
# tests/zero2fsd/test_preview.py
"""The preview registry mirrors the grade registry: every graded exercise has a preview of the right kind.

The playground runs these previews on the learner's own code (no verdict): drive previews build the Car it
drives, stats previews turn a canned drive into labelled numbers, image previews draw the learner's masks.
"""
import re
import typing

import numpy as np
import pytest

import zero2fsd.preview as preview
from zero2fsd.car import Car, run
from zero2fsd.car.provided import K_HEAD, K_OFF, estimate_lane, paint_masks, steer_p
from zero2fsd.grade._report import EXERCISES, Failure
from zero2fsd.preview import PREVIEWS, Kind, get, register
from zero2fsd.preview.units import OVERLAY, STATS_ROAD
from zero2fsd.score import metrics
from zero2fsd.sim import BicycleState, Camera, make_scenario
from zero2fsd.sim.scenarios import SCENARIOS

KINDS = {
    "0.1.1.a": "stats", "0.1.1.b": "stats", "0.1.2.a": "image",
    "0.1.3.a": "drive", "0.1.3.b": "drive", "0.1.4.a": "drive", "0.1.4.b": "drive",
}
STEPS = 100  # enough steps to tell one car from another without driving a whole lap
STILL_PROVIDED = {"0.1.3.a": ["controller"], "0.1.3.b": ["controller"], "0.1.4.a": ["perception"], "0.1.4.b": ["perception", "controller"]}

# Each answer behaves as the provided box but is a different object, so the Car counts the slot as the learner's.
own_perception = lambda obs: estimate_lane(obs)  # noqa: E731
own_steer = lambda est, k_off, k_head: steer_p(est, k_off, k_head)  # noqa: E731
PROVIDED_ANSWER = {"0.1.3.a": own_perception, "0.1.3.b": own_perception, "0.1.4.a": own_steer, "0.1.4.b": Car()}


def ids_of(kind):
    return [i for i, p in sorted(PREVIEWS.items()) if p.kind == kind]


def good_lane_error_stats(lat):
    return {"mean_abs": float(np.abs(lat).mean()), "max_abs": float(np.abs(lat).max()), "rms": float(np.sqrt(np.mean(lat**2)))}


def good_steps_off_lane(lat, threshold=0.9):
    return int(np.count_nonzero(np.abs(lat) > threshold))


STATS_ANSWER = {"0.1.1.a": good_lane_error_stats, "0.1.1.b": good_steps_off_lane}


# --- the registry -------------------------------------------------------------------------------------------------

def test_every_graded_exercise_has_exactly_one_preview():
    assert set(PREVIEWS) == set(EXERCISES)


def test_each_exercise_has_the_kind_its_lab_needs():
    assert {i: p.kind for i, p in PREVIEWS.items()} == KINDS
    assert {p.kind for p in PREVIEWS.values()} <= set(typing.get_args(Kind))


def test_drive_and_stats_previews_name_a_real_scenario_and_image_previews_none():
    for exercise_id, p in PREVIEWS.items():
        assert (p.road == "") if p.kind == "image" else (p.road in SCENARIOS), exercise_id


def test_an_unknown_id_is_a_key_error_that_lists_the_known_ids():
    with pytest.raises(KeyError) as err:
        get("9.9.9.z")
    assert all(i in str(err.value) for i in KINDS)


def test_register_stores_the_function_untouched_and_refuses_a_second_preview(monkeypatch):
    monkeypatch.setattr(preview, "PREVIEWS", {})
    build = lambda fn: Car()  # noqa: E731
    assert register("x.1", "drive", road="gentle")(build) is build
    assert preview.get("x.1") == preview.Preview("drive", build, "gentle")
    with pytest.raises(ValueError, match="x.1"):
        register("x.1", "drive")(build)


# --- drive previews: a Car whose learner-owned slots are the learner's code --------------------------------------

def same_drive(a: Car, b: Car, road: str):
    ta, tb = run(a, road, max_steps=STEPS), run(b, road, max_steps=STEPS)
    return np.array_equal(ta.lat, tb.lat) and np.array_equal(ta.steer, tb.steer)


@pytest.mark.parametrize("exercise_id", ids_of("drive"))
def test_a_drive_preview_builds_a_car_that_drives_its_road_and_leaves_the_other_slot_provided(exercise_id):
    p = get(exercise_id)
    car = p.build(PROVIDED_ANSWER[exercise_id])
    assert isinstance(car, Car)
    assert car.uses_provided() == STILL_PROVIDED[exercise_id]
    assert same_drive(car, Car(), p.road)  # each answer is the provided box in disguise, so the car drives as the provided one


def test_the_steering_preview_uses_the_provided_gains_and_the_provided_perception():
    seen = []

    def my_steer(est, k_off, k_head):
        seen.append((k_off, k_head))
        return 0.0

    run(get("0.1.4.a").build(my_steer), "gentle", max_steps=STEPS)
    assert seen == [(K_OFF, K_HEAD)] * STEPS


def test_the_all_yours_preview_returns_the_learners_own_car_and_refuses_anything_else():
    mine = Car(perception=estimate_lane)
    assert get("0.1.4.b").build(mine) is mine  # a preview may run on the provided boxes too: only the check demands own ones
    with pytest.raises(Failure, match="my_car must be a Car.*got NoneType"):
        get("0.1.4.b").build(None)


# --- stats previews: the learner's function applied to the canned drive ------------------------------------------

@pytest.fixture(scope="module")
def canned():
    return run(Car(), STATS_ROAD)


@pytest.mark.parametrize("road", ["straight", STATS_ROAD])
@pytest.mark.parametrize("exercise_id", ids_of("stats"))
def test_a_stats_preview_returns_labelled_floats(exercise_id, road):
    values = get(exercise_id).build(STATS_ANSWER[exercise_id], run(Car(), road, max_steps=STEPS))
    assert values and all(isinstance(k, str) and type(v) is float for k, v in values.items())


def test_the_stats_previews_agree_with_the_score_module_on_the_canned_drive(canned):
    ref = metrics(canned)
    stats = get("0.1.1.a").build(good_lane_error_stats, canned)
    assert stats["mean_abs"] == pytest.approx(ref["mean_abs_lat"]) and stats["max_abs"] == pytest.approx(ref["max_abs_lat"])
    assert get("0.1.1.b").build(good_steps_off_lane, canned) == {"steps_off_lane": float(ref["steps_off_lane"])}
    assert ref["steps_off_lane"] > 0  # the canned drive has something to count: on `gentle` this lab line would read 0


def test_a_stats_preview_labels_a_bare_number_and_keeps_a_dicts_own_labels(canned):
    assert get("0.1.1.b").build(lambda lat: 7, canned) == {"steps_off_lane": 7.0}
    assert get("0.1.1.a").build(lambda lat: {"mean_abs": 1, "bias": -0.5}, canned) == {"mean_abs": 1.0, "bias": -0.5}


def test_a_stats_preview_fails_loudly_on_a_non_number_and_never_changes_the_telemetry(canned):
    def scribble(lat):
        lat.fill(0.0)
        return 0

    before = canned.lat.copy()
    with pytest.raises(TypeError):
        get("0.1.1.a").build(lambda lat: None, canned)
    get("0.1.1.b").build(scribble, canned)
    assert np.array_equal(canned.lat, before)


# --- image previews: the learner's masks drawn on the frame ------------------------------------------------------

@pytest.fixture(scope="module")
def noisy():
    road = make_scenario("gentle").road
    return Camera().render(road, BicycleState(*road.ego_pose(5.0, 0.3, 0.02), 8.0, 0.0), noise_sigma=6.0, seed=3)


def test_the_image_preview_paints_each_mask_in_its_own_colour_and_leaves_the_rest(noisy):
    before = noisy.copy()
    shown = get("0.1.2.a").build(paint_masks, noisy)
    yellow, white = paint_masks(noisy)
    assert shown.dtype == np.uint8 and shown.shape == noisy.shape
    assert yellow.any() and white.any() and not (yellow & white).any()
    assert (shown[yellow] == OVERLAY["yellow"]).all() and (shown[white] == OVERLAY["white"]).all()
    rest = ~(yellow | white)
    assert np.array_equal(shown[rest], noisy[rest]) and np.array_equal(noisy, before)


@pytest.mark.parametrize("answer, said", [
    (lambda frame: None, "returned NoneType"),
    (lambda frame: (np.zeros((2, 2), bool), np.zeros((2, 2), bool)), "the yellow mask must be a bool array of shape (180, 320)"),
])
def test_the_image_preview_reports_a_wrong_answer_as_a_failure(noisy, answer, said):
    with pytest.raises(Failure, match=re.escape(said)):
        get("0.1.2.a").build(answer, noisy)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/zero2fsd/test_preview.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'zero2fsd.preview'` ("Interrupted: 1 error during collection").

Run: `python -m pytest tests/zero2fsd/test_browser_import_surface.py -v`
Expected: both cases FAILED (2 failed), stderr ends `ModuleNotFoundError: No module named 'zero2fsd.preview'`.

- [ ] **Step 3: Write the minimal implementation**

First the three grade-file edits. They are behaviour-preserving, and `test_grade.py` (69 tests) is their guard.

```diff
--- a/zero2fsd/grade/_drive.py
+++ b/zero2fsd/grade/_drive.py
@@ -47,6 +47,13 @@
                       given=f"Car with provided: {car.uses_provided()}")
 
 
+def require_car(car) -> Car:
+    """The all-yours exercise's answer must be a Car (it is `None` until the learner builds one); returns it."""
+    if not isinstance(car, Car):
+        raise Failure(f"my_car must be a Car(perception=..., controller=...), got {type(car).__name__}")
+    return car
+
+
 def gate(car: Car, show: bool) -> str:
     """Fail unless the car completes the gate scenario with mean |lat| under the limit; else say how it did."""
     tel = drive(car, GATE_SCENARIO, f"your car on {GATE_SCENARIO}", show)
```

```diff
--- a/zero2fsd/grade/units/u0_1_2.py
+++ b/zero2fsd/grade/units/u0_1_2.py
@@ -16,7 +16,7 @@
     return np.count_nonzero(a & b) / union if union else 1.0  # both empty: nothing to find, nothing found
 
 
-def _as_masks(got, shape, given):
+def as_masks(got, shape, given):
     """The learner's answer as two bool masks of the frame's shape, or a Failure saying what is off."""
     masks = list(got) if isinstance(got, tuple | list) else None
     if masks is None or len(masks) != 2:
@@ -39,7 +39,7 @@
     for k, pose in enumerate(poses):
         frame, labels = camera.render(pose.road, pose.state, noise_sigma=NOISE_SIGMA, seed=k, return_labels=True)
         given = f"a {frame.shape[1]}x{frame.shape[0]} frame rendered at {pose}"
-        masks = _as_masks(call(fn, frame.copy(), given=given), frame.shape[:2], given)
+        masks = as_masks(call(fn, frame.copy(), given=given), frame.shape[:2], given)
         for name, mask, paint in zip(("yellow", "white"), masks, (YELLOW, WHITE)):
             truth = labels == paint
             iou = _iou(mask, truth)
```

```diff
--- a/zero2fsd/grade/units/u0_1_4.py
+++ b/zero2fsd/grade/units/u0_1_4.py
@@ -5,9 +5,9 @@
 
 import numpy as np
 
-from ...car import Car, LaneEstimate
+from ...car import LaneEstimate
 from ...sim.dynamics import MAX_STEER
-from .._drive import drive, gate, outcome, require_own
+from .._drive import drive, gate, outcome, require_car, require_own
 from .._report import Failure, call, exercise, fmt, is_real
 
 TOL = 1e-9
@@ -56,8 +56,7 @@
     hint='lecture 0.1.4, section "Closing the loop"',
 )
 def _my_car(car, show):
-    if not isinstance(car, Car):
-        raise Failure(f"my_car must be a Car(perception=..., controller=...), got {type(car).__name__}")
+    require_car(car)
     require_own(car)
     passed = gate(car, show)
     curvy = drive(car, "curvy", "your cliffhanger on curvy", show)
```

Then the registry package:

```python
# zero2fsd/preview/__init__.py
"""The playground's preview registry: how each graded exercise runs the learner's code without grading it.

It mirrors `grade.EXERCISES`, one entry per exercise id.  `Preview.kind` says what the learner sees:
  drive  `build(learner_obj) -> Car`, which the playground drives on `road` and streams step by step
  stats  `build(learner_fn, tel) -> {label: number}`, the learner's function applied to a drive on `road`
  image  `build(learner_fn, frame) -> uint8[H, W, 3]`, the learner's function drawn on a camera frame
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

Kind = Literal["drive", "stats", "image"]


@dataclass(frozen=True)
class Preview:
    kind: Kind
    build: Callable[..., Any]
    road: str = ""  # drive and stats: the scenario the preview drives; image: empty (the caller picks the frame)


PREVIEWS: dict[str, Preview] = {}


def register(exercise_id: str, kind: Kind, road: str = "") -> Callable:
    """Decorator: store the function as the preview of `exercise_id`; a second preview for one id is a bug."""

    def store(build):
        if exercise_id in PREVIEWS:
            raise ValueError(f"exercise {exercise_id!r} already has a preview")
        PREVIEWS[exercise_id] = Preview(kind, build, road)
        return build

    return store


def get(exercise_id: str) -> Preview:
    if exercise_id not in PREVIEWS:
        raise KeyError(f"unknown exercise {exercise_id!r}; known: {', '.join(sorted(PREVIEWS))}")
    return PREVIEWS[exercise_id]


from . import units  # noqa: E402,F401  (importing it registers every exercise's preview)
```

```python
# zero2fsd/preview/units.py
"""One preview per graded exercise (grade/units/u0_1_*.py): the learner's code, run for a look instead of a verdict."""
from __future__ import annotations

from collections.abc import Mapping

from ..car import Car, provided
from ..grade._drive import GATE_SCENARIO, require_car
from ..grade.units.u0_1_2 import as_masks
from . import register

STATS_ROAD = "curvy"  # the lab's own stats drive: `gentle` never leaves the lane, so a count of off-lane steps would be 0
OVERLAY = {"yellow": (255, 0, 255), "white": (0, 255, 255)}  # where the learner's (yellow, white) masks say the paint is


def _as_stats(got, name: str) -> dict[str, float]:
    """A function's answer as {label: float}: a dict keeps its keys, a single number is labelled `name`."""
    pairs = got.items() if isinstance(got, Mapping) else [(name, got)]
    return {str(label): float(value) for label, value in pairs}


@register("0.1.1.a", "stats", road=STATS_ROAD)
def _lane_error_stats(fn, tel):
    return _as_stats(fn(tel.lat.copy()), "lane_error_stats")


@register("0.1.1.b", "stats", road=STATS_ROAD)
def _steps_off_lane(fn, tel):
    return _as_stats(fn(tel.lat.copy()), "steps_off_lane")


@register("0.1.2.a", "image")
def _paint_masks(fn, frame):
    masks = as_masks(fn(frame.copy()), frame.shape[:2], "the preview frame")
    shown = frame.copy()
    for colour, mask in zip(OVERLAY.values(), masks):
        shown[mask] = colour
    return shown


@register("0.1.3.a", "drive", road=GATE_SCENARIO)
@register("0.1.3.b", "drive", road=GATE_SCENARIO)
def _own_perception(fn):
    return Car(perception=fn)


@register("0.1.4.a", "drive", road=GATE_SCENARIO)
def _own_steering(fn):
    return Car(controller=lambda est, obs: fn(est, provided.K_OFF, provided.K_HEAD))


@register("0.1.4.b", "drive", road=GATE_SCENARIO)
def _own_car(car):
    return require_car(car)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/zero2fsd/test_preview.py tests/zero2fsd/test_browser_import_surface.py tests/zero2fsd/test_grade.py -v`
Expected: PASS (92 tests: 21 in `test_preview.py`, 2 in the import-surface file, 69 in `test_grade.py`).

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests -q`
Expected: all green; number of tests = 263 + 21 = 284.

- [ ] **Step 6: Commit**

```bash
git add zero2fsd/preview zero2fsd/grade/_drive.py zero2fsd/grade/units/u0_1_2.py zero2fsd/grade/units/u0_1_4.py tests/zero2fsd/test_preview.py tests/zero2fsd/test_browser_import_surface.py
git commit -m "feat(zero2fsd): preview registry, one entry per graded exercise

Drive previews build the Car the playground drives, stats previews turn a
canned curvy drive into labelled numbers, the image preview paints the
learner's masks on a frame. as_masks and require_car are shared with the
graders instead of copied.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Verdict margins and the borderline note

> NOTE (signature): `borderline(value, threshold)` takes no `unit`: the spec fixes the sentence verbatim (`docs/superpowers/specs/2026-10-11-playground-design.md:70`), so the note never mentions a number or a unit. The metric, the limit and the margin are printed by a second helper, `vs_limit(value, limit, unit="")`, which appends the sentence only when the margin is within `BORDERLINE_FRACTION`. `margin(value, limit)` and `BORDERLINE_NOTE` are exported for the test.

> NOTE (the "lift the literals" step): the gate scenario and limit were already constants (`_drive.py:11`), and the pose sampler, tolerances and hit rate already sit in `u0_1_3.py:13` and `u0_1_2.py:11`. What remained were the same numbers re-typed inside the `checked=` strings (`u0_1_3.py:46`, `u0_1_4.py:55`). Those now interpolate the constants, so the printed limit, the checked limit and the test limit are one value.

> NOTE (three checks): the drive gate, the hit rate and, from spec 4.1 (e), the paint IoU (`u0_1_2.py`, `IOU_MIN = 0.85`). The test covers all three. To reach the numbers the graders compute without a second copy, each grader's loop moves into a generator the test also calls: `pose_trials(fn)` in `u0_1_3.py` and `paint_trials(fn)` in `u0_1_2.py`. The graders now consume those generators.

> NOTE (reference solutions): the margin test measures the reference solutions through `load_solution(REPO, unit)` (`scripts/_course.py:39`, the same helper `tests/zero2fsd/test_solutions.py:43` uses). The graders never read them. Measured Oct 11 2026: gate mean |lat| 0.3757 m (24.9 % under 0.5), hit rate 40 of 40 (11 % over 90), lowest IoU 1.000 (18 % over 0.85). All clear `MARGIN = 0.05`.

**Files:**
- Modify: `zero2fsd/grade/_report.py:53-58` (insert the margin helpers after `is_real`, which ends at `:55`, before `@contextmanager` at `:58`)
- Modify: `zero2fsd/grade/_drive.py:9` (import), `:57-65` (`gate`: print the limit and margin)
- Modify: `zero2fsd/grade/units/u0_1_2.py:9` (import), `:31-52` (generator plus checker)
- Modify: `zero2fsd/grade/units/u0_1_3.py:9`, `:11` (imports), `:14` (insert two helpers after the constants), `:22-41` (checker), `:46` (`checked=`)
- Modify: `zero2fsd/grade/units/u0_1_4.py:10` (import), `:55` (`checked=`)
- Test: `tests/zero2fsd/test_margins.py`

**Interfaces:**
- Consumes: `GATE_SCENARIO, GATE_MEAN_LAT` (`_drive.py:11`); `N_POSES, MIN_HIT_RATE, OFFSET_TOL, HEADING_TOL = 40, 0.9, 0.25, 0.05` (`u0_1_3.py:13`); `NOISE_SIGMA, IOU_MIN, N_FRAMES = 6.0, 0.85, 6` (`u0_1_2.py:11`); `sample_poses` (`grade/_poses.py:37`); `metrics(tel)["mean_abs_lat"]` (`score.py:11`); `load_solution(root, unit_id)` and `REPO` (`scripts/_course.py:39`, `:12`; `tests/zero2fsd/conftest.py:7` puts `scripts/` on `sys.path`).
- Produces, in `zero2fsd/grade/_report.py`: `BORDERLINE_FRACTION = 0.02`; `BORDERLINE_NOTE = "Borderline: tiny differences between computers can flip this. Aim for a clearer margin."`; `margin(value, limit) -> float` (`abs(value - limit) / abs(limit)`); `borderline(value, threshold) -> str | None`; `vs_limit(value, limit, unit="") -> str`, e.g. `(limit 0.5 m, margin 25 %)`, followed by `. ` and the sentence when borderline. Also `pose_trials(fn)` in `u0_1_3.py` (yields `(pose, given, est, hit)`) and `paint_trials(fn)` in `u0_1_2.py` (yields `(given, name, mask, truth, iou)`). Every thresholded detail line now reads metric, limit and margin; real output:
  - `✓ 0.1.4.b: completed gentle, score 58, mean |lat| 0.376 m (limit 0.5 m, margin 25 %); cliffhanger (not graded): left the lane after 48 of 218 m of curvy`
  - `✓ 0.1.3.a: 40 of 40 poses within 0.25 m and 0.05 rad: hit rate 100.0 % (limit 90 %, margin 11 %)`
  - `✓ 0.1.2.a: both masks overlap the true paint on all 6 noisy frames; lowest IoU 1.000 (limit 0.85, margin 18 %)`
  - `✗ 0.1.2.a: the yellow mask overlaps the true yellow paint too little: IoU 0.839 (limit 0.85, margin 1.3 %). Borderline: tiny differences between computers can flip this. Aim for a clearer margin.`

- [ ] **Step 1: Write the failing test**

The test imports the real constants, so a change to any limit moves the test with it. Stats checks are not in it: they compare the learner with a reference computed in the same runtime (tolerance 1e-9), so no computer can split them (the module docstring says so).

```python
# tests/zero2fsd/test_margins.py   (this is the Task 6 version; Task 6b edits one line)
"""Verdict margins: the checks that compare a simulation-derived number with a threshold.

Three checks do (the drive gate, the perception hit rate, the paint IoU).  A verdict that sits too close to its
threshold could read PASS on one computer and FAIL on another, so the reference solution must clear each by
MARGIN, and a learner's answer that lands within BORDERLINE_FRACTION gets the borderline sentence.

The other checks compare the learner's function with a reference computed in the same runtime (tolerance 1e-9),
so the computer cannot split them and they need no margin.
"""
import dataclasses
import operator

import numpy as np
import pytest

from _course import REPO, load_solution
from zero2fsd.car import Car, LaneEstimate, provided, run
from zero2fsd.grade import _drive, check
from zero2fsd.grade._drive import GATE_MEAN_LAT, GATE_SCENARIO
from zero2fsd.grade._report import BORDERLINE_FRACTION, BORDERLINE_NOTE, borderline, margin, vs_limit
from zero2fsd.grade.units.u0_1_2 import IOU_MIN, paint_trials
from zero2fsd.grade.units.u0_1_3 import MIN_HIT_RATE, N_POSES, pose_trials
from zero2fsd.score import metrics

MARGIN = 0.05  # the reference solution clears every threshold by at least this fraction of the threshold


def gate_mean_lat():
    tel = run(load_solution(REPO, "0.1.4").my_car, GATE_SCENARIO)
    assert tel.completed
    return metrics(tel)["mean_abs_lat"]


def hit_rate():
    return sum(hit for *_, hit in pose_trials(load_solution(REPO, "0.1.3").estimate_lane)) / N_POSES


def lowest_iou():
    return min(iou for *_, iou in paint_trials(load_solution(REPO, "0.1.2").paint_masks))


# check -> (measure on the reference solution, its threshold, the side of the threshold that passes)
THRESHOLDED = {
    "drive gate: mean |lat|": (gate_mean_lat, GATE_MEAN_LAT, operator.lt),
    "perception hit rate": (hit_rate, MIN_HIT_RATE, operator.ge),
    "paint IoU": (lowest_iou, IOU_MIN, operator.ge),
}


@pytest.mark.parametrize("name", THRESHOLDED)
def test_the_reference_solution_clears_every_thresholded_check_by_the_margin(name):
    measure, threshold, passes = THRESHOLDED[name]
    value = measure()
    assert passes(value, threshold), f"{name}: {value} against {threshold}"
    assert margin(value, threshold) >= MARGIN, f"{name}: {value} is only {margin(value, threshold):.1%} from {threshold}"


# --- the sentence and the printed margin ---------------------------------------------------------------------------

@pytest.mark.parametrize("value, near", [(0.509, True), (0.491, True), (0.5, True), (0.512, False), (0.488, False), (0.0, False)])
def test_a_value_within_two_percent_of_the_threshold_is_borderline_on_either_side(value, near):
    assert BORDERLINE_FRACTION == 0.02
    assert borderline(value, 0.5) == (BORDERLINE_NOTE if near else None)


def test_the_borderline_sentence_is_the_one_the_spec_fixes_and_a_negative_threshold_works():
    assert BORDERLINE_NOTE == "Borderline: tiny differences between computers can flip this. Aim for a clearer margin."
    assert borderline(-1.01, -1.0) == BORDERLINE_NOTE and borderline(-1.1, -1.0) is None


@pytest.mark.parametrize(
    "value, limit, unit, printed",
    [
        (0.376, 0.5, "m", "(limit 0.5 m, margin 25 %)"),
        (0.92, 0.85, "", "(limit 0.85, margin 8.2 %)"),
        (0.0, 0.5, "m", "(limit 0.5 m, margin 100 %)"),
        (92.5, 90, "%", "(limit 90 %, margin 2.8 %)"),
        (0.499, 0.5, "m", f"(limit 0.5 m, margin 0.2 %). {BORDERLINE_NOTE}"),
    ],
)
def test_vs_limit_prints_the_limit_and_the_margin_and_the_sentence_only_when_borderline(value, limit, unit, printed):
    assert vs_limit(value, limit, unit) == printed


# --- every thresholded line of a real check prints metric, limit and margin ---------------------------------------

def text(result):
    return "\n".join(result.details)


def own_perception(obs):
    return provided.estimate_lane(obs)


def blind_for(first: int):
    """Provided perception that sees nothing on its first `first` frames, so exactly 40 - first poses are hits."""
    calls = []

    def perception(obs):
        calls.append(1)
        return LaneEstimate(float("nan"), float("nan"), False) if len(calls) <= first else provided.estimate_lane(obs)

    return perception


def keeping(share: float):
    """Provided paint masks with all but the first `share` of each mask's pixels dropped: IoU is about `share`."""

    def paint_masks(frame):
        kept = []
        for mask in provided.paint_masks(frame):
            out = np.zeros_like(mask)
            out.flat[np.flatnonzero(mask)[: int(share * np.count_nonzero(mask))]] = True
            kept.append(out)
        return tuple(kept)

    return paint_masks


# (check, learner, passes?, borderline?, what the printed line says)
@pytest.mark.parametrize(
    "exercise_id, learner, passed, near, says",
    [
        ("0.1.3.a", provided.estimate_lane, True, False, "hit rate 100.0 % (limit 90 %, margin 11 %)"),
        ("0.1.3.a", blind_for(4), True, True, "hit rate 90.0 % (limit 90 %, margin 0.0 %)"),
        ("0.1.3.a", blind_for(5), False, False, "hit rate 87.5 % (limit 90 %, margin 2.8 %)"),
        ("0.1.2.a", provided.paint_masks, True, False, "IoU 1.000 (limit 0.85, margin 18 %)"),
        ("0.1.2.a", keeping(0.86), True, True, "(limit 0.85, margin 1."),
        ("0.1.2.a", keeping(0.84), False, True, "(limit 0.85, margin 1."),
        ("0.1.2.a", keeping(0.5), False, False, "(limit 0.85, margin 41 %)"),
    ],
)
def test_the_hit_rate_and_the_iou_print_their_limit_and_margin_and_flag_a_thin_one(exercise_id, learner, passed, near, says):
    result = check(exercise_id, learner, show=False)
    assert result.passed is passed, text(result)
    assert says in text(result), text(result)
    assert (BORDERLINE_NOTE in text(result)) is near, text(result)


def tel_that(make_tel, *, completed, mean_lat):
    tel = make_tel(lat=[mean_lat, mean_lat], progress=[0.0, 100.0], scenario="gentle", route_length=100.0)
    return dataclasses.replace(tel, completed=completed)


@pytest.mark.parametrize("exercise_id, subject", [
    ("0.1.3.b", own_perception),
    ("0.1.4.b", Car(perception=own_perception, controller=lambda est, obs: provided.steer_p(est))),
])
@pytest.mark.parametrize(
    "completed, mean_lat, passed, near, says",
    [
        (True, 0.376, True, False, "mean |lat| 0.376 m (limit 0.5 m, margin 25 %)"),
        (True, 0.495, True, True, "mean |lat| 0.495 m (limit 0.5 m, margin 1.0 %)"),
        (True, 0.505, False, True, "mean |lat| 0.505 m (limit 0.5 m, margin 1.0 %)"),
        (False, 0.8, False, False, "mean |lat| 0.800 m (limit 0.5 m, margin 60 %)"),
    ],
)
def test_both_gate_exercises_print_the_gate_limit_and_margin_and_flag_a_thin_one(
        monkeypatch, make_tel, exercise_id, subject, completed, mean_lat, passed, near, says):
    monkeypatch.setattr(_drive, "run", lambda car, scenario: tel_that(make_tel, completed=completed, mean_lat=mean_lat))
    result = check(exercise_id, subject, show=False)
    assert result.passed is passed, text(result)
    assert says in text(result), text(result)
    assert (BORDERLINE_NOTE in text(result)) is near, text(result)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/zero2fsd/test_margins.py -v`
Expected: collection error, `ImportError: cannot import name 'BORDERLINE_FRACTION' from 'zero2fsd.grade._report'`.

- [ ] **Step 3: Write the minimal implementation**

The helpers in `_report.py`:

```diff
--- a/zero2fsd/grade/_report.py
+++ b/zero2fsd/grade/_report.py
@@ -55,6 +55,29 @@
     return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
 
 
+BORDERLINE_FRACTION = 0.02  # a verdict within this fraction of its limit can flip between computers
+BORDERLINE_NOTE = "Borderline: tiny differences between computers can flip this. Aim for a clearer margin."
+
+
+def margin(value: float, limit: float) -> float:
+    """How far `value` is from `limit`, as a fraction of the limit."""
+    return abs(value - limit) / abs(limit)
+
+
+def borderline(value: float, threshold: float) -> str | None:
+    """The borderline sentence when `value` is within BORDERLINE_FRACTION of `threshold`, else None."""
+    return BORDERLINE_NOTE if margin(value, threshold) <= BORDERLINE_FRACTION else None
+
+
+def vs_limit(value: float, limit: float, unit: str = "") -> str:
+    """`(limit 0.5 m, margin 25 %)`, then the borderline sentence when the margin is that thin."""
+    percent = 100 * margin(value, limit)
+    shown = f"{percent:.1f}" if percent < 10 else f"{percent:.0f}"
+    text = f"(limit {limit:g}{' ' + unit if unit else ''}, margin {shown} %)"
+    note = borderline(value, limit)
+    return f"{text}. {note}" if note else text
+
+
 @contextmanager
 def learner_code(given: str):
     """Run learner code: NotImplementedError and any other exception become a `Failure` naming the input."""
```

The gate prints its limit and margin (the `Failure.got` text and the pass message share one `seen` string):

```diff
--- a/zero2fsd/grade/_drive.py
+++ b/zero2fsd/grade/_drive.py
@@ -6,7 +6,7 @@
 from ..car import Car, Telemetry, run
 from ..dashboard import dashboard
 from ..score import driving_score, metrics, metrics_table
-from ._report import Failure, learner_code
+from ._report import Failure, learner_code, vs_limit
 
 GATE_SCENARIO, GATE_MEAN_LAT = "gentle", 0.5  # the car must complete the scenario with mean |lat| below this (m)
 
@@ -58,8 +58,9 @@
     """Fail unless the car completes the gate scenario with mean |lat| under the limit; else say how it did."""
     tel = drive(car, GATE_SCENARIO, f"your car on {GATE_SCENARIO}", show)
     mean_lat = metrics(tel)["mean_abs_lat"]
+    seen = f"mean |lat| {mean_lat:.3f} m {vs_limit(mean_lat, GATE_MEAN_LAT, 'm')}"
     if not tel.completed or mean_lat >= GATE_MEAN_LAT:
         raise Failure(f"your car did not pass the {GATE_SCENARIO} gate",
                       expected=f"completes {GATE_SCENARIO} with mean |lat| < {GATE_MEAN_LAT} m",
-                      got=f"{outcome(tel)}, mean |lat| {mean_lat:.2f} m", given=f"your car driving {GATE_SCENARIO!r}")
-    return f"{outcome(tel)}, mean |lat| {mean_lat:.2f} m (< {GATE_MEAN_LAT} m), score {driving_score(tel)}"
+                      got=f"{outcome(tel)}, {seen}", given=f"your car driving {GATE_SCENARIO!r}")
+    return f"{outcome(tel)}, score {driving_score(tel)}, {seen}"
```

The paint IoU, as a generator plus a thin checker:

```diff
--- a/zero2fsd/grade/units/u0_1_2.py
+++ b/zero2fsd/grade/units/u0_1_2.py
@@ -6,7 +6,7 @@
 from ...sim import Camera
 from ...sim.camera import WHITE, YELLOW
 from .._poses import sample_poses
-from .._report import Failure, call, exercise
+from .._report import Failure, call, exercise, vs_limit
 
 NOISE_SIGMA, IOU_MIN, N_FRAMES = 6.0, 0.85, 6
 
@@ -28,12 +28,8 @@
     return masks
 
 
-@exercise(
-    "0.1.2.a",
-    checked=f"yellow and white masks on {N_FRAMES} noisy (sigma={NOISE_SIGMA:g}) frames against the renderer's true paint, IoU >= {IOU_MIN}",
-    hint='lecture 0.1.2, section "Paint is a colour threshold"',
-)
-def _paint_masks(fn, show):
+def paint_trials(fn):
+    """Run `fn` on the seeded noisy frames: yield (given, name, mask, truth, iou) for the yellow, then the white mask of each."""
     camera, rng = Camera(), np.random.default_rng(201)
     poses = sample_poses(rng, N_FRAMES, ("straight", "gentle", "curvy"), straight_only=False)
     for k, pose in enumerate(poses):
@@ -42,11 +38,26 @@
         masks = as_masks(call(fn, frame.copy(), given=given), frame.shape[:2], given)
         for name, mask, paint in zip(("yellow", "white"), masks, (YELLOW, WHITE)):
             truth = labels == paint
-            iou = _iou(mask, truth)
-            if iou < IOU_MIN:
-                raise Failure(
-                    f"the {name} mask overlaps the true {name} paint too little (IoU {iou:.2f}, need {IOU_MIN})", given=given,
-                    expected=f"{np.count_nonzero(truth)} {name} pixels",
-                    got=f"{np.count_nonzero(mask)} pixels marked, {np.count_nonzero(mask & truth)} of them on {name} paint",
-                )
-    return f"both masks reach IoU >= {IOU_MIN} on all {N_FRAMES} noisy frames"
+            yield given, name, mask, truth, _iou(mask, truth)
+
+
+def _iou_text(iou: float) -> str:
+    return f"IoU {iou:.3f} {vs_limit(iou, IOU_MIN)}"
+
+
+@exercise(
+    "0.1.2.a",
+    checked=f"yellow and white masks on {N_FRAMES} noisy (sigma={NOISE_SIGMA:g}) frames against the renderer's true paint, IoU >= {IOU_MIN}",
+    hint='lecture 0.1.2, section "Paint is a colour threshold"',
+)
+def _paint_masks(fn, show):
+    worst = 1.0  # an IoU cannot be higher
+    for given, name, mask, truth, iou in paint_trials(fn):
+        if iou < IOU_MIN:
+            raise Failure(
+                f"the {name} mask overlaps the true {name} paint too little: {_iou_text(iou)}", given=given,
+                expected=f"{np.count_nonzero(truth)} {name} pixels",
+                got=f"{np.count_nonzero(mask)} pixels marked, {np.count_nonzero(mask & truth)} of them on {name} paint",
+            )
+        worst = min(worst, iou)
+    return f"both masks overlap the true paint on all {N_FRAMES} noisy frames; lowest {_iou_text(worst)}"
```

The perception hit rate, and the `checked=` literal replaced by the constants:

```diff
--- a/zero2fsd/grade/units/u0_1_3.py
+++ b/zero2fsd/grade/units/u0_1_3.py
@@ -6,13 +6,30 @@
 from ...car import Car, LaneEstimate, Observation
 from ...sim import Camera
 from ...sim.scenarios import TARGET_SPEED
-from .._drive import gate, require_own
+from .._drive import GATE_MEAN_LAT, GATE_SCENARIO, gate, require_own
 from .._poses import sample_poses
-from .._report import Failure, call, exercise, learner_code
+from .._report import Failure, call, exercise, learner_code, vs_limit
 
 N_POSES, MIN_HIT_RATE, OFFSET_TOL, HEADING_TOL = 40, 0.9, 0.25, 0.05
 
 
+def pose_trials(fn):
+    """Run `fn` on the seeded straight-road poses: yield (pose, given, est, hit) for each; a non-LaneEstimate is a Failure."""
+    camera, rng = Camera(), np.random.default_rng(301)
+    for pose in sample_poses(rng, N_POSES, ("straight", "gentle"), straight_only=True):
+        given = f"an Observation from a frame rendered at {pose}"
+        est = call(fn, Observation(camera.render(pose.road, pose.state), TARGET_SPEED, 0.0), given=given)
+        if not isinstance(est, LaneEstimate):
+            raise Failure(f"returned {type(est).__name__}, expected a LaneEstimate", given=given, got=est)
+        yield pose, given, est, bool(est.valid and abs(est.offset_m - pose.offset) < OFFSET_TOL
+                                     and abs(est.heading_rad - pose.heading) < HEADING_TOL)
+
+
+def _hit_rate_text(hits: int) -> str:
+    rate = 100 * hits / N_POSES
+    return f"hit rate {rate:.1f} % {vs_limit(rate, 100 * MIN_HIT_RATE, '%')}"
+
+
 @exercise(
     "0.1.3.a",
     checked=(f"offset and heading on {N_POSES} seeded straight-road poses (up to 1 m off centre, up to 0.15 rad yawed): "
@@ -20,30 +37,24 @@
     hint='lecture 0.1.3, section "From masks to a lane estimate"',
 )
 def _estimate_lane(fn, show):
-    camera, rng = Camera(), np.random.default_rng(301)
     hits, first_miss = 0, None
-    for pose in sample_poses(rng, N_POSES, ("straight", "gentle"), straight_only=True):
-        given = f"an Observation from a frame rendered at {pose}"
-        est = call(fn, Observation(camera.render(pose.road, pose.state), TARGET_SPEED, 0.0), given=given)
-        if not isinstance(est, LaneEstimate):
-            raise Failure(f"returned {type(est).__name__}, expected a LaneEstimate", given=given, got=est)
-        if est.valid and abs(est.offset_m - pose.offset) < OFFSET_TOL and abs(est.heading_rad - pose.heading) < HEADING_TOL:
-            hits += 1
-        elif first_miss is None:
+    for pose, given, est, hit in pose_trials(fn):
+        hits += hit
+        if not hit and first_miss is None:
             first_miss = (pose, est, given)
     if hits < MIN_HIT_RATE * N_POSES:
         pose, est, given = first_miss
         raise Failure(
-            f"only {hits} of {N_POSES} poses were within tolerance (need {MIN_HIT_RATE:.0%}); first miss shown", given=given,
+            f"only {hits} of {N_POSES} poses were within tolerance: {_hit_rate_text(hits)}", given=f"first miss: {given}",
             expected=f"offset {pose.offset:+.2f} m (within {OFFSET_TOL}), heading {pose.heading:+.3f} rad (within {HEADING_TOL}), valid=True",
             got=f"offset {est.offset_m:+.2f} m, heading {est.heading_rad:+.3f} rad, valid={est.valid}",
         )
-    return f"{hits} of {N_POSES} poses within {OFFSET_TOL} m and {HEADING_TOL} rad"
+    return f"{hits} of {N_POSES} poses within {OFFSET_TOL} m and {HEADING_TOL} rad: {_hit_rate_text(hits)}"
 
 
 @exercise(
     "0.1.3.b",
-    checked="your perception inside the provided car on `gentle`: it must complete with mean |lat| < 0.5 m",
+    checked=f"your perception inside the provided car on `{GATE_SCENARIO}`: it must complete with mean |lat| < {GATE_MEAN_LAT} m",
     hint='lecture 0.1.3, section "Swap your perception into the car"',
 )
 def _perception_gate(fn, show):
```

The `checked=` literal in 0.1.4.b:

```diff
--- a/zero2fsd/grade/units/u0_1_4.py
+++ b/zero2fsd/grade/units/u0_1_4.py
@@ -7,7 +7,7 @@
 
 from ...car import LaneEstimate
 from ...sim.dynamics import MAX_STEER
-from .._drive import drive, gate, outcome, require_car, require_own
+from .._drive import GATE_MEAN_LAT, GATE_SCENARIO, drive, gate, outcome, require_car, require_own
 from .._report import Failure, call, exercise, fmt, is_real
 
 TOL = 1e-9
@@ -52,7 +52,7 @@
 
 @exercise(
     "0.1.4.b",
-    checked="my_car (your perception and your controller) on `gentle`: it must complete with mean |lat| < 0.5 m; then `curvy` is shown, not graded",
+    checked=f"my_car (your perception and your controller) on `{GATE_SCENARIO}`: it must complete with mean |lat| < {GATE_MEAN_LAT} m; then `curvy` is shown, not graded",
     hint='lecture 0.1.4, section "Closing the loop"',
 )
 def _my_car(car, show):
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/zero2fsd/test_margins.py tests/zero2fsd/test_grade.py -v`
Expected: PASS (99 tests: 30 in `test_margins.py`, 69 in `test_grade.py`; the existing grader tests need no edit; `test_grade.py:140` still finds `first miss`, which moved from the 0.1.3.a headline into its `given` line).

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests -q`
Expected: all green; number of tests = 284 + 30 = 314.

- [ ] **Step 6: Commit**

```bash
git add zero2fsd/grade/_report.py zero2fsd/grade/_drive.py zero2fsd/grade/units/u0_1_2.py zero2fsd/grade/units/u0_1_3.py zero2fsd/grade/units/u0_1_4.py tests/zero2fsd/test_margins.py
git commit -m "feat(zero2fsd): every thresholded verdict prints metric, limit and margin

The drive gate, the perception hit rate and the paint IoU show how far they
sit from their limit, and add the borderline sentence within 2 %. A margin
test holds the reference solutions at least 5 % clear of all three limits.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6b: Stream the graded drive

> NOTE (the brief's example): the dispatch brief's `check("0.1.4.b", Car(), ...)` never reaches a drive: `require_own` fails first on the provided boxes (`_drive.py:42`). The positive streaming tests therefore use a Car with the learner's own boxes. The `Car()` case stays as the negative test: it fails before driving and streams nothing.

> NOTE (what streams): only the gate drive streams (0.1.3.b and 0.1.4.b). The curvy cliffhanger after the gate in 0.1.4.b is not streamed (spec section 5), so the `drive(car, "curvy", ...)` call at `u0_1_4.py:62` is unchanged. Task 7's `check_code` must forward `on_step` to `check(...)`.

> NOTE (uniform grader signature): every graded checker now takes `(target, show, on_step)`, and `exercise()` passes all three positionally. The dispatch in `check` stays one call with no per-exercise cases, and a future driving exercise gets the hook by passing `on_step` to `drive` or `gate`. Non-driving graders accept the argument and never call it.

> NOTE (existing tests): 15 existing tests replace `_drive.run` with `lambda car, scenario: ...`. Once `drive` passes `on_step=`, those lambdas raise `TypeError: ... got an unexpected keyword argument 'on_step'` (6 + 1 in `test_grade.py`, 8 in `test_margins.py`; verified). Their signature becomes `lambda car, scenario, on_step=None: ...`. The edit belongs to Step 1, and it is harmless before the implementation.

**Files:**
- Modify: `zero2fsd/grade/__init__.py:4-5` (import), `:18-25` (`check`), `:22` (the dispatch)
- Modify: `zero2fsd/grade/_report.py:3` (docstring), `:37` (`EXERCISES` type), `:109` (docstring), `:113` (`graded`), `:115` (the checker call)
- Modify: `zero2fsd/grade/_drive.py:4` (import), `:32-37` (`drive`), `:57-59` (`gate`)
- Modify: `zero2fsd/grade/units/u0_1_1.py:29`, `:49`; `u0_1_2.py:53`; `u0_1_3.py:39`, `:60`, `:64`; `u0_1_4.py:41`, `:58`, `:61`
- Modify: `tests/zero2fsd/test_grade.py:9-10` (imports), `:208`, `:218` (lambdas), new block after `:240`
- Modify: `tests/zero2fsd/test_margins.py:160` (lambda)

**Interfaces:**
- Consumes: `run(..., on_step=None)` and `COLUMNS` from Task 1 (`zero2fsd/car/runner.py`); `drive`, `gate` (`_drive.py:32`, `:57`); `exercise()` and `EXERCISES` (`_report.py:108`, `:37`).
- Produces: `check(exercise_id, fn_or_box, show=True, on_step=None) -> Result`. `on_step(row)` receives one 14-tuple (the `COLUMNS` order) per step of the gate drive, the same rows `run` appends, before `check` returns. `drive(car, scenario, label, show, on_step=None)` and `gate(car, show, on_step=None)` forward it to `run`. `EXERCISES[id]` is `(target, show, on_step) -> Result`. Exercises that do not drive never call it.

- [ ] **Step 1: Write the failing test**

The test file edits (imports, the three lambdas, and the new streaming block with its own `own_car()` helper):

```diff
--- a/tests/zero2fsd/test_grade.py
+++ b/tests/zero2fsd/test_grade.py
@@ -6,8 +6,10 @@
 import pytest
 
 import zero2fsd.grade as grade_pkg
-from zero2fsd.car import Car, LaneEstimate, provided
+from zero2fsd.car import Car, LaneEstimate, provided, run
+from zero2fsd.car.runner import COLUMNS
 from zero2fsd.grade import _drive, check
+from zero2fsd.grade._drive import GATE_SCENARIO
 from zero2fsd.grade._poses import LOOKAHEAD, sample_poses
 from zero2fsd.grade._report import EXERCISES, NOT_IMPLEMENTED
 from zero2fsd.sim.dynamics import MAX_STEER
@@ -205,7 +207,7 @@
 )
 def test_the_gate_decides_on_completion_and_mean_lateral_error(monkeypatch, make_tel, completed, mean_lat, off_road, verdict):
     tel = telemetry_that(make_tel, completed=completed, mean_lat=mean_lat, off_road=off_road)
-    monkeypatch.setattr(_drive, "run", lambda car, scenario: tel)
+    monkeypatch.setattr(_drive, "run", lambda car, scenario, on_step=None: tel)
     if verdict == "pass":
         assert "completed gentle" in _drive.gate(Car(), show=False)
     else:
@@ -215,7 +217,7 @@
 
 
 def test_show_draws_the_dashboard_once_per_run_only_when_asked(monkeypatch, make_tel):
-    monkeypatch.setattr(_drive, "run", lambda car, scenario: telemetry_that(make_tel, completed=True, mean_lat=0.1))
+    monkeypatch.setattr(_drive, "run", lambda car, scenario, on_step=None: telemetry_that(make_tel, completed=True, mean_lat=0.1))
     shown = []
     monkeypatch.setattr(_drive, "show_figure", shown.append)
     _drive.gate(Car(), show=False)
@@ -238,6 +240,43 @@
     assert result.passed and "completed gentle" in text(result), text(result)
 
 
+# --- streaming: `on_step` sees every step of the gate drive, and only of a drive ----------------------------------
+
+def own_car():
+    return Car(perception=own_perception, controller=lambda est, obs: provided.steer_p(est))
+
+
+@pytest.mark.parametrize(
+    "exercise_id, target, reference",
+    [
+        ("0.1.3.b", own_perception, lambda: Car(perception=own_perception)),
+        ("0.1.4.b", own_car(), own_car),  # the curvy cliffhanger after the gate is not streamed
+    ],
+)
+def test_the_gate_drive_streams_every_step_to_on_step(exercise_id, target, reference):
+    rows = []
+    result = check(exercise_id, target, show=False, on_step=rows.append)
+    assert result.passed, text(result)
+    steps = len(run(reference(), GATE_SCENARIO).t)
+    assert steps > 0 and len(rows) == steps
+    assert all(len(row) == len(COLUMNS) for row in rows)
+
+
+@pytest.mark.parametrize("exercise_id", ["0.1.1.a", "0.1.1.b", "0.1.2.a", "0.1.3.a", "0.1.4.a"])
+def test_a_grader_that_does_not_drive_never_calls_on_step(exercise_id):
+    calls = []
+    result = check(exercise_id, GOOD[exercise_id], show=False, on_step=calls.append)
+    assert result.passed, text(result)
+    assert calls == []
+
+
+def test_a_car_that_fails_before_driving_streams_nothing():
+    calls = []
+    result = check("0.1.4.b", Car(), show=False, on_step=calls.append)
+    assert not result.passed and "still uses the provided" in text(result)
+    assert calls == []
+
+
 # --- the pose sampler and the grader's own hygiene ---------------------------------------------------------------
 
 def test_straight_poses_have_no_turn_in_the_next_lookahead_metres_and_are_seeded():
```

and the one lambda in `tests/zero2fsd/test_margins.py`:

```diff
--- a/tests/zero2fsd/test_margins.py
+++ b/tests/zero2fsd/test_margins.py
@@ -157,7 +157,7 @@
 )
 def test_both_gate_exercises_print_the_gate_limit_and_margin_and_flag_a_thin_one(
         monkeypatch, make_tel, exercise_id, subject, completed, mean_lat, passed, near, says):
-    monkeypatch.setattr(_drive, "run", lambda car, scenario: tel_that(make_tel, completed=completed, mean_lat=mean_lat))
+    monkeypatch.setattr(_drive, "run", lambda car, scenario, on_step=None: tel_that(make_tel, completed=completed, mean_lat=mean_lat))
     result = check(exercise_id, subject, show=False)
     assert result.passed is passed, text(result)
     assert says in text(result), text(result)
```

The streamed rows are checked against an independent `run(...)` of the same car, so the test proves the rows are the drive's own, not merely that something was called. `0.1.3.b` and `0.1.4.b` are two different gate exercises, and five non-driving graders prove "never called" across the class.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/zero2fsd/test_grade.py tests/zero2fsd/test_margins.py -v`
Expected: `8 failed, 99 passed`; each failure is `TypeError: check() got an unexpected keyword argument 'on_step'` (the 8 new tests; the lambda edits are compatible with the old `drive`).

- [ ] **Step 3: Write the minimal implementation**

The dispatch and the public signature (`grade/__init__.py`):

```diff
--- a/zero2fsd/grade/__init__.py
+++ b/zero2fsd/grade/__init__.py
@@ -3,6 +3,7 @@
 
 import importlib
 import pkgutil
+from collections.abc import Callable
 
 import numpy as np
 
@@ -15,11 +16,13 @@
 GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"
 
 
-def check(exercise_id: str, fn_or_box, show: bool = True) -> Result:
-    """Run one exercise's checks on the learner's function (or Car), print the verdict, return it."""
+def check(exercise_id: str, fn_or_box, show: bool = True, on_step: Callable[[tuple], None] | None = None) -> Result:
+    """Run one exercise's checks on the learner's function (or Car), print the verdict, return it.
+
+    `on_step` is called with each telemetry row while a graded drive runs; exercises that do not drive never call it."""
     if exercise_id not in EXERCISES:
         raise ValueError(f"unknown exercise {exercise_id!r}; known: {', '.join(sorted(EXERCISES))}")
-    result = EXERCISES[exercise_id](fn_or_box, show)
+    result = EXERCISES[exercise_id](fn_or_box, show, on_step)
     head, *rest = result.details
     print(f"{GREEN if result.passed else RED}{head}{RESET}", *rest, sep="\n")
     return result
```

The uniform checker signature (`_report.py`):

```diff
--- a/zero2fsd/grade/_report.py
+++ b/zero2fsd/grade/_report.py
@@ -1,6 +1,6 @@
 """What a graded exercise is: a checker that raises `Failure` on the first thing wrong.
 
-`exercise(...)` registers a checker in `EXERCISES` and turns it into `(target, show) -> Result`.
+`exercise(...)` registers a checker in `EXERCISES` and turns it into `(target, show, on_step) -> Result`.
 `call` and `learner_code` are the one place learner code is run, so an exception in it becomes a
 readable red result instead of a traceback and a missing implementation can never count as a pass.
 """
@@ -34,7 +34,7 @@
         self.headline, self.checked, self.shown = headline, checked, shown
 
 
-EXERCISES: dict[str, Callable[[Any, bool], Result]] = {}
+EXERCISES: dict[str, Callable[[Any, bool, Callable[[tuple], None] | None], Result]] = {}
 
 
 def fmt(value) -> str:
@@ -106,13 +106,13 @@
 
 
 def exercise(exercise_id: str, *, checked: str, hint: str):
-    """Register a checker `(target, show) -> str` (what passed); it raises `Failure` for the first problem."""
+    """Register a checker `(target, show, on_step) -> str` (what passed); it raises `Failure` for the first problem."""
 
     def register(checker):
         @functools.wraps(checker)
-        def graded(target, show):
+        def graded(target, show, on_step):
             try:
-                message = checker(target, show)
+                message = checker(target, show, on_step)
             except Failure as failure:
                 return Result(False, _lines(failure, exercise_id, checked, hint))
             return Result(True, [f"\N{CHECK MARK} {exercise_id}: {message}"])
```

The hook reaches `run` (`_drive.py`):

```diff
--- a/zero2fsd/grade/_drive.py
+++ b/zero2fsd/grade/_drive.py
@@ -2,6 +2,7 @@
 from __future__ import annotations
 
 import io
+from collections.abc import Callable
 
 from ..car import Car, Telemetry, run
 from ..dashboard import dashboard
@@ -29,10 +30,12 @@
     return f"{how} after {tel.progress_m[-1]:.0f} of {tel.route_length:.0f} m of {tel.scenario}"
 
 
-def drive(car: Car, scenario: str, label: str, show: bool) -> Telemetry:
-    """Run the car on a scenario (learner exceptions become a Failure), print its metrics and maybe the dashboard."""
+def drive(car: Car, scenario: str, label: str, show: bool, on_step: Callable[[tuple], None] | None = None) -> Telemetry:
+    """Run the car on a scenario (learner exceptions become a Failure), print its metrics and maybe the dashboard.
+
+    `on_step` receives each telemetry row as the car drives (see `run`)."""
     with learner_code(f"your car driving {scenario!r}"):
-        tel = run(car, scenario)
+        tel = run(car, scenario, on_step=on_step)
     print(metrics_table(tel, labels=[label]))
     if show:
         show_figure(dashboard(tel, labels=[label]))
@@ -54,9 +57,9 @@
     return car
 
 
-def gate(car: Car, show: bool) -> str:
+def gate(car: Car, show: bool, on_step: Callable[[tuple], None] | None = None) -> str:
     """Fail unless the car completes the gate scenario with mean |lat| under the limit; else say how it did."""
-    tel = drive(car, GATE_SCENARIO, f"your car on {GATE_SCENARIO}", show)
+    tel = drive(car, GATE_SCENARIO, f"your car on {GATE_SCENARIO}", show, on_step)
     mean_lat = metrics(tel)["mean_abs_lat"]
     seen = f"mean |lat| {mean_lat:.3f} m {vs_limit(mean_lat, GATE_MEAN_LAT, 'm')}"
     if not tel.completed or mean_lat >= GATE_MEAN_LAT:
```

Each unit grader takes the argument; the two gate exercises pass it on:

```diff
--- a/zero2fsd/grade/units/u0_1_1.py
+++ b/zero2fsd/grade/units/u0_1_1.py
@@ -26,7 +26,7 @@
     checked=f"mean_abs, max_abs and rms of seeded arrays (negatives included) against a python-loop computation, to {TOL}",
     hint='lecture 0.1.1, section "Summarising lane error"',
 )
-def _lane_error_stats(fn, show):
+def _lane_error_stats(fn, show, on_step):
     rng = np.random.default_rng(101)
     arrays = [rng.normal(0.0, sd, n) for sd, n in ((0.3, 50), (0.6, 200), (1.0, 7), (0.05, 1000), (0.8, 2))]
     arrays.append(np.array([-0.4]))  # one negative sample: |e|, not e
@@ -46,7 +46,7 @@
     checked="the count of steps with |error| strictly greater than the threshold, at the default 0.9 and at custom thresholds",
     hint='lecture 0.1.1, section "Counting steps off the lane"',
 )
-def _steps_off_lane(fn, show):
+def _steps_off_lane(fn, show, on_step):
     rng = np.random.default_rng(102)
     edge = np.array([0.9, -0.9, 0.9000001, -0.95, 0.2, 0.0])  # values exactly at the threshold are not off the lane
     cases = [(edge, None), (edge, 0.5)] + [(rng.normal(0.0, 0.7, n), thr) for n, thr in ((30, None), (200, 0.3), (1, None))]
```

```diff
--- a/zero2fsd/grade/units/u0_1_2.py
+++ b/zero2fsd/grade/units/u0_1_2.py
@@ -50,7 +50,7 @@
     checked=f"yellow and white masks on {N_FRAMES} noisy (sigma={NOISE_SIGMA:g}) frames against the renderer's true paint, IoU >= {IOU_MIN}",
     hint='lecture 0.1.2, section "Paint is a colour threshold"',
 )
-def _paint_masks(fn, show):
+def _paint_masks(fn, show, on_step):
     worst = 1.0  # an IoU cannot be higher
     for given, name, mask, truth, iou in paint_trials(fn):
         if iou < IOU_MIN:
```

```diff
--- a/zero2fsd/grade/units/u0_1_3.py
+++ b/zero2fsd/grade/units/u0_1_3.py
@@ -36,7 +36,7 @@
              f"at least {MIN_HIT_RATE:.0%} within {OFFSET_TOL} m and {HEADING_TOL} rad and valid"),
     hint='lecture 0.1.3, section "From masks to a lane estimate"',
 )
-def _estimate_lane(fn, show):
+def _estimate_lane(fn, show, on_step):
     hits, first_miss = 0, None
     for pose, given, est, hit in pose_trials(fn):
         hits += hit
@@ -57,8 +57,8 @@
     checked=f"your perception inside the provided car on `{GATE_SCENARIO}`: it must complete with mean |lat| < {GATE_MEAN_LAT} m",
     hint='lecture 0.1.3, section "Swap your perception into the car"',
 )
-def _perception_gate(fn, show):
+def _perception_gate(fn, show, on_step):
     with learner_code("Car(perception=your function)"):
         car = Car(perception=fn)
     require_own(car, "perception")  # Car(perception=None) would quietly drive on the provided box
-    return gate(car, show)
+    return gate(car, show, on_step)
```

```diff
--- a/zero2fsd/grade/units/u0_1_4.py
+++ b/zero2fsd/grade/units/u0_1_4.py
@@ -38,7 +38,7 @@
     checked=f"p_steer against -(k_off*offset + k_head*heading) clipped to +-{MAX_STEER}: signs, clipping, invalid estimate, 30 random cases",
     hint='lecture 0.1.4, section "The P controller"',
 )
-def _p_steer(fn, show):
+def _p_steer(fn, show, on_step):
     cases = _cases()
     for what, est, k_off, k_head, want in cases:
         given = f"est = LaneEstimate(offset_m={fmt(est.offset_m)}, heading_rad={fmt(est.heading_rad)}, valid={est.valid}), k_off={fmt(k_off)}, k_head={fmt(k_head)}"
@@ -55,9 +55,9 @@
     checked=f"my_car (your perception and your controller) on `{GATE_SCENARIO}`: it must complete with mean |lat| < {GATE_MEAN_LAT} m; then `curvy` is shown, not graded",
     hint='lecture 0.1.4, section "Closing the loop"',
 )
-def _my_car(car, show):
+def _my_car(car, show, on_step):
     require_car(car)
     require_own(car)
-    passed = gate(car, show)
+    passed = gate(car, show, on_step)
     curvy = drive(car, "curvy", "your cliffhanger on curvy", show)
     return f"{passed}; cliffhanger (not graded): {outcome(curvy)}"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/zero2fsd/test_grade.py tests/zero2fsd/test_margins.py -v`
Expected: PASS (107 tests: 77 in `test_grade.py`, 30 in `test_margins.py`).

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests -q`
Expected: all green; number of tests = 314 + 8 = 322.

- [ ] **Step 6: Commit**

```bash
git add zero2fsd/grade tests/zero2fsd/test_grade.py tests/zero2fsd/test_margins.py
git commit -m "feat(zero2fsd): check() streams the graded gate drive through on_step

The gate drive of 0.1.3.b and 0.1.4.b hands every telemetry row to the
caller, so the playground can show the graded run as it happens. All graders
share the (target, show, on_step) signature; non-driving ones never call it.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

## Part B: Playground v1

### Task 7: The browser shim, `playground/shim.py`

**Files:**
- Create: `playground/shim.py` (new directory `playground/`; Tasks 9 to 12 add the page files beside it)
- Create: `tests/playground/conftest.py` (`tests/playground/` gets no `__init__.py`: with `--import-mode=importlib` (`pytest.ini:2`) a second package named like `tests/zero2fsd/__init__.py` would only add module-shadowing trouble)
- Modify: `zero2fsd/car/car.py:40` (the controller's wrong-type message says "must return one number")
- Modify: `tests/zero2fsd/test_car.py:123` (pin that message)
- Test: `tests/playground/test_shim.py` (create, 52 tests)

Precondition for Tasks 7 and 8: the venv's editable install must point at the checkout you are in. Run `python -c "import zero2fsd; print(zero2fsd.__file__)"` from the repo root. If the path is not under this checkout, apply the "worktree imports" NOTE at the top of Part A (never `pip install -e`: it rewires the shared venv). Under pytest the `tests/zero2fsd/__init__.py` package shadows `zero2fsd`, so a module that is not yet imported (`zero2fsd.export`, `zero2fsd.preview`) is found only through that install.

**Interfaces:**
- Consumes (Tasks 1 to 6 of this plan, then existing code with file:line):
  - `zero2fsd/car/runner.py`: `COLUMNS` (14 names, Task 1; above `Telemetry`, which is `runner.py:20` today) and `run(car, scenario, max_steps=None, seed=0, noise_sigma=0.0, on_step=None) -> Telemetry` (`runner.py:43` today, gains `on_step` in Task 1: called once per step with the 14-tuple that was just appended).
  - `zero2fsd/car/__init__.py:2-5`: exports `Action, Car, LaneEstimate, Observation, Telemetry, provided, run`.
  - `zero2fsd/car/car.py:16`, `:34`, `:40`: the course's own `TypeError`s for a wrong box or wrong return type. They carry no learner frame in their traceback, which is how the shim tells them from the learner's own errors. Task 7 rewords `:40` only (see Step 3).
  - `zero2fsd/grade/__init__.py:18`: `check(exercise_id, fn_or_box, show=True, on_step=None) -> Result` (`on_step` is added in Task 6b and only the gate drives of 0.1.3.b and 0.1.4.b call it). `check` raises `ValueError` for an unknown id (`:20-21`) and prints the verdict with ANSI colour (`:24`); `show=False` stops the gate drive from drawing a figure (`_drive.py:37-38`). `Result` is `zero2fsd/grade/_report.py:23-25` (`passed: bool`, `details: list[str]`, `details[0]` is the summary line). Whatever Task 6 puts into `details` (the borderline note, margins) reaches the page as it is: the shim never formats a verdict.
  - `zero2fsd/grade/_report.py:28-34`: `Failure(headline, ...)`, what the preview builds raise for a wrong answer.
  - `zero2fsd/grade/units/u0_1_2.py:11`: `NOISE_SIGMA = 6.0`, the noise the 0.1.2 grader and lab use.
  - `zero2fsd/palette.py` (Task 4): `SCENE`, `#RRGGBB` strings for `sky, grass, road, yellow, white`, derived from the camera's `PALETTE`.
  - `zero2fsd/preview` (Task 5): `get(exercise_id) -> Preview(kind, build, road)` (raises `KeyError` for an unknown id) and `zero2fsd/preview/units.py` `OVERLAY = {"yellow": (255, 0, 255), "white": (0, 255, 255)}`:

    | kind | `build` | `road` |
    | --- | --- | --- |
    | `drive` | `build(obj) -> Car` | the gate road (`gentle`) |
    | `stats` | `build(fn, tel) -> dict[str, float]`, `fn` gets a copy of `tel.lat` | `curvy` (`STATS_ROAD`) |
    | `image` | `build(fn, frame) -> uint8[H, W, 3]`, the learner's masks painted in the `OVERLAY` colours | `""` (the caller picks the frame) |

  - `zero2fsd/score.py:11`: `metrics(tel)` returns a dict with `mean_abs_lat` and `max_abs_lat` (`:16-17`); `zero2fsd/dashboard.py:12`: `dashboard(*tels, labels=None) -> Figure` (matplotlib is imported inside it since Task 2).
  - `zero2fsd/sim/__init__.py:2-6`: `BicycleState, Camera, make_scenario`; `zero2fsd/sim/scenarios.py:76`: `make_scenario(name)` raises `ValueError` for an unknown name; `zero2fsd/sim/camera.py:79`: `Camera.render(road, state, noise_sigma=0.0, seed=0, return_labels=False)`.
  - `zero2fsd/car/provided.py:26-66`: `paint_masks, lane_point, estimate_lane, hold_speed, steer_p` and the constants the prelude re-exports.
  - The lab setup cells whose bindings the prelude must cover: `course/0.1.1-meet-your-car/lab.py:15`, `course/0.1.2-images-are-arrays/lab.py:16`, `course/0.1.3-your-first-perception-box/lab.py:16`, `course/0.1.4-closing-the-loop/lab.py:15` (each is `# %% tags=["setup"]`).
  - `scripts/_course.py:12` `REPO`, `:18` `load_yaml`, `:32` `syllabus_units(root, status=None)`, `:54` `parse_lab(text) -> list[Cell]` (`Cell` at `:48`: `kind`, `tags`, `source`).
- Produces (Tasks 9 to 13 rely on exactly these, all in `playground/shim.py`):
  - `PRELUDE: str` (35 names: `math, time, np, Action, Car, LaneEstimate, Observation, provided, run, FAR_ROWS, K_HEAD, K_OFF, MIN_PIXELS, NEAR_ROWS, estimate_lane, hold_speed, lane_point, paint_masks, steer_p, metrics, metrics_table, BicycleState, Camera, make_scenario, GRASS, PALETTE, ROAD, SKY, WHITE, YELLOW, DT, MAX_STEER, WHEELBASE, TARGET_SPEED, camera`) and `LEARNER_FILENAME = "<learner>"`.
  - `learner_object(code: str, name: str) -> Any`: `NameError("define `<name>` in the editor; the check looks for that exact name")` when the code does not bind `name` itself.
  - `format_error(exc) -> {"message": "<Class>: <text>", "line": int | None, "traceback": str}`; `line` is the innermost `<learner>` frame, or the `SyntaxError`'s own line, else `None`.
  - `row_dict(row) -> dict` (keys `COLUMNS`, values plain `float | int | bool`); `summary(tel) -> {"completed", "steps", "mean_abs_lat", "max_abs_lat", "progress_m", "off_road"}`.
  - `preview_run(obj, exercise_id, road, on_row) -> summary`; `preview_stats(obj, exercise_id) -> dict[str, float]`; `preview_image(obj, exercise_id, seed=0) -> {"width": 320, "height": 180, "frame": bytes, "overlay": bytes}` (two RGB buffers of `width * height * 3` bytes; the overlay is black except where a mask is set, painted in the `SCENE` yellow or white); `canned_telemetry(road: str) -> Telemetry` (cached, one drive of the provided `Car()` per road); `check_code(obj, exercise_id, on_row=None) -> {"passed": bool, "details": list[str]}`; `frame_rgb(road, k, x, y, yaw, speed, seed, noise_sigma=0.0) -> bytes` (320 x 180 x 3); `dashboard_png() -> bytes`; `probe_steps_per_s(n: int = 100) -> float`; module globals `LAST_RUN: Telemetry | None` and `IMAGE_ROAD = "straight"`.
  - Errors the worker relays through `format_error`: `ValueError("<id> is a <kind> exercise; this call is for <kind> exercises")` when a call does not match the exercise kind; `ValueError("unknown exercise '<id>'; known: ...")` (the registry's message, translated from its `KeyError`) for an id the registry lacks; `TypeError("<exercise id>: <what is wrong>")` for a wrong return type, for example `0.1.4.a: the controller box must return one number (a steering angle in radians) or an Action, got tuple`, `0.1.3.a: the perception box must return a LaneEstimate, got tuple`, `0.1.1.a: expected a number or a dict of numbers, got str`, `0.1.2.a: returned str, expected (yellow, white)`; the learner's own exceptions untouched; `RuntimeError("run the car first: ...")` from `dashboard_png` before any run.

> NOTE: `check_code` takes an optional `on_row` (default `None`). The worker passes it so a drive check streams its gate run to the canvas; with `on_row=None` the call is the plain `check_code(obj, exercise_id)`.
> NOTE: `PRELUDE` binds 35 names. It is derived from the four lab setup cells and a test fails when a lab binds a name the prelude lacks, so no list drifts by hand. `plt, dashboard, check, practice, ends` stay out (the page has buttons for them).
> NOTE: `check()` prints its verdict with ANSI colour codes (`grade/__init__.py:24`). Task 9's `setStdout` forwards them untouched; Task 11's `stdout` handler strips `\x1b[...m` (`ANSI_COLOUR`) before writing to `#stdout`.
> NOTE (Task 6b): `check_code` calls `check(exercise_id, obj, show=False, on_step=...)` with the signature Task 6b gives it, and `on_step` is `None` when `on_row` is `None`. Only the gate drives stream, so a stats or image check never calls `on_row`. Task 7 needs nothing else of 6b.
> NOTE (stats road): `preview_stats` does not summarise `run(Car(), "gentle")`: a `gentle` drive never leaves the lane, so a count of off-lane steps would read 0. Task 5 stores `STATS_ROAD = "curvy"` on the stats previews, so `canned_telemetry` takes the road and the shim runs `get(exercise_id).road`. Two stats exercises on one road share one drive.
> NOTE (image road): Task 5 gives image previews `road == ""` (the caller picks the frame), so the frame's road is the shim's own constant `IMAGE_ROAD = "straight"`, the road the 0.1.2 lab photographs. Records built in Task 8 carry `road == ""` for the image exercise; Task 10 and Task 11 must not read it.
> NOTE (wrong return type text): `car.py:40` said "an Action or a steering angle (float)". Task 12 asserts "must return one number". Rather than add a second return-type check in the shim (a copy of `car.py:38`'s rule), Task 7 rewords the message where it is raised, which also helps a Colab learner, and `_at_boundary` prefixes the exercise id. What the learner sees is the wrapper's message, with the exercise id, and not `Car.step`'s bare error: the wrapper is the one place that adds the exercise.
> NOTE (stats numbers): Task 5's `_as_stats` turns whatever the learner returns into floats with `float()`, so `"3"` would become `3.0` and `True` would become `1.0`. `_numbers` in the shim checks the type first. If `_as_stats` is changed to raise `Failure` for a non-number, delete `_numbers` and keep the tests.
> NOTE (read-only telemetry): the first draft froze the cached `lat` array to stop a stats function editing the shared run. Task 5's stats build hands the function `tel.lat.copy()`, so the freezing code is gone and `test_a_stats_function_that_edits_its_input_leaves_the_shared_drive_alone` keeps the guarantee.
> NOTE: `Failure` is a private name (`grade/_report.py:28`) imported across packages. It is the one exception type the registry uses for a wrong answer; making it public would be a rename in a file this task does not otherwise touch.

**Review focus:** (a) `format_error` returns the learner's line, not the shim's: the pin is `test_format_error_reports_the_learner_line_and_nothing_else` (a three-line function raising on line 2 gives `line == 2`; a nested helper gives the helper's line; an error raised outside learner code, and a stray `NameError`, give `None`; a `SyntaxError` gives its own line), and `test_an_error_in_the_mask_function_keeps_the_learners_own_line` for the image path. (b) A wrong return type fails in ONE place with the exercise id and the expected type in plain words: `_at_boundary` in `shim.py`. Pins: `test_a_wrong_return_type_is_a_plain_error_naming_the_exercise` (`preview_run`: a tuple, `None` and a string steer, a tuple lane estimate, a non-`Car`), `test_a_stats_function_returning_the_wrong_type_is_a_plain_error` (`preview_stats`: a string, an array, `None`, a dict with a string value, a bool) and `test_a_wrong_mask_answer_is_a_plain_error_naming_the_exercise` (`preview_image`). The learner's own `TypeError` is left alone (`test_the_learners_own_type_error_keeps_its_own_message_and_line`).

- [ ] **Step 1: Write the failing test**

Edit `tests/zero2fsd/test_car.py`. `test_a_controller_returning_the_wrong_type_raises_a_readable_type_error` (lines 120-124) pins only the type name; make it pin the wording the playground relies on. Line 123 becomes:

```python
@pytest.mark.parametrize("bad, message", [(None, "NoneType"), ("left", "str"), (True, "bool")])
def test_a_controller_returning_the_wrong_type_raises_a_readable_type_error(bad, message):
    car = Car(perception=lambda obs: LaneEstimate(0.0, 0.0, True), controller=lambda est, obs: bad)
    with pytest.raises(TypeError, match=f"must return one number.*got {message}"):
        car.step(Observation(np.zeros((1, 1, 3), np.uint8), 8.0, 0.0))
```

Create `tests/playground/conftest.py`. It loads the page file by path (it is not part of the `zero2fsd` package and the browser fetches it as its own file), and it makes the course scripts importable the way `tests/zero2fsd/conftest.py:8` does.

```python
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # before any matplotlib import: tests never open a window
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))  # the course scripts are importable modules

import pytest


@pytest.fixture(scope="session")
def shim():
    """playground/shim.py loaded by path: it is a page file, not part of the zero2fsd package."""
    spec = importlib.util.spec_from_file_location("playground_shim", ROOT / "playground" / "shim.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
```

Create `tests/playground/test_shim.py`:

```python
import ast
import dataclasses
import json

import numpy as np
import pytest
from _course import REPO, load_yaml, parse_lab, syllabus_units

from zero2fsd.car import Car, provided, run
from zero2fsd.car.runner import COLUMNS
from zero2fsd.grade import _drive as grade_drive
from zero2fsd.grade.units.u0_1_2 import NOISE_SIGMA
from zero2fsd.palette import SCENE
from zero2fsd.sim.camera import PALETTE

NOT_IN_BROWSER = {"plt", "dashboard", "check", "practice", "ends"}  # notebook-only names: the page has its own buttons

P_STEER = """\
def p_steer(est, k_off, k_head):
    if not est.valid:
        return 0.0
    return float(np.clip(-(k_off * est.offset_m + k_head * est.heading_rad), -MAX_STEER, MAX_STEER))
"""
MY_CAR = """\
def estimate_lane(obs):
    return provided.estimate_lane(obs)


def my_controller(est, obs):
    return steer_p(est, K_OFF, K_HEAD)


my_car = Car(perception=estimate_lane, controller=my_controller)
"""
LANE_ERROR_STATS = """\
def lane_error_stats(lat_err):
    return {"mean_abs": np.abs(lat_err).mean(), "max_abs": np.abs(lat_err).max(), "rms": np.sqrt(np.mean(lat_err**2))}
"""
COUNT_STEPS = "def steps_off_lane(lat_err):\n    return len(lat_err)\n"
PAINT_MASKS = "def paint_masks(frame):\n    return provided.paint_masks(frame)\n"
SWAPPED_MASKS = "def paint_masks(frame):\n    yellow, white = provided.paint_masks(frame)\n    return white, yellow\n"
NO_MASKS = "def paint_masks(frame):\n    return np.zeros(frame.shape[:2], bool), np.zeros(frame.shape[:2], bool)\n"


def prelude_names(shim):
    ns = {}
    exec(shim.PRELUDE, ns)
    return {name for name in ns if not name.startswith("__")}


def lab_cells():
    """(unit entry, unit.yaml, parsed lab cells) for every ready unit of the real syllabus."""
    for entry in syllabus_units(REPO, "ready"):
        folder = REPO / "course" / entry["folder"]
        yield entry, load_yaml(folder / "unit.yaml"), parse_lab((folder / "lab.py").read_text(encoding="utf-8"))


def bound_names(source):
    """Names a module-level code cell binds: imports, defs, classes and assignments."""
    names = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)}
    return names


def test_prelude_binds_every_name_the_lab_setup_cells_bind(shim):
    lab_names = {n for _, _, cells in lab_cells() for c in cells if "setup" in c.tags for n in bound_names(c.source)}
    assert lab_names - NOT_IN_BROWSER <= prelude_names(shim)
    assert NOT_IN_BROWSER.isdisjoint(prelude_names(shim))


def test_every_exercise_starter_runs_after_the_prelude_and_defines_its_name(shim):
    for _, meta, cells in lab_cells():
        for exercise in meta["exercises"]:
            starter = "\n\n".join(c.source for c in cells if c.kind == "code" and f"exercise:{exercise['id']}" in c.tags)
            shim.learner_object(starter, exercise["name"])  # raises NameError when the starter does not define the name


def test_learner_object_returns_what_the_learner_defined(shim):
    assert shim.learner_object("def f(x):\n    return x + 1\n", "f")(1) == 2
    assert shim.learner_object("def estimate_lane(obs):\n    return 7\n", "estimate_lane")(None) == 7  # a redefinition counts


def test_a_provided_box_or_a_missing_name_is_not_an_answer(shim):
    with pytest.raises(NameError, match="define `p_steer`"):
        shim.learner_object("x = 1", "p_steer")
    for name in sorted(prelude_names(shim)):  # the class: any name the prelude binds, left untouched
        with pytest.raises(NameError, match=f"define `{name}`"):
            shim.learner_object("pass", name)


def test_format_error_reports_the_learner_line_and_nothing_else(shim):
    fn = shim.learner_object("def p_steer(est, k_off, k_head):\n    raise ValueError('boom')\n    return 0.0\n", "p_steer")
    with pytest.raises(ValueError) as caught:
        fn(None, 0.35, 2.7)
    info = shim.format_error(caught.value)
    assert info["line"] == 2 and info["message"] == "ValueError: boom" and "<learner>" in info["traceback"]
    nested = shim.learner_object("def helper():\n    raise ValueError('deep')\n\n\ndef p_steer(est, k_off, k_head):\n    return helper()\n", "p_steer")
    with pytest.raises(ValueError) as inner:
        nested(None, 0.35, 2.7)
    assert shim.format_error(inner.value)["line"] == 2  # the innermost learner frame, not the call site on line 6
    with pytest.raises(NameError) as elsewhere:
        shim.learner_object("pass", "p_steer")
    assert shim.format_error(elsewhere.value)["line"] is None
    with pytest.raises(SyntaxError) as syntax:
        shim.learner_object("x = 1\ndef f(:\n", "f")
    assert shim.format_error(syntax.value)["line"] == 2


def test_row_dict_is_plain_json(shim):
    rows = []
    run(Car(), "straight", max_steps=3, on_step=rows.append)
    for row in rows:
        plain = shim.row_dict(row)
        assert tuple(plain) == COLUMNS
        assert {type(v) for v in plain.values()} <= {float, int, bool}
        json.dumps(plain)
    with pytest.raises(ValueError):
        shim.row_dict(rows[0][:-1])


def test_summary_of_a_complete_run_and_of_a_run_that_leaves_the_road(shim):
    tel = run(Car(), "straight", max_steps=40)
    kept = shim.summary(tel)
    assert set(kept) == {"completed", "steps", "mean_abs_lat", "max_abs_lat", "progress_m", "off_road"}
    assert kept["steps"] == 40 and kept["completed"] is False and kept["off_road"] is False
    assert kept["progress_m"] == pytest.approx(tel.progress_m[-1]) and kept["progress_m"] > tel.progress_m[0]
    assert kept["mean_abs_lat"] == pytest.approx(np.abs(tel.lat).mean()) and kept["max_abs_lat"] == pytest.approx(np.abs(tel.lat).max())
    left = shim.summary(run(Car(controller=lambda est, obs: 0.5), "straight"))
    assert left["off_road"] is True and left["completed"] is False and left["steps"] < 200
    json.dumps([kept, left])


@pytest.mark.parametrize("road", ["straight", "gentle"])
def test_preview_run_streams_one_row_per_step(shim, road):
    rows = []
    done = shim.preview_run(shim.learner_object(P_STEER, "p_steer"), "0.1.4.a", road, rows.append)
    assert len(rows) == done["steps"] == len(shim.LAST_RUN.t) and done["completed"] is True
    assert rows[0]["t"] == 0.0 and tuple(rows[0]) == COLUMNS


def test_a_learner_that_raises_mid_run_stops_the_run_and_the_next_run_works(shim):
    code = "def p_steer(est, k_off, k_head, _n=[0]):\n    _n[0] += 1\n    if _n[0] > 20:\n        raise ValueError('boom')\n    return 0.0\n"
    good, rows = shim.learner_object(P_STEER, "p_steer"), []
    shim.preview_run(good, "0.1.4.a", "straight", rows.append)
    assert shim.LAST_RUN is not None
    rows.clear()
    with pytest.raises(ValueError) as caught:
        shim.preview_run(shim.learner_object(code, "p_steer"), "0.1.4.a", "straight", rows.append)
    assert len(rows) == 20 and shim.format_error(caught.value)["line"] == 4 and shim.LAST_RUN is None
    assert shim.preview_run(good, "0.1.4.a", "straight", rows.append)["completed"]


@pytest.mark.parametrize("exercise, code, name, expected, got", [
    ("0.1.4.a", "def p_steer(est, k_off, k_head):\n    return (0.1, 0.2)\n", "p_steer", "must return one number", "got tuple"),
    ("0.1.4.a", "def p_steer(est, k_off, k_head):\n    return None\n", "p_steer", "must return one number", "got NoneType"),
    ("0.1.4.a", "def p_steer(est, k_off, k_head):\n    return 'left'\n", "p_steer", "must return one number", "got str"),
    ("0.1.3.a", "def estimate_lane(obs):\n    return (0.0, 0.0, True)\n", "estimate_lane", "must return a LaneEstimate", "got tuple"),
    ("0.1.4.b", "my_car = 5\n", "my_car", "must be a Car", "got int"),
])
def test_a_wrong_return_type_is_a_plain_error_naming_the_exercise(shim, exercise, code, name, expected, got):
    with pytest.raises(TypeError) as caught:
        shim.preview_run(shim.learner_object(code, name), exercise, "straight", lambda row: None)
    message = str(caught.value)
    assert message.startswith(f"{exercise}: ") and expected in message and got in message and "numpy" not in message.lower()


def test_the_learners_own_type_error_keeps_its_own_message_and_line(shim):
    fn = shim.learner_object("def p_steer(est, k_off, k_head):\n    return 'a' + 1\n", "p_steer")
    with pytest.raises(TypeError) as caught:
        shim.preview_run(fn, "0.1.4.a", "straight", lambda row: None)
    assert "0.1.4.a" not in str(caught.value) and shim.format_error(caught.value)["line"] == 2


CALLS = {  # the kind of exercise a call is for -> a call of it that never gets as far as the learner's code
    "drive": lambda shim, exercise: shim.preview_run(lambda: None, exercise, "straight", lambda row: None),
    "stats": lambda shim, exercise: shim.preview_stats(lambda lat: 0, exercise),
    "image": lambda shim, exercise: shim.preview_image(lambda frame: None, exercise),
}
EXERCISE_OF = {"drive": "0.1.4.a", "stats": "0.1.1.a", "image": "0.1.2.a"}


@pytest.mark.parametrize("call, kind", [(call, kind) for call in CALLS for kind in EXERCISE_OF if kind != call])
def test_a_call_for_one_kind_refuses_an_exercise_of_another(shim, call, kind):
    with pytest.raises(ValueError, match=f"{EXERCISE_OF[kind]} is a {kind} exercise; this call is for {call} exercises"):
        CALLS[call](shim, EXERCISE_OF[kind])


@pytest.mark.parametrize("call", list(CALLS))
def test_an_unknown_exercise_id_is_a_plain_error_listing_the_valid_ids(shim, call):
    with pytest.raises(ValueError, match=r"unknown exercise '9\.9\.9\.z'; known: .*0\.1\.4\.a"):
        CALLS[call](shim, "9.9.9.z")


def test_preview_stats_summarises_the_canned_drive_by_label(shim):
    lat = np.asarray(shim.canned_telemetry(shim.get("0.1.1.a").road).lat)
    stats = shim.preview_stats(shim.learner_object(LANE_ERROR_STATS, "lane_error_stats"), "0.1.1.a")
    assert stats == pytest.approx({"mean_abs": np.abs(lat).mean(), "max_abs": np.abs(lat).max(), "rms": np.sqrt(np.mean(lat**2))})
    counter = shim.learner_object("def steps_off_lane(lat_err, threshold=0.9):\n    return int((np.abs(lat_err) > threshold).sum())\n", "steps_off_lane")
    assert list(shim.preview_stats(counter, "0.1.1.b").values()) == [float((np.abs(lat) > 0.9).sum())]


def test_stats_run_on_the_road_the_registry_names_and_share_one_drive_per_road(shim, monkeypatch):
    shim.canned_telemetry.cache_clear()
    road, roads, real_run = shim.get("0.1.1.a").road, [], shim.run
    assert shim.get("0.1.1.b").road == road
    monkeypatch.setattr(shim, "run", lambda car, scenario: roads.append(scenario) or real_run(car, scenario))
    steps = shim.learner_object(COUNT_STEPS, "steps_off_lane")
    for exercise in ("0.1.1.a", "0.1.1.b"):
        assert list(shim.preview_stats(steps, exercise).values()) == [float(len(real_run(Car(), road).t))]
    assert roads == [road]  # two exercises, one road: one drive


def test_a_stats_exercise_on_another_road_gets_that_roads_drive(shim, monkeypatch):
    real_get, straight = shim.get, len(run(Car(), "straight").t)
    assert straight != len(run(Car(), real_get("0.1.1.b").road).t)  # the two roads give different drives
    monkeypatch.setattr(shim, "get", lambda exercise_id: dataclasses.replace(real_get(exercise_id), road="straight"))
    assert shim.preview_stats(shim.learner_object(COUNT_STEPS, "steps_off_lane"), "0.1.1.b") == {"steps_off_lane": float(straight)}


@pytest.mark.parametrize("exercise, code, name, what, got", [
    ("0.1.1.a", "def lane_error_stats(lat_err):\n    return 'oops'\n", "lane_error_stats", "a number or a dict of numbers", "got str"),
    ("0.1.1.b", "def steps_off_lane(lat_err, threshold=0.9):\n    return '3'\n", "steps_off_lane", "a number or a dict of numbers", "got str"),
    ("0.1.1.a", "def lane_error_stats(lat_err):\n    return lat_err[:3]\n", "lane_error_stats", "a number or a dict of numbers", "got ndarray"),
    ("0.1.1.b", "def steps_off_lane(lat_err, threshold=0.9):\n    pass\n", "steps_off_lane", "a number or a dict of numbers", "got NoneType"),
    ("0.1.1.a", "def lane_error_stats(lat_err):\n    return {'mean': 'high'}\n", "lane_error_stats", "a number for 'mean'", "got str"),
    ("0.1.1.b", "def steps_off_lane(lat_err, threshold=0.9):\n    return True\n", "steps_off_lane", "a number or a dict of numbers", "got bool"),
])
def test_a_stats_function_returning_the_wrong_type_is_a_plain_error(shim, exercise, code, name, what, got):
    with pytest.raises(TypeError) as caught:
        shim.preview_stats(shim.learner_object(code, name), exercise)
    assert str(caught.value) == f"{exercise}: expected {what}, {got}"


def test_a_stats_function_that_edits_its_input_leaves_the_shared_drive_alone(shim):
    road = shim.get("0.1.1.b").road
    before = shim.canned_telemetry(road).lat.copy()
    wipe = shim.learner_object("def steps_off_lane(lat_err):\n    lat_err[:] = 99.0\n    return 0\n", "steps_off_lane")
    assert shim.preview_stats(wipe, "0.1.1.b") == {"steps_off_lane": 0.0}
    assert np.array_equal(shim.canned_telemetry(road).lat, before) and shim.canned_telemetry(road) is shim.canned_telemetry(road)


def scene_rgb(name):
    return tuple(int(SCENE[name][i:i + 2], 16) for i in (1, 3, 5))


def picture(shim, code, seed=0):
    """(frame, overlay) arrays of the image preview of 0.1.2.a for the learner's `paint_masks` in `code`."""
    shown = shim.preview_image(shim.learner_object(code, "paint_masks"), "0.1.2.a", seed)
    shape = (shown["height"], shown["width"], 3)
    return tuple(np.frombuffer(shown[key], np.uint8).reshape(shape) for key in ("frame", "overlay"))


@pytest.mark.parametrize("seed", [0, 3])
def test_the_image_frame_is_what_the_car_sees_at_the_start_of_the_straight_road(shim, seed):
    seen = []

    def perception(obs):
        seen.append(obs.frame.copy())
        return provided.estimate_lane(obs)

    run(Car(perception=perception), "straight", max_steps=1, seed=seed, noise_sigma=NOISE_SIGMA)
    frame, overlay = picture(shim, PAINT_MASKS, seed)
    assert frame.shape == overlay.shape == (180, 320, 3) and np.array_equal(frame, seen[0])


def test_the_same_seed_gives_the_same_picture_and_another_seed_other_noise(shim):
    first, again, other = (picture(shim, PAINT_MASKS, seed)[0] for seed in (0, 0, 1))
    assert np.array_equal(first, again) and not np.array_equal(first, other)


@pytest.mark.parametrize("code, order", [(PAINT_MASKS, 1), (SWAPPED_MASKS, -1)], ids=["as returned", "yellow and white swapped"])
def test_the_overlay_paints_each_mask_in_the_scene_colour_of_its_paint(shim, code, order):
    frame, overlay = picture(shim, code)
    first, second = provided.paint_masks(frame.copy())[::order]  # the masks the learner's function returned, in order
    assert first.any() and second.any()
    expected = np.zeros_like(frame)
    expected[first], expected[second] = scene_rgb("yellow"), scene_rgb("white")
    assert np.array_equal(overlay, expected)


def test_a_function_that_marks_nothing_gives_a_black_overlay(shim):
    assert not picture(shim, NO_MASKS)[1].any()


def test_no_camera_colour_is_near_the_markers_the_registry_paints_masks_in(shim):
    nearest = min(np.abs(np.array(marker) - colour).max() for marker in shim.OVERLAY.values() for colour in PALETTE)
    assert nearest > 5 * NOISE_SIGMA  # so reading a mask back from the painted frame cannot pick up a scenery pixel


@pytest.mark.parametrize("code, expected", [
    ("def paint_masks(frame):\n    return 'oops'\n", "returned str, expected (yellow, white)"),
    ("def paint_masks(frame):\n    return provided.paint_masks(frame)[0]\n", "returned ndarray, expected (yellow, white)"),
    ("def paint_masks(frame):\n    return frame[:, :, 0], frame[:, :, 1]\n", "the yellow mask must be a bool array of shape (180, 320)"),
])
def test_a_wrong_mask_answer_is_a_plain_error_naming_the_exercise(shim, code, expected):
    with pytest.raises(TypeError) as caught:
        shim.preview_image(shim.learner_object(code, "paint_masks"), "0.1.2.a")
    assert str(caught.value) == f"0.1.2.a: {expected}"


def test_an_error_in_the_mask_function_keeps_the_learners_own_line(shim):
    fn = shim.learner_object("def paint_masks(frame):\n    return frame.nothing\n", "paint_masks")
    with pytest.raises(AttributeError) as caught:
        shim.preview_image(fn, "0.1.2.a")
    assert shim.format_error(caught.value)["line"] == 2


def test_check_code_gives_the_verdict_and_streams_rows_only_for_drive_checks(shim):
    ok = shim.check_code(shim.learner_object(P_STEER, "p_steer"), "0.1.4.a")
    bad = shim.check_code(shim.learner_object("def p_steer(est, k_off, k_head):\n    return 0.0\n", "p_steer"), "0.1.4.a")
    assert ok["passed"] is True and bad["passed"] is False
    assert all(isinstance(line, str) for line in ok["details"] + bad["details"]) and bad["details"]
    assert shim.check_code(shim.learner_object(LANE_ERROR_STATS, "lane_error_stats"), "0.1.1.a")["passed"] is True
    drive_rows, stats_rows = [], []
    assert shim.check_code(shim.learner_object(MY_CAR, "my_car"), "0.1.4.b", on_row=drive_rows.append)["passed"] is True
    shim.check_code(shim.learner_object(LANE_ERROR_STATS, "lane_error_stats"), "0.1.1.a", on_row=stats_rows.append)
    assert len(drive_rows) > 500 and tuple(drive_rows[0]) == COLUMNS and stats_rows == []
    refused_rows = []
    refused = shim.check_code(Car(), "0.1.4.b", on_row=refused_rows.append)  # the provided boxes: refused before any drive
    assert refused["passed"] is False and "still uses the provided" in refused["details"][0] and refused_rows == []


def test_check_code_draws_no_figure(shim, monkeypatch):
    drawn = []
    monkeypatch.setattr(grade_drive, "show_figure", drawn.append)  # what a check does with `show=True`: matplotlib has no window here
    assert shim.check_code(shim.learner_object(MY_CAR, "my_car"), "0.1.4.b")["passed"] is True
    assert drawn == []


@pytest.mark.parametrize("seed, noise_sigma", [(0, 0.0), (7, 6.0)])
def test_frame_rgb_is_the_frame_the_car_saw(shim, seed, noise_sigma):
    seen = []

    def perception(obs):
        seen.append(obs.frame.copy())
        return provided.estimate_lane(obs)

    tel = run(Car(perception=perception), "gentle", max_steps=6, seed=seed, noise_sigma=noise_sigma)
    frames = [shim.frame_rgb("gentle", k, tel.x[k], tel.y[k], tel.yaw[k], tel.speed[k], seed, noise_sigma) for k in (0, 5)]
    assert [len(f) for f in frames] == [320 * 180 * 3] * 2 and frames == [seen[0].tobytes(), seen[5].tobytes()]
    assert frames[0] != frames[1]


def test_dashboard_png_shows_the_last_completed_run(shim):
    shim.LAST_RUN = None
    with pytest.raises(RuntimeError, match="run the car first"):
        shim.dashboard_png()
    shim.preview_run(shim.learner_object(P_STEER, "p_steer"), "0.1.4.a", "straight", lambda row: None)
    assert shim.dashboard_png().startswith(b"\x89PNG\r\n\x1a\n")


def test_probe_steps_per_s_is_a_positive_rate(shim):
    assert shim.probe_steps_per_s(30) > 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/zero2fsd/test_car.py tests/playground/test_shim.py -v`
Expected: FAIL, `3 failed, 28 passed, 52 errors`. The 3 failures are `test_car.py::test_a_controller_returning_the_wrong_type_raises_a_readable_type_error[...]`, each `AssertionError: Regex pattern did not match.` (pattern `must return one number.*got NoneType`, `...got str`, `...got bool`, against "the controller box must return an Action or a steering angle (float), got ..."). The 52 errors are every test in `test_shim.py`, each `FileNotFoundError: [Errno 2] No such file or directory: '<repo>/playground/shim.py'` raised at `tests/playground/conftest.py:18` (`spec.loader.exec_module`) while setting up the `shim` fixture. No shim test body runs.

- [ ] **Step 3: Write the minimal implementation**

Edit `zero2fsd/car/car.py`, the last line of `Car.step` (line 40). Only the message changes:

```python
        out = self._slots["controller"][0](est, obs)
        if isinstance(out, Action):
            return est, out
        if isinstance(out, Real) and not isinstance(out, bool):  # a bare number is a steering angle
            return est, Action(float(out), provided.hold_speed(obs.speed))
        raise TypeError(f"the controller box must return one number (a steering angle in radians) or an Action, got {type(out).__name__}")
```

Create `playground/shim.py`:

```python
"""The Python side of the playground: the worker calls these functions; nothing here touches the page.

Importable natively (the tests import it by path); in the browser it is fetched as its own file and
imported after bundle.zip has been unpacked.
"""
from __future__ import annotations

import functools
import io
import time
import traceback
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from numbers import Real
from typing import Any

import numpy as np

from zero2fsd.car import Car, Telemetry, run
from zero2fsd.car.runner import COLUMNS
from zero2fsd.dashboard import dashboard
from zero2fsd.grade import check
from zero2fsd.grade._report import Failure
from zero2fsd.grade.units.u0_1_2 import NOISE_SIGMA
from zero2fsd.palette import SCENE
from zero2fsd.preview import get
from zero2fsd.preview.units import OVERLAY
from zero2fsd.score import metrics
from zero2fsd.sim import BicycleState, Camera, make_scenario

# What every exercise cell sees: the names the labs' setup cells bind, minus the notebook-only ones
# (plt, dashboard, check, practice, ends), plus the provided boxes a learner may build on.
PRELUDE = """\
import math
import time

import numpy as np

from zero2fsd.car import Action, Car, LaneEstimate, Observation, provided, run
from zero2fsd.car.provided import (FAR_ROWS, K_HEAD, K_OFF, MIN_PIXELS, NEAR_ROWS, estimate_lane, hold_speed,
                                   lane_point, paint_masks, steer_p)
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
    provided = dict(ns)
    exec(compile(code, LEARNER_FILENAME, "exec"), ns)
    if name not in ns or (name in provided and ns[name] is provided[name]):  # a provided box is not an answer
        raise NameError(f"define `{name}` in the editor; the check looks for that exact name")
    return ns[name]


def _learner_line(exc: BaseException) -> int | None:
    """The line in the learner's code that raised `exc` (innermost learner frame, or a SyntaxError's own line)."""
    if isinstance(exc, SyntaxError) and exc.filename == LEARNER_FILENAME:
        return exc.lineno
    lines = [lineno for frame, lineno in traceback.walk_tb(exc.__traceback__) if frame.f_code.co_filename == LEARNER_FILENAME]
    return lines[-1] if lines else None


def format_error(exc: BaseException) -> dict[str, Any]:
    return {"message": f"{type(exc).__name__}: {exc}", "line": _learner_line(exc),
            "traceback": "".join(traceback.format_exception(exc))}


@contextmanager
def _at_boundary(exercise_id: str):
    """The one place a wrong answer type is reported. The registry's checks (a `Failure`: my_car is not a Car, the
    masks are not masks) and the course's own (a `TypeError` from Car.step) come from code that is not the
    learner's: say which exercise. The learner's own exceptions have a `<learner>` frame and pass through."""
    try:
        yield
    except Failure as exc:
        raise TypeError(f"{exercise_id}: {exc.headline}") from exc
    except TypeError as exc:
        if _learner_line(exc) is not None:
            raise
        raise TypeError(f"{exercise_id}: {exc}") from exc


def _preview(exercise_id: str, kind: str):
    try:
        preview = get(exercise_id)
    except KeyError as exc:  # the worker shows the error text; a KeyError would print its message in quotes
        raise ValueError(exc.args[0]) from exc
    if preview.kind != kind:
        raise ValueError(f"{exercise_id} is a {preview.kind} exercise; this call is for {kind} exercises")
    return preview


def row_dict(row: tuple) -> dict[str, Any]:
    return {name: value.item() if isinstance(value, np.generic) else value for name, value in zip(COLUMNS, row, strict=True)}


def summary(tel: Telemetry) -> dict[str, Any]:
    m = metrics(tel)
    return {"completed": tel.completed, "steps": len(tel.t), "mean_abs_lat": m["mean_abs_lat"],
            "max_abs_lat": m["max_abs_lat"], "progress_m": float(tel.progress_m[-1]), "off_road": bool(tel.off_road[-1])}


def preview_run(obj: Any, exercise_id: str, road: str, on_row: Callable[[dict], None]) -> dict[str, Any]:
    """Drive the learner's car on `road`, handing each step to `on_row`; return the run's summary."""
    global LAST_RUN
    preview = _preview(exercise_id, "drive")
    LAST_RUN = None  # a run that raises must not leave the previous run behind for the dashboard
    with _at_boundary(exercise_id):
        tel = run(preview.build(obj), road, on_step=lambda row: on_row(row_dict(row)))
    LAST_RUN = tel
    return summary(tel)


@functools.cache
def canned_telemetry(road: str) -> Telemetry:
    """The provided car's run on `road`, the data a stats exercise summarises: one run per road, shared."""
    return run(Car(), road)


def _numbers(fn: Callable) -> Callable:
    """`fn` with its answer checked to be a number or a dict of numbers: the registry would turn "3" into 3.0."""

    def answer(*args):
        got = fn(*args)
        for label, value in (got.items() if isinstance(got, Mapping) else [(None, got)]):
            if not isinstance(value, Real) or isinstance(value, bool):
                what = "a number or a dict of numbers" if label is None else f"a number for {label!r}"
                raise TypeError(f"expected {what}, got {type(value).__name__}")
        return got

    return answer


def preview_stats(obj: Any, exercise_id: str) -> dict[str, float]:
    """The numbers the learner's stats function produces from the canned drive on the exercise's road, by label."""
    preview = _preview(exercise_id, "stats")
    with _at_boundary(exercise_id):
        return preview.build(_numbers(obj), canned_telemetry(preview.road))


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
    """The verdict Colab gives, plus the rows of any run the check makes when `on_row` is given."""
    on_step = None if on_row is None else lambda row: on_row(row_dict(row))
    result = check(exercise_id, obj, show=False, on_step=on_step)
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
```

Why it is shaped this way (read before changing it):
- `PRELUDE` is the union of the lab setup cells' bindings minus the notebook-only names. `test_prelude_binds_every_name_the_lab_setup_cells_bind` parses those cells, so a lab that starts using a new name breaks this test, not a learner's page.
- `learner_object` compares each prelude name by identity after the learner's code ran. `estimate_lane` is in the prelude (exercise 0.1.4.b builds on it), so presence in the namespace alone would make an untouched provided box pass as the learner's answer.
- `_at_boundary` is the only place a wrong answer is re-labelled with the exercise id. It catches the registry's `Failure` (not a `Car`, masks of the wrong shape) and the course's own `TypeError`s from `car.py` (no learner frame in the traceback). A `TypeError` with a `<learner>` frame is the learner's bug and passes through with its own line.
- `_preview` is the one lookup of the registry. It turns the registry's `KeyError` into a `ValueError` with the same text, because the worker shows `Class: message` and a `KeyError` would print the message inside quotes (`test_an_unknown_exercise_id_is_a_plain_error_listing_the_valid_ids`).
- `preview_stats` asks the registry for the road and runs `Car()` on it once per road (`functools.cache` on the road name). `_numbers` rejects a string, an array, `None` or a bool before the registry's `float()` can hide them.
- `preview_image` draws the car's first view of `IMAGE_ROAD` with the grader's `NOISE_SIGMA` and the caller's seed. The registry paints the learner's masks in `OVERLAY` colours no camera pixel comes near (`test_no_camera_colour_is_near_the_markers_the_registry_paints_masks_in`), so the shim reads them back exactly and repaints them in the scene's own yellow and white on black.
- `preview_run` clears `LAST_RUN` before it runs, so a run that raises leaves no stale dashboard behind.
- `frame_rgb` rebuilds a frame from the recorded pose and `seed + k`, the same call the runner makes, so the page never stores 180 x 320 x 3 bytes per step.
- `check_code` returns `check(...).details` as it is and passes `show=False` so no figure is drawn (`test_check_code_draws_no_figure`).

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/zero2fsd/test_car.py tests/playground/test_shim.py -v`
Expected: PASS (83 tests: 31 in `test_car.py`, 52 in `test_shim.py`).

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests -q`
Expected: all green; number of tests = 322 + 52 = 374 (322 is the total after Task 6b; `test_car.py` keeps its test count).

- [ ] **Step 6: Commit**

```bash
git add playground/shim.py tests/playground/conftest.py tests/playground/test_shim.py zero2fsd/car/car.py tests/zero2fsd/test_car.py
git commit -m "feat(playground): shim.py, the Python side of the in-browser runner

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: `scripts/build_playground.py` and its wiring into the course build

**Files:**
- Create: `scripts/build_playground.py`
- Modify: `scripts/_course.py:22-27` (add `colab_url` between `unit_slug` and `heading_slug`)
- Modify: `scripts/build_course_bundle.py:1-7` (docstring), `:18` (imports), `:119` (use `colab_url`), `:168-179` (`main` writes the playground)
- Modify: `course/0.1.1-meet-your-car/unit.yaml:4`, `course/0.1.2-images-are-arrays/unit.yaml:4`, `course/0.1.3-your-first-perception-box/unit.yaml:4`, `course/0.1.4-closing-the-loop/unit.yaml:4` (add `car_box:` after the prose `box:` line)
- Modify: `tests/zero2fsd/fixtures/mini_repo/course/0.1.1-mini/unit.yaml:7` (add `car_box:` after `box: none`)
- Modify: `tests/zero2fsd/conftest.py:11-12` (import), `:40-47` (`built_repo` also carries the real `playground/` and `zero2fsd/`)
- Modify: `tests/zero2fsd/test_bundle.py:80-84` (`test_main_exit_codes`) and add one test below it
- Modify: `tests/playground/test_shim.py:77-83` (delete one test that `test_build.py` supersedes)
- Test: `tests/playground/test_build.py` (create, 38 tests)

**Interfaces:**
- Consumes:
  - `scripts/_course.py:12` `REPO`, `:13` `REPO_SLUG`, `:18` `load_yaml`, `:22-24` `unit_slug(unit_id)` (dots to hyphens), `:32-36` `syllabus_units(root, status=None)`, `:48-51` `Cell(kind, tags, source)`, `:54` `parse_lab(text)`.
  - `scripts/build_course_bundle.py:126` `build_bundle(root) -> (bundle, copies)`, `:156` `write_bundle(root) -> Path`, `:168-179` `main(argv)`; `:110` and `:136` show the bundle keys `units` by `unit_slug(id)`.
  - `zero2fsd.preview.get(exercise_id) -> Preview(kind, build, road)` (Task 5), `zero2fsd.export.roads_dict()` (Task 3; coordinates rounded by `COORD_DECIMALS = 3`), `zero2fsd.palette.palette_dict()` (Task 4), `zero2fsd.grade._report.EXERCISES` (`_report.py:37`, tests only).
  - Course data: `unit.yaml` `exercises:` entries `{id, name}` with an optional `task` (for example `course/0.1.4-closing-the-loop/unit.yaml:15-17`); lab cells `# %% tags=["exercise:0.1.4.a"]` (`course/0.1.4-closing-the-loop/lab.py:168`; two cells for `exercise:0.1.4.b` at `:215` and `:233`).
  - `.gitignore:39` `!course/dist/`: `course/dist/playground/` is tracked without `git add -f`.
  - The site keys all progress by the unit slug: `/Users/vivek/Antigravity/vivekn/vivekn/src/lib/learnfsd-progress.ts:32-35,38,42-45` (`p.units[slug]`, `unit.slug`).
- Produces:
  - `scripts/_course.py`: `colab_url(folder: str) -> str`.
  - `scripts/build_playground.py`: `exercise_records(repo_root) -> list[dict]`, `validate_lecture_links(repo_root, records) -> None` (raises `ValueError` listing every bad link), `bundle_zip(package: Path) -> bytes`, `git_revision(repo_root) -> str`, `write_playground(repo_root, out_dir) -> None` (replaces `out_dir`), constants `PLAYGROUND_LINK`, `WRONG_FORM_LINK`, `ZIP_EPOCH`.
  - Record keys in `exercises.json` (list, syllabus order, only ready units): `id, unit_id, unit_key, week, unit_title, label, name, task, car_box, kind, road, presets, colab_url`. `road` is the preview's road, `""` for the image exercise 0.1.2.a (Task 5). `unit_key` is the slug (`"0-1-4"`), `task` is `""` unless `unit.yaml` gives one, `car_box` is a non-empty string from `unit.yaml`, `presets` is `{"starter": <tagged cells joined by a blank line>}`.
  - `course/dist/playground/`: every non-hidden file of `playground/`, `bundle.zip` (every `zero2fsd/**/*.py`, under `zero2fsd/`, stored, fixed date, sorted; no solutions, no tests, no caches), `exercises.json`, `roads.json` (`roads_dict()` unchanged), `palette.json`, and `manifest.json` (`{"generated_from": "<short hash or unknown>", "files": [{"path", "bytes", "sha256"}, ...]}`, sorted, not listing itself). JSON files use `sort_keys=True` and a trailing newline.
  - `exercises.json` carries `colab_url` on every record: the link from the playground to the lab (the page's "Open in Colab" button). The reverse link, from each lab exercise cell to its playground page, is deviation P15 and is not built here.
  - Lecture links the build accepts: `playground/index.html?exercise=<id>[&preset=<key>]`, relative with no leading slash.
  - `tests/zero2fsd/conftest.py::built_repo` now also holds the real `playground/` and `zero2fsd/`.

> NOTE: `unit_key` is the unit slug in the dash form, `"0-1-4"`, not the dotted id. `scripts/_course.py:22-24` (`unit_slug`) is the one place that derives it; `build_playground.py` imports it, so `course.json` (`build_course_bundle.py:110,136` keys `units` by that slug) and `exercises.json` cannot disagree. The site reads progress by the same string (`learnfsd-progress.ts:32-35,38,42-45`).
> NOTE: there is no shared `week_number(unit_id)` helper in `_course.py`. `course.json` carries no week, so it would have one consumer. The week is the 1-based position among ALL listed units, ready or not (`enumerate(syllabus_units(root), start=1)` in `exercise_records`); `test_the_week_counts_every_listed_unit_not_only_the_ready_ones` pins it. The site validator (site plan, Task 2) needs no shared helper either.
> NOTE: `colab_url` must equal the URL `build_course_bundle` already produces, but that is an inline f-string at `build_course_bundle.py:119`, not a function. Task 8 extracts it to `_course.colab_url` and calls it from both places; `course.json` stays byte-identical (checked in Step 4).
> NOTE: the lecture-link form is `playground/index.html?exercise=`, not `playground/?exercise=` (the site serves `public/` by exact path; the bare directory URL redirects and then 404s). A lecture link in the old form would match nothing and pass silently, so `WRONG_FORM_LINK` makes any other `playground/...exercise=` URL a build error too (one regex, three lines, one parametrised test).
> NOTE (two link validators): the site plan's Task 6b validator is the stricter of the two lecture-link checks, and the two regexes are deliberately not unified. Its pattern makes the `?exercise=` group optional, so a malformed `playground/index.html?...` link is reported instead of skipped, accepts `&amp;` before `preset=` (rendered HTML escapes the ampersand), and stops at `'`, `<`, `>` and `#` so a raw `<a href='...'>` is read correctly. Both accept the canonical `playground/index.html?exercise=<id>[&preset=<name>]`.
> NOTE: two record fields need explicit rules: `task` (optional `exercises[].task` in `unit.yaml`, default `""`) and `car_box` (required `unit.yaml` key, no default; a unit with exercises and no `car_box` fails the build naming the unit). The site plan's Task 2 validator requires `car_box` (non-empty string) and does not check `task` (a string, possibly empty).
> NOTE: spec section 6 (`docs/superpowers/specs/2026-10-11-playground-design.md:140`, branch `feat/playground-design`) says an unknown exercise id opens the first Week 1 exercise and says why. Task 11 follows the spec (first exercise, state `ready`, `#status` "Unknown exercise <id>; showing <label>") and Task 12 pins it. Task 8 only guarantees `records[0]` is the first exercise of the first ready unit (`test_one_record_per_graded_exercise_in_syllabus_order`).
> NOTE: Task 8 does not commit `course/dist/playground/` (it would hold no page files yet). The task that adds the last page file rebuilds with `python scripts/build_course_bundle.py` and commits the directory; Step 4 below deletes the smoke-run output.
> NOTE: `build_course_bundle.main` now exits 1 on a `ValueError` from the playground build as well as on a `BundleError`. The only `ValueError`s on that path are the build's own messages (missing `car_box`, untagged exercise, unknown preview id, bad lecture link).

- [ ] **Step 1: Write the failing tests**

Create `tests/playground/test_build.py`. It builds the real repo once per module and uses a copy of the one-unit fixture repo (`tests/zero2fsd/fixtures/mini_repo`) for every failure case.

```python
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

import pytest

import build_playground as bp
from _course import REPO, syllabus_units
from build_course_bundle import build_bundle
from zero2fsd.export import roads_dict
from zero2fsd.grade._report import EXERCISES
from zero2fsd.palette import palette_dict
from zero2fsd.preview import get

MINI = REPO / "tests" / "zero2fsd" / "fixtures" / "mini_repo"
DOTTED_ID = re.compile(r"\d+\.\d+\.\d+")
GENERATED = {"bundle.zip", "exercises.json", "roads.json", "palette.json", "manifest.json"}
RECORD_KEYS = {"id", "unit_id", "unit_key", "week", "unit_title", "label", "name", "task", "car_box", "kind", "road", "presets", "colab_url"}
UNIT_YAML = "course/0.1.1-mini/unit.yaml"


@pytest.fixture(scope="module")
def records():
    return bp.exercise_records(REPO)


@pytest.fixture(scope="module")
def dist(tmp_path_factory):
    out = tmp_path_factory.mktemp("playground") / "dist"
    bp.write_playground(REPO, out)
    return out


@pytest.fixture
def mini_repo(tmp_path):
    """A writable copy of the one-unit fixture repo the course-script tests use."""
    root = tmp_path / "repo"
    shutil.copytree(MINI, root)
    return root


def rewrite(root, rel, old, new):
    path = root / rel
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{old!r} not in {rel}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def link_in_lecture(root, folder, link):
    path = root / "course" / folder / "lecture.md"
    path.write_text(path.read_text(encoding="utf-8") + f"\n\n{link}\n", encoding="utf-8")


def starter_of(records, exercise_id):
    return next(r for r in records if r["id"] == exercise_id)["presets"]["starter"]


# --- exercise_records ---------------------------------------------------------------------------------------


def test_one_record_per_graded_exercise_in_syllabus_order(records):
    ids = [r["id"] for r in records]
    assert sorted(ids) == sorted(EXERCISES) and len(set(ids)) == len(ids)
    assert ids[0] == "0.1.1.a"  # the first exercise of the first unit: what the page opens for an id it does not know
    units = list(dict.fromkeys(r["unit_id"] for r in records))
    assert units == [u["id"] for u in syllabus_units(REPO, "ready")]


def test_week_and_label_follow_the_unit_position(records):
    labels = {r["id"]: r["label"] for r in records}
    assert labels["0.1.4.b"] == "Week 4 · Closing the loop · Exercise B"
    assert labels["0.1.1.a"] == "Week 1 · Meet your car · Exercise A"
    assert [r["week"] for r in records] == [1, 1, 2, 3, 3, 4, 4]


def test_labels_never_show_a_dotted_id(records):
    assert not [r["id"] for r in records if DOTTED_ID.search(r["label"]) or DOTTED_ID.search(r["unit_title"])]


def test_unit_key_and_colab_url_are_the_ones_the_course_bundle_publishes(records):
    units = build_bundle(REPO)[0]["units"]
    for r in records:
        unit = units[r["unit_key"]]  # the site keys progress by this slug, so it must be a unit of course.json
        assert (unit["id"], unit["title"], unit["colab_url"]) == (r["unit_id"], r["unit_title"], r["colab_url"])
    assert {r["unit_id"]: r["unit_key"] for r in records} == {"0.1.1": "0-1-1", "0.1.2": "0-1-2", "0.1.3": "0-1-3", "0.1.4": "0-1-4"}


def test_every_record_has_exactly_the_keys_the_page_reads(records):
    assert all(set(r) == RECORD_KEYS for r in records)


def test_the_starter_is_every_cell_tagged_for_the_exercise_joined_by_a_blank_line(records):
    two_cells = starter_of(records, "0.1.4.b")
    assert "raise NotImplementedError\n\nK_OFF, K_HEAD = 0.35, 2.7" in two_cells
    assert "def my_controller(est, obs):" in two_cells and "my_car = None" in two_cells
    assert "check(" not in two_cells and "inside_the_bend" not in two_cells  # untagged cells stay out
    assert "my_car" not in starter_of(records, "0.1.4.a")  # another exercise's cells stay out


def test_one_cell_tagged_for_two_exercises_is_the_starter_of_both(records):
    assert starter_of(records, "0.1.3.a") == starter_of(records, "0.1.3.b")
    assert "def estimate_lane(obs):" in starter_of(records, "0.1.3.a")


def test_every_starter_runs_after_the_prelude_and_defines_the_exercise_name(records, shim):
    for r in records:
        shim.learner_object(r["presets"]["starter"], r["name"])  # raises NameError when the name is not the learner's own


def test_kind_and_road_come_from_the_preview_registry_and_every_named_road_exists(records):
    assert {(r["id"], r["kind"], r["road"]) for r in records} == {(r["id"], get(r["id"]).kind, get(r["id"]).road) for r in records}
    assert {"drive", "stats", "image"} <= {r["kind"] for r in records}
    assert all(r["road"] in roads_dict() for r in records if r["road"])
    assert {r["kind"] for r in records if not r["road"]} == {"image"}  # a picture exercise names no road: the shim picks its frame


def test_car_box_is_the_units_own_key_in_unit_yaml(records):
    assert {r["unit_id"]: r["car_box"] for r in records} == {
        "0.1.1": "instruments", "0.1.2": "perception", "0.1.3": "perception", "0.1.4": "steer"}


def test_a_unit_with_exercises_and_no_car_box_fails_naming_it_and_one_with_a_car_box_emits_it(mini_repo):
    assert {r["car_box"] for r in bp.exercise_records(mini_repo)} == {"instruments"}
    rewrite(mini_repo, UNIT_YAML, "car_box: instruments", "car_box: steer")
    assert {r["car_box"] for r in bp.exercise_records(mini_repo)} == {"steer"}
    rewrite(mini_repo, UNIT_YAML, "car_box: steer\n", "")
    with pytest.raises(ValueError, match=r"0\.1\.1: unit\.yaml has exercises but no `car_box`"):
        bp.exercise_records(mini_repo)


def test_a_unit_without_exercises_needs_no_car_box(mini_repo):
    rewrite(mini_repo, UNIT_YAML, "car_box: instruments\n", "")
    rewrite(mini_repo, UNIT_YAML, 'exercises:\n  - {id: "0.1.1.a", name: lane_error_stats}\n  - {id: "0.1.1.b", name: steps_off_lane}', "exercises: []")
    assert bp.exercise_records(mini_repo) == []


def test_task_is_empty_until_unit_yaml_gives_one(mini_repo):
    assert [r["task"] for r in bp.exercise_records(mini_repo)] == ["", ""]
    rewrite(mini_repo, "course/0.1.1-mini/unit.yaml", "{id: \"0.1.1.b\", name: steps_off_lane}",
            "{id: \"0.1.1.b\", name: steps_off_lane, task: Count the steps off the lane}")
    assert [r["task"] for r in bp.exercise_records(mini_repo)] == ["", "Count the steps off the lane"]


def test_the_week_counts_every_listed_unit_not_only_the_ready_ones(mini_repo):
    assert {r["week"] for r in bp.exercise_records(mini_repo)} == {1}
    rewrite(mini_repo, "course/syllabus.yaml", "          - {id: \"0.1.1\"",
            "          - {id: \"0.0.9\", folder: 0.0.9-earlier, title: Earlier, status: planned}\n          - {id: \"0.1.1\"")
    records = bp.exercise_records(mini_repo)
    assert {r["week"] for r in records} == {2} and [r["unit_id"] for r in records] == ["0.1.1", "0.1.1"]
    assert records[0]["label"] == "Week 2 · Mini unit · Exercise A" and records[1]["label"].endswith("Exercise B")


def test_an_exercise_without_a_tagged_lab_cell_fails_naming_it(mini_repo):
    rewrite(mini_repo, "course/0.1.1-mini/lab.py", 'tags=["exercise:0.1.1.b"]', 'tags=["exercise:0.1.1.x"]')
    with pytest.raises(ValueError, match="0.1.1.b"):
        bp.exercise_records(mini_repo)


# --- validate_lecture_links ----------------------------------------------------------------------------------


@pytest.mark.parametrize("link", [
    "[Open in the playground](playground/index.html?exercise=0.1.1.a)",
    "[Try it](https://intervuhq.com/learnfsd/playground/index.html?exercise=0.1.1.b&preset=starter)",
    '<a href="playground/index.html?exercise=0.1.1.a&preset=starter">try it</a>',
    "The playground (playground/index.html) opens on its own; the folder `playground/` holds its files.",
])
def test_lecture_links_that_resolve_pass(mini_repo, link):
    link_in_lecture(mini_repo, "0.1.1-mini", link)
    bp.validate_lecture_links(mini_repo, bp.exercise_records(mini_repo))


def test_every_lecture_link_in_the_repo_resolves(records):
    bp.validate_lecture_links(REPO, records)


@pytest.mark.parametrize(("link", "bad_link", "why"), [
    ("[Try it](playground/index.html?exercise=9.9.9.z)", "playground/index.html?exercise=9.9.9.z", "no such exercise"),
    ("[Try it](playground/index.html?exercise=9.9.9.z&preset=starter)", "playground/index.html?exercise=9.9.9.z&preset=starter", "no such exercise"),
    ("[Try it](playground/index.html?exercise=0.1.1.a&preset=nope)", "playground/index.html?exercise=0.1.1.a&preset=nope", "exercise 0.1.1.a has presets ['starter']"),
])
def test_a_lecture_link_to_a_missing_exercise_or_preset_names_the_lecture_the_link_and_why(mini_repo, link, bad_link, why):
    link_in_lecture(mini_repo, "0.1.1-mini", link)
    with pytest.raises(ValueError) as caught:
        bp.validate_lecture_links(mini_repo, bp.exercise_records(mini_repo))
    assert f"course/0.1.1-mini/lecture.md: {bad_link} ({why}" in str(caught.value)


@pytest.mark.parametrize("link", [
    "[Try it](/learnfsd/playground/?exercise=0.1.1.a)",
    "[Try it](playground/?exercise=0.1.1.a&preset=starter)",
    "[Try it](playground/index.html?preset=starter&exercise=0.1.1.a)",
])
def test_a_link_in_a_form_the_site_does_not_serve_fails_even_when_the_exercise_exists(mini_repo, link):
    link_in_lecture(mini_repo, "0.1.1-mini", link)
    with pytest.raises(ValueError, match=re.escape("use playground/index.html?exercise=<id>")):
        bp.validate_lecture_links(mini_repo, bp.exercise_records(mini_repo))


def test_every_bad_link_of_every_lecture_is_listed_at_once(mini_repo):
    (mini_repo / "course" / "0.9.9-other").mkdir()
    (mini_repo / "course" / "0.9.9-other" / "lecture.md").write_text("[x](playground/index.html?exercise=8.8.8.y)\n")
    link_in_lecture(mini_repo, "0.1.1-mini", "[x](playground/index.html?exercise=9.9.9.z)")
    with pytest.raises(ValueError) as caught:
        bp.validate_lecture_links(mini_repo, bp.exercise_records(mini_repo))
    message = str(caught.value)
    assert "course/0.9.9-other/lecture.md" in message and "8.8.8.y" in message
    assert "course/0.1.1-mini/lecture.md" in message and "9.9.9.z" in message


def test_a_bad_link_stops_the_build_before_the_previous_one_is_touched(mini_repo, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("previous build")
    link_in_lecture(mini_repo, "0.1.1-mini", "[x](playground/index.html?exercise=9.9.9.z)")
    with pytest.raises(ValueError, match="9.9.9.z"):
        bp.write_playground(mini_repo, out)
    assert (out / "keep.txt").read_text() == "previous build"


# --- write_playground ----------------------------------------------------------------------------------------


def test_dist_holds_the_page_files_and_the_generated_ones_and_nothing_else(dist):
    page = {p.name for p in (REPO / "playground").iterdir() if p.is_file() and not p.name.startswith(".")}
    assert "shim.py" in page
    assert {p.name for p in dist.iterdir()} == page | GENERATED
    assert all(p.is_file() for p in dist.iterdir())


def test_hidden_files_and_folders_of_playground_are_not_published(mini_repo, tmp_path):
    page = mini_repo / "playground"
    (page / "__pycache__").mkdir(parents=True)
    (page / "__pycache__" / "shim.cpython-312.pyc").write_bytes(b"x")
    (page / ".DS_Store").write_bytes(b"x")
    (page / "app.js").write_text("export {};\n")
    (mini_repo / "zero2fsd").mkdir()
    (mini_repo / "zero2fsd" / "__init__.py").write_text('"""stub"""\n')
    bp.write_playground(mini_repo, tmp_path / "out")
    assert {p.name for p in (tmp_path / "out").iterdir()} == {"app.js"} | GENERATED


def test_page_files_are_copied_byte_for_byte(dist):
    for source in (REPO / "playground").iterdir():
        if source.is_file() and not source.name.startswith("."):
            assert (dist / source.name).read_bytes() == source.read_bytes()


def test_the_data_files_read_back_as_what_they_were_built_from(dist, records):
    assert json.loads((dist / "exercises.json").read_text(encoding="utf-8")) == records
    assert json.loads((dist / "roads.json").read_text()) == roads_dict()
    assert json.loads((dist / "palette.json").read_text()) == palette_dict()
    assert (dist / "roads.json").read_text() == json.dumps(roads_dict(), sort_keys=True) + "\n"


def test_the_zip_holds_the_package_sources_and_no_solutions_tests_or_caches(dist):
    with zipfile.ZipFile(dist / "bundle.zip") as archive:
        names = archive.namelist()
        assert "zero2fsd/__init__.py" in names and names == sorted(names)
        assert all(n.startswith("zero2fsd/") and n.endswith(".py") for n in names)
        assert not [n for n in names if any(word in n for word in ("solutions", "tests", "__pycache__"))]
        assert archive.read("zero2fsd/car/car.py") == (REPO / "zero2fsd" / "car" / "car.py").read_bytes()


def test_the_zip_can_be_imported_without_the_repo(dist, tmp_path):
    code = "import sys; sys.path.insert(0, sys.argv[1]); import zero2fsd.car, zero2fsd.grade; print(zero2fsd.car.__file__)"
    done = subprocess.run([sys.executable, "-c", code, str(dist / "bundle.zip")], capture_output=True, text=True, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith(str(dist / "bundle.zip"))


def test_the_zip_bytes_depend_on_content_only(tmp_path):
    copy = tmp_path / "zero2fsd"
    shutil.copytree(REPO / "zero2fsd", copy)
    (copy / "__pycache__").mkdir(exist_ok=True)
    (copy / "__pycache__" / "stray.cpython-312.pyc").write_bytes(b"x")
    for path in copy.rglob("*"):
        os.utime(path, (0, 0))
    assert bp.bundle_zip(copy) == bp.bundle_zip(REPO / "zero2fsd")


def test_manifest_lists_every_other_file_with_its_size_and_hash(dist):
    manifest = json.loads((dist / "manifest.json").read_text())
    paths = [f["path"] for f in manifest["files"]]
    assert paths == sorted(p.name for p in dist.iterdir() if p.name != "manifest.json")
    for entry in manifest["files"]:
        data = (dist / entry["path"]).read_bytes()
        assert (entry["bytes"], entry["sha256"]) == (len(data), hashlib.sha256(data).hexdigest())
    assert manifest["generated_from"] == bp.git_revision(REPO)


def test_git_revision_is_the_short_hash_in_a_checkout_and_unknown_outside_one(tmp_path):
    assert bp.git_revision(tmp_path) == "unknown"
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@example.org"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    head = subprocess.run([*git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert bp.git_revision(tmp_path) == head and re.fullmatch(r"[0-9a-f]{7,}", head)


def test_write_replaces_the_directory_and_creates_missing_parents(tmp_path):
    out = tmp_path / "deep" / "er" / "dist"
    bp.write_playground(REPO, out)
    (out / "stale.js").write_text("old")
    bp.write_playground(REPO, out)
    assert not (out / "stale.js").exists() and (out / "manifest.json").is_file()
```

Add `car_box` to the fixture unit, `tests/zero2fsd/fixtures/mini_repo/course/0.1.1-mini/unit.yaml`, after line 7:

```yaml
box: none
car_box: instruments
go_deeper:
```

Edit `tests/zero2fsd/conftest.py`. Add the import after line 11 (`scripts/` is already on `sys.path` from line 8):

```python
import numpy as np
import pytest

from _course import REPO
from zero2fsd.car.runner import Telemetry
```

Replace `built_repo` (lines 40-47) so the main-wiring tests have the sources `write_playground` packs:

```python
@pytest.fixture
def built_repo(mini_repo):
    """The fixture repo with its notebook and figures generated by the real scripts, and the real playground/ and
    zero2fsd/ sources that scripts/build_playground.py packs into course/dist/playground/."""
    import build_labs
    import make_figures

    assert build_labs.main(["--root", str(mini_repo)]) == 0 and make_figures.main(["--root", str(mini_repo)]) == 0
    for source in ("playground", "zero2fsd"):
        shutil.copytree(REPO / source, mini_repo / source, ignore=shutil.ignore_patterns("__pycache__"))
    return mini_repo
```

Edit `tests/zero2fsd/test_bundle.py`. Replace `test_main_exit_codes` (lines 80-84) and add the second test after it:

```python
def test_main_exit_codes(built_repo, capsys):
    assert bundler.main(["--root", str(built_repo)]) == 0 and "1 ready units" in capsys.readouterr().out
    assert (built_repo / "course" / "dist" / "playground" / "manifest.json").is_file()
    (built_repo / UNIT / "quiz.yaml").unlink()
    assert bundler.main(["--root", str(built_repo)]) == 1
    assert "quiz.yaml" in capsys.readouterr().err


def test_main_fails_on_a_lecture_link_to_an_exercise_that_does_not_exist(built_repo, capsys):
    edit(built_repo, f"{UNIT}/lecture.md", "# Mini unit\n", "# Mini unit\n\n[Try it](playground/index.html?exercise=9.9.9.z)\n")
    assert bundler.main(["--root", str(built_repo)]) == 1
    assert "9.9.9.z" in capsys.readouterr().err
```

Edit `tests/playground/test_shim.py`: delete `test_every_exercise_starter_runs_after_the_prelude_and_defines_its_name` (lines 77-83: the test and its two trailing blank lines). `test_every_starter_runs_after_the_prelude_and_defines_the_exercise_name` in `test_build.py` runs the same check over the exercise records, which are the data the page really loads, so the older test (which re-derived the starters from the lab cells) is dead weight. `lab_cells()` stays: `test_prelude_binds_every_name_the_lab_setup_cells_bind` still uses it.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/zero2fsd/test_bundle.py::test_main_exit_codes tests/zero2fsd/test_bundle.py::test_main_fails_on_a_lecture_link_to_an_exercise_that_does_not_exist -v`
Expected: FAIL, 2 failed. `test_main_exit_codes`: `AssertionError: assert False` where `False = is_file()` on `<tmp>/repo/course/dist/playground/manifest.json`. The link test: `AssertionError: assert 0 == 1` (`bundler.main` returned 0 because nothing validates links yet).

Run: `python -m pytest tests/playground/test_build.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'build_playground'` at `tests/playground/test_build.py:12` (`import build_playground as bp`), "Interrupted: 1 error during collection".

- [ ] **Step 3: Write the minimal implementation**

Create `scripts/build_playground.py`:

```python
"""Build course/dist/playground/: the page sources, the Python bundle and the data the playground reads.

    python scripts/build_course_bundle.py     # writes course.json, then calls write_playground

Fails loudly (ValueError, exit 1 through build_course_bundle) when a unit with exercises has no `car_box`, an
exercise has no lab.py cell, or a lecture links to a playground exercise or preset that does not exist.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import string
import subprocess
import zipfile
from pathlib import Path

from _course import colab_url, load_yaml, parse_lab, syllabus_units, unit_slug
from zero2fsd.export import roads_dict
from zero2fsd.palette import palette_dict
from zero2fsd.preview import get

PLAYGROUND_LINK = re.compile(r'playground/index\.html\?exercise=([^&"\s)]+)(?:&preset=([^&"\s)]+))?')  # the form the site serves
WRONG_FORM_LINK = re.compile(r'playground/(?!index\.html\?exercise=)[^\s)"]*exercise=[^\s)"]*')  # e.g. playground/?exercise=: a 404
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)  # the earliest date a zip can hold: the same bytes whenever it is built


def exercise_records(repo_root: Path) -> list[dict]:
    """One record per exercise of every ready unit, in syllabus order: everything the page needs to open it."""
    records = []
    for week, unit in enumerate(syllabus_units(repo_root), start=1):  # the week counts every listed unit, ready or not
        if unit["status"] != "ready":
            continue
        folder = repo_root / "course" / unit["folder"]
        meta = load_yaml(folder / "unit.yaml")
        if meta["exercises"] and not meta.get("car_box"):
            raise ValueError(f"{unit['id']}: unit.yaml has exercises but no `car_box` (the part of the car they build)")
        cells = parse_lab((folder / "lab.py").read_text(encoding="utf-8"))
        for index, exercise in enumerate(meta["exercises"]):
            tag = f"exercise:{exercise['id']}"
            starter = "\n\n".join(c.source for c in cells if c.kind == "code" and tag in c.tags)
            if not starter:
                raise ValueError(f"{exercise['id']}: no code cell of {unit['folder']}/lab.py is tagged {tag!r}")
            preview = get(exercise["id"])
            records.append({
                "id": exercise["id"], "unit_id": unit["id"], "unit_key": unit_slug(unit["id"]), "week": week,
                "unit_title": meta["title"], "label": f"Week {week} · {meta['title']} · Exercise {string.ascii_uppercase[index]}",
                "name": exercise["name"], "task": exercise.get("task", ""), "car_box": meta["car_box"],
                "kind": preview.kind, "road": preview.road, "presets": {"starter": starter}, "colab_url": colab_url(unit["folder"]),
            })
    return records


def validate_lecture_links(repo_root: Path, records: list[dict]) -> None:
    """Raise ValueError naming every lecture link to a playground exercise or preset that does not exist."""
    presets = {r["id"]: r["presets"] for r in records}
    bad = []
    for lecture in sorted((repo_root / "course").glob("*/lecture.md")):
        text, where = lecture.read_text(encoding="utf-8"), lecture.relative_to(repo_root).as_posix()
        bad += [f"{where}: {m[0]} (use playground/index.html?exercise=<id>)" for m in WRONG_FORM_LINK.finditer(text)]
        for link in PLAYGROUND_LINK.finditer(text):
            exercise, preset = link.groups()
            if exercise not in presets:
                bad.append(f"{where}: {link[0]} (no such exercise)")
            elif preset and preset not in presets[exercise]:
                bad.append(f"{where}: {link[0]} (exercise {exercise} has presets {sorted(presets[exercise])})")
    if bad:
        raise ValueError("lecture links to the playground that do not resolve:\n  " + "\n  ".join(bad))


def bundle_zip(package: Path) -> bytes:
    """Every .py file of `package`, sorted, under its own name; stored with a fixed date so the bytes depend on content only."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for path in sorted(package.rglob("*.py")):
            info = zipfile.ZipInfo(f"{package.name}/{path.relative_to(package).as_posix()}", ZIP_EPOCH)
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def git_revision(repo_root: Path) -> str:
    """The short commit hash of `repo_root`, or "unknown" when it is not a git checkout."""
    done = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else "unknown"


def _json(data) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False) + "\n"


def write_playground(repo_root: Path, out_dir: Path) -> None:
    """Replace `out_dir` with the build: the playground/ files, bundle.zip, three data files and manifest.json."""
    records = exercise_records(repo_root)
    validate_lecture_links(repo_root, records)
    shutil.rmtree(out_dir, ignore_errors=True)  # a source renamed since the last build must not leave its old copy
    out_dir.mkdir(parents=True)
    for source in sorted((repo_root / "playground").iterdir()):
        if source.is_file() and not source.name.startswith("."):
            shutil.copyfile(source, out_dir / source.name)
    (out_dir / "bundle.zip").write_bytes(bundle_zip(repo_root / "zero2fsd"))
    for name, data in (("exercises.json", records), ("roads.json", roads_dict()), ("palette.json", palette_dict())):
        (out_dir / name).write_text(_json(data), encoding="utf-8")
    files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(out_dir.iterdir())]  # manifest.json does not exist yet, so it is not listed in itself
    (out_dir / "manifest.json").write_text(_json({"generated_from": git_revision(repo_root), "files": files}), encoding="utf-8")
```

Edit `scripts/_course.py`. Insert `colab_url` between `unit_slug` (lines 22-24) and `heading_slug` (line 27):

```python
def unit_slug(unit_id: str) -> str:
    """URL slug of a unit id: dots become hyphens."""
    return unit_id.replace(".", "-")


def colab_url(folder: str) -> str:
    """The Colab link of a unit's lab notebook; `folder` is its directory under course/."""
    return f"https://colab.research.google.com/github/{REPO_SLUG}/blob/main/course/{folder}/lab.ipynb"


def heading_slug(text: str) -> str:
```

Edit `scripts/build_course_bundle.py`. Four changes.

Docstring, lines 1-7:

```python
"""Build course/dist/course.json (+ figures): the ONE interface the site reads.  Schema version 1.

    python scripts/build_course_bundle.py [--root DIR]

Fails loudly (BundleError, exit 1) when a ready unit is missing a file, a field, a figure the lecture
references, or a quiz `section` that is not a heading of the lecture.  It then writes course/dist/playground/
(scripts/build_playground.py), which fails the same way on a lecture link to an exercise that does not exist.
"""
```

Imports, line 18 (add the new import below it):

```python
from _course import FENCE, REPO, REPO_SLUG, colab_url, heading_slug, load_yaml, syllabus_units, unit_slug
from build_playground import write_playground
```

Line 119, inside `build_unit`:

```python
        "colab_url": colab_url(entry["folder"]),
```

`main`, lines 168-179:

```python
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO)
    root = parser.parse_args(argv).root
    playground = root / "course" / "dist" / "playground"
    try:
        out = write_bundle(root)
        write_playground(root, playground)
    except (BundleError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    bundle = json.loads(out.read_text(encoding="utf-8"))
    print(f"wrote {out} ({len(bundle['units'])} ready units, {len(bundle['figures'])} figures)")
    print(f"wrote {playground}")
    return 0
```

Add one `car_box:` line to each real unit, directly after the prose `box:` on line 4 (the key names the part of the car the unit's exercises build; the page uses it, nothing enumerates the values):

```yaml
# course/0.1.1-meet-your-car/unit.yaml, after line 4
car_box: instruments
# course/0.1.2-images-are-arrays/unit.yaml, after line 4
car_box: perception
# course/0.1.3-your-first-perception-box/unit.yaml, after line 4
car_box: perception
# course/0.1.4-closing-the-loop/unit.yaml, after line 4
car_box: steer
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/playground tests/zero2fsd/test_bundle.py -v`
Expected: PASS (129 tests: 51 in `test_shim.py`, 38 in `test_build.py`, 40 in `test_bundle.py`).

Then check the real build once, and that the refactor left `course.json` alone:

Run: `python scripts/build_course_bundle.py && git status --short course/dist && ls course/dist/playground`
Expected: two lines, `wrote <repo>/course/dist/course.json (4 ready units, 16 figures)` and `wrote <repo>/course/dist/playground`; `git status` prints only `?? course/dist/playground/` (so `course.json` and the figures are byte-identical to the committed ones); `ls` prints `bundle.zip exercises.json manifest.json palette.json roads.json shim.py`.

Then remove the smoke-run output (the page files do not exist yet, so it is not a build to commit): `rm -r course/dist/playground`.

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests -q`
Expected: all green; number of tests = 374 + 38 = 412 (38 new in `test_build.py`, 1 new in `test_bundle.py`, 1 removed from `test_shim.py`).

- [ ] **Step 6: Commit**

```bash
git add scripts/build_playground.py scripts/_course.py scripts/build_course_bundle.py \
  course/0.1.1-meet-your-car/unit.yaml course/0.1.2-images-are-arrays/unit.yaml \
  course/0.1.3-your-first-perception-box/unit.yaml course/0.1.4-closing-the-loop/unit.yaml \
  tests/playground/test_build.py tests/playground/test_shim.py \
  tests/zero2fsd/conftest.py tests/zero2fsd/test_bundle.py \
  tests/zero2fsd/fixtures/mini_repo/course/0.1.1-mini/unit.yaml
git commit -m "feat(playground): build course/dist/playground from the lab cells, the preview registry and the roads

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 9: Module worker (`playground/worker.js`)

**Files:**
- Create: `playground/worker.js`
- Test: `/tmp/playground-checks/check_worker.py` and `/tmp/playground-checks/solutions.py` (throwaway scripts outside the repo; Task 12 holds the repeatable pytest version of the browser assertions)

**Interfaces:**
- Consumes:
  - The Task 7 shim in `playground/shim.py`: `learner_object(code, name)`, `preview_run(obj, exercise_id, road, on_row)`, `preview_stats(obj, exercise_id)`, `check_code(obj, exercise_id, on_row=None)` (third parameter added, see the NOTE), `preview_image(obj, exercise_id, seed=0)` (see the NOTE), `frame_rgb(road, k, x, y, yaw, speed, seed, noise_sigma=0.0)`, `dashboard_png()`, `format_error(exc)`, `probe_steps_per_s(n=100)`.
  - The Task 8 output folder `course/dist/playground/`: `bundle.zip`, `shim.py`, `exercises.json`, `roads.json`, `palette.json`, and whichever `playground/` source files exist.
  - `zero2fsd/sim/camera.py:53-54`, `Camera(width=320, height=180, ...)`. The worker reads the frame size from this class, so `320` and `180` are typed nowhere in JS.
  - Pyodide 314.0.7 at `https://cdn.jsdelivr.net/pyodide/v314.0.7/full/`.
- Produces (Tasks 11 and 12 rely on these; `exercise` is always the full record from `exercises.json`):
  - Page to worker (`e.data.type`): `init {}`; `run {code, exercise, road}`; `check {code, exercise}`; `stats {code, exercise}`; `image {code, exercise}`; `frame {road, k, x, y, yaw, speed, seed}`; `dashboard {}`.
  - Worker to page: `progress {stage: "pyodide" | "packages" | "bundle" | "probe", detail}`; `ready {versions: {pyodide, python, numpy, pillow}, steps_per_s_probe}`; `row {row}` (the 14 `COLUMNS` of Task 1 plus `road`); `stdout {text}`; `done {metrics}`; `verdict {passed, details}`; `stats {values}`; `image {width, height, frame, overlay}` (two RGB `ArrayBuffer`s); `frame {k, width, height, rgb}`; `dashboard {png}`; `error {message, line, traceback, request}`.
  - What B6 asserts for this task (`check_worker.py`, and cases 2, 5, 6, 12 of `check_page.py` in Task 11): the four load stages in order and then `ready` with `314.0.7`; no `stdout` from package loading; a run streams one `row` per step, all tagged with the requested road; a check of a driving exercise streams exactly one drive (the gate drive), tagged with the exercise's road, and a check of a function or image exercise streams nothing; a stats run returns floats; an image run returns two 320 x 180 RGB buffers that differ; a `1 / 0` body reports `line == 2` and `request == "run"`; a missing function name and an unknown message type are each reported by name; the dashboard PNG starts with the PNG magic bytes.

> NOTE: Week 1 has an image exercise (0.1.2.a), which needs a message of its own, so the worker adds `image {code, exercise}` and replies `image {width, height, frame, overlay}`. Task 7 must provide `preview_image(obj, exercise_id, seed=0) -> {"width": int, "height": int, "frame": bytes, "overlay": bytes}`. The overlay is the learner's masks on black, in the scene's yellow and white. The page and the worker never look at its colours.

> NOTE: `check_code` gains a third parameter, `on_row` (default `None`), and Task 7 already adds it. It streams only the gate drive (`zero2fsd/grade/_drive.py:32-39` and `:11`, `GATE_SCENARIO = "gentle"`), and its rows are the 14 `COLUMNS` with no road. The worker therefore tags a check's rows with `exercise.road`, which is the gate road for every drive exercise (Task 5 registers each with `road=GATE_SCENARIO`). A row the shim tags itself keeps its own tag, so a future exercise that is graded on another road only needs the shim to say so. A row with no known road stops the page with a named error at the first step. That is deliberate: a guessed road would draw the car on the wrong map.

> NOTE: the `error` message gains `request`, the type of the message that failed. The page treats `request == "init"` as a failed load (message, Retry, Colab link) and anything else as a failed run.

> NOTE: an integrity hash cannot apply to a dynamic `import()` inside a module worker, so Pyodide is pinned by exact version only (header deviation P8). The import is dynamic on purpose: a load failure then reaches the page as an `error` message that names the URL.

> NOTE: Tasks 9 to 11 land one at a time. Task 8's `write_playground` copies every file in `playground/`, so each build holds whatever sources exist so far, and the Step 2 failures below depend on that.

> NOTE: `check_worker.py` was run in headless Chromium against a scratch shim that follows the interfaces above. The first run against the real Task 7 shim is Step 4 here, so a mismatch shows up as a named assertion.

> NOTE (a blocked package): spec section 6 row 5 wants a failed download to be an error that names the file. Checked on 2026-10-11 in headless Chromium (Pyodide 314.0.7, a scratch page that loads `pyodide.mjs` and aborts the request for the matplotlib wheel): `loadPackage` resolved without throwing; its error callback received "Failed to load matplotlib" and "Failed to fetch"; `loadedPackages` had no `matplotlib`; and the next `import` raised "'matplotlib' is included in the Pyodide distribution, but it is not installed" with micropip advice the learner cannot use. So `loadPackages` reads `loadedPackages` and names the wheel from `pyodide.lockfile.packages[key].file_name` (read back in the same probe: `matplotlib-3.10.8-cp314-cp314-pyemscripten_2026_0_wasm32.whl`). It serves both `loadPackage` calls, the start-up one (numpy, pillow) and the Dashboard one, so a blocked numpy wheel is the same kind of error. The probe ran `loadPackage` in a scratch page, not this worker: the first run of `loadPackages` is Task 12 `test_route_abort_names_the_matplotlib_wheel`.

- [ ] **Step 1: Write the failing test**

Prerequisite. Run from the course repo root with the repo `.venv` active. This installs a browser (about 300 MB) and changes nothing in the repo. Task 12 adds the `test` extra later, so for now install it by hand:

```bash
python -m pip install "playwright>=1.40"
python -m playwright install chromium
mkdir -p /tmp/playground-checks
```

Save the reference solutions as `/tmp/playground-checks/solutions.py`. Each one is self-contained (its own imports), so it runs in the editor exactly as typed. They use no preset, because v1 presets hold only `starter`.

```python
"""Reference solutions for four exercises, each self-contained (its own imports) so it runs in the editor as typed."""

SOLUTIONS = {
    "0.1.4.a": '''import numpy as np
from zero2fsd.sim.dynamics import MAX_STEER


def p_steer(est, k_off, k_head):
    if not est.valid:
        return 0.0
    return float(np.clip(-(k_off * est.offset_m + k_head * est.heading_rad), -MAX_STEER, MAX_STEER))
''',
    "0.1.4.b": '''from zero2fsd.car import Car, provided

my_car = Car(
    perception=lambda obs: provided.estimate_lane(obs),
    controller=lambda est, obs: provided.controller(est, obs),
)
''',
    "0.1.1.a": '''import numpy as np


def lane_error_stats(lat_err):
    lat_err = np.asarray(lat_err)
    return {
        "mean_abs": float(np.mean(np.abs(lat_err))),
        "max_abs": float(np.max(np.abs(lat_err))),
        "rms": float(np.sqrt(np.mean(lat_err**2))),
    }
''',
    "0.1.2.a": '''from zero2fsd.car import provided


def paint_masks(frame):
    return provided.paint_masks(frame)
''',
}
```

Save the worker check as `/tmp/playground-checks/check_worker.py`. It starts the worker the way the page will, sends every message type, and asserts on the replies.

```python
# Throwaway check for Task 9 (worker protocol); Task 12 re-uses the same assertions inside pytest.
import sys

from playwright.sync_api import sync_playwright
from solutions import SOLUTIONS

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765/"
START = """
() => {
  window.messages = [];
  window.worker = new Worker("./worker.js", { type: "module" });
  window.worker.onmessage = ({ data }) => {
    const copy = { ...data };
    if (data.type === "image") {
      const frame = new Uint8Array(data.frame), overlay = new Uint8Array(data.overlay);
      let changed = 0;
      for (let i = 0; i < frame.length; i += 3) if (frame[i] !== overlay[i] || frame[i + 1] !== overlay[i + 1] || frame[i + 2] !== overlay[i + 2]) changed++;
      Object.assign(copy, { frame: frame.byteLength, overlay: overlay.byteLength, changed });
    }
    if (data.rgb) { copy.rgb_bytes = data.rgb.byteLength; delete copy.rgb; }
    if (data.png) { copy.png_magic = Array.from(new Uint8Array(data.png).slice(0, 4)); delete copy.png; }
    window.messages.push(copy);
  };
  window.worker.onerror = (e) => window.messages.push({ type: "onerror", message: e.message });
  window.worker.postMessage({ type: "init" });
}
"""


def send(page, message):
    seen = page.evaluate("window.messages.length")
    page.evaluate("m => window.worker.postMessage(m)", message)
    return seen


def reply(page, kind, since, timeout=120_000):
    """The first `kind` message after index `since`; a worker script error fails the wait at once."""
    handle = page.wait_for_function(
        "([kind, since]) => window.messages.slice(since).find((m) => m.type === kind || m.type === 'onerror')",
        arg=[kind, since], timeout=timeout)
    message = handle.json_value()
    assert message["type"] == kind, message
    return message


def since(page, n):
    return page.evaluate("n => window.messages.slice(n)", n)


def exercises(page):
    return {r["id"]: r for r in page.evaluate("fetch('exercises.json').then((r) => r.json())")}


def test_worker_protocol(page):
    page.goto(URL + "worker.js")  # any same-origin document will do; the worker needs no DOM
    page.evaluate(START)
    ready = reply(page, "ready", 0, timeout=90_000)
    stages = [m["stage"] for m in page.evaluate("window.messages") if m["type"] == "progress"]
    assert stages == ["pyodide", "packages", "bundle", "probe"], stages
    assert ready["versions"]["pyodide"] == "314.0.7" and ready["steps_per_s_probe"] > 0, ready
    assert not [m for m in page.evaluate("window.messages") if m["type"] == "stdout"], "package loading is not learner output"

    ex = exercises(page)
    a, b, stats, paint = ex["0.1.4.a"], ex["0.1.4.b"], ex["0.1.1.a"], ex["0.1.2.a"]

    n = send(page, {"type": "run", "code": SOLUTIONS["0.1.4.a"], "exercise": a, "road": "gentle"})
    done = reply(page, "done", n)
    rows = [m["row"] for m in since(page, n) if m["type"] == "row"]
    assert len(rows) == done["metrics"]["steps"] and rows[0]["t"] == 0 and done["metrics"]["completed"] is True
    assert {r["road"] for r in rows} == {"gentle"}, "a run tags every step with the road it was asked to drive"
    assert {"t", "x", "y", "yaw", "speed", "steer", "lat", "est_valid", "progress_m", "off_road"} <= set(rows[0])

    n = send(page, {"type": "run", "code": "print('hello from the learner')\n" + SOLUTIONS["0.1.4.a"], "exercise": a, "road": "straight"})
    reply(page, "done", n)
    assert any(m["type"] == "stdout" and "hello from the learner" in m["text"] for m in since(page, n))

    n = send(page, {"type": "check", "code": SOLUTIONS["0.1.4.a"], "exercise": a})
    verdict = reply(page, "verdict", n)
    assert verdict["passed"] is True and verdict["details"], verdict
    n = send(page, {"type": "check", "code": "def p_steer(est, k_off, k_head):\n    return 0.0\n", "exercise": a})
    assert reply(page, "verdict", n)["passed"] is False
    assert not [m for m in since(page, n) if m["type"] == "row"], "a function check drives nothing, so it streams nothing"

    n = send(page, {"type": "check", "code": SOLUTIONS["0.1.4.b"], "exercise": b})  # a second exercise; its check drives the gate road
    verdict = reply(page, "verdict", n)
    streamed = [m["row"] for m in since(page, n) if m["type"] == "row"]
    assert verdict["passed"] is True, verdict
    assert len(streamed) > 500 and {r["road"] for r in streamed} == {b["road"]}, "the gate drive streams, tagged with the exercise's road"
    assert [r["t"] for r in streamed].count(0) == 1, "one drive, not the cliffhanger after it"

    n = send(page, {"type": "stats", "code": SOLUTIONS["0.1.1.a"], "exercise": stats})
    values = reply(page, "stats", n)["values"]
    assert values and all(isinstance(v, float) for v in values.values()), values

    n = send(page, {"type": "image", "code": SOLUTIONS["0.1.2.a"], "exercise": paint})  # a third kind: masks marked on a camera frame
    image = reply(page, "image", n)
    assert (image["width"], image["height"], image["frame"], image["overlay"]) == (320, 180, 320 * 180 * 3, 320 * 180 * 3), image
    assert image["changed"] > 100, "the overlay differs from the frame where the learner's masks mark pixels"
    n = send(page, {"type": "image", "code": "def paint_masks(frame):\n    return None\n", "exercise": paint})
    error = reply(page, "error", n)
    assert "NoneType" in error["message"] and error["request"] == "image", error
    n = send(page, {"type": "check", "code": SOLUTIONS["0.1.2.a"], "exercise": paint})
    assert reply(page, "verdict", n)["passed"] is True
    assert not [m for m in since(page, n) if m["type"] == "row"], "an image check streams no drive"

    last = rows[400]
    n = send(page, {"type": "frame", "road": "gentle", "k": 400, **{key: last[key] for key in ("x", "y", "yaw", "speed")}, "seed": 0})
    frame = reply(page, "frame", n)
    assert (frame["k"], frame["width"], frame["height"], frame["rgb_bytes"]) == (400, 320, 180, 320 * 180 * 3), frame

    # The dashboard draws the last completed run (the straight run above printing 'hello'); asked after a failed run it errors by design.
    n = send(page, {"type": "dashboard"})
    png = reply(page, "dashboard", n, timeout=180_000)
    assert png["png_magic"] == [137, 80, 78, 71], png

    n = send(page, {"type": "run", "code": "def p_steer(est, k_off, k_head):\n    return 1 / 0\n", "exercise": a, "road": "gentle"})
    error = reply(page, "error", n)
    assert "ZeroDivisionError" in error["message"] and error["line"] == 2 and error["request"] == "run", error
    assert "Traceback" in error["traceback"]

    n = send(page, {"type": "run", "code": "x = 1\n", "exercise": a, "road": "gentle"})
    assert "p_steer" in reply(page, "error", n)["message"], "a missing function name is reported by name"

    n = send(page, {"type": "nonsense"})
    assert "nonsense" in reply(page, "error", n)["message"]


if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.on("pageerror", lambda e: print("pageerror", e)); page.on("console", lambda m: print("console", m.text)); page.on("requestfailed", lambda r: print("requestfailed", r.url))
        try:
            test_worker_protocol(page)
            print("PASS: worker protocol")
        finally:
            print("messages:", page.evaluate("JSON.stringify(window.messages).slice(0, 1500)"))
            browser.close()
```

> NOTE (the dashboard comes before the failing runs): a dashboard asked for after a failed run errors by design with `RuntimeError("run the car first: ...")`, because `preview_run` clears the previous run before it starts and keeps nothing when the learner's code raises. `tests/playground/test_shim.py` pins both halves: `test_a_learner_that_raises_mid_run_stops_the_run_and_the_next_run_works` (a run that raises leaves `LAST_RUN` as `None`) and `test_dashboard_png_shows_the_last_completed_run` (no run, no dashboard). So `check_worker.py` asks for the dashboard while the last run is the completed straight run, before the `1 / 0` and `x = 1` runs; a check (`check_code`) does not touch the last run. The first draft asked after those two failed runs and could not pass against the Task 7 shim.

- [ ] **Step 2: Run the test to verify it fails**

```bash
python scripts/build_course_bundle.py
python -m http.server 8765 --directory course/dist/playground &
until curl -s -o /dev/null http://localhost:8765/; do :; done
(cd /tmp/playground-checks && python -u check_worker.py http://localhost:8765/)
```

Expected: FAIL with `AssertionError: {'type': 'onerror', 'message': None}`. The folder has no `worker.js` yet, so the worker's error event is the first message.

- [ ] **Step 3: Write the minimal implementation**

Create `playground/worker.js`. The curl line in its header is the verified result for the Pyodide URL.

```js
// Module worker: the protocol between the page and Pyodide, nothing else. Python decides, JS carries.
// Pyodide 314.0.7 from jsDelivr, verified 2026-10-11:
//   $ curl -sI https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs | head -1
//   HTTP/2 200
// It must stay a module worker (`new Worker("./worker.js", {type: "module"})`): the classic worker
// failed in the spike because importScripts threw. The import is dynamic so that a failed load
// reaches the page as an `error` message that names the URL, instead of a bare worker error event.
const PYODIDE_VERSION = "314.0.7";
const PYODIDE_BASE = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const PACKAGES = ["numpy", "pillow"];
const QUIET = { messageCallback: () => {} }; // package progress goes out as `progress` messages; it is not the learner's output

let pyodide;
let shim;
let frameSize; // [width, height] of the camera image, read from the camera that renders it

const post = (type, fields = {}, transfer = []) => postMessage({ type, ...fields }, transfer);

/** Await `work`; a failure is re-thrown naming the file it was fetching. */
async function loading(url, work) {
  try {
    return await work;
  } catch (cause) {
    throw new Error(`Could not load ${url}: ${cause.message}`);
  }
}

/** `loadPackage` reports a failed download through its error callback and still resolves, so read what loaded: a package
 *  the lockfile lists that did not load is a failure that names its wheel. `loadedPackages` is keyed by the lockfile's
 *  display name ("Pillow"), the lockfile by the package key ("pillow"). */
async function loadPackages(keys) {
  await pyodide.loadPackage(keys, QUIET);
  const failed = keys.map((key) => pyodide.lockfile.packages[key]).filter(({ name }) => !pyodide.loadedPackages[name]);
  if (failed.length) throw new Error(`the download failed for ${failed.map(({ file_name }) => file_name).join(", ")}`);
}

async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.arrayBuffer();
}

/** Python value to plain JS; the proxy is released. */
function plain(proxy) {
  try {
    return proxy.toJs({ dict_converter: Object.fromEntries });
  } finally {
    proxy.destroy();
  }
}

/** A Python bytes object as a standalone ArrayBuffer (a copy, so it can be transferred). */
function bytesOf(proxy) {
  try {
    return proxy.toJs().buffer;
  } finally {
    proxy.destroy();
  }
}

/** `on_row` for Python: each step goes to the page the moment it exists, tagged with `road`. A step the shim already
 *  tagged keeps its own tag. A run passes the road it was asked for. A check passes the exercise's own road: only a
 *  gate drive streams, and every drive exercise's record road is that gate road (GATE_SCENARIO, grade/_drive.py:11). */
const onRow = (road) => (rowProxy) => post("row", { row: { road, ...rowProxy.toJs({ dict_converter: Object.fromEntries }) } });

async function init() {
  post("progress", { stage: "pyodide", detail: `Pyodide ${PYODIDE_VERSION}` });
  const { loadPyodide } = await loading(`${PYODIDE_BASE}pyodide.mjs`, import(`${PYODIDE_BASE}pyodide.mjs`));
  pyodide = await loading(PYODIDE_BASE, loadPyodide({ indexURL: PYODIDE_BASE }));
  pyodide.setStdout({ batched: (text) => post("stdout", { text }) });
  post("progress", { stage: "packages", detail: PACKAGES.join(", ") });
  await loading(`${PYODIDE_BASE} (${PACKAGES.join(", ")})`, loadPackages(PACKAGES));
  post("progress", { stage: "bundle", detail: "bundle.zip" });
  pyodide.unpackArchive(await loading("bundle.zip", fetchBytes("bundle.zip")), "zip");
  pyodide.FS.writeFile("shim.py", new Uint8Array(await loading("shim.py", fetchBytes("shim.py"))));
  shim = pyodide.pyimport("shim");
  frameSize = plain(pyodide.runPython("from zero2fsd.sim.camera import Camera\ncamera = Camera()\n[camera.width, camera.height]"));
  post("progress", { stage: "probe", detail: "timing the simulator" });
  const stepsPerS = shim.probe_steps_per_s(100);
  const [python, numpy, pillow] = plain(pyodide.runPython("import sys, numpy, PIL\n[sys.version.split()[0], numpy.__version__, PIL.__version__]"));
  post("ready", { versions: { pyodide: pyodide.version, python, numpy, pillow }, steps_per_s_probe: stepsPerS });
}

/** Build the learner's object, hand it to `use`, and release it. */
async function withLearner({ code, exercise }, use) {
  const learner = shim.learner_object(code, exercise.name);
  try {
    return await use(learner);
  } finally {
    learner.destroy?.(); // a function or Car is a proxy; anything else was already converted to a JS value
  }
}

const HANDLERS = {
  init,

  run: (m) => withLearner(m, (learner) => {
    const metrics = plain(shim.preview_run(learner, m.exercise.id, m.road, onRow(m.road)));
    post("done", { metrics });
  }),

  check: (m) => withLearner(m, (learner) => {
    post("verdict", plain(shim.check_code(learner, m.exercise.id, onRow(m.exercise.road))));
  }),

  stats: (m) => withLearner(m, (learner) => {
    post("stats", { values: plain(shim.preview_stats(learner, m.exercise.id)) });
  }),

  frame: ({ road, k, x, y, yaw, speed, seed }) => {
    const rgb = bytesOf(shim.frame_rgb(road, k, x, y, yaw, speed, seed));
    const [width, height] = frameSize;
    post("frame", { k, width, height, rgb }, [rgb]);
  },

  image: (m) => withLearner(m, (learner) => {
    const { width, height, frame, overlay } = plain(shim.preview_image(learner, m.exercise.id));
    post("image", { width, height, frame: frame.buffer, overlay: overlay.buffer }, [frame.buffer, overlay.buffer]);
  }),

  dashboard: async () => {
    if (!pyodide.loadedPackages.matplotlib) {
      post("progress", { stage: "packages", detail: "matplotlib" });
      await loading(`${PYODIDE_BASE} (matplotlib)`, loadPackages(["matplotlib"]));
    }
    const png = bytesOf(shim.dashboard_png());
    post("dashboard", { png }, [png]);
  },
};

/** The page's one error shape. A Python exception is formatted by the shim, which finds the learner's line. */
function errorFields(error) {
  if (shim && error instanceof pyodide.ffi.PythonError) return plain(shim.format_error(pyodide.pyimport("sys").last_exc));
  return { message: error.message, line: null, traceback: error.stack ?? "" };
}

onmessage = async ({ data }) => {
  try {
    const handler = HANDLERS[data.type];
    if (!handler) throw new Error(`The worker has no handler for message type "${data.type}".`);
    await handler(data);
  } catch (error) {
    post("error", { ...errorFields(error), request: data.type });
  }
};
```

- [ ] **Step 4: Run the test to verify it passes**

The build replaces the folder, so stop the old server first and start a new one after the build.

```bash
pkill -f "[h]ttp.server 8765" || true
python scripts/build_course_bundle.py
python -m http.server 8765 --directory course/dist/playground &
until curl -s -o /dev/null http://localhost:8765/; do :; done
(cd /tmp/playground-checks && python -u check_worker.py http://localhost:8765/)
```

Expected: `PASS: worker protocol`, after about 35 s (a cold Pyodide download is most of it). The script then prints the first 1500 characters of the message log; the first lines show the four `progress` stages, then `ready` with `"pyodide":"314.0.7"`.

The Pyodide URL in the header was checked on 2026-10-11:

```bash
curl -sI https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs | head -1
# HTTP/2 200
```

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests -q`
Expected: all green; 412 passed, the same as after Task 8 (this task adds no Python tests).

- [ ] **Step 6: Commit**

```bash
git add playground/worker.js
git commit -m "feat(playground): module worker that runs the exercises in Pyodide

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Drawing and playback (`playground/view.js`, `playground/player.js`)

**Files:**
- Create: `playground/view.js`
- Create: `playground/player.js`
- Test: `/tmp/playground-checks/check_view_player.mjs` (throwaway node script; the drawing itself is checked in the browser by cases 2 and 14 of `check_page.py` in Task 11)

**Interfaces:**
- Consumes:
  - The Task 3 road dict: `closed`, `dash_period`, `dash_fraction`, and `lines.{tarmac_left, tarmac_right, yellow, white, ego_center}`, each a list of `[x, y]` world points.
  - The Task 4 palette: `quantity.{e_off, psi, delta, perception}`, `scene.{road, yellow, white}`, `chrome.{background, panel, border, text}`.
  - The Task 1 step row: `t, x, y, yaw, speed, steer, accel, lat, head_err, est_offset, est_heading, est_valid, progress_m, off_road`. `lat` and `head_err` are positive to the left.
  - `DT = 0.05` at `zero2fsd/sim/dynamics.py:8`: rows arrive 20 per simulated second.
- Produces (Task 11 relies on these):
  - `view.js`: `fitCanvas(canvas) -> {ctx, width, height}`; `fitTransform(road, width, height, padding = 24) -> {scale, ox, oy, width, height}`; `worldToCanvas(t, x, y) -> [px, py]`; `draw(ctx, road, rows, k, palette, t)`; `drawStrips(ctx, rows, k, palette, windowS = 10)`; `readoutText(row) -> {t, lat, speed, steer, progress}`; `describeRun(metrics, road, seconds) -> string`. None keeps state, so the lecture sliders can reuse them.
  - `player.js`: `rowIndexAt(rows, k, clock) -> number`, and `class Player` with `constructor(canvas, road, palette)`, `reset(road)`, `setRows(rows)`, `append(row)`, `finish()`, `fit()`, `play()`, `pause()`, `toggle()`, `seek(k)`, `setSpeed(multiplier)`, getters `k`, `length`, `rows`, `playing`, and `onTick(cb)` where `cb({k, row, playing})` runs on every drawn frame.
  - What B6 asserts for this task (cases 2 and 14 of `check_page.py`): the canvas holds more than one colour after a run; the strips canvas is drawn; dragging `#scrub` to step 100 shows `5.0 s`; Play and Pause flip the button label; `#to-end` moves the car to the last step.

> NOTE: `view.js` reads only `palette.chrome`, so it follows Task 4, which sets `CHROME` to `background #090d16`, `panel #0d1117` (the spec's values).

> NOTE: `Player` carries more than a bare tick loop, and Task 11 needs all of it: `reset` for a new drive, `finish` when a run ends, `fit` on resize, `rows` for the strips, and `playing` for the button label. The tick payload is `{k, row}`.

> NOTE: `draw` also paints the `e_off` brace (a line from the car to the nearest lane-centre point). The car is drawn at 4.5 x 1.8 m but never below 12 px long, because a 700 m loop on a phone would otherwise draw a car under one pixel.

- [ ] **Step 1: Write the failing test**

Save as `/tmp/playground-checks/check_view_player.mjs`. It imports the two modules by path, so it needs no DOM: `rowIndexAt`, `fitTransform`, `worldToCanvas`, `readoutText` and `describeRun` are pure. It checks both a completed and a failed run, and both a left and a right offset.

```js
// Usage: node check_view_player.mjs <path to the playground folder>
import assert from "node:assert/strict";
import path from "node:path";
import { pathToFileURL } from "node:url";

const load = (name) => import(pathToFileURL(path.resolve(process.argv[2], name)).href);
const { rowIndexAt } = await load("player.js");
const { describeRun, fitTransform, readoutText, worldToCanvas } = await load("view.js");

const rows = Array.from({ length: 11 }, (_, i) => ({ t: i * 0.05 }));
assert.equal(rowIndexAt(rows, 0, 0), 0);
assert.equal(rowIndexAt(rows, 0, 0.26), 5);
assert.equal(rowIndexAt(rows, 10, 0.12), 2, "moves backwards too");
assert.equal(rowIndexAt(rows, 3, 99), 10, "clamps at the last row");
assert.equal(rowIndexAt([], 0, 1), 0);
assert.equal(rowIndexAt([{ t: 0 }], 5, 1), 0, "a stale k is clamped");

const road = { lines: { tarmac_left: [[0, 0], [100, 0]], tarmac_right: [[0, -10], [100, -10]] } };
const t = fitTransform(road, 200, 100, 10);
assert.ok(Math.abs(t.scale - 1.8) < 1e-9, "100 m wide in 180 px");
const [x0, y0] = worldToCanvas(t, 0, 0);
const [x1, y1] = worldToCanvas(t, 100, -10);
assert.ok(Math.abs(x0 - 10) < 1e-9 && Math.abs(x1 - 190) < 1e-9, "x spans the padded width");
assert.ok(y0 < y1, "world y runs up the screen");
assert.ok(Math.abs((y0 + y1) / 2 - 50) < 1e-9, "centred vertically");

assert.deepEqual(readoutText(undefined), { t: "-", lat: "-", speed: "-", steer: "-", progress: "-" });
assert.deepEqual(readoutText({ t: 5, lat: -0.004, speed: 8, steer: 0.1234, progress_m: 40.4 }),
  { t: "5.0 s", lat: "+0.00 m", speed: "8.0 m/s", steer: "+0.123 rad", progress: "40 m" });
assert.equal(readoutText({ t: 1, lat: -0.31, speed: 8, steer: -0.2, progress_m: 8 }).lat, "-0.31 m");

assert.equal(describeRun({ completed: true, mean_abs_lat: 0.381 }, "gentle", 88.9), "Completed gentle in 88.9 s, mean |lat| 0.38 m");
assert.equal(describeRun({ completed: false, off_road: true, progress_m: 120.4, mean_abs_lat: 1.234 }, "curvy", 15),
  "Left the road on curvy after 15.0 s and 120 m, mean |lat| 1.23 m");
assert.equal(describeRun({ completed: false, off_road: false, progress_m: 5, mean_abs_lat: 0 }, "gentle", 120),
  "Ran out of time on gentle after 120.0 s and 5 m, mean |lat| 0.00 m");
console.log("check_view_player.mjs: all assertions passed");
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node /tmp/playground-checks/check_view_player.mjs playground`
Expected: FAIL with `Error [ERR_MODULE_NOT_FOUND]: Cannot find module '.../playground/player.js'`.

- [ ] **Step 3: Write the minimal implementation**

Create `playground/view.js`. Colours come only from the `palette` argument.

```js
// Pure presentation, shared by the playground and the lecture sliders: drawing functions take a 2D context
// plus data, text functions take numbers; none keeps state. Colours come only from the palette argument.

const CAR_LENGTH_M = 4.5;
const CAR_WIDTH_M = 1.8;
const MIN_CAR_PX = 12; // a 700 m loop on a phone would draw a true-size car under one pixel; the symbol never goes below this
const LINE_PX = 2;
const STRIPS = [
  { label: "e_off", unit: "m", truth: "lat", estimate: "est_offset", color: "e_off", floor: 0.5 },
  { label: "ψ", unit: "rad", truth: "head_err", estimate: "est_heading", color: "psi", floor: 0.1 },
  { label: "δ", unit: "rad", truth: "steer", color: "delta", floor: 0.1 },
];
const LABEL_MARGIN_PX = 76; // room right of the strip lines for the quantity names
const LABEL_GAP_PX = 12;
const FONT = '10px "IBM Plex Mono", ui-monospace, monospace';

/** Size a canvas's backing store to its CSS box at the device pixel ratio; drawing then uses CSS pixels. */
export function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

/** Scale and offsets that fit the whole tarmac into width x height pixels, centred, world y up. */
export function fitTransform(road, width, height, padding = 24) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x, y] of [...road.lines.tarmac_left, ...road.lines.tarmac_right]) {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  const scale = Math.min((width - 2 * padding) / (maxX - minX), (height - 2 * padding) / (maxY - minY));
  return { scale, ox: (width - scale * (minX + maxX)) / 2, oy: (height + scale * (minY + maxY)) / 2, width, height };
}

export const worldToCanvas = (t, x, y) => [t.ox + t.scale * x, t.oy - t.scale * y];

function tracePath(ctx, t, points, close) {
  points.forEach(([x, y], i) => {
    const [px, py] = worldToCanvas(t, x, y);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  if (close) ctx.closePath();
}

function strokeLine(ctx, t, points, road, style) {
  ctx.beginPath();
  tracePath(ctx, t, points, road.closed);
  Object.assign(ctx, { strokeStyle: style.color, lineWidth: style.width, globalAlpha: style.alpha ?? 1 });
  ctx.setLineDash(style.dash ?? []);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;
}

function fillTarmac(ctx, t, road, colour) {
  const { tarmac_left: left, tarmac_right: right } = road.lines;
  ctx.beginPath();
  if (road.closed) {
    tracePath(ctx, t, left, true); // two loops: the fill between them is the road
    tracePath(ctx, t, right, true);
  } else {
    tracePath(ctx, t, [...left, ...[...right].reverse()], true);
  }
  ctx.fillStyle = colour;
  ctx.fill("evenodd");
}

function nearestPoint(points, x, y) {
  let best = points[0], bestD = Infinity;
  for (const p of points) {
    const d = (p[0] - x) ** 2 + (p[1] - y) ** 2;
    if (d < bestD) { bestD = d; best = p; }
  }
  return best;
}

function drawCar(ctx, t, row, fill, outline) {
  const [px, py] = worldToCanvas(t, row.x, row.y);
  const grow = Math.max(1, MIN_CAR_PX / (CAR_LENGTH_M * t.scale));
  const length = CAR_LENGTH_M * t.scale * grow;
  const width = CAR_WIDTH_M * t.scale * grow;
  ctx.save();
  ctx.translate(px, py);
  ctx.rotate(-row.yaw); // world angles run counter-clockwise, canvas rotation clockwise
  ctx.beginPath();
  ctx.rect(-length / 2, -width / 2, length, width);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = LINE_PX;
  ctx.strokeStyle = outline;
  ctx.stroke();
  ctx.beginPath(); // the nose: which way the car points
  ctx.moveTo(0, 0);
  ctx.lineTo(length / 2, 0);
  ctx.stroke();
  ctx.restore();
}

/** Top-down picture of step `k`: road, e_off brace, the path driven so far, the car. `t` comes from fitTransform. */
export function draw(ctx, road, rows, k, palette, t) {
  const { quantity, scene, chrome } = palette;
  const { lines } = road;
  ctx.fillStyle = chrome.panel;
  ctx.fillRect(0, 0, t.width, t.height);
  fillTarmac(ctx, t, road, scene.road);
  strokeLine(ctx, t, lines.ego_center, road, { color: chrome.text, width: 1, alpha: 0.25 });
  strokeLine(ctx, t, lines.white, road, { color: scene.white, width: LINE_PX });
  const dashOn = road.dash_period * road.dash_fraction * t.scale;
  const dashOff = road.dash_period * (1 - road.dash_fraction) * t.scale;
  strokeLine(ctx, t, lines.yellow, road, { color: scene.yellow, width: LINE_PX, dash: [dashOn, dashOff] });
  const row = rows[k];
  if (!row) return;
  strokeLine(ctx, t, rows.slice(0, k + 1).map((r) => [r.x, r.y]), { closed: false }, { color: quantity.e_off, width: LINE_PX });
  const [bx, by] = nearestPoint(lines.ego_center, row.x, row.y); // the e_off brace: car to the lane centre
  strokeLine(ctx, t, [[row.x, row.y], [bx, by]], { closed: false }, { color: quantity.e_off, width: LINE_PX });
  drawCar(ctx, t, row, chrome.background, row.off_road ? quantity.perception : chrome.text);
}

function plotSeries(ctx, rows, xOf, yOf, key, usable) {
  let pen = false, last = null;
  ctx.beginPath();
  for (const row of rows) {
    const v = row[key];
    if (!usable(row) || !Number.isFinite(v)) { pen = false; continue; } // a gap, never an invented value
    last = [xOf(row), yOf(v)];
    if (pen) ctx.lineTo(...last); else ctx.moveTo(...last);
    pen = true;
  }
  ctx.stroke();
  return last;
}

/** Three strip charts of the last `windowS` seconds ending at step `k`; each quantity's name sits at its line's end. */
export function drawStrips(ctx, rows, k, palette, windowS = 10) {
  const { clientWidth: width, clientHeight: height } = ctx.canvas;
  const { quantity, chrome } = palette;
  ctx.fillStyle = chrome.panel;
  ctx.fillRect(0, 0, width, height);
  if (!rows[k]) return;
  const start = rows[k].t - windowS;
  let first = k;
  while (first > 0 && rows[first - 1].t >= start) first--;
  const visible = rows.slice(first, k + 1);
  const band = height / STRIPS.length;
  const plotWidth = width - LABEL_MARGIN_PX;
  const xOf = (row) => ((row.t - start) / windowS) * plotWidth;
  ctx.font = FONT;
  ctx.lineWidth = LINE_PX;
  STRIPS.forEach((strip, i) => {
    const mid = band * (i + 0.5);
    const estimateOk = (row) => strip.estimate !== undefined && row.est_valid;
    const spread = visible.flatMap((r) => [r[strip.truth], estimateOk(r) ? r[strip.estimate] : NaN]).filter(Number.isFinite);
    const range = Math.max(strip.floor, ...spread.map(Math.abs)) * 1.1;
    const yOf = (v) => mid - (v / range) * (band / 2 - 6);
    strokeLine(ctx, { ox: 0, oy: 0, scale: 1 }, [[0, yOf(0)], [plotWidth, yOf(0)]], { closed: false }, { color: chrome.border, width: 1 });
    ctx.fillStyle = chrome.text;
    ctx.globalAlpha = 0.6;
    ctx.fillText(`±${range.toFixed(2)} ${strip.unit}`, 4, band * i + 11);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = quantity[strip.color];
    const end = plotSeries(ctx, visible, xOf, yOf, strip.truth, () => true);
    const labels = [{ text: strip.label, end }];
    if (strip.estimate !== undefined) {
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = quantity.perception;
      labels.push({ text: `${strip.label} est.`, end: plotSeries(ctx, visible, xOf, yOf, strip.estimate, estimateOk) });
      ctx.setLineDash([]);
    }
    let lastY = -Infinity;
    for (const label of labels.filter((l) => l.end).sort((a, b) => a.end[1] - b.end[1])) {
      const y = Math.max(label.end[1], lastY + LABEL_GAP_PX); // two names that would overlap stack instead
      ctx.fillStyle = label.text.endsWith("est.") ? quantity.perception : quantity[strip.color];
      ctx.fillText(label.text, plotWidth + 6, y + 3);
      lastY = y;
    }
  });
}

/** `+0.12`: fixed digits with an explicit sign, so left and right offsets read at a glance. */
function signed(value, digits) {
  const rounded = Number(value.toFixed(digits));
  return (rounded < 0 ? "" : "+") + rounded.toFixed(digits);
}

/** The five live readouts for one step; a dash when there is no step yet. */
export function readoutText(row) {
  if (!row) return { t: "-", lat: "-", speed: "-", steer: "-", progress: "-" };
  return {
    t: `${row.t.toFixed(1)} s`,
    lat: `${signed(row.lat, 2)} m`,
    speed: `${row.speed.toFixed(1)} m/s`,
    steer: `${signed(row.steer, 3)} rad`,
    progress: `${row.progress_m.toFixed(0)} m`,
  };
}

/** One sentence for a finished drive. `metrics` is the shim's summary(); `seconds` is the last step's t. */
export function describeRun(metrics, road, seconds) {
  const lat = `mean |lat| ${metrics.mean_abs_lat.toFixed(2)} m`;
  if (metrics.completed) return `Completed ${road} in ${seconds.toFixed(1)} s, ${lat}`;
  const how = metrics.off_road ? "Left the road on" : "Ran out of time on";
  return `${how} ${road} after ${seconds.toFixed(1)} s and ${metrics.progress_m.toFixed(0)} m, ${lat}`;
}
```

Create `playground/player.js`. The player owns the clock and the scrub position. It replays rows by their `t`, so a slow device shows the same drive later, not a different drive.

```js
// The playback clock. The worker produces rows at full speed; the player replays them by row.t, so
// scrubbing, speed and pause never touch the worker. It owns the rows and calls view.js to draw them.
import { draw, fitCanvas, fitTransform } from "./view.js";

const MAX_FRAME_S = 0.25; // a backgrounded tab must not fast-forward the lap when it wakes

/** Index of the newest row shown at `clock` seconds, searching from `k` (rows are in time order). */
export function rowIndexAt(rows, k, clock) {
  let i = Math.max(0, Math.min(k, rows.length - 1));
  while (i + 1 < rows.length && rows[i + 1].t <= clock) i++;
  while (i > 0 && rows[i].t > clock) i--;
  return i;
}

export class Player {
  #canvas;
  #palette;
  #road;
  #transform;
  #ctx;
  #rows = [];
  #k = 0;
  #clock = 0;
  #speed = 1;
  #playing = false;
  #finished = true; // no more rows are coming
  #dirty = true;
  #frame = 0;
  #last = null;
  #ticks = [];

  constructor(canvas, road, palette) {
    this.#canvas = canvas;
    this.#palette = palette;
    this.reset(road);
    this.#finished = true;
  }

  get k() { return this.#k; }
  get length() { return this.#rows.length; }
  get rows() { return this.#rows; }
  get playing() { return this.#playing; }

  onTick(callback) { this.#ticks.push(callback); }

  /** Start a new drive on `road`: rows cleared, paused, more rows expected until finish(). */
  reset(road) {
    this.#road = road;
    this.#rows = [];
    this.#k = 0;
    this.#clock = 0;
    this.#playing = false;
    this.#finished = false;
    this.fit();
  }

  /** Replace all rows with a finished drive. */
  setRows(rows) {
    this.#rows = rows;
    this.#finished = true;
    this.seek(0);
  }

  /** Live: add the newest step; a playing player keeps following it. */
  append(row) {
    this.#rows.push(row);
    this.#dirty = true;
    this.#kick();
  }

  /** The producer is done: playback stops at the last row instead of waiting for more. */
  finish() {
    this.#finished = true;
    this.#kick();
  }

  /** Re-measure the canvas (window or layout resize) and redraw. */
  fit() {
    const { ctx, width, height } = fitCanvas(this.#canvas);
    this.#ctx = ctx;
    this.#transform = fitTransform(this.#road, width, height);
    this.#dirty = true;
    this.#kick();
  }

  play() {
    if (this.#finished && this.#k >= this.#rows.length - 1) this.seek(0); // finished lap: play it again
    this.#playing = true;
    this.#last = null;
    this.#dirty = true; // listeners learn that playback started
    this.#kick();
  }

  pause() {
    this.#playing = false;
    this.#dirty = true;
    this.#kick();
  }

  toggle() { this.#playing ? this.pause() : this.play(); }

  seek(k) {
    this.#k = Math.max(0, Math.min(Math.trunc(k), this.#rows.length - 1));
    this.#clock = this.#rows[this.#k]?.t ?? 0;
    this.#dirty = true;
    this.#kick();
  }

  setSpeed(multiplier) { this.#speed = multiplier; }

  #kick() {
    if (!this.#frame) this.#frame = requestAnimationFrame(this.#step);
  }

  #step = (now) => {
    this.#frame = 0;
    if (this.#playing) {
      const dt = this.#last === null ? 0 : Math.min((now - this.#last) / 1000, MAX_FRAME_S);
      this.#last = now;
      const newest = this.#rows.at(-1)?.t ?? 0;
      this.#clock = Math.min(this.#clock + dt * this.#speed, newest);
      const k = rowIndexAt(this.#rows, this.#k, this.#clock);
      this.#dirty ||= k !== this.#k;
      this.#k = k;
      if (this.#finished && k >= this.#rows.length - 1) this.#playing = false; // reached the end
    }
    if (this.#dirty) this.#render();
    if (this.#playing) this.#kick();
  };

  #render() {
    this.#dirty = false;
    draw(this.#ctx, this.#road, this.#rows, this.#k, this.#palette, this.#transform);
    const tick = { k: this.#k, row: this.#rows[this.#k], playing: this.#playing };
    this.#ticks.forEach((callback) => callback(tick));
  }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `node /tmp/playground-checks/check_view_player.mjs playground`
Expected: `check_view_player.mjs: all assertions passed`

The picture is checked by eye and by browser in Task 11, Steps 2 and 4.

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests -q`
Expected: all green; 412 passed, the same as after Task 8 (this task adds no Python tests).

- [ ] **Step 6: Commit**

```bash
git add playground/view.js playground/player.js
git commit -m "feat(playground): canvas drawing and the replay clock

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: The page, its editor and the progress store (`index.html`, `app.css`, `app.js`, `progress.js`)

**Files:**
- Create: `playground/progress.js`
- Create: `playground/index.html`
- Create: `playground/app.css`
- Create: `playground/app.js`
- Create (generated, committed): `course/dist/playground/` (rebuilt here; Tasks 9 and 10 leave it untracked)
- Test: `/tmp/playground-checks/check_page.py` (19 numbered cases, 20 runs because case 13 runs once per missing feature; uses `solutions.py` from Task 9) and `/tmp/playground-checks/check_progress.mjs`

**Interfaces:**
- Consumes:
  - Task 9 worker messages (the tables under Task 9) and Task 10 exports (`Player`, `describeRun`, `drawStrips`, `fitCanvas`, `readoutText`).
  - `exercises.json` records: `id`, `unit_key` (the slug, `"0-1-4"`), `label`, `name`, `kind`, `road`, `presets`, `colab_url`; an optional one-sentence `task`; `car_box` is ignored by this page. `roads.json` is keyed by road name; `palette.json` is Task 4's `palette_dict()`.
  - The site's progress store, `src/lib/learnfsd-progress.ts` in the vivekn repo: `STORAGE_KEY` at `:7`, `parseProgress` at `:22`, `logs: PilotLog[]` at `:13`.
- Produces (Task 12 and the site plan rely on these):
  - The page URL `playground/index.html?exercise=<id>[&preset=<name>]`. Self-links use `index.html`; module imports are relative.
  - DOM ids: `#exercise-label`, `#exercise-task`, `#unit-link` (`../unit/<unit_key>/`, relative: the lecture sits beside the playground wherever the course is served, spec section 12), `#open-colab`, `#editor` (`#editor-fallback` is the textarea when CodeMirror does not load), `#run`, `#check`, `#stop`, `#road-label`, `#road`, `#status` (`data-state` is `loading | ready | running | done | error`), `#retry`, `#colab-fallback`, `#traceback`, `#verdict` (`data-passed`), `#stdout`, `#stats` (a `<dl>`), `#drive-view`, `#player-view`, `#canvas`, `#play`, `#to-end`, `#scrub`, `#speed`, `#r-t`, `#r-lat`, `#r-speed`, `#r-steer`, `#r-progress`, `#strips`, `#frame` (320 x 180), `#mask` (320 x 180), `#dashboard-view`, `#dashboard`, `#dashboard-img`, `#engine`.
  - `window.__setCode(text)`: replaces the code, whether CodeMirror or the textarea is active.
  - `progress.js`: `STORAGE_KEY = "learnfsd.v2.progress"`, `readProgress() -> object | null` (never throws), `markPassed(unitKey, exerciseId, now = new Date()) -> boolean` (true when written, false when storage is unavailable; the first pass keeps its timestamp; other units and unknown keys are kept).
  - What B6 asserts for this task: all 19 numbered cases (20 runs) of `check_page.py`. Task 12's pytest cases drive the same page through `window.__setCode`.

> CARRY (site plan): `#unit-link` is `../unit/<slug>/`, so the site must serve the playground at `<prefix>/playground/` (the iframe's `src` is `<prefix>/playground/index.html?exercise=<id>`), and then `../unit/<slug>/` resolves to `<prefix>/unit/<slug>/`. Served from any other depth the link breaks, so the site plan's iframe task must use that path.

> NOTE: an unknown `?exercise=` does not show an error page. This page opens the FIRST exercise in state `ready` and writes "Unknown exercise <id>; showing <label>." in `#status`, as Task 12 pins. An unknown `preset=` falls back to `starter` with "Unknown preset <name>; showing the starter."
> 
> This block was rewritten at assembly and has not yet run in a browser; the executor's Task 12 run is its first test.

> NOTE: the `app.js` below defines `window.__setCode` (`window.__setCode = editor.set` in `setUp`) and writes a run error's learner-facing line to `#stdout` as well as `#status` (the `error` entry of `FROM_WORKER`). Task 12's browser tests rely on both and make no `app.js` edit.

> NOTE: load errors (`request == "init"`, a missing JSON file, a bad kind, a missing browser feature) write to `#status` only and show Retry (except a missing feature, which cannot be retried) and the Colab link. Run errors write the same learner-facing line to both `#status` and `#stdout`, with the traceback in `#traceback`. A message the page cannot use (for example an untagged row) is a load failure too: the page terminates the worker and shows Retry, never a spinner.

> NOTE: exercise kinds are a table in `app.js` (`KINDS`), and the visible page groups come from it. `drive` shows the road, the player and the dashboard; `stats` shows only the editor, the buttons and `#stats`; `image` shows the camera frame and `#mask`. The page dispatches on `exercise.kind`, never on an id. An unknown kind is a named load error.

> NOTE: `check()` prints its verdict with ANSI colour codes, and they arrive in `stdout` messages. The page strips them. `--ok` and `--error` in `app.css` are status colours that are not in the palette, so they stay CSS-only.

> NOTE: the wrong-return-type text for drive exercises, "must return one number", is produced in Python (Task 7 rewords `car.py:40`, and its `_at_boundary` prefixes the exercise id). The page shows the error verbatim and never rewords it.

> NOTE (image colours): Task 7's `preview_image` returns the overlay in the scene's yellow and white, and the page draws exactly what it receives. Task 12's `MASK_PIXELS_JS` looks for `palette.json` `scene.yellow` and `scene.white` in `#mask`, which holds the overlay alone.

> NOTE: the camera-frame request is throttled to one per 100 ms and is sent only when the worker is idle, because the worker is single-threaded. The frame size comes from the Python `Camera`.

> NOTE: `progress.js` writes `logs: []` for a new unit, because the site's `addPilotLog` needs it, and keeps the first `passedAt`. Task 3 of the site plan adds `reload()` and `watchStorage` on the site side.

> NOTE: the editor needs sixteen packages: six that `app.js` imports and ten that they import in turn. CodeMirror 6 breaks if two copies of `@codemirror/state` load, and jsDelivr's `+esm` bundles each pin a different sibling version (they fail with "multiple instances of @codemirror/state"). The page therefore maps every package to its raw `dist` file with an import map. If the import fails for any reason, the page falls back to a textarea and logs a `console.warn`, with no error state.

> NOTE: 0.1.4.b answers with a `Car` (`my_car`), not a function, and its Check drives the gate road. Case 18 covers that path, and the worker check covers a function check that streams nothing.

CodeMirror pins. Every URL was fetched with `curl -sI` on 2026-10-11 and returned the status line shown. The page was also rendered in headless Chromium with them: python tokens were highlighted, typing worked, and `EditorView.findFromDOM` found the view.

| Package | Pinned version | How it is used | File | `curl -sI` result |
|---|---|---|---|---|
| `@codemirror/view` | 6.43.14 | imported by `app.js` | `dist/index.js` | `HTTP/2 200` |
| `@codemirror/state` | 6.7.6 | imported by `app.js` | `dist/index.js` | `HTTP/2 200` |
| `@codemirror/lang-python` | 6.2.1 | imported by `app.js` | `dist/index.js` | `HTTP/2 200` |
| `@codemirror/commands` | 6.11.1 | imported by `app.js` | `dist/index.js` | `HTTP/2 200` |
| `@codemirror/language` | 6.13.1 | imported by `app.js` | `dist/index.js` | `HTTP/2 200` |
| `@codemirror/theme-one-dark` | 6.1.3 | imported by `app.js` | `dist/index.js` | `HTTP/2 200` |
| `@codemirror/autocomplete` | 6.20.3 | needed by another package | `dist/index.js` | `HTTP/2 200` |
| `@codemirror/streamparser` | 6.0.0 | needed by another package | `dist/index.js` | `HTTP/2 200` |
| `@lezer/common` | 1.5.3 | needed by another package | `dist/index.js` | `HTTP/2 200` |
| `@lezer/highlight` | 1.2.5 | needed by another package | `dist/index.js` | `HTTP/2 200` |
| `@lezer/lr` | 1.4.11 | needed by another package | `dist/index.js` | `HTTP/2 200` |
| `@lezer/python` | 1.1.19 | needed by another package | `dist/index.js` | `HTTP/2 200` |
| `crelt` | 1.0.7 | needed by another package | `index.js` | `HTTP/2 200` |
| `style-mod` | 4.1.4 | needed by another package | `src/style-mod.js` | `HTTP/2 200` |
| `w3c-keyname` | 2.2.8 | needed by another package | `index.js` | `HTTP/2 200` |
| `@marijn/find-cluster-break` | 1.0.4 | needed by another package | `src/index.js` | `HTTP/2 200` |

- [ ] **Step 1: Write the failing test**

Save as `/tmp/playground-checks/check_progress.mjs`. It stubs `localStorage`, so it runs in node. It covers a first pass, a second exercise in the same unit, a second unit, unknown keys, malformed storage, and a store that refuses writes or is absent.

```js
// Usage: node check_progress.mjs <path to the playground folder>
import assert from "node:assert/strict";
import path from "node:path";
import { pathToFileURL } from "node:url";

const { STORAGE_KEY, markPassed, readProgress } = await import(pathToFileURL(path.resolve(process.argv[2], "progress.js")).href);

// progress.js against a stubbed localStorage (the module reads the global at call time)
const store = new Map();
globalThis.localStorage = { getItem: (k) => store.get(k) ?? null, setItem: (k, v) => store.set(k, v) };
assert.equal(readProgress(), null);
assert.equal(markPassed("0-1-4", "0.1.4.a", new Date("2026-10-11T00:00:00Z")), true);
let saved = JSON.parse(store.get(STORAGE_KEY));
assert.deepEqual(saved, { version: 1, units: { "0-1-4": { logs: [], exercises: { "0.1.4.a": { passedAt: "2026-10-11T00:00:00.000Z", via: "playground" } } } }, deck: {} });
markPassed("0-1-4", "0.1.4.b", new Date("2026-10-12T00:00:00Z"));
markPassed("0-1-4", "0.1.4.a", new Date("2026-10-13T00:00:00Z"));
saved = JSON.parse(store.get(STORAGE_KEY));
assert.equal(saved.units["0-1-4"].exercises["0.1.4.a"].passedAt, "2026-10-11T00:00:00.000Z", "first pass keeps its timestamp");
assert.equal(saved.units["0-1-4"].exercises["0.1.4.b"].passedAt, "2026-10-12T00:00:00.000Z", "a second exercise in the same unit");
store.set(STORAGE_KEY, JSON.stringify({ version: 1, units: { "0-1-1": { logs: [{ at: "x" }], quiz: { best: 1, passedAt: "p" } } }, deck: { "0-1-1:q1": { box: 2 } }, extra: 7 }));
markPassed("0-1-3", "0.1.3.a");
saved = JSON.parse(store.get(STORAGE_KEY));
assert.deepEqual(saved.units["0-1-1"], { logs: [{ at: "x" }], quiz: { best: 1, passedAt: "p" } }, "other units untouched");
assert.deepEqual(saved.deck, { "0-1-1:q1": { box: 2 } });
assert.equal(saved.extra, 7, "unknown keys survive");
markPassed("0-1-1", "0.1.1.a");
assert.equal(JSON.parse(store.get(STORAGE_KEY)).units["0-1-1"].quiz.best, 1, "a unit's quiz survives a pass");
store.set(STORAGE_KEY, "{not json");
assert.equal(readProgress(), null);
assert.equal(markPassed("0-1-4", "0.1.4.a"), true, "unreadable storage reads as empty, like the site");
globalThis.localStorage = { getItem: () => null, setItem: () => { throw new Error("quota"); } };
assert.equal(markPassed("0-1-4", "0.1.4.a"), false, "a refusing store reports false");
delete globalThis.localStorage;
assert.equal(markPassed("0-1-4", "0.1.4.a"), false, "no storage at all reports false");
console.log("check_progress.mjs: all assertions passed");
```

Save as `/tmp/playground-checks/check_page.py`. Run `python -u check_page.py http://localhost:8765/ 3 12` to run only cases 3 and 12. Each case opens a fresh browser context, so each one downloads Pyodide again from the CDN.

```python
# Throwaway page-level check for Tasks 10 and 11; Task 12 turns each case into a pytest test.
import json
import re
import sys
import time

from playwright.sync_api import sync_playwright
from solutions import SOLUTIONS

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765/"  # the playground folder, ending in /
FAIL = []


def case(name):
    def wrap(fn):
        def run(browser):
            context = browser.new_context(viewport={"width": 1280, "height": 900})
            page = context.new_page()
            page.on("pageerror", lambda e: FAIL.append(f"{name}: pageerror {e}"))
            started = time.time()
            try:
                fn(page, context)
                print(f"PASS {name} ({time.time() - started:.1f}s)")
            except Exception as e:  # noqa: BLE001
                FAIL.append(f"{name}: {type(e).__name__}: {str(e)[:400]}")
                print(f"FAIL {name}: {str(e)[:300]}")
            finally:
                context.close()
        run.case_name = name
        CASES.append(run)
        return fn
    return wrap


CASES = []


def state(page, want, timeout=90_000):
    page.wait_for_function("w => document.getElementById('status').dataset.state === w", arg=want, timeout=timeout)


def open_playground(page, query="exercise=0.1.4.a"):
    page.goto(f"{URL}index.html?{query}")
    state(page, "ready", 120_000)


def set_code(page, code):
    page.evaluate("code => window.__setCode(code)", code)  # the page's own test hook; works for CodeMirror and the textarea


def canvas_shows(page, selector, key):
    """True when the canvas holds exactly the RGB bytes the worker sent under `key` of its last `image` message."""
    return page.evaluate("""([sel, key]) => {
      const c = document.querySelector(sel); const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
      const rgb = window.__image[key];
      if ((d.length / 4) * 3 !== rgb.length) return false;
      for (let i = 0, j = 0; j < rgb.length; i += 4, j += 3) if (d[i] !== rgb[j] || d[i + 1] !== rgb[j + 1] || d[i + 2] !== rgb[j + 2]) return false;
      return true;
    }""", [selector, key])


WATCH_IMAGES = """
const Native = window.Worker;
window.Worker = class extends Native {
  constructor(...args) {
    super(...args);
    this.addEventListener("message", ({ data }) => {
      if (data.type === "image") window.__image = { frame: new Uint8Array(data.frame.slice(0)), overlay: new Uint8Array(data.overlay.slice(0)) };
    });
  }
};
"""


def canvas_has_non_background(page, selector="#canvas"):
    return page.evaluate("""sel => {
      const c = document.querySelector(sel); const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
      const seen = new Set();
      for (let i = 0; i < d.length; i += 4) seen.add((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]);
      return seen.size;
    }""", selector)


@case("1 ready, label has no dotted id, editor is CodeMirror")
def _(page, ctx):
    open_playground(page)
    label = page.inner_text("#exercise-label")
    assert label and not re.search(r"\d\.\d", label), label
    assert page.locator(".cm-editor").count() == 1 and page.locator("#editor-fallback").count() == 0
    assert page.locator(".cm-content span[class]").count() > 0, "python tokens are highlighted"
    assert page.inner_text("#engine").startswith("Pyodide 314.0.7"), page.inner_text("#engine")
    task = page.evaluate("fetch('exercises.json').then((r) => r.json()).then((rs) => rs.find((r) => r.id === '0.1.4.a').task ?? '')")
    assert page.locator("#exercise-task").is_visible() == bool(task), "the task line shows only when the record has one"
    assert page.locator("#retry").is_hidden()
    assert page.locator("#player-view").is_visible() and page.locator("#mask").is_hidden(), "a drive shows the player, not the masks"
    set_code(page, "marker_line = 1\n")
    assert "marker_line = 1" in page.inner_text(".cm-content"), "__setCode replaces the CodeMirror document"


@case("2 run reaches done with progress, picture and strips drawn")
def _(page, ctx):
    open_playground(page)
    set_code(page, SOLUTIONS["0.1.4.a"])
    page.click("#run")
    state(page, "done", 120_000)
    assert "Completed gentle" in page.inner_text("#status"), page.inner_text("#status")
    assert page.inner_text("#r-progress") not in ("-", "0 m"), page.inner_text("#r-progress")
    assert canvas_has_non_background(page) > 5
    assert canvas_has_non_background(page, "#strips") > 3
    page.wait_for_function("document.getElementById('frame').getContext('2d').getImageData(0,0,1,1).data[3] === 255", timeout=15_000)


@case("3 check passes and saves progress; second pass keeps first timestamp")
def _(page, ctx):
    open_playground(page)
    set_code(page, SOLUTIONS["0.1.4.a"])
    page.click("#check")
    page.wait_for_selector('#verdict[data-passed="true"]', timeout=120_000)
    stored = json.loads(page.evaluate("localStorage.getItem('learnfsd.v2.progress')"))
    entry = stored["units"][json.loads(page.evaluate("fetch('exercises.json').then(r => r.json()).then(rs => JSON.stringify(rs.find(r => r.id === '0.1.4.a').unit_key))"))]
    assert stored["version"] == 1 and entry["logs"] == [] and entry["exercises"]["0.1.4.a"]["via"] == "playground", stored
    first = entry["exercises"]["0.1.4.a"]["passedAt"]
    time.sleep(1.1)
    page.click("#check")
    page.wait_for_function("document.getElementById('status').textContent.includes('saved')", timeout=120_000)
    again = json.loads(page.evaluate("localStorage.getItem('learnfsd.v2.progress')"))
    assert [v["exercises"]["0.1.4.a"]["passedAt"] for v in again["units"].values() if "exercises" in v] == [first]
    assert page.locator("#verdict li").count() >= 1
    printed = page.inner_text("#stdout")
    assert "0.1.4.a" in printed and "\x1b" not in printed, repr(printed)


@case("4 failing body gives data-passed=false and does not save")
def _(page, ctx):
    open_playground(page)
    set_code(page, "def p_steer(est, k_off, k_head):\n    return 0.0\n")
    page.click("#check")
    page.wait_for_selector('#verdict[data-passed="false"]', timeout=120_000)
    assert page.evaluate("localStorage.getItem('learnfsd.v2.progress')") is None


@case("5 stop kills an infinite loop within 5 s, then a run works")
def _(page, ctx):
    open_playground(page)
    set_code(page, "def p_steer(est, k_off, k_head):\n    while True:\n        pass\n")
    page.click("#run")
    state(page, "running", 10_000)
    time.sleep(1.5)
    t0 = time.time()
    page.click("#stop")
    assert "Stopped." in page.inner_text("#status"), page.inner_text("#status")
    state(page, "ready", 60_000)
    print(f"    stop to ready: {time.time() - t0:.1f}s")
    assert time.time() - t0 < 5, "ready within 5 s"
    set_code(page, SOLUTIONS["0.1.4.a"])
    page.click("#run")
    state(page, "done", 120_000)


@case("6 ZeroDivisionError names the learner line")
def _(page, ctx):
    open_playground(page)
    set_code(page, "def p_steer(est, k_off, k_head):\n    return 1 / 0\n")
    page.click("#run")
    state(page, "error", 60_000)
    text = page.inner_text("#status")
    assert "ZeroDivisionError" in text and "line 2" in text, text
    assert text in page.inner_text("#stdout"), "the same line is in the printed output"
    assert page.locator("#traceback").is_visible()
    assert page.locator("#run").is_enabled(), "the learner can fix the code and run again"


@case("7 CodeMirror aborted -> textarea fallback, run still works")
def _(page, ctx):
    page.route("**/@codemirror/**", lambda route: route.abort())
    open_playground(page)
    assert page.locator("#editor-fallback").is_visible()
    set_code(page, SOLUTIONS["0.1.4.a"])
    assert page.input_value("#editor-fallback") == SOLUTIONS["0.1.4.a"]
    page.click("#run")
    state(page, "done", 120_000)


@case("8 unknown exercise and unknown preset show a notice, state stays ready")
def _(page, ctx):
    open_playground(page, "exercise=9.9.9.z")
    text = page.inner_text("#status")
    assert "Unknown exercise 9.9.9.z" in text and text.endswith("Ready."), text
    assert page.inner_text("#exercise-label")
    page.goto(f"{URL}index.html?exercise=0.1.4.a&preset=toString")
    state(page, "ready", 120_000)
    assert "Unknown preset toString" in page.inner_text("#status")


@case("9 phone width has no horizontal scroll, drive, stats and image exercises")
def _(page, ctx):
    page.set_viewport_size({"width": 375, "height": 700})
    for query in ("exercise=0.1.4.a", "exercise=0.1.1.a", "exercise=0.1.2.a"):
        open_playground(page, query)
        assert page.evaluate("document.documentElement.scrollWidth") <= 375, query
    open_playground(page, "exercise=0.1.4.a")
    set_code(page, SOLUTIONS["0.1.4.a"])
    page.click("#run")
    state(page, "done", 120_000)
    assert page.evaluate("document.documentElement.scrollWidth") <= 375, "after a run"


@case("10 stats exercise shows #stats values and hides the drive view")
def _(page, ctx):
    open_playground(page, "exercise=0.1.1.a")
    set_code(page, SOLUTIONS["0.1.1.a"])
    page.click("#run")
    state(page, "done", 120_000)
    assert page.locator("#stats dt").count() >= 1 and page.locator("#stats dt").count() == page.locator("#stats dd").count()
    assert page.locator("#stats").is_visible() and page.locator("#drive-view").is_hidden()
    assert page.locator("#road-label").is_hidden()


@case("11 preset= runs once after ready; a second exercise too")
def _(page, ctx):
    for exercise in ("0.1.4.a", "0.1.4.b", "0.1.2.a"):
        page.goto(f"{URL}index.html?exercise={exercise}&preset=starter")
        page.wait_for_function("['done', 'error'].includes(document.getElementById('status').dataset.state)", timeout=120_000)
        assert page.inner_text("#r-t") != "-" or page.get_attribute("#status", "data-state") == "error", exercise
    assert "NotImplementedError" in page.inner_text("#status") and "line" in page.inner_text("#status"), "the starter of an image exercise ran and failed in the learner's code"


@case("12 load failure: aborted bundle.zip names the file, Retry recovers")
def _(page, ctx):
    page.route("**/bundle.zip", lambda route: route.abort())
    page.goto(f"{URL}index.html?exercise=0.1.4.a")
    state(page, "error", 120_000)
    assert "bundle.zip" in page.inner_text("#status"), page.inner_text("#status")
    assert page.locator("#retry").is_visible() and page.locator("#colab-fallback").is_visible()
    page.unroute("**/bundle.zip")
    page.click("#retry")
    state(page, "ready", 120_000)
    assert page.locator("#retry").is_hidden()


for missing, label in (("WebAssembly", "WebAssembly"), ("Worker", "Web Workers")):  # label: the name missingFeatures() in app.js gives the page

    @case(f"13 no {missing}: one sentence naming it, Colab link, no Retry")
    def _(page, ctx, missing=missing, label=label):
        ctx.add_init_script(f"delete window.{missing}")
        page.goto(f"{URL}index.html?exercise=0.1.4.a")
        state(page, "error", 30_000)
        assert f"This browser has no {label}" in page.inner_text("#status"), page.inner_text("#status")
        assert page.locator("#colab-fallback").is_visible() and "colab" in page.get_attribute("#colab-fallback", "href")
        assert page.locator("#retry").is_hidden()


@case("14 scrub, play/pause and to-end move the car")
def _(page, ctx):
    open_playground(page)
    set_code(page, SOLUTIONS["0.1.4.a"])
    page.click("#run")
    state(page, "done", 120_000)
    page.click("#to-end")
    end = page.inner_text("#r-progress")
    page.fill("#scrub", "100")
    page.wait_for_function("document.getElementById('r-t').textContent === '5.0 s'", timeout=5_000)
    assert page.inner_text("#r-progress") != end
    for label in ("Pause", "Play"):  # the label follows the next drawn frame, so wait for it rather than read at once
        page.click("#play")
        page.wait_for_function("l => document.getElementById('play').textContent === l", arg=label, timeout=2_000)


@case("15 storage blocked: the verdict still shows and says progress was not saved")
def _(page, ctx):
    ctx.add_init_script("Storage.prototype.setItem = () => { throw new Error('blocked'); }")
    open_playground(page)
    set_code(page, SOLUTIONS["0.1.4.a"])
    page.click("#check")
    page.wait_for_selector('#verdict[data-passed="true"]', timeout=120_000)
    assert "blocked saving" in page.inner_text("#status"), page.inner_text("#status")


@case("16 chrome colours come from palette.json at runtime")
def _(page, ctx):
    chrome = {"background": "#123456", "panel": "#234567", "border": "#345678", "text": "#456789", "link": "#56789a"}

    def other_palette(route):
        palette = route.fetch().json()
        palette["chrome"] = chrome
        route.fulfill(json=palette)

    page.route("**/palette.json", other_palette)
    open_playground(page)
    seen = page.evaluate("names => names.map((n) => getComputedStyle(document.documentElement).getPropertyValue('--' + n).trim())", list(chrome))
    assert seen == list(chrome.values()), seen
    assert page.evaluate("getComputedStyle(document.body).backgroundColor") == "rgb(18, 52, 86)"


@case("17 image exercise: masks shown beside the frame, player hidden, wrong answer reported")
def _(page, ctx):
    ctx.add_init_script(WATCH_IMAGES)
    open_playground(page, "exercise=0.1.2.a")
    for hidden in ("#player-view", "#dashboard-view", "#road-label", "#canvas", "#strips", "#scrub", "#speed", "#stats"):
        assert page.locator(hidden).is_hidden(), hidden
    assert page.locator("#drive-view").is_visible() and page.locator("#frame").is_visible() and page.locator("#mask").is_visible()
    set_code(page, SOLUTIONS["0.1.2.a"])
    page.click("#run")
    state(page, "done", 120_000)
    assert page.evaluate("[document.getElementById('frame').width, document.getElementById('frame').height, document.getElementById('mask').width]") == [320, 180, 320]
    assert canvas_shows(page, "#frame", "frame") and canvas_shows(page, "#mask", "overlay"), "each canvas holds the bytes the worker sent it"
    assert page.evaluate("window.__image.frame.some((v, i) => v !== window.__image.overlay[i])"), "the two pictures differ"
    set_code(page, "def paint_masks(frame):\n    return None\n")
    page.click("#run")
    state(page, "error", 60_000)
    assert "NoneType" in page.inner_text("#status") and "NoneType" in page.inner_text("#stdout")
    set_code(page, SOLUTIONS["0.1.2.a"])
    page.click("#check")
    page.wait_for_selector('#verdict[data-passed="true"]', timeout=120_000)


@case("18 a check that drives streams the gate drive to the player")
def _(page, ctx):
    open_playground(page, "exercise=0.1.4.b")
    set_code(page, SOLUTIONS["0.1.4.b"])
    page.click("#check")
    page.wait_for_selector('#verdict[data-passed="true"]', timeout=120_000)
    assert page.inner_text("#r-t") != "-" and int(page.get_attribute("#scrub", "max")) > 100, "the graded drive reached the player"
    assert canvas_has_non_background(page) > 5


@case("19 a streamed step with no road is reported on the page, with Retry, not left running")
def _(page, ctx):
    def untag(route):
        response = route.fetch()
        source = response.text()
        assert "onRow(m.road)" in source
        route.fulfill(response=response, body=source.replace("onRow(m.road)", "onRow()"))

    page.route("**/worker.js", untag)
    open_playground(page)
    set_code(page, SOLUTIONS["0.1.4.a"])
    page.click("#run")
    state(page, "error", 60_000)
    assert 'names the road "undefined"' in page.inner_text("#status"), page.inner_text("#status")
    assert page.locator("#retry").is_visible()


if __name__ == "__main__":
    only = sys.argv[2:] if len(sys.argv) > 2 else None
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for run in CASES:
            if not only or run.case_name.split()[0] in only:
                run(browser)
        browser.close()
    print("FAILURES:" if FAIL else "ALL PASS")
    print("\n".join(FAIL))
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
node /tmp/playground-checks/check_progress.mjs playground
pkill -f "[h]ttp.server 8765" || true
python scripts/build_course_bundle.py
python -m http.server 8765 --directory course/dist/playground &
until curl -s -o /dev/null http://localhost:8765/; do :; done
(cd /tmp/playground-checks && python -u check_page.py http://localhost:8765/ 1)
```

Expected, first command: FAIL with `Error [ERR_MODULE_NOT_FOUND]: Cannot find module '.../playground/progress.js'`.
Expected, last command: `FAIL 1 ready, label has no dotted id, editor is CodeMirror: Page.wait_for_function: TypeError: Cannot read properties of null (reading 'dataset')`. The folder has no `index.html` yet, so there is no `#status`.

- [ ] **Step 3: Write the minimal implementation**

Create `playground/progress.js`:

```js
// Learner progress, shared with the course site: the same localStorage entry and shape as its
// src/lib/learnfsd-progress.ts, so a pass written here is read there (same origin).
export const STORAGE_KEY = "learnfsd.v2.progress"; // must equal the site's src/lib/learnfsd-progress.ts:7

const isRecord = (v) => typeof v === "object" && v !== null && !Array.isArray(v);

/** The stored progress object, or null when absent, malformed or unreadable. Never throws. */
export function readProgress() {
  try {
    const data = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null");
    return isRecord(data) && data.version === 1 && isRecord(data.units) && isRecord(data.deck) ? data : null;
  } catch {
    return null;
  }
}

/** Record that `exerciseId` passed in unit `unitKey`; the first pass keeps its timestamp. Returns false when storage refuses. */
export function markPassed(unitKey, exerciseId, now = new Date()) {
  try {
    const progress = readProgress() ?? { version: 1, units: {}, deck: {} };
    const unit = { logs: [], ...progress.units[unitKey] }; // `logs` is required by the site's addPilotLog
    const passed = unit.exercises?.[exerciseId] ?? { passedAt: now.toISOString(), via: "playground" };
    progress.units[unitKey] = { ...unit, exercises: { ...unit.exercises, [exerciseId]: passed } };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
    return true;
  } catch {
    return false;
  }
}
```

Create `playground/index.html`. The import map is the sixteen pins above. The worker does not use it.

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Playground</title>
  <link rel="stylesheet" href="./app.css">
  <!-- CodeMirror 6, every package pinned to an exact version and mapped to its raw dist file so that one
       copy of @codemirror/state and @codemirror/view is loaded. (jsDelivr's +esm bundles each pin a
       different sibling version and fail with "multiple instances of @codemirror/state".) The worker
       does not use this map. -->
  <script type="importmap">
  {
    "imports": {
      "@codemirror/view": "https://cdn.jsdelivr.net/npm/@codemirror/view@6.43.14/dist/index.js",
      "@codemirror/state": "https://cdn.jsdelivr.net/npm/@codemirror/state@6.7.6/dist/index.js",
      "@codemirror/lang-python": "https://cdn.jsdelivr.net/npm/@codemirror/lang-python@6.2.1/dist/index.js",
      "@codemirror/commands": "https://cdn.jsdelivr.net/npm/@codemirror/commands@6.11.1/dist/index.js",
      "@codemirror/language": "https://cdn.jsdelivr.net/npm/@codemirror/language@6.13.1/dist/index.js",
      "@codemirror/theme-one-dark": "https://cdn.jsdelivr.net/npm/@codemirror/theme-one-dark@6.1.3/dist/index.js",
      "@codemirror/autocomplete": "https://cdn.jsdelivr.net/npm/@codemirror/autocomplete@6.20.3/dist/index.js",
      "@codemirror/streamparser": "https://cdn.jsdelivr.net/npm/@codemirror/streamparser@6.0.0/dist/index.js",
      "@lezer/common": "https://cdn.jsdelivr.net/npm/@lezer/common@1.5.3/dist/index.js",
      "@lezer/highlight": "https://cdn.jsdelivr.net/npm/@lezer/highlight@1.2.5/dist/index.js",
      "@lezer/lr": "https://cdn.jsdelivr.net/npm/@lezer/lr@1.4.11/dist/index.js",
      "@lezer/python": "https://cdn.jsdelivr.net/npm/@lezer/python@1.1.19/dist/index.js",
      "crelt": "https://cdn.jsdelivr.net/npm/crelt@1.0.7/index.js",
      "style-mod": "https://cdn.jsdelivr.net/npm/style-mod@4.1.4/src/style-mod.js",
      "w3c-keyname": "https://cdn.jsdelivr.net/npm/w3c-keyname@2.2.8/index.js",
      "@marijn/find-cluster-break": "https://cdn.jsdelivr.net/npm/@marijn/find-cluster-break@1.0.4/src/index.js"
    }
  }
  </script>
  <script type="module" src="./app.js"></script>
</head>
<body>
  <header>
    <h1 id="exercise-label"></h1>
    <p id="exercise-task" hidden></p>
    <nav>
      <a id="unit-link" target="_top">Back to the lecture</a>
      <a id="open-colab" target="_blank" rel="noopener">Open in Colab</a>
    </nav>
  </header>

  <main>
    <section id="code-pane" aria-label="Your code">
      <div id="editor"></div>
      <div class="toolbar">
        <button id="run" type="button" disabled>Run</button>
        <button id="check" type="button" disabled>Check</button>
        <button id="stop" type="button" disabled>Stop</button>
        <label id="road-label">Road <select id="road"></select></label>
      </div>
      <p id="status" data-state="loading" role="status" aria-live="polite">Loading...</p>
      <div class="status-actions">
        <button id="retry" type="button" hidden>Retry</button>
        <a id="colab-fallback" target="_blank" rel="noopener" hidden>Open this exercise in Colab</a>
      </div>
      <details id="traceback" hidden><summary>Traceback</summary><pre></pre></details>
      <div id="verdict" hidden><ul></ul></div>
      <pre id="stdout" aria-label="Printed output"></pre>
      <dl id="stats" hidden></dl>
    </section>

    <section id="drive-view" aria-label="The drive">
      <div id="player-view">
        <canvas id="canvas" aria-label="Top-down view of the car on the road"></canvas>
        <div class="transport">
          <button id="play" type="button">Play</button>
          <button id="to-end" type="button">To end</button>
          <input id="scrub" type="range" min="0" max="0" value="0" step="1" aria-label="Step">
          <select id="speed" aria-label="Speed">
            <option value="0.5">0.5x</option>
            <option value="1" selected>1x</option>
            <option value="2">2x</option>
            <option value="4">4x</option>
          </select>
        </div>
        <dl class="readouts">
          <div><dt>time</dt><dd id="r-t">-</dd></div>
          <div><dt>off centre (left +)</dt><dd id="r-lat">-</dd></div>
          <div><dt>speed</dt><dd id="r-speed">-</dd></div>
          <div><dt>steering</dt><dd id="r-steer">-</dd></div>
          <div><dt>progress</dt><dd id="r-progress">-</dd></div>
        </dl>
        <canvas id="strips" aria-label="Offset, heading error and steering over the last 10 seconds"></canvas>
      </div>
      <div class="camera">
        <canvas id="frame" width="320" height="180" aria-label="What the car's camera sees"></canvas>
        <canvas id="mask" width="320" height="180" aria-label="The pixels your masks marked" hidden></canvas>
      </div>
      <div id="dashboard-view" class="dashboard">
        <button id="dashboard" type="button" disabled>Dashboard</button>
        <img id="dashboard-img" alt="Dashboard of the last run" hidden>
      </div>
    </section>
  </main>

  <footer><p id="engine"></p></footer>
</body>
</html>
```

Create `playground/app.css`. The hex values are first-paint fallbacks only: `app.js` overwrites them from `palette.json` as soon as it loads, and they are copied from Task 4's `CHROME`.

```css
/* Dark only. app.js sets --background, --panel, --border, --text and --link from palette.json's "chrome" once it has
   fetched it. The five values below are only for the first paint before that fetch; they are copied from CHROME in
   zero2fsd/palette.py (Task 4). --error and --ok are status colours that are not in the palette. */
:root {
  color-scheme: dark;
  --background: #090d16;
  --panel: #0d1117;
  --border: #30363d;
  --text: #c9d1d9;
  --link: #58a6ff;
  --error: #FC6255;
  --ok: #83C167;
  --sans: Inter, system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, monospace;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  padding: 12px 16px 24px;
  background: var(--background);
  color: var(--text);
  font: 15px/1.5 var(--sans);
}

a { color: var(--link); }

header h1 { margin: 0 0 4px; font-size: 1.1rem; }
header p { margin: 0 0 4px; }
nav { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 0.9rem; }

main { display: grid; gap: 16px; margin-top: 12px; }
@media (min-width: 720px) {
  main { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); align-items: start; }
}
section { min-width: 0; }
[hidden] { display: none !important; }

button, select, input { font: inherit; }
button, select {
  min-height: 36px;
  padding: 4px 12px;
  color: var(--text);
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
}
button:not(:disabled) { cursor: pointer; }
button:not(:disabled):hover { border-color: var(--link); }
button:disabled { opacity: 0.45; }
#run:not(:disabled) { border-color: var(--ok); }

#editor { border: 1px solid var(--border); border-radius: 6px; overflow: hidden; background: var(--panel); }
#editor .cm-editor { max-height: 24rem; font-size: 13px; }
#editor .cm-scroller { overflow: auto; font-family: var(--mono); }
#editor-fallback {
  display: block;
  width: 100%;
  min-height: 14rem;
  padding: 8px;
  color: var(--text);
  background: var(--panel);
  border: 0;
  font: 13px/1.5 var(--mono);
  resize: vertical;
}

.toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 8px 0; }
.toolbar label { margin-left: auto; }

#status { margin: 8px 0 0; padding: 6px 10px; border-left: 3px solid var(--border); background: var(--panel); overflow-wrap: anywhere; }
#status[data-state="running"] { border-left-color: var(--link); }
#status[data-state="done"], #status[data-state="ready"] { border-left-color: var(--ok); }
#status[data-state="error"] { border-left-color: var(--error); }
.status-actions { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; margin-top: 8px; }

pre { margin: 8px 0 0; padding: 8px; overflow-x: auto; background: var(--panel); border: 1px solid var(--border); border-radius: 6px; font: 12px/1.5 var(--mono); }
pre:empty { display: none; }
details { margin-top: 8px; }

#verdict { margin-top: 8px; padding: 6px 10px; background: var(--panel); border-left: 3px solid var(--error); }
#verdict[data-passed="true"] { border-left-color: var(--ok); }
#verdict ul { margin: 0; padding-left: 1.2em; }
#verdict li { overflow-wrap: anywhere; }

#stats { display: grid; grid-template-columns: max-content 1fr; gap: 2px 16px; margin: 8px 0 0; font-family: var(--mono); }
#stats dd { margin: 0; }

canvas { display: block; width: 100%; background: var(--panel); border: 1px solid var(--border); border-radius: 6px; }
#canvas { aspect-ratio: 4 / 3; }
#strips { height: 180px; margin-top: 8px; }
/* The camera picture; an image exercise shows its masks in a second canvas beside it. */
.camera { display: grid; grid-auto-flow: column; grid-auto-columns: minmax(0, 1fr); gap: 8px; margin-top: 8px; }
#frame, #mask { aspect-ratio: 16 / 9; image-rendering: pixelated; }

.transport { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 8px; }
.transport input[type="range"] { flex: 1 1 120px; min-width: 0; }

.readouts { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 4px 12px; margin: 8px 0 0; }
.readouts dt { font-size: 0.75rem; opacity: 0.7; }
.readouts dd { margin: 0; font-family: var(--mono); }

.dashboard { margin-top: 8px; }
#dashboard-img { display: block; width: 100%; margin-top: 8px; border-radius: 6px; }

footer { margin-top: 16px; font-size: 0.75rem; opacity: 0.6; }
footer p { margin: 0; }
```

Create `playground/app.js`:

```js
// Wiring: the URL picks an exercise, the worker runs it, the player replays it. All the DOM lives here.
import { Player } from "./player.js";
import { describeRun, drawStrips, fitCanvas, readoutText } from "./view.js";
import { markPassed } from "./progress.js";

const FRAME_INTERVAL_MS = 100; // camera-frame requests are throttled to one per interval
// What each exercise kind does and shows. The kind comes from exercises.json; no exercise id appears in this file.
// `shows` lists the page groups that kind uses; every other group is hidden (hidden, never removed).
const KINDS = {
  drive: { message: "run", shows: ["road-label", "drive-view", "player-view", "dashboard-view"] },
  stats: { message: "stats", shows: [] },
  image: { message: "image", shows: ["drive-view", "mask"] },
};
const ANSI_COLOUR = /\x1b\[[0-9;]*m/g; // check() prints its verdict in colour; the page shows the words only
const GROUPS = [...new Set(Object.values(KINDS).flatMap((kind) => kind.shows))];
const RUN_SEED = 0; // the seed run() uses, so the frame shown is the frame the car saw
const EDITOR_MODULES = ["view", "state", "lang-python", "commands", "language", "theme-one-dark"].map((name) => `@codemirror/${name}`);

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);

let data; // { records, roads, palette }, loaded once
let app; // the page's long-lived pieces, built once: { record, roads, palette, player, strips, editor, notice, autoRun }
let worker;
let workerReady = false;
let busy = null; // the worker message in flight: "run" | "stats" | "image" | "check" | "dashboard"
let loadNote = ""; // "Stopped." while the runtime restarts after Stop
let hasDashboard = false;
let driveRoad = ""; // the road of the drive the player holds
let dashboardUrl = "";
const frames = { wanted: null, inFlight: false, timer: 0, sentAt: 0 };

// ---- status and controls ----------------------------------------------------------------------

function setStatus(state, text) {
  $("status").dataset.state = state;
  $("status").textContent = text;
}

function controls() {
  const idle = workerReady && !busy;
  $("run").disabled = $("check").disabled = !idle;
  $("stop").disabled = !busy;
  $("road").disabled = Boolean(busy);
  $("dashboard").disabled = !(idle && hasDashboard);
}

/** The worker cannot be used any more (it failed to start, or sent something the page cannot use): stop it and say why, with Retry. */
function failWorker(message) {
  worker?.terminate();
  workerReady = false;
  busy = null;
  showLoadFailure(message);
}

function showLoadFailure(message, { retry = true } = {}) {
  setStatus("error", message);
  $("retry").hidden = !retry;
  if (app) {
    $("colab-fallback").href = app.record.colab_url;
    $("colab-fallback").hidden = false;
  }
  controls();
}

function clearFailure() {
  $("retry").hidden = true;
  $("colab-fallback").hidden = true;
}

function clearOutputs() {
  $("stdout").textContent = "";
  $("verdict").hidden = true;
  $("stats").hidden = true;
  $("traceback").hidden = true;
}

// ---- data and page setup -----------------------------------------------------------------------

async function fetchJson(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (cause) {
    throw new Error(`Could not load ${url}: ${cause.message}`);
  }
}

async function loadData() {
  const [records, roads, palette] = await Promise.all(["exercises.json", "roads.json", "palette.json"].map(fetchJson));
  return { records, roads, palette };
}

/**
 * The exercise and preset the URL asks for. An unknown exercise falls back to the first record and an unknown
 * preset to the starter, each with a notice. `autoRun` is true only when the URL named both and both exist.
 */
function choose({ records }) {
  const id = params.get("exercise");
  const known = records.find((r) => r.id === id);
  const record = known ?? records[0];
  const notices = [];
  if (!known && id) notices.push(`Unknown exercise ${id}; showing ${record.label}.`);
  const asked = params.get("preset");
  const preset = asked !== null && Object.hasOwn(record.presets, asked) ? asked : "starter";
  if (asked !== null && preset !== asked) notices.push(`Unknown preset ${asked}; showing the starter.`);
  return { record, code: record.presets[preset], notice: notices.join(" "), autoRun: Boolean(known) && asked === preset };
}

/** CodeMirror 6 from the import map in index.html; any failure leaves a plain textarea. Returns { get, set } for the code. */
async function mountEditor(parent, doc) {
  try {
    const [view, state, python, commands, language, dark] = await Promise.all(EDITOR_MODULES.map((name) => import(name)));
    const editor = new view.EditorView({
      parent,
      state: state.EditorState.create({
        doc,
        extensions: [
          view.lineNumbers(), view.drawSelection(), language.indentOnInput(), language.bracketMatching(), commands.history(),
          view.keymap.of([...commands.defaultKeymap, ...commands.historyKeymap, commands.indentWithTab]),
          python.python(), dark.oneDark,
        ],
      }),
    });
    return {
      get: () => editor.state.doc.toString(),
      set: (text) => editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: text } }),
    };
  } catch (error) {
    console.warn("CodeMirror did not load; using a plain textarea.", error);
    const textarea = Object.assign(document.createElement("textarea"), { id: "editor-fallback", value: doc, spellcheck: false });
    parent.replaceChildren(textarea);
    return { get: () => textarea.value, set: (text) => { textarea.value = text; } };
  }
}

/** The chrome colours come from palette.json at runtime, so no hex is typed twice: each key becomes `--<key>`. */
function applyChrome(chrome) {
  if (!chrome) throw new Error('Could not use palette.json: it has no "chrome" colours.');
  for (const [name, value] of Object.entries(chrome)) document.documentElement.style.setProperty(`--${name}`, value);
}

async function setUp({ records, roads, palette }) {
  const { record, code, notice, autoRun } = choose({ records });
  document.title = `${record.label} - Playground`;
  $("exercise-label").textContent = record.label;
  $("exercise-task").textContent = record.task ?? "";
  $("exercise-task").hidden = !record.task;
  $("unit-link").href = `../unit/${record.unit_key}/`; // relative: the lecture sits beside the playground wherever the course is served
  $("open-colab").href = record.colab_url;
  const kind = KINDS[record.kind];
  if (!kind) throw new Error(`Exercise ${record.id} has kind "${record.kind}", which this page does not know how to run.`);
  GROUPS.forEach((id) => { $(id).hidden = !kind.shows.includes(id); });
  const roadName = record.road || Object.keys(roads)[0]; // an image exercise names no road ("")
  $("road").replaceChildren(...Object.keys(roads).map((name) => new Option(name, name, false, name === roadName)));

  const player = new Player($("canvas"), roads[roadName], palette);
  driveRoad = roadName;
  const strips = { ctx: fitCanvas($("strips")).ctx };
  const editor = await mountEditor($("editor"), code);
  window.__setCode = editor.set; // test hook: the browser tests paste reference solutions with it
  player.onTick(({ k, row, playing }) => {
    $("play").textContent = playing ? "Pause" : "Play";
    $("scrub").max = Math.max(0, player.length - 1);
    $("scrub").value = k;
    Object.entries(readoutText(row)).forEach(([name, text]) => { $(`r-${name}`).textContent = text; });
    drawStrips(strips.ctx, player.rows, k, palette);
    if (row) wantFrame(k, row);
  });
  new ResizeObserver(() => {
    player.fit();
    strips.ctx = fitCanvas($("strips")).ctx;
  }).observe(document.body);
  return { record, kind, roads, palette, player, strips, editor, notice, autoRun };
}

// ---- the worker --------------------------------------------------------------------------------

const missingFeatures = () =>
  [["Web Workers", "Worker"], ["WebAssembly", "WebAssembly"]].filter(([, name]) => typeof globalThis[name] === "undefined").map(([label]) => label);

function startWorker(note = "") {
  worker?.terminate();
  loadNote = note;
  workerReady = false;
  busy = null;
  frames.wanted = null;
  frames.inFlight = false;
  clearTimeout(frames.timer);
  frames.timer = 0;
  clearFailure();
  controls();
  const missing = missingFeatures();
  if (missing.length) {
    return showLoadFailure(`This browser has no ${missing.join(" or ")}, which the playground needs to run Python.`, { retry: false });
  }
  setStatus("loading", note ? `${note} Restarting Python...` : "Starting Python in your browser...");
  worker = new Worker("./worker.js", { type: "module" });
  worker.onmessage = ({ data: message }) => {
    try {
      (FROM_WORKER[message.type] ?? unknownMessage)(message);
    } catch (error) {
      failWorker(error.message); // a bug in the page or the worker is shown, never left as "Running..."
    }
  };
  worker.onerror = (event) => failWorker(`Could not start the worker (${event.filename || "worker.js"}): ${event.message || "see the browser console"}`);
  worker.postMessage({ type: "init" });
}

function unknownMessage(message) {
  console.error("Unknown message from the worker", message);
}

function start(message) {
  clearOutputs();
  busy = message;
  hasDashboard = false;
  setStatus("running", message === "check" ? "Checking your code..." : "Running your code...");
  worker.postMessage({ type: message, code: app.editor.get(), exercise: app.record, road: $("road").value });
  controls();
}

function stop() {
  app.player.finish(); // the rows so far stay on screen
  startWorker("Stopped.");
}

function onRow(row) {
  if (row.t === 0) { // a new drive begins at t = 0; a check can drive several roads in turn
    const road = app.roads[row.road];
    if (!road) throw new Error(`A streamed step names the road "${row.road}", which roads.json does not have.`);
    driveRoad = row.road;
    app.player.reset(road);
    app.player.play();
  }
  app.player.append(row);
}

function finishWork() {
  busy = null;
  app.player.finish();
  controls();
}

function wantFrame(k, row) {
  frames.wanted = { k, row };
  pumpFrames();
}

function pumpFrames() {
  if (!frames.wanted || frames.inFlight || frames.timer || !workerReady || busy) return; // the worker is single-threaded: frames wait for the run
  const wait = Math.max(0, FRAME_INTERVAL_MS - (performance.now() - frames.sentAt));
  frames.timer = setTimeout(() => {
    frames.timer = 0;
    sendFrame();
  }, wait);
}

function sendFrame() {
  if (!frames.wanted || !workerReady || busy) return;
  const { k, row } = frames.wanted;
  frames.wanted = null;
  frames.inFlight = true;
  frames.sentAt = performance.now();
  worker.postMessage({ type: "frame", road: driveRoad, k, x: row.x, y: row.y, yaw: row.yaw, speed: row.speed, seed: RUN_SEED });
}

function paintRgb(canvas, width, height, rgb) {
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  const image = new ImageData(width, height);
  const source = new Uint8Array(rgb);
  for (let i = 0, j = 0; i < source.length; i += 3, j += 4) {
    image.data.set([source[i], source[i + 1], source[i + 2], 255], j);
  }
  canvas.getContext("2d").putImageData(image, 0, 0);
}

function showStats(values) {
  const items = Object.entries(values).flatMap(([label, value]) => [
    Object.assign(document.createElement("dt"), { textContent: label }),
    Object.assign(document.createElement("dd"), { textContent: Number(value.toPrecision(4)).toString() }),
  ]);
  $("stats").replaceChildren(...items);
  $("stats").hidden = false;
}

const FROM_WORKER = {
  progress: ({ detail }) => setStatus(workerReady ? "running" : "loading", [loadNote, `Loading ${detail}...`].filter(Boolean).join(" ")),

  ready: ({ versions, steps_per_s_probe: stepsPerS }) => {
    workerReady = true;
    loadNote = "";
    $("engine").textContent =
      `Pyodide ${versions.pyodide}, Python ${versions.python}, NumPy ${versions.numpy}, Pillow ${versions.pillow}; ` +
      `the simulator runs about ${Math.round(stepsPerS)} steps per second on this device.`;
    setStatus("ready", [app.notice, "Ready."].filter(Boolean).join(" "));
    controls();
    pumpFrames();
    if (app.autoRun) {
      app.autoRun = false; // once: Stop or a later Retry must not start another run
      start(app.kind.message);
    }
  },

  row: ({ row }) => onRow(row),

  stdout: ({ text }) => {
    $("stdout").textContent += `${text.replace(ANSI_COLOUR, "")}\n`;
  },

  done: ({ metrics }) => {
    const seconds = app.player.rows.at(-1)?.t ?? 0;
    hasDashboard = true;
    finishWork();
    setStatus("done", describeRun(metrics, driveRoad, seconds));
    pumpFrames();
  },

  stats: ({ values }) => {
    finishWork();
    showStats(values);
    setStatus("done", `Ran ${app.record.name} on a recorded drive; ${Object.keys(values).length} values below.`);
  },

  image: ({ width, height, frame, overlay }) => {
    finishWork();
    paintRgb($("frame"), width, height, frame);
    paintRgb($("mask"), width, height, overlay);
    setStatus("done", `Ran ${app.record.name} on the lab's noisy camera frame. Left: the frame. Right: the pixels your masks marked.`);
  },

  verdict: ({ passed, details }) => {
    finishWork();
    $("verdict").dataset.passed = String(passed);
    $("verdict").querySelector("ul").replaceChildren(...details.map((text) => Object.assign(document.createElement("li"), { textContent: text })));
    $("verdict").hidden = false;
    const saved = passed && markPassed(app.record.unit_key, app.record.id);
    setStatus("done", !passed ? "Check did not pass. Details below." : saved ? "Check passed. Your progress is saved." : "Check passed, but this browser blocked saving your progress.");
    pumpFrames();
  },

  frame: (message) => {
    frames.inFlight = false;
    paintRgb($("frame"), message.width, message.height, message.rgb);
    pumpFrames();
  },

  dashboard: ({ png }) => {
    URL.revokeObjectURL(dashboardUrl);
    dashboardUrl = URL.createObjectURL(new Blob([png], { type: "image/png" }));
    $("dashboard-img").src = dashboardUrl;
    $("dashboard-img").hidden = false;
    busy = null;
    hasDashboard = true;
    setStatus("done", "Dashboard drawn below.");
    controls();
  },

  error: ({ message, line, traceback, request }) => {
    if (request === "init") return failWorker(message);
    frames.inFlight = false;
    finishWork();
    const said = line ? `${message} (your code, line ${line})` : message;
    setStatus("error", said);
    $("stdout").textContent += `${said}\n`; // the same line where the learner's own prints appear
    $("traceback").querySelector("pre").textContent = traceback;
    $("traceback").hidden = !traceback;
  },
};

// ---- user input ---------------------------------------------------------------------------------

async function boot() {
  try {
    data ??= await loadData();
    applyChrome(data.palette.chrome);
    app ??= await setUp(data);
    startWorker();
  } catch (error) {
    showLoadFailure(error.message);
  }
}

$("run").addEventListener("click", () => start(app.kind.message));
$("check").addEventListener("click", () => start("check"));
$("stop").addEventListener("click", stop);
$("retry").addEventListener("click", boot);
$("play").addEventListener("click", () => app.player.toggle());
$("to-end").addEventListener("click", () => app.player.seek(app.player.length - 1));
$("scrub").addEventListener("input", (event) => app.player.seek(event.target.valueAsNumber));
$("speed").addEventListener("change", (event) => app.player.setSpeed(Number(event.target.value)));
$("road").addEventListener("change", (event) => {
  driveRoad = event.target.value;
  app.player.reset(app.roads[driveRoad]); // a different road starts a different drive; the old rows do not belong on it
  app.player.finish();
});
$("dashboard").addEventListener("click", () => {
  busy = "dashboard";
  setStatus("running", "Drawing the dashboard (the first time this downloads matplotlib)...");
  worker.postMessage({ type: "dashboard" });
  controls();
});

boot();
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
node /tmp/playground-checks/check_progress.mjs playground
pkill -f "[h]ttp.server 8765" || true
python scripts/build_course_bundle.py
python -m http.server 8765 --directory course/dist/playground &
until curl -s -o /dev/null http://localhost:8765/; do :; done
(cd /tmp/playground-checks && python -u check_page.py http://localhost:8765/)
```

Expected: `check_progress.mjs: all assertions passed`, then 20 lines `PASS 1 ...` to `PASS 19 ...` (case 13 appears twice: no WebAssembly, no Worker), then `ALL PASS`, after about two to three minutes.

Manual smoke. Keep the server running and open these in Chrome:

1. `http://localhost:8765/index.html?exercise=0.1.4.a`. The heading shows the exercise label (for example "Week 4 · Closing the loop · Exercise A") and no dotted id. Under it, the status line reads "Starting Python in your browser..." and then "Ready." The engine line reads "Pyodide 314.0.7, Python 3.14.2, NumPy 2.4.6, Pillow 12.2.0; the simulator runs about 190 steps per second on this device." (the rate depends on your machine). The editor shows the starter in a dark CodeMirror with line numbers.
2. Paste the `0.1.4.a` solution from `solutions.py`, press Run. The car drives the loop; the readouts count up; the three strip charts scroll with names at the line ends; the status ends "Completed gentle in 88.9 s, mean |lat| 0.38 m". Drag the scrub bar: the car and the camera frame move together. Press Check: the verdict reads "all 36 steering cases match" and the status reads "Check passed. Your progress is saved."
3. `http://localhost:8765/index.html?exercise=0.1.1.a`. The drive area is gone. Paste the `0.1.1.a` solution and press Run: the status reads "Ran lane_error_stats on a recorded drive; 3 values below." with `mean_abs`, `max_abs` and `rms` listed.
4. `http://localhost:8765/index.html?exercise=0.1.2.a`. Paste the `0.1.2.a` solution and press Run: the status reads "Ran paint_masks on the lab's noisy camera frame. Left: the frame. Right: the pixels your masks marked." and the two 320 x 180 pictures sit side by side. The player, the scrub bar and the strips are absent.
5. `http://localhost:8765/index.html?exercise=nonsense`. The first exercise opens and the status reads "Unknown exercise nonsense; showing ..." and then "Ready."

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests -q`
Expected: all green; 412 passed, the same as after Task 8 (this task adds no Python tests).

- [ ] **Step 6: Commit**

```bash
git add playground/progress.js playground/index.html playground/app.css playground/app.js course/dist/playground
git commit -m "feat(playground): the page, the editor and the progress store

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 12: Browser tests (playwright-python)

**Files:**
- Create: `scripts/playground_browser.py` (helpers shared by the tests, `scripts/measure_phone.py` and the manual steps)
- Create: `tests/playground/test_browser.py`
- Modify: `pytest.ini:2` (deselect `browser` by default) and add line 5 (register the marker)
- Modify: `pyproject.toml:12-13` (add the `test` extra after the `dev` line)
- Test: `tests/playground/test_browser.py`

**Interfaces:**
- Consumes:
  - `scripts/build_playground.py`: `exercise_records(repo_root: Path) -> list[dict]` and `write_playground(repo_root: Path, out_dir: Path) -> None` (Task 8). Records carry `id, unit_id, unit_key, label, name, kind, road, colab_url`.
  - `scripts/_course.py:12` `REPO`, `scripts/_course.py:39` `load_solution(root, unit_id)` (loads `solutions/course/<unit_id>/solution.py` by path).
  - `zero2fsd/grade/__init__.py:18` `check(exercise_id, fn_or_box, show=True) -> Result`; `zero2fsd/grade/_report.py:23` `Result(passed: bool, details: list[str])`, `details[0]` is the one-line summary.
  - `zero2fsd/preview` `get(exercise_id) -> Preview(kind, build, road="")` (Task 5). Stats: `build(fn, tel) -> dict[str, float]`, run on `run(Car(), preview.road)`. Image: `build(fn, frame) -> uint8[H, W, 3]` with `road == ""`. Task 7's `preview_image` paints the learner's masks in the `SCENE` yellow and white, which the page serves as `palette.json` `scene.yellow` and `scene.white`. Week 1 holds seven exercises: stats 0.1.1.a and 0.1.1.b, image 0.1.2.a, drive 0.1.3.a, 0.1.3.b, 0.1.4.a and 0.1.4.b.
  - `zero2fsd/car/__init__.py:7` `Car`, `run`, `provided`; `zero2fsd/car/provided.py:60` `steer_p`; `zero2fsd/sim/dynamics.py:8` `DT = 0.05`.
  - Task 11 DOM ids: `#exercise-label #unit-link #open-colab #editor #editor-fallback #run #check #stop #canvas #play #scrub #status[data-state] #verdict[data-passed] #stdout #r-progress #frame #dashboard #dashboard-img`, plus the ids this task adds to the contract: `#stats` (dl), `#retry`, `#strips`, `#to-end`, `#exercise-task`, `#mask` (canvas, image exercises only).
  - Task 9 worker: `row` messages stream during Check for the checks that drive the car (spec 5.3).
  - Task 11 `window.__setCode(text)`: replaces the code, whether CodeMirror or the textarea is active. Task 11's `error` handler writes the learner-facing line to both `#status` and `#stdout`.
- Produces (Task 13 and the manual acceptance steps rely on these):
  - `scripts/playground_browser.py`: `P_STEER`, `EDITOR_READY`, `pasteable(unit_id) -> str`, `paint_colours(base_url) -> list[list[int]]`, `serve(directory)` (context manager yielding a base URL with a trailing slash), `exercise_url(base_url, exercise_id, query="") -> str`, `wait_state(page, state, timeout_ms=60_000)`, `open_ready(page, base_url, exercise_id, query="")`, `set_code(page, code)`.
  - pytest marker `browser`, deselected by default; `PLAYGROUND_TIMING=1` turns on the S-1 timing test and tightens the Stop budget to the S-4 figure.

> NOTE (unknown id): an unknown `?exercise=` opens the FIRST exercise in state `ready`, and `#status` contains "Unknown exercise <id>". Source: spec section 6 table. Task 11's `app.js` already does this.
> NOTE (page URL): the canonical page URL is `<base>/index.html?exercise=<id>[&preset=<name>]`. Next serves `public/` by exact path, so the directory URL does not work on the site. Every navigation in this task uses `index.html`. The site plan's playground panel (Task 4) builds the iframe `src` and the "Open full screen" link as `/learnfsd/playground/index.html?exercise=...`, and Task 8's `validate_lecture_links` already accepts only that form.
> NOTE (page hooks): this task makes no `app.js` edit. Task 11's `app.js` already defines `window.__setCode` and writes the learner-facing error line to both `#status` and `#stdout`, which the pinned cases read.
> NOTE (wrong return type text): the pinned case expects "must return one number" in `#stdout`. Task 7 rewords `zero2fsd/car/car.py:40` to "the controller box must return one number (a steering angle in radians) or an Action, got <type>" and its `_at_boundary` wrapper prefixes the exercise id, so the page shows `0.1.4.a: the controller box must return one number ...` verbatim. The stats case reads the same way: Task 7's `_numbers` raises "expected a number or a dict of numbers, got <type>" and `_at_boundary` prefixes the exercise id.
> NOTE (`#stats`): one `<dt>` (the label) and one `<dd>` (the number at 4 significant digits) per value. The Task 11 draft's `showStats` (`app.js:243-250`) does this. `#retry` (hidden until a load failure, click boots again with a new Worker) is also part of the contract.
> NOTE (`unit_key`): `unit_key` in `exercises.json` is the unit slug in dash form (`0-1-4`). The progress test pins the literal slug, so a wrong key form fails it.
> NOTE (image exercise; spec S10 is out of date): Week 1's paint-mask exercise 0.1.2.a has kind `image` (Task 5 table), so the tests cover three Run kinds. Spec S10 ("`drive` and `stats`", "`image` has no Week 1 exercise") predates it. What Task 11 does for an image exercise: Run shows one camera frame in `#frame` and the worker's `overlay` bytes alone in the canvas `#mask` (black except where a mask is set), the yellow mask in `palette.json` `scene.yellow` and the white mask in `scene.white`; both lines of paint are in that frame; the player controls `#play #scrub #to-end` carry the `hidden` attribute. For drive and stats exercises `#mask` is hidden and the controls are visible.
> NOTE (`paint_colours`): the mask tests read the two paints from the served `palette.json` (`scene.yellow`, then `scene.white`) through `paint_colours(base_url)` in `scripts/playground_browser.py`, so a palette change cannot leave them passing on stale colours. This block was rewritten at assembly and has not yet run in a browser; the executor's Task 12 run is its first test.
> NOTE (stats road): `shim.preview_stats` must not run `run(Car(), "gentle")`. Task 5 fixes the stats road as `STATS_ROAD = "curvy"` (a `gentle` drive never leaves the lane, so a count of off-lane steps would read 0) and stores it on each stats preview. Task 7's `preview_stats` must run `run(Car(), get(exercise_id).road)`. The stats test below takes the road from the preview, so a hard-coded `gentle` fails it with both numbers printed.

Prerequisite before Step 2 (so the red run is a real failure and not a skip). Run from the course repo root with the repo `.venv` active. This installs a browser, about 300 MB, and changes nothing in the repo:

```bash
python -m pip install 'playwright>=1.40'
python -m playwright install chromium webkit
```

- [ ] **Step 1: Write the failing test**

```python
# tests/playground/test_browser.py
"""Browser tests for the playground (spec section 9). Opt-in: python -m pytest tests/playground/test_browser.py -m browser -v

Setup once: python -m pip install 'playwright>=1.40' && python -m playwright install chromium
Each test opens a fresh browser context, so each one downloads Pyodide again (about 10 MB).
The module takes 15 to 25 minutes. PLAYGROUND_TIMING=1 also runs the S-1 timing test and holds Stop to the S-4 budget.
"""
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))  # the course scripts are importable modules

from playwright.sync_api import expect, sync_playwright  # noqa: E402

from _course import REPO, load_solution  # noqa: E402
from build_playground import exercise_records, write_playground  # noqa: E402
from playground_browser import P_STEER, exercise_url, open_ready, paint_colours, pasteable, serve, set_code, wait_state  # noqa: E402
from zero2fsd.car import Car, provided, run  # noqa: E402
from zero2fsd.grade import check  # noqa: E402
from zero2fsd.preview import get as get_preview  # noqa: E402
from zero2fsd.sim.dynamics import DT  # noqa: E402

pytestmark = pytest.mark.browser

TIMING = bool(os.environ.get("PLAYGROUND_TIMING"))
LOAD_BUDGET_S = 3.0  # spec S-1
REALTIME_FACTOR = 5  # spec §7: a lap of gentle with camera runs at >= 5x real time on a Mac (measured 10.3x)
STOP_BUDGET_S = 3.0 if TIMING else 5.0  # spec S-4 is 3 s; the 5 s default ceiling keeps a busy machine from failing the default run
STEER_ID = "0.1.4.a"
# A literal on purpose: this key is the contract with the site (src/lib/learnfsd-progress.ts:7), not something to read back from progress.js.
PROGRESS_KEY = "learnfsd.v2.progress"
# (exercise id, the unit slug the site files its progress under)
PASSING = (("0.1.4.a", "0-1-4"), ("0.1.3.a", "0-1-3"))
STREAMING = ("0.1.4.b", "0.1.3.b")  # spec 5.3: the two checks that drive the car, so their rows stream
# What a learner holds before writing anything, pasted and as the Mac holds it: the provided boxes.
# check() refuses them in require_own before it drives, so a Check on them streams no rows.
UNDRIVEN = {
    "0.1.4.b": ("from zero2fsd.car import Car\nmy_car = Car()\n", Car()),
    "0.1.3.b": ("from zero2fsd.car import provided\nestimate_lane = provided.estimate_lane\n", provided.estimate_lane),
}
# The exercises S-2 covers, one case each. A registry that lost one would shrink the loop without failing it.
WEEK_1 = ("0.1.1.a", "0.1.1.b", "0.1.2.a", "0.1.3.a", "0.1.3.b", "0.1.4.a", "0.1.4.b")

RECORDS = exercise_records(REPO)
BY_ID = {record["id"]: record for record in RECORDS}
RUN_KINDS = ("drive", "stats", "image")  # zero2fsd.preview.Kind: Week 1 has an exercise of each
STATS = [record for record in RECORDS if record["kind"] == "stats"]
IMAGES = [record for record in RECORDS if record["kind"] == "image"]
PLAYER_CONTROLS = ("#play", "#scrub", "#to-end")  # visible on a drive exercise, hidden on an image exercise


def first_of_kind(kind):
    found = [record for record in RECORDS if record["kind"] == kind]
    assert found, f"no exercise has kind {kind!r}; these tests need one per preview kind"
    return found[0]


ONE_PER_KIND = [first_of_kind(kind) for kind in RUN_KINDS]
# Per kind: the exercise whose answer is a number, and the plain-language text the page must show when it is not (Task 7's
# `_at_boundary` and `_numbers`). The other drive exercises answer with a LaneEstimate or a Car: Task 7 tests those messages.
WRONG_RETURN = {
    "drive": (BY_ID[STEER_ID], "must return one number"),
    "stats": (first_of_kind("stats"), "expected a number or a dict of numbers"),
}

RETURN_ZERO = "def p_steer(est, k_off, k_head):\n    return 0.0\n"
FOREVER = "def p_steer(est, k_off, k_head):\n    while True:\n        pass\n"
DIVIDES_BY_ZERO = "def p_steer(est, k_off, k_head):\n    return 1 / 0\n"
RAISES_AFTER_50_STEPS = '''calls = 0


def p_steer(est, k_off, k_head):
    global calls
    calls += 1
    if calls > 50:
        raise ValueError("steering table exhausted")
    return 0.0
'''

# Records every data-state the status line takes, so a test can read the order the page went through.
STATE_RECORDER = """
window.__states = [];
new MutationObserver((records) => {
  for (const r of records) if (r.target.id === "status") window.__states.push(r.target.dataset.state);
}).observe(document, { subtree: true, attributes: true, attributeFilter: ["data-state"] });
"""
# localStorage exists but every read throws, as in a browser with site data blocked.
BLOCK_STORAGE = """Object.defineProperty(window, "localStorage", {
  configurable: true,
  get() { throw new DOMException("storage is blocked", "SecurityError"); },
});"""
# How many pixels of a canvas differ from its corner pixel (the background).
DRAWN_PIXELS_JS = """(selector) => {
  const c = document.querySelector(selector), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  const key = (i) => ((d[i] << 24) | (d[i + 1] << 16) | (d[i + 2] << 8) | d[i + 3]) >>> 0, corner = key(0);
  let drawn = 0;
  for (let i = 0; i < d.length; i += 4) if (key(i) !== corner) drawn++;
  return drawn;
}"""
# On a canvas: how many pixels are not black, and for each colour how many pixels are within 8 per channel of it
# (a canvas that blends the overlay with alpha rounds the colour a little).
MASK_PIXELS_JS = """({ selector, colours }) => {
  const c = document.querySelector(selector), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  const near = (i, rgb) => rgb.every((v, k) => Math.abs(d[i + k] - v) <= 8);
  const counts = { nonBlack: 0, colour: colours.map(() => 0) };
  for (let i = 0; i < d.length; i += 4) {
    if (d[i] || d[i + 1] || d[i + 2]) counts.nonBlack++;
    colours.forEach((rgb, k) => { if (near(i, rgb)) counts.colour[k]++; });
  }
  return counts;
}"""
FRAME_SUM_JS = """() => {
  const c = document.querySelector('#frame');
  return c.getContext('2d').getImageData(0, 0, c.width, c.height).data.reduce((a, b) => a + b, 0);
}"""
# Longest gap between two 20 ms timer ticks on the page's main thread: a frozen page shows up as a big number.
TICKER_JS = """() => {
  window.__ticks = { last: performance.now(), maxGap: 0 };
  setInterval(() => {
    const now = performance.now();
    window.__ticks.maxGap = Math.max(window.__ticks.maxGap, now - window.__ticks.last);
    window.__ticks.last = now;
  }, 20);
}"""
# (progress text, number of verdict lines on screen) every time the progress readout changes.
TRACE_PROGRESS_JS = """() => {
  window.__trace = [];
  const readout = document.querySelector('#r-progress');
  new MutationObserver(() => window.__trace.push([readout.textContent, document.querySelectorAll('#verdict li').length]))
    .observe(readout, { childList: true, characterData: true, subtree: true });
}"""


@pytest.fixture(scope="module")
def base_url(tmp_path_factory):
    dist = tmp_path_factory.mktemp("playground-dist")
    write_playground(REPO, dist)
    with serve(dist) as url:
        yield url


@pytest.fixture(scope="module")
def chromium():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        yield browser
        browser.close()


@pytest.fixture
def page_errors():
    return []


@pytest.fixture
def page(chromium, page_errors):
    context = chromium.new_context()
    context.add_init_script(STATE_RECORDER)
    page = context.new_page()
    page.set_default_timeout(60_000)
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    yield page
    context.close()


def visited(page):
    """The data-state values the status line went through, repeats collapsed."""
    states = page.evaluate("window.__states")
    return [state for i, state in enumerate(states) if i == 0 or state != states[i - 1]]


def line_of(code, needle):
    """1-based line number of the first line containing `needle`: the number the page must report for that line."""
    return next(i for i, line in enumerate(code.splitlines(), 1) if needle in line)


def first_number(text):
    match = re.search(r"\d+(?:\.\d+)?", text)
    assert match, f"no number in {text!r}"
    return float(match.group())


def press_check(page, code):
    """Put `code` in the editor, press Check, return the verdict's lines once the page reports done."""
    set_code(page, code)
    page.click("#check")
    wait_state(page, "done")
    return page.locator("#verdict li").all_text_contents()


def native_check(record):
    """The verdict the Mac prints for the reference solution of `record`."""
    return check(record["id"], getattr(load_solution(REPO, record["unit_id"]), record["name"]), show=False)


def stored_progress(page):
    return page.evaluate("key => localStorage.getItem(key)", PROGRESS_KEY)


def scroll_width(page):
    return page.evaluate("document.documentElement.scrollWidth")


# ---- opening ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["id"])
def test_every_exercise_opens_ready_with_its_label_and_links(page, base_url, record):
    open_ready(page, base_url, record["id"])
    expect(page.locator("#exercise-label")).to_have_text(record["label"])
    assert not re.search(r"\d+\.\d+\.\d+", page.inner_text("#exercise-label")), "a dotted id reached the learner"
    expect(page.locator("#unit-link")).to_have_attribute("href", f"../unit/{record['unit_key']}/")
    expect(page.locator("#open-colab")).to_have_attribute("href", record["colab_url"])


def test_the_registry_holds_the_seven_week_1_exercises():
    """S-2 and the opening test loop over the registry, so this is what keeps both loops at seven."""
    assert set(WEEK_1) <= set(BY_ID)
    assert {BY_ID[exercise_id]["kind"] for exercise_id in WEEK_1} == set(RUN_KINDS)


@pytest.mark.parametrize("bad_id", ["9.9.9.z", "nonsense"])
def test_unknown_exercise_opens_the_first_exercise_and_says_why(page, base_url, bad_id):
    page.goto(exercise_url(base_url, bad_id))
    wait_state(page, "ready")
    expect(page.locator("#status")).to_contain_text("Unknown exercise")
    expect(page.locator("#exercise-label")).to_have_text(RECORDS[0]["label"])


@pytest.mark.skipif(not TIMING, reason="timing budgets need an idle, wired machine: set PLAYGROUND_TIMING=1")
def test_cold_load_reaches_ready_within_3s(page, base_url):
    started = time.monotonic()
    page.goto(exercise_url(base_url, STEER_ID))
    wait_state(page, "ready", timeout_ms=int(LOAD_BUDGET_S * 1000))
    print(f"load_s={time.monotonic() - started:.2f}")


@pytest.mark.parametrize("record", ONE_PER_KIND, ids=lambda record: record["kind"])
def test_preset_starter_runs_automatically(page, base_url, record):
    open_ready(page, base_url, record["id"])
    assert "running" not in visited(page), "a plain URL started a run on its own"
    page.goto(exercise_url(base_url, record["id"], "&preset=starter"))
    page.wait_for_function("() => window.__states.includes('running')")
    states = visited(page)
    assert states.index("ready") < states.index("running")


# ---- running ------------------------------------------------------------------------------------------------


def test_run_provided_steer_completes_and_draws(page, base_url):
    record = BY_ID[STEER_ID]
    open_ready(page, base_url, STEER_ID)
    expect(page.locator("#editor .cm-content")).to_have_count(1)  # CodeMirror itself loaded, not the textarea fallback
    set_code(page, P_STEER)
    started = time.monotonic()
    page.click("#run")
    wait_state(page, "done")
    elapsed = time.monotonic() - started
    assert "running" in visited(page)
    expect(page.locator("#status")).to_contain_text(f"Completed {record['road']}")
    assert first_number(page.inner_text("#r-progress")) > 0
    expect(page.locator("#mask")).to_be_hidden()  # the image preview's canvas belongs to image exercises only
    for control in PLAYER_CONTROLS:
        expect(page.locator(control)).to_be_visible()
    for canvas in ("#canvas", "#strips"):
        assert page.evaluate(DRAWN_PIXELS_JS, canvas) > 0, f"{canvas} has only background pixels"
    steps = int(page.get_attribute("#scrub", "max")) + 1
    assert abs(steps - len(run(Car(), record["road"]).t)) <= 2  # the page drove the same lap the Mac drives
    print(f"real_time_x={steps * DT / elapsed:.1f}")  # recorded in the acceptance PR (step C9)
    assert elapsed * REALTIME_FACTOR < steps * DT, f"the lap ran at {steps * DT / elapsed:.1f}x real time; spec section 7 wants {REALTIME_FACTOR}x"
    page.click("#to-end")
    expect(page.locator("#scrub")).to_have_value(str(steps - 1))


def test_scrubbing_redraws_the_camera_frame(page, base_url):
    open_ready(page, base_url, STEER_ID)
    set_code(page, P_STEER)
    page.click("#run")
    wait_state(page, "done")
    if page.inner_text("#play") == "Pause":
        page.click("#play")  # hold the playhead still so only the scrubber moves it
    page.click("#to-end")
    page.wait_for_function(f"({FRAME_SUM_JS})() > 0")
    at_end = page.evaluate(FRAME_SUM_JS)
    page.fill("#scrub", "0")
    page.wait_for_function(f"({FRAME_SUM_JS})() !== {at_end}")


def test_dashboard_library_loads_only_on_demand(page, base_url):
    requested = []
    page.on("request", lambda request: requested.append(request.url))
    open_ready(page, base_url, STEER_ID)
    set_code(page, P_STEER)
    page.click("#run")
    wait_state(page, "done")
    assert not [url for url in requested if "matplotlib" in url.lower()], "matplotlib was fetched before Dashboard was pressed"
    page.click("#dashboard")
    expect(page.locator("#dashboard-img")).to_be_visible(timeout=120_000)
    assert [url for url in requested if "matplotlib" in url.lower()], "pressing Dashboard did not fetch matplotlib"


@pytest.mark.parametrize("record", STATS, ids=lambda record: record["id"])
def test_stats_exercise_shows_native_values(page, base_url, record):
    reference = getattr(load_solution(REPO, record["unit_id"]), record["name"])
    preview = get_preview(record["id"])
    native = preview.build(reference, run(Car(), preview.road))  # the canned drive is the preview's own road
    open_ready(page, base_url, record["id"])
    set_code(page, pasteable(record["unit_id"]))
    page.click("#run")
    wait_state(page, "done")
    shown = dict(zip(page.locator("#stats dt").all_text_contents(), page.locator("#stats dd").all_text_contents()))
    assert shown.keys() == native.keys()
    for label, value in native.items():
        assert math.isclose(float(shown[label]), value, rel_tol=1e-3), f"{label}: page {shown[label]}, Mac {value}"
    expect(page.locator("#mask")).to_be_hidden()


@pytest.mark.parametrize("record", IMAGES, ids=lambda record: record["id"])
def test_image_exercise_paints_the_masks_and_hides_the_player(page, base_url, record):
    """Week 1's paint-mask exercise: one camera frame in #frame, the learner's masks in #mask, no player."""
    open_ready(page, base_url, record["id"])
    set_code(page, pasteable(record["unit_id"]))
    page.click("#run")
    wait_state(page, "done")
    expect(page.locator("#frame")).to_be_visible()
    expect(page.locator("#mask")).to_be_visible()
    for control in PLAYER_CONTROLS:
        expect(page.locator(control)).to_be_hidden()
    painted = page.evaluate(MASK_PIXELS_JS, {"selector": "#mask", "colours": paint_colours(base_url)})
    assert painted["nonBlack"] > 0
    assert all(count > 0 for count in painted["colour"]), f"a mask was not painted in its colour: {painted}"


@pytest.mark.parametrize("record", IMAGES, ids=lambda record: record["id"])
def test_image_exercise_with_empty_masks_paints_no_overlay(page, base_url, record):
    """The control for the test above: a learner who marks nothing gets no overlay colour, so the count can tell the two apart."""
    open_ready(page, base_url, record["id"])
    nothing = f"import numpy as np\n\n\ndef {record['name']}(frame):\n    none = np.zeros(frame.shape[:2], dtype=bool)\n    return none, none\n"
    set_code(page, nothing)
    page.click("#run")
    wait_state(page, "done")
    colours = paint_colours(base_url)
    painted = page.evaluate(MASK_PIXELS_JS, {"selector": "#mask", "colours": colours})
    assert painted["colour"] == [0] * len(colours), painted


# ---- check --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["id"])
def test_check_matches_native_verdict_on_the_reference(page, base_url, record):
    """S-2: the verdict Pyodide prints for the reference solution is the verdict the Mac prints, line for line.

    One case per exercise in the registry: the seven of Week 1, the image exercise 0.1.2.a among them. No kind is skipped.
    """
    native = native_check(record)
    assert native.passed, f"the reference for {record['id']} fails natively: {native.details}"
    open_ready(page, base_url, record["id"])
    lines = press_check(page, pasteable(record["unit_id"]))
    expect(page.locator("#verdict")).to_have_attribute("data-passed", "true")
    assert lines == native.details


def test_failing_check_has_the_native_shape(page, base_url):
    """A failing verdict carries numpy-formatted arrays, which differ by version: compare headline, line count and indentation."""
    namespace = {}
    exec(RETURN_ZERO, namespace)
    native = check(STEER_ID, namespace["p_steer"], show=False)
    assert not native.passed
    open_ready(page, base_url, STEER_ID)
    lines = press_check(page, RETURN_ZERO)
    expect(page.locator("#verdict")).to_have_attribute("data-passed", "false")
    assert lines[0] == native.details[0]
    indent = lambda text: len(text) - len(text.lstrip())  # noqa: E731
    assert [indent(line) for line in lines] == [indent(line) for line in native.details]


def test_pass_is_recorded_in_progress_store(page, base_url):
    for exercise_id, _slug in PASSING:
        open_ready(page, base_url, exercise_id)
        press_check(page, pasteable(BY_ID[exercise_id]["unit_id"]))
        expect(page.locator("#verdict")).to_have_attribute("data-passed", "true")
    store = json.loads(stored_progress(page))  # read-modify-write: the second pass must not erase the first
    assert store["version"] == 1
    for exercise_id, slug in PASSING:
        entry = store["units"][slug]["exercises"][exercise_id]
        assert entry["via"] == "playground"
        assert entry["passedAt"]


def test_failing_check_records_nothing(page, base_url):
    open_ready(page, base_url, STEER_ID)
    press_check(page, RETURN_ZERO)
    expect(page.locator("#verdict")).to_have_attribute("data-passed", "false")
    assert STEER_ID not in (stored_progress(page) or "")


@pytest.mark.parametrize("exercise_id", STREAMING)
def test_check_streams_rows_before_the_verdict(page, base_url, exercise_id):
    """Pastes the reference solution, which holds the learner's own steer and perception. An empty Car would not do:
    check("0.1.4.b", Car()) fails in require_own before it drives, so nothing would stream."""
    open_ready(page, base_url, exercise_id)
    page.evaluate(TRACE_PROGRESS_JS)
    press_check(page, pasteable(BY_ID[exercise_id]["unit_id"]))
    before_verdict = {text for text, verdict_lines in page.evaluate("window.__trace") if verdict_lines == 0}
    assert len(before_verdict) >= 2, f"#r-progress never moved while the check drove: {before_verdict}"
    expect(page.locator("#verdict")).to_have_attribute("data-passed", "true")


@pytest.mark.parametrize("exercise_id", STREAMING)
def test_check_that_fails_before_driving_streams_nothing(page, base_url, exercise_id):
    """The control for the test above: the provided boxes are refused before the drive, so #r-progress holds still."""
    pasted, held_natively = UNDRIVEN[exercise_id]
    native = check(exercise_id, held_natively, show=False)
    assert not native.passed and "still uses the provided" in native.details[0], native.details
    open_ready(page, base_url, exercise_id)
    page.evaluate(TRACE_PROGRESS_JS)
    lines = press_check(page, pasted)
    expect(page.locator("#verdict")).to_have_attribute("data-passed", "false")
    assert lines[0] == native.details[0]
    before_verdict = {text for text, verdict_lines in page.evaluate("window.__trace") if verdict_lines == 0}
    assert len(before_verdict) <= 1, f"#r-progress moved although nothing drove: {before_verdict}"  # one reset at the start is allowed


# ---- stop ---------------------------------------------------------------------------------------------------


def test_stop_halts_while_true_and_page_stays_responsive(page, base_url):
    open_ready(page, base_url, STEER_ID)
    page.evaluate(TICKER_JS)
    set_code(page, FOREVER)
    page.click("#run")
    wait_state(page, "running")
    page.wait_for_timeout(1000)  # let the worker spin inside the loop before Stop
    started = time.monotonic()
    page.click("#stop")
    wait_state(page, "ready", timeout_ms=int(STOP_BUDGET_S * 1000))
    print(f"restart_s={time.monotonic() - started:.2f}")
    assert page.evaluate("window.__ticks.maxGap") < 500, "the page froze while the loop ran"
    assert visited(page)[-3:] == ["running", "loading", "ready"]
    set_code(page, P_STEER)
    page.click("#run")
    wait_state(page, "done")


def test_stop_during_a_normal_run_keeps_the_rows(page, base_url):
    open_ready(page, base_url, STEER_ID)
    set_code(page, P_STEER)
    page.click("#run")
    page.wait_for_function("() => /[1-9]/.test(document.querySelector('#r-progress').textContent)")
    page.click("#stop")
    wait_state(page, "ready")
    assert first_number(page.inner_text("#r-progress")) > 0
    assert int(page.get_attribute("#scrub", "max")) > 0


# ---- learner errors -----------------------------------------------------------------------------------------


def test_learner_exception_reports_class_and_line(page, base_url):
    open_ready(page, base_url, STEER_ID)
    set_code(page, DIVIDES_BY_ZERO)
    page.click("#run")
    wait_state(page, "error")
    status = page.inner_text("#status")
    assert "ZeroDivisionError" in status
    assert f"line {line_of(DIVIDES_BY_ZERO, '1 / 0')}" in status


def test_raise_at_step_50(page, base_url):
    open_ready(page, base_url, STEER_ID)
    page.evaluate("window.__survives = true")  # a reload would clear it
    set_code(page, RAISES_AFTER_50_STEPS)
    page.click("#run")
    wait_state(page, "error")
    assert f"line {line_of(RAISES_AFTER_50_STEPS, 'raise ValueError')}" in page.inner_text("#stdout")
    assert page.locator("#verdict li").count() == 0
    assert first_number(page.inner_text("#r-progress")) > 0
    set_code(page, P_STEER)
    page.click("#run")
    wait_state(page, "done")
    assert page.evaluate("window.__survives") is True


@pytest.mark.parametrize(
    "kind, returned, type_name",
    [("drive", "(0.1, 0.2)", "tuple"), ("drive", "None", "NoneType"), ("drive", '"left"', "str"), ("stats", '"big"', "str")],
    ids=["drive-tuple", "drive-none", "drive-text", "stats-text"],
)
def test_wrong_return_type(page, base_url, kind, returned, type_name):
    record, message = WRONG_RETURN[kind]
    open_ready(page, base_url, record["id"])
    set_code(page, f"def {record['name']}(*args, **kwargs):\n    return {returned}\n")  # one signature fits every kind
    page.click("#run")
    wait_state(page, "error")
    shown = page.inner_text("#stdout")
    assert message in shown
    assert record["id"] in shown  # the shim's `_at_boundary` prefixes the exercise id
    assert type_name in shown
    assert "numpy" not in shown.lower()


def test_storage_blocked(page, base_url, page_errors):
    page.add_init_script(BLOCK_STORAGE)
    open_ready(page, base_url, STEER_ID)
    press_check(page, P_STEER)
    expect(page.locator("#verdict")).to_have_attribute("data-passed", "true")
    assert page_errors == []


# ---- load failures ------------------------------------------------------------------------------------------


# The editor modules come from the import map in index.html:15-20, confirmed:
#   $ curl -sI https://cdn.jsdelivr.net/npm/@codemirror/view@6.43.14/dist/index.js | head -1
#   HTTP/2 200
def test_editor_import_blocked_falls_back_to_textarea(page, base_url):
    page.route(re.compile(r"/npm/@codemirror/"), lambda route: route.abort())
    open_ready(page, base_url, STEER_ID)
    expect(page.locator("#editor-fallback")).to_be_visible()
    set_code(page, P_STEER)
    page.click("#run")
    wait_state(page, "done")


def abort_then_retry(page, base_url, pattern, named):
    """Abort every request matching `pattern`: the page must say which file failed and offer Retry; Retry must recover."""
    page.route(pattern, lambda route: route.abort())
    page.goto(exercise_url(base_url, STEER_ID))
    wait_state(page, "error")
    expect(page.locator("#status")).to_contain_text(re.compile(named))
    expect(page.locator("#retry")).to_be_visible()
    page.unroute(pattern)
    page.click("#retry")
    wait_state(page, "ready")


# The URL the worker imports (worker.js:9, :59), confirmed:
#   $ curl -sI https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs | head -1
#   HTTP/2 200
# The pattern takes any version, so a Pyodide bump does not turn this test into one that aborts nothing.
def test_route_abort_pyodide_js(page, base_url):
    abort_then_retry(page, base_url, re.compile(r"/pyodide/v[^/]+/full/pyodide\.m?js$"), r"pyodide\.m?js")


@pytest.mark.parametrize("filename", ["bundle.zip", "exercises.json"])
def test_route_abort_names_the_failed_file(page, base_url, filename):
    abort_then_retry(page, base_url, re.compile(rf"/{re.escape(filename)}$"), re.escape(filename))


# The wheel a Dashboard press downloads (worker.js, the `dashboard` handler), confirmed in headless Chromium on 2026-10-11:
#   https://cdn.jsdelivr.net/pyodide/v314.0.7/full/matplotlib-3.10.8-cp314-cp314-pyemscripten_2026_0_wasm32.whl
# The pattern takes any version and any wheel tag.
MATPLOTLIB_WHEEL = re.compile(r"/pyodide/v[^/]+/full/matplotlib-[^/]+\.whl$")


def test_route_abort_names_the_matplotlib_wheel(page, base_url):
    """Spec section 6: a charting library that fails to load is reported, and the rest of the page keeps working."""
    open_ready(page, base_url, STEER_ID)
    set_code(page, P_STEER)
    page.click("#run")
    wait_state(page, "done")
    page.route(MATPLOTLIB_WHEEL, lambda route: route.abort())
    page.click("#dashboard")
    wait_state(page, "error")
    for surface in ("#status", "#stdout"):  # the two places Task 11's `error` handler writes the learner-facing line
        shown = page.inner_text(surface)
        assert "matplotlib" in shown and re.search(r"matplotlib-\S+\.whl", shown), f"{surface}: {shown!r}"
    expect(page.locator("#dashboard-img")).to_be_hidden()
    page.unroute(MATPLOTLIB_WHEEL)
    page.click("#run")  # the page is not left busy: a new run completes
    wait_state(page, "done")
    page.click("#dashboard")  # and the failed download left nothing half-loaded: this press fetches the wheel again
    expect(page.locator("#dashboard-img")).to_be_visible(timeout=120_000)


# ---- phone width --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("record", ONE_PER_KIND, ids=lambda record: record["kind"])
def test_no_horizontal_scroll_at_375(page, base_url, record):
    page.set_viewport_size({"width": 375, "height": 700})
    open_ready(page, base_url, record["id"])
    assert scroll_width(page) <= 375
    set_code(page, pasteable(record["unit_id"]))
    page.click("#run")
    wait_state(page, "done")
    assert scroll_width(page) <= 375


def test_no_horizontal_scroll_with_a_failing_verdict(page, base_url):
    """Failing verdict lines are the longest text the page shows."""
    page.set_viewport_size({"width": 375, "height": 700})
    open_ready(page, base_url, STEER_ID)
    press_check(page, RETURN_ZERO)
    assert scroll_width(page) <= 375
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/playground/test_browser.py -m browser -v`
Expected: collection error, `ERROR tests/playground/test_browser.py` with `ModuleNotFoundError: No module named 'playground_browser'`, then `Interrupted: 1 error during collection`.

- [ ] **Step 3: Write the minimal implementation**

New file `scripts/playground_browser.py`:

```python
"""Browser helpers shared by tests/playground/test_browser.py, scripts/measure_phone.py and the manual acceptance steps."""
import inspect
import json
import re
import sys
import threading
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote
from urllib.request import urlopen

from playwright.sync_api import expect

from _course import REPO

sys.path.insert(0, str(REPO))  # `python scripts/<name>.py` puts scripts/ first, not the repo root

from zero2fsd.car import provided  # noqa: E402

# The provided steering law under the exercise's function name: learner code that passes 0.1.4.a.
P_STEER = inspect.getsource(provided.steer_p).replace("def steer_p(", "def p_steer(", 1)
# CodeMirror mounted, or the textarea fallback shown: either way the learner can type.
EDITOR_READY = ("() => !!document.querySelector('#editor .cm-content') || "
                "((f) => f !== null && f.offsetParent !== null)(document.querySelector('#editor-fallback'))")
LOADED_SOLUTION = re.compile(r'^(?P<name>\w+) = load_solution\([^\n]*?"(?P<unit>[\d.]+)"\)\.(?P<attr>\w+)$', re.M)


def pasteable(unit_id: str) -> str:
    """A reference solution as one cell a learner could paste: the sibling units it loads by path are inlined above it."""
    source = (REPO / "solutions" / "course" / unit_id / "solution.py").read_text()
    siblings = [m["unit"] for m in LOADED_SOLUTION.finditer(source)]
    source = LOADED_SOLUTION.sub(r"\g<name> = \g<attr>", source).replace("from _course import load_solution\n", "")
    assert "load_solution" not in source, f"solutions/course/{unit_id}/solution.py loads something this helper cannot inline"
    return "".join(pasteable(unit) + "\n" for unit in siblings) + source


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@contextmanager
def serve(directory):
    """Serve `directory` on a free localhost port; yields the base URL (with a trailing slash)."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_Quiet, directory=str(directory)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()


def exercise_url(base_url, exercise_id, query=""):
    """The canonical page URL. Next serves public/ by exact path, so the directory URL does not work on the site."""
    return f"{base_url}index.html?exercise={quote(exercise_id, safe='')}{query}"


def paint_colours(base_url):
    """The two mask paints as [r, g, b] lists: palette.json scene.yellow, then scene.white (what the shim paints and the page draws)."""
    with urlopen(f"{base_url}palette.json") as response:
        scene = json.load(response)["scene"]
    return [[int(scene[name][i:i + 2], 16) for i in (1, 3, 5)] for name in ("yellow", "white")]


def wait_state(page, state, timeout_ms=60_000):
    expect(page.locator("#status")).to_have_attribute("data-state", state, timeout=timeout_ms)


def open_ready(page, base_url, exercise_id, query=""):
    """Open an exercise and wait until the runtime is ready and the editor is usable."""
    page.goto(exercise_url(base_url, exercise_id, query))
    wait_state(page, "ready")
    page.wait_for_function(EDITOR_READY)


def set_code(page, code):
    """Replace the editor's text: the textarea when it is the fallback, else CodeMirror through window.__setCode."""
    if page.locator("#editor-fallback").is_visible():
        page.fill("#editor-fallback", code)
    else:
        page.evaluate("text => window.__setCode(text)", code)
```

`pytest.ini` (the whole file after the edit; line 2 changes, line 5 is new). `-m "not browser"` in `addopts` deselects the browser tests by default, so `python -m pytest tests -q` stays the whole-suite command. Chosen over relying on `importorskip` alone because playwright is installed in some environments, and there `importorskip` would let the network-dependent module run by default. A command-line `-m browser` replaces the `addopts` expression, so every command in this task carries `-m browser`.

```ini
[pytest]
addopts = --import-mode=importlib -m "not browser"
markers =
    slow: measurement or long-running tests (still run by default)
    browser: drives a real browser through playwright (opt-in: run with -m browser; needs the [test] extra)
```

`pyproject.toml` (lines 12-15 today; the new line goes between 13 and 14):

```toml
[project.optional-dependencies]
dev = ["pytest", "nbformat"]
test = ["playwright>=1.40"]

[tool.setuptools.packages.find]
```

- [ ] **Step 4: Run the test to verify it passes**

The totals below are for today's registry. Confirm it first: `python -c "import sys; sys.path.insert(0, 'scripts'); from collections import Counter; from _course import REPO; from build_playground import exercise_records; print(Counter(r['kind'] for r in exercise_records(REPO)))"` prints `Counter({'drive': 4, 'stats': 2, 'image': 1})`. Seven exercises, so S-2 and the opening test run 7 cases each; the stats test runs 2, the image tests 1 each and the wrong-return test 4 (three drive answers, one stats answer).

Run: `PLAYGROUND_TIMING=1 python -m pytest tests/playground/test_browser.py -m browser -v -s`
Expected: PASS (52 tests plus the timing test, so `53 passed`); the output prints `load_s=` (budget 3.0) and `restart_s=` (budget 3.0). Without `PLAYGROUND_TIMING=1` the result is `52 passed, 1 skipped`. To run one case: `python -m pytest tests/playground/test_browser.py -m browser -k test_raise_at_step_50 -v`.

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests -q`
Expected: all green; 412 passed, the same as after Task 11 (this task adds none, because `-m "not browser"` deselects its tests), plus `53 deselected`. In an environment without playwright the module is skipped at collection instead: `1 skipped`.

- [ ] **Step 6: Commit**

```bash
git add scripts/playground_browser.py tests/playground/test_browser.py pytest.ini pyproject.toml
git commit -m "test(playground): playwright browser tests, opt-in behind -m browser

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Part C: Acceptance

### Task 13: Acceptance (Part C)

**Files:**
- Create: `scripts/measure_phone.py`
- Create: `tests/playground/test_measure_phone.py`
- Modify: `docs/superpowers/specs/2026-10-11-playground-design.md:185` (the S10 row, step C0b)
- Test: `tests/playground/test_measure_phone.py`

**Interfaces:**
- Consumes:
  - `scripts/playground_browser.py` (Task 12): `P_STEER`, `open_ready`, `serve`, `set_code`, `wait_state`.
  - `scripts/build_playground.py` `write_playground(repo_root, out_dir)` (Task 8).
  - `zero2fsd/sim/dynamics.py:8` `DT = 0.05`; `scripts/_course.py:12` `REPO`.
  - Status text from Task 11: "Completed gentle in 88.9 s, mean |lat| 0.38 m"; `#scrub` `max` is the last step index.
- Produces:
  - `scripts/measure_phone.py`: `lap_report(status_text: str, scrub_max: int, wall_s: float) -> dict` (keys `road, steps, wall_s, steps_per_s, real_time_x`); `main() -> int`; run as `python scripts/measure_phone.py`.
  - The acceptance checklist below, executed once before the PR leaves draft. Its numbers go in the PR description.

> NOTE (directory URL): the site leg reports that `/learnfsd/playground/` answers 308 and then 404, so the S-1, S-3 and S-5 steps below open `index.html?exercise=...`. Step C7 re-checks this with `curl`. See the page-URL NOTE under Task 12 for the URL form Task 8 and the site plan share.
> NOTE (emulation limits): Playwright device emulation sets viewport, user agent and touch only. It does not slow the CPU. The script's number is a ceiling for the phone. The phone itself is read with a stopwatch (C4).

- [ ] **Step 1: Write the failing test**

```python
# tests/playground/test_measure_phone.py
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))  # the course scripts are importable modules

from measure_phone import lap_report  # noqa: E402
from zero2fsd.sim.dynamics import DT  # noqa: E402

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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/playground/test_measure_phone.py -v`
Expected: collection error, `ERROR tests/playground/test_measure_phone.py` with `ModuleNotFoundError: No module named 'measure_phone'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# scripts/measure_phone.py
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/playground/test_measure_phone.py -v`
Expected: PASS (5 tests). Then run the script once as a smoke check: `python scripts/measure_phone.py` prints two lines, the second reading like `NNN steps/s, N.Nx real time`.

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests -q`
Expected: all green; 417 passed (412 after Task 12, plus 5; these tests need no browser, so they run by default).

- [ ] **Step 6: Commit**

```bash
git add scripts/measure_phone.py tests/playground/test_measure_phone.py
git commit -m "feat(playground): measure_phone.py reports steps per second on an emulated iPhone 12

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

#### The criteria, copied verbatim from the spec, each with its check

Spec section 1 (`docs/superpowers/specs/2026-10-11-playground-design.md`, lines 16-23):

| # | Criterion | How it is checked |
|---|---|---|
| S-1 | From a Week 1 lecture page, one click opens the playground with the car driving the lesson's road, in ≤ 3 s on a wired Mac | Playwright test; measured cold load today is 1.71 s without the charting library |
| S-2 | `check()` prints the identical verdict in Colab and in the playground for every Week 1 exercise on the reference solutions | Python test runs the grader natively; Playwright test runs it in Pyodide; the two texts are compared |
| S-3 | A PASS in the playground appears on the unit page and the course map without a reload | Playwright test on the site |
| S-4 | *Stop* halts `while True: pass` and the page stays responsive; a new run starts within 3 s | Playwright test; measured today: page ticker steady at 20 ms, restart 2.45 s |
| S-5 | Works on a phone. Slower is fine; wrong is not | Acceptance step: Vivek opens the URL on his phone and runs one exercise |
| S-6 | A learner cannot be told PASS on the Mac and FAIL in the browser (or the reverse) for the reference solutions | Margin test (§4.1 e) |

**S-1. Verified by** `tests/playground/test_browser.py::test_cold_load_reaches_ready_within_3s` for the seconds (runs with `PLAYGROUND_TIMING=1`; cold browser context, course files local, Pyodide from jsdelivr) and `test_every_exercise_opens_ready_with_its_label_and_links` for every exercise opening. The "one click from a lecture page" half is manual: step C2.

**S-2. Verified by** `tests/playground/test_browser.py::test_check_matches_native_verdict_on_the_reference` (one case per exercise id: all seven of Week 1, namely 0.1.1.a, 0.1.1.b, 0.1.2.a, 0.1.3.a, 0.1.3.b, 0.1.4.a and 0.1.4.b, the image exercise 0.1.2.a included; `test_the_registry_holds_the_seven_week_1_exercises` keeps the loop at seven and no kind is skipped). Each case runs `zero2fsd.grade.check(id, reference, show=False)` on the Mac, pastes the same reference source into the page, presses Check, and asserts the `#verdict` lines equal `Result.details` line by line. All seven references come from `solutions/course/<unit>/solution.py`, loaded by path with `load_solution` natively and with `pasteable(unit_id)` in the page; none needs `inspect.getsource` of the provided box renamed. `test_failing_check_has_the_native_shape` covers a failing verdict (headline, line count, indentation). The human read side by side with Colab is step C6.

**S-3. Verified by** a manual procedure (step C3) for the "without a reload" half. The write half is automated: `test_pass_is_recorded_in_progress_store` pins the key `learnfsd.v2.progress` and the shape `units["0-1-4"].exercises["0.1.4.a"] = {passedAt, via: "playground"}`. The read half is automated in the site repo by site plan Task 3 (`tests/learnfsd/progress.test.ts`, the `storage` listener in `useProgress.ts`).

**S-4. Verified by** `tests/playground/test_browser.py::test_stop_halts_while_true_and_page_stays_responsive`, run with `PLAYGROUND_TIMING=1` so the restart budget is the spec's 3 s (5 s otherwise). It asserts the main-thread ticker never stalls 500 ms, the status goes `running`, `loading`, `ready`, and a new Run completes. `test_stop_during_a_normal_run_keeps_the_rows` covers a Stop on a healthy run.

**S-5. Verified by** a manual procedure (step C4): Vivek's phone, one exercise, plus `python scripts/measure_phone.py` for the emulated number.

**S-6. Verified by** `tests/zero2fsd/test_margins.py` (Task 6): `python -m pytest tests/zero2fsd/test_margins.py -v`. The in-browser counterpart is the S-2 test above.

Spec section 11 (lines 174-185):

| # | Decision | Recommended | Alternatives and why not |
|---|---|---|---|
| S1 | Where it lives | Static folder in the course repo, served by the site | React feature of the site: dies at the GitHub Pages move. Separate repo: a third place to keep in sync |
| S2 | Run model | Producer at full speed + player | Pace the sim to real time: a lap takes 89 s to see and a verdict; slow devices stutter instead of merely waiting |
| S3 | Mac vs browser numbers | Margin test + borderline note; both official | Chase bit-identity: libm/numpy differ per platform, open-ended. "Colab is official": makes the playground a toy |
| S4 | Editor | CodeMirror 6 from CDN, textarea fallback | Monaco: 10× heavier. Textarea only: no highlighting for a beginner |
| S5 | Stop | terminate + restart | Interrupt buffer: needs cross-origin isolation headers GitHub Pages cannot set |
| S6 | Camera frames | Re-render the scrubbed frame on demand | Stream and keep every frame: 307 MB per lap |
| S7 | Progress | Additive `exercises` field, playground passes only | Record Colab passes too: Colab cannot write the browser's storage; a "mark done" button is self-reported and would lie on the map |
| S8 | Charting library | Loaded on first *Dashboard* press | Always loaded: +9 MB, +8.5 s on slow links, for a panel most runs never open |
| S9 | Phones | Supported, slower; measured in acceptance | Desktop-only: breaks "no kit, no install" |
| S10 | Week 1 kinds | `drive` and `stats` | All three now: `image` has no Week 1 exercise |

**S1. Verified by** `tests/playground/test_build.py` (Task 8: the dist folder, the manifest and its sha256) and, on the site, step C7 (`index.html` answers 200 after `sync:course`).
**S2. Verified by** `test_run_provided_steer_completes_and_draws`: the lap runs at least `REALTIME_FACTOR` (5) times faster than real time, which is the spec section 7 budget and implies the producer is not paced, and the step count equals the Mac's. On the phone, `measure_phone.py` prints the real-time factor.
**S3. Verified by** the Task 6 margin and borderline tests (`tests/zero2fsd/test_margins.py`) and the S-2 browser test.
**S4. Verified by** `test_run_provided_steer_completes_and_draws` (asserts the CodeMirror content element exists, so a broken CDN pin cannot hide behind the fallback) and `test_editor_import_blocked_falls_back_to_textarea`.
**S5. Verified by** the two Stop tests listed under S-4.
**S6. Verified by** `test_scrubbing_redraws_the_camera_frame` (the `#frame` pixels change when the scrubber moves, with the playhead paused).
**S7. Verified by** `test_pass_is_recorded_in_progress_store`, `test_failing_check_records_nothing` and `test_storage_blocked`; Colab passes are not recorded by construction, which needs no test.
**S8. Verified by** `test_dashboard_library_loads_only_on_demand` (no matplotlib request before Dashboard, one after).
**S9. Verified by** `test_no_horizontal_scroll_at_375` (all three kinds: drive, stats, image), `test_no_horizontal_scroll_with_a_failing_verdict`, `scripts/measure_phone.py`, and step C4 on the real phone.
**S10. Verified by** `test_stats_exercise_shows_native_values` (stats kind), `test_run_provided_steer_completes_and_draws` (drive kind), `test_image_exercise_paints_the_masks_and_hides_the_player` with its control `test_image_exercise_with_empty_masks_paints_no_overlay` (image kind), and the module-level guard `first_of_kind`, which fails collection if a kind loses its exercise.

> NOTE (S10 is out of date): the row above is copied verbatim and says Week 1 has `drive` and `stats` only. The registry (Task 5) gives 0.1.2.a kind `image`, so the checks above cover three kinds. Step C0b below corrects the spec's S10 row in the same PR.

Performance budgets (spec section 7, lines 146-153), the numbers C1, C4 and C5 compare against:

| Item | Budget | Measured |
|---|---|---|
| First load, wired | ≤ 3 s, ≤ 11 MB | 1.71 s, 10.36 MB without matplotlib |
| First load, 10 Mbit/s + 40 ms | ≤ 12 s | 9.84 s |
| Lap of `gentle` with camera | ≥ 5× real time on a Mac | 10.3× (205 steps/s) in Chrome, 11.6× in Safari |
| Same, CPU throttled 4× | ≥ 2× real time | 2.33× |
| Heap | ≤ 120 MB | 74.9 MB (worker, one lap) |
| Phone | not predicted; measured in acceptance (S-5) | — |

#### Acceptance run

Run once, on a wired Mac, after Tasks 1 to 13 are merged on the feature branch and the site plan (`2026-10-11-playground-site-seams.md`) is built. Record every number in the PR description (blocks C8 and C9).

**C0. Automated, default suite and build.** From the course repo root:

```bash
python scripts/build_course_bundle.py
python -m pytest tests -q
```

Read: all green, the browser module reported as deselected.

**C0b. Spec edit: S10 reflects the Week 1 image exercise (deviation P14).** The spec's S10 row still says `image` has no Week 1 exercise. In `docs/superpowers/specs/2026-10-11-playground-design.md` line 185, replace this row:

```text
| S10 | Week 1 kinds | `drive` and `stats` | All three now: `image` has no Week 1 exercise |
```

with this row:

```text
| S10 | Week 1 kinds | Week 1 kinds are `drive`, `stats` and one `image` exercise (0.1.2.a); the image preview is one frame beside the mask overlay, no player | Label 0.1.2.a "Colab only": contradicts S-2, which covers every Week 1 exercise |
```

Then commit it on its own:

```bash
git add docs/superpowers/specs/2026-10-11-playground-design.md
git commit -m "docs(spec): S10 reflects the Week 1 image exercise

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

Read: `git show --stat HEAD` lists one file with 1 insertion and 1 deletion. If Vivek strikes the image preview (P14), this row changes to say 0.1.2.a is Colab only.

**C1. Automated, browser and margin tests.**

```bash
PLAYGROUND_TIMING=1 python -m pytest tests/playground/test_browser.py -m browser -v -s
python -m pytest tests/zero2fsd/test_margins.py -v
python scripts/measure_phone.py
```

Read off: every case passed; the `load_s=` line (S-1, budget 3.0 s); the `restart_s=` line (S-4, budget 3.0 s); the margin tests (S-6); the `steps/s` line from `measure_phone.py` (S-5, emulated ceiling).

**C2. S-1 click path (manual).**

1. `cd /Users/vivek/Antigravity/vivekn/vivekn && npm run sync:course -- --from /Users/vivek/Antigravity/self-driving-ai-course && npm run dev`.
2. Open `http://localhost:3000/learnfsd/unit/0-1-4`.
3. Click the lecture's playground link for Exercise A.
4. Read off: the page opens on "Week 4 · Closing the loop · Exercise A", the road is `gentle`, and the status reads "Ready." with the car drawn. Record yes or no. The seconds come from C1.

**C3. S-3 pass appears without a reload (manual).**

1. Copy the passing body: `python -c "import sys; sys.path.insert(0, 'scripts'); from playground_browser import P_STEER; print(P_STEER)" | pbcopy`.
2. In tab A open `http://localhost:3000/learnfsd` (the course map). In tab B open `http://localhost:3000/learnfsd/unit/0-1-4`. Leave both open.
3. In tab B's playground, select all in the editor, paste, press Check.
4. Read off: the verdict says passed; tab B's exercise list shows the tick for Exercise A, and tab A's car box for unit 0.1.4 changes fill. Neither tab was reloaded. Record yes or no for each tab.

**C4. S-5 phone (manual, Vivek).**

1. On the Mac: `python scripts/build_course_bundle.py && python -m http.server 8000 --bind 0.0.0.0 --directory course/dist/playground`, and read the Mac's address with `ipconfig getifaddr en0`.
2. On the phone (same Wi-Fi): open `http://<that address>:8000/index.html?exercise=0.1.4.a`.
3. Replace the editor text with the single line `p_steer = steer_p` (the page's starter namespace already defines `steer_p`).
4. Start a stopwatch as you press Run. Stop it when the status reads "Completed gentle in ...".
5. Read off: wall seconds `W`. The phone's rate is `1778 / W` steps per second (a `gentle` lap is 1778 steps at `DT = 0.05`). Press Check: it must say passed. Try the 375 px layout by eye: no sideways scroll, buttons reachable.
6. Record: phone model, browser, `W`, steps per second, and whether Check passed. Slower than the Mac is fine; wrong is not.

**C5. Cold load at 10 Mbit/s (manual, Chrome DevTools).** Budget: 12 s and 11 MB.

1. Serve the dist: `python -m http.server 8000 --directory course/dist/playground`. Open Chrome with a fresh profile or an incognito window.
2. Open DevTools (Cmd+Option+I), Network tab. Tick "Disable cache".
3. Throttling dropdown, "Add...", then "Add profile". Name `10 Mbit 40 ms`, Download `10000` kbit/s, Upload `10000` kbit/s, Latency `40` ms. Save, and select that profile in the dropdown.
4. Load `http://localhost:8000/index.html?exercise=0.1.4.a`. Start a stopwatch at Enter. Stop it when the status reads "Ready."
5. Read the transferred size in the Network panel's bottom bar (the "transferred" figure).
6. Repeat three times. Record all three times and the size.

**C6. S-2 human read, Colab against the playground (manual).**

1. In the playground (`index.html?exercise=0.1.4.a`) paste the body from C3 step 1 and press Check. In DevTools Console run `copy([...document.querySelectorAll('#verdict li')].map(li => li.textContent).join('\n'))`, then `pbpaste > /tmp/playground.txt`.
2. Press "Open in Colab". Run the setup cells down to the Exercise A cell, replace its body with the same pasted text, run the check cell, and save the printed verdict to `/tmp/colab.txt`.
3. `diff /tmp/playground.txt /tmp/colab.txt`.
4. Read off: both say PASS and every verdict line matches. A framing line Colab prints around the details is not a difference; a changed number or word is. Repeat for Exercise B with the body from `python -c "import sys; sys.path.insert(0, 'scripts'); from playground_browser import pasteable; print(pasteable('0.1.4'))" | pbcopy` and the `my_car` cell.

**C7. URL probe (site).** With `npm run dev` running:

```bash
curl -sI http://localhost:3000/learnfsd/playground/ | head -1
curl -sI "http://localhost:3000/learnfsd/playground/index.html?exercise=0.1.4.a" | head -1
```

Expected: the first line is not `200` (the site leg saw 308, then 404 on the follow-up); the second is `HTTP/1.1 200 OK`. If the first ever answers 200, the page-URL NOTE under Task 12 can be relaxed; until then every link uses `index.html`.

**C8. PR description block.** Paste this into the PR and replace each `<...>` with the number or yes/no you read:

```text
Acceptance (Part C)
S-1  cold load to Ready: <load_s> s (budget 3.0); lecture click path: <yes/no>
S-2  check verdicts identical on the 7 references, 0.1.2.a (image) included: <7 of 7 tests passed>; Colab vs playground diff, Exercise A: <identical/differs>, Exercise B: <identical/differs>
S-3  pass shown without reload: unit page <yes/no>, course map <yes/no>
S-4  Stop restart: <restart_s> s (budget 3.0)
S-5  phone <model, browser>: lap <W> s = <steps/s> steps/s, Check <passed/failed>; emulated iPhone 12, best of 3 laps: <steps/s> steps/s
S-6  margin tests: <N passed>
10 Mbit/s 40 ms cold load: <t1>, <t2>, <t3> s (budget 12); transferred <MB> MB (budget 11)
Machine: <Mac model, macOS, Chrome version>
```

**C9. Budget read-offs (spec §7, manual, Chrome DevTools).** Two budgets no test asserts (deviation P16). Serve the dist as in C5 step 1 (`python -m http.server 8000 --directory course/dist/playground`) and use Chrome in a fresh profile or an incognito window.

1. **CPU 4x slowdown, budget at least 2x real time.** Open `http://localhost:8000/index.html?exercise=0.1.4.a`. In DevTools (Cmd+Option+I) open the Performance tab, click the gear icon (Capture settings), set the CPU dropdown to "4x slowdown", and wait for the status to read "Ready.". Replace the editor text with `p_steer = steer_p`. Start a stopwatch as you press Run and stop it when `#status` reads "Completed gentle in ...": that is the wall time `W`. In the Console run `Number(document.querySelector('#scrub').max) + 1` for the step count `N` (1778 for `gentle`). Record `N * 0.05 / W`. Bound: at least 2. Compare it with the unthrottled `real_time_x=` line from C1: a factor at or above that line means the throttle did not reach the worker, so record "not throttled", not a pass.
2. **Heap, budget at most 120 MB.** Set the CPU dropdown back to "No throttling" and press Run once more on the same page. Open the Memory tab, choose the dedicated worker (`worker.js`) in the target dropdown at the top of the panel (not the page), select "Heap snapshot" and press "Take snapshot". Read the total shown beside the snapshot in the left panel. Bound: at most 120 MB.
3. **Record** both numbers, with the `real_time_x=` line the Task 12 lap test printed in C1, in the PR description:

```text
Budgets (spec §7)
Lap of gentle with camera, Mac, no throttle: <real_time_x> x real time (budget at least 5; asserted by test_run_provided_steer_completes_and_draws)
CPU 4x slowdown: <N> steps in <W> s = <factor> x real time (budget at least 2)
Heap, worker after one lap: <MB> MB (budget at most 120)
```
