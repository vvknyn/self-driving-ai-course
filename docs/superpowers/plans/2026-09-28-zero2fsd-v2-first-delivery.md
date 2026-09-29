# Zero2FSD v2 — First Delivery Implementation Plan

> **For agentic workers:** Execute task-by-task with TDD (test red → implement → green → commit). Steps use `- [ ]`. Follow `~/.claude/skills/writing-lean-code` Core rules.

**Goal:** Ship the v2 course spine: the `zero2fsd` package (sim, boxes, grader), Loop 0 units 0.1.1–0.1.4, the course bundle, and the site's course map and unit pages, deployed to vivekn.xyz.

**Architecture:** The course repo is the single source of content. It produces `course/dist/course.json` plus figures. The site syncs that bundle at build time and renders it statically; learner progress lives in localStorage. Labs run in Colab against the pip-installed `zero2fsd` package, whose grader never substitutes the reference.

**Tech Stack:** Python ≥ 3.10 (numpy, pillow, matplotlib, pyyaml; dev: pytest, nbformat). Next.js 15 (app router, output standalone) with marked and KaTeX. Docker on Contabo behind Caddy.

**Spec:** `docs/superpowers/specs/2026-09-28-zero2fsd-v2-design.md` (course repo).

## Global Constraints
- Repos:
  - Course: `~/Antigravity/self-driving-ai-course`, branch `feat/zero2fsd-v2`.
  - Site: `~/Antigravity/vivekn/vivekn` (the nested clone at origin/main 238c98c; NOT the outer `~/Antigravity/vivekn`), branch `feat/learnfsd-v2-course`.
  - Both repos flow feature → PR → `main`, and neither has a staging branch.
- Python venv at `<course repo>/.venv` (gitignored): `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`. Run tests as `.venv/bin/python -m pytest tests/zero2fsd -q`.
- Runtime deps are ONLY numpy, pillow, matplotlib and pyyaml; no torch or opencv in `zero2fsd`.
- Sign conventions (never flip):
  - `LaneEstimate.offset_m > 0` means the car is LEFT of ego-lane centre.
  - `heading_rad > 0` means the car points LEFT of the lane direction.
  - `Action.steer_rad > 0` turns LEFT.
  - Truth and estimates are measured at the camera point (front axle).
- World:
  - Two lanes of 3.6 m; ego drives the RIGHT lane, so the ego-lane centre is at lateral −1.8 m from the road centreline.
  - Yellow dashed line on the road centreline; white solid line at lateral −3.6 m (right edge). No left white line.
- Palette (RGB, exact): sky (135,190,235), grass (70,130,60), road (90,90,95), yellow (230,200,40), white (240,240,240).
- Camera:
  - 320×180, HFOV 90°, 1.4 m high, pitched down 5°, at the front axle.
  - Renders ground 0.5–60 m ahead along the route; polygons are clipped at the near plane (forward ≥ 0.3 m).
- Dynamics:
  - Kinematic bicycle, wheelbase 2.7, dt 0.05.
  - Steer clip ±0.5 rad, steer rate ≤ 1.0 rad/s, accel clip [−6, 3].
  - Provided target speed 8 m/s.
- Episodes:
  - Complete when progress ≥ route length.
  - Terminate when |true offset| > 1.8 m.
  - `max_steps` defaults to ceil(3 × length / (8 × 0.05)).
- Driving score = round(100 × completion × clip(1 − mean|lat|/0.9, 0, 1)); off-lane means |lat| > 0.9 m.
- The grader NEVER imports `solutions/` and never substitutes a reference. `NotImplementedError` prints `✗ not implemented yet — write your code in the cell above` and fails.
- Claims policy: nothing about Tesla or any company without a citation; every URL in content was fetched and resolved by the author; every paper fact was checked against the paper.

## Review Focus
1. A learner function that raises NotImplementedError, raises another exception, or returns the wrong type must produce a readable red failure, never a traceback wall or a pass. The test lives in Task 5.
2. The untouched skeleton of every exercise must fail. A grader that passes on `return None` or on a constant is the v1 defect again. The test is verify_units (Task 6).
3. Perception on frames where a line is out of view (sharp curve, car drifting) must return `valid=False` rather than crash; the controller must handle `valid=False`. The test lives in Task 3 (provided perception) and in 0.1.3's checks.
4. Near-plane clipping: on `curvy` the road bends behind the camera, and the renderer must not draw garbage polygons. The test lives in Task 2: render every 10th step of `curvy` and assert that the sky region above the horizon holds only sky and grass colours.
5. localStorage missing or throwing (private window) must leave the site pages rendering and usable. The test lives in Task 10 (storage wrapper test).

---

## Part A — `zero2fsd` package (course repo)

### Task 1: Scaffolding, spec and plan commit
**Files:**
- Create: `pyproject.toml`, `zero2fsd/__init__.py`, `docs/superpowers/specs/2026-09-28-zero2fsd-v2-design.md` (copy of the spec), `docs/superpowers/plans/2026-09-28-zero2fsd-v2-first-delivery.md` (copy of this plan), `tests/zero2fsd/__init__.py`.
- Modify: `.gitignore` (add `.venv/`).

Steps:
- [ ] Create the branch.
- [ ] Write `pyproject.toml`: setuptools; `packages = find(include=["zero2fsd*"])`; `requires-python = ">=3.10"`; deps as above; `[project.optional-dependencies] dev = ["pytest", "nbformat"]`.
- [ ] Create the venv, install, and confirm that `python -c "import zero2fsd"` works.
- [ ] Commit.

### Task 2: Sim — world, scenarios, dynamics, camera, top-down
**Files:** `zero2fsd/sim/{__init__,world,scenarios,dynamics,camera,topdown}.py`, `tests/zero2fsd/test_sim.py`

**Interfaces (Produces):**
- `Road(centerline: np.ndarray[N,2], closed: bool)`:
  - `.length`
  - `.pose_at(s) -> (x, y, heading)`
  - `.project(x, y) -> (s, lateral)`: lateral is the signed distance from the centreline, left positive.
- `Scenario(name, road, start_s, target_speed=8.0)`; `make_scenario(name: str) -> Scenario` for `straight`, `gentle` and `curvy`. Unknown names raise ValueError listing the valid names.
- `BicycleState(x, y, yaw, speed, steer)` (the pose is the FRONT axle); `step(state, steer_cmd, accel, dt=0.05) -> BicycleState`.
- `Camera()` with defaults from Global Constraints:
  - `.render(road, state, noise_sigma=0.0, seed=0, return_labels=False)` returns `uint8[180,320,3]`, or `(frame, labels uint8[180,320])` with labels 0 = sky, 1 = grass, 2 = road, 3 = yellow, 4 = white.
  - `.pixel_to_ground(u, v) -> (forward_m, left_m)` in vehicle frame; NaN at or above the horizon; vectorised.
  - `.ground_to_pixel(forward_m, left_m) -> (u, v)`.
- `render_topdown(road, trajectories: list[np.ndarray[N,2]] = (), labels=(), ax=None) -> matplotlib Axes`.

**Scenario geometry:**
- `straight`: 200 m straight.
- `gentle`: closed stadium, two 100 m straights plus two semicircles of radius 60 m (one lap).
- `curvy`: 30 m straight, then alternating ±90° arcs of radius 20 m (≥ 6 arcs).

**Tests:**
- `pixel_to_ground(ground_to_pixel(p)) ≈ p` round-trip within 1e-6 over random ground points 1–50 m ahead.
- The horizon row is ≈ 76 ± 1.
- Rendering at lane centre on `straight`: the yellow label pixels' mean column < 160 < the white label pixels' mean column.
- Yellow is dashed along its length (labels show gaps).
- Driving the centreline pose sequence of `curvy` in 1 m steps and rendering every 10th leaves no road, yellow or white labels above row 70 (Review Focus 4).
- `step` with steer 0 keeps yaw constant; steer clip and rate limit are enforced; accel is clipped.
- `project(pose_at(s) + normal*d) == (s, d)` within 1e-6.
- Unknown scenario → ValueError.
- A rendering benchmark test asserts ≥ 100 renders/s. Mark it `@pytest.mark.slow` and report the measured rate.

### Task 3: Car — types, provided boxes, car, runner
**Files:** `zero2fsd/car/{__init__,types,provided,car,runner}.py`, `tests/zero2fsd/test_car.py`

**Interfaces:**
- Frozen dataclasses:
  - `Observation(frame, speed, t)`
  - `LaneEstimate(offset_m, heading_rad, valid)`
  - `Action(steer_rad, accel)`
- `provided`:
  - `paint_masks(frame) -> (yellow_bool, white_bool)`, a colour-distance threshold that is robust to noise σ ≤ 8.
  - `estimate_lane(obs) -> LaneEstimate`:
    - Use two image rows.
    - Back-project the yellow and white mean columns with `pixel_to_ground`.
    - Lane-centre left position = mean of the two lines.
    - offset = −centre_left.
    - heading = −atan2(Δleft, Δforward).
    - `valid=False` if either line is missing in a row.
  - `hold_speed(speed, target=8.0) -> accel` = clip(0.5 × (target − speed)).
  - `steer_p(est, k_off=0.35, k_head=1.2) -> float` = −(k_off × offset + k_head × heading) clipped to ±0.5; returns 0.0 when not valid. Tune the gains so the Task 3 property tests pass.
- `Car(perception=None, controller=None)`:
  - None → provided.
  - Accepts objects with `.estimate` / `.act` or plain callables.
  - A controller returning a float → `Action(steer, hold_speed(obs.speed))`.
  - `car.step(obs) -> (LaneEstimate, Action)`; `car.uses_provided() -> list[str]` names the slots (`perception`, `controller`) still on provided boxes.
- `run(car, scenario, max_steps=None, seed=0, noise_sigma=0.0) -> Telemetry`.
- `Telemetry` has numpy arrays `t, x, y, yaw, speed, steer, accel, lat, head_err, est_offset, est_heading, est_valid, progress_m, off_road`, plus `scenario: str`, `completed: bool` and `route_length: float`.
- Exceptions raised by learner boxes propagate (the grader catches them).

**Tests (property):**
- `run(Car(), "gentle").completed is True` and mean|lat| < 0.5.
- `run(Car(), "curvy").completed is False` (leaves the lane).
- On `straight` at offsets {−0.8, 0, 0.8}: the provided estimate is within 0.15 m and 0.03 rad of truth.
- Provided perception returns valid=False (no exception) on a frame that is all grass.
- A float-returning controller is adapted.
- A box raising ValueError propagates out of `run`.

### Task 4: Score and dashboard
**Files:** `zero2fsd/score.py`, `zero2fsd/dashboard.py`, `tests/zero2fsd/test_score.py`

**Interfaces:**
- `metrics(tel) -> dict(completion, mean_abs_lat, max_abs_lat, steps_off_lane, rms_steer_rate)`.
- `driving_score(tel) -> int`.
- `metrics_table(*tels, labels) -> str`, a fixed-width text table.
- `dashboard(*tels, labels=None) -> matplotlib.figure.Figure`: 2×2 panels of top-down trajectories, lat vs t, est vs true offset, and a metrics table. Uses the Agg-safe API; the caller shows it.

**Tests:**
- Hand-built telemetry → exact metrics and score.
- completion is progress / length capped at 1.
- dashboard returns a Figure with 4 axes.

### Task 5: Grader
**Files:** `zero2fsd/grade/{__init__,_report}.py`, `zero2fsd/grade/units/{__init__,u0_1_1,u0_1_2,u0_1_3,u0_1_4}.py`, `tests/zero2fsd/test_grade.py`

**Interfaces:**
- `check(exercise_id: str, fn_or_box, show=True) -> Result`.
- `Result(passed: bool, details: list[str])`.
- The registry maps exercise id → check function via each unit module's `EXERCISES: dict[str, Callable[[Any, bool], Result]]`. It is built by importing the modules in `grade/units/`; an unknown id → ValueError listing the known ids.
- The runner wraps the call: `NotImplementedError` → the red not-implemented message; any other exception → a red "your code raised <Type>: <msg>" plus the input that triggered it.
- Each failure detail says: what was checked, the input, expected, got, and a hint naming the lecture section.
- Prints a one-line green ✓ or red ✗ summary (ANSI colours).

**Exercise checks:** expected values are computed independently inside the grader, never by calling provided/solution code on the same path.
- `0.1.1.a` `lane_error_stats(lat_err: np.ndarray) -> dict` with keys `mean_abs`, `max_abs`, `rms`. Uses ≥ 5 random seeded arrays incl. negatives; compared with a python-loop computation; tolerance 1e-9.
- `0.1.1.b` `steps_off_lane(lat_err, threshold=0.9) -> int`: counts |e| > threshold (strict). Includes an array with values exactly at 0.9 and a custom threshold.
- `0.1.2.a` `paint_masks(frame) -> (yellow, white)` bool[H,W]:
  - Frames: 6 poses across scenarios rendered with noise σ=6.
  - Pass if IoU ≥ 0.85 against the renderer labels for BOTH masks on every frame.
  - The shape/dtype is checked first with a clear message.
- `0.1.3.a` `estimate_lane(obs) -> LaneEstimate`:
  - Sample 40 seeded poses on `straight` and `gentle` with offset ∈ [−1.0, 1.0] and heading ∈ [−0.15, 0.15].
  - Pass if ≥ 90% have |Δoffset| < 0.25 m and |Δheading| < 0.05 rad with valid=True.
  - Gate (`0.1.3.b`, graded name `estimate_lane`): `run(Car(perception=fn), "gentle")` must satisfy completed and mean|lat| < 0.5. Print the metrics table; show the dashboard if `show`.
- `0.1.4.a` `p_steer(est, k_off, k_head) -> float`:
  - Checks sign (left offset → negative steer), linearity, clipping to ±0.5, and 0.0 on invalid.
  - Random cases against the formula `−(k_off×offset + k_head×heading)` clipped.
- `0.1.4.b` graded object `my_car` (a `Car`):
  - Gate: `gentle` completed and mean|lat| < 0.5.
  - Then runs `curvy` and prints its metrics as "your cliffhanger" (not gated).
  - Rejects a `my_car` built with no learner perception or no learner controller (`car.uses_provided()` returns the names of provided boxes; the gate requires an empty list).

**Tests (in `test_grade.py`):**
- For every exercise:
  - A NotImplementedError fn fails with the not-implemented message.
  - A fn returning None fails.
  - A fn raising TypeError fails with the "your code raised" message.
  - Unknown id → ValueError.
- `0.1.4.b` with `Car()` (all provided) fails.

### Task 6: Scripts — build_labs, verify_units, make_figures, bundle
**Files:** `scripts/build_labs.py`, `scripts/verify_units.py`, `scripts/make_figures.py`, `scripts/build_course_bundle.py`, `course/syllabus.yaml`, `solutions/course/0.1.{1,2,3,4}/solution.py`, `tests/zero2fsd/{test_build_labs,test_verify_units,test_bundle}.py`

- `build_labs.py [unit_dir...]`:
  - Parses percent-format `lab.py`: `# %%` code, `# %% [markdown]` markdown (strip a leading `# `), and `# %% tags=["exercise:0.1.1.a"]` or `tags=["setup"]`.
  - Writes `lab.ipynb` with nbformat v4 (kernel python3, cell metadata `tags`).
  - `--check` exits 1 if any committed `lab.ipynb` is stale.
- `solutions/course/<id>/solution.py` defines exactly the graded names from that unit's `unit.yaml` exercises.
  - Reference `p_steer` gains plus reference `estimate_lane` must pass the 0.1.4.b gate on `gentle` AND fail `curvy`; assert both in a test.
- `verify_units.py`, for each ready unit in `syllabus.yaml` and each exercise in `unit.yaml`:
  - skeleton = exec, in one namespace and in order, the notebook code cells tagged `setup` or `exercise:*` up to and including the LAST cell tagged with this exercise's id (an exercise may span several cells, e.g. 0.1.4 has a "paste your estimate_lane from 0.1.3" cell); take `ns[name]`.
  - `check(id, skeleton, show=False)` must be False AND `check(id, solution, show=False)` must be True.
  - Exit 1 listing each violation.
  - Uses the matplotlib Agg backend; bootstrap cells are not tagged `setup` (they pip-install).
- `make_figures.py [unit_dir...]`: for each unit dir with `figures.py`, imports it and calls `make(out_dir=unit_dir/"figures")`. Output must be deterministic (seeded).
- `course/syllabus.yaml`:
  - Structure: `loops: [{id, title, summary, modules: [{id, title, summary, planned_units: int, paper?: {title, url}, units?: [{id, folder, status: ready|planned}]}]}]`.
  - Content: all loops and modules from spec §3.
  - Loop 0 lists units 0.1.1–0.1.4 with `status: planned` until Part C flips them to `ready`.
- `build_course_bundle.py` → `course/dist/course.json` + `course/dist/figures/<slug>/*.png` (schema below). Fails loudly if a `ready` unit is missing a required file or a figure referenced by the lecture.
- `tests/zero2fsd/test_verify_units.py` runs verify on all ready units; it is marked slow if > 30 s.

**`course.json` schema (the ONLY site interface; version 1):**
```json
{
  "version": 1,
  "repo": "vvknyn/self-driving-ai-course",
  "loops": [{"id": "0", "title": "", "summary": "", "modules": [
    {"id": "0.1", "title": "", "summary": "", "planned_units": 4, "paper": {"title": "", "url": ""},
     "units": [{"id": "0.1.1", "slug": "0-1-1", "title": "", "status": "ready", "minutes": 210}]}]}],
  "units": {"0-1-1": {
    "id": "0.1.1", "slug": "0-1-1", "title": "", "minutes": 210, "box": "",
    "outcomes": [{"can": "", "proof": ""}],
    "prerequisites": ["0-1-0"],
    "go_deeper": [{"title": "", "url": ""}],
    "paper": {"title": "", "url": "", "questions": [""]},
    "tutor_prompt": "",
    "lecture_md": "markdown; figure refs rewritten to /learnfsd-course/figures/0-1-1/<file>",
    "colab_url": "https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/course/<folder>/lab.ipynb",
    "quiz": [{"id": "", "type": "mcq", "prompt": "", "choices": [""], "answer": 0, "explain": "", "section": "heading-slug"},
             {"id": "", "type": "numeric", "prompt": "", "answer": 1.5, "tolerance": 0.05, "explain": "", "section": ""}],
    "cards": [{"id": "", "front": "", "back": ""}]
  }},
  "figures": ["0-1-1/lat_error.png"]
}
```
- `paper` is null when absent. `units` holds only `ready` units. A module's `units` may be empty for later loops (the site shows `planned_units`).
- `section` = GitHub-style heading slug: lowercase, spaces → `-`, drop characters other than `[a-z0-9-]`.

### Task 7: v1 cleanup
- [ ] Grep the site repo (`~/Antigravity/vivekn/vivekn`, including `scripts/` and `src/data/`) for `01_driving_perception_gym` and `02_camera_geometry_and_ipm`. If either is referenced, do NOT delete it; report it instead.
- [ ] If there are no references, `git rm` the two stubs and fix `notebooks/README.md`.
- [ ] Add a top section to the root `README.md`: "v2 course (start here)" (link to `course/` and vivekn.xyz/learnfsd) plus "v1 reference library" labels for `notebooks/`, `modules/` and `docs/`.

## Part C — Loop 0 content (course repo; after Part A)

### Task 8: Units 0.1.1–0.1.4 (one author, one pass, coherent across units)
Per unit folder `course/0.1.<n>-<slug>/`:
- `unit.yaml` with the fields from spec §7.1 plus `exercises: [{id, name}]`.
- `lecture.md` follows spec §4 part 3: ONE big idea, intuition → picture → math → worked example → common misconception, figures, and a "Go deeper" canonical source. Write in second person, around 1,200–2,000 words, with `##` headings whose slugs the quiz `section` fields use.
- `lab.py` covers parts 2, 4, 5 and 6: a hook with a written prediction, worked → faded problems, the lab (open the box), and break-it-fix-it.
  - Cell 1 is the Colab bootstrap (not tagged).
  - Setup cells are tagged `setup`.
  - Each exercise cell is tagged `exercise:<id>`; its skeleton body raises `NotImplementedError`.
  - After each exercise, a `check(...)` cell.
- Also in each folder: `quiz.yaml` (5–8 questions, each mapped to an outcome), `cards.yaml`, `figures.py` and the generated `figures/`.
- Update the `solutions/` files as needed.
- Flip the unit to `ready` in `syllabus.yaml`.

Units:
- 0.1.1 Meet your car.
- 0.1.2 Images are arrays.
- 0.1.3 Your first perception box.
- 0.1.4 Closing the loop. Includes the ALVINN paper questions, verified against the paper.

Done when:
- `build_labs.py --check`, `make_figures.py`, `verify_units.py`, `build_course_bundle.py` and pytest all exit 0.
- Every notebook executes top-to-bottom locally with the skeletons: check cells print red; there are no uncaught exceptions except in the learner exercises; the grader catches those.

## Part B — Site (site repo; parallel with A/C against a fixture)

### Task 9: Sync
- `scripts/sync-course-bundle.mjs` (`npm run sync:course`) mirrors the style of `scripts/sync-course-notebooks.mjs`:
  - Default source: GitHub raw `main`. `--from <course repo dir>` reads `course/dist/`.
  - Writes `src/data/learnfsd-course.json` and `public/learnfsd-course/figures/<slug>/*`, following the `figures` list.
- Commit a small hand-written fixture that conforms to the schema until the real bundle exists.

### Task 10: Course lib, review scheduler, storage
- `src/lib/learnfsd-course.ts`: types for the schema plus accessors `getCourse()`, `getUnit(slug)`, `readyUnits()`.
- `src/lib/learnfsd-review.ts` (pure):
  - Leitner boxes 1..5 with intervals [1, 3, 7, 21, 60] days.
  - `enrollUnit(deck, unit, today)`: cards = quiz questions + cards.yaml, keyed `<slug>:<id>`, box 1, due today + 1 day.
  - `grade(deck, key, correct, today)`: correct → next box (max 5) with due = today + interval[newBox]; wrong → box 1, due today + 1.
  - `dueCards(deck, today, limit=5)`: oldest due first.
- `src/lib/learnfsd-progress.ts`:
  - localStorage key `learnfsd.v2.progress`, shape `{version: 1, units: {[slug]: {quiz?: {best: number, passedAt?: string}, teachBack?: string, logs: [{at, minutes, confusing, confidence}]}}, deck: {[key]: {box, due}}}`.
  - Every access goes through try/catch and falls back to an in-memory state (Review Focus 5).
  - `exportProgress()` downloads JSON.
- Tests use `node:test`, run via `node --experimental-strip-types --test` (Node ≥ 22.6), on the scheduler and on storage with a throwing localStorage. Add the script `npm run test:learnfsd`.

### Task 11: `/learnfsd` course map
- Above the existing v1 portal, add a course map:
  - Loops → modules → units, with status ready / coming soon (showing `planned_units`) and per-unit progress (quiz passed ✓).
  - A "Start here" link to the first ready unit whose quiz isn't passed.
  - An "Export my progress" button.
- The existing v1 chapter library stays below under a heading like "v1 chapter library (reference)". Existing `/learnfsd/[id]` URLs stay unchanged.
- Match the site's existing design language and components; the map must work at 375 px width with no horizontal scroll.

### Task 12: `/learnfsd/unit/[slug]`
- Static via `generateStaticParams` from `readyUnits()`; unknown slug → `notFound()`.
- Server-rendered:
  - A unit card: outcomes with their proofs, prerequisites (links), minutes, box.
  - The lecture via the existing `src/lib/learnfsd-markdown.ts`. Headings must carry `id` = the same slug rule; add this if missing, without changing how existing chapters render.
  - Go-deeper links and the paper block.
  - An "Open lab in Colab" button (`colab_url`).
- Client islands:
  - WarmupReview: up to 5 due cards across units; reveal answer → "got it" / "missed it" → `grade`.
  - Quiz:
    - MCQ radio plus numeric input with tolerance; submit shows the score.
    - Wrong answers show `explain` plus a link to `#section`.
    - Pass ≥ 80% records `quiz.passedAt` and enrolls the unit's cards.
  - TeachBack: textarea (saved) plus "Copy tutor prompt", which copies `tutor_prompt` + "\n\nMy explanation:\n" + the text.
  - PilotLog: minutes, most confusing moment, confidence 1–5 → append to logs.
- Must work at 375 px width.

### Task 13: Site verification
- `npx tsc --noEmit`, `npm run test:learnfsd` and `npm run build` all exit 0.
- `npm start` locally, then curl `/learnfsd`, `/learnfsd/00`, `/learnfsd/00b` and `/learnfsd/unit/0-1-1`: all return 200 with the expected text.
- Screenshots at 1280 px and 375 px of the map and of a unit page (quiz answered) go into the scratchpad.

## Part D — Integration, merge, deploy
### Task 14: Integrate
- Run `sync:course --from` the course repo, rebuild, re-verify (Task 13), commit.
- Push both branches; open the PRs (feature → main) and merge. Course first, so the Colab bootstrap `@main` resolves.
### Task 15: Deploy
- Kit = `git archive --prefix=vivekn/ -o vivekn-source-<short>.tar.gz <merged main sha>` plus `remote-deploy.sh`, cloned from `/private/tmp/vivekn-deploy-238c98c/remote-deploy.sh` with a new SHA. Canary and public checks are extended:
  - `/learnfsd` contains the course-map marker text.
  - `/learnfsd/unit/0-1-1` returns 200 and contains "Meet your car".
  - The v1 checks are kept.
- scp both to the box `/tmp/`, then `ssh subspace@194.163.147.154 'bash /tmp/remote-deploy.sh'`. If the auto-mode classifier refuses, hand Vivek ONE `!` command.
- Afterwards: pixels verified on prod, with screenshots of the map and unit pages at 1280 and 375 px.
