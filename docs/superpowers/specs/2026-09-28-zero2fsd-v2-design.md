# Zero2FSD v2 — Course Redesign Spec

Date: 2026-09-28 · Author: Claude (design delegated by Vivek: "take a pass at all sections… I will leave it to you… once done, deploy")
Status: approved-by-delegation. Supersedes the v1 chapter structure (v1 stays online as a reference library until each chapter is superseded).

## 1. Goal and learner

- **Outcome:** take a learner from zero to PhD-level in autonomous driving: by the end they can read a new CoRL/CVPR/ICRA driving paper, reproduce it at small scale, extend it, and write it up.
- **Learner #1 (pilot):** Vivek. Rusty math, no ML, fluent programmer. 3–5 h/week. Mac (Apple Silicon) + free Google Colab only. Course must later be shareable publicly.
- **Pace:** ~144 one-week units ≈ 3 years. Content is authored just-in-time, staying ~4 units ahead of the pilot learner.

### Audit findings this design fixes (v1, measured 2026-09-28)
| v1 defect | v2 fix |
|---|---|
| Exercise checks silently substitute the reference solution and print ✅ (`notebooks/00_driving_ml_gym.ipynb` cell 71) | Grader never substitutes; untouched skeleton must fail (automated meta-check, §6.5) |
| Core algorithms are black-box calls into `modules/*.py` | Each module's lab *is* writing the box; your code runs in your car |
| No objectives, prerequisites, quizzes, projects | Unit template §4 |
| Chapter number ≠ notebook number ≠ module number | One id scheme `L.M.U` everywhere (§7) |
| Lecture prose (course_docs) and notebooks written separately, contradict, overclaim Tesla | One folder per unit authored in one pass; claims policy §4.3 |
| All data synthetic and presented as realistic | Course-owned sim is openly a toy; real frames (nuScenes-mini) enter in Loop 2 |

## 2. Pedagogical architecture — "Build your own self-driving car, one box at a time"

Week 1 the learner gets a complete car that drives **badly** in a simulator. Its stack is made of *provided boxes*. Every module opens one box: learn the math it needs just-in-time, write your own version, swap it in, watch the driving score move. By the end every box is the learner's own code.

Principles (from the learning-science literature; cited as design rationale):
1. **Whole game first** (Perkins, *Making Learning Whole*; fast.ai top-down) — Loop 0 plays a junior version of the whole game.
2. **Spiral curriculum** (Bruner) — three loops around the full stack, each deeper.
3. **Just-in-time math** (cognitive load theory, Sweller) — math arrives when a box needs it.
4. **Retrieval + spacing** (Roediger & Karpicke) — weekly review cards; old boxes stay in the car and keep being exercised.
5. **Mastery gates** (Bloom) — a box is "done" when it passes its tests and meets the unit's driving-score gate.
6. **Productive failure** (Kapur) — every unit opens with the car failing and a written prediction.
7. **Fading scaffolding** — early labs give most of the code; by Loop 3 the learner writes the tests too.
8. **Research from week 1** — one paper per module with guided questions, starting with classics.

## 3. Syllabus

Ids: `L.M.U` = loop.module.unit (e.g. `1.3.2`). Loop 0 is specified to unit level; later loops at module level and are refined as the pilot approaches them.

### Loop 0 — The whole game (4 units, ~1 month)
| Unit | Title | Lab (learner writes) | Gate |
|---|---|---|---|
| 0.1.1 | Meet your car | `lane_error_stats(lat_err)` → mean |e|, max |e|, RMS; `steps_off_lane` | tests pass |
| 0.1.2 | Images are arrays | `paint_masks(frame)` → yellow / white boolean masks | tests pass |
| 0.1.3 | Your first perception box | `estimate_lane(obs)` → `LaneEstimate` from masks; swap into car | car completes `gentle` with your perception, mean |lat| < 0.5 m |
| 0.1.4 | Closing the loop | `p_steer(est, k_off, k_head)` controller; tune gains | all-yours car (your perception + your controller) completes `gentle`; you predict and observe its failure on `curvy` (cliffhanger → lookahead in 1.3) |
Module paper: Pomerleau, *ALVINN: An Autonomous Land Vehicle in a Neural Network* (NIPS 1988) — guided questions (in 0.1.4).
Python/NumPy fundamentals are taught inside these labs (arrays, slicing, boolean masks, dtypes, vectorisation).

### Loop 1 — Classical car (~48 units, ~12 months). Every classical box becomes yours.
| Module | Units | Box opened | Just-in-time math |
|---|---|---|---|
| 1.1 Frames & transforms | 5 | coordinate transforms used everywhere | vectors, rotation matrices, homogeneous transforms |
| 1.2 Longitudinal control & vehicle motion | 6 | speed controller; the sim's bicycle model | derivatives, Euler integration, PID |
| 1.3 Lateral control | 5 | steering controller (pure pursuit, Stanley) | geometry, stability intuition. Paper: Thrun et al., *Stanley* (2006) |
| 1.4 Camera geometry | 6 | the sim's camera renderer; IPM | pinhole model, intrinsics/extrinsics, projection |
| 1.5 Classical lane perception | 6 | perception (lanes) | least squares, RANSAC, polynomial fits |
| 1.6 State estimation | 7 | localisation from noisy odometry/GNSS | probability, Gaussians, Bayes, KF, EKF. Paper: Kalman (1960) |
| 1.7 Multi-object tracking | 5 | tracker | data association, Hungarian algorithm |
| 1.8 Planning | 6 | planner (lattice, A*, speed profiles, overtaking a slow car) | graph search, cost functions |
| 1.9 Loop 1 capstone | 2 | — | all-yours classical car beats the provided car on the full scenario suite |

### Loop 2 — Learned car (~52 units, ~12 months). Learned boxes must beat *your own* classical boxes.
2.1 Autograd from scratch (5, reuses v1 00b) · 2.2 ML fundamentals (5, reuses v1 ML gym) · 2.3 Neural nets & CNNs in PyTorch (5) · 2.4 Learned lane perception: sim → real nuScenes-mini frames, the domain gap (5) · 2.5 Object detection on real frames (6) · 2.6 Multi-task nets (3, reuses v1 HydraNet) · 2.7 Attention & transformers from scratch (5) · 2.8 BEV perception & point clouds (6, reuses v1 BEV) · 2.9 Occupancy & temporal fusion (4, reuses v1 occupancy) · 2.10 Motion forecasting (4) · 2.11 Imitation-learned planner, covariate shift, DAgger; capstone (4)

### Loop 3 — Research car (~40 units, ~9 months + thesis)
3.1 Reading & reproducing papers (3) · 3.2 RL for driving in the course sim (6) · 3.3 End-to-end driving (5) · 3.4 World models & generative simulation (5) · 3.5 Vision-language-action models (4) · 3.6 Data engines & the long tail (4, reuses v1 ch09) · 3.7 Safety, verification, scenario-based evaluation (5) · 3.8 Thesis: reproduce + extend + write-up (8+)
Specific papers for Loop 3 are chosen when the pilot arrives (field moves fast).

Compute constraint (Mac + free Colab): real data = small subsets (nuScenes-mini class); reproductions are small-scale; every long training lab checkpoints to survive Colab disconnects.

## 4. Unit template (one week, ~3.5–4 h, 3–4 sessions)

| # | Part | Time | Where | Content |
|---|---|---|---|---|
| 0 | Unit card | 2 min | site | 1–3 can-do outcomes each with its proof; prerequisites (unit links); time; box touched |
| 1 | Warm-up review | 10 min | site | ≤5 due review cards from earlier units (§6.6) |
| 2 | Hook | 10 min | lab | run the car, watch a specific failure; **write a prediction first** |
| 3 | Lecture | 25 min | site | ONE big idea: intuition → picture → math → worked example → common misconception; figures required; "Go deeper" canonical source |
| 4 | Worked → faded | 30 min | lab | one fully worked example, then 2–3 problems with less given each time |
| 5 | Lab: open the box | 60–90 min | lab | implement against a fixed interface; tests red→green; swap into car; dashboard vs provided |
| 6 | Break it, fix it | 20 min | lab | a deliberately broken variant to diagnose |
| 7 | Checkpoint quiz | 15 min | site | 5–8 questions (concept / calculation / predict-the-output / transfer); pass ≥80%; wrong answers link to the lecture section; every question becomes a review card |
| 8 | Teach-back | 10 min | site | explain in 5 sentences; copy a Socratic tutor prompt into Claude |
| 9 | Pilot log | 2 min | site | minutes spent, most confusing moment, confidence 1–5 |
Module level (every 4–6 units): module project with rubric + one paper with guided questions.

### 4.1 Split
**Read on the site, do in Colab.** Site: parts 0, 1, 3, 7, 8, 9. Lab notebook: parts 2, 4, 5, 6.

### 4.2 Quality bar (unit Definition of Done — checked before the pilot reaches the unit)
1. Lab fails on the untouched skeleton and passes with the reference solution (automated, §6.5).
2. Every number in the lecture is computed by the course code (figure/scripts) or cited.
3. Math cross-checked against the "Go deeper" source.
4. Every quiz question maps to a can-do outcome; outcomes without a question are cut.
5. Lecture figures are generated from the course sim by a script, never hand-invented.

### 4.3 Voice and claims policy
Second person, one concrete driving scene per lecture, no emoji walls. Any statement about Tesla or any company is cited or labelled as speculation. The sim is described as what it is: a flat-world teaching simulator.

## 5. The course simulator (course-owned, numpy + Pillow)

Decision (2026-09-28 spike, `scratchpad/simspike/REPORT.md`): MetaDrive camera renders washed-out on Mac; highway-env has no camera; gym-duckietown is unmaintained, pins numpy≤1.23, differential-drive. → Build a small transparent sim; it becomes a box the learner opens in Loop 1 (renderer = camera geometry, dynamics = bicycle model).

- **World:** flat ground. Road = centerline polyline (arc-length parameterised), two lanes of 3.6 m; ego drives the right lane. Markings: yellow dashed centre line (left of ego lane), white solid edge line (right). Grass outside the road.
- **Ego dynamics:** kinematic bicycle, wheelbase 2.7 m, dt = 0.05 s, steer clip ±0.5 rad, steer-rate limit, accel clip [-6, 3] m/s².
- **Camera:** pinhole 320×180, horizontal FOV 90°, mounted 1.4 m high at the car's front axle, pitched down 5°. Renders sky, grass, road surface, and markings by projecting ground-plane quads sampled along the road ahead (0.5–60 m) and filling polygons with Pillow. Optional seeded pixel noise (off in Loop 0).
- **Top-down render** for dashboards.
- **Ground truth** in telemetry: lateral offset to ego-lane centre, heading error, progress along route, off-road flag.
- **Scenarios** (seeded, deterministic): `straight` (200 m), `gentle` (closed loop, radius ≥ 60 m), `curvy` (S-curves, radius ~20 m). Scenario property checked in tests: the provided Loop 0 car completes `gentle` and leaves the road on `curvy`.
- Performance target: ≥100 steps/s with camera on in Colab CPU (measured, not assumed).

## 6. Labs, boxes, scoring, grading

### 6.1 Box interfaces (`zero2fsd.car`), added only when a module needs them
- `Observation(frame: uint8[H,W,3], speed: float, t: float)`
- `LaneEstimate(offset_m: float, heading_rad: float, valid: bool)` — offset > 0 means the car is **left** of lane centre; heading > 0 means the car points **left** of the lane direction.
- `Action(steer_rad: float, accel: float)` — steer > 0 turns left.
- `Perception.estimate(obs) -> LaneEstimate`; `Controller.act(est, obs) -> Action`. Plain functions are accepted and adapted. A controller that returns a bare float is a steering angle; the car pairs it with the provided speed controller (longitudinal control is its own box, opened in 1.2).
- Truth and estimates are measured at the camera point (front axle), so an estimate and its ground truth describe the same quantity.
- `Car(perception=None, controller=None)` → provided boxes where None.

### 6.2 Provided boxes (Loop 0)
Provided perception: colour threshold → lane centre in pixel rows → metres via flat-ground back-projection → heading from two rows. Provided controller: P on offset + heading, constant target speed 8 m/s (no lookahead — deliberately fails `curvy`).

### 6.3 Runner and telemetry
`run(car, scenario, max_steps=None, seed=0) -> Telemetry` — numpy arrays per step: t, x, y, yaw, speed, steer, accel, true lateral offset, true heading error, estimated offset/heading, progress_m, off_road; episode ends when complete or |true offset| > 1.8 m.

### 6.4 Driving score
Shown as a metrics table (completion %, mean |lat|, max |lat|, steps off lane (|lat| > 0.9 m), RMS steer rate) plus a headline **driving score = round(100 × completion × clip(1 − mean|lat| / 0.9, 0, 1))**. Gates use explicit metric thresholds stated in the unit card, never the headline alone. `dashboard(*telemetries, labels=...)` plots top-down trajectories, lateral error vs time, estimated vs true offset, and the metrics table.

### 6.5 Grader contract (`zero2fsd.grade.check(exercise_id, fn_or_box)`)
- Each lab has one or more exercises with ids `L.M.U.x` (e.g. `0.1.1.a`). `unit.yaml` declares them as `exercises: [{id, name}]`, where `name` is the notebook global that gets graded. The exercise cell is tagged `exercise:<id>` in the notebook.
- Runs that exercise's checks (visible in `zero2fsd/grade/units/`); randomised property checks plus comparison against an independently computed expected value.
- **Never substitutes the reference.** `NotImplementedError` → red "✗ not implemented yet — write your code in the cell above". Each failure prints: what was checked, input, expected, got, and a hint pointing to the lecture section.
- Box labs: after unit checks pass, run the gate scenario(s), print the metrics, show the dashboard, report gate pass/fail.
- Returns a `Result(passed: bool, details)`; prints a one-line summary.
- **Meta-check `scripts/verify_units.py`** (and a pytest): for every exercise, grading the untouched skeleton must fail and grading `solutions/` must pass. The skeleton comes from executing the notebook's cells tagged `setup` plus the exercise cells up to that exercise, in order, and taking the declared `name`. This is the automated Definition-of-Done item 1.

### 6.6 Quiz and review cards
- `quiz.yaml`: list of `{id, type: mcq|numeric, prompt, choices?, answer, tolerance?, explain, section}` (`section` = lecture heading anchor for "review this").
- `cards.yaml`: list of `{id, front, back}`; quiz questions also become cards.
- Scheduler (site, per learner): Leitner boxes with intervals 1, 3, 7, 21, 60 days; correct → next box, wrong → box 1; a unit's cards enter the deck when its quiz is passed; warm-up shows ≤5 due cards, oldest due first.
- Stored in the browser (localStorage); an "export my progress" button downloads JSON (progress, review deck, pilot logs). Cross-device sync is out of scope for v2.0.

## 7. Repository layout and delivery

### 7.1 Course repo (`vvknyn/self-driving-ai-course`) — single source of truth for content
```
pyproject.toml                  # package "zero2fsd": numpy, pillow, matplotlib, pyyaml
zero2fsd/
  sim/ (world.py, dynamics.py, camera.py, topdown.py, scenarios.py)
  car/ (types.py, provided.py, car.py, runner.py)
  score.py  dashboard.py
  grade/ (__init__.py, units/u0_1_1.py …)
course/
  syllabus.yaml                 # loops → modules → units (id, title, status: ready|planned)
  0.1.1-meet-your-car/ unit.yaml lecture.md lab.ipynb quiz.yaml cards.yaml figures/
  …
solutions/course/<unit-id>/     # reference answers, never imported by labs
scripts/ make_figures.py  verify_units.py  build_course_bundle.py
tests/zero2fsd/                 # package tests + verify_units test
```
- `unit.yaml`: `{id, title, minutes, outcomes: [{can, proof}], prerequisites: [unit ids], box, go_deeper: [{title, url}], paper?: {title, url, questions}, tutor_prompt}`.
- `build_course_bundle.py` → `course/dist/course.json` (syllabus + every ready unit's card, lecture markdown, quiz, cards, figure paths, Colab URL). **This JSON is the only interface the site depends on.**
- Lab bootstrap cell (first cell of every lab): `pip install "git+https://github.com/vvknyn/self-driving-ai-course@main"` in Colab; locally `pip install -e .`. Python ≥ 3.10 (Colab's current runtime; local dev is 3.13).
- Labs are authored as `lab.py` (percent format: `# %%` code cells, `# %% [markdown]` cells, `# %% tags=["exercise:0.1.1.a"]`) and compiled to the committed `lab.ipynb` by `scripts/build_labs.py`, which keeps the sources diffable. Each unit may have a `figures.py` exposing `make(out_dir)`, which `scripts/make_figures.py` runs.
- v1 material (`notebooks/`, `modules/`, `docs/`) stays in place so existing Colab links keep working; README marks it "v1 reference library". Superseded v1 stub notebooks (`01_driving_perception_gym.ipynb`, `02_camera_geometry_and_ipm.ipynb`) are removed and the notebooks README fixed.

### 7.2 Site (`vvknyn/vivekn`, route `/learnfsd`)
- Sync step copies `course.json` + figures into the site at build time (static; no runtime dependency on GitHub).
- `/learnfsd` → course map: loops → modules → units with status (ready / coming soon), per-unit progress from the browser, "Start here" pointing at the next unit, "Export my progress". Below it, the **v1 chapter library** (existing `/learnfsd/[id]` pages, unchanged URLs) labelled as legacy reference.
- `/learnfsd/unit/[unitId]` → unit page: card, warm-up review, lecture (existing markdown + KaTeX renderer), "Open lab in Colab", quiz, teach-back (textarea + "copy tutor prompt"), pilot log.
- Follows the site's existing design language; mobile-width safe.
- URL slug = unit id with dots replaced by hyphens (`0.1.1` → `/learnfsd/unit/0-1-1`). The synced bundle lives at `src/data/learnfsd-course.json` and the figures at `public/learnfsd-course/figures/<slug>/`, both committed, so the Docker build needs no network.

### 7.3 Delivery mechanics (verified 2026-09-28)
- Both repos: feature branch → PR → `main`. Neither repo has a `staging` branch.
- Site deploy (the same path as the 2026-09-27 deploy of `238c98c`):
  1. `git archive` of the merged `main` goes to the box, which holds it at `/opt/vivekn-releases/<sha>`.
  2. `docker build` runs with the build secret `/opt/interview-os/deploy/contabo-web/vivekn.env`.
  3. A canary runs on `127.0.0.1:13001`.
  4. `docker compose up -d --force-recreate --no-build vivekn` replaces the running container.
  5. App-origin and public checks run, with automatic rollback to a tagged previous image.

  The deploy checks must include the new pages.

## 8. First delivery (this session) and what comes after

**In:** this spec; `zero2fsd` package (sim, boxes, provided car, runner, score, dashboard, grader, verify_units); Loop 0 units 0.1.1–0.1.4 complete per template; course bundle; site course map + unit pages + quiz + review + teach-back + pilot log; v1 stub cleanup; deploy of the site; pixels verified on prod.
**Next (just-in-time, ~4 units ahead):** Loop 1 module 1.1 onward, driven by the pilot logs.
**Out of scope for v2.0:** video lectures, accounts / cross-device sync, certificates, anti-cheating (the goal is "no accidental green", not proctoring), renaming v1 files.

## 9. Risks
| Risk | Mitigation |
|---|---|
| AI-authored lectures thin or wrong | DoD §4.2; math cross-checked against Go-deeper source; pilot logs flag confusion; review before pilot reaches a unit |
| Toy sim teaches toy intuitions | Stated openly; real frames from Loop 2; the sim is itself a box the learner inspects |
| Colab environment drift | package depends only on numpy/pillow/matplotlib/pyyaml; labs pin nothing exotic |
| Browser-only progress lost | export button; sync deferred |
| 3-year dropout | fixed weekly units, visible car improvement every module, review deck |
