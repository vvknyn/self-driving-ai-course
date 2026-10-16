# Zero2FSD Playground — design

**Date:** 2026-10-11 · **Status:** draft for Vivek's review · **Inputs already decided:** D1–D6 on the rebuild plan (Oct 11, "yes to all").
**Scope:** the playground subsystem and every seam it needs in the course repo (`vvknyn/self-driving-ai-course`) and the site (`vvknyn/vivekn`). **Not in scope:** the Week 1 lecture rewrite under writing standard v2, and the open-source move (plan Phase 4). Both are sequenced in the implementation plan, not designed here.

Every file:line below was read this session by a read-only leg; "measured" numbers come from the Oct 10 spike (`scratchpad/pyodide-spike/REPORT.md`).

---

## 1. Purpose and success

**Purpose.** The playground is the car's home. A learner writes one box of the car in the browser, watches their car drive a course road, reads its gauges, presses *Check my work*, and gets the same verdict Colab gives. A pass lights that box on the course map. No install, no kit, no account.

**Success criteria (each one is a test or an acceptance step):**

| # | Criterion | How it is checked |
|---|---|---|
| S-1 | From a Week 1 lecture page, one click opens the playground with the car driving the lesson's road, in ≤ 3 s on a wired Mac | Playwright test; measured cold load today is 1.71 s without the charting library |
| S-2 | `check()` prints the identical verdict in Colab and in the playground for every Week 1 exercise on the reference solutions | Python test runs the grader natively; Playwright test runs it in Pyodide; the two texts are compared |
| S-3 | A PASS in the playground appears on the unit page and the course map without a reload | Playwright test on the site |
| S-4 | *Stop* halts `while True: pass` and the page stays responsive; a new run starts within 3 s | Playwright test; measured today: page ticker steady at 20 ms, restart 2.45 s |
| S-5 | Works on a phone. Slower is fine; wrong is not | Acceptance step: Vivek opens the URL on his phone and runs one exercise |
| S-6 | A learner cannot be told PASS on the Mac and FAIL in the browser (or the reverse) for the reference solutions | Margin test (§4.1 e) |

## 2. The one idea, applied to the playground

| Surface | What it gives the playground | What it takes from the playground |
|---|---|---|
| Simulator (`zero2fsd`) | The engine, unchanged, via Pyodide | One per-step hook and lazy charting imports (§4.1) |
| Lecture | "Open in playground" links beside figures, preloaded with the figure's road and gains | The same drawing code (`view.js`) draws the lecture's sliders |
| Colab lab | Starter code for each exercise (the same cells) | A "Try it live" link per exercise |
| "Check my work" | `zero2fsd.grade.check` runs inside the worker | Nothing new: same ids, same text |
| Course map | Boxes light from per-exercise passes | A new `exercises` field in the progress store |
| Colours | `palette.json` from the Python palette module | Nothing |
| Quiz, cards | — | Nothing in v1 |

## 3. Architecture

```
course repo                                          site (vivekn; later zero2fsd.github.io)
zero2fsd/ (python) ───────────► bundle.zip ─┐
playground/ (static ES modules) ────────────┼► course/dist/playground/ ──sync──► public/learnfsd/playground/
scripts/build_playground.py ── exercises.json, roads.json, palette.json ┘               │
                                                                       unit page <iframe> · lecture sliders import view.js

browser:  page (editor · player · gauges · console)  ◄── postMessage ──►  module worker (Pyodide + zero2fsd)
                                                                            ▲ pyodide pinned from cdn.jsdelivr.net
```

**Why a static folder in the course repo, not a React feature of the site.** The playground must survive the move to zero2fsd.github.io, which serves files, not Next.js. The site already mirrors `course/dist/` by copy (`scripts/sync-course-bundle.mjs:2,13-15`), so the playground ships through the same door. The site keeps only thin seams (§4.6).

**Why a producer and a player, not a paced simulation.** The simulator runs the lap at full speed in the worker (measured 205 steps/s; a 1,778-step lap in 8.7 s) and streams every step to the page. The page replays at 1×, 4× or "skip to end", with a scrubber. The verdict is ready the moment the run finishes, even while playback is still going. A slow phone only makes the run take longer to produce; it never changes a number.

Alternatives considered are in §11.

## 4. Components

### 4.1 Course package seams (Python, `zero2fsd/`)

Each is a small, testable change that also makes the package better outside the playground.

**a. Per-step hook.** `run(car, scenario, max_steps=None, seed=0, noise_sigma=0.0)` (`car/runner.py:43`) gains `on_step=None`. Right after the runner appends a step's record to `rows` (`:62-63`) it calls `on_step(row)` with that same record, so the stream is exactly the telemetry, one row at a time, and the shim never recomputes `lat` or `progress_m`. `Telemetry` (`:20-40`) is unchanged. *Test:* the hook is called 1,778 times on `gentle`; the rows it saw, stacked, equal the returned telemetry; telemetry with and without the hook is equal.

**b. Lazy charting imports.** `sim/topdown.py:5` and `dashboard.py:5` import `matplotlib.figure` at module level, so `import zero2fsd.sim` (`sim/__init__.py:5`) and `import zero2fsd.grade` (`grade/_drive.py:7`) both load matplotlib. The import moves inside the functions that draw. *Test:* a subprocess with `sys.modules["matplotlib"] = None` imports `zero2fsd.car` and `zero2fsd.grade` and drives `gentle` to completion. *Effect measured:* first load 1.71 s / 10.36 MB instead of 2.94 s / 19.27 MB; 9.8 s instead of 18.3 s at 10 Mbit/s.

**c. Road export.** A public `Road.centerline_points(spacing_m)` (built on `Road.offset(s, 0)`, `sim/world.py:57`) replaces reaching into the private `_a`. The build writes `roads.json`: for each of the three scenarios (`sim/scenarios.py:73`), the centre line at 0.5 m spacing, the lane edges at `EGO_LANE_CENTER ± LANE_WIDTH/2` (`world.py:13-14`), `closed`, and `length`. *Test:* the exported polyline's length is within 0.1 % of `road.length`.

**d. Preview registry.** `zero2fsd/preview/` mirrors the grade registry (`grade/units/u0_1_*.py`): one entry per exercise id, `Preview(kind, build)`. Kinds in v1: `drive` (`build(fn) -> Car`, e.g. for `p_steer`: the provided perception plus the learner's steer at `K_OFF, K_HEAD = 0.35, 2.7`) and `stats` (`build(fn) -> str`, e.g. `lane_error_stats` over the provided car's `tel.lat`). `image` (paint masks on a frame) is added in Phase 3. *Test:* the registry holds exactly the exercise ids the grade registry holds (mirrors `tests/zero2fsd/test_grade.py:51`).

**e. Verdict margins.** Three Week 1 checks compare simulation-derived metrics to thresholds: the drive gate (`_drive.py:11`, mean |lat| < 0.5 m and completed), the perception hit rate (`u0_1_3.py:13`, ≥ 90 % within 0.25 m / 0.05 rad), and the paint IoU (`u0_1_2.py:11`, ≥ 0.85). The other checks compare the learner's function with a computation done in the same runtime, so Mac-vs-browser cannot split them. New `tests/zero2fsd/test_margins.py` runs every thresholded check on the reference solution and asserts the metric sits ≥ 5 % of the threshold away from it (today's gate margin: 0.376 vs 0.5, 25 %; the measured Mac/browser gap is 3 × 10⁻⁵ m). For the learner's own answers, `_report.py` prints metric, threshold and margin on every thresholded line, and when the margin is under 2 % adds one sentence: *"Borderline: tiny differences between computers can flip this. Aim for a clearer margin."* That is the honest answer to non-bit-identity; chasing bit-identity across libm versions is not attempted.

**f. Frame on demand.** The worker shim (not the package) exposes `render_frame(scenario, k, x, y, yaw, speed, seed)` by calling `Camera.render` (`sim/camera.py:79-86`) with the same seed rule the runner uses (`runner.py:59`, `seed + k`). Scrubbing to a step re-renders that one 320×180 frame (`camera.py:54`) in 1–2 ms instead of keeping 1,778 frames (307 MB) in memory.

### 4.2 Build and bundle

`scripts/build_playground.py`, called from `build_course_bundle.py` so the output lands under `course/dist/playground/` and the site's sync copies it unchanged:

| File | Source | Notes |
|---|---|---|
| `bundle.zip` | `zero2fsd/` + `playground/py/shim.py` | No `solutions/`, no tests, no `.pyc`. Measured 31 KB. |
| `exercises.json` | `course/syllabus.yaml` (week number = position), `unit.yaml` (`exercises: [{id, name}]`), `lab.py` (the `exercise:<id>` tagged cell's source is the starter, via `parse_lab`, `scripts/_course.py:54`), preview registry (kind, road) | Learner-visible labels are "Week N · Title · Exercise A"; dotted ids stay internal |
| `roads.json` | §4.1 c | |
| `palette.json` | the palette module (plan Phase 1) | Quantity colours only |
| `index.html`, `app.js`, `worker.js`, `player.js`, `view.js`, `app.css` | `playground/` in the course repo | Dependency-free ES modules; no bundler |

Presets: `exercises.json` carries named presets per exercise (e.g. `k_off=1.5` for the gain-sweep figure). URL parameters select a preset by name; they are never evaluated as code.

### 4.3 Worker protocol

Module worker (a classic worker failed in the spike: `importScripts` of pyodide.js threw). Messages, page → worker: `init`, `run {code, exercise, road}`, `check {code, exercise}`, `frame {k}`, `dashboard`. Worker → page: `ready {versions, steps_per_s_probe}`, `row {k, t, x, y, yaw, speed, steer, lat, head_err, est_offset, est_heading, est_valid}` (one per step), `frame {k, rgb}` (transferable `ArrayBuffer`), `stdout {text}`, `done {completed, score, metrics}`, `verdict {passed, details}`, `error {traceback}`. The learner's code is `exec`'d in a fresh namespace on every *Run* or *Check*; the package state persists across runs so the second run does not re-import.

*Stop* is `worker.terminate()` followed by a fresh worker (measured: ~2.1 s of CPU tail after terminate, restart 2.45 s warm). No `SharedArrayBuffer`, so no cross-origin-isolation headers; the site sets none today (`next.config.ts` has no `headers()`; `src/middleware.ts` only rewrites by host) and GitHub Pages cannot set any.

### 4.4 Page

Layout (≥ 960 px): editor left; top-down view and gauges right; console and verdict below. Phones (< 640 px): the same blocks stacked, editor first. Chrome colours mirror the site's learnfsd pages (`bg #090d16`, text `#c9d1d9`, surfaces `#0d1117`, borders `#30363d`, accent `#58a6ff` — hardcoded Tailwind values in `unit/[slug]/page.tsx:49-50`, `CourseMap.tsx:63`), written once in `app.css` with a comment naming the source. Quantity colours come only from `palette.json`.

Controls: road select (gentle / curvy / straight), **Run**, **Stop**, **Check my work**, **Dashboard** (loads matplotlib on first press, renders the Colab dashboard PNG; measured 0.26 s once loaded), playback 1× / 4× / end, scrubber.

States: *loading* (progress bar over the 10 MB; names what is loading) → *ready* → *running* → *done* or *error*. Nothing in the page pretends to be ready before `ready` arrives.

Editor: CodeMirror 6, pinned version, loaded as ES modules from cdn.jsdelivr.net. If that import fails the page falls back to a plain `<textarea>` and says so in the console panel (a real editor with no highlighting, not a dead button).

Header: "Week N · Unit title · Exercise A — name", the exercise's one-line task, and the unit page link.

### 4.5 Player and view

`player.js` holds the rows of the current run, the playback clock and the scrubber position. `view.js` draws: the road from `roads.json` (surface grey, centre line in paint yellow, edge in paint white, course palette), the car as a rectangle at (x, y, yaw) with the e_off brace in blue, and three strip charts for the last 10 s: e_off (blue) with the estimate (coral, dashed), ψ (pink) with its estimate, δ (green). Names sit at the line ends, no legends (writing standard v2, rule 24). Camera panel shows the frame for the scrubbed step.

`view.js` is a dependency-free ES module with one export, `draw(canvasCtx, road, rows, k, palette)`. The site's lecture sliders (D5) import it at runtime from `/learnfsd/playground/view.js` and feed it precomputed rows from `sweeps/<name>.json`, so the lecture picture and the playground picture are the same picture.

### 4.6 Site seams (`vvknyn/vivekn`)

1. **Sync:** `scripts/sync-course-bundle.mjs` also mirrors `course/dist/playground/` to `public/learnfsd/playground/`; `validate-course-bundle.mjs` checks `exercises.json` parses and every exercise id in the course JSON has an entry.
2. **Unit page panel:** a *Playground* panel with `<iframe src="/learnfsd/playground/?exercise=<id>">` plus "Open full tab" — the pattern at `src/app/learnfsd/[id]/ChapterTabs.tsx:90-94, :73-75`. Same origin, so the iframe shares `localStorage`.
3. **Progress:** `UnitProgress` (`src/lib/learnfsd-progress.ts:9-15`) gains an optional `exercises?: Record<string, { passedAt: string; via: "playground" }>`. `version` stays 1 — additive and optional, so `parseProgress` (`:22-30`) keeps accepting old data. The playground's `progress.js` writes the same key (`STORAGE_KEY = "learnfsd.v2.progress"`, `:7`) and shape. `useProgress` (`src/app/learnfsd/useProgress.ts`) listens for the `storage` event so a pass inside the iframe updates the page (S-3). *Test:* `tests/learnfsd/progress.test.ts` parses a fixture written by the playground's `progress.js` (copied by the sync).
4. **Course map:** `CourseMap.tsx` gains the car drawing — the loop boxes camera → perception → steer → move plus an instruments box — each lit when the exercise that writes it has passed. Week 1 lights *instruments* (telemetry exercises); Weeks 2–4 light paint, perception, steer. The quiz state stays as it is (`isQuizPassed`, `learnfsd-progress.ts:38`).
5. **Lecture links:** written in `lecture.md` as `../../playground/?exercise=<id>&preset=<name>` so the same text resolves on `/learnfsd/unit/<slug>` today and on `/unit/<slug>/` on GitHub Pages. The bundle validator checks each link against `exercises.json`.

### 4.7 Hosting

Site: static files under `public/`, copied into the image as-is (`Dockerfile:26`); no CSP or headers exist to amend. Pyodide and CodeMirror come from cdn.jsdelivr.net at pinned versions (Pyodide 314.0.7 is the measured one). GitHub Pages later: the same folder at `/playground/`.

## 5. Data flow — one *Check my work*

1. Page posts `check {code, exercise}`.
2. Worker `exec`s the code in a fresh namespace, pulls the function named in `exercises.json`, and calls `zero2fsd.grade.check(id, fn, show=False)` (`grade/__init__.py:18`), capturing stdout.
3. While the gate drives (`0.1.4.b` and `0.1.3.b`), the `on_step` hook streams `row` messages, so the learner watches the graded drive itself.
4. Worker posts `verdict {passed, details}` from the returned `Result` (`_report.py:22-25`).
5. Page shows `details` verbatim in the console panel, and on `passed` writes `exercises[id] = {passedAt, via: "playground"}` to the progress key.
6. The parent unit page and the course map receive the `storage` event and re-render.

## 6. Error handling

| Situation | Behaviour |
|---|---|
| Learner code raises | Traceback in the console; run ends at that step; car stays where it stopped; verdict panel says nothing |
| Infinite loop | *Stop* terminates the worker; page shows "stopped" and reloads the runtime (≈2.5 s) |
| CDN or network failure during load | Loading state turns into an error naming the file that failed and a *Retry* button; no silent fallback |
| Unknown exercise id in the URL | The page opens on the first Week 1 exercise and says why |
| Charting library fails to load on *Dashboard* | The button reports the failure; the rest of the page keeps working |
| Browser without `Worker` or `WebAssembly` | One sentence naming the missing feature and a link to Colab |

## 7. Performance budget (from the spike)

| Item | Budget | Measured |
|---|---|---|
| First load, wired | ≤ 3 s, ≤ 11 MB | 1.71 s, 10.36 MB without matplotlib |
| First load, 10 Mbit/s + 40 ms | ≤ 12 s | 9.84 s |
| Lap of `gentle` with camera | ≥ 5× real time on a Mac | 10.3× (205 steps/s) in Chrome, 11.6× in Safari |
| Same, CPU throttled 4× | ≥ 2× real time | 2.33× |
| Heap | ≤ 120 MB | 74.9 MB (worker, one lap) |
| Phone | not predicted; measured in acceptance (S-5) | — |

## 8. Security and privacy

Learner code runs only in the learner's own browser; there is no server and nothing leaves the page. URL parameters are names looked up in `exercises.json`, never evaluated. The iframe is same-origin and shows our own page, so no `sandbox` attribute is needed (adding one would cut it off from `localStorage` and break S-3). Pinned CDN versions; the loader script carries an integrity hash.

## 9. Testing

**Course repo (pytest, `tests/zero2fsd/`):** hook (§4.1 a), lazy import subprocess (b), road export (c), preview registry (d), margins (e), `build_playground` output schema, and `exercises.json` starters equal the lab cells. **Browser (playwright-python, optional `[test]` extra):** load → run the reference `0.1.1.a` and `0.1.4.b` → verdict text equals the native text (S-2); 1,778 rows arrive; *Stop* works (S-4); textarea fallback when the editor import is blocked. **Site (`npm run test:learnfsd`):** progress fixture round-trips; sync copies the playground folder; validator rejects a lecture link to a missing exercise.

## 10. Sequencing (for the implementation plan)

1. **Foundation** — §4.1 a–f and the palette module; all pure Python with tests; no site change. The lazy import alone is a visible win for Colab too.
2. **Playground v1** — §4.2–4.5 with the two Week 1 kinds; browser test; served from the course repo's `course/dist/playground/`.
3. **Site seams** — §4.6; the Week 1 unit page gets the panel; map lights.
4. **Acceptance** — S-1…S-6, including Vivek's phone and a Colab-vs-playground verdict read side by side.

Weeks 2–4 then add the `image` kind and their presets; nothing structural.

## 11. Decisions in this spec

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
| S10 | Week 1 kinds | Week 1 kinds are `drive`, `stats` and one `image` exercise (0.1.2.a); the image preview is one frame beside the mask overlay, no player | Label 0.1.2.a "Colab only": contradicts S-2, which covers every Week 1 exercise |

## 12. Non-goals and open points

- Weeks that need PyTorch stay Colab-only; Pyodide has no torch. The playground can still *play* their saved runs later (not v1).
- Colab passes are not recorded on the map (S7).
- The GitHub Pages move changes the path prefix; relative links (§4.6 5) are the only coupling, and the validator guards them.
- The spike's harness is throwaway; nothing from it is copied into the repo.
