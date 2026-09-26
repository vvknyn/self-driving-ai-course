# LearnFSD Modules 00–01 — Locked Design

Status: **build from this doc**. Tweaks welcome after Week 1 Colab is walkable.

## Audience & bar

- **Door:** normal person — Python OK, rusty linear algebra, maybe trained a net once.
- **Exit after Week 2:** operates like a first-year Stanford/MIT AV masters student in a geometry + perception lab: ships artifacts, explains first principles, writes testable code.
- **Calendar:** ~6 months of material compressed into ~10 weeks; Modules 00–01 are Weeks 1–2.
- **Tracks:** one spine. Practice / Why / Derive on the same page. Optional hardware never blocks.

## Pedagogy contracts (global)

### 1. First principles (required, not optional Derive-only)

Every module states a **governing principle in plain words**, shows why it must be true (sketch or short derivation), then **assignments force applying that principle** to the solution. Passing by memorizing an API call is a fail.

Pattern in Colab + `.py`:
1. Principle box (3–6 sentences, no jargon wall)
2. Tiny derivation or “why this formula”
3. Scaffold uses the principle correctly nearby
4. Student fill / from-scratch must re-derive the critical step in code
5. Rubric question: “Which principle did your fix apply?” (short written cell)

### 2. Silent Python + engineering

Apprenticeship-by-scaffold. Strict from Week 1 (light bar):

- Type hints on every public function students implement
- Docstrings with Args / Returns / tensor shapes `(B,C,H,W)` etc.
- Student code lives in `modules/…/*.py`; Colab imports it
- `pytest` green; students add ≥1 test for their from-scratch island
- Config as dataclass or small dict (no magic numbers in fills)
- Seeded runs; `artifacts/m0X/<run_id>/` with `metrics.json` + plots
- Shape asserts with helpful messages; NaN loss fails

### 3. Assignment shape

Per module:
- **Scaffold (~60–70%):** data, plots, train loop shell, artifact I/O
- **Load-bearing fills (~25–30%):** `NotImplementedError` stubs for the week’s ideas
- **One from-scratch island (~5–10%):** empty body; tests only
- **Break-it:** engineering or principle broken; student repairs
- Solutions in `solutions/` (or hidden) after attempt

### 4. Stack (current, JIT — max one new tool/week)

Week 1: Python 3.11+, PyTorch, torchvision, numpy, matplotlib, pytest.  
Week 2: + OpenCV.  
Later modules may add einops, timm/HF, Lightning *optional*, etc. Never dump the whole modern stack in Week 1.

### 5. Data honesty

No `torch.randn` as the main “driving” path. Use real or realistic checked-in frames (tiny in-repo set required so Colab works offline-ish; optional larger download documented).

### 6. Tone

Prefer “why vision-first stacks do X” with Tesla / OpenDriveLab / UniAD as *examples*, not brand worship or hype (“production-grade”, “PhD product tiers”).

### 7. Autograd

Optional **00b** appendix. Not on the default Week 1 critical path.

---

## Defaults locked for 00–01

| Decision | Choice |
|---|---|
| Module 00 task | **Multi-class classifier on real crops** (road / vehicle / pedestrian / lane). Stretch: tiny seg head. |
| Dataset | **Tiny in-repo sample frames** + manifest; optional larger download noted in README |
| Module 01 cams | **1-cam IPM required first**, then **3-cam stitch required** for pass |
| Engineering | **Strict light bar from Week 1** (types + tests) |
| First principles | In spine + forced in assignment cells |

---

## Module 00 — Driving ML Gym (Week 1)

### Governing first principles

1. **A model is a function from data to scores; learning is adjusting parameters to make a chosen loss small on the training distribution.**
2. **The loss defines what “good” means.** If the loss ignores rare classes, the model will too (imbalance).
3. **You cannot improve what you do not inspect.** Error galleries beat vanity accuracy.

Students must apply (2) and (3) in code to pass — not only call `CrossEntropyLoss()`.

### Outcomes (Practice)

- Explain `(B,C,H,W)` in one sentence each
- Train tiny CNN (or frozen backbone + head) on real crops
- Compare CE vs focal (or class-weighted CE) on minority recall
- Export error gallery + metrics + checkpoint
- Break/fix one training bug with principle-based reasoning

### Repo layout (target)

```
modules/00_ml_gym/
  README.md                 # thin hub: Practice / Why / Derive links
  config.py                 # dataclass TrainConfig
  dataset.py                # real-frame Dataset (scaffold)
  model.py                  # tiny CNN (scaffold + optional student tweak)
  losses.py                 # CE wrapper scaffold; focal_loss FILL
  metrics.py                # accuracy scaffold; minority_recall FILL
  train.py                  # loop scaffold
  error_gallery.py          # FROM SCRATCH island
  break_it_fix_it.py
  tests/
artifacts/m00/              # gitkeep + example structure
notebooks/00_driving_ml_gym.ipynb   # PRIMARY Colab-style lecture
data/m00_sample/            # tiny real crops + labels.csv
solutions/00_ml_gym/        # reference fills (not imported by default)
```

### Colab flow

1. Principle boxes for (1)(2)(3)
2. Run cold-open gallery on sample frames
3. Inspect class counts (principle 2 foreshadow)
4. Train CE baseline → confusion matrix
5. **Fill:** `focal_loss` from the formula (must match math cell)
6. Retrain → compare minority recall (**Fill:** metric)
7. **From scratch:** `build_error_gallery(...)`
8. Written cell: “Which principle did focal loss apply?”
9. Break-it + pytest
10. Export `artifacts/m00/run_*/`

### Rubric

- [ ] Trains on real frames (not randn)
- [ ] Focal (or weighted CE) improves minority recall vs CE by documented delta on sample set
- [ ] Error gallery artifact exists and is sorted by confidence error
- [ ] Types + tests green; student-authored test for gallery
- [ ] Written principle check answered

### Why / Derive (same page)

- Why: Data Engine intuition; accuracy lies under imbalance
- Derive: softmax → CE; focal modulating factor; optional chain-rule pointer → 00b
- Links: 3Blue1Brown backprop, StatQuest CE

### Non-goals

Cameras, IPM, multitask heads, writing autograd.

---

## Module 01 — Cameras & IPM (Week 2)

### Governing first principles

1. **A pinhole camera maps 3D rays to 2D pixels by similar triangles** — depth divides lateral position.
2. **A homography exists between two planes.** Under flat-road \(Z=0\), image ↔ ground is a \(3\times3\) \(H\) built from \(K\) and extrinsics.
3. **Wrong extrinsics → wrong meters.** Small pitch error grows with range (geometry, not “ML vibes”).

Assignments force building \(K\), \(H\), and measuring pitch sensitivity from principle (3).

### Outcomes (Practice)

- Build \(K\); project points; explain each term
- Build planar \(H\); IPM-warp one real front image
- Stitch 3 overlapping cams into ground canvas
- Quantify lane shift vs pitch; compensate or diagnose
- Ship `ground_bev.png` + `calib.json`

### Repo layout (target)

```
modules/01_camera_geometry/
  README.md
  config.py
  camera_model.py      # build_K FILL; project points scaffold
  extrinsics.py        # scaffold
  ipm.py               # build_H FILL; warp scaffold
  stitch.py            # multi-cam blend scaffold
  pitch_sensitivity.py # FROM SCRATCH island
  calibrate_rig.py     # demo entry
  break_it_fix_it.py
  tests/
data/m01_sample/       # frames + calib json for 3 cams
notebooks/01_cameras_and_ipm.ipynb
artifacts/m01/
solutions/01_camera_geometry/
```

### Colab flow

1. Principle boxes (1)(2)(3) + visual explainer
2. Pinhole playground
3. **Fill:** `build_intrinsic_matrix`
4. Extrinsics ego↔cam demo
5. **Fill:** `build_ground_homography` from first principles (drop \(r_3\) when \(Z=0\))
6. Warp + show
7. 3-cam stitch
8. **From scratch:** `pitch_shift_meters(calib, delta_deg, range_m)`
9. Written: which principle explains the smear?
10. Assignment: fix miscalibrated cam to pass lane alignment metric
11. Artifacts out

### Rubric

- [ ] IPM within pixel/ground error threshold on known points
- [ ] 3-cam stitch without catastrophic seams
- [ ] Pitch experiment plot + numbers
- [ ] Types + tests; student test for pitch function
- [ ] Principle check cell

### Hardware twin (optional)

Stage 1: USB webcam + chessboard. Sim twin always available.

### Non-goals

Learned BEV/LSS (Module 03), full 8-cam Tesla rig.

---

## First-principles in grading (both modules)

Every load-bearing fill is paired with:
- A markdown cell stating the principle
- A test that a wrong-but-running implementation can fail (e.g. focal with \(\gamma=0\) equals CE; \(H\) with identity \(R\) wrong for pitched cam)
- A short free-response: apply the principle to explain a failure mode we inject

---

## Build order

1. This doc → `docs/redesign/03_modules_00_01_design.md`
2. Implement Module 00 fully (data sample, fills, tests, notebook, artifacts contract)
3. Implement Module 01 to the same contract
4. Thin mobile/README hubs; defer portal polish
5. Do **not** mass-rewrite Modules 02–09 until 00–01 feel right

## Anti-BS checklist for implementers

- No randn-as-driving
- No essay-only README as the course
- No “production-grade” claims
- Notebook is dense: principle → run → fill → test → artifact
- Student must *apply* principles in fills and written cells
