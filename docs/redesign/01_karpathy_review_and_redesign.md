# LearnFSD: Karpathy-hat review and second-pass redesign

Sources consulted: live portal at https://www.vivekn.xyz/learnfsd, repo https://github.com/vvknyn/self-driving-ai-course, Duckietown MOOC syllabus, Udacity Self-Driving Car Engineer Nanodegree outlines, Karpathy nn-zero-to-hero, OpenDriveLab / UniAD / end-to-end AD survey, comma.ai openpilot hardware path, MIT/MITx-style accessibility norms.

---

## 1. If Andrej Karpathy reviewed LearnFSD

**What he would respect**
- Vision-first, Tesla-shaped spine (multi-cam → HydraNet → BEV → occupancy → vector space → plan → control). That matches how modern FSD actually thinks.
- Desire to demystify black boxes and include break-it drills.
- Interactive HTML explainers as a medium. Visual debugging is how intuition forms.
- Ambition to go from foundations to research frontier.

**What he would cut hard**
- Long essay chapters with thin notebooks (often ~3 cells, ~40 lines). That is the anti-pattern of Zero-to-Hero: there, the notebook *is* the lecture, typed line by line.
- Synthetic `randn` training sold as "driving perception." If the tensors never came from a camera, you have not learned driving ML. You have learned PyTorch shapes.
- "Production-grade / 5-tier JIT / PhD track / verified modules" chrome before the practice path is honest. Marketing language is the opposite of high signal.
- Pedigree name-dropping (Ng, Karpathy, Thrun, Duckietown, UniAD) as UI. Cite them in reading lists. Do not use them as badges.
- Micrograd as Chapter 00 of a public FSD path. Autograd belongs as a spelled-out optional appendix when a learner hits "I don't trust loss.backward()." Forcing it first blocks people who already have some NN experience and delays the first real driving artifact.

**What he would demand instead**
1. Every concept appears first as running code on real (or realistic recorded) sensor data.
2. Then peel the abstraction: shapes, gradients, failure modes, one intentional bug.
3. Then optional math depth with derivations, not the other way around.
4. Capstone must consume artifacts from earlier modules, not re-fake everything.
5. Information density: fewer pages, more "watch this tensor, then change this line."

Blunt summary in his voice: *Cute website. Weak gym. Fix the notebooks or stop calling it a course.*

---

## 2. What Coursera, Udacity, and Duckietown get right (steal the structure, not the LIDAR bias)

### Andrew Ng / Coursera pattern
- Intuition before formalism.
- Bias/variance and error-analysis checklists after every model.
- Short videos + programming assignments with clear rubrics.
- Math appears when needed, with optional deeper notes.
- Clear "you are here" progression and weekly effort estimates.

### Udacity Self-Driving Car Nanodegree pattern
- Stack sliced into CV → (sensor fusion) → localization → planning → control.
- Each slice ends in a **project with a rubric**, often on real datasets (Waymo Open) or CARLA.
- Optional advanced electives (UKF, MPC, functional safety) after core competence.
- Lesson: do not bury electives in the default path.

### Duckietown MOOC pattern (ETH / edX)
- Modules 0–9: autonomy intro → architectures → modeling/PID → robot vision → detection → estimation → planning → RL.
- **Simulation track and hardware track** with honest time estimates (sim 3–5h/week, hardware 6–10h/week).
- Learning Experiences: activities (solved tutorials) vs exercises (ungraded challenges).
- Software-first possible; hardware is optional amplification, not a gate.
- Non-EE friendly: buy a kit, follow recipes, focus on algorithms.

### Implication for LearnFSD
Keep Tesla/OpenDriveLab vision-first content (unlike classic Udacity LIDAR-heavy arcs), but adopt:
- Ng: diagnostics + short units
- Udacity: project rubrics + dataset realism
- Duckietown: sim-vs-hardware tracks + kit recipes for non-electronics learners
- Karpathy: code-is-the-lecture density

---

## 3. OpenDriveLab and research: cut marketing, keep signal

Use as **curriculum fuel**, not homepage decoration:

| Resource | Role in course |
|---|---|
| UniAD (CVPR 2023 Best Paper) | Capstone philosophy: planning-oriented stack; perception exists to serve planning |
| VAD / vectorized scene | Module on vector space vs dense BEV tradeoffs |
| End-to-end AD survey (270+ papers) | PhD-track reading map; challenges list (causal confusion, closed-loop eval, world models) |
| nuScenes / NAVSIM benchmarks | Evaluation discipline in later modules |
| Tesla AI Day talks + openpilot | Systems intuition and deployable reference code |

Rule: every paper earns a place only if a practice module cites a figure, a loss, or an evaluation metric the learner will implement in simplified form.

---

## 4. Second-pass course design (beginner → PhD-ready, self-contained)

### North star (one sentence)
Build a vision-first mini-FSD stack you understand line by line, then read modern papers without getting lost, with an optional path onto physical hardware that does not require an electronics degree.

### Entry bar
- Python comfort
- Basic calculus + linear algebra intuition
- MITx-style probability (Bayes, Gaussians, expectation) — course will re-teach locally when used
- No electronics background required

### Dual tracks (Duckietown-style)
- **Laptop track (default):** Colab or local GPU/CPU, recorded multi-cam / nuScenes mini / checked-in sample frames, browser sims, HTML labs.
- **Hardware track (optional):** staged kit ladder below. Never blocks graduation of laptop track.

### Depth modes (replace "5-tier JIT product")
Every lesson has three expandable layers on one mobile-friendly page:
1. **Practice** (default): run → see → break → fix → ship artifact
2. **Why it works:** Ng-style intuition + error analysis
3. **Derive + go deeper:** spelled-out derivation of every formula used, plus links to MIT OCW / 3Blue1Brown / Khan / specific tutorials for the missing math prerequisite; plus 1–3 paper paragraphs for research track

Beginners stay on 1–2. Experts open 3. Same URL. No separate "PhD product."

### Module map (single numbering everywhere)

| ID | Title | Practice milestone (artifact) | Research taste |
|---|---|---|---|
| 00 | Driving ML gym | Train classifier/segmenter on real frames; focal loss; error gallery | Imbalanced learning literature |
| 00b | Autograd appendix (optional) | Micrograd-style scalar engine | — |
| 01 | Cameras and IPM | Calibrated multi-cam → metric ground warp | Geometric CV / calibration papers |
| 02 | HydraNet perception | Shared backbone, multi-head, uncertainty weighting | Multi-task learning (Kendall & Gal) |
| 03 | BEV lift | Image features → BEV grid | LSS, BEVFormer |
| 04 | Occupancy + time | Short-horizon occupancy / flow | OccNet / UniAD occ |
| 05 | Vector world | Tracks + lane graph | VAD / vectorized scenes |
| 06 | Planning | Lattice / learned scorer | UniAD planning-oriented design |
| 07 | Control | Stanley / bicycle closed loop | Thrun / Stanford racing |
| 08 | Integration | Replay drive: cams→plan→control log | E2E survey challenges |
| 09 | Frontier studio | Replicate one paper figure or metric | OpenDriveLab + latest CVPR/ICCV |

### Session shape (Ng + Karpathy + FastAI)
1. Visual explainer (60–90s worth of interaction on phone)
2. Run the notebook (outputs first)
3. Break-it / fix-it
4. Ship artifact into `artifacts/mXX/`
5. Optional derive layer + math tutorial links
6. One "why Tesla / why UniAD" note

### Math rule (global)
Whenever a formula appears:
- State it in words first
- Show how it is obtained (derivation or clear derivation sketch)
- Name the prerequisite (e.g. "chain rule," "multivariate Gaussian")
- Link 1–2 high-quality tutorials (MIT OCW clip, 3Blue1Brown, specific chapter)
- Show the corresponding 5–15 lines of code

### Mobile-friendly delivery
- Lesson pages: single column, large tap targets, KaTeX that reflows, no hover-only UI
- Visual labs: touch-first (sliders, steppers), canvas sized to viewport
- Code: phone for reading + quizzes; "Open in Colab" for editing (honest about phone limits)
- Offline-friendly: downloadable notebook + short lesson markdown
- Avoid autoplay video walls; prefer short silent animations + captions

### Hardware ladder (non-EE)
Costs are approximate USD and will drift; treat as order-of-magnitude.

| Stage | What you buy | Approx | Skills | Course use |
|---|---|---|---|---|
| 0 | Laptop only | $0 extra | Python | Modules 00–08 |
| 1 | USB webcam + printed chessboard | $20–60 | Mount camera, run OpenCV calib | Module 01 live calib |
| 2 | Raspberry Pi 5 or similar + cam | $80–150 | Flash OS from recipe, SSH, run Python | Deploy Module 01–02 node |
| 3 | Ready-made Duckiebot / similar kit OR DIY robot car kit with prebuilt motor HAT | $300–800 | Follow vendor recipe; no soldering required if kit is chosen carefully | Closed-loop lane follow |
| 4 | Optional: comma device + supported car (advanced, real vehicle) | ~$700–1000+ harness | Install per vendor guide; safety-critical | Read openpilot; **not required** |

Rules for hardware chapters:
- Prefer **plug-and-play kits** and step-by-step photos
- Provide a shopping list with "buy this exact SKU" when possible
- Always offer a simulator twin so lack of hardware never blocks learning
- Explicit safety: never imply street autonomy without legal/safety framing; comma/openpilot is ADAS with driver responsibility

### Quality bar = Ng / Thrun level
- Every module has a rubric: inputs, outputs, metrics, failure cases
- Probability and estimation taught the Thrun way: belief, prediction, update, on a driving example
- No orphan theory chapters
- Capstone demo video (even screen capture) as graduation

### Honest anti-BS policy
- Ban "production-grade" until tests + real data + closed-loop eval exist
- Prefer "toy but truthful" over "impressive but fake"
- Cite OpenDriveLab, MIT lectures, papers as pointers, not as borrowed prestige

---

## 5. Practical next build order
1. Rewrite Module 00 on real sample frames + focal loss + error gallery + mobile lesson page
2. Unify numbering site ↔ repo ↔ Colabs
3. Artifact contract `artifacts/` across 01–08
4. Fix DNS or stop advertising learnfsd.vivekn.xyz
5. Only then deepen research studio (Module 09)

