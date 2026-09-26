# LearnFSD Redesign Research Brief
**Target course:** [vvknyn/self-driving-ai-course](https://github.com/vvknyn/self-driving-ai-course) (LearnFSD / vision-first Mini-FSD)  
**Researched:** 2026-09-25 (America/Toronto)  
**Method:** Primary sources via WebSearch/WebFetch + local clone inspection of notebooks/docs/modules.

**Repo facts that ground this brief (verified locally):**
- README advertises “production-grade,” Karpathy/Ng/Thrun/Duckietown/OpenDriveLab lineage, PhD track, Tesla FSD modeling.
- Lecture essays: `docs/*.md` ≈ **2,700 lines** total across ~16 docs.
- Notebooks: most chapters are **3 cells** (1 code + 2 markdown); only `00_*` is thicker (9 cells). Several core notebooks train/demo on **synthetic `torch.randn` / toy tensors**, not nuScenes/Waymo/KITTI.
- HTML visual explainers exist under `visual_explainers/` (~11 pages).
- Modules under `modules/` are richer than notebooks (Python scripts + `break_it_fix_it.py` + pytest) — redesign should treat notebooks as the weak layer, not necessarily delete module code.

---

## 1. How Karpathy would likely critique this course

**Karpathy’s public teaching standard (citeable):**
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — multi-hour live notebooks that *spell out* backprop → GPT; prerequisite honesty (“solid programming, intro-level math”); transferability argument for focusing on one stack deeply.
- micrograd / makemore / nanoGPT pattern: build the whole mechanism in code, then scale; companion repos keep git history walkable ([karpathy.ai/zero-to-hero](https://karpathy.ai/zero-to-hero.html), [build-nanogpt lineage described in secondary analyses](https://blakecrosley.com/blog/engineering-philosophy-andrej-karpathy)).
- CS231n assignments: multi-question notebooks with fill-in regions, vectorized NumPy, CIFAR-10 real data, graded checkpoints ([Assignment 1 2019](https://cs231n.github.io/assignments2019/assignment1/)).
- Eureka Labs / LLM101n framing: teacher designs materials; AI TA scales tutoring — still assumes *substantial* build-your-own projects ([announcement thread](https://threadreaderapp.com/thread/1813263734707790301.html), [Reuters](https://www.reuters.com/technology/artificial-intelligence/former-openai-tesla-engineer-andrej-karpathy-starts-ai-education-platform-2024-07-16/)).

**Likely critique points mapped to LearnFSD’s actual artifacts:**

| Claim / pattern in LearnFSD | Karpathy-shaped objection |
|---|---|
| “Karpathy style / from scratch” in README while most notebooks are ~3 cells | From-scratch means *hours of typed reasoning*, not a one-cell ToyLiftSplat + diagnostic table. Compare Zero-to-Hero lecture lengths (56m–2h25m) vs a single forward pass on `torch.randn`. |
| Long lecture essays (`docs/`) + thin executable notebooks | Essays without runnable verification are slides-by-another-name. Karpathy’s medium is the notebook/repo; prose supports code, not the reverse. |
| Synthetic `randn` training for HydraNet/LSS/etc. | Toy tensors teach API shape, not failure modes of driving data (class imbalance, calibration error, distribution shift). CS231n starts students on CIFAR-10 immediately. |
| HTML visual explainers as primary “intuition” | Useful as *secondary* demos (like interactive underactuated notes), but not a substitute for inspecting tensors, losses, and metrics under your own random seed. |
| “Production-grade” / “PhD track” / “Tesla FSD-like” branding | Credibility debt. Production FSD implies fleet data, safety case, latency budgets, eval harnesses. PhD track implies open problems + paper reading + nontrivial experiments. Claiming both without real datasets, closed-loop metrics, or ablations invites distrust. |
| Name-dropping five “titans” as pedagogy | Pedigree is not pedagogy. Either *do* the method (Ng diagnostics with real bias/variance plots; Thrun with particle filters on noisy measurements; Duckietown with sim→robot gate) or drop the claim. |
| Module scripts exist but notebooks don’t force students through them | Fast.ai-style “run first” is fine; Karpathy-style still requires students to *re-implement or deeply modify* inner loops, not only run `train.py`. |

**Constructive Karpathy redesign direction (inferred, not attributed as a quote):**
1. One chapter = one *long* notebook (or paired notebook + clean `.py`) that a student can pause and remix.
2. Real pixels early: nuScenes mini / KITTI / comma2k19 / webcam — even if models stay tiny.
3. Explicit “pedagogy vs production” fence (as micrograd ≠ PyTorch).
4. Honest leveling: “intro vision stack for hobbyists/engineers,” not PhD candidacy prep, unless Tier 4 is a separate paper-implementation track with benchmarks.

---

## 2. Structural patterns from reference curricula

### Coursera / Andrew Ng (ML + DL Specialization)
- **Structure:** Short video lectures → quizzes → guided notebooks with `# YOUR CODE HERE` cells → (sometimes) application notebooks ([Deep Learning Specialization](https://www.deeplearning.ai/specializations/deep-learning)).
- **Scaffolding:** Step-by-step build (`Building_your_Deep_Neural_Network_Step_by_Step`) then Application notebook on real-ish tasks (cat images, YOLO car detection) — see public mirrors e.g. [rahulg-101 notes layout](https://github.com/rahulg-101/Deep-Learning-Specialization-Coursera).
- **Math posture:** Intuition-first; optional deeper notes; chain-rule intuition before full matrix calc ([Coursera NN&DL](https://www.coursera.org/learn/neural-networks-deep-learning)).
- **Strategy course:** Explicit ML project diagnostics (bias/variance, data splits) — Course 3 — transferable to AV eval design.
- **Takeaway for LearnFSD:** Keep diagnostic tables, but attach them to *graded* notebook exercises with unit tests and real metrics, not markdown-only field guides.

### Udacity Self-Driving Car Engineer Nanodegree
- **Current outline (marketing page, verify dates):** 14 courses / 39 lessons / **7 projects**; ~78 hours claimed; partners historically Waymo/Mercedes ([Udacity ND0013](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd0013)).
- **Project spine (concrete):** Urban object detection (Waymo), 3D lidar detection, sensor fusion + EKF tracking, scan-matching localization (ICP/NDT + CARLA maps), motion planning, PID trajectory tracking.
- **Legacy pattern (v8 era):** Lane finding → traffic signs → behavioral cloning → EKF → kidnapped vehicle → highway planner → system integration on physical car ([archived syllabus mirrors](https://us-udacity.com/CourseCatalog/self/sdce.html)).
- **Strengths:** Project-gated progress; industry datasets; stack breadth (perception→control).
- **Weaknesses (skeptical):** Price/subscription; simulator flakiness reported in reviews on same page; marketing salary claims; “advanced” label with heavy scaffolding can feel fill-in-the-blank.
- **Takeaway:** LearnFSD should gate progression on **submitted projects with rubrics**, not completed HTML tours.

### Duckietown / Duckietown MOOC / textbooks
- **MOOC:** ETHx “Self-Driving Cars with Duckietown” — 9 modules; sim track 3–5 h/wk, hardware 6–10 h/wk; optional Duckiebot ([edX course](https://www.edx.org/learn/technology/eth-zurich-self-driving-cars-with-duckietown); [syllabus PDF](https://courses.edx.org/asset-v1:ETHx+DT-01x+3T2023+type@asset+block@Syllabus_Self-Driving_Cars_with_Duckietown_MOOC_2022Dec02.pdf)).
- **Learning Experiences (LXs):** Plug-and-play week units: videos + notes + Jupyter + sim/robot activities ([What is an LX](https://docs.duckietown.com/ente/duckietown-manual/60-learning-experiences/02-what-is-a-duckietown-learning-experience.html); [exercises list](https://docs.duckietown.com/ente/duckietown-manual/80-instructor-manual/available-resources/exercises.html)).
- **Curriculum modules:** Braitenberg → modeling/PID → vision → detection → state estimation → planning → RL ([educational resources](https://duckietown.com/educational-resources/)).
- **Strengths:** Honest hardware optional track; ROS/Docker reality; assessment via challenges; classical + ML coexistence.
- **Weaknesses:** Abstraction layer (ROS/Docker) can dominate learning time; Duckietown is *scaled city*, not highway FSD; Jetson/kit cost barrier.
- **Takeaway:** Split LearnFSD into **sim-required** vs **hardware-optional** paths with the same autograders where possible.

### Sebastian Thrun / Probabilistic Robotics / Udacity CS373 era
- **Pattern:** Short videos + immediate coding exercises; handwritten intuition; Kalman/particle filters implemented by students ([Thrun “Learn. Think. Do.”](https://www.udacity.com/blog/2012/09/sebastian-thrun-learn-think-do-thats.html); [CS373 notes](https://medium.com/self-driving-cars/cs373-4c30dbecc428)).
- **Book DNA (*Probabilistic Robotics*):** Bayes filter → Gaussian → nonparametric → occupancy grids → SLAM — math with algorithmic pseudocode, then implement.
- **Takeaway:** For tracking/localization chapters, force **noise injection + filter comparison plots** (histogram vs particle vs EKF) on 1D then 2D before “vector space tracker” branding.

### MIT OCW / MITx-relevant
- **16.485 VNAV (Fall 2020):** Grad-level vision navigation; C++/ROS/OpenCV labs; theory + hardware race car/drone; final project ICRA-style report ([OCW syllabus](https://ocw.mit.edu/courses/16-485-visual-navigation-for-autonomous-vehicles-vnav-fall-2020/pages/syllabus/)). Recommended texts: Barfoot *State Estimation for Robotics*; Ma et al. *Invitation to 3-D Vision*.
- **Underactuated Robotics (Tedrake):** HTML-first interactive textbook + Drake Python; spiral through model systems → planning/control → estimation/learning ([underactuated.csail.mit.edu](https://underactuated.csail.mit.edu/)).
- **MIT 6.S094 (Lex Fridman era):** Lectures + browser DeepTraffic RL competition ([DeepTraffic repo](https://github.com/lexfridman/deeptraffic)) — low friction interactive eval, thinner “build full stack” expectation.
- **Takeaway:** Prefer **interactive notes that execute** (Underactuated) and **lab+demo+report** (VNAV) over essay PDFs; optional competition hooks like DeepTraffic for engagement.

### University of Toronto Self-Driving Cars Specialization (Coursera)
- Four courses: Intro → State Estimation → Visual Perception → Motion Planning; CARLA projects; ~3 months @ 10 h/wk claimed ([Coursera specialization](https://www.coursera.org/specializations/self-driving-cars)).
- Strong modular stack teaching; still mostly classical+ML hybrid, not Tesla-like E2E transformers.
- **Takeaway:** Best reference for **course ordering** and CARLA project design for a serious non-E2E track.

---

## 3. What OpenDriveLab contributes (absorb ideas, not marketing spam)

**Primary hub:** [opendrivelab.com](https://opendrivelab.com/#news) · [Publications](https://opendrivelab.com/publications)

### Ideas worth teaching (concept → student exercise)

| Contribution | Why it matters pedagogically | Absorb without spam |
|---|---|---|
| **UniAD** — planning-oriented unified stack; query interfaces across track/map/motion/occ/plan; CVPR 2023 Best Paper ([paper](https://arxiv.org/abs/2212.10156), [code](https://github.com/OpenDriveLab/UniAD)) | Reframes “perception metrics” as servants of planning collision/L2 | Teach the *philosophy* + simplified query toy; do **not** claim students “built UniAD.” Point to official code for advanced track. |
| **VAD / VADv2** — vectorized scene reps for efficient E2E ([VAD](https://arxiv.org/abs/2303.12077), [VADv2](https://arxiv.org/abs/2402.13243)) | Motivates LearnFSD’s “vector space” module with a real paper lineage | Contrast raster BEV vs vector agents/map elements on a tiny synthetic scene. |
| **DriveLM** — Graph VQA linking perception→prediction→planning with language ([repo](https://github.com/OpenDriveLab/DriveLM/)) | Excellent for *reasoning* assignments and eval rubrics | One exercise: write GVQA chains for a scene; optional small VLM — not a “language FSD product.” |
| **E2E survey** — 270+ papers roadmap/challenges ([IEEE TPAMI survey page via OpenDriveLab](https://opendrivelab.com/publications); [GitHub survey repo](https://github.com/OpenDriveLab/End-to-end-Autonomous-Driving)) | Honest map of modular vs E2E, imitation vs RL, closed-loop eval | Assign as reading + 1-page critique; kill “PhD roadmap” original essays that paraphrase surveys. |
| **OpenLane / OpenLane-V2** — lane + topology benchmarks ([OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2)) | Real vision-centric map learning tasks | Optional advanced project: topology metrics, not required for core course. |
| **NAVSIM** — data-driven non-reactive planning benchmark (NeurIPS 2024 D&B; listed on publications) | Better planning eval than “looks good in HTML canvas” | Capstone: submit planner to NAVSIM-style metrics if compute allows. |
| **BEVFormer / LSS lineage** (OpenDriveLab ecosystem + Philion LSS) | Matches vision-first thesis | Implement *tiny* LSS; cite papers; run pretrained only in advanced track. |
| **World models / Vista / ReSim / World Engine** (recent pubs) | Frontier, unstable for intro curricula | “Reading list only” until APIs stabilize; avoid README hero banners. |

**Anti-spam rules for LearnFSD:**
- Cite paper + year + one figure reproduced with attribution; no “we unify UniAD+VAD+DriveLM+Tesla v12.”
- Separate **Core** (implementable in a weekend) vs **Frontier** (read + run official checkpoints).
- Never equate completing course modules with “SOTA on nuScenes.”

---

## 4. Other public FSD / AV courses worth mirroring

| Course | URL | Does well | Does poorly / caveats |
|---|---|---|---|
| **U of T Self-Driving Cars Specialization** | https://www.coursera.org/specializations/self-driving-cars | Full stack; CARLA; strong localization/perception/planning sequence; academic rigor | Paid; heavy prerequisites; not vision-only E2E; CARLA hardware requirements |
| **Udacity SDC Nanodegree** | https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd0013 | Project portfolio; Waymo data exposure; C++ + Python realism | Cost; uneven project freshness (reviews mention outdated detection); sim reliability |
| **Duckietown MOOC (ETHx)** | https://www.edx.org/learn/technology/eth-zurich-self-driving-cars-with-duckietown | Sim↔hardware parity; LX design; free audit | Ecosystem lock-in; not highway FSD |
| **MIT 6.S094 + DeepTraffic** | Lectures on YouTube; https://github.com/lexfridman/deeptraffic | Instant feedback competition; low setup | Shallow vs full AV stack; dated materials |
| **MIT OCW 16.485 VNAV** | https://ocw.mit.edu/courses/16-485-visual-navigation-for-autonomous-vehicles-vnav-fall-2020/ | Grad-level geometry + labs + project report culture | Assumes C++/estimation background; not self-paced MOOC polish |
| **Underactuated Robotics** | https://underactuated.csail.mit.edu/ | Best-in-class interactive notes + Drake | Control-heavy, not perception-FSD |
| **freeCodeCamp Perception for SDC** | https://www.freecodecamp.org/news/perception-for-self-driving-cars-deep-learning-course/ | Free; YOLO/DeepSORT/BEV topics; real datasets in demos | ~2h survey depth; not a sequenced curriculum |
| **CS231n** | https://cs231n.github.io/ | Gold standard CV assignments from scratch | Not AV-specific; must bridge to driving |
| **Donkey Car docs / community** | http://docs.donkeycar.com/ | Hardware-first behavioral cloning path; simulator | Hobby RC ≠ urban AV; limited theory |
| **comma / openpilot (learn-by-reading-code)** | https://github.com/commaai/openpilot · https://docs.comma.ai/CARS/ | Real vision-based driving stack in production-ish form | Not a course; safety/legal constraints; car+device cost |
| **Sensor Fusion Nanodegree / related Udacity** | Udacity SFND repos e.g. https://github.com/udacity/SFND_Lidar_Obstacle_Detection | Excellent lidar/radar pedagogy | Off-axis for vision-first thesis |

**Mirroring priority for LearnFSD:** U of T ordering + Duckietown LX packaging + CS231n assignment hardness + OpenDriveLab reading track + Donkey/webcam as hardware lab — not Udacity price or Tesla marketing voice.

---

## 5. Practical hardware ladder (software-first, not EE student)

Approximate USD street prices as of research date; verify before publishing. Skills assume Python comfort.

| Tier | What | Approx. cost | Skills needed | Learning payoff | Caveats |
|---|---|---|---|---|---|
| **0. Laptop-only** | Colab/local PyTorch; nuScenes mini; CARLA if GPU; course sims | **$0–$50** (storage); GPU optional via Colab | Python, git, basic Linux | 95% of algorithms | CARLA needs decent GPU for local |
| **1. Webcam desk rig** | 1–3 USB webcams + tape lanes + toys | **$15–$60** | OpenCV `VideoCapture`, chessboard calibration | Intrinsics/extrinsics, IPM, latency | Sync/multi-cam hard; lighting sensitivity |
| **2a. Cheap RC + Pi (DIY Donkey)** | RC chassis + Raspberry Pi + camera + PCA9685 | **$250–$300** parts ([Donkey docs](http://docs.donkeycar.com/)) | Flash SD, SSH, `myconfig.py`, collect bags, train CNN | Behavioral cloning loop end-to-end | Debugging electromechanical issues; not “zero electronics,” but no soldering if kitized |
| **2b. Prebuilt PiRacer-class** | Waveshare PiRacer Pro | **~$194–$262** ([Donkey docs recommendation](http://docs.donkeycar.com/)) | Same as 2a, less assembly | Fastest path to moving robot | Still hobby-scale dynamics |
| **2c. JetRacer / Jetson Nano RC** | NVIDIA JetRacer-class | **~$400–$600** typical DIY | Jetson flash, TensorRT basics | Onboard DNN inference | Parts availability varies |
| **3. Duckiebot-ish** | Duckiebot DB-J kit | **$429** list ([Duckietown store](https://get.duckietown.com/products/duckiebot-db21)); city/track kits extra | Docker/ROS, assembly 3–4 h DIY | Curriculum-aligned autonomy in model city | Cost; ROS learning curve; not street-legal FSD |
| **4. comma-like real car** | comma four + harness | **$999** device ([comma shop](https://comma.ai/shop/comma-four)) + harness; **requires compatible car** ([supported cars](https://docs.comma.ai/CARS/)) | Read openpilot docs, safety discipline, SSH, possibly carporting | Closest “vision drives car” experience | Legal/insurance/driver responsibility; **not** a beginner lab; software does not ship controlling a car out of box per shop FAQ |

**Recommended cheapest path for LearnFSD audience:**  
`Tier 0 (required) → Tier 1 webcam weekend ($30) → Tier 2b prebuilt RC (~$250) → optional Duckiebot or comma only if student already has compatible interests/car.`

**Do not** market comma as “course hardware” equivalent to a webcam — different liability and prerequisites.

---

## 6. How strong courses make math accessible

Observed patterns to copy:

1. **Derive in-place, then link out**  
   - Karpathy: write the derivative code, then finite-difference check (micrograd lecture).  
   - Ng: show chain-rule intuition in slides; optional deeper notes; point to 3Blue1Brown for visual calc.  
   - Thrun: Bayes update on a discrete grid *before* Kalman equations.

2. **Two-track math callouts**  
   - Box in notes: “Intuition (required)” vs “Derivation (optional Tier 3)” — Underactuated and Duckietown both do layered depth.  
   - LearnFSD already *names* JIT tiers; enforce them with collapsible sections / separate notebooks, not parallel essay files students skip.

3. **Symbol → shape → unit test**  
   - CS231n: implement loss, then `grad_check`.  
   - Coursera: assert shapes.  
   - Pattern: every equation followed by `assert tensor.shape == ...` and one numeric smoke test.

4. **Worked micro-examples before general form**  
   - 1D localization before SE(3); pinhole with numbers before full camera matrix; constant-velocity KF before EKF.

5. **External tutorial whitelist (point, don’t rewrite)**  
   - 3Blue1Brown Essence of Linear Algebra / Calculus  
   - Barfoot chapters for estimation (VNAV recommends)  
   - Official PyTorch tensor tutorials  
   - Specific paper appendices (LSS depth distribution)  
   Avoid rewriting MITx probability inside the AV course.

6. **Diagnostic tables tied to observables**  
   - Ng-style symptom→cause is good **iff** the notebook can reproduce the symptom (e.g., turn off ReLU and show collapse). LearnFSD tables currently often lack a reproducible trigger cell.

---

## 7. Mobile-friendly delivery for code-heavy STEM

What works in practice:

| Pattern | Why it works | Limits |
|---|---|---|
| **Short vertical video / clip per concept** (3–8 min) + separate desktop lab | Mobile for watching/commute; desktop for coding (Coursera/edX model) | Don’t pretend phone = IDE |
| **Read-only lecture HTML that is responsive** (Underactuated-style; LearnFSD visual_explainers) | Math + diagrams on phone | Interaction ≠ training networks |
| **Colab / cloud notebooks with “Open in Colab” badges** | Phone can *kick off* runs; results checked later | Editing on phone is miserable |
| **Exercise checklist + flashcards / quizzes in mobile app** | Spaced recall for definitions (pinhole, IoU, bicycle model) | Not a substitute for projects |
| **Progressive disclosure**: summary card → full derivation link | Reduces scroll fatigue | Must not hide required labs behind walls of prose |
| **Downloadable PDF/EPUB of notes** for offline | Transit learning | Keep equations as images carefully |
| **Native Jupyter apps (Juno etc.)** | Power users only ([Juno](https://juno.sh/)) | Not the default path |

**Anti-patterns:** 200-line markdown essays as the only mobile experience; requiring local CARLA on a phone; autoplaying heavy WebGL explainers on cellular.

**Recommended LearnFSD packaging:**  
`Mobile: explainers + quizzes + 5-min videos` / `Desktop: notebooks + modules + hardware labs` / sync progress via git or LMS.

---

## Implications for LearnFSD redesign

- **Kill credibility debt:** Drop or quarantine “production-grade / PhD track / Tesla FSD” claims until projects use real datasets, rubrics, and published metrics; rename tiers honestly (Hobbyist → Practitioner → Research reading).
- **Invert essay/notebook ratio:** Cap lecture docs; expand each chapter to CS231n-hardness notebooks (many code cells, grad checks, real images); treat HTML explainers as optional previews.
- **Replace `randn` demos with photons:** nuScenes mini / webcam capture / KITTI samples as default; synthetic only for unit tests of geometry.
- **Adopt LX packaging (Duckietown):** Each module = video + notes + notebook activities + graded exercise + optional robot challenge; sim track complete without hardware.
- **Project spine like Udacity/U of T:** 5–7 portfolio projects with autograder + human rubric (calibration, multitask perception on real frames, tiny LSS, tracker with noise, planner metric, closed-loop sim).
- **OpenDriveLab as reading+benchmark track:** UniAD philosophy, VAD vectorization, DriveLM reasoning homework, survey paper; advanced students run official code — course does not rebrand their SOTA.
- **Hardware ladder in README must match store prices and skills:** $0 → ~$30 webcam → ~$250 PiRacer/Donkey → $429 Duckiebot optional → $999 comma only with loud safety/legal warnings.
- **Math pattern:** in-notebook derivation + finite-difference/shape asserts + outbound links; delete orphaned Tier-3 essays that aren’t executed.
- **Mobile:** ship responsive explainers + quizzes; keep coding desktop/Colab-first; stop implying phone completes STEM labs.
- **Keep what’s working:** module `break_it_fix_it.py`, pytest hooks, vision-first curriculum order (geometry → multitask → BEV → track → plan → control), and JIT *idea* — rebuild the delivery so claims match artifacts.

### Key primary URLs (quick index)
- Course repo: https://github.com/vvknyn/self-driving-ai-course  
- Karpathy Zero to Hero: https://karpathy.ai/zero-to-hero.html  
- CS231n A1: https://cs231n.github.io/assignments2019/assignment1/  
- Udacity SDC: https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd0013  
- U of T Coursera: https://www.coursera.org/specializations/self-driving-cars  
- Duckietown MOOC: https://www.edx.org/learn/technology/eth-zurich-self-driving-cars-with-duckietown  
- OpenDriveLab: https://opendrivelab.com/ · UniAD https://github.com/OpenDriveLab/UniAD · DriveLM https://github.com/OpenDriveLab/DriveLM/  
- Donkey Car: http://docs.donkeycar.com/  
- Duckiebot store: https://get.duckietown.com/products/duckiebot-db21  
- comma four: https://comma.ai/shop/comma-four  
- Underactuated: https://underactuated.csail.mit.edu/  
- MIT VNAV OCW: https://ocw.mit.edu/courses/16-485-visual-navigation-for-autonomous-vehicles-vnav-fall-2020/pages/syllabus/
