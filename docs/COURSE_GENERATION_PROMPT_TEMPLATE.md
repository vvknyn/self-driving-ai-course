# Master Course Generation Meta-Prompt
*A generalized, field-agnostic framework for synthesizing world-class, production-grade, self-contained interactive courses with Just-In-Time (JIT) multi-depth pedagogy.*

---

## How to Use This Prompt
Copy the text in the **Prompt Template** section below into an LLM session (Claude 3.5 Sonnet, GPT-4o, Gemini 1.5 Pro, or Antigravity). Fill in the bracketed variables `[...]` at the top with your target domain. An example filled-in configuration for **Computational Structural Biology & Drug Discovery** is provided at the bottom for reference.

---

```markdown
# MISSION: BUILD THE DEFINITIVE, ZERO-TO-PHD MASTER COURSE & REPOSITORY FOR [FIELD_NAME]

You are tasked with building a world-class, production-grade, fully self-contained interactive course and open-source codebase for:
Target Field: [FIELD_NAME]
Core Focus: [CORE_PARADIGM, e.g., Vision-Centric Autonomous Driving, AlphaFold/Equivariant Drug Design, Quantum Circuit Compilation]

## 1. PEDAGOGICAL ARCHETYPE (THE 5 TITANS PRINCIPLE)
You must synthesize the distinct teaching styles of the field's foremost pioneers into a cohesive curriculum:
1. THE INTUITION & DIAGNOSTICS MASTER (e.g., Andrew Ng): Build rock-solid mental models, intuitive analogies, diagnostic error analysis tables ("Why did this fail?"), and progressive ML/domain gyms.
2. THE FIRST-PRINCIPLES CODER (e.g., Andrej Karpathy): Every core algorithm must be implemented from raw scratch in pure, minimal code (pure Python/NumPy/PyTorch or raw language primitives). Every tensor shape, data structure, and step must be explicitly traced. Zero black-box libraries for the core mechanics.
3. THE FORMAL MATHEMATICIAN & SYSTEM THEORIST (e.g., Sebastian Thrun / Field Pioneer): Every formula must be derived step-by-step from first principles (Bayes' rule, Hamiltonians, conservation laws, Lyapunov stability, etc.). Map formulas directly to rigorous online open courses (e.g., MIT OCW, Stanford Online, edX).
4. THE ACCESSIBLE MAKER / ZERO-GATEKEEPING PIONEER (e.g., MIT Duckietown): Demystify physical implementation. Design a zero-barrier hardware/lab roadmap: Level 1 requires only a standard laptop with synthetic data; Level 2 uses commodity <$50 sensors or kits; Level 3 bridges to professional-grade hardware. No gatekeeping, no prerequisites assumed.
5. THE CUTTING-EDGE SOTA & PHD BENCHMARK PIONEER (e.g., OpenDriveLab, Broad Institute, CERN): Synthesize the 2020–present literature, state-of-the-art benchmarks, multi-task unified architectures, open unsolved PhD research questions, and PhD technical interview preparation.

---

## 2. STRICT TRUTH-FIRST & ANTI-HALLUCINATION PROTOCOL
- NO HALLUCINATING any papers, citations, mathematical proofs, or benchmark metrics.
- Every citation must reference real, peer-reviewed or standard preprint literature with exact author groups, years, and venue names.
- Every mathematical derivation must be mathematically sound. If a simplification is made for computational tractability, explicitly declare the assumptions (e.g., "assuming Gaussian noise", "small-angle approximation").
- Every code implementation must be runnable, fully typed, unit-tested, and verified to run in sub-second test suites without requiring external proprietary data.

---

## 3. JUST-IN-TIME (JIT) 5-TIER MULTI-DEPTH LEARNING MODEL
Every single module across the course must be structured to accommodate learners at any level:
- TIER 1: CORE INTUITION (5–10 min read): High-level concepts, plain-English explanations, visual diagrams, and real-world failure mode tables.
- TIER 2: CODE IMPLEMENTATION FROM RAW SCRATCH: Zero-dependency reference implementation with typed function signatures, shape assertions, and inline educational docstrings.
- TIER 3: DEEP MATHEMATICAL DERIVATION & ACADEMIC LINEAGE: Complete step-by-step formula derivations, LaTeX math blocks, and exact curated links to free MIT/Stanford open course lectures.
- TIER 4: CUTTING-EDGE RESEARCH & PHD ROADMAP: Critical analysis of 2020–present foundational papers, paradigm shifts, open frontier problems, and interview-level theoretical challenges.
- TIER 5: PRACTICAL LAB & HARDWARE BENCHMARK: Step-by-step guide to reproducing the concept on accessible hardware or cloud compute with latency, compute, and memory budgets.

---

## 4. MULTI-MODAL INTERACTIVE VISUAL EXPLAINERS
For each module, generate a standalone, zero-dependency interactive HTML5/Canvas/CSS visual explainer:
- Modern Dark-Themed Aesthetic: Jet-black backgrounds (`#0a0b10`), glassmorphic panels, glowing neon accent colors.
- Interactive Parameter Controls: Real-time sliders, buttons, presets, and diagnostic probes.
- Mobile-Responsive: Viewport meta tag, CSS `aspect-ratio` canvas scaling, and unified touch/pointer event handlers (`touchstart`, `touchmove`, `touchend` mirroring mouse events) so it is silky smooth on smartphones.
- Embedded Diagnostic HUD: Live telemetry displays, frame rates, error metrics, and mathematical state inspection.

---

## 5. COMPLETE COURSE CURRICULUM BLUEPRINT
Organize the course into a logical 8-to-10 module progression:
- Module 00: Domain Gym & Foundations (Linear algebra, coordinate transforms, statistical mechanics, or tensor gyms).
- Module 01–03: Foundational Perceptions & Transformations (Sensors, signal transforms, feature extraction, representations).
- Module 04–05: Unified Latent Space & Multi-Task Representations (3D volumetric/graph/manifold representations, tracking, state estimation).
- Module 06–07: Planning, Control, Optimization & Simulation (Cost functions, optimization, closed-loop dynamics, feedback stability).
- Module 08: End-to-End Capstone System & Mission Control Dashboard (Full integration, latency budgeting, unified evaluator).
- Module 09: Frontier Research, PhD Roadmap & Career Horizons (Causal bottlenecks, foundational world models, open research directions).

---

## 6. DELIVERABLE SPECIFICATIONS
1. CODEBASE: Clean directory structure (`modules/00_...` to `08_...`, `tests/`, `scripts/run_all_tests.py`, `evaluate.py`).
2. DOCUMENTATION: Comprehensive Markdown guides for each module following the 5-Tier JIT standard, plus master hardware and PhD roadmaps.
3. INTERACTIVE SUITE: All HTML5 visual explainers in `visual_explainers/` with an `index.html` hub.
4. HOSTING & EMBEDDING: Ready for one-click deployment to a Next.js / Astro web portal with responsive layouts, iframe modal runners, and subdomain routing.
```

---

## Concrete Example: Computational Structural Biology & Drug Design

Here is how the variables are populated when applying this prompt to another domain:

| Variable | Value for Autonomous Driving (This Course) | Value for Computational Biology & Drug Discovery |
| :--- | :--- | :--- |
| `[FIELD_NAME]` | Vision-First Autonomous Driving (Tesla FSD) | AI for Molecular Biology & Drug Discovery |
| `[CORE_PARADIGM]` | Multi-Camera Pure Vision to BEV Occupancy & Trajectory Planning | Sequence-to-Structure-to-Function (AlphaFold & Equivariant Diffusion) |
| `[TITAN 1: INTUITION]` | Andrew Ng (Error analysis, perception diagnostics) | Eric Lander & David Baker (Biophysical intuitions, energy landscapes) |
| `[TITAN 2: SCRATCH CODE]` | Andrej Karpathy (Raw tensor tracing, NumPy backprop) | John Jumper & Mohammed AlQuraishi (Scratch Evoformer & SE(3) attention) |
| `[TITAN 3: FORMAL MATH]` | Sebastian Thrun (Bayesian filtering, Kalman, Stanley) | Ken Dill & Martin Karplus (Statistical mechanics, partition functions, Langevin) |
| `[TITAN 4: ACCESSIBLE LAB]` | MIT Duckietown ($100 RC car, laptop synthetic camera) | Folding@home & Google Colab (Free T4 GPUs, open PDB files, PyMOL web) |
| `[TITAN 5: CUTTING-EDGE SOTA]`| OpenDriveLab UniAD, Wayve GAIA-1, BEVFormer | AlphaFold 3, RFdiffusion, ESM-3, Chroma, OpenFold |

---

## Course Module Mapping Comparison

| Module # | Autonomous Driving (Tesla FSD) | Computational Structural Biology |
| :--- | :--- | :--- |
| **00** | **ML Gym & Coordinates**: Tensor math, camera matrices, vehicle frames | **Biophysical Gym**: PDB parsing, dihedral angles ($\phi, \psi, \omega$), Ramachandran plots |
| **01** | **Camera Geometry & IPM**: Pinhole cameras, extrinsic $[R \mid T]$, planar homography | **Protein Geometry & SE(3) Invariance**: Rigid body frames, quaternions, Lie groups |
| **02** | **Multi-Task HydraNet**: Shared vision backbone, task heads, GradNorm loss | **Multiple Sequence Alignment (MSA) & Transformers**: Co-evolution, axial attention |
| **03** | **BEV Transformation**: Monocular depth ambiguity, Lift-Splat-Shoot | **Invariant Point Attention (IPA)**: AlphaFold2 structure module, frame updates |
| **04** | **3D Occupancy Networks**: Voxel grids, volume rendering, temporal ConvGRU | **Equivariant Graph Neural Networks (EGNNs)**: Molecular graphs, coordinates, forces |
| **05** | **Vector Space Tracking**: Kalman filters, Mahalanobis distance, Hungarian matching | **Molecular Dynamics & Langevin Integrators**: Velocity Verlet, thermostats, free energy |
| **06** | **Trajectory Planner**: Quintic splines, cost maps, collision avoidance | **Generative De Novo Design**: RFdiffusion, SE(3) diffusion forward/reverse drift |
| **07** | **Control & Kinematics**: Bicycle model, Stanley controller, latency compensation | **Docking & Virtual Screening**: Scoring functions, Monte Carlo pose search, AutoDock Vina |
| **08** | **Capstone FSD System**: End-to-end multi-camera to steering simulation | **Capstone Lead Optimization**: Target protein to de novo generated nanomolar binder |
| **09** | **PhD Frontier Roadmap**: Causal confusion, world models, safety filters | **PhD Frontier Roadmap**: Allosteric dynamics, protein-nucleic acid complexes, lab synthesis |
