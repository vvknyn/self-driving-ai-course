# Just-In-Time (JIT) Multi-Depth Learning Guide

> "Learn as broadly or as deeply as you need, precisely when you need it. No artificial barriers, no hidden math, and zero hand-waving."

---

## 🎯 The Philosophy: Multi-Tier Just-In-Time Learning

In traditional engineering courses, students face a frustrating binary:
1. **The Pop-Science / Marketing Fluff**: High-level hand-waving with cartoon diagrams, zero code, and zero math. You finish the course feeling entertained but incapable of implementing a single algorithm.
2. **The Inaccessible Academic Monograph**: Dense graduate-level textbooks that plunge straight into measure-theoretic probability, differential forms, or unannotated 5,000-line C++ codebases without explaining *why* any of it matters.

This course pioneers the **Just-In-Time (JIT) 5-Tier Architecture**, blending the teaching styles of five legendary educators and open-source programs:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                THE 5 JIT DEPTH TIERS                                        │
├───────────────┬──────────────────────────────┬──────────────────────────────────────────────┤
│ 🟢 Tier 1     │ Intuition & Diagnostics      │ Andrew Ng Style: 30,000-ft mental models,    │
│               │ (Beginner Friendly)          │ analogies, bias/variance diagnostic trees.   │
├───────────────┼──────────────────────────────┼──────────────────────────────────────────────┤
│ 🟡 Tier 2     │ Code From Raw Scratch        │ Andrej Karpathy Style: Pure PyTorch/NumPy,   │
│               │ (Software Engineer)          │ line-by-line tensor shape tracing, no magic. │
├───────────────┼──────────────────────────────┼──────────────────────────────────────────────┤
│ 🔴 Tier 3     │ Mathematical Derivations     │ Sebastian Thrun & MIT 6.041x: Bayesian proof,│
│               │ (Probabilist & Theorist)     │ first-principles calculus, Lyapunov stability│
├───────────────┼──────────────────────────────┼──────────────────────────────────────────────┤
│ 🎓 Tier 4     │ Cutting-Edge Research / PhD  │ OpenDriveLab / CVPR 2023-2026: UniAD, VAD,   │
│               │ (Graduate Researcher)        │ World Models, 3DGS, open research problems.  │
├───────────────┼──────────────────────────────┼──────────────────────────────────────────────┤
│ 🟣 Tier 5     │ Hardware & Edge Robotics     │ MIT Duckietown & Comma AI: Camera latency,   │
│               │ (Hands-on Roboticist)        │ distortion calibration, zero-electronics kit.│
└───────────────┴──────────────────────────────┴──────────────────────────────────────────────┘
```

---

## 🧭 How to Choose Your Depth Path

### Pathway A: The Probability & Math Student (e.g., MITx 6.041x)
- **Your Background**: Comfortable with Bayes' rule, continuous random variables, normal distributions, and matrix multiplication.
- **Your Recommended Path**:
  1. Read **🟢 Tier 1** for the physical driving intuition.
  2. Jump straight to **🔴 Tier 3** to see how every self-driving equation is a direct application of probability axioms (Categorical depth, Bernoulli occupancy fields, Gaussian Kalman conditioning).
  3. Interact with the visual explainers in `visual_explainers/` to watch the probability distributions update in real time.
  4. Skip heavy PyTorch implementation details until you feel curious.

### Pathway B: The Software Engineer / FastAI Hacker
- **Your Background**: Strong Python skills, familiar with PyTorch tensors or NumPy arrays, prefers seeing code run before reading math proofs.
- **Your Recommended Path**:
  1. Read **🟢 Tier 1** for the architectural problem definition.
  2. Immerse yourself in **🟡 Tier 2**: trace tensor shapes `(B, N, C, H, W)` -> `(B, D, X, Y, Z)` in raw PyTorch.
  3. Run the module script: `python modules/XX/run_*.py`.
  4. Deliberately break the system using `break_it_fix_it.py` and debug it.
  5. Consult **🔴 Tier 3** only when you need to understand *why* a particular hyperparameter or formula behaves as it does.

### Pathway C: The Robotics Student (Thrun / Duckietown / ROS)
- **Your Background**: Familiar with kinematic state spaces, coordinate transforms, PID loops, or Duckiebots.
- **Your Recommended Path**:
  1. Focus on **Modules 01 (Geometry)**, **05 (Tracking)**, **07 (Control)**, and **08 (Capstone)**.
  2. Study **🔴 Tier 3** (Lyapunov stability proofs, Frenet frame transformations) and **🟣 Tier 5** (hardware calibration and latency budgets).
  3. Connect your webcam or Duckiebot using the instructions in `docs/00_practical_hardware_guide.md`.

### Pathway D: The Aspiring PhD Applicant / Research Scientist
- **Your Goal**: Apply to top graduate programs (MIT, Stanford, Berkeley, CMU, Oxford) or interview for Autonomous Vehicle Research Scientist roles (Tesla Autopilot, Wayve, Waymo, OpenDriveLab).
- **Your Recommended Path**:
  1. Read every tier across all modules, treating Tier 2 as your baseline implementation capability.
  2. Deeply study **🎓 Tier 4** in each module and read `docs/09_cutting_edge_research_and_phd_roadmap.md`.
  3. Complete the open research extension prompts provided at the end of each module.
  4. Build an end-to-end portfolio project by extending the Module 08 Capstone with a novel contribution (e.g., diffusion trajectory generation, NeRF/3DGS simulation, or Control Barrier Function safety filters).

---

## 🔬 How Each Tier Teaches: A Concrete Example

To understand how JIT depth works in practice, examine how **Depth Estimation in Lift-Splat-Shoot (Module 03)** is taught across all 5 tiers:

### 🟢 Tier 1: Intuition (Andrew Ng Style)
> *The Mental Model*: When you look at a photograph of a red sports car, you can tell it's a car, but you cannot tell whether it is a full-sized Ferrari 50 meters away or a matchbox toy car 50 centimeters away. A 2D camera projects an entire 3D ray of physical points onto a single pixel. To build a top-down Bird's-Eye View (BEV) map, we cannot guess a single depth point—that would cause fatal errors when the model is uncertain. Instead, we cast a net: the neural network assigns a probability score to each distance bin along the ray.

### 🟡 Tier 2: Code From Scratch (Andrej Karpathy Style)
> *The Raw PyTorch Mechanics*: Let's look at the exact tensor operations. We take an image feature map of shape `(B, N, C, H, W)` and a depth logit tensor of shape `(B, N, D, H, W)`. We compute:
```python
# Andrej Karpathy style: inspect every tensor shape
# B: batch size, N: cameras (6), D: depth bins (41), C: feature channels (64)
prob_depth = F.softmax(depth_logits, dim=2)          # Shape: (B, N, D, H, W)
# Outer product along ray: broadcast and multiply
frustum_features = prob_depth.unsqueeze(3) * features.unsqueeze(2) 
# Result: (B, N, D, C, H, W) -> exactly D discrete 3D points per pixel!
```

### 🔴 Tier 3: Mathematical Derivation (Sebastian Thrun / MIT 6.041x Style)
> *The First-Principles Derivation*: The true depth $D$ along optical ray $(u, v)$ is a continuous random variable with conditional probability density function $f_{D \mid U, V}(d \mid u, v)$. We discretize the continuous ray into $K$ mutually exclusive and collectively exhaustive intervals $I_k = [d_k - \frac{\Delta d}{2}, d_k + \frac{\Delta d}{2}]$. By the Law of Total Probability:
$$P(D \in I_k \mid u, v) = \int_{I_k} f_{D \mid U, V}(t \mid u, v) \, dt = \frac{\exp(z_k / T)}{\sum_{j=1}^K \exp(z_j / T)}$$
The expected feature representation $\mathbb{E}[F(x, y, z)]$ pooled in voxel cell $V$ is:
$$\mathbb{E}[F_V] = \sum_{(u, v, k) \in V} P(D \in I_k \mid u, v) \cdot c(u, v)$$

### 🎓 Tier 4: Cutting-Edge Research & PhD Track (CVPR / OpenDriveLab Style)
> *The Research Frontier*: While Philion & Fidler (ECCV 2020) pioneered categorical depth distribution lifting, modern state-of-the-art models like **BEVDepth** (Li et al., AAAI 2023) showed that unguided depth logits suffer from supervision starvation. By introducing LiDAR-assisted depth supervision during training (without requiring LiDAR at test time), depth accuracy improves by 18%. Furthermore, **BEVFormer** (Li et al., ECCV 2022) replaces explicit depth unprojection with deformable cross-attention queries that probe camera images directly, eliminating the memory-heavy $D \times H \times W$ frustum tensor entirely.

### 🟣 Tier 5: Hardware & Edge Robotics (Duckietown / Edge AI Style)
> *The Physical Reality*: On an embedded robot (NVIDIA Jetson Orin Nano, Apple M-series GPU, or Raspberry Pi 5), storing a `(1, 6, 64, 80, 45, 80)` float32 frustum tensor consumes over 1.2 GB of GPU VRAM per forward pass, triggering out-of-memory kernel panics. In real-world edge deployment, we apply 3 optimizations:
> 1. Discretize depth into non-uniform logarithmic bins (dense nearby, sparse far away).
> 2. Quantize features to INT8 using TensorRT or CoreML.
> 3. Use cumulative sum (`torch.cumsum`) voxel pooling to reduce memory traffic by $10\times$.

---

## 📋 Module JIT Navigation Matrix

Use this index to jump directly to any topic at your desired depth tier:

| Module | 🟢 Tier 1: Intuition | 🟡 Tier 2: Code From Scratch | 🔴 Tier 3: Math Derivations | 🎓 Tier 4: PhD Research | 🟣 Tier 5: Hardware |
|---|---|---|---|---|---|
| **00: ML Gym** | [Intuition](00_course_philosophy_and_mit_open_curriculum.md) | [model.py](../modules/00_ml_gym/model.py) | [Focal Loss Proof](00_mathematical_foundations_and_tutorial_guide.md) | Class-imbalance in NuScenes | MPS / CUDA Benchmark |
| **01: Camera Geometry** | [Pinhole & IPM](01_geometry_and_cameras.md#1-the-pinhole-camera-model) | [calibrate_rig.py](../modules/01_camera_geometry/calibrate_rig.py) | [Similar Triangles & Homography](00_mathematical_foundations_and_tutorial_guide.md#1-pinhole-projection-from-similar-triangles-module-01) | Online Self-Calibration Nets | Desk Webcam Calibration |
| **02: HydraNet** | [Multi-Task Bottle](02_hydranet_multitask.md#1-the-multi-task-computational-bottleneck) | [backbone.py & heads.py](../modules/02_hydranet/backbone.py) | [Homoscedastic MLE Derivation](00_mathematical_foundations_and_tutorial_guide.md#3-multi-task-uncertainty-loss-via-maximum-likelihood-module-02) | Gradient Surgery (PCGrad) | MobileNetV4 / EdgeTPU |
| **03: BEV Transform** | [The 2D to 3D Ray](03_bev_transformation.md#1-the-fundamental-dilemma-of-monocular-vision) | [lift_splat_shoot.py](../modules/03_bev_transform/lift_splat_shoot.py) | [Categorical Softmax Expectation](00_mathematical_foundations_and_tutorial_guide.md#4-lift-splat-shoot-expectation-weighting-module-03) | BEVFormer & MatrixVT | Quick Cumsum Pooling |
| **04: 3D Occupancy** | [Voxel World & Flow](04_occupancy_networks.md#1-why-3d-boxes-fail) | [voxel_grid.py](../modules/04_occupancy_network/voxel_grid.py) | [Bernoulli Random Fields](00_course_philosophy_and_mit_open_curriculum.md#3-mitx-probability-connection) | Tesla AI Day 2022 Occupancy | Sparse Conv (Minkowski) |
| **05: Vector Tracker** | [Hungarian & Swaps](05_vector_space_tracking.md#1-the-data-association-problem) | [kalman_tracker.py](../modules/05_vector_space/kalman_tracker.py) | [Kalman Gaussian & Curvature](00_mathematical_foundations_and_tutorial_guide.md#5-kalman-filter-state-update--covariance-contraction-module-05) | Graph Neural Net Trackers | Low-Latency Matrix Math |
| **06: Trajectory Plan** | [Lattice Fanout](06_trajectory_planning.md#1-frenet-coordinate-frame) | [lattice_planner.py](../modules/06_trajectory_planner/lattice_planner.py) | [Quintic BVP Analytical Matrix](00_mathematical_foundations_and_tutorial_guide.md#7-quintic-boundary-value-problem-module-06) | Neural Planners & UniAD | Real-Time C++ Cost Maps |
| **07: Control Sim** | [Bicycle Steering](07_control_and_kinematics.md#1-the-kinematic-bicycle-model) | [controllers.py](../modules/07_control_sim/controllers.py) | [Lyapunov Stability Proof](00_mathematical_foundations_and_tutorial_guide.md#8-stanley-controller-exponential-stability-proof-module-07) | Model Predictive Path Int. | Steering Actuator Delays |
| **08: Capstone FSD** | [End-to-End Stack](08_fsd_system_architecture.md#1-the-unified-information-highway) | [pipeline.py](../modules/08_capstone_fsd/pipeline.py) | [Integrated Probabilistic Flow](08_fsd_system_architecture.md#3-mathematical-derivation-the-perception-action-markov-chain) | Foundation Models (GAIA-1) | 80 Hz Hardware Budget |

---

## 🏆 Self-Assessment Rubric: Are You PhD / FAANG Ready?

Before you finish the course, test your depth across all dimensions:

1. **Intuition Test (Ng Style)**: Can you explain to a non-engineer why driving pitch changes cause an autonomous vehicle to misestimate distances to obstacles?
2. **Implementation Test (Karpathy Style)**: Can you implement a vectorized LSS frustum pooling operation without using nested Python `for` loops in under 50 lines of PyTorch?
3. **Mathematical Proof Test (Thrun Style)**: Can you derive the Stanley steering law from the kinematic bicycle error dynamics and prove that cross-track error converges exponentially using a Lyapunov function?
4. **Research Mastery Test (OpenDriveLab Style)**: Can you critique the trade-offs between dense 3D occupancy grids and sparse vectorized map queries (VAD/MapTR), explaining when each representation dominates?
5. **Practical Robotics Test (Duckietown Style)**: Can you calibrate a distorted pinhole camera using a printed checkerboard, calculate metric cross-track error, and run closed-loop control under 150 ms of actuator latency?
