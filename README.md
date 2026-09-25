# Vision-First Autonomous Driving: Building a Mini-FSD Stack from Scratch

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Interactive Visuals](https://img.shields.io/badge/Visual_Explainers-Interactive_HTML%2FSVG-green.svg)](visual_explainers/index.html)

A hands-on, production-grade course and open-source codebase for mastering modern, **vision-centric autonomous driving** — modeled directly after **Tesla's Full Self-Driving (FSD) architecture** (multi-camera geometry, HydraNet multi-task perception, Lift-Splat-Shoot Bird's-Eye-View projection, 3D occupancy networks, vector space tracking, and learned trajectory planning).

---

## 🎯 Pedagogical Philosophy

This course marries the best educational principles from the pioneers of AI and robotics:

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  │   Andrew Ng's Rigor             Sebastian Thrun's Robotics             │
  │   - Mathematical clarity        - Kinematic bicycle dynamics           │
  │   - Explicit tensor dimensions  - Stanley & MPC controllers            │
  │   - Assert-based unit tests     - Kalman state estimation              │
  │                                                                        │
  │                    \                  /                                │
  │                     ▼                ▼                                 │
  │            ┌──────────────────────────────────┐                        │
  │            │     MINI-FSD COURSE ENGINE       │                        │
  │            └──────────────────────────────────┘                        │
  │                     ▲                ▲                                 │
  │                    /                  \                                │
  │                                                                        │
  │   FastAI Intuitive Code-First   Tesla FSD Vision Architecture          │
  │   - Run & see output in 60s     - Zero-LiDAR multi-camera rig          │
  │   - "Break It & Fix It" drills  - BEV metric space + 3D Occupancy      │
  │   - Concepts earned on real data- Vector space scene graph             │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘
```

1. **FastAI Code-First Flow**: Every module opens with a runnable script that immediately outputs visual driving predictions. No gatekeeping theory before seeing working code.
2. **Break It & Fix It**: Every module features an intentional bug drill (e.g. calibration pitch perturbation, loss gradient starvation, occupancy memory amnesia, controller latency fishtailing). You break it, observe the failure mode, and engineer the fix.
3. **Andrew Ng Diagnostic Clarity**: Every equation is paired with tensor dimensions `(B, N_cams, C, H, W) -> (B, X, Y, Z, C)`, coordinate frame conventions, and unit tests.
4. **Sebastian Thrun Control Foundations**: Grounded in vehicle kinematics, path lattices, cost-map evaluations, and Stanley/MPC steering control.
5. **Interactive Visual Explainers**: Dedicated, zero-dependency browser-based visual animations and interactive simulators for every major mathematical concept.

---

## 🗺️ Curriculum & Architecture Roadmap

The course is structured into 9 modules (Module 0 through 8). Each module is a **fully functional, standalone project**, yet all modules chain together to form the unified end-to-end Mini-FSD pipeline.

| Module | Title | Key Concepts | Hands-On Project | Interactive Visual Explainer |
|---|---|---|---|---|
| **00** | **The Driving ML Gym** | Tensors, autograd, backpropagation, focal loss, transfer learning, data loaders | Fine-tune a real-time driving scene classifier on multi-class road crops | [`00_ml_gym`](modules/00_ml_gym/) |
| **01** | **Camera Rig & Multi-View Geometry** | Pinhole model, intrinsics $K$, extrinsics $[R \mid T]$, lens distortion, Inverse Perspective Mapping (IPM) | Calibrate a 3-camera rig (Front, Left, Right) & warp road to metric ground plane | [`01_camera_geometry_ipm.html`](visual_explainers/01_camera_geometry_ipm.html) |
| **02** | **Semantic Perception (HydraNet)** | Shared backbone, branched multi-task heads, multi-task uncertainty loss weighting | Build a 4-head driving network (lanes, vehicles, freespace, traffic lights) | [`02_hydranet_architecture.html`](visual_explainers/02_hydranet_architecture.html) |
| **03** | **Bird's-Eye View (BEV) Transformation** | Monocular depth ambiguity, Lift-Splat-Shoot (LSS), frustum voxelization, vertical pooling | Lift 2D multi-camera features into a top-down metric BEV feature map | [`03_lift_splat_shoot_bev.html`](visual_explainers/03_lift_splat_shoot_bev.html) |
| **04** | **3D Occupancy & Temporal Dynamics** | 3D voxel grids, Spatiotemporal GRU fusion, dynamic motion flow vectors $(\Delta x, \Delta y, \Delta z)$ | Train a temporal occupancy network to track visible and occluded obstacles | [`04_occupancy_network_3d.html`](visual_explainers/04_occupancy_network_3d.html) |
| **05** | **Vector Space Tracking & HD Maps** | Kalman filtering, Hungarian bipartite matching, cubic spline lane graph extraction | Build an online HD-map-free vector scene graph with persistent object IDs | [`05_vector_space_tracker.html`](visual_explainers/05_vector_space_tracker.html) |
| **06** | **Learned Trajectory Planning** | Quintic polynomial sampling, lattice planners, multi-objective cost evaluation | Generate candidate path fanouts and select optimal collision-free trajectories | [`06_trajectory_planner.html`](visual_explainers/06_trajectory_planner.html) |
| **07** | **Closed-Loop Control & Kinematics** | Bicycle vehicle dynamics, Pure Pursuit, Stanley lateral control, Model Predictive Control (MPC) | Drive through complex obstacle courses and highway merges in closed-loop sim | [`07_closed_loop_simulator.html`](visual_explainers/07_closed_loop_simulator.html) |
| **08** | **Capstone: Full Mini-FSD Integration** | End-to-end pipeline wiring, sensor-to-actuation streaming, failure diagnostics | Run the complete multi-cam → BEV → Occupancy → Planner → Controller pipeline | [`08_fsd_dashboard.html`](visual_explainers/08_fsd_dashboard.html) |

---

## ⚡ Quick Start

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/vvknyn/self-driving-ai-course.git
cd self-driving-ai-course

# Create or activate your virtual environment (conda or venv)
conda create -n fsd-course python=3.11 -y
conda activate fsd-course

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Entire Test Suite
Verify that all math, tensor operations, geometry projections, and planners pass their unit tests:
```bash
python scripts/run_all_tests.py
```

### 3. Launch the Interactive Visual Explainer Portal
Open the web-based interactive visual portal in your browser:
```bash
python scripts/launch_portal.py
# Opens http://localhost:8000/visual_explainers/index.html
```

---

## 🚀 How Each Session Works (FastAI Style)

Each module is self-contained under `modules/XX_name/` with four essential files:
1. `run_*.py` / `train.py`: **Run this first.** Produces visual output and metrics immediately.
2. `break_it_fix_it.py`: **The core intuition drill.** An intentional defect is introduced; your job is to run it, diagnose the symptom, and apply the fix.
3. `tests/test_*.py`: **Andrew Ng verification.** Run `pytest` to verify mathematical and dimensional correctness.
4. `README.md`: **Deep dive.** Contains geometric diagrams, equations with labeled dimensions, and the *"Why Tesla Does It This Way"* design rationale.

---

## 📂 Repository Structure

```
self-driving-ai-course/
├── README.md                          # Master curriculum & setup guide
├── requirements.txt                   # Tested package versions
├── docs/                              # Deep-dive theory articles & mathematical proofs
│   ├── 00_foundations_ml.md           # Autograd, Loss Landscapes & Focal Loss
│   ├── 01_geometry_and_cameras.md     # Pinhole Camera Models & IPM Geometry
│   ├── 02_hydranet_multitask.md       # Multi-Task Learning & GradNorm
│   ├── 03_bev_transformation.md       # Lift-Splat-Shoot & Spatial Cross-Attention
│   ├── 04_occupancy_networks.md       # 3D Voxel Occupancy & Temporal Dynamics
│   ├── 05_vector_space_tracking.md    # Kalman Filtering & Vector Lane Graphs
│   ├── 06_trajectory_planning.md      # Quintic Splines & Cost-Map Optimization
│   ├── 07_control_and_kinematics.md   # Stanley, Pure Pursuit, and MPC Formulations
│   └── 08_fsd_system_architecture.md  # End-to-End System Integration & FSD V12
├── modules/                           # Standalone executable projects
│   ├── 00_ml_gym/                     # FastAI ML Primer for Autonomous Driving
│   ├── 01_camera_geometry/            # Multi-Camera Rig Calibration & IPM
│   ├── 02_hydranet/                   # Shared Backbone Multi-Task Perception
│   ├── 03_bev_transform/              # Lift-Splat-Shoot 2D-to-BEV Network
│   ├── 04_occupancy_network/          # 3D Spatiotemporal Occupancy Network
│   ├── 05_vector_space/               # Multi-Object Tracker & Vector Map Builder
│   ├── 06_trajectory_planner/         # Multi-Modal Trajectory Sampler & Scorer
│   ├── 07_control_sim/                # Closed-Loop Vehicle Sim (Stanley & MPC)
│   └── 08_capstone_fsd/               # Unified End-to-End Mini-FSD Stack
├── visual_explainers/                 # Zero-dependency interactive web apps
│   ├── index.html                     # Visual Portal & Curriculum Hub
│   ├── 01_camera_geometry_ipm.html    # 3D Camera Ray & IPM Distortion Visualizer
│   ├── 02_hydranet_architecture.html  # Animated Multi-Task Feature Flow
│   ├── 03_lift_splat_shoot_bev.html   # 3D Frustum-to-BEV Splatting Animation
│   ├── 04_occupancy_network_3d.html   # Interactive 3D Voxel Grid & Flow Field
│   ├── 05_vector_space_tracker.html   # Vector Map & Kalman Bipartite Matching
│   ├── 06_trajectory_planner.html     # Interactive Cost-Map & Trajectory Tuner
│   ├── 07_closed_loop_simulator.html  # Dynamic Canvas Vehicle Simulator
│   └── 08_fsd_dashboard.html          # Mission Control Multi-Camera FSD Dashboard
└── scripts/
    ├── run_all_tests.py               # One-click test runner across all modules
    └── launch_portal.py               # Local server for visual explainers
```

---

## 🔬 Citations & Recommended Reading
- Philion, J., & Fidler, S. (2020). *Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D.* ECCV 2020.
- Li, Z., et al. (2022). *BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers.* ECCV 2022.
- Kendall, A., Gal, Y., & Cipolla, R. (2018). *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.* CVPR 2018.
- Hoffmann, G. M., Tomlin, C. J., Montemerlo, M., & Thrun, S. (2007). *Autonomous Automobile Trajectory Tracking for Off-Road Driving: Controller Design, Experimental Validation and Issues.* IEEE Transactions on Control Systems Technology.
- Tesla AI Day 2021 & 2022 Presentations (Andrei Karpathy, Ashok Elluswamy, Milan Kovac).
- Hu, Y., et al. (2023). *Planning-oriented Autonomous Driving (UniAD).* CVPR 2023 Best Paper.

---

## 📜 License
MIT License. Built for students, engineers, and researchers passionate about autonomous robotics and vision-first AI.
