# Deep Dive: End-to-End FSD System Architecture & Industry Comparisons

> "The architectural evolution from heuristic rules to deep learning: How 300,000 lines of C++ code were replaced by a single neural network in Tesla FSD V12."

---

## 1. Architectural Comparison: Tesla Vision vs. Waymo / Cruise

| Architectural Dimension | Tesla FSD (Vision-First) | Waymo / Cruise (LiDAR + HD Maps) |
|---|---|---|
| **Primary Sensors** | 8 Surround CMOS Cameras (Vision only) | 5+ LiDARs, 6 Radars, 29 Cameras |
| **Mapping Strategy** | **Zero HD Maps** (Generates vector map on the fly) | **Pre-recorded HD Maps** (Pre-scanned millimeter point clouds) |
| **Compute Hardware** | Custom FSD Chip (Dual NPU, ~144 TOPS HW3 / ~500 TOPS HW4) | Industrial Server Rack (Dual Intel Xeon + Multiple Nvidia GPUs) |
| **Hardware Cost** | $\approx \$1,500$ per vehicle | $\approx \$50,000 - \$100,000+$ per vehicle |
| **Scalability** | Global (drives anywhere without prior mapping) | Geo-fenced to pre-mapped cities |
| **Failure Modes** | Harsh glare, extreme lens occlusion, uncalibrated pitch | Stale HD maps, road construction shifts, rain/fog LiDAR scattering |

---

## 2. Latency Budget & Real-Time Scheduling
In autonomous driving, safety dictates that the perception-to-actuation latency cannot exceed **100 milliseconds** (running at $\ge 20$–$36\text{ Hz}$).

```
0 ms ───────────────► 15 ms ──────────────► 40 ms ──────────────► 55 ms ────────► 65 ms
Camera Exposure       HydraNet Trunk       BEV Lift-Splat        Lattice         Stanley / MPC
& ISP Debayering      & Multi-Heads        & 3D Occupancy        Planner         CAN Actuation
```

---

## 3. The FSD V12 End-to-End Neural Planner Revolution
Prior to FSD V12, Tesla's stack used deep learning for perception, but C++ heuristic state machines for planning:
- 300,000+ lines of handwritten C++ code handled merges, four-way stops, lane cuts, and pedestrian yields.
- Every edge case required adding another `if/else` condition, which often broke five other edge cases!

In **FSD V12 (2024)**, Ashok Elluswamy and the Tesla AI team trained an **End-to-End Neural Planner**:
- Sensor video and vector state enter a large transformer backbone.
- The network directly outputs trajectory waypoint tokens and steering/acceleration commands.
- The model was trained via imitation learning on millions of hours of 5-star human driving video.
- All 300,000 lines of heuristic C++ rules were deleted!

---

## 4. How Your Mini-FSD Capstone Connects
The codebase you built in this course directly mirrors this production blueprint:
1. **Module 01**: Calibrated camera rig $[R \mid T]$ and $K$.
2. **Module 02**: HydraNet shared trunk caching.
3. **Module 03**: Lift-Splat-Shoot categorical depth unprojection.
4. **Module 04**: 3D Occupancy grid with temporal ConvGRU memory.
5. **Module 05**: Online vector lane splines and Kalman tracking.
6. **Module 06**: Quintic lattice trajectory optimization (UniAD paradigm).
7. **Module 07**: Sebastian Thrun Stanley closed-loop tracking.
8. **Module 08**: Full end-to-end integration and telemetry streaming.
