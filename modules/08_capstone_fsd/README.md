# Module 08: Capstone Portfolio Project: The Unified Mini-FSD Stack

> "This is the culmination of your journey: Multi-Camera Vision → BEV Frustum Splatting → 3D Temporal Occupancy → Vector Space Tracking → Learned Trajectory Planning → Closed-Loop Actuation."

---

## 🎯 Purpose & Learning Goals
In Modules 00 through 07, you built each component of an autonomous vehicle from scratch. Now, you wire them into a single, cohesive, production-grade autonomous driving stack.

In this Capstone module, you will:
1. Construct the **End-to-End Mini-FSD Pipeline**:
   ```
   [3 Surround Cameras: Front, Left, Right]
                     │
                     ▼
       [Module 02: HydraNet Shared Trunk]
                     │
                     ▼
       [Module 03: Lift-Splat-Shoot BEV Projection]
                     │
                     ▼
       [Module 04: 3D Temporal Occupancy & ConvGRU]
                     │
                     ▼
       [Module 05: Vector Space Scene Graph & Kalman Tracking]
                     │
                     ▼
       [Module 06: Quintic Lattice Trajectory Optimization]
                     │
                     ▼
       [Module 07: Stanley Steering & PID Closed-Loop Control]
                     │
                     ▼
       [Physical / Simulated Vehicle Actuation (δ, a)]
   ```
2. Benchmark your stack against **Safety & Comfort Metrics**:
   - Zero collisions in dynamic traffic scenarios.
   - Lane-keeping cross-track error $< 0.30\text{ meters}$.
   - Passenger comfort jerk $< 2.5\text{ m/s}^3$.
3. Stream telemetry directly into the **FSD Mission Control Dashboard** in [`visual_explainers/08_fsd_dashboard.html`](../../visual_explainers/08_fsd_dashboard.html).

---

## 🏎️ Why This Is Your Definitive Portfolio Project
Most self-driving courses leave students with disconnected toy notebooks (one for OpenCV edge detection, one for a simple MNIST classifier) that never talk to each other.
- **This Capstone is different.** It is an integrated, vision-first autonomous driving stack modeled directly after Tesla's modern architecture.
- You can explain every coordinate frame transformation, every loss balancing term, every probability distribution, and every controller constraint during interviews.

---

## 🧪 Quick Run
```bash
# 1. Run the end-to-end Mini-FSD drive replay
python modules/08_capstone_fsd/evaluate.py

# 2. Verify complete integration tests
pytest modules/08_capstone_fsd/tests/test_integration.py

# 3. Launch the Mission Control Dashboard
python scripts/launch_portal.py
# Select "08: End-to-End FSD Dashboard" to watch the stack drive live!
```
