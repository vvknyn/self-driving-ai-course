# Module 03: Bird's-Eye View (BEV) Transformation (Lift-Splat-Shoot)

> "In 2020, Tesla made a historic breakthrough: we abandoned operating in 2D image pixels and lifted all multi-camera video into a shared 3D vector Bird's-Eye View space." — Ashok Elluswamy

---

## 🎯 Purpose & Learning Goals
Looking at separate 2D camera images is how traditional ADAS works. But humans and self-driving cars need a unified top-down map of the world to make safe navigation decisions.

In this module, you will:
1. Understand **Monocular Depth Ambiguity**: Why a 2D image cannot uniquely determine 3D location.
2. Implement **Lift-Splat-Shoot (LSS)** (Philion & Fidler, ECCV 2020):
   - **Lift**: Predict a discrete probability distribution over depth bins $P(D = d \mid u, v)$ for every pixel.
   - **Splat**: Unproject camera rays into a 3D metric point cloud of features in the vehicle coordinate frame.
   - **Shoot**: Pool features vertically into a 2D Bird's-Eye View grid (Pillar Pooling).
3. Connect to **MITx Probability Foundations**:
   - The "Lift" step is fundamentally a **Categorical Random Variable** $D \in \{d_1, \dots, d_K\}$ with probabilities $p_k = \text{Softmax}(z_k)$.
   - The 3D feature representation is the expected feature emission along the optical ray: $\mathbb{E}[c(u, v)] = \sum_{k=1}^K P(D = d_k) \cdot c(u, v)$.
4. FastAI "Break It & Fix It": Witness what happens when depth prediction collapses to uniform probabilities (radial smearing disaster) and fix it.

---

## 📐 Mathematical Formulation (MITx Probability Connection)

```
2D Camera Image Feature (C)       Categorical Depth Dist P(D=d_k)
   ┌─────────┐                      ┌────────────────────────┐
   │ c(u, v) │                      │ [0.01, 0.04, 0.85, ...]│
   └────┬────┘                      └───────────┬────────────┘
        │                                       │
        └───────────────┬───────────────────────┘
                        ▼ (Outer Product / Expectation)
        ┌───────────────────────────────────────────────┐
        │  3D Frustum Points: p(u, v, d_k)              │
        │  Shape: (D, H, W, C)                          │
        └───────────────────────┬───────────────────────┘
                                ▼ (Unproject via K and [R|T])
        ┌───────────────────────────────────────────────┐
        │  Splat into 3D Metric Voxel Grid (X, Y, Z)    │
        └───────────────────────┬───────────────────────┘
                                ▼ (Shoot: Pool along Z-axis)
        ┌───────────────────────────────────────────────┐
        │  2D Metric BEV Grid (BEV_X, BEV_Y, C)         │
        └───────────────────────────────────────────────┘
```

### 1. Depth Discretization (The "Lift" Step)
We discretize the continuous depth range $[D_{\text{min}}, D_{\text{max}}]$ into $D$ bins:
$$d_k = D_{\text{min}} + k \cdot \Delta d, \quad k \in \{0, 1, \dots, D-1\}$$
For each 2D feature vector $c \in \mathbb{R}^C$ at pixel $(u, v)$, the network outputs depth logits $\alpha \in \mathbb{R}^D$:
$$P(D = d_k \mid u, v) = \frac{\exp(\alpha_k)}{\sum_{j=1}^D \exp(\alpha_j)}$$
The outer product yields the lifted frustum tensor of shape $(B, D, H, W, C)$:
$$F(d_k, u, v) = P(D = d_k \mid u, v) \otimes c(u, v)$$

### 2. Geometry Splatting
Using camera intrinsics $K$ and extrinsics $[R \mid T]$, each frustum point $(u, v, d_k)$ maps to an exact metric coordinate in vehicle ego space:
$$P_{\text{ego}} = R^T \left( d_k \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} - T \right)$$

### 3. Voxel Pooling (The "Shoot" Step)
Points falling into the same BEV grid cell $(x_{\text{bev}}, y_{\text{bev}})$ are summed and vertically pooled across $Z$:
$$\text{BEV}(x, y) = \sum_{z} \sum_{p \in \text{Voxel}(x, y, z)} F(p)$$

---

## 🏎️ Why Tesla Does It This Way
- In 2020, Tesla Autopilot switched completely from per-image heuristic bounding boxes to multi-camera BEV space.
- Occlusions (e.g. a vehicle visible in the left camera but blocked by a pillar in the front camera) seamlessly merge in the shared BEV grid.
- Downstream modules (occupancy, tracking, trajectory planning) can operate in true metric distance (meters) rather than arbitrary image pixels.

---

## 🧪 Quick Run
```bash
# 1. Run the Multi-Camera Lift-Splat-Shoot transformation
python modules/03_bev_transform/run_bev.py

# 2. Run the Break-It & Fix-It drill (depth collapse radial smearing)
python modules/03_bev_transform/break_it_fix_it.py

# 3. Verify unit tests
pytest modules/03_bev_transform/tests/test_bev.py
```

🎨 **Interactive Visual Explainer**: Open [`visual_explainers/03_lift_splat_shoot_bev.html`](../../visual_explainers/03_lift_splat_shoot_bev.html) to interactively drag depth distribution peaks and watch 3D rays splat into BEV cells!
