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

## 📐 How the Formulas Are Obtained

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

### 1. The "Lift" Step as Probabilistic Expectation
A single pixel $(u, v)$ with feature vector $c(u, v) \in \mathbb{R}^C$ corresponds to an entire optical ray in 3D space.
We discretize the continuous depth range $[D_{\text{min}}, D_{\text{max}}]$ into $D$ bins:
$$d_k = D_{\text{min}} + k \cdot \Delta d, \quad k \in \{0, 1, \dots, D-1\}$$

For each feature vector, the network predicts depth logits $\alpha \in \mathbb{R}^D$. Applying Softmax yields categorical probabilities:
$$P(D = d_k \mid u, v) = \frac{\exp(\alpha_k)}{\sum_{j=1}^D \exp(\alpha_j)}$$

By the **Law of Total Probability**, the expected feature representation along the ray at distance $d_k$ is the outer product:
$$F(d_k, u, v) = P(D = d_k \mid u, v) \otimes c(u, v) \in \mathbb{R}^{D \times H \times W \times C}$$

### 2. Geometry Splatting Equation
Each frustum point $(u, v, d_k)$ is unprojected into the camera coordinate frame:
$$P_c = d_k \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$
And transformed into the vehicle ego frame via extrinsic inverse $[R \mid T]^{-1}$:
$$P_{\text{ego}} = R^T (P_c - T)$$

### 3. Voxel Pooling ("Shoot")
Points falling into the same BEV grid cell $(x_{\text{bev}}, y_{\text{bev}})$ are summed and vertically pooled across $Z$:
$$\text{BEV}(x, y) = \sum_{z} \sum_{p \in \text{Voxel}(x, y, z)} F(p)$$

---

## 📺 Recommended Free Video Tutorials to Learn the Math
- **Linear Transformations & Inverses**: [3Blue1Brown: Inverse Matrices & Column Space](https://www.youtube.com/watch?v=uQhTuRlWM3E)
- **Categorical & Discrete Random Variables**: [MIT 6.041x: Discrete Probability Distributions](https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/)
- **Lift-Splat-Shoot Paper Presentation**: [Jonah Philion: ECCV 2020 Talk on LSS](https://www.youtube.com/watch?v=sobq_z6y-i8)

---

## 🏎️ Why Tesla Does It This Way
- In 2020, Tesla Autopilot switched completely from per-image heuristic bounding boxes to multi-camera BEV space.
- Occlusions seamlessly merge in the shared BEV grid.
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
