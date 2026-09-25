# Module 04: 3D Occupancy Networks & Temporal Dynamics

> "The real world doesn't come with 3D bounding boxes. It is full of arbitrary shapes—curved barriers, semi-trailers, debris. With Occupancy Networks, we predict the 3D geometry of everything." — Ashok Elluswamy, Tesla AI Day 2022

---

## 🎯 Purpose & Learning Goals
In traditional robotics, perception tries to fit nice rectangular 3D boxes around known objects. But when a truck carries an irregularly shaped pole or a construction zone has weird barricades, box detectors fail catastrophically.

In this module, you will:
1. Discretize the 3D world around the car into a **3D Voxel Grid** (e.g. $0.4\text{m} \times 0.4\text{m} \times 0.4\text{m}$).
2. Predict two properties for each voxel:
   - **Occupancy Probability**: $P(\text{occupied} = 1)$ (a Bernoulli random variable).
   - **Velocity Flow Vector**: $\vec{v} = (\Delta x, \Delta y, \Delta z)$ (how fast this voxel is moving).
3. Connect to **MITx Probability Foundations**:
   - Each voxel is a **Bernoulli Trial** with parameter $p = \sigma(z) \in [0, 1]$.
   - Temporal fusion acts as a **Recursive Bayesian Filter**: updating the prior occupancy probability with new observation likelihoods across video frames.
4. Build a **Spatiotemporal ConvGRU**:
   - Single-frame models suffer from **"Occlusion Amnesia"**—if a pedestrian walks behind a parked truck, a single-frame detector thinks they vanished into thin air!
   - A temporal recurrent memory unit preserves the hidden state across time, carrying forward occluded objects and projecting them along their velocity vectors.

---

## 📐 Mathematical Formulation

### 1. 3D Voxel Representation
Let vehicle ego space be bounded by:
$$X \in [X_{\text{min}}, X_{\text{max}}], \quad Y \in [Y_{\text{min}}, Y_{\text{max}}], \quad Z \in [Z_{\text{min}}, Z_{\text{max}}]$$
Discretized with voxel resolution $\Delta s = 0.4\text{m}$, creating a 3D grid of shape $(N_x, N_y, N_z)$.

### 2. Spatiotemporal Recurrent Update (ConvGRU)
At frame $t$, the current perception features $X_t$ and previous hidden memory state $H_{t-1}$ are fused:
$$\text{Reset Gate: } R_t = \sigma(W_r * X_t + U_r * H_{t-1})$$
$$\text{Update Gate: } Z_t = \sigma(W_z * X_t + U_z * H_{t-1})$$
$$\text{Candidate State: } \tilde{H}_t = \tanh(W_h * X_t + U_h * (R_t \odot H_{t-1}))$$
$$\text{New Hidden Memory: } H_t = (1 - Z_t) \odot H_{t-1} + Z_t \odot \tilde{H}_t$$

### 3. Dual-Head Prediction
From the updated temporal memory $H_t$:
$$\text{Occupancy Logits: } \hat{O}_t = \text{Conv}_{\text{occ}}(H_t) \in \mathbb{R}^{N_x \times N_y \times N_z}$$
$$\text{Velocity Flow: } \vec{V}_t = \text{Conv}_{\text{vel}}(H_t) \in \mathbb{R}^{3 \times N_x \times N_y \times N_z}$$

---

## 🏎️ Why Tesla Does It This Way
- Ashok Elluswamy revealed at AI Day 2022 that Tesla trains Occupancy Networks using **NeRF-style volume rendering** across billions of fleet video clips without manual labeling.
- The occupancy grid directly feeds Tesla's trajectory planner, enabling safe braking for unknown debris, fallen trees, and odd construction vehicles where traditional classifiers fail.

---

## 🧪 Quick Run
```bash
# 1. Run the temporal occupancy network through an occlusion scenario
python modules/04_occupancy_network/run_occupancy.py

# 2. Run the Break-It & Fix-It drill (occlusion amnesia vs temporal memory)
python modules/04_occupancy_network/break_it_fix_it.py

# 3. Verify unit tests
pytest modules/04_occupancy_network/tests/test_occupancy.py
```

🎨 **Interactive Visual Explainer**: Open [`visual_explainers/04_occupancy_network_3d.html`](../../visual_explainers/04_occupancy_network_3d.html) to interact with a 3D orbital voxel renderer and slice through occupancy layers!
