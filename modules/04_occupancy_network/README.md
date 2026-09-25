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

## 📐 How the Formulas Are Obtained

### 1. Spatial Bernoulli Random Field
We treat 3D physical space as a collection of independent binary random variables:
$$O(i_x, i_y, i_z) \sim \text{Bernoulli}(p), \quad p = \sigma(z) = \frac{1}{1 + \exp(-z)}$$
Where $z$ is the unnormalized network logit.
The probability mass function is:
$$P(O = o) = p^o (1 - p)^{1 - o}, \quad o \in \{0, 1\}$$
Binary Cross-Entropy loss is the exact negative log-likelihood of this Bernoulli distribution:
$$\mathcal{L}_{\text{BCE}} = -[o \log p + (1 - o) \log(1 - p)]$$

### 2. Spatiotemporal ConvGRU Gating Equations
A standard GRU operates on 1D vectors. A **ConvGRU** replaces matrix multiplications with 2D/3D spatial convolutions ($*$) so temporal memory retains spatial topology:
$$\text{Reset Gate: } R_t = \sigma(W_r * X_t + U_r * H_{t-1})$$
$$\text{Update Gate: } Z_t = \sigma(W_z * X_t + U_z * H_{t-1})$$
$$\text{Candidate Memory: } \tilde{H}_t = \tanh(W_h * X_t + U_h * (R_t \odot H_{t-1}))$$
$$\text{Updated Memory: } H_t = (1 - Z_t) \odot H_{t-1} + Z_t \odot \tilde{H}_t$$

### 3. Bayesian Filtering Interpretation (MITx Probability)
- The update gate $Z_t \in [0, 1]$ acts as a dynamic **Kalman Gain**:
  - When $Z_t \approx 0$: the network ignores the noisy new visual frame and relies 100% on prior memory $H_{t-1}$ (critical during camera occlusion or sensor glare).
  - When $Z_t \approx 1$: the network overwrites memory with fresh visual evidence.

---

## 📺 Recommended Free Video Tutorials to Learn the Math
- **Bernoulli & Binomial Random Variables**: [MIT 6.041x: Bernoulli Processes](https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/)
- **Recurrent Neural Networks & GRU Gates**: [StatQuest: Recurrent Neural Networks (RNNs) and GRUs](https://www.youtube.com/watch?v=LHXXI4-IEns)
- **Understanding LSTM/GRU Architectures**: [Christopher Olah: Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

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
