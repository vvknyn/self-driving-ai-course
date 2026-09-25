# Module 05: Vector Space Tracking & HD Maps (Online Vector Map Builder)

> "HD maps are a really bad idea. They are brittle and the world changes constantly. The car must see and construct its own HD vector map on the fly." — Elon Musk

---

## 🎯 Purpose & Learning Goals
Waymo and Cruise spend billions driving cars around cities to scan centimeter-accurate LiDAR HD maps beforehand. If a lane is repainted or construction begins tomorrow, the car gets stuck.

Tesla takes the opposite approach: **Zero pre-recorded HD maps**. The vehicle constructs a live, vectorized representation of lanes and dynamic obstacles in real time.

In this module, you will:
1. Connect to **MITx Probability Foundations**:
   - Master the **Kalman Filter as Recursive Bayesian Estimation with Gaussian Random Variables**.
   - Understand prior beliefs, transition likelihoods, and posterior updates via Bayes' Theorem.
2. Build a multi-object tracker that maintains persistent IDs and velocity estimates for surrounding traffic.
3. Solve **Data Association** using the **Hungarian Algorithm** and **Mahalanobis Distance**.
4. Convert raw lane perception masks into smooth, differentiable **Vector Splines** $y(x) = c_0 + c_1 x + c_2 x^2 + c_3 x^3$ ready for trajectory optimization.
5. FastAI "Break It & Fix It": Fix track identity swapping when two vehicles pass closely in an intersection.

---

## 📐 Mathematical Formulation (MITx Probability Connection)

### 1. The Kalman Filter as Bayes' Rule
In MITx Probability, Bayes' rule for continuous random variables is:
$$f_{X \mid Z}(x \mid z) = \frac{f_{Z \mid X}(z \mid x) \cdot f_X(x)}{f_Z(z)}$$

When the prior and observation likelihood are multivariate Gaussians:
- Prior State Belief: $X_{t-1} \sim \mathcal{N}(\mu_{t-1}, \Sigma_{t-1})$
- Kinematic Motion Model: $X_t = F X_{t-1} + W_t$, where $W_t \sim \mathcal{N}(0, Q)$
- Sensor Observation: $Z_t = H X_t + V_t$, where $V_t \sim \mathcal{N}(0, R)$

**Predict Step (Law of Total Probability):**
$$\mu_t^- = F \mu_{t-1}$$
$$\Sigma_t^- = F \Sigma_{t-1} F^T + Q$$

**Update Step (Bayes Conditioning):**
$$K_t = \Sigma_t^- H^T (H \Sigma_t^- H^T + R)^{-1} \quad \text{(Kalman Gain)}$$
$$\mu_t = \mu_t^- + K_t (Z_t - H \mu_t^-) \quad \text{(Posterior Mean)}$$
$$\Sigma_t = (I - K_t H) \Sigma_t^- \quad \text{(Posterior Covariance Contraction)}$$

### 2. Mahalanobis Distance for Data Association
When matching incoming detections to tracks, Euclidean distance ignores covariance uncertainty. We use the **Mahalanobis Distance** (the number of standard deviations the measurement lies from the track's predicted distribution):
$$d_M(z, \mu) = \sqrt{(z - H\mu)^T (H \Sigma^- H^T + R)^{-1} (z - H\mu)}$$
By thresholding $d_M^2 < \chi^2_k(\alpha)$, we mathematically reject improbable false associations!

---

## 🏎️ Why Tesla Does It This Way
- In Tesla AI Day 2021, Andrei Karpathy showed how the "Spatial RNN" produces clean vector splines for highway lane forks and intersections.
- The downstream neural planner needs analytical derivatives (tangent angles, curvature $\kappa = \frac{|y''|}{(1 + y'^2)^{3/2}}$) to calculate lateral steering acceleration limits, which cannot be computed from noisy pixel blobs.

---

## 🧪 Quick Run
```bash
# 1. Run the multi-object tracker and vector map builder
python modules/05_vector_space/run_tracking.py

# 2. Run the Break-It & Fix-It drill (track identity swap vs Mahalanobis gating)
python modules/05_vector_space/break_it_fix_it.py

# 3. Verify unit tests
pytest modules/05_vector_space/tests/test_tracker.py
```

🎨 **Interactive Visual Explainer**: Open [`visual_explainers/05_vector_space_tracker.html`](../../visual_explainers/05_vector_space_tracker.html) to interact with live Kalman covariance ellipses, Hungarian bipartite cost matrices, and lane spline control points!
