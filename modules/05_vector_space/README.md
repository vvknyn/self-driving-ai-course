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

## 📐 How the Formulas Are Obtained

### 1. Step-by-Step Derivation of the Kalman Filter (Bayes' Rule with Gaussians)
In MITx Probability, Bayes' rule states:
$$p(x \mid z) = \frac{p(z \mid x) p(x)}{p(z)} \propto p(z \mid x) \cdot p(x)$$

Let prior belief be Gaussian: $x \sim \mathcal{N}(\mu^-, P^-)$, with PDF:
$$p(x) \propto \exp\left( -\frac{1}{2} (x - \mu^-)^T (P^-)^{-1} (x - \mu^-) \right)$$
And measurement likelihood be linear Gaussian $z = Hx + v$, with $v \sim \mathcal{N}(0, R)$:
$$p(z \mid x) \propto \exp\left( -\frac{1}{2} (z - Hx)^T R^{-1} (z - Hx) \right)$$

Multiplying the two PDFs adds their exponents:
$$J(x) = (x - \mu^-)^T (P^-)^{-1} (x - \mu^-) + (z - Hx)^T R^{-1} (z - Hx)$$

By completing the square with respect to $x$:
The quadratic term in $x$ gives the **posterior covariance inverse (precision matrix)**:
$$P^{-1} = (P^-)^{-1} + H^T R^{-1} H$$

Applying the **Woodbury Matrix Identity** $(A + U C V)^{-1} = A^{-1} - A^{-1} U (C^{-1} + V A^{-1} U)^{-1} V A^{-1}$:
$$P = (I - K H) P^-, \quad \text{where } K = P^- H^T (H P^- H^T + R)^{-1}$$

Setting $\nabla_x J(x) = 0$ yields the **posterior mean update**:
$$\mu = \mu^- + K (z - H \mu^-)$$

### 2. Derivation of Analytical Road Curvature $\kappa(x)$
Curvature $\kappa$ is defined as the magnitude of heading angle change per unit arc length:
$$\kappa = \left| \frac{d\psi}{ds} \right|$$
By the chain rule: $\frac{d\psi}{ds} = \frac{d\psi/dx}{ds/dx}$.
From calculus:
$$ds = \sqrt{dx^2 + dy^2} = \sqrt{1 + (y'(x))^2} dx \implies \frac{ds}{dx} = \sqrt{1 + (y'(x))^2}$$
Since heading tangent is $\psi(x) = \arctan(y'(x))$:
$$\frac{d\psi}{dx} = \frac{d}{dx}[\arctan(y')] = \frac{y''(x)}{1 + (y'(x))^2}$$
Dividing numerator by denominator yields the exact curvature formula used in our spline code:
$$\kappa(x) = \frac{\frac{y''(x)}{1 + (y'(x))^2}}{\sqrt{1 + (y'(x))^2}} = \frac{|y''(x)|}{(1 + (y'(x))^2)^{3/2}}$$

---

## 📺 Recommended Free Video Tutorials to Learn the Math
- **Kalman Filter from First Principles**: [Brian Douglas: State-Space & The Kalman Filter](https://www.youtube.com/watch?v=mwn8xhgNpFY)
- **Gaussian Conditioning & Bayes' Rule**: [MIT 6.041x: Jointly Normal Random Variables](https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/)
- **Differential Geometry & Curvature**: [Khan Academy: Curvature Formula Derivation](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/curvature)

---

## 🏎️ Why Tesla Does It This Way
- In Tesla AI Day 2021, Andrei Karpathy showed how the "Spatial RNN" produces clean vector splines for highway lane forks and intersections.
- The downstream neural planner needs analytical derivatives (tangent angles, curvature $\kappa$) to calculate lateral steering acceleration limits, which cannot be computed from noisy pixel blobs.

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
