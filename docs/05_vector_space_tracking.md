# Deep Dive: Vector Space Tracking, Kalman Filtering & Online HD Maps

> "Zero pre-recorded HD maps. The vehicle must construct its own HD vector map on the fly in real-time." — Elon Musk

---

## 1. The Kalman Filter as Recursive Bayesian Estimation (MITx Probability)
Let the state vector of a surrounding vehicle be:
$$\mathbf{x} = [x, y, v_x, v_y]^T \in \mathbb{R}^4$$

### The Continuous Probability Flow
In MITx Probability, Bayes' rule states:
$$p(\mathbf{x}_t \mid \mathbf{z}_{1:t}) \propto p(\mathbf{z}_t \mid \mathbf{x}_t) \cdot p(\mathbf{x}_t \mid \mathbf{z}_{1:t-1})$$

When the prior and likelihood are multivariate Gaussians:
1. **Prior Belief at step $t-1$**:
   $$\mathbf{x}_{t-1} \sim \mathcal{N}(\mu_{t-1}, P_{t-1})$$

2. **Kinematic Motion Prediction (Total Probability / Chapman-Kolmogorov)**:
   $$\mathbf{x}_t = F \mathbf{x}_{t-1} + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(0, Q)$$
   $$\mu_t^- = F \mu_{t-1}$$
   $$P_t^- = F P_{t-1} F^T + Q$$

3. **Sensor Measurement Likelihood**:
   $$\mathbf{z}_t = H \mathbf{x}_t + \mathbf{v}_t, \quad \mathbf{v}_t \sim \mathcal{N}(0, R)$$
   Innovation residual: $\mathbf{y}_t = \mathbf{z}_t - H \mu_t^-$
   Innovation covariance: $S_t = H P_t^- H^T + R$

4. **Bayes Conditioning (Posterior Update)**:
   Kalman Gain: $K_t = P_t^- H^T S_t^{-1}$
   Posterior Mean: $\mu_t = \mu_t^- + K_t \mathbf{y}_t$
   Posterior Covariance: $P_t = (I - K_t H) P_t^-$

Notice that $(I - K_t H)$ is positive semi-definite; therefore:
$$\text{Tr}(P_t) < \text{Tr}(P_t^-)$$
Observing photons strictly decreases state uncertainty!

---

## 2. Data Association: Mahalanobis Distance vs. Euclidean Distance
When assigning incoming sensor detections to existing tracks, Euclidean distance $\|\mathbf{z} - H\mu\|^2$ fails when tracks have directional velocities (e.g. at highway merges or intersections).

The **Mahalanobis Distance** measures distance in units of standard deviations:
$$d_M^2(\mathbf{z}, \mu_t^-) = (\mathbf{z} - H\mu_t^-)^T S_t^{-1} (\mathbf{z} - H\mu_t^-)$$
Under Gaussian assumptions, $d_M^2$ follows a **Chi-Squared Distribution** with $k=2$ degrees of freedom:
$$d_M^2 \sim \chi^2_2$$
A gating threshold of $d_M^2 \le 5.99$ accepts 95% of true associations while mathematically rejecting 95% of false identity cross-swaps!

---

## 3. Parametric Vector Lane Splines
Raster lane masks (pixel grids) cannot be differentiated analytically. We fit cubic polynomials:
$$y(x) = c_0 + c_1 x + c_2 x^2 + c_3 x^3$$
Where $x$ is forward longitudinal distance and $y$ is lateral offset.

### Analytical Road Geometry
- **Lateral Offset at distance $x$**: $y(x)$
- **Tangent Heading Angle**: $\psi(x) = \arctan(y'(x)) = \arctan(c_1 + 2 c_2 x + 3 c_3 x^2)$
- **Road Curvature**:
  $$\kappa(x) = \frac{|y''(x)|}{(1 + (y'(x))^2)^{3/2}} = \frac{|2 c_2 + 6 c_3 x|}{(1 + (c_1 + 2 c_2 x + 3 c_3 x^2)^2)^{3/2}}$$
Curvature $\kappa(x)$ dictates maximum safe cornering speed: $v_{\text{max}} = \sqrt{\frac{a_{\text{lat, max}}}{\kappa}}$.
