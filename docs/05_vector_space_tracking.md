# Module 05: Vector Space Tracking, Kalman Filtering & Online HD Maps

> "Zero pre-recorded HD maps. The vehicle must construct its own HD vector map on the fly in real-time." — Elon Musk

---

## 🟢 Tier 1: Intuition & Diagnostics (Andrew Ng Style)

### Why Can't We Plan Directly from Raw Detections?
Imagine your neural network detects a lead vehicle at distance $X=25\text{m}$.
- On Frame 100: It sees the car at $25.2\text{m}$.
- On Frame 101: A truck momentarily blocks the view; detection disappears!
- On Frame 102: The car reappears at $24.8\text{m}$.

If your planner reacted to raw detections, the vehicle would slam on the brakes on Frame 100, accelerate aggressively on Frame 101, and panic on Frame 102.
Furthermore, **raw 2D/3D detections do not provide velocity!** A single static bounding box cannot tell you whether a car is parked or reversing at 50 km/h.

We need **Recursive State Estimation**:
1. Maintain persistent object tracks across temporary occlusions.
2. Estimate full kinematic state vectors: position $(x, y)$ **and** velocity $(v_x, v_y)$.
3. Assign incoming detections to existing tracks without causing **Identity Switches**.

```
    DETECTION FLICKER (RAW SENSORS)               KALMAN FILTER ESTIMATION (VECTOR SPACE)
    t=1: [Car Detected at 20m]                    t=1: State = 20.0m, v = 0.0 m/s
    t=2: [LOST - 0 Detections!]   ─────────────>  t=2: State = 20.6m, v = 14.1 m/s (Propagated!)
    t=3: [Car Detected at 21.2m]                  t=3: State = 21.2m, v = 14.2 m/s (Fused!)
    Result: Spastic phantom braking               Result: Butter-smooth cruise control
```

### Andrew Ng Diagnostic Table: Tracking & Association Failures

| Symptom | Root Cause | Diagnostic Test | Solution |
|---|---|---|---|
| Two cars cross at an intersection and swap track IDs | Naive Euclidean distance association | Measure distance between tracks during closest approach ($d < 1.0\text{m}$) | Use Mahalanobis distance gating ($d_M^2 \le 5.99$) |
| Covariance matrix $P$ blows up to infinity | Process noise $Q$ too high relative to measurement frequency | Inspect trace $\text{Tr}(P)$ over time | Re-tune process noise covariance $Q$; check sensor timestamp delta $\Delta t$ |
| Track lags behind the physical car during hard braking | Constant velocity model assumes zero acceleration | Calculate innovation residual $y = z - H\mu^-$ | Implement Extended Kalman Filter (EKF) with constant acceleration or turn-rate |
| Phantom tracks persist for 10 seconds after obstacle leaves | Death threshold (max missed frames) set too high | Check track lifetime when $z$ is absent | Delete unassigned tracks after $M_{\text{missed}} \ge 3$ consecutive frames |

---

## 🟡 Tier 2: Code From Scratch (Andrej Karpathy Style)

Let's build a self-contained 2D Kalman Filter and Hungarian Data Associator in raw NumPy.

```python
import numpy as np

class KalmanVehicleTrack:
    """
    2D Linear Kalman Filter for Vehicle State Tracking.
    State vector: x = [pos_x, pos_y, vel_x, vel_y]^T
    """
    def __init__(self, track_id: int, init_x: float, init_y: float):
        self.track_id = track_id
        # State vector
        self.x = np.array([init_x, init_y, 0.0, 0.0], dtype=np.float64)
        
        # Initial state covariance P (high uncertainty on velocities)
        self.P = np.diag([1.0, 1.0, 10.0, 10.0])
        
        # Measurement matrix H (we measure position x, y directly)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)
        
        # Sensor measurement noise covariance R
        self.R = np.diag([0.25, 0.25])
        
        # Process noise covariance Q
        self.Q = np.diag([0.05, 0.05, 0.5, 0.5])
        self.missed_frames = 0

    def predict(self, dt: float):
        """Kinematic state propagation: x_t = F * x_{t-1}"""
        F = np.array([
            [1.0, 0.0,  dt, 0.0],
            [0.0, 1.0, 0.0,  dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        self.missed_frames += 1

    def compute_mahalanobis(self, z: np.ndarray) -> float:
        """Computes statistical Mahalanobis distance squared to incoming detection z."""
        innovation = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        d_m_sq = float(innovation.T @ np.linalg.inv(S) @ innovation)
        return d_m_sq

    def update(self, z: np.ndarray):
        """Bayesian conditioning: fuses sensor measurement z into track belief."""
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S) # Kalman Gain
        
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        self.missed_frames = 0
```

---

## 🔴 Tier 3: Mathematical Derivations & Proofs (MITx Probability)

### 1. Bayesian Derivation of the Kalman Filter Update
- **Starting Point**: Let the prior state belief be $x \sim \mathcal{N}(\mu^-, P^-)$. The likelihood of sensor observation $z$ is $z \mid x \sim \mathcal{N}(Hx, R)$.
- **Proof via Product of Gaussians**:
  By Bayes' Rule:
  $$p(x \mid z) \propto p(z \mid x) \cdot p(x)$$
  $$\propto \exp\left( -\frac{1}{2} \left[ (z - Hx)^T R^{-1} (z - Hx) + (x - \mu^-)^T (P^-)^{-1} (x - \mu^-) \right] \right)$$
  Expanding the quadratic terms in $x$:
  $$J(x) = x^T (H^T R^{-1} H + (P^-)^{-1}) x - 2 x^T (H^T R^{-1} z + (P^-)^{-1} \mu^-) + \text{const}$$
  Completing the square, the posterior precision matrix is:
  $$P^{-1} = (P^-)^{-1} + H^T R^{-1} H$$
  Applying the **Woodbury Matrix Identity** to invert $P^{-1}$:
  $$P = (I - K H) P^-, \quad \text{where } K = P^- H^T (H P^- H^T + R)^{-1}$$
  The posterior mean is:
  $$\mu = \mu^- + K (z - H \mu^-)$$

### 2. Analytical Road Curvature $\kappa(x)$ Proof
- **Theorem**: For a cubic lane polynomial $y(x) = c_0 + c_1 x + c_2 x^2 + c_3 x^3$, road curvature is given by:
  $$\kappa(x) = \frac{|2 c_2 + 6 c_3 x|}{\left(1 + (c_1 + 2 c_2 x + 3 c_3 x^2)^2\right)^{3/2}}$$
- **Proof**:
  Curvature is defined as the angular rate of change with respect to arc length: $\kappa = |\frac{d\psi}{ds}|$.
  Since $\psi(x) = \arctan(y'(x))$, by the chain rule:
  $$\frac{d\psi}{dx} = \frac{y''(x)}{1 + (y'(x))^2}$$
  From the differential line element $ds = \sqrt{1 + (y'(x))^2} \, dx$:
  $$\kappa(x) = \frac{d\psi/dx}{ds/dx} = \frac{\frac{y''(x)}{1 + (y'(x))^2}}{\sqrt{1 + (y'(x))^2}} = \frac{|y''(x)|}{(1 + (y'(x))^2)^{3/2}}$$
  Substituting $y'(x) = c_1 + 2 c_2 x + 3 c_3 x^2$ and $y''(x) = 2 c_2 + 6 c_3 x$ completes the proof.

---

## 🎓 Tier 4: Cutting-Edge Research & PhD Track

### 1. Vectorized Online HD Map Construction: MapTR (ICCV 2023)
Traditional self-driving systems relied on pre-mapped centimeter-accurate HD Maps. When construction changes lane geometry, pre-recorded maps cause fatal collisions.
- **MapTR & MapTRv2 (ICCV 2023 / TPAMI 2024)**: Models lane dividers and road boundaries as **structured point sets** learned via hierarchical bipartite matching queries.
- Generates fully vectorized cubic lane splines in real time directly from surround video at 35 FPS, completely eliminating the need for pre-surveyed HD maps!

### 2. Graph Neural Network (GNN) Tracking
Classic Kalman tracking treats each object independently.
- In dense traffic, vehicles interact: when Lead Car A brakes, Following Car B must brake.
- Modern research uses **Spatio-Temporal Graph Neural Networks** where nodes represent tracked agents and edges represent interaction attention, predicting joint future covariances over time.

---

## 🟣 Tier 5: Real-World Hardware & Practical Robotics

### Multi-Sensor Latency Synchronization
In physical robotics (e.g. Comma 3X or Duckiebot):
- Camera frames arrive with $33\text{ ms}$ exposure latency.
- IMU gyro updates arrive at $200\text{ Hz}$ ($5\text{ ms}$ interval).
- Wheel speed encoders arrive asynchronously via CAN bus.

**The Solution**: Maintain a circular timestamp buffer. When a delayed camera measurement arrives with timestamp $t_{\text{cam}} = t_{\text{now}} - 45\text{ms}$:
1. Roll back the Kalman state to $t_{\text{cam}}$.
2. Execute the `update()` step with the new detection.
3. Fast-forward the state back to $t_{\text{now}}$ by re-applying the buffered IMU and wheel odometry updates!
