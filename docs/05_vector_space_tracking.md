# Chapter 06: Vector Space Tracking & Multi-Object Association

> **The Big Question**: *In video frame 101, your camera detects a white sedan at coordinates $(15.2, 2.1)$. In frame 102, camera noise places a detection at $(15.9, 2.3)$. How does the vehicle mathematically prove that this is the exact same sedan continuing along its path rather than a brand-new obstacle—and how does a Kalman filter filter out sensor jitter to calculate exact physical velocity?*

---

## 1. 🚨 The Real-World Dilemma: The Identity Swap & Velocity Jitter Trap

Every sensor in the real world is noisy:
- Camera detection coordinates jitter by $\pm 0.3\text{ m}$ from frame to frame due to rolling shutter and suspension vibration.
- If you compute velocity using simple naive finite differences:
  $$v_x = \frac{x_t - x_{t-1}}{\Delta t}$$
  At $\Delta t = 0.033\text{ s}$ (30 FPS), a tiny measurement noise of $\Delta x = 0.4\text{ m}$ produces a calculated velocity spike of:
  $$v = \frac{0.4}{0.033} = \mathbf{12.1\text{ m/s}} \approx \mathbf{27\text{ mph!}}$$
- A stationary car waiting at a red light will appear to violently vibrate forward and backward at 27 mph, causing your autonomous planner to slam on the brakes!

### The Multi-Object Association Dilemma
Now imagine highway traffic with 6 identical silver sedans driving closely together:
- When two cars overtake and cross paths, how does the vehicle ensure it doesn't swap their identities (Track ID Switch)?
- If Track ID 14 (a car turning away safely) swaps with Track ID 15 (a car cutting directly into your lane), the vehicle will fail to brake in time.

```
Frame t:                             Frame t+1:
  Track 1 (Fast) ──►                   Track 1 ──┐
                     CROSSING                    ├──► ??? Identity Swap!
  Track 2 (Slow) ──►                   Track 2 ──┘
```

---

## 2. 💡 The Mental Model: The Kalman Filter & Mahalanobis Distance

### The Kalman Filter: Physics Prediction + Sensor Correction
Think of how you track a baseball in flight:
1. **Physics Prediction (Internal Model)**: When you blink your eyes, you know the ball doesn't teleport. You use high school kinematics ($\mathbf{x} = \mathbf{x}_0 + \mathbf{v} t$) to predict where the ball *should* be.
2. **Sensor Measurement (Observation)**: You open your eyes and see a blurry shape near your predicted point.
3. **Optimal Fusion (Kalman Gain)**: You weigh your physics prediction against your visual measurement based on **which one has less uncertainty**.

```
State Uncertainty (Covariance P):
      Prediction Step (Uncertainty Grows): 
             [─────── Uncertainty Blob ───────]
      Measurement Fusion (Uncertainty Contracts):
                     [── Tight Belief ──]
```

### Why Euclidean Distance Fails for Tracking
If a detected vehicle was traveling at 65 mph in the center lane:
- In the next frame, finding the detection $1.5\text{ m}$ further along the lane is **completely expected** (consistent with high velocity).
- But finding the detection $1.5\text{ m}$ *perpendicular* across the lane divider means the car just made an aggressive emergency swerve!

Naive Euclidean distance treats both errors identically ($1.5\text{ m}$).  
**Mahalanobis Distance** warps space using the covariance matrix $S$:

$$d_M = \sqrt{(\mathbf{z} - \hat{\mathbf{z}})^T S^{-1} (\mathbf{z} - \hat{\mathbf{z}})}$$

It measures distance in units of **standard deviations along the ellipse of motion**, making association robust to velocity and heading!

---

## 3. 🧪 Lab Mission: Hands-On Simulator Experiments

Scroll to the **Interactive Vector Space Tracker Studio** at the top of this chapter:

1. **Experiment 1 (Sensor Noise vs Filtered State)**:
   - Turn **Measurement Noise ($\sigma_{\text{meas}}$)** up to `1.5m`.
   - Watch the raw red detection dots bounce erratically across the lane.
   - *Observation*: Look at the solid green Kalman filter trajectory. Despite the violent measurement jitter, the filtered velocity vector remains smooth and steady!
2. **Experiment 2 (Tracking Through Occlusion)**:
   - Click the **Occlude Target (3s)** button. The red sensor detections stop completely.
   - *Observation*: The green tracker continues to advance forward monotonically using pure physical kinematics ($\mathbf{x} = F \mathbf{x}$). Notice how the blue covariance ellipse steadily expands, reflecting growing uncertainty until the sensor re-acquires the car!

---

## 4. 🛠️ The Karpathy Build: 2D Kalman Filter from Scratch

Here is the exact linear-quadratic estimation engine implemented in pure Python with zero black box libraries:

```python
import numpy as np

class KalmanFilter2D:
    """
    Constant-Velocity 2D Kalman Filter tracking position [x, y] and velocity [vx, vy].
    State vector: x = [px, py, vx, vy]^T
    """
    def __init__(self, dt: float = 0.05, std_pos: float = 0.5, std_vel: float = 1.0):
        self.dt = dt
        
        # State transition matrix F: px' = px + dt * vx
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # Measurement matrix H: we only observe position [px, py]
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        
        # State estimation covariance matrix P
        self.P = np.eye(4) * 10.0
        
        # Process noise covariance Q (kinematic acceleration uncertainty)
        self.Q = np.eye(4) * 0.1
        
        # Measurement noise covariance R (sensor resolution uncertainty)
        self.R = np.eye(2) * (std_pos ** 2)
        
        # Estimated state
        self.x = np.zeros((4, 1))

    def predict(self):
        """Propagates state and covariance forward using physics model."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z: np.ndarray):
        """
        Fuses noisy measurement z = [px_meas, py_meas]^T into belief state.
        """
        # Innovation (measurement residual)
        y = z.reshape(2, 1) - self.H @ self.x
        
        # Innovation covariance S
        S = self.H @ self.P @ self.H.T + self.R
        
        # Optimal Kalman Gain K
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Updated state estimate
        self.x = self.x + K @ y
        
        # Updated covariance estimate (Joseph form for stability)
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        return self.x
```

---

## 5. 📐 Mathematical Rigor: The Kalman Filter Equations

### Phase 1: Prediction (Time Update)
$$\mathbf{x}_{k \mid k-1} = F_k \mathbf{x}_{k-1 \mid k-1}$$
$$P_{k \mid k-1} = F_k P_{k-1 \mid k-1} F_k^T + Q_k$$

### Phase 2: Correction (Measurement Update)
$$\mathbf{y}_k = \mathbf{z}_k - H_k \mathbf{x}_{k \mid k-1} \quad \text{(Innovation)}$$
$$S_k = H_k P_{k \mid k-1} H_k^T + R_k \quad \text{(Innovation Covariance)}$$
$$K_k = P_{k \mid k-1} H_k^T S_k^{-1} \quad \text{(Optimal Kalman Gain)}$$
$$\mathbf{x}_{k \mid k} = \mathbf{x}_{k \mid k-1} + K_k \mathbf{y}_k$$
$$P_{k \mid k} = (I - K_k H_k) P_{k \mid k-1}$$

> [!NOTE]
> **Bayesian Contraction Theorem**  
> Notice that the updated covariance $P_{k|k} = (I - K H) P_{k|k-1}$ is strictly smaller than the prior covariance $P_{k|k-1}$. In probability theory, observing independent information **always contracts entropy and reduces uncertainty**!

---

## 6. 🩺 Andrew Ng's Diagnostic Field Guide

| Observed Symptom | Underlying Mathematical Mechanism | Verification Test | Production Fix |
| :--- | :--- | :--- | :--- |
| **Track ID switches when two vehicles pass each other** | **Euclidean Gating Overlap**: Naive association gates overlap during close proximity. | Compute cross-track Mahalanobis distance between tracks $i$ and $j$. | Use Hungarian Algorithm (Munkres) with joint cost incorporating appearance embeddings (Re-ID). |
| **Filter diverges or covariance $P$ becomes non-positive definite** | **Numerical Roundoff in Covariance**: The update $(I - KH)P$ loses symmetry due to floating point error. | Check eigenvalues of $P$; if any $\lambda_i \le 0$, filter has collapsed. | Use Joseph Form: $P = (I - KH)P(I - KH)^T + KRK^T$, or force symmetry: $P = \frac{1}{2}(P + P^T)$. |
| **Filtered position lags several meters behind turning cars** | **Process Noise $Q$ Underestimated**: The constant-velocity model refuses to believe the car is accelerating or turning. | Measure innovation residual magnitude $\|\mathbf{y}\|$ during maneuvers. | Increase acceleration variance in $Q$ or upgrade to an Unscented Kalman Filter (UKF) with a bicycle motion model. |

---

## 7. 🎯 Self-Check: Test Your Mental Model

<details>
<summary><b>Q1: What happens to the Kalman Gain $K$ if sensor measurement noise becomes infinitely large ($R \to \infty$)?</b></summary>

<br>

**Answer**: 
From the Kalman Gain formula:
$$K = P H^T (H P H^T + R)^{-1}$$
As $R \to \infty$, the term $(H P H^T + R)^{-1} \to 0$, causing $K \to 0$.  
The updated state becomes $\mathbf{x} = \mathbf{x}_{\text{pred}} + 0 \cdot \mathbf{y} = \mathbf{x}_{\text{pred}}$.  
The filter completely ignores the noisy sensor and relies $100\%$ on its physical kinematic prediction!
</details>

<details>
<summary><b>Q2: Why is the state vector $\mathbf{x}$ tracked in the global metric world frame rather than the vehicle's ego-camera frame?</b></summary>

<br>

**Answer**: If tracks are stored in the ego-camera frame, whenever your own car accelerates, brakes, or turns its steering wheel, every other object in the world will appear to accelerate or rotate in the opposite direction! Tracking in the stationary metric world frame decouples the motion of external vehicles from the ego-car's own driving maneuvers.
</details>
