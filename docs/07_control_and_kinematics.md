# Module 07: Closed-Loop Control & Kinematics (Sebastian Thrun & Duckietown)

> "A controller connects abstract coordinate trajectories to physical tires rolling on asphalt."

---

## 🟢 Tier 1: Intuition & Diagnostics (Andrew Ng Style)

### Why Open-Loop Control Always Crashes
Imagine aiming your car down a straight highway lane, setting the steering wheel perfectly straight, and closing your eyes.
Within 5 seconds, slight tire imbalances, wind gusts, or road crowns will nudge the car 10 centimeters off-center.
Because errors accumulate over time, an open-loop vehicle will veer off the road and crash.

**Closed-Loop Control**:
Every 10 to 20 milliseconds, the vehicle measures its actual position relative to the planned path and computes corrective steering:
- **Cross-Track Error ($e$)**: How many meters the car is to the left or right of the centerline.
- **Heading Error ($\theta_e$)**: The angular difference between the car's nose and the road's direction.

### The Two Titans of Robotics: Stanley vs Pure Pursuit
- **Stanley Controller** (Sebastian Thrun / Stanford DARPA Grand Challenge): Measures error at the **front axle**. It directly sums heading alignment error with an arctangent cross-track correction. Known for razor-sharp tracking on roads.
- **Pure Pursuit Controller** (MIT Duckietown / Carnegie Mellon): Measures error at the **rear axle**. It fits a smooth circular arc from the rear wheels to a "carrot" lookahead point $L_d$ ahead of the car. Known for extreme smoothness and simplicity.

```
       STANLEY CONTROLLER (Front Axle)               PURE PURSUIT CONTROLLER (Rear Axle)
      Path  ─────────────────────────               Path  ─────────────● Lookahead Carrot
                   ^ e (front error)                                  /
                   │                                                /  Circular
             [Front Axle]                                         /    Arc
                  │                                        [Front Wheels]
                  │ Car Body                                      │
                  │                                        [Rear Wheels]
             [Rear Axle]                                        (Fits arc to carrot)
```

### Andrew Ng Diagnostic Table: Steering & Control Failures

| Symptom | Root Cause | Diagnostic Test | Solution |
|---|---|---|---|
| High-speed fishtailing / wild side-to-side oscillation | Actuator latency $\tau$ introduces phase lag $\omega \tau$ | Test tracking at 20 km/h vs 80 km/h | Add Smith Predictor or reduce Stanley gain $k$ at high speeds |
| Car cuts the inside curb of tight corners | Pure pursuit lookahead distance $L_d$ set too large | Measure distance to path apex during turn | Dynamically scale lookahead: $L_d = \max(L_{\text{min}}, k_v \cdot v)$ |
| Car settles into permanent 10cm offset to one side | Road banking / cross-wind creates steady-state bias | Check if CTE error does not converge to zero | Add an Integral term ($K_i \int e \, dt$) to create PID-Stanley |
| Steering violently snaps left-and-right at near-zero speed | Division by zero: $\arctan\left(\frac{k e}{v}\right)$ as $v \to 0$ | Stop vehicle and inspect commanded steering angle | Add velocity softening parameter $\epsilon = 0.5\text{ m/s}$ in denominator |

---

## 🟡 Tier 2: Code From Scratch (Andrej Karpathy Style)

Let's build a Kinematic Bicycle Model and both controllers from scratch in raw NumPy.

```python
import numpy as np

class KinematicBicycleModel:
    """
    Standard Ackermann Steering Kinematic Bicycle Model.
    Wheelbase L: distance between front and rear axles.
    """
    def __init__(self, L: float = 2.8, dt: float = 0.05):
        self.L = L
        self.dt = dt
        self.x = 0.0
        self.y = 0.0
        self.psi = 0.0 # Heading angle in radians
        self.v = 0.0   # Forward speed in m/s

    def step(self, accel: float, delta: float):
        """Integrates motion using 4th-order Runge-Kutta or Euler integration."""
        # Clamp front wheel steering to physical rack limits (+/- 35 degrees)
        delta = np.clip(delta, -np.deg2rad(35), np.deg2rad(35))
        
        # State derivatives
        dx = self.v * np.cos(self.psi)
        dy = self.v * np.sin(self.psi)
        dpsi = (self.v / self.L) * np.tan(delta)
        dv = accel

        # Euler forward integration
        self.x += dx * self.dt
        self.y += dy * self.dt
        self.psi += dpsi * self.dt
        self.v += dv * self.dt

def stanley_control(car: KinematicBicycleModel, target_x: float, target_y: float, 
                    target_psi: float, k: float = 0.85, eps: float = 0.5) -> float:
    """
    Sebastian Thrun's Stanley Steering Controller.
    Computes steering angle delta for the front axle.
    """
    # 1. Front axle position
    fx = car.x + car.L * np.cos(car.psi)
    fy = car.y + car.L * np.sin(car.psi)

    # 2. Heading alignment error (normalized to [-pi, pi])
    heading_err = target_psi - car.psi
    heading_err = np.arctan2(np.sin(heading_err), np.cos(heading_err))

    # 3. Cross-track error vector from path to front axle
    dx = fx - target_x
    dy = fy - target_y
    # Signed cross-track error: positive if car is to the right of path
    cte = np.sin(target_psi) * dx - np.cos(target_psi) * dy

    # 4. Stanley control law with low-speed softening epsilon
    cte_term = np.arctan2(k * -cte, car.v + eps)
    cmd_delta = heading_err + cte_term
    return float(cmd_delta)
```

---

## 🔴 Tier 3: Mathematical Derivations & Proofs (Sebastian Thrun / Lyapunov)

### 1. Complete Lyapunov Direct Method Proof for Stanley Stability
- **Starting Point**: Let $e(t)$ be the signed cross-track error at the front axle. From the kinematic bicycle geometry, the error rate of change is:
  $$\dot{e}(t) = -v(t) \sin(\theta_e(t) - \delta(t))$$
- **Substituting the Stanley Law**:
  The Stanley steering command is $\delta = \theta_e + \arctan\left(\frac{k \cdot e}{v}\right)$, meaning:
  $$\theta_e - \delta = -\arctan\left(\frac{k \cdot e}{v}\right)$$
  Using the exact trigonometric identity $\sin(\arctan(u)) = \frac{u}{\sqrt{1 + u^2}}$:
  $$\sin\left(-\arctan\left(\frac{ke}{v}\right)\right) = -\frac{\frac{ke}{v}}{\sqrt{1 + \left(\frac{ke}{v}\right)^2}} = -\frac{k \cdot e}{\sqrt{v^2 + k^2 e^2}}$$
- **Error Dynamics**:
  $$\dot{e}(t) = -v \left(-\frac{k \cdot e}{\sqrt{v^2 + k^2 e^2}}\right) = -\frac{k \cdot v(t) \cdot e(t)}{\sqrt{v(t)^2 + k^2 e(t)^2}}$$
- **Lyapunov Stability Candidate**:
  Choose the continuously differentiable, positive-definite candidate function:
  $$V(e) = \frac{1}{2} e^2 > 0 \quad (\forall e \neq 0)$$
  Taking the time derivative:
  $$\dot{V}(e) = e \cdot \dot{e} = -\frac{k \cdot v(t) \cdot e^2}{\sqrt{v(t)^2 + k^2 e(t)^2}}$$
  For all $v(t) > 0$, $k > 0$, and $e \neq 0$:
  $$\dot{V}(e) < 0 \quad \text{strictly!}$$
  By **Lyapunov's Direct Method for Non-Autonomous Systems**, the origin $e = 0$ is **globally asymptotically stable**. Furthermore, for small $e \ll v/k$, $\dot{e} \approx -k \cdot e$, proving **exponential convergence** with time constant $\tau_{\text{conv}} = 1/k$!

---

## 🎓 Tier 4: Cutting-Edge Research & PhD Track

### 1. Model Predictive Control (MPC) vs. End-to-End Neural Control
- **The Limit of Stanley/Pure Pursuit**: Classical controllers assume linear tire adhesion. If a vehicle hydroplanes on ice or takes an evasive maneuver with lateral acceleration $a_{\text{lat}} > 0.6\text{g}$, the tires saturate (governed by Pacejka's Non-linear Magic Formula). Stanley will over-steer into a spin!
- **Nonlinear Model Predictive Control (NMPC)**: Solves an online quadratic program (QP) over a receding horizon of 2 seconds, explicitly penalizing tire slip angles and actuator slew rates.
- **Model Predictive Path Integral (MPPI)**: GPU-parallelized stochastic sampling (running 4,096 rollout simulations concurrently via PyTorch/CUDA) to achieve drift control and extreme obstacle avoidance without analytical gradients.

### 2. Open PhD Research Questions
- *How can neural network end-to-end driving policies guarantee stability under unmodeled actuator delays without falling victim to high-frequency oscillation?*
- *Can we synthesize neural Control Barrier Functions (CBFs) that formally prevent tire slip saturation during emergency obstacle evasion on wet pavement?*

---

## 🟣 Tier 5: Real-World Hardware & Practical Robotics

### Tuning Controllers on Low-Cost Hardware (Duckiebot / RC Car)
When deploying on an actual physical chassis:
1. **Actuator Slew Rate Limiting**: Electric servos cannot teleport from $-30^\circ$ to $+30^\circ$ instantly. Typical limit: $\dot{\delta}_{\text{max}} \approx 35^\circ/\text{s}$. Enforce this in software to avoid burning servo gears!
2. **Steering Trim Calibration**: Even with $\delta = 0$, physical wheel alignment is never perfect. Include a software calibration offset:
   $$\delta_{\text{physical}} = \delta_{\text{command}} + \delta_{\text{trim}}$$
3. **Deadband Compensation**: Cheap RC servos have $1\text{--}2^\circ$ of mechanical deadband around center. If commanded $|\delta| < 1.0^\circ$, output 0 to prevent motor jitter.
