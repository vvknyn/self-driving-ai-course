# Chapter 07: Lattice Trajectory Planning & Quintic Splines

> **The Big Question**: *A path planner can easily find a path around an obstacle using standard video game algorithms like A\* or Dijkstra. But if an autonomous vehicle follows piecewise line segments, its steering wheel must teleport instantaneously, demanding infinite tire friction and violently snapping passengers' necks! How do we mathematically formulate trajectories with zero jerk that guarantee passenger comfort while dodging dynamic highway traffic at 70 mph?*

---

## 1. 🚨 The Real-World Dilemma: The Instantaneous Acceleration Snap

In classical robotics (like a Roomba vacuum cleaner or warehouse robot), moving around an obstacle is simple:
1. Drive straight until you reach the obstacle.
2. Stop, rotate $45^\circ$, and drive diagonally.
3. Rotate $-45^\circ$ and drive straight again.

### Why Piecewise Paths Kill at Highway Speeds
If an autonomous car tries to make an instantaneous heading change at 65 mph:
- Lateral acceleration is $a_{\text{lat}} = v^2 \kappa$, where $\kappa$ is road curvature.
- Changing curvature instantly ($\Delta \kappa > 0$ at $\Delta t = 0$) demands **infinite rate of change of acceleration**:

$$\text{Jerk} = j(t) = \frac{da}{dt} = \frac{d^3 x}{dt^3} \to \infty$$

- The car's physical steering rack cannot move infinitely fast.
- The tires break traction with the asphalt, sending the car into an unrecoverable spin!
- Human passengers feel violent nausea when lateral jerk exceeds **$2.0\text{ m/s}^3$**.

```
Naive Piecewise Path:                   Quintic Spline Smooth Trajectory:
┌───────────────────────────┐           ┌───────────────────────────┐
│              /\           │           │              ╭──╮         │
│             /  \          │           │             ╭╯  ╰╮        │
│   ─────────┘    └──────── │           │   ─────────╯      ╰────── │
└───────────────────────────┘           └───────────────────────────┘
Infinite Jerk! Steering snaps           Continuous Jerk! Butter-smooth
and tires break traction.               and physically achievable.
```

---

## 2. 💡 The Mental Model: The Frenet Frame & Quintic Boundary Matching

### The Frenet Coordinate System: Uncurling the Highway
Trying to plan a highway lane change in global $(X, Y)$ Cartesian space is a mathematical nightmare because the highway itself curves through mountains.

Instead, we project the world onto the **Frenet Frame $(s, d)$**:
- $s(t)$: Longitudinal distance along the curved road centerline (station).
- $d(t)$: Lateral offset perpendicular to the road centerline ($d = 0$ is dead center, $d = +3.5\text{ m}$ is the left lane).

```
Curved Highway:                        Frenet Frame (Flat & Decoupled):
        (d = +3.5m) Left Lane                   d (Lateral Offset)
       ╭───────────────────╮                     ▲
      ╭╯ (d = 0m) Center   ╰╮                    │   [Left Lane: d = +3.5m]
     ╭╯                   ╰╮                     │   [Centerline: d = 0m]
    ╭╯                     ╰╮                    │   [Right Lane: d = -3.5m]
                                                 └─────────────────────────► s (Longitudinal)
```

In the Frenet frame, changing lanes is as simple as moving from $d(0) = 0$ to $d(T) = 3.5\text{ m}$!

### The 6 Boundary Conditions of a Smooth Maneuver
When you begin a lane change at time $t = 0$ and complete it at time $t = T$:
1. Initial lateral position: $d(0) = d_0$
2. Initial lateral velocity: $\dot{d}(0) = \dot{d}_0$
3. Initial lateral acceleration: $\ddot{d}(0) = \ddot{d}_0$
4. Target lateral position: $d(T) = d_1$
5. Target lateral velocity: $\dot{d}(T) = \dot{d}_1 = 0$ (parallel to new lane)
6. Target lateral acceleration: $\ddot{d}(T) = \ddot{d}_1 = 0$ (no residual lateral roll)

To satisfy **6 independent constraints**, our polynomial must have **6 degrees of freedom** ($a_0$ through $a_5$). This is a **Quintic ($5^{\text{th}}\text{-order}$) Polynomial**:

$$d(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$$

---

## 3. 🧪 Lab Mission: Hands-On Simulator Experiments

Scroll to the **Interactive Trajectory Planner Studio** at the top of this chapter:

1. **Experiment 1 (The Obstacle Fan Sampling)**:
   - Place a stationary obstacle in the ego vehicle's lane at distance $s = 40\text{ m}$.
   - Observe the candidate trajectory lattice fan out across multiple target lane offsets ($d \in [-3.5, 0, 3.5]$).
   - *Observation*: Red candidate splines that clip the obstacle bounding box are instantly discarded by the collision evaluator.
2. **Experiment 2 (Jerk Cost vs Passenger Comfort)**:
   - Set the **Jerk Penalty ($w_{\text{jerk}}$)** slider to `0.0`.
   - Watch the selected green trajectory aggressively swerve at the last possible moment.
   - Now increase $w_{\text{jerk}}$ to `5.0`.
   - *Observation*: The planner initiates the lane change 20 meters earlier, creating a long, gradual, elegant curve with low peak jerk.

---

## 4. 🛠️ The Karpathy Build: Quintic Boundary Value Solver from Scratch

Here is the exact linear algebra solver implemented in pure Python:

```python
import numpy as np

class QuinticPolynomial:
    """
    1D Quintic Polynomial trajectory satisfying 6 boundary conditions:
    d(0), d'(0), d''(0) and d(T), d'(T), d''(T).
    """
    def __init__(self, x0: float, v0: float, a0: float, 
                       x1: float, v1: float, a1: float, T: float):
        self.T = T
        # 1. The first 3 coefficients are given directly by initial state
        self.a0 = x0
        self.a1 = v0
        self.a2 = 0.5 * a0

        # 2. Solve 3x3 linear system for [a3, a4, a5]^T
        # Matrix evaluated at terminal time T
        M = np.array([
            [   T**3,       T**4,        T**5],
            [ 3*T**2,     4*T**3,      5*T**4],
            [    6*T,    12*T**2,     20*T**3]
        ])
        
        # Target residuals after subtracting known lower-order terms
        b = np.array([
            x1 - (self.a0 + self.a1 * T + self.a2 * T**2),
            v1 - (self.a1 + 2 * self.a2 * T),
            a1 - (2 * self.a2)
        ])
        
        # Solve M * a = b
        self.a3, self.a4, self.a5 = np.linalg.solve(M, b)

    def calc_pos(self, t: float) -> float:
        return self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4 + self.a5*t**5

    def calc_vel(self, t: float) -> float:
        return self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3 + 5*self.a5*t**4

    def calc_acc(self, t: float) -> float:
        return 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2 + 20*self.a5*t**3

    def calc_jerk(self, t: float) -> float:
        return 6*self.a3 + 24*self.a4*t + 60*self.a5*t**2
```

---

## 5. 📐 Mathematical Rigor: The Trajectory Cost Functional

In modern end-to-end planners (such as **UniAD**, CVPR 2023 Best Paper), we sample $K$ candidate trajectories and evaluate each against a composite cost functional:

$$\mathcal{J}(\tau) = w_{\text{coll}} \mathcal{J}_{\text{collision}} + w_{\text{jerk}} \int_0^T \left( \dddot{d}(t) \right)^2 dt + w_{\text{lane}} (d(T) - d_{\text{center}})^2 + w_{\text{speed}} (v(T) - v_{\text{target}})^2$$

### Collision Distance Potential Field
For any candidate waypoint $\mathbf{p}(t)$ and obstacle centroid $\mathbf{o}_i$ with bounding radius $r_{\text{safe}}$:

$$\mathcal{J}_{\text{collision}}(\tau) = \sum_{t=0}^T \sum_{i \in \text{obstacles}} \exp\left( -\frac{\|\mathbf{p}(t) - \mathbf{o}_i(t)\|^2}{2 \sigma_{\text{safe}}^2} \right)$$

If any point on the trajectory penetrates the safety envelope ($\|\mathbf{p} - \mathbf{o}\| < r_{\text{safe}}$), the exponential cost surges toward infinity, strictly eliminating that trajectory from selection.

---

## 6. 🩺 Andrew Ng's Diagnostic Field Guide

| Observed Symptom | Underlying Mathematical Mechanism | Verification Test | Production Fix |
| :--- | :--- | :--- | :--- |
| **Vehicle swerves aggressively then snaps back** | **Horizon Time $T$ Too Short**: Forcing a 3.5m lane change in $T = 1.0\text{ s}$ requires peak lateral acceleration $> 8\text{ m/s}^2$ exceeding tire friction. | Check peak lateral acceleration $\max |a_{\text{lat}}(t)| > \mu g$. | Enforce dynamic horizon scaling: $T_{\min} \ge \sqrt{\frac{2 \Delta d}{a_{\text{comfort}}}}$. |
| **Planner freezes / output trajectory oscillates between lanes** | **Symmetric Cost Well**: Obstacle directly in path creates identical costs for left swerve and right swerve. | Check if cost difference $|J_{\text{left}} - J_{\text{right}}| < 10^{-3}$. | Add lane bias hysteresis: favor the current lane or following the rules of the road (e.g. pass on left). |
| **Vehicle fails to brake for a decelerating lead car** | **Decoupled Longitudinal Planning**: Planning $s(t)$ and $d(t)$ independently without checking temporal collision intersections. | Plot space-time diagram $s(t)$ vs obstacle trajectory $s_{\text{obs}}(t)$. | Formulate joint spatiotemporal lattice or use Model Predictive Path Integral (MPPI) control. |

---

## 7. 🎯 Self-Check: Test Your Mental Model

<details>
<summary><b>Q1: Why is a cubic ($3^{\text{rd}}\text{-order}$) polynomial insufficient for passenger-comfortable autonomous trajectory planning?</b></summary>

<br>

**Answer**: A cubic polynomial $p(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3$ only has 4 degrees of freedom. It can satisfy initial position $x_0$, initial velocity $v_0$, terminal position $x_1$, and terminal velocity $v_1$, but **cannot control acceleration**. As a result, initial acceleration $\ddot{p}(0)$ will almost never match the vehicle's current physical acceleration, causing an instantaneous jump in steering angle and infinite jerk at $t = 0$.
</details>

<details>
<summary><b>Q2: What is the physical meaning of minimizing $\int_0^T (\dddot{x}(t))^2 dt$ in the planning cost function?</b></summary>

<br>

**Answer**: Minimizing integrated squared jerk minimizes the rate of change of lateral force transferred through the suspension springs and tires. In human biomechanics, inner-ear vestibular balance organs and neck muscles react painfully to high jerk. Minimizing squared jerk produces smooth, human-like, flowing trajectories that keep tire contact patches firmly within their linear friction limits.
</details>
