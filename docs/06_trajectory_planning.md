# Module 06: Quintic Lattice Trajectory Planning & Cost Maps

> "Perception is not the goal. Driving safely is. Planning is where the vehicle turns probabilistic beliefs into physical action." — OpenDriveLab (UniAD)

---

## 🟢 Tier 1: Intuition & Diagnostics (Andrew Ng Style)

### Path Planning vs. Trajectory Planning
Many beginners confuse **path planning** with **trajectory planning**:
- **A Path** is a geometric curve in space: $y = f(x)$. It tells the car *where* to go, but says nothing about *when* to be there.
- **A Trajectory** is a time-parameterized curve: $(x(t), y(t), v(t), a(t))$.

You **cannot** navigate dynamic traffic with paths alone! If you plan a static path across an intersection, but a truck is barreling through at 60 km/h, your safety depends entirely on **time**:
- Arrive at $t=3.0\text{s} \implies$ Fatal T-bone collision.
- Arrive at $t=4.5\text{s} \implies$ Safe, clean passage behind the truck.

### The Frenet Coordinate Frame: Longitudinal vs Lateral Decoupling
Planning directly in Cartesian coordinates $(X, Y)$ is painful when roads curve.
Sebastian Thrun and Moritz Werling introduced the **Frenet Coordinate Frame**:
- **$s(t)$ (Longitudinal)**: Distance traveled *along* the curved lane centerline.
- **$d(t)$ (Lateral)**: Perpendicular deviation *away from* the centerline (positive = left, negative = right).

This decouples 2D driving into two simple 1D problems:
1. $s(t)$: Speed regulation, following distance, and braking.
2. $d(t)$: Lane centering, nudging around obstacles, and lane changes.

```
                     CARTESIAN (X, Y) vs FRENET (s, d) COORDINATES
      Y ^                                               Centerline (d=0)
        │      Road Centerline                           ─────────────────
        │         .~~~~~~.            Frenet Transform      d > 0 (Left)
        │       .~        ~.        ─────────────────>   ───────────────
        │     .~    [Car]   ~.                             d < 0 (Right)
        └─────┼───────────────┼──> X                     ─────────────────
              0               s                         0 ─────────────> s (Distance)
```

### Andrew Ng Diagnostic Table: Trajectory Planning Failures

| Symptom | Root Cause | Diagnostic Test | Solution |
|---|---|---|---|
| Passenger feels severe nausea during lane change | Excessive lateral jerk $\dddot{d}(t) > 3.0\text{ m/s}^3$ | Plot analytical 3rd derivative of lateral position | Increase trajectory time horizon $T$ (e.g. from 2.0s to 3.5s) |
| Vehicle hesitates and freezes behind parked cars | Collision cost bubble set too conservative | Inspect candidate rejection log for all candidates | Implement exponential decay safety cost or nudge maneuver |
| Car oscillates laterally back and forth down straight road | Lane centering weight $w_{\text{lane}}$ fighting jerk weight | Compare $w_{\text{lane}} \cdot J_{\text{lane}}$ vs $w_{\text{jerk}} \cdot J_{\text{jerk}}$ | Increase jerk penalty $w_{\text{jerk}}$ and reduce lane stiffness |
| Planned path cuts sharp corners dangerously | Kinematic curvature limit $\kappa_{\text{max}} = \frac{\tan(\delta_{\text{max}})}{L}$ violated | Check max curvature of candidate paths | Prune candidates that exceed steering actuator limits before evaluation |

---

## 🟡 Tier 2: Code From Scratch (Andrej Karpathy Style)

Let's build a self-contained Quintic Boundary Value Solver and Lattice Candidate Evaluator in raw NumPy.

```python
import numpy as np

class QuinticPolynomial:
    """
    5th-order Polynomial: s(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    Uniquely minimizes the integral of squared jerk: min integral(jerk^2 dt).
    """
    def __init__(self, x0: float, v0: float, a0: float, 
                       xT: float, vT: float, aT: float, T: float):
        self.a0 = x0
        self.a1 = v0
        self.a2 = 0.5 * a0

        # Linear 3x3 system for [a3, a4, a5]
        # A * [a3, a4, a5]^T = B
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T

        A = np.array([
            [T3,      T4,      T5],
            [3 * T2,  4 * T3,  5 * T4],
            [6 * T,   12 * T2, 20 * T3]
        ], dtype=np.float64)

        b = np.array([
            xT - self.a0 - self.a1 * T - self.a2 * T2,
            vT - self.a1 - 2 * self.a2 * T,
            aT - 2 * self.a2
        ], dtype=np.float64)

        # Solve system
        a3, a4, a5 = np.linalg.solve(A, b)
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5

    def calc_pos(self, t: float) -> float:
        return self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4 + self.a5*t**5

    def calc_jerk(self, t: float) -> float:
        return 6.0 * self.a3 + 24.0 * self.a4 * t + 60.0 * self.a5 * t**2

def evaluate_lattice(candidates: list[QuinticPolynomial], obstacles: list[tuple[float, float]], 
                     T: float = 3.0, dt: float = 0.1) -> QuinticPolynomial:
    """Evaluates multi-objective cost over a candidate fanout and selects optimal path."""
    best_cost = float('inf')
    best_poly = candidates[0]

    for poly in candidates:
        total_cost = 0.0
        # Check collision, lane centering, and jerk along trajectory
        for t in np.arange(0.0, T, dt):
            pos_lat = poly.calc_pos(t)
            jerk = poly.calc_jerk(t)
            
            # Jerk cost (passenger comfort)
            total_cost += 0.5 * (jerk / 5.0) ** 2
            # Lane centering cost
            total_cost += 2.0 * (pos_lat / 1.5) ** 2
            
            # Collision cost against obstacles
            for ox, oy in obstacles:
                dist = np.hypot(t * 15.0 - ox, pos_lat - oy) # approximate forward progress
                if dist < 2.5: # 2.5 meter safety bubble
                    total_cost += np.exp((2.5 - dist) * 2.0) * 1000.0

        if total_cost < best_cost:
            best_cost = total_cost
            best_poly = poly

    return best_poly
```

---

## 🔴 Tier 3: Mathematical Derivations & Proofs (Sebastian Thrun / Optimal Control)

### 1. Proof that the Quintic Polynomial Minimizes Squared Jerk
- **Problem Formulation**: We seek a trajectory $s(t)$ connecting initial state $(s_0, v_0, a_0)$ at $t=0$ to final state $(s_T, v_T, a_T)$ at $t=T$ that minimizes total lateral jerk:
  $$J = \int_0^T (\dddot{s}(t))^2 \, dt$$
- **Calculus of Variations (Euler-Lagrange Equation)**:
  Let the Lagrangian be $L(t, s, \dot{s}, \ddot{s}, \dddot{s}) = (\dddot{s})^2$.
  The generalized Euler-Lagrange equation for higher-order derivatives is:
  $$\frac{\partial L}{\partial s} - \frac{d}{dt}\left[\frac{\partial L}{\partial \dot{s}}\right] + \frac{d^2}{dt^2}\left[\frac{\partial L}{\partial \ddot{s}}\right] - \frac{d^3}{dt^3}\left[\frac{\partial L}{\partial \dddot{s}}\right] = 0$$
  Since $L$ depends only on $\dddot{s}$:
  $$\frac{\partial L}{\partial s} = 0, \quad \frac{\partial L}{\partial \dot{s}} = 0, \quad \frac{\partial L}{\partial \ddot{s}} = 0$$
  $$\frac{\partial L}{\partial \dddot{s}} = 2 \dddot{s}(t)$$
  Substituting into Euler-Lagrange:
  $$-\frac{d^3}{dt^3} [2 \dddot{s}(t)] = 0 \implies \frac{d^6 s(t)}{dt^6} = 0$$
- **Integration**:
  Integrating the 6th derivative $\frac{d^6 s}{dt^6} = 0$ six times consecutively with respect to $t$:
  $$s(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$$
  This proves analytically that **a 5th-order polynomial is the exact unique mathematical minimum-jerk trajectory!**

### 2. Proof of Boundary Matrix Invertibility
- **Theorem**: The $3 \times 3$ linear system for coefficients $[a_3, a_4, a_5]$ has non-zero determinant $\det(A) = 2 T^9 \neq 0$ for all $T > 0$.
- **Proof**:
  $$A = \begin{bmatrix} T^3 & T^4 & T^5 \\ 3 T^2 & 4 T^3 & 5 T^4 \\ 6 T & 12 T^2 & 20 T^3 \end{bmatrix}$$
  Factoring $T^3$ from row 1, $T^2$ from row 2, and $T$ from row 3:
  $$\det(A) = T^3 \cdot T^2 \cdot T \cdot \det \begin{bmatrix} 1 & T & T^2 \\ 3 & 4 T & 5 T^2 \\ 6 & 12 T & 20 T^2 \end{bmatrix} = T^6 \cdot T^3 \cdot \det \begin{bmatrix} 1 & 1 & 1 \\ 3 & 4 & 5 \\ 6 & 12 & 20 \end{bmatrix}$$
  Evaluating the numerical $3 \times 3$ determinant:
  $$\det \begin{bmatrix} 1 & 1 & 1 \\ 3 & 4 & 5 \\ 6 & 12 & 20 \end{bmatrix} = 1(80 - 60) - 1(60 - 30) + 1(36 - 24) = 20 - 30 + 12 = 2$$
  Thus:
  $$\det(A) = 2 T^9$$
  Because $T > 0$ strictly for any forward trajectory duration, $\det(A) > 0$ always. The system is unconditionally non-singular and never encounters division by zero!

---

## 🎓 Tier 4: Cutting-Edge Research & PhD Track

### 1. From Heuristic Lattice Planners to End-to-End Neural Planners
- **The Limitation of Lattice Planners**: Handcrafted cost weights ($w_{\text{coll}}, w_{\text{lane}}, w_{\text{jerk}}$) work well on structured highways, but fail in chaotic urban environments (e.g. negotiating an aggressive unprotected left turn across two oncoming lanes).
- **UniAD (CVPR 2023 Best Paper)** & **VAD (ICCV 2023)**: Replaces lattice sampling with **Planning Queries**. A transformer cross-attention decoder queries the unified perception space directly to predict future waypoints:
  $$\tau = \text{MLP}(\text{CrossAttention}(Q_{\text{plan}}, K_{\text{scene}}, V_{\text{scene}}))$$
- **Diffusion Planners**: Diffusion models (e.g. Diffusion-Policy, NoMaD) generate multi-modal trajectories, avoiding the common mode-collapse failure mode where neural planners average left and right paths and crash directly into the center obstacle.

### 2. Open PhD Research Questions
- *How can we integrate formal Control Barrier Functions (CBFs) directly into the loss function of a neural trajectory planner to provide certified zero-collision guarantees?*
- *Can we formulate a game-theoretic planner where the ego-vehicle reasons about how other human drivers will react to its own planned nudges?*

---

## 🟣 Tier 5: Real-World Hardware & Practical Robotics

### Real-Time Performance & Human G-Force Limits
On automotive hardware, trajectory generation must complete within a strict **$20\text{ ms}$ budget** (50 Hz cycle).

**ISO 2631 Passenger Comfort Limits**:
- Lateral Acceleration: $|a_{\text{lat}}| \le 2.0\text{ m/s}^2$ ($0.2\text{ g}$).
- Longitudinal Deceleration: $|a_{\text{long}}| \le 3.5\text{ m/s}^2$ (normal braking); $> 6.0\text{ m/s}^2$ triggers seatbelt pre-tensioners.
- Maximum Jerk: $|j| \le 2.5\text{ m/s}^3$.

In your robotics code, evaluate candidate paths against these thresholds and immediately discard any trajectory that violates them before checking obstacle collisions!
