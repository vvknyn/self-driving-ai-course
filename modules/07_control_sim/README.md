# Module 07: Closed-Loop Control & Kinematic Simulation (Stanley, Pure Pursuit, MPC)

> "A great trajectory plan is useless if your car oscillates like a drunk driver or cuts corners into oncoming traffic. Control is where math meets the road." — Sebastian Thrun

---

## 🎯 Purpose & Learning Goals
In robotics and self-driving cars, plans exist in the abstract realm of coordinates. But cars are heavy physical machines governed by Newton's laws and tire friction.

In this module, you will:
1. Build a **Kinematic Bicycle Vehicle Model**: Learn how front wheel steering angle $\delta$ and acceleration $a$ govern vehicle position $(x, y)$ and heading $\psi$.
2. Implement **Sebastian Thrun's Stanley Controller** (which won the DARPA Grand Challenge with Stanford's *Stanley* autonomous car).
3. Implement the **Pure Pursuit Controller** (the battle-tested standard in MIT Duckietown and robotics).
4. Build a **Closed-Loop Simulator** that benchmarks controllers on tight S-curves and highway lane merges.
5. FastAI "Break It & Fix It": Discover **Control Latency Fishtailing**—when 100ms of compute delay turns a stable car into a violent oscillatory wreck—and fix it with predictive lookahead compensation.

---

## 📐 How the Formulas Are Obtained

### 1. Step-by-Step Derivation & Lyapunov Stability of Stanley Controller
Let:
- $e(t)$ be the cross-track error measured from the front axle to the nearest path point.
- $\theta_e(t) = \psi_{\text{path}} - \psi_{\text{car}}$ be heading error.
- $\delta(t)$ be the front wheel steering angle.

The differential kinematic equation for cross-track error rate of change is:
$$\dot{e}(t) = -v(t) \sin(\theta_e(t) - \delta(t))$$

Sebastian Thrun designed the Stanley steering law to cancel out heading error while adding a non-linear arctangent correction for lateral offset:
$$\delta(t) = \theta_e(t) + \arctan\left(\frac{k \cdot e(t)}{v(t) + \epsilon}\right)$$

Substituting $\delta(t)$ back into $\dot{e}(t)$:
$$\theta_e(t) - \delta(t) = -\arctan\left(\frac{k \cdot e(t)}{v(t)}\right)$$
$$\dot{e}(t) = -v(t) \sin\left(-\arctan\left(\frac{k \cdot e(t)}{v(t)}\right)\right)$$

Using the exact trigonometric identity $\sin(\arctan(u)) = \frac{u}{\sqrt{1 + u^2}}$:
$$\dot{e}(t) = -v(t) \left( -\frac{\frac{k \cdot e}{v}}{\sqrt{1 + \left(\frac{k \cdot e}{v}\right)^2}} \right) = -\frac{k \cdot v(t) \cdot e(t)}{\sqrt{v(t)^2 + k^2 e(t)^2}}$$

Now, choose the quadratic **Lyapunov Candidate Function**:
$$V(e) = \frac{1}{2} e(t)^2 \ge 0$$
Differentiating with respect to time:
$$\dot{V}(e) = e \cdot \dot{e} = -\frac{k \cdot v(t) \cdot e(t)^2}{\sqrt{v(t)^2 + k^2 e(t)^2}}$$

Because $k > 0$ and $v(t) > 0$:
$$\dot{V}(e) < 0 \quad \forall e \neq 0$$
By **Lyapunov's Direct Method**, $\dot{V}$ is strictly negative-definite! This proves mathematically that the cross-track error $e(t)$ converges exponentially to zero from any starting offset.

### 2. Pure Pursuit Law of Sines Derivation (MIT Duckietown)
Consider a vehicle with wheelbase $L$ steering toward a lookahead point at distance $L_d$ with heading angle $\alpha$.
The vehicle follows an instantaneous circular arc of radius $R$.
By the **Law of Sines** on the triangle connecting rear axle, center of curvature, and lookahead point:
$$\frac{L_d}{\sin(2\alpha)} = \frac{R}{\sin(\frac{\pi}{2} - \alpha)} \implies \frac{L_d}{2 \sin(\alpha) \cos(\alpha)} = \frac{R}{\cos(\alpha)} \implies R = \frac{L_d}{2 \sin(\alpha)}$$
The path curvature is $\kappa = \frac{1}{R} = \frac{2 \sin(\alpha)}{L_d}$.
From kinematic bicycle geometry, steering angle $\delta$ satisfies $\tan(\delta) = \frac{L}{R} = L \kappa$:
$$\delta = \arctan\left(\frac{2 L \sin(\alpha)}{L_d}\right)$$

---

## 📺 Recommended Free Video Tutorials to Learn the Math
- **Stanley Controller & Proof**: [Sebastian Thrun / Hoffmann DARPA Landmark Paper (IEEE 2007)](https://ai.stanford.edu/~thrun/papers/hoffmann.stanley.control07.pdf)
- **Lyapunov Stability Theorem**: [Brian Douglas: Introduction to Lyapunov Stability](https://www.youtube.com/watch?v=1Fq-XG198u8)
- **Pure Pursuit Geometric Derivation**: [MIT Duckietown: Lane Following & Pure Pursuit](https://docs.duckietown.org/)

---

## 🏎️ Why Tesla Does It This Way
- Drive-by-wire steering systems have mechanical and safety constraints: steering rate limits ($\le 30^\circ/\text{s}$) and lateral acceleration comfort limits ($a_{\text{lat}} = v^2 \kappa \le 3.0\text{ m/s}^2$).
- Controllers must run at **50Hz to 100Hz** with deterministic latency.
- In modern Tesla FSD, the neural planner outputs target waypoint paths, and the low-level controller translates these into exact steering wheel torque and drive motor commands.

---

## 🧪 Quick Run
```bash
# 1. Run the closed-loop controller benchmark
python modules/07_control_sim/simulator.py

# 2. Run the Break-It & Fix-It drill (control latency fishtailing vs lookahead compensation)
python modules/07_control_sim/break_it_fix_it.py

# 3. Verify unit tests
pytest modules/07_control_sim/tests/test_control.py
```

🎨 **Interactive Visual Explainer**: Open [`visual_explainers/07_closed_loop_simulator.html`](../../visual_explainers/07_closed_loop_simulator.html) to drive a live canvas car in real-time, toggle Stanley vs Pure Pursuit, and watch oscilloscope telemetry of cross-track error!
