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

## 📐 Mathematical Formulation

### 1. The Kinematic Bicycle Model
We simplify a 4-wheel car into a 2-wheel bicycle with wheelbase $L$ (front axle to rear axle distance):

```
       Front Axle (Steering δ)
           ┌───┐
           │   │ ──► Velocity v
           └───┘
             │
             │ Wheelbase L
             │
           ┌───┐
           │   │ ──► Rear Axle (Heading ψ)
           └───┘
```

The differential equations of motion are:
$$\dot{x} = v \cos(\psi)$$
$$\dot{y} = v \sin(\psi)$$
$$\dot{\psi} = \frac{v}{L} \tan(\delta)$$
$$\dot{v} = a$$

### 2. Sebastian Thrun's Stanley Controller
The Stanley controller calculates the steering angle $\delta(t)$ at the front axle using two intuitive terms:
$$\delta(t) = \underbrace{(\psi_{\text{path}} - \psi_{\text{car}})}_{\text{Heading Alignment Error}} + \underbrace{\arctan\left(\frac{k \cdot e(t)}{v(t) + \epsilon}\right)}_{\text{Cross-Track Error Correction}}$$

Where:
- $e(t)$ is the perpendicular distance (cross-track error) from the front axle to the nearest path point.
- When $e(t)$ is large, $\arctan\left(\frac{k \cdot e}{v}\right) \to \pm \frac{\pi}{2}$ ($90^\circ$), pointing the wheels aggressively back toward the track.
- As the vehicle converges to the path ($e(t) \to 0$), the correction vanishes and the car matches the road heading $\psi_{\text{path}}$.

### 3. MIT Duckietown Pure Pursuit Controller
Pure pursuit looks ahead at distance $L_d = k_{\text{look}} \cdot v$:
$$\delta(t) = \arctan\left(\frac{2 L \sin(\alpha)}{L_d}\right)$$
Where $\alpha$ is the angle between the vehicle's heading vector and the lookahead point.

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
