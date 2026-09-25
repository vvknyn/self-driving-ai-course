# Chapter 08: Closed-Loop Control & Vehicle Kinematics

> **The Big Question**: *A self-driving car can compute the most beautiful mathematical trajectory in the world. But if the steering actuator commands turn the wheels even $1^\circ$ too aggressively at 70 mph, the car enters an uncontrollable sinusoidal death wobble that throws passengers across the road. How do we mathematically command physical steering actuators to track a millimeter-precise path through rain, speed variations, and tire dynamics?*

---

## 1. 🚨 The Real-World Dilemma: The 70 MPH Death Wobble

Imagine building a basic lateral steering controller for an autonomous vehicle:
- You measure cross-track error $e(t)$ (how many meters you are away from the lane center).
- You implement a standard Proportional (P) controller: $\delta = k \cdot e$.
  - If you are $0.5\text{ m}$ to the right, turn left by $5^\circ$.
  - If you are $0.5\text{ m}$ to the left, turn right by $5^\circ$.

### The Parking Lot vs The Highway
- In a parking lot at **$5\text{ mph}$ ($2.2\text{ m/s}$)**: The car smoothly glides into the center of the lane.
- On the freeway at **$75\text{ mph}$ ($33.5\text{ m/s}$)**:
  - The car drifts $0.2\text{ m}$ to the right.
  - The controller turns the steering wheel left by $2^\circ$.
  - Because velocity is high, the car shoots across the lane in **$0.15\text{ seconds}$**.
  - By the time the steering motor reverses, the car has overshot $0.4\text{ m}$ to the left!
  - It snaps back harder, overshooting by $0.8\text{ m}$, then $1.6\text{ m}$!
  - In less than 3 seconds, the car is in a violent, divergent **death wobble** that rolls the vehicle!

```
Cross-Track Error e(t) at 75 mph (Naive P-Controller):
   e(t) ▲
        │         ╭──╮               ╭──────╮ (Crash!)
        │        ╭╯  ╰╮             ╭╯      ╰──► Divergent Oscillation!
   0.0m ┼───────┼──────┼───────────┼───────────► Time (t)
        │      ╭╯      ╰╮         ╭╯
        │     ╭╯        ╰╮       ╭╯
        ▼    ╭╯          ╰───────╯
```

> [!CAUTION]
> **Why Linear Control Fails in Automotive Systems**  
> Vehicle heading change rate is governed by $\dot{\psi} = \frac{v}{L} \tan(\delta)$. **Lateral dynamics are multiplicatively proportional to speed ($v$)**. A steering command that produces a safe $0.1\text{ m/s}^2$ lateral acceleration at 10 mph produces a lethal $5.0\text{ m/s}^2$ sideways slide at 70 mph!

---

## 2. 💡 The Mental Model: The Kinematic Bicycle & The Stanley Controller

### The Kinematic Bicycle Model
A full 4-wheel passenger car with suspension geometry is mathematically complex. But for lateral steering control below $0.4g$ lateral acceleration, we collapse the two front wheels into a single virtual wheel and the two rear wheels into a single rear wheel:

```
            Front Wheel (Steered by δ)
                   \
                    \
                     ══════════════ [Wheelbase L] ══════════════
                                                               │
                                                               │
                                                       Rear Wheel (Fixed)
```

- Wheelbase $L$: Distance between front and rear axles (e.g. $2.8\text{ m}$).
- Heading angle $\psi$: Orientation of the car chassis.
- Steering angle $\delta$: Angle of the front steered tire.

### The Sebastian Thrun Breakthrough: The Stanley Controller
In 2005, Sebastian Thrun and his Stanford team won the historic **DARPA Grand Challenge** by inventing the **Stanley Controller**.

Instead of measuring error at the rear axle or center of gravity, Stanley measures cross-track error $e(t)$ **at the front axle**:

```
                              Front Axle (Error e measured HERE)
                                     │
                 Cross-Track Error e │
          ═══════════════════════════* ◄── Steering command points HERE
         Path Centerline
```

The Stanley steering law combines two independent terms:
1. **Heading Alignment**: $\theta_e = \psi_{\text{path}} - \psi_{\text{car}}$ (aligns wheels parallel to road).
2. **Non-Linear Cross-Track Correction**: Steers wheels toward the path center, **damped by speed in the denominator**:

$$\delta(t) = \theta_e(t) + \arctan\left( \frac{k \cdot e(t)}{v(t) + k_{\text{soft}}} \right)$$

Look at the denominator: **$v(t) + k_{\text{soft}}$**:
- At **low speed** ($v = 2\text{ m/s}$): $\frac{k \cdot e}{2}$ is large. The wheels turn aggressively to park.
- At **high speed** ($v = 35\text{ m/s}$): $\frac{k \cdot e}{35}$ is tiny! The controller automatically turns down its steering gain, **completely eliminating high-speed oscillations**!
- At **zero speed** ($v = 0$): $k_{\text{soft}}$ prevents division by zero!

---

## 3. 🧪 Lab Mission: Hands-On Simulator Experiments

Scroll to the **Interactive Closed-Loop Control Studio** at the top of this chapter:

1. **Experiment 1 (The Speed Instability Demonstration)**:
   - Set **Velocity ($v$)** to `30 m/s` (67 mph).
   - Set **Stanley Damping Parameter ($k_{\text{soft}}$)** to `0.0` and turn off velocity scaling.
   - Click **Run Simulation**.
   - *Observation*: Watch the car enter a violent sinusoidal oscillation, swinging outside the lane boundaries.
2. **Experiment 2 (Activating Stanley Speed Damping)**:
   - Reset the car. Enable **Stanley Speed Damping** ($k = 0.8, k_{\text{soft}} = 1.0$).
   - Run the simulation again at `30 m/s`.
   - *Observation*: The car tracks the curved S-bend lane with millimeter precision! Cross-track error stays below $0.03\text{ m}$.
3. **Experiment 3 (Actuator Limits & Steering Saturation)**:
   - Set **Max Steering Limit ($\delta_{\max}$)** to $10^\circ$ and drive through a sharp hairpin curve.
   - *Observation*: The actuator saturates, demonstrating why kinematic controllers must know physical hardware limits.

---

## 4. 🛠️ The Karpathy Build: Stanley Controller from Scratch

Here is the exact kinematic simulation and controller implemented in pure Python:

```python
import numpy as np

class KinematicBicycleModel:
    """
    Bicycle kinematic motion model.
    State: [x, y, yaw, v]
    """
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, L=2.87):
        self.x = x
        self.y = y
        self.yaw = yaw      # Radians
        self.v = v          # m/s
        self.L = L          # Wheelbase in meters

    def update(self, throttle_acc: float, steer_delta: float, dt: float = 0.05):
        """Advances physical state by time step dt using forward Euler."""
        # 1. Update positions based on current heading and velocity
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        
        # 2. Update heading based on front steering angle delta
        self.yaw += (self.v / self.L) * np.tan(steer_delta) * dt
        
        # 3. Update velocity based on acceleration
        self.v += throttle_acc * dt
        self.v = max(0.0, self.v)  # No reverse in forward model

class StanleyController:
    """
    Front-axle Stanley lateral tracking controller.
    Ref: Hoffmann et al. (Stanford Racing Team, 2007)
    """
    def __init__(self, k: float = 0.8, k_soft: float = 1.0, max_steer_deg: float = 35.0):
        self.k = k
        self.k_soft = k_soft
        self.max_steer = np.deg2rad(max_steer_deg)

    def compute_steering(self, vehicle: KinematicBicycleModel, 
                               path_x: float, path_y: float, path_yaw: float) -> float:
        """
        Computes front-wheel steering command delta.
        """
        # 1. Calculate front axle coordinates
        fx = vehicle.x + vehicle.L * np.cos(vehicle.yaw)
        fy = vehicle.y + vehicle.L * np.sin(vehicle.yaw)
        
        # 2. Heading error normalized to [-pi, pi]
        heading_error = path_yaw - vehicle.yaw
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        
        # 3. Cross-track error (vector from path point to front axle)
        dx = fx - path_x
        dy = fy - path_y
        # Cross product with path heading vector gives signed cross-track error
        cross_track_error = -dx * np.sin(path_yaw) + dy * np.cos(path_yaw)
        
        # 4. Stanley control law with velocity damping in denominator
        crosstrack_steering = np.arctan2(self.k * cross_track_error, 
                                        vehicle.v + self.k_soft)
        
        steer_cmd = heading_error + crosstrack_steering
        
        # 5. Strict actuator hardware clamp
        return float(np.clip(steer_cmd, -self.max_steer, self.max_steer))
```

---

## 5. 📐 Mathematical Rigor: Non-Linear Lyapunov Stability Proof

Why is the Stanley controller mathematically guaranteed to converge to the path without oscillating?

Consider the rate of change of cross-track error $\dot{e}(t)$ as the vehicle travels with velocity $v$:

$$\dot{e}(t) = -v \sin(\theta_e - \delta)$$

Substitute the Stanley steering law $\delta = \theta_e + \arctan\left(\frac{ke}{v}\right)$:

$$\dot{e}(t) = -v \sin\left( \theta_e - \left( \theta_e + \arctan\left(\frac{ke}{v}\right) \right) \right) = -v \sin\left( -\arctan\left(\frac{ke}{v}\right) \right)$$

Using the identity $\sin(\arctan(u)) = \frac{u}{\sqrt{1 + u^2}}$:

$$\dot{e}(t) = -v \left( -\frac{\frac{ke}{v}}{\sqrt{1 + (ke/v)^2}} \right) = -\frac{k e(t)}{\sqrt{1 + \left(\frac{k e(t)}{v}\right)^2}}$$

For small cross-track errors ($ke \ll v$), the square root denominator $\approx 1$, yielding:

$$\dot{e}(t) \approx -k \cdot e(t) \implies e(t) = e(0) \cdot e^{-k t}$$

This is an **exponentially decaying differential equation**! The error decays to zero at an exponential rate determined strictly by gain $k$, with **zero imaginary roots and zero oscillatory overshoot**!

---

## 6. 🩺 Andrew Ng's Diagnostic Field Guide

| Observed Symptom | Underlying Mathematical Mechanism | Verification Test | Production Fix |
| :--- | :--- | :--- | :--- |
| **Steady-state cross-track offset on banked highways** | **Gravitational Lateral Force**: Banked roads produce constant lateral acceleration $g \sin(\phi)$ that pure P-terms cannot cancel. | Measure mean cross-track error over 10 seconds of constant banking. | Add an integral anti-windup term ($K_i \int e \, dt$) or feedforward banking gravity compensation. |
| **Vehicle cuts corners aggressively on sharp curves** | **Preview Horizon Missing**: Front axle tracks current target point instead of looking ahead along the path curve. | Compare trajectory curvature $\kappa_{\text{path}}$ to vehicle path. | Add curvature feedforward: $\delta_{\text{ff}} = \arctan(L \cdot \kappa)$. |
| **Violent steering shudder at near-zero speeds ($v < 0.2\text{ m/s}$)** | **Singularity at Zero Velocity**: Division by $v$ explodes when $k_{\text{soft}} = 0$. | Check if steering command spikes when stopping at a red light. | Set $k_{\text{soft}} \ge 1.0\text{ m/s}$ and deadband steering when $v < 0.1\text{ m/s}$. |

---

## 7. 🎯 Self-Check: Test Your Mental Model

<details>
<summary><b>Q1: Why does the Stanley controller measure cross-track error at the FRONT axle rather than the REAR axle?</b></summary>

<br>

**Answer**: Because the front wheels are the steered wheels. If you measure error at the front axle, turning the steering wheel directly changes the rate of change of that error $\dot{e}(t)$ with zero lag. If you measure error at the rear axle, the front wheels must turn first, rotate the vehicle body, and only then translate the rear axle—introducing a non-minimum phase zero and physical transport delay that causes control instability.
</details>

<details>
<summary><b>Q2: What is the physical meaning of parameter $k_{\text{soft}}$ in the Stanley equation?</b></summary>

<br>

**Answer**: $k_{\text{soft}}$ has units of velocity ($\text{m/s}$). It prevents numerical singularity ($\frac{ke}{0}$) when the vehicle comes to a complete stop, and limits the maximum steering sensitivity at low speeds, preventing the steering motor from aggressively hunting when creeping through intersections.
</details>
