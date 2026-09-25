# Deep Dive: Closed-Loop Control & Kinematics (Sebastian Thrun & Duckietown)

> "A controller connects abstract coordinate trajectories to physical tires rolling on asphalt."

---

## 1. The Kinematic Bicycle Model
Real passenger vehicles have four wheels with complex suspension. For trajectory tracking below tire slip limits ($a_{\text{lat}} \le 4\text{ m/s}^2$), the **Kinematic Bicycle Model** provides an accurate representation:

$$\dot{x} = v \cos(\psi)$$
$$\dot{y} = v \sin(\psi)$$
$$\dot{\psi} = \frac{v}{L} \tan(\delta)$$
$$\dot{v} = a$$
Where:
- $L$ is the wheelbase (distance from rear axle to front axle).
- $\delta$ is the front wheel steering angle.
- $\psi$ is the vehicle heading.
- $v$ is vehicle forward speed.
- $a$ is longitudinal acceleration (throttle/brake).

---

## 2. Sebastian Thrun's Stanley Controller
Developed for Stanford's autonomous car *Stanley*, winner of the 2005 DARPA Grand Challenge.

The steering law at the front axle is:
$$\delta(t) = \theta_e(t) + \arctan\left(\frac{k \cdot e(t)}{v(t) + \epsilon}\right)$$
Where:
- $\theta_e(t) = \psi_{\text{path}} - \psi_{\text{car}}$ is the heading alignment error.
- $e(t)$ is the signed cross-track error measured from the center of the front axle to the nearest point on the reference path.
- $k$ is a positive tuning gain.
- $\epsilon > 0$ is a small softening factor preventing division by zero at low speeds.

### Stability Proof (Lyapunov Candidate)
Let the cross-track error dynamics be:
$$\dot{e}(t) = -v(t) \sin(\theta_e - \delta)$$
Substituting the Stanley control law:
$$\dot{e}(t) = -v(t) \sin\left(-\arctan\left(\frac{k \cdot e(t)}{v(t)}\right)\right) = \frac{v(t) \cdot k \cdot e(t)}{\sqrt{v(t)^2 + k^2 e(t)^2}}$$
For a Lyapunov candidate $V(e) = \frac{1}{2} e^2$:
$$\dot{V}(e) = e \dot{e} = -\frac{k \cdot v(t) \cdot e^2}{\sqrt{v(t)^2 + k^2 e(t)^2}} \le 0$$
Since $\dot{V}(e) < 0$ for all $e \neq 0$ and $v > 0$, the cross-track error is **exponentially stable**!

---

## 3. Pure Pursuit Controller (MIT Duckietown Standard)
Pure pursuit fits a circular arc from the vehicle's rear axle to a lookahead point $(x_L, y_L)$ on the target path at distance $L_d$:
$$\delta = \arctan\left(\frac{2 L \sin(\alpha)}{L_d}\right)$$
Where $\alpha$ is the angle between the vehicle's heading vector and the lookahead vector.
- Lookahead distance is dynamically scheduled with speed: $L_d = k_{\text{look}} \cdot v + L_{\text{min}}$.

---

## 4. Actuation Latency & Delay Compensation (Smith Predictor)
In physical vehicles, computing neural perception, running the planner, and sending CAN bus messages introduces $\tau \approx 100\text{ms}$ of latency.
- Uncompensated latency introduces a phase lag of $\omega \tau$, causing high-speed fishtailing.
- **Lookahead Delay Compensation**: We project the vehicle's position forward by $\tau$ seconds before evaluating the controller:
  $$x_{\text{pred}} = x_f + v \cos(\psi) \cdot \tau$$
  $$y_{\text{pred}} = y_f + v \sin(\psi) \cdot \tau$$
  $$\psi_{\text{pred}} = \psi + \frac{v}{L} \tan(\delta_{\text{prev}}) \cdot \tau$$
Evaluating the Stanley control law at $(x_{\text{pred}}, y_{\text{pred}}, \psi_{\text{pred}})$ cancels out the latency phase lag, restoring tracking stability!
