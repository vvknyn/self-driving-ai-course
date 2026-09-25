# Deep Dive: Trajectory Planning & Cost Maps (OpenDriveLab UniAD Paradigm)

> "Perception is not the goal. Driving safely is. Planning is where the car turns beliefs into physical action." — OpenDriveLab (UniAD)

---

## 1. The Planning-Oriented Paradigm (OpenDriveLab UniAD)
Traditional autonomous driving treated perception, tracking, prediction, and planning as isolated, modular blocks.
- **Error Propagation Flaw**: A 5% error in bounding box estimation caused a 15% error in tracking, which caused a 40% error in trajectory prediction, causing the planner to panic or collide.
- **UniAD Revolution (CVPR 2023 Best Paper)**: All perception queries (BEV queries, agent queries, map queries) flow directly into a **Planning Query**. The entire stack is optimized end-to-end to minimize trajectory planning error.

---

## 2. Boundary Value Problem with Quintic Polynomials
A planned trajectory over time horizon $T$ must connect current vehicle conditions to desired terminal conditions smoothly:
$$s(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$$

Boundary conditions:
- **At $t = 0$**: Current position $s_0$, current velocity $v_0 = \dot{s}(0)$, current acceleration $a_0 = \ddot{s}(0)$.
- **At $t = T$**: Target position $s_T$, target velocity $v_T = \dot{s}(T)$, target acceleration $a_T = \ddot{s}(T)$.

Setting up the boundary system:
$$s(0) = a_0 = s_0$$
$$\dot{s}(0) = a_1 = v_0$$
$$\ddot{s}(0) = 2 a_2 = a_0 \implies a_2 = \frac{a_0}{2}$$

For the remaining three unknowns $[a_3, a_4, a_5]^T$:
$$\begin{bmatrix} T^3 & T^4 & T^5 \\ 3 T^2 & 4 T^3 & 5 T^4 \\ 6 T & 12 T^2 & 20 T^3 \end{bmatrix} \begin{bmatrix} a_3 \\ a_4 \\ a_5 \end{bmatrix} = \begin{bmatrix} s_T - (s_0 + v_0 T + \frac{1}{2} a_0 T^2) \\ v_T - (v_0 + a_0 T) \\ a_T - a_0 \end{bmatrix}$$

Solving this $3 \times 3$ system produces the unique 5th-order polynomial that minimizes the integral of squared jerk:
$$\min \int_0^T (\dddot{s}(t))^2 dt$$
Guaranteeing maximum passenger comfort!

---

## 3. Multi-Objective Cost Function
For every candidate trajectory $\tau$, we compute its expected loss:
$$J(\tau) = w_{\text{coll}} J_{\text{coll}} + w_{\text{lane}} J_{\text{lane}} + w_{\text{jerk}} J_{\text{jerk}} + w_{\text{speed}} J_{\text{speed}}$$

1. **Collision Risk ($J_{\text{coll}}$)**:
   $$J_{\text{coll}} = \sum_{t} \sum_{\text{obs}} \exp\left( \frac{r_{\text{safe}} - d(p(t), \text{obs})}{\sigma} \right)$$
2. **Lane Centering ($J_{\text{lane}}$)**:
   $$J_{\text{lane}} = \int_0^T (y(t) - y_{\text{lane}}(x(t)))^2 dt$$
3. **Passenger Jerk Comfort ($J_{\text{jerk}}$)**:
   $$J_{\text{jerk}} = \int_0^T (\dddot{s}(t))^2 dt$$
4. **Traffic Speed Progress ($J_{\text{speed}}$)**:
   $$J_{\text{speed}} = \int_0^T (v(t) - v_{\text{target}})^2 dt$$

The optimal trajectory is selected via:
$$\tau^* = \arg\min_{\tau \in \mathcal{T}} J(\tau)$$
