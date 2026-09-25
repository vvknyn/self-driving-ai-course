# Module 06: Trajectory Planning (Neural & Cost-Map Lattice Planner)

> "Perception is not the goal. Safe, comfortable navigation is. Planning is where the car turns beliefs into physical action." — OpenDriveLab (UniAD)

---

## 🎯 Purpose & Learning Goals
You have multi-camera images lifted into Bird's-Eye View, 3D occupancy voxels, and dynamic vehicle tracks. Now comes the defining question of autonomous driving: **Where should the vehicle drive over the next 5 seconds?**

In this module, you will:
1. Master **Quintic Polynomial Trajectories**: Solve boundary value problems that guarantee continuous position, velocity, and acceleration while minimizing passenger jerk.
2. Build a **Lattice Trajectory Sampler**: Fan out candidate trajectories spanning lane-keeping, obstacle nudging, and lane-changing maneuvers.
3. Formulate a **Multi-Objective Cost Function**:
   - Connect to **MITx Expected Loss Minimization**: We evaluate candidate paths across collision risk, lane centering, comfort (jerk), and progress.
4. Adopt the **OpenDriveLab Planning-Oriented Paradigm**: Direct end-to-end evaluation against dynamic obstacles and occupancy fields.
5. FastAI "Break It & Fix It": Encounter the famous **"Freezing Robot Problem"** where overly cautious cost weights paralyze the vehicle in traffic, and implement probabilistic gap acceptance to fix it.

---

## 📐 Mathematical Formulation

```
                                  ┌── Candidate 1 (Overtake Left) ── Cost: 18.4
                                  │
Ego State [x0, y0, v0, a0] ───────┼── Candidate 2 (Stay in Lane) ─── Cost: 4.2  (OPTIMAL)
                                  │
                                  └── Candidate 3 (Dodge Right) ──── Cost: 142.0 (COLLISION)
```

### 1. Why Quintic (5th-Order) Polynomials?
A 5th-order polynomial has 6 unknown coefficients:
$$s(t) = c_0 + c_1 t + c_2 t^2 + c_3 t^3 + c_4 t^4 + c_5 t^5$$

Its derivatives are:
$$\dot{s}(t) = \text{Velocity: } c_1 + 2 c_2 t + 3 c_3 t^2 + 4 c_4 t^3 + 5 c_5 t^4$$
$$\ddot{s}(t) = \text{Acceleration: } 2 c_2 + 6 c_3 t + 12 c_4 t^2 + 20 c_5 t^3$$
$$\dddot{s}(t) = \text{Jerk (Comfort): } 6 c_3 + 24 c_4 t + 60 c_5 t^2$$

At time $t=0$, initial conditions give:
$$c_0 = s_0, \quad c_1 = v_0, \quad c_2 = \frac{1}{2} a_0$$

At target planning horizon $t=T$, target boundary conditions $[s_T, v_T, a_T]$ form a solvable $3 \times 3$ linear system for $[c_3, c_4, c_5]$:
$$\begin{bmatrix} T^3 & T^4 & T^5 \\ 3 T^2 & 4 T^3 & 5 T^4 \\ 6 T & 12 T^2 & 20 T^3 \end{bmatrix} \begin{bmatrix} c_3 \\ c_4 \\ c_5 \end{bmatrix} = \begin{bmatrix} s_T - (s_0 + v_0 T + \frac{1}{2} a_0 T^2) \\ v_T - (v_0 + a_0 T) \\ a_T - a_0 \end{bmatrix}$$

### 2. Multi-Objective Cost Function (MITx Risk Minimization)
For each sampled trajectory $\tau$, we compute its expected cost:
$$J(\tau) = w_{\text{coll}} J_{\text{coll}}(\tau) + w_{\text{lane}} J_{\text{lane}}(\tau) + w_{\text{jerk}} J_{\text{jerk}}(\tau) + w_{\text{speed}} J_{\text{speed}}(\tau)$$
Where:
- $J_{\text{coll}} = \sum_{t} \exp\left(-\frac{d(p(t), \text{obstacle})^2}{2 \sigma_{\text{safety}}^2}\right)$ (Gaussian safety bubble).
- $J_{\text{lane}} = \int_0^T (y(t) - y_{\text{lane}}(x(t)))^2 dt$ (lane centering).
- $J_{\text{jerk}} = \int_0^T (\dddot{s}(t))^2 dt$ (passenger comfort / motion sickness).
- $J_{\text{speed}} = \int_0^T (v(t) - v_{\text{target}})^2 dt$ (traffic progress).

The optimal trajectory is selected via:
$$\tau^* = \arg\min_{\tau \in \mathcal{T}} J(\tau)$$

---

## 🏎️ Why Tesla Does It This Way
- Tesla FSD V10/V11 used a C++ Monte Carlo Tree Search (MCTS) optimizer over cost maps with hundreds of thousands of lines of handwritten heuristics.
- In **FSD V12**, Elon Musk and Ashok Elluswamy replaced the heuristic planner with an **end-to-end neural network planner** that mimics expert human trajectories conditioned on the vector space scene graph, avoiding brittle edge-case rules!

---

## 🧪 Quick Run
```bash
# 1. Run the lattice trajectory planner through an obstacle avoidance scenario
python modules/06_trajectory_planner/run_planner.py

# 2. Run the Break-It & Fix-It drill (freezing robot problem vs risk discounting)
python modules/06_trajectory_planner/break_it_fix_it.py

# 3. Verify unit tests
pytest modules/06_trajectory_planner/tests/test_planner.py
```

🎨 **Interactive Visual Explainer**: Open [`visual_explainers/06_trajectory_planner.html`](../../visual_explainers/06_trajectory_planner.html) to interactively drag obstacles, adjust jerk weights, and watch trajectory candidate fanouts re-rank in real time!
