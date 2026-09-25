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

## 📐 How the Formulas Are Obtained

```
                                  ┌── Candidate 1 (Overtake Left) ── Cost: 18.4
                                  │
Ego State [x0, y0, v0, a0] ───────┼── Candidate 2 (Stay in Lane) ─── Cost: 4.2  (OPTIMAL)
                                  │
                                  └── Candidate 3 (Dodge Right) ──── Cost: 142.0 (COLLISION)
```

### 1. Step-by-Step Derivation of the Quintic Polynomial System
Why a 5th-order polynomial? Because we have 6 physical boundary conditions to satisfy simultaneously:
- At start $t=0$: Initial position $s_0$, initial speed $v_0$, initial acceleration $a_0$.
- At end $t=T$: Target position $s_T$, target speed $v_T$, target acceleration $a_T$.

Let:
$$s(t) = c_0 + c_1 t + c_2 t^2 + c_3 t^3 + c_4 t^4 + c_5 t^5$$
Taking derivatives:
$$\dot{s}(t) = c_1 + 2 c_2 t + 3 c_3 t^2 + 4 c_4 t^3 + 5 c_5 t^4$$
$$\ddot{s}(t) = 2 c_2 + 6 c_3 t + 12 c_4 t^2 + 20 c_5 t^3$$
$$\dddot{s}(t) = 6 c_3 + 24 c_4 t + 60 c_5 t^2 \quad \text{(Jerk)}$$

Evaluating at $t = 0$:
$$s(0) = c_0 = s_0$$
$$\dot{s}(0) = c_1 = v_0$$
$$\ddot{s}(0) = 2 c_2 = a_0 \implies c_2 = \frac{a_0}{2}$$

Evaluating at $t = T$ and moving known terms ($c_0, c_1, c_2$) to the right-hand side:
$$\begin{bmatrix} T^3 & T^4 & T^5 \\ 3 T^2 & 4 T^3 & 5 T^4 \\ 6 T & 12 T^2 & 20 T^3 \end{bmatrix} \begin{bmatrix} c_3 \\ c_4 \\ c_5 \end{bmatrix} = \begin{bmatrix} s_T - (s_0 + v_0 T + \frac{1}{2} a_0 T^2) \\ v_T - (v_0 + a_0 T) \\ a_T - a_0 \end{bmatrix}$$

This matrix $A$ has non-zero determinant $\det(A) = 2 T^9$.
Because $T > 0$, the inverse $A^{-1}$ always exists, giving the unique minimum-jerk trajectory:
$$\min \int_0^T (\dddot{s}(t))^2 dt$$

---

## 📺 Recommended Free Video Tutorials to Learn the Math
- **Boundary Value Problems & Trajectory Optimization**: [MIT 6.832: Underactuated Robotics (Prof. Russ Tedrake)](https://underactuated.csail.mit.edu/)
- **Quintic Frenet Frame Planning**: [Werling et al. ICRA 2010 Landmark Paper](https://www.researchgate.net/publication/224155184_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame)
- **Calculus of Variations & Jerk**: [3Blue1Brown: Higher Derivatives & Taylor Series](https://www.youtube.com/watch?v=3d6DsjIBzJ4)

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
