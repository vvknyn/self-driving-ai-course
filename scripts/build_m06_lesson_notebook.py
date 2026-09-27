#!/usr/bin/env python3
"""Regenerate the Module 06 lesson notebook and execute it.

Writes ``notebooks/07_lattice_trajectory_planning.ipynb`` next to this course
staging tree, with outputs saved. Every backticked number in the prose has to
appear in an earlier cell output.

    python staging/self-driving-ai-course/scripts/build_m06_lesson_notebook.py
"""

from __future__ import annotations

import re
import textwrap
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

COURSE_CWD = Path("/tmp/self-driving-ai-course")


def md(source: str):
    return new_markdown_cell(textwrap.dedent(source).strip() + "\n")


def code(source: str):
    return new_code_cell(textwrap.dedent(source).strip() + "\n")


def build() -> nbformat.NotebookNode:
    cells = []

    cells.append(md("""
    # Module 06 — Pick a path that does not hit the car ahead

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/notebooks/07_lattice_trajectory_planning.ipynb)

    A car is stopped in your lane. You could stay behind it, nudge left, or change lanes. Each of those is a path. This notebook draws a few paths, watches the shortest one run through the car, then scores them so a sideways path wins.

    Each section explains one idea, then runs code. **Predict first**, then execute the cell.
    """))

    cells.append(code("""
    import os, subprocess, sys
    from pathlib import Path

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    def keep_inline():
        # Put figures on the notebook backend so plt.show() renders here.
        matplotlib.use("module://matplotlib_inline.backend_inline", force=True)
        ip = None
        try:
            from IPython import get_ipython
            ip = get_ipython()
        except Exception:
            ip = None
        if ip is not None:
            ip.run_line_magic("matplotlib", "inline")

    %matplotlib inline

    def find_repo(start: Path) -> Path:
        for p in [start, start.parent]:
            if (p / "modules" / "06_trajectory_planner").is_dir():
                return p.resolve()
        return start.resolve()

    REPO = find_repo(Path.cwd())
    if not (REPO / "modules" / "06_trajectory_planner").is_dir():
        dest = Path.cwd() / "self-driving-ai-course"
        if not (dest / "modules" / "06_trajectory_planner").is_dir():
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/vvknyn/self-driving-ai-course.git", str(dest)],
                check=True,
            )
        REPO = dest.resolve()
        os.chdir(REPO)
    else:
        os.chdir(REPO)

    MOD = REPO / "modules" / "06_trajectory_planner"
    sys.path.insert(0, str(MOD))

    from cost_functions import TrajectoryCostEvaluator
    from lattice_planner import LatticePlanner, QuinticPolynomial, TrajectoryCandidate

    print("Repo:", REPO)
    print("module: modules/06_trajectory_planner")
    keep_inline()
    """))

    cells.append(md("## 1. Many possible futures"))
    cells.append(md("""
    A stopped car sits ahead. Over the next few seconds the ego car could hold the lane, drift halfway to the next lane, or move a full lane width to either side. Those are different futures. No score yet.

    **Predict:** the paths start at the same point. Do they finish at the same distance along the road?
    """))
    cells.append(code("""
    ego = {"x0": 0.0, "y0": 0.0, "v0": 15.0, "a0": 0.0}
    offsets = [-3.5, -1.75, 0.0, 1.75, 3.5]
    planner = LatticePlanner(target_speed=15.0)
    futures = planner.generate_candidate_trajectories(
        ego,
        lateral_offsets=offsets,
        planning_horizon=3.0,
        dt=0.1,
    )
    cruise = {}
    for traj in futures:
        if abs(traj.v[-1] - 15.0) < 1e-6:
            cruise[round(traj.y[-1], 2)] = traj

    print(f"candidates: {len(futures)}")
    print("car ahead x=24.0 y=0.0")
    for key in sorted(cruise):
        traj = cruise[key]
        print(f"end y={traj.y[-1]:.2f}  end x={traj.x[-1]:.2f}  end v={traj.v[-1]:.1f}")

    fig, ax = plt.subplots(figsize=(8, 3.4))
    for key in sorted(cruise):
        traj = cruise[key]
        ax.plot(traj.x, traj.y, label=f"end y={traj.y[-1]:.2f}")
    ax.scatter([24.0], [0.0], c="black", s=36, zorder=3)
    ax.annotate("car ahead", (24.0, 0.0), textcoords="offset points", xytext=(6, 8))
    ax.set_xlabel("x (m)  along the road")
    ax.set_ylabel("y (m)  sideways")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Five futures, same start")
    plt.tight_layout()
    plt.show()
    """))
    cells.append(md("""
    The cell prints `candidates: 15`. Five sideways targets and three end speeds make those fifteen paths. The five drawn here all finish at end speed `15.0` and at `end x=45.00`. Their sideways ends are `-3.50`, `-1.75`, `0.00`, `1.75`, and `3.50`. The stopped car is printed at `x=24.0` `y=0.0`, sitting on the middle path. Along-road distance matched. Sideways position did not.
    """))

    cells.append(md("## 2. Frenet: along the road, and sideways"))
    cells.append(md("""
    A **Frenet** frame is tied to the road. **s** is distance traveled along the road. **d** is sideways offset from the lane center, positive to the left.

    On a straight road pointing along +x, s is x and d is y. Rotate the road and the same pair lands somewhere else:

    x = s cos θ − d sin θ

    y = s sin θ + d cos θ

    **Predict:** s = 10 m and d = 3.5 m. Heading 0 (road along +x) should put that point at x = 10, y = 3.5. Heading 90° (road along +y) points "forward" along +y and "left" along −x.
    """))
    cells.append(code("""
    def road_xy(s, d, heading_rad):
        x = s * np.cos(heading_rad) - d * np.sin(heading_rad)
        y = s * np.sin(heading_rad) + d * np.cos(heading_rad)
        return float(x), float(y)

    print("sideways d")
    print("  +3.5 | . . . . .   left lane")
    print("   0.0 | . . car .   center lane")
    print("  -3.5 | . . . . .   right lane")
    print("       +---------->  along the road, s")

    s, d = 10.0, 3.5
    xy0 = road_xy(s, d, 0.0)
    xy90 = road_xy(s, d, np.pi / 2)
    print(f"s={s:.1f} d={d:.1f}")
    print(f"heading 0 deg   -> x={xy0[0]:.4f} y={xy0[1]:.4f}")
    print(f"heading 90 deg  -> x={xy90[0]:.4f} y={xy90[1]:.4f}")
    print("planner stores x along the road and y sideways (heading 0)")
    """))
    cells.append(md("""
    Heading 0 prints `x=10.0000` and `y=3.5000` for `s=10.0` `d=3.5`. Heading 90 prints `x=-3.5000` and `y=10.0000`. The diagram's lane lines are the same sideways offsets the fan used: `+3.5`, `0.0`, and `-3.5`. This planner keeps the heading-0 case. It writes along-road position into `x` and sideways position into `y` on a straight road.
    """))

    cells.append(md("## 3. A smooth lane change"))
    cells.append(md("""
    A lane change has to match six numbers: sideways position, speed, and acceleration at the start, and the same three at the end. A polynomial with powers t⁰ through t⁵ has six coefficients, so those six numbers pick one curve. That curve is a **quintic**.

    This lateral move starts at d = 0 with zero sideways speed and acceleration, and ends at 3.5 m after 3 seconds, again with zero sideways speed and acceleration. Along the road, speed stays 15 m/s, so s runs from 0 to 45 m.

    **Predict:** at 1.5 seconds, is the car already in the new lane, or still near the middle of the lane change?
    """))
    cells.append(code("""
    T = 3.0
    lat = QuinticPolynomial(0.0, 0.0, 0.0, 3.5, 0.0, 0.0, T)
    lon = QuinticPolynomial(0.0, 15.0, 0.0, 45.0, 15.0, 0.0, T)
    ts = np.linspace(0.0, T, 301)
    d_of_t = np.array([lat.calc_point(float(t)) for t in ts])
    s_of_t = np.array([lon.calc_point(float(t)) for t in ts])
    peak_jerk = max(abs(lat.calc_third_derivative(float(t))) for t in ts)

    print(f"d(0)={lat.calc_point(0.0):.3f}  d(1.5)={lat.calc_point(1.5):.3f}  d(3)={lat.calc_point(T):.3f}")
    print(f"s(0)={lon.calc_point(0.0):.3f}  s(1.5)={lon.calc_point(1.5):.3f}  s(3)={lon.calc_point(T):.3f}")
    print(
        "lateral speed at ends: "
        f"{lat.calc_first_derivative(0.0):.3f} and {lat.calc_first_derivative(T):.3f}"
    )
    print(f"peak |lateral jerk|={peak_jerk:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(ts, s_of_t)
    axes[0].set_xlabel("t (s)")
    axes[0].set_ylabel("s (m)")
    axes[0].set_title("along the road")
    axes[1].plot(ts, d_of_t)
    axes[1].axhline(1.75, color="0.55", ls="--", lw=1)
    axes[1].set_xlabel("t (s)")
    axes[1].set_ylabel("d (m)")
    axes[1].set_title("sideways")
    plt.tight_layout()
    plt.show()
    """))
    cells.append(md("""
    The prints are `d(0)=0.000`, `d(1.5)=1.750`, and `d(3)=3.500`. Halfway in time is halfway sideways. The end sideways speeds print as `0.000` and `-0.000` (a rounded zero). Along the road, `s(1.5)=22.500` and `s(3)=45.000`. Peak sideways jerk is `7.778`. The curve is smooth. It has not looked at the car yet.
    """))

    cells.append(md("## 4. The shortest path hits the car"))
    cells.append(md("""
    First attempt: keep the shortest of three cruise paths at 15 m/s. One stays in lane, one ends 1.75 m to the left, one ends a full 3.5 m lane to the left. The stopped car is at x = 24 m, y = 0, radius 1.5 m.

    **Predict:** the straight path is the shortest. Does its closest point sit outside the car?
    """))
    cells.append(code("""
    def cruise_traj(d_target, v_target=15.0, T=3.0, dt=0.1, v0=15.0):
        s_target = 0.5 * (v0 + v_target) * T
        lon_poly = QuinticPolynomial(0.0, v0, 0.0, s_target, v_target, 0.0, T)
        lat_poly = QuinticPolynomial(0.0, 0.0, 0.0, d_target, 0.0, 0.0, T)
        traj = TrajectoryCandidate()
        for t in np.arange(0.0, T + 1e-9, dt):
            traj.t.append(float(t))
            traj.x.append(float(lon_poly.calc_point(t)))
            traj.y.append(float(lat_poly.calc_point(t)))
            traj.v.append(float(lon_poly.calc_first_derivative(t)))
            traj.a.append(float(lon_poly.calc_second_derivative(t)))
            traj.jerk.append(float(lon_poly.calc_third_derivative(t)))
        return traj

    def path_length(traj):
        return float(sum(
            np.hypot(traj.x[i + 1] - traj.x[i], traj.y[i + 1] - traj.y[i])
            for i in range(len(traj.x) - 1)
        ))

    def min_clearance(traj, ox=24.0, oy=0.0):
        return float(min(np.hypot(x - ox, y - oy) for x, y in zip(traj.x, traj.y)))

    obstacle = {"x": 24.0, "y": 0.0, "radius": 1.5}
    names = [("straight", 0.0), ("half lane", 1.75), ("full lane", 3.5)]
    ranked = []
    print(f"obstacle x={obstacle['x']:.1f} y={obstacle['y']:.1f} radius={obstacle['radius']:.1f}")
    for name, d_target in names:
        traj = cruise_traj(d_target)
        length = path_length(traj)
        clearance = min_clearance(traj)
        ranked.append((length, name, traj, clearance))
        print(f"{name:10} length={length:.3f} m  min clearance={clearance:.3f} m  end y={traj.y[-1]:.2f}")

    ranked.sort()
    shortest = ranked[0]
    by_name = {name: (length, clearance) for length, name, _, clearance in ranked}
    extra = by_name["full lane"][0] - by_name["straight"][0]
    print(f"shortest: {shortest[1]}  length={shortest[0]:.3f} m  clearance={shortest[3]:.3f} m")
    print(f"extra length of full lane: {extra:.3f} m")
    print("hits the car:", shortest[3] < obstacle["radius"])

    fig, ax = plt.subplots(figsize=(8, 3.4))
    for _length, name, traj, _clearance in ranked:
        ax.plot(traj.x, traj.y, label=name)
    ax.add_patch(plt.Circle((obstacle["x"], obstacle["y"]), obstacle["radius"], fill=False, color="black"))
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Shortest path vs the car")
    plt.tight_layout()
    plt.show()
    """))
    cells.append(md("""
    The shortest path is `straight` at `45.000` m, and its clearance is `0.000` m. The car's radius is `1.5` m, so that path runs through the car. The half-lane path is `45.049` m long with clearance `0.984` m. The full-lane path is `45.193` m long, an extra `0.193` m, with clearance `1.968` m. Length picked the path that hits the car. That attempt fails.
    """))

    cells.append(md("## 5. Four costs, three candidates"))
    cells.append(md("""
    The fix is a score on every path. `TrajectoryCostEvaluator` averages four terms over the samples, then multiplies each average by a weight:

    - **collision** — inside a bubble of radius (obstacle radius + safety radius), add exp(3 × (bubble − distance))
    - **lane centering** — sideways distance from y = 0, squared
    - **jerk** — longitudinal jerk, squared
    - **speed** — (speed − target speed), squared

    **Predict:** the straight path's collision term swamps its total. Among the three, the full-lane path has the smallest total. Its clearance from the previous cell, `1.968` m, sits inside the bubble, so it can still pay a collision cost.
    """))
    cells.append(code("""
    evaluator = TrajectoryCostEvaluator()
    print(f"w_collision={evaluator.w_coll:.1f}")
    print(f"w_lane_center={evaluator.w_lane:.1f}")
    print(f"w_jerk={evaluator.w_jerk:.1f}")
    print(f"w_speed_progress={evaluator.w_speed:.1f}")
    print(f"safety_radius={evaluator.safety_radius:.1f}")
    r_eff = obstacle["radius"] + evaluator.safety_radius
    print(f"r_eff={r_eff:.1f}")

    scored = []
    for name, d_target in names:
        traj = cruise_traj(d_target)
        evaluator.score_trajectory(traj, [obstacle], target_lane_y=0.0, target_speed=15.0)
        costs = traj.costs
        scored.append((traj.total_cost, name))
        print(
            f"{name:10} collision={costs['collision']:.3f}  lane={costs['lane_centering']:.3f}  "
            f"jerk={costs['jerk_comfort']:.3f}  speed={costs['speed_progress']:.3f}  "
            f"total={traj.total_cost:.3f}"
        )
    scored.sort()
    print(f"lowest total: {scored[0][1]}  total={scored[0][0]:.3f}")
    """))
    cells.append(md("""
    The defaults print `w_collision=500.0`, `w_lane_center=2.0`, `w_jerk=0.5`, `w_speed_progress=1.0`, and `safety_radius=2.2`. The bubble radius is `r_eff=3.7`.

    Straight total `1091252.950` is the collision term `1091252.950`. Its lane, jerk, and speed terms are `0.000`. Half lane totals `65827.618` (collision `65825.197`, lane `2.421`). Full lane totals `4402.900` (collision `4393.216`, lane `9.684`). Jerk and speed stay `0.000` on all three because this cruise holds speed.

    The lowest total is the full lane, `4402.900`. Clearance `1.968` m is inside `r_eff=3.7`, so that winner still pays collision cost `4393.216`. The score moved the choice sideways. The bubble is a penalty, and the path still enters it.
    """))

    cells.append(md("## 6. The repo planner picks one"))
    cells.append(md("""
    `modules/06_trajectory_planner/run_planner.py` builds the fan and scores it with its own weights: collision 1000, lane 3, jerk 0.2. The speed weight stays at the class default. Sideways targets are −3.5, −2, 0, 2, and 3.5 m. End speeds are 4 m/s under the target, the target, and 2 m/s over it. The target here is 15 m/s. The stopped car is the same one.

    **Predict:** the chosen path ends off the center of the lane. A mirror path can tie it.
    """))
    cells.append(code("""
    ego_state = {"x0": 0.0, "y0": 0.0, "v0": 15.0, "a0": 0.0}
    obstacles = [{"x": 24.0, "y": 0.0, "radius": 1.5}]
    runner = LatticePlanner(target_speed=15.0)
    runner_eval = TrajectoryCostEvaluator(w_collision=1000.0, w_lane_center=3.0, w_jerk=0.2)
    candidates = runner.generate_candidate_trajectories(
        ego_state,
        lateral_offsets=[-3.5, -2.0, 0.0, 2.0, 3.5],
        planning_horizon=3.0,
        dt=0.1,
    )
    print(f"sampled: {len(candidates)}")
    print(f"w_collision={runner_eval.w_coll:.1f}")
    print(f"w_lane_center={runner_eval.w_lane:.1f}")
    print(f"w_jerk={runner_eval.w_jerk:.1f}")
    print(f"w_speed_progress={runner_eval.w_speed:.1f}")

    best = runner_eval.select_optimal_trajectory(
        candidates, obstacles, target_lane_y=0.0, target_speed=15.0
    )
    print(f"{'rank':<6}{'end_y':<8}{'end_v':<8}{'collision':<12}{'total':<12}{'clearance':<10}decision")
    for i, cand in enumerate(candidates[:6]):
        decision = "chosen" if i == 0 else "rejected"
        clear = min_clearance(cand)
        print(
            f"{i + 1:<6}{cand.y[-1]:<8.2f}{cand.v[-1]:<8.1f}{cand.costs['collision']:<12.3f}"
            f"{cand.total_cost:<12.3f}{clear:<10.3f}{decision}"
        )

    center = next(
        c for c in candidates if abs(c.y[-1]) < 1e-6 and abs(c.v[-1] - 15.0) < 1e-6
    )
    center_clear = min_clearance(center)
    print(
        f"center {center.y[-1]:.2f} {center.v[-1]:.1f} {center.costs['collision']:.3f} "
        f"{center.total_cost:.3f} {center_clear:.3f} rejected"
    )
    print(
        f"chosen end y={best.y[-1]:.2f} end v={best.v[-1]:.1f} "
        f"total={best.total_cost:.3f} clearance={min_clearance(best):.3f}"
    )

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(center.x, center.y, color="C3", label=f"rejected end y={center.y[-1]:.2f}")
    ax.plot(best.x, best.y, color="C0", lw=2.2, label=f"chosen end y={best.y[-1]:.2f}")
    ax.add_patch(plt.Circle((24.0, 0.0), 1.5, fill=False, color="black"))
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Chosen path and the rejected center path")
    plt.tight_layout()
    plt.show()
    """))
    cells.append(md("""
    The cell prints `sampled: 15`. Runner weights are `w_collision=1000.0`, `w_lane_center=3.0`, `w_jerk=0.2`, and `w_speed_progress=1.0`.

    Rank 1 is `chosen` at end y `-3.50`, end speed `11.0`, collision `5687.759`, total `5708.800`, clearance `2.183`. Rank 2 ends at `3.50` with the same total `5708.800`. The mirror paths tie, and the sort keeps the one built first.

    The center path at end speed `15.0` is rejected: clearance `0.000`, total `2182505.900`. The plot draws that path through the car and the chosen path off to the side.

    Clearance `2.183` is outside the car body (`1.5`) and inside the bubble (`3.7`). Collision is `5687.759` of the total `5708.800`. The same end y `-3.50` at end speed `15.0` has collision `8786.432`, higher than `5687.759`, so the slower sample ranks first.
    """))

    cells.append(md("## 7. Try this: raise the collision weight"))
    cells.append(md("""
    The runner's collision weight is large. Turn it down, watch which path wins, then put it back.

    **Predict:** a tiny collision weight lets a center path win. The runner's weight moves the chosen end off the center of the lane.
    """))
    cells.append(code("""
    def choose(w_collision):
        fan = LatticePlanner(target_speed=15.0).generate_candidate_trajectories(
            ego_state,
            lateral_offsets=[-3.5, -2.0, 0.0, 2.0, 3.5],
            planning_horizon=3.0,
            dt=0.1,
        )
        ev = TrajectoryCostEvaluator(w_collision=w_collision, w_lane_center=3.0, w_jerk=0.2)
        picked = ev.select_optimal_trajectory(
            fan, obstacles, target_lane_y=0.0, target_speed=15.0
        )
        return picked

    for w_collision in (0.001, 1000.0):
        picked = choose(w_collision)
        shown = f"{w_collision:.3f}" if w_collision < 1 else f"{w_collision:.1f}"
        clear = min_clearance(picked)
        print(
            f"w_collision={shown} end y={picked.y[-1]:.2f} end v={picked.v[-1]:.1f} "
            f"total={picked.total_cost:.3f} min clearance={clear:.3f} "
            f"coll={picked.costs['collision']:.3f}"
        )
    """))
    cells.append(md("""
    With `w_collision=0.001` the chosen path ends at `y=0.00` and `v=17.0`. Its total is `2.048`. The closest sample is `0.667` m from the car's center, inside the `1.5` m body, and the collision term is `0.420`. The other terms are willing to drive through the car when collision barely counts.

    With `w_collision=1000.0` the chosen path ends at `y=-3.50` and `v=11.0`. Total `5708.800`, clearance `2.183`, collision `5687.759`. Raising the weight is what moves the car sideways.
    """))

    cells.append(md("## 8. Exercises"))
    cells.append(md("""
    **Exercise — `min_clearance`.** Closest distance from a list of points to one obstacle center. Return that distance in meters. Leave the `TODO` as it is to use the reference implementation.
    """))
    cells.append(code("""
    def min_clearance_student(xs, ys, ox, oy):
        # TODO: minimum distance from the points (xs, ys) to (ox, oy)
        raise NotImplementedError

    def min_clearance_reference(xs, ys, ox, oy):
        return float(min(np.hypot(x - ox, y - oy) for x, y in zip(xs, ys)))

    def get_min_clearance_fn():
        try:
            min_clearance_student([0.0], [0.0], 0.0, 0.0)
        except NotImplementedError:
            print("Using reference min_clearance (TODO not implemented)")
            return min_clearance_reference
        return min_clearance_student

    clear_fn = get_min_clearance_fn()
    toy_xs = [0.0, 24.0, 45.0]
    toy_ys = [0.0, 0.0, 0.0]
    print("toy xs:", toy_xs)
    print("toy ys:", toy_ys)
    print("obstacle: (24.0, 0.0)")
    toy_clear = clear_fn(toy_xs, toy_ys, 24.0, 0.0)
    print("min_clearance:", f"{toy_clear:.4f}")
    assert abs(toy_clear - 0.0) < 1e-9
    print("✅ correct: min_clearance =", round(toy_clear, 4))
    """))
    cells.append(md("""
    The check prints `✅`. The toy points include `(24.0, 0.0)`, the same place as the obstacle, so the minimum distance is `0.0000`. The reference function is used because the `TODO` still raises. Replace the `TODO` and run the cell again if you want the check to call your function instead.
    """))
    cells.append(md("""
    <details><summary>Solution</summary>

    ```python
    return float(min(np.hypot(x - ox, y - oy) for x, y in zip(xs, ys)))
    ```

    </details>
    """))

    cells.append(md("""
    **Exercise — `quintic_row1`.** First row of the 3×3 system inside `QuinticPolynomial`: the coefficients of a₃, a₄, a₅ in the position constraint at time T. Return a tuple `(T**3, T**4, T**5)`. Leave the `TODO` as it is to use the reference implementation.
    """))
    cells.append(code("""
    def quintic_row1_student(T):
        # TODO: coefficients of a3, a4, a5 in p(T)
        raise NotImplementedError

    def quintic_row1_reference(T):
        return (T ** 3, T ** 4, T ** 5)

    def get_quintic_row1_fn():
        try:
            quintic_row1_student(1.0)
        except NotImplementedError:
            print("Using reference quintic_row1 (TODO not implemented)")
            return quintic_row1_reference
        return quintic_row1_student

    row_fn = get_quintic_row1_fn()
    row = tuple(float(v) for v in row_fn(3.0))
    print("T: 3.0")
    print("row1:", row)
    assert all(abs(a - b) < 1e-9 for a, b in zip(row, (27.0, 81.0, 243.0)))
    print("✅ correct: quintic_row1 =", row)
    """))
    cells.append(md("""
    The check prints `✅`. For `T: 3.0` the first row is `(27.0, 81.0, 243.0)`, which is T³, T⁴, and T⁵. The reference function is used because the `TODO` still raises.
    """))
    cells.append(md("""
    <details><summary>Solution</summary>

    ```python
    return (T ** 3, T ** 4, T ** 5)
    ```

    </details>
    """))

    cells.append(md("""
    The checked-in tests cover the same two ideas: a quintic hits its start and end boundaries, and the planner's chosen path ends off the obstacle.
    """))
    cells.append(code("""
    try:
        import pytest  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "modules/06_trajectory_planner/tests/test_planner.py", "-q"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    print(proc.stdout)
    if proc.stderr.strip():
        print(proc.stderr)
    print("returncode", proc.returncode)
    assert proc.returncode == 0
    """))
    cells.append(md("""
    The log contains `2 passed`, and the cell prints `returncode 0`.
    """))

    cells.append(md("## 9. Recap"))
    cells.append(md("""
    - Five sideways targets and three speeds produce `15` candidate paths. The 15 m/s slice finishes at `end x=45.00` with end y `-3.50`, `-1.75`, `0.00`, `1.75`, and `3.50`.
    - Frenet on this straight road: `s=10.0` and `d=3.5` at heading 0 is `x=10.0000`, `y=3.5000`. The same pair at heading 90 is `x=-3.5000`, `y=10.0000`.
    - The lane-change quintic prints `d(1.5)=1.750` and `d(3)=3.500`. Along the road, `s(3)=45.000`. Peak sideways jerk is `7.778`.
    - The shortest cruise path is `45.000` m with clearance `0.000` m. It runs through the car of radius `1.5` m. The full-lane path is only `0.193` m longer.
    - Default weights `500.0`, `2.0`, `0.5`, `1.0` and safety radius `2.2` give bubble `r_eff=3.7`. The full-lane total is `4402.900`, and collision is still `4393.216` because clearance `1.968` is inside the bubble.
    - The repo runner (`w_collision=1000.0`, `w_lane_center=3.0`, `w_jerk=0.2`) chooses end y `-3.50` at `11.0` m/s, total `5708.800`. The mirror path ties at the same total.
    - Collision weight `0.001` chooses a center path with clearance `0.667` m. Weight `1000.0` chooses end y `-3.50`.
    - The module tests print `2 passed`.

    ### Go deeper
    - [Werling, Ziegler, Kammel, and Thrun — Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame (ICRA 2010)](https://doi.org/10.1109/ROBOT.2010.5509799)

    Module 07 takes the chosen (x, y) samples and turns them into a steering angle and a speed command.
    """))

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nb.metadata["colab"] = {"provenance": []}
    return nb


def _outputs_text(cell) -> str:
    chunks = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            chunks.append(out.get("text", ""))
        elif out.get("output_type") in {"execute_result", "display_data"}:
            data = out.get("data", {})
            chunks.append(data.get("text/plain", ""))
    return "".join(chunks)


def assert_lesson(nb: nbformat.NotebookNode) -> None:
    banned = re.compile(r"\b(beat|scaffold|contracts?)\b", re.I)
    agg = re.compile(r"matplotlib\.use\(\s*['\"]Agg['\"]")
    number = re.compile(r"`(-?\d+(?:\.\d+)?)`")
    seen = ""
    headings = []
    for cell in nb.cells:
        source = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        hit = banned.search(source)
        if hit:
            raise SystemExit(f"banned word {hit.group(0)!r} in cell")
        if agg.search(source):
            raise SystemExit("Agg backend is not allowed")
        if cell.cell_type == "markdown":
            for match in number.finditer(source):
                token = match.group(1)
                if token not in seen:
                    raise SystemExit(f"number `{token}` is not in an earlier output")
            for line in source.splitlines():
                if line.startswith("#"):
                    headings.append(line)
        elif cell.cell_type == "code":
            if cell.execution_count is None:
                raise SystemExit("code cell was not executed")
            seen += _outputs_text(cell)
    print("headings:")
    for line in headings:
        print(" ", line)


def main() -> None:
    if not (COURSE_CWD / "modules" / "06_trajectory_planner" / "lattice_planner.py").is_file():
        raise SystemExit(f"course repo missing at {COURSE_CWD}")
    out = Path(__file__).resolve().parents[1] / "notebooks" / "07_lattice_trajectory_planning.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(COURSE_CWD)}},
    )
    started = time.perf_counter()
    client.execute()
    elapsed = time.perf_counter() - started
    assert_lesson(nb)
    nbformat.write(nb, out)
    print(f"Wrote {out} ({len(nb.cells)} cells) in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
