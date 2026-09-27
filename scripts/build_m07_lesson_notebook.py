#!/usr/bin/env python3
"""Regenerate the Module 07 lesson notebook.

Writes ``staging/self-driving-ai-course/notebooks/08_closed_loop_control_stanley.ipynb``.
The notebook is the lesson: run it top to bottom. This script does not execute it.

    python scripts/build_m07_lesson_notebook.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(source: str):
    return new_markdown_cell(textwrap.dedent(source).strip() + "\n")


def code(source: str):
    return new_code_cell(textwrap.dedent(source).strip() + "\n")


def build() -> nbformat.NotebookNode:
    cells = []

    cells.append(md("""
    # Module 07 — Steer so the plan actually happens

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/notebooks/08_closed_loop_control_stanley.ipynb)

    A planner can hand you a list of points. The car still needs a steering angle every few hundredths of a second. This notebook takes one step of the kinematic bicycle in `modules/07_control_sim`, watches a first rule (point the wheels at the next waypoint) overshoot and cut a corner, then follows a curve with the repo's Stanley controller.

    Each section explains one idea, then runs code. **Predict first**, then execute the cell.

    Everything here is simulation.
    """))

    cells.append(code("""
    import os, subprocess, sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    %matplotlib inline

    def find_repo(start: Path) -> Path:
        for p in [start, *start.parents]:
            if (p / "modules" / "07_control_sim" / "bicycle_model.py").is_file():
                return p.resolve()
        return start.resolve()

    REPO = find_repo(Path.cwd())
    if not (REPO / "modules" / "07_control_sim" / "bicycle_model.py").is_file():
        dest = Path.cwd() / "self-driving-ai-course"
        if not (dest / "modules" / "07_control_sim" / "bicycle_model.py").is_file():
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/vvknyn/self-driving-ai-course.git", str(dest)],
                check=True,
            )
        REPO = dest.resolve()

    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "modules" / "07_control_sim"))

    from bicycle_model import KinematicBicycleModel, VehicleState
    from controllers import PIDLongitudinalController, StanleyController
    from simulator import generate_curved_track

    def wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    print("Repo:", REPO)
    print("wheelbase:", KinematicBicycleModel().L, "m")
    """))

    cells.append(md("## 1. A planned path is not a drive"))
    cells.append(md("""
    Module 06 ends with arrays of `x` and `y`. Those arrays are a drawing. The bicycle moves only when you pass a steering angle and an acceleration into `KinematicBicycleModel.step`.

    Hold the wheel at 0 and the car travels along its current heading. The path can bend away from that line.

    **Predict:** 8 m/s for 4 seconds, heading 0, steer 0. The car's `y` stays 0. Does the path `y = 4 sin(x / 15)` stay 0 at the car's `x`?
    """))

    cells.append(code("""
    path_x = np.arange(0.0, 40.0, 0.5)
    path_y = 4.0 * np.sin(path_x / 15.0)

    car = KinematicBicycleModel()
    state = VehicleState(x=0.0, y=0.0, psi=0.0, v=8.0)
    dt = 0.05
    xs, ys = [state.x], [state.y]
    for _ in range(80):
        state = car.step(state, throttle_accel=0.0, steer_delta=0.0, dt=dt)
        xs.append(state.x)
        ys.append(state.y)

    nearest = int(np.argmin(np.abs(path_x - state.x)))
    print("speed 8.0 m/s for 4.00 s, steer 0 deg")
    print(f"car x, y = {state.x:.3f}, {state.y:.3f}")
    print(f"path y at that x = {path_y[nearest]:.3f}")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(path_x, path_y, "k--", label="planned path")
    ax.plot(xs, ys, label="steer held at 0")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()
    ax.set_title("A path does not steer the car")
    plt.show()
    """))

    cells.append(md("""
    After 4.00 s the car is at `x = 32.000`, `y = 0.000`. The path at that `x` is `y = 3.384`. Straight integration never sent a steering command, so the plan and the car have already separated by more than 3 m.
    """))

    cells.append(md("## 2. One step of the bicycle"))
    cells.append(md("""
    Rear-axle state is `(x, y, psi, v)`. Heading `psi` is in radians, and 0 points along +x. `L` is the wheelbase, 2.8 m here. `delta` is the front-wheel angle.

    `step` is one explicit Euler update. Position moves along the heading you have **now**. Heading then changes by `(v / L) tan(delta) dt`. Acceleration is clipped to [-6, 3.5] m/s², steering to ±35°, and speed is kept at 0 or above.

    **Predict:** `v = 8`, `delta = 10°`, `L = 2.8`, `dt = 0.5`, starting heading `0.4` rad. The heading change is `(v / L) tan(delta) dt`, a bit more than 0.25 rad. The new heading should land near 0.65 rad. `y` should increase, because the old heading is not zero.
    """))

    cells.append(code("""
    v = 8.0
    delta = np.deg2rad(10.0)
    dt = 0.5
    x0, y0, psi0 = 0.0, 1.0, 0.4

    car = KinematicBicycleModel()
    heading_change = (v / car.L) * np.tan(delta) * dt
    state = VehicleState(x=x0, y=y0, psi=psi0, v=v)
    new_state = car.step(state, throttle_accel=0.0, steer_delta=delta, dt=dt)

    print(f"v = {v:.1f} m/s")
    print(f"delta = {np.degrees(delta):.2f} deg")
    print(f"L = {car.L:.1f} m")
    print(f"dt = {dt:.1f} s")
    print(f"start x, y, psi = {x0:.3f}, {y0:.3f}, {psi0:.4f} rad")
    print(f"heading change = {heading_change:.4f} rad")
    print(f"new x, y, psi = {new_state.x:.3f}, {new_state.y:.3f}, {new_state.psi:.4f} rad")
    """))

    cells.append(md("""
    The cell prints `v = 8.0`, `delta = 10.00 deg`, `L = 2.8`, `dt = 0.5`. Heading change is `0.2519` rad, so the new heading is `0.4000 + 0.2519 = 0.6519` rad. The new position is `(3.684, 2.558)`. That `y` step used `sin(0.4)`, the heading from before the update.
    """))

    cells.append(md("## 3. Point the wheels at the next waypoint"))
    cells.append(md("""
    First controller: take the nearest path point to the rear axle, then aim the wheels at the **next** waypoint. The command is the full angle from the car's heading to that point, clipped to ±35°.

    **Predict:** the car starts 2 m left of a straight road whose waypoints are 0.5 m apart. The next waypoint is almost beside you, not far ahead. The aim angle is large. The car will cross the line, not settle onto it.

    The second picture is a quarter-circle of radius 15 m, stored as a waypoint every 30°. Aiming at the next of those sparse points cuts inside the corner.
    """))

    cells.append(code("""
    def aim_at_next(state, path_x, path_y, car):
        dist = np.hypot(path_x - state.x, path_y - state.y)
        nearest = int(np.argmin(dist))
        idx = min(nearest + 1, len(path_x) - 1)
        alpha = wrap_angle(np.arctan2(path_y[idx] - state.y, path_x[idx] - state.x) - state.psi)
        delta = float(np.clip(alpha, -car.max_steer, car.max_steer))
        return delta, alpha, idx

    straight_x = np.arange(0.0, 80.0, 0.5)
    straight_y = np.zeros_like(straight_x)
    car = KinematicBicycleModel()
    state = VehicleState(x=0.0, y=2.0, psi=0.0, v=8.0)
    aim_x, aim_y = [], []
    for i in range(160):
        delta, alpha, idx = aim_at_next(state, straight_x, straight_y, car)
        if i == 0:
            first_alpha, first_delta = alpha, delta
        state = car.step(state, throttle_accel=0.0, steer_delta=delta, dt=0.05)
        aim_x.append(state.x)
        aim_y.append(state.y)
    aim_y = np.array(aim_y)
    crossings = int(np.sum(aim_y[1:] * aim_y[:-1] < 0))
    print(f"first aim {np.degrees(first_alpha):.2f} deg")
    print(f"clamped steer {np.degrees(first_delta):.2f} deg")
    print(f"y crossings {crossings}")
    print(f"lowest y {aim_y.min():.3f} m")

    R = 15.0
    step_deg = 30.0
    theta = np.deg2rad(np.arange(0.0, 90.0 + step_deg, step_deg))
    corner_x = R * np.sin(theta)
    corner_y = R * (1.0 - np.cos(theta))
    state = VehicleState(x=0.0, y=0.0, psi=0.0, v=10.0)
    cut_x, cut_y = [], []
    for _ in range(120):
        delta, alpha, idx = aim_at_next(state, corner_x, corner_y, car)
        state = car.step(state, throttle_accel=0.0, steer_delta=delta, dt=0.05)
        cut_x.append(state.x)
        cut_y.append(state.y)
        if state.x > R * 0.95 and state.y > R * 0.65:
            break
    cut_x = np.array(cut_x)
    cut_y = np.array(cut_y)
    radius = np.hypot(cut_x, cut_y - R)
    print(f"corner radius {R:.1f} m")
    print(f"closest radius {radius.min():.3f} m")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(straight_x, straight_y, "k--", label="path")
    axes[0].plot(aim_x, aim_y, label="aim at next waypoint")
    axes[0].set_xlim(0, 40)
    axes[0].set_ylim(-2, 2.5)
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_title("Straight road, overshoot")
    axes[0].legend()

    arc = np.linspace(0.0, np.pi / 2, 200)
    axes[1].plot(R * np.sin(arc), R * (1.0 - np.cos(arc)), color="0.75", label="15 m radius")
    axes[1].plot(corner_x, corner_y, "k.", label="waypoints")
    axes[1].plot(cut_x, cut_y, label="aim at next waypoint")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].set_title("Coarse corner, cut inside")
    axes[1].legend()
    plt.tight_layout()
    plt.show()
    """))

    cells.append(md("""
    The first aim is `-75.96` deg. The bicycle clips it to `-35.00` deg, a full lock toward the line. `y` then crosses 0 **15** times and reaches `-1.386` m. Pointing at a nearby waypoint turns the whole geometry error into a steering angle, so the car swings past the path.

    On the coarse corner the closest approach is radius `14.540` m, inside the `15.0` m arc. The next waypoint sits across the chord, and the car follows that chord.
    """))

    cells.append(md("## 4. Stanley, computed by hand"))
    cells.append(md("""
    Stanley, as implemented in `StanleyController.compute_steering`, is two angles added together.

    1. **Heading error:** wrap `path_psi - vehicle_psi` into `[-π, π]`.
    2. **Cross-track angle:** `atan2(k * (-e), max(0.1, v) + soft_factor)`.

    `e` is the signed distance from the nearest path point to the **front** axle, using the path normal `(-sin ψ_path, cos ψ_path)`. The second angle has tangent `(k * e) / (speed + a floor)`. Faster speed, smaller extra steer. In the tests `k = 0.8` and `soft_factor = 1`, so at 10 m/s the denominator is 11.

    On a road along +x the normal is `(0, 1)`. A front axle at `y = 2` has `e = +2` m. The unit test expects a **negative** steer: turn right, back toward the line.

    **Predict:** heading error 0, `e = 2` m, `v = 10` m/s, `k = 0.8`. Is this correction close to the `-35°` clip from the last section, or a much smaller angle?
    """))

    cells.append(code("""
    k_gain = 0.8
    soft_factor = 1.0
    vehicle_v = 10.0
    cte_hand = 2.0
    heading_hand = 0.0
    denom = max(0.1, vehicle_v) + soft_factor
    cross = float(np.arctan2(k_gain * (-cte_hand), denom))
    steer_hand = heading_hand + cross

    path_x = np.linspace(0.0, 50.0, 100)
    path_y = np.zeros_like(path_x)
    path_psi = np.zeros_like(path_x)
    steer, cte, heading = StanleyController(k_gain=k_gain, soft_factor=soft_factor).compute_steering(
        front_x=5.0,
        front_y=2.0,
        vehicle_psi=0.0,
        vehicle_v=vehicle_v,
        path_x=path_x,
        path_y=path_y,
        path_psi=path_psi,
    )
    cross_slow = float(np.arctan2(k_gain * (-cte_hand), max(0.1, 2.0) + soft_factor))

    print(f"cross-track error {cte_hand:.3f} m")
    print(f"heading error {heading_hand:.3f} rad")
    print(f"denominator {denom:.1f}")
    print(f"cross-track angle {cross:.4f} rad ({np.degrees(cross):.3f} deg)")
    print(f"hand steer {steer_hand:.4f} rad")
    print(f"repo steer {steer:.4f} rad")
    print(f"repo cross-track {cte:.3f} m")
    print(f"repo heading error {heading:.3f} rad")
    print(f"same error at 2 m/s: {np.degrees(cross_slow):.3f} deg")
    assert abs(steer - steer_hand) < 1e-12
    assert abs(cte - cte_hand) < 1e-12
    """))

    cells.append(md("""
    Hand and repo agree: cross-track error `2.000` m, heading error `0.000` rad, denominator `11.0`, steer `-0.1444` rad (`-8.276` deg). That is a gentle right turn, far from the `-35°` clip.

    The same 2 m error at 2 m/s is `-28.072` deg. The speed in the denominator is what kept the 10 m/s correction small. The waypoint rule had no such term, so it went to the clip.
    """))

    cells.append(md("## 5. Closed loop on a curve"))
    cells.append(md("""
    Now send Stanley's steer, and a speed command, back into `step` on every tick. The path is the S-curve from `simulator.py`: `y = 4 sin(x / 15)` over 100 m. Gain `k = 0.75`, `soft_factor = 1`, control step `0.05` s. The car starts at `(0, 1.5)` with heading 0 and speed 8 m/s. The speed controller aims at 12 m/s with `kp = 1.5`, `ki = 0.05`, `kd = 0.1`.

    **Predict:** mean `|cross-track error|` over the run should come down from the 1.5 m offset. The first speed error is 4 m/s. The derivative term alone is `0.1 * 4 / 0.05 = 8` m/s², so the requested acceleration should hit the bicycle's 3.5 m/s² clip.
    """))

    cells.append(code("""
    car = KinematicBicycleModel(wheelbase=2.8, max_steer_deg=35.0)
    stanley = StanleyController(k_gain=0.75, soft_factor=1.0)
    pid = PIDLongitudinalController(kp=1.5, ki=0.05, kd=0.1)
    ref_x, ref_y, ref_psi = generate_curved_track(length_m=100.0)
    state = VehicleState(x=0.0, y=1.5, psi=0.0, v=8.0)
    dt = 0.05
    target_speed = 12.0

    err0 = target_speed - state.v
    p_term = pid.kp * err0
    i_term = pid.ki * err0 * dt
    d_term = pid.kd * err0 / dt
    print(f"P {p_term:.3f}  I {i_term:.3f}  D {d_term:.3f}  sum {p_term + i_term + d_term:.3f}")

    ctes = []
    ego_x, ego_y = [], []
    for step in range(180):
        front_x, front_y = car.front_axle_position(state)
        delta, cte, heading_err = stanley.compute_steering(
            front_x, front_y, state.psi, state.v, ref_x, ref_y, ref_psi
        )
        accel = pid.compute_acceleration(state.v, target_speed, dt)
        if step == 0:
            print(f"first accel request {accel:.3f} m/s^2")
            print(f"accel clip {car.max_accel:.1f} m/s^2")
        state = car.step(state, accel, delta, dt)
        ctes.append(abs(cte))
        ego_x.append(state.x)
        ego_y.append(state.y)
        if step in (0, 80, 160):
            print(f"t={step * dt:.2f} s  x={state.x:.3f}  y={state.y:.3f}  cte={cte:.3f} m")
        if state.x >= ref_x[-1] - 5.0:
            break

    ctes = np.array(ctes)
    print(f"steps {len(ctes)}")
    print(f"mean |CTE| {ctes.mean():.3f} m")
    print(f"max |CTE| {ctes.max():.3f} m")
    print(f"steady-state |CTE| {ctes[-40:].mean():.3f} m")

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.plot(ref_x, ref_y, "k--", label="planned path")
    ax.plot(ego_x, ego_y, label="Stanley + speed controller")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Closed loop on the S-curve")
    ax.legend()
    plt.show()
    """))

    cells.append(md("""
    The speed split is `P 6.000`, `I 0.010`, `D 8.000`, sum `14.010` m/s². The bicycle clips that request to `3.5` m/s².

    Cross-track error at the first command is `0.733` m (`t = 0.00` s, after the step the car is still at `y = 1.500`). At `t = 4.00` s the error is `0.091` m. At `t = 8.00` s it is `-0.039` m. Over `165` steps, mean `|CTE|` is `0.162` m, the max is `0.733` m, and the last 40 samples average `0.047` m. The plot shows the car leaving the 1.5 m offset and staying with the sine wave.
    """))

    cells.append(md("## 6. Try this: raise the gain"))
    cells.append(md("""
    Stanley's `k` scales the cross-track angle. The S-curve above used `k = 0.75` and a `0.05` s step.

    **Try this.** Same straight road, start at `y = 1.5` m and `8` m/s. Run gain `0.75` and gain `15`, holding each steer command for `0.20` s. Then run gain `15` again at `0.05` s.

    **Predict:** which of those three runs crosses the centerline?
    """))

    cells.append(code("""
    def stanley_straight(k_gain, dt, seconds=8.0):
        path_x = np.arange(0.0, 200.0, 0.5)
        path_y = np.zeros_like(path_x)
        path_psi = np.zeros_like(path_x)
        car = KinematicBicycleModel()
        stanley = StanleyController(k_gain=k_gain, soft_factor=1.0)
        state = VehicleState(x=0.0, y=1.5, psi=0.0, v=8.0)
        ys = []
        steps = int(round(seconds / dt))
        for _ in range(steps):
            front_x, front_y = car.front_axle_position(state)
            delta, cte, heading_err = stanley.compute_steering(
                front_x, front_y, state.psi, state.v, path_x, path_y, path_psi
            )
            state = car.step(state, throttle_accel=0.0, steer_delta=delta, dt=dt)
            ys.append(state.y)
        ys = np.array(ys)
        crossings = int(np.sum(ys[1:] * ys[:-1] < 0))
        print(f"k={k_gain:g} dt={dt:.2f} s  crossings={crossings}  lowest y={ys.min():.3f} m")
        return np.arange(1, len(ys) + 1) * dt, ys

    t_lo, y_lo = stanley_straight(0.75, 0.20)
    t_hi, y_hi = stanley_straight(15, 0.20)
    stanley_straight(15, 0.05)

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(t_lo, y_lo, label="k = 0.75, dt = 0.20 s")
    ax.plot(t_hi, y_hi, label="k = 15, dt = 0.20 s")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("y (m)")
    ax.set_title("Same step, two gains")
    ax.legend()
    plt.show()
    """))

    cells.append(md("""
    Gain `0.75` held for `0.20` s crosses the line `0` times. The lowest `y` is `0.006` m: it eases in and stays on the positive side.

    Gain `15` at that same `0.20` s step crosses `37` times and reaches `-0.271` m. The trace snakes around the line. The wheel is still commanded hard after the car has already passed the path, and the long step keeps that command in force.

    Gain `15` at the `0.05` s step from the curve demo crosses `0` times (lowest `y` is `0.000` m). The gain that weaves at `0.20` s still settles when the command is updated four times as often.
    """))

    cells.append(md("## 7. Exercises"))
    cells.append(md("""
    **Exercise — `yaw_change`.** Return the heading change of one Euler step, `(v / wheelbase) * tan(steer_rad) * dt`, in radians. Leave the `TODO` as it is to use the reference implementation.
    """))

    cells.append(code("""
    def yaw_change_student(v, steer_rad, wheelbase, dt):
        # TODO: (v / wheelbase) * tan(steer_rad) * dt
        raise NotImplementedError

    def yaw_change_reference(v, steer_rad, wheelbase, dt):
        return (v / wheelbase) * np.tan(steer_rad) * dt

    def get_yaw_change():
        try:
            yaw_change_student(1.0, 0.0, 2.8, 0.1)
        except NotImplementedError:
            print("Using reference yaw_change (TODO not implemented)")
            return yaw_change_reference
        return yaw_change_student

    yaw_change = get_yaw_change()
    before = VehicleState(x=0.0, y=1.0, psi=0.4, v=8.0)
    after = KinematicBicycleModel().step(before, throttle_accel=0.0, steer_delta=np.deg2rad(10.0), dt=0.5)
    expected_dyaw = after.psi - before.psi
    got_dyaw = yaw_change(8.0, np.deg2rad(10.0), 2.8, 0.5)
    print(f"yaw change {got_dyaw:.4f} rad")
    assert abs(got_dyaw - expected_dyaw) < 1e-12
    print("✅ yaw change matches KinematicBicycleModel.step")
    """))

    cells.append(md("""
    The check prints `✅`. Heading change is `0.2519` rad, the same increment as the one-step bicycle above (`0.6519 − 0.4000`). The reference is used because the `TODO` still raises. Replace the `TODO` and run the cell again if you want the check to call your function.
    """))

    cells.append(md("""
    <details><summary>Solution</summary>

    ```python
    def yaw_change(v, steer_rad, wheelbase, dt):
        return (v / wheelbase) * np.tan(steer_rad) * dt
    ```

    </details>
    """))

    cells.append(md("""
    **Exercise — `stanley_steer`.** Return the repo's steer command in radians: heading error plus `atan2(k * (-cross_track_error), max(0.1, speed) + soft_factor)`. Leave the `TODO` as it is to use the reference implementation.
    """))

    cells.append(code("""
    def stanley_steer_student(heading_error, cross_track_error, speed, k_gain, soft_factor):
        # TODO: heading_error + atan2(k_gain * (-cross_track_error), max(0.1, speed) + soft_factor)
        raise NotImplementedError

    def stanley_steer_reference(heading_error, cross_track_error, speed, k_gain, soft_factor):
        cross = np.arctan2(k_gain * (-cross_track_error), max(0.1, speed) + soft_factor)
        return float(heading_error + cross)

    def get_stanley_steer():
        try:
            stanley_steer_student(0.0, 0.0, 1.0, 0.8, 1.0)
        except NotImplementedError:
            print("Using reference stanley_steer (TODO not implemented)")
            return stanley_steer_reference
        return stanley_steer_student

    stanley_steer = get_stanley_steer()
    path_x = np.linspace(0.0, 50.0, 100)
    path_y = np.zeros_like(path_x)
    path_psi = np.zeros_like(path_x)
    expected_steer, expected_cte, expected_heading = StanleyController(0.8, 1.0).compute_steering(
        5.0, 2.0, 0.0, 10.0, path_x, path_y, path_psi
    )
    got_steer = stanley_steer(expected_heading, expected_cte, 10.0, 0.8, 1.0)
    print(f"stanley steer {got_steer:.4f} rad ({np.degrees(got_steer):.3f} deg)")
    assert abs(got_steer - expected_steer) < 1e-12
    print("✅ stanley steer matches StanleyController.compute_steering")
    """))

    cells.append(md("""
    The check prints `✅`. The steer is `-0.1444` rad (`-8.276` deg), the same command as the hand step: positive cross-track error, zero heading error, a right turn. The reference is used because the `TODO` still raises.
    """))

    cells.append(md("""
    <details><summary>Solution</summary>

    ```python
    def stanley_steer(heading_error, cross_track_error, speed, k_gain, soft_factor):
        cross = np.arctan2(k_gain * (-cross_track_error), max(0.1, speed) + soft_factor)
        return float(heading_error + cross)
    ```

    </details>
    """))

    cells.append(code("""
    try:
        import pytest  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "modules/07_control_sim/tests/test_control.py", "-q"],
        check=True,
    )
    """))

    cells.append(md("## 8. Recap"))
    cells.append(md("""
    - A path is coordinates. With steer held at 0 for 4.00 s at 8 m/s, the car is at `(32.000, 0.000)` while the path is at `y = 3.384`.
    - One Euler step with `v = 8`, `delta = 10°`, `L = 2.8`, `dt = 0.5`, start `(0, 1, 0.4)` lands at `(3.684, 2.558)` with heading `0.6519` rad. The heading moved by `0.2519` rad.
    - Aiming at the next waypoint from 2 m left commands `-75.96` deg, clips to `-35.00` deg, crosses the line 15 times, and reaches `y = -1.386` m. On a 15 m corner sampled every 30°, the same rule comes in to radius `14.540` m.
    - Stanley adds heading error and `atan2(k (-e), max(0.1, v) + soft_factor)`. For `e = 2` m at 10 m/s and `k = 0.8`, that is `-0.1444` rad (`-8.276` deg). At 2 m/s the same error is `-28.072` deg.
    - On the S-curve, mean `|CTE|` is `0.162` m, max is `0.733` m, and the last 40 samples average `0.047` m. The first speed request is `14.010` m/s² and the bicycle applies `3.5`.
    - Gain 15 held for 0.20 s crosses the line 37 times (lowest `y = -0.271` m). The same gain at 0.05 s crosses 0 times.
    - `pytest modules/07_control_sim/tests/test_control.py` reports `3 passed`: straight motion, the 35° steer clip, and a left offset producing a right steer.

    ### Go deeper
    - [Thrun et al., Stanley: The Robot that Won the DARPA Grand Challenge (Stanford AI Lab PDF)](http://robots.stanford.edu/papers/thrun.stanley05.pdf)
    - [MIT 6.003, Signals and Systems (feedback)](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/)
    """))

    nb = new_notebook(cells=cells)
    nb["metadata"] = {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }
    return nb


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "staging" / "self-driving-ai-course" / "notebooks" / "08_closed_loop_control_stanley.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    nbformat.write(nb, out)
    print(f"Wrote {out} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
