#!/usr/bin/env python3
"""Regenerate the Module 08 lesson notebook.

Writes ``notebooks/09_full_fsd_system_architecture.ipynb`` next to this course
staging tree. The notebook is the lesson: run it top to bottom. This script
does not execute it.

    python staging/self-driving-ai-course/scripts/build_m08_lesson_notebook.py
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
    # Module 08 — Wire the stack into one loop

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/notebooks/09_full_fsd_system_architecture.ipynb)

    Six runners, one baton. The camera runs first, then features, a map, tracks, a path, and the wheel. **End to end** here means the baton finishes that relay. It does not mean a magic net that glances at a photo and turns a steering wheel.

    Skip the tracker and the wheel twitches. Same planner, same Stanley controller, same bicycle. The only change is that each frame's boxes are handed over raw.

    This process is not a car, and it is not a safety case. The cameras are noise. The planner never reads the occupancy it just computed. A passing test means the function returned and the bicycle rolled forward.

    Each idea shows up three times: a picture, a handful of numbers, then a few lines of code. **Predict first**, then run the cell. The paragraph after the cell says what was weird, and what a driver should care about.
    """))

    cells.append(code("""
    import inspect
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.use("module://matplotlib_inline.backend_inline", force=True)
    get_ipython().run_line_magic("matplotlib", "inline")

    def find_repo(start: Path) -> Path:
        for p in [start, *start.parents]:
            if (p / "modules" / "08_capstone_fsd" / "pipeline.py").is_file():
                return p.resolve()
        return start.resolve()

    REPO = find_repo(Path.cwd())
    if not (REPO / "modules" / "08_capstone_fsd" / "pipeline.py").is_file():
        dest = Path.cwd() / "self-driving-ai-course"
        if not (dest / "modules" / "08_capstone_fsd" / "pipeline.py").is_file():
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/vvknyn/self-driving-ai-course.git", str(dest)],
                check=True,
            )
        REPO = dest.resolve()

    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "modules" / "08_capstone_fsd"))

    def _missing(mod_name: str) -> bool:
        try:
            __import__(mod_name)
            return False
        except Exception:
            return True

    if _missing("torch") or _missing("cv2") or _missing("scipy") or _missing("pytest"):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision",
             "--index-url", "https://download.pytorch.org/whl/cpu"],
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
            check=True,
        )

    import torch
    from pipeline import FullFSDPipeline
    from scenario_generator import DrivingScenario

    SEED = 0
    DT = 0.1
    N_STEPS = 15
    FLICKER_SIGMA = 0.8
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Repo:", REPO)
    print("torch", torch.__version__, "device cpu")
    print("dt", DT, "steps in the closed-loop cells", N_STEPS)
    """))

    cells.append(md("## 1. The wheel twitches"))
    cells.append(md("""
    Two cars, smooth paths, fifteen steps. The lead car starts near 22 m. The right-lane car starts near 10 m, at `y = -3.5` m. Every box gets independent noise, `sigma = 0.8` m, from `np.random.default_rng(0)`.

    One run throws the tracker out and hands the noisy boxes straight to the lattice. The lattice offsets are `-3.5, -1.8, 0.0, 1.8, 3.5` meters. The other run sends the **same** noise through `MultiObjectTracker` (`max_age` 3, `min_hits` 1, `distance_threshold` 4.0). Planner, Stanley, PID, and bicycle stay put.

    **Predict:** without a tracker the wheel kicks when the chosen plan jumps to another offset. With a tracker those kicks disappear. The true cars stay far away either way. A twitch is not a crash. The tracker may still invent an extra box. That is a different failure from the kick.
    """))
    cells.append(code("""
    def step_switched(pipe, camera_frames, gt, dt=DT, use_occ=True, use_tracker=True, noise=None):
        \"\"\"Same calls as FullFSDPipeline.step, with two switches and optional box noise.\"\"\"
        times = {}
        t_all = time.perf_counter()
        with torch.no_grad():
            B, N_cams, C, H, W = camera_frames.shape
            flat = camera_frames.view(B * N_cams, C, H, W)
            t0 = time.perf_counter()
            feats = pipe.hydranet.backbone(flat)["p3"]
            times["hydra"] = (time.perf_counter() - t0) * 1000.0
            fh, fw = feats.shape[-2:]
            feats = feats.view(B, N_cams, 128, fh, fw)
            t0 = time.perf_counter()
            bev = pipe.lss(feats, pipe.K_tensor, pipe.R_tensor, pipe.T_tensor)
            times["lss"] = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            if use_occ:
                _occ, _vel, pipe.hidden_state = pipe.occ_net(bev, pipe.hidden_state)
            times["occ"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        det = np.array([[o["x"], o["y"]] for o in gt], dtype=float)
        if noise is not None:
            det = det + noise
        if use_tracker:
            tracks = pipe.tracker.update(det)
            boxes = [{"x": float(p[0]), "y": float(p[1]), "radius": 1.6} for _, p, _ in tracks]
        else:
            boxes = [{"x": float(p[0]), "y": float(p[1]), "radius": 1.6} for p in det]
        times["track"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        ego = {"x0": pipe.state.x, "y0": pipe.state.y, "v0": pipe.state.v, "a0": 0.0}
        candidates = pipe.planner.generate_candidate_trajectories(
            ego, lane_centerline_y=0.0,
            lateral_offsets=[-3.5, -1.8, 0.0, 1.8, 3.5],
            planning_horizon=2.5, dt=dt,
        )
        optimal = pipe.evaluator.select_optimal_trajectory(
            candidates, boxes, target_lane_y=0.0, target_speed=14.0
        )
        times["plan"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        xf, yf = pipe.vehicle.front_axle_position(pipe.state)
        path_x = np.asarray(optimal.x, dtype=float)
        path_y = np.asarray(optimal.y, dtype=float)
        path_psi = np.arctan2(np.gradient(path_y), np.gradient(path_x))
        steer, cte, _heading = pipe.stanley.compute_steering(
            xf, yf, pipe.state.psi, pipe.state.v, path_x, path_y, path_psi
        )
        accel = pipe.pid_speed.compute_acceleration(pipe.state.v, target_v=float(optimal.v[1]), dt=dt)
        pipe.state = pipe.vehicle.step(pipe.state, accel, steer, dt)
        times["control"] = (time.perf_counter() - t0) * 1000.0
        times["total"] = (time.perf_counter() - t_all) * 1000.0
        row = {
            "ego_x": float(pipe.state.x),
            "ego_y": float(pipe.state.y),
            "ego_v": float(pipe.state.v),
            "steer": float(steer),
            "cte": float(cte),
            "end_y": float(path_y[-1]),
            "cost": float(optimal.total_cost),
            "n_boxes": len(boxes),
            "min_true": float(min(np.hypot(pipe.state.x - o["x"], pipe.state.y - o["y"]) for o in gt)),
        }
        return row, times

    def rollout(mode, n=N_STEPS, sigma=FLICKER_SIGMA, seed=SEED):
        rng = np.random.default_rng(seed)
        pipe = FullFSDPipeline(device="cpu")
        sc = DrivingScenario(num_frames=n, dt=DT)
        rows, times = [], []
        for _ in range(n):
            cam_i, gt = sc.step()
            noise = None
            use_occ = mode != "no_occ"
            use_tracker = mode not in ("no_tracker", "flicker_raw")
            if mode in ("flicker_raw", "flicker_tracked"):
                noise = rng.normal(0.0, sigma, size=(len(gt), 2))
            row, timing = step_switched(
                pipe, cam_i, gt, use_occ=use_occ, use_tracker=use_tracker, noise=noise
            )
            rows.append(row)
            times.append(timing)
        ends = [round(r["end_y"], 2) for r in rows]
        jumps = sum(a != b for a, b in zip(ends, ends[1:]))
        return {"rows": rows, "times": times, "ends": ends, "jumps": jumps, "lead_v": sc.lead_v}

    def show_run(name, run):
        rows = run["rows"]
        print(name)
        print("plan end y:", run["ends"])
        print("offset jumps:", run["jumps"])
        print("boxes per step:", [r["n_boxes"] for r in rows])
        print("steer deg:", [f"{np.degrees(r['steer']):.2f}" for r in rows])
        print(f"final ego_x: {rows[-1]['ego_x']:.4f}")
        print(f"final ego_y: {rows[-1]['ego_y']:.4f}")
        print(f"final ego_v: {rows[-1]['ego_v']:.4f}")
        print(f"mean |cte|: {float(np.mean([abs(r['cte']) for r in rows])):.4f}")
        print(f"min true clearance: {min(r['min_true'] for r in rows):.4f}")
        print(f"step0 cost: {rows[0]['cost']:.4f}")
        print(f"lead v after loop: {run['lead_v']:.1f}")

    raw_run = rollout("flicker_raw")
    show_run("skip tracker, flickering boxes", raw_run)
    """))
    cells.append(md("""
    Without a tracker the plan end `y` is

    `3.5, 3.5, 3.5, 3.5, 3.5, -1.8, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, -1.8, 3.5, 3.5`

    and the wheel, in degrees, is

    `1.23, 0.61, 0.29, 0.15, 0.07, -1.84, 0.94, 0.47, 0.23, 0.11, 0.05, 0.02, -1.87, 0.93, 0.46`

    Four offset jumps. The two dips to `-1.8` m are the two kicks, `-1.84` deg and `-1.87` deg. Final ego `x` is `17.9906` m. Mean `|cte|` is `0.0226` m. Minimum distance to the **true** cars is `10.9731` m. Step-0 cost is `207.2194`. Both cars are present on every step. Lead speed after these 15 steps is `11.4` m/s: four brake ticks of `0.4` after step 10, down from `13`.

    The weird part is already here, before the filter. The bicycle was never within `10.9731` m of either car, and the lane error is a couple of centimeters, and the wheel still jerks. A driver should care about the jerk. A score that only watches clearance will call this fine.
    """))
    cells.append(code("""
    tracked_run = rollout("flicker_tracked")
    show_run("same flicker, through the tracker", tracked_run)
    print("offset jumps fell:", tracked_run["jumps"], "<", raw_run["jumps"])
    """))
    cells.append(md("""
    Same noise, through the tracker. Offset jumps fall from `4` to `0`. Every plan end `y` is `3.5`. The wheel, in degrees, is

    `1.23, 0.61, 0.30, 0.14, 0.07, 0.03, 0.01, -0.00, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01`

    The `-1.84` and `-1.87` kicks are gone. Final ego `x` is `17.9814`, `y` is `0.2789`, `v` is `11.9776`. Mean `|cte|` is `0.0234` m. Minimum true clearance is again `10.9731` m.

    Boxes per step are `2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2, 3, 3, 3`. A third box shows up on six of the fifteen steps. The chosen offset still does not jump. Step-0 cost is still `207.2194`: the first track is born on the noisy measurement, so the filter has not helped yet on that step. A driver should care that "the tracker is on" is not the same sentence as "the tracker counted two cars."
    """))
    cells.append(code("""
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    raw_deg = [np.degrees(r["steer"]) for r in raw_run["rows"]]
    tracked_deg = [np.degrees(r["steer"]) for r in tracked_run["rows"]]
    axes[0].plot(raw_deg, marker="o", label="no tracker")
    axes[0].plot(tracked_deg, marker="o", label="tracker")
    axes[0].set_ylabel("steer (deg)")
    axes[0].set_title(f"The wheel. Flicker sigma {FLICKER_SIGMA} m, seed {SEED}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(raw_run["ends"], marker="o", label="no tracker")
    axes[1].plot(tracked_run["ends"], marker="o", label="tracker")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("plan end y (m)")
    axes[1].set_title("Why the wheel moved")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
    """))
    cells.append(md("""
    Top is the wheel. Bottom is the plan that the wheel was chasing. Without a tracker the wheel dives to about `-1.8` deg twice, and the plan end `y` dives to `-1.8` m on those same two steps. With a tracker the wheel settles near zero and the plan stays on `3.5` m. The picture and the lists are the same fact: skipping a runner makes the last runner twitch.
    """))

    cells.append(md("## 2. The relay, not a magic trick"))
    cells.append(md("""
    You have seen the twitch. Here is the race that produced it. One function, `FullFSDPipeline.step`, takes a camera tensor and a list of obstacle dicts and returns a steer command and a new ego state.

    The cameras run through the networks. The planner scores boxes from `MultiObjectTracker.update` on the scenario's ground-truth positions. Occupancy is computed and then dropped on the floor.

    ```
    camera_frames (1, 3, 3, 128, 256)
            |
            v
    HydraNet.backbone(flat)["p3"]                 (1, 3, 128, 16, 32)
            |
            v
    LiftSplatShoot(feats, K, R, T)                bev (1, 32, 40, 60)
            |
            v
    OccupancyNetwork(bev, hidden_state)           occ, vel, hidden
            |                                     (computed, then dropped)
            X
    DrivingScenario obstacles  -- positions only -->
            |
            v
    MultiObjectTracker.update(xy)
            |
            v
    LatticePlanner.generate_candidate_trajectories
    TrajectoryCostEvaluator.select_optimal_trajectory
            |
            v
    StanleyController.compute_steering
    PIDLongitudinalController.compute_acceleration
            |
            v
    KinematicBicycleModel.step                    telemetry dict
    ```

    `VectorLane` is built in `__init__`. `step` still passes `lane_centerline_y=0.0` into the planner.

    **Predict:** `step`'s return dict will list ego state, steer, throttle, cross-track error, track count, waypoints, and cost. It will not list `occ_probs` or `bev`.
    """))
    cells.append(code("""
    pipe = FullFSDPipeline(device="cpu")
    step_src = inspect.getsource(FullFSDPipeline.step)
    return_src = step_src.split("return", 1)[1]
    print("cameras", pipe.cam_keys)
    print("K", tuple(pipe.K_tensor.shape), "R", tuple(pipe.R_tensor.shape), "T", tuple(pipe.T_tensor.shape))
    print("hydranet", type(pipe.hydranet).__name__)
    print("lss", type(pipe.lss).__name__)
    print("occ_net", type(pipe.occ_net).__name__)
    print("tracker", type(pipe.tracker).__name__,
          "max_age", pipe.tracker.max_age,
          "min_hits", pipe.tracker.min_hits,
          "distance_threshold", pipe.tracker.distance_threshold)
    print("planner", type(pipe.planner).__name__, "target_speed", pipe.planner.target_speed)
    print("evaluator", type(pipe.evaluator).__name__)
    print("stanley k", pipe.stanley.k, "pid kp", pipe.pid_speed.kp, "wheelbase L", pipe.vehicle.L)
    print("initial v", pipe.state.v)
    print("vector_lane object:", type(pipe.vector_lane).__name__)
    print("step() mentions vector_lane:", "vector_lane" in step_src)
    print("step() passes lane_centerline_y=0.0:", "lane_centerline_y=0.0" in step_src)
    print("return mentions occ_probs:", "occ_probs" in return_src)
    print("return mentions bev:", "bev" in return_src)
    """))
    cells.append(md("""
    Three cameras: front, left, right. `K` and `R` are `(1, 3, 3, 3)`. `T` is `(1, 3, 3, 1)`. The tracker forgets a box after `max_age` 3, publishes a track on the first hit, and associates inside `4.0` m. Initial speed is `12.0` m/s. Stanley's `k` is `0.8`, the speed PID's `kp` is `1.5`, the wheelbase is `2.8` m, and the lattice's target speed is `14.0` m/s.

    The weird part is what gets built and then ignored. A `VectorLane` exists. `step` never mentions it, and the planner is handed `lane_centerline_y=0.0` anyway. The return block names neither `occ_probs` nor `bev`. The map leg draws a picture the wheel never sees. A driver should care: an occupancy grid can update every frame and still not move the steering.
    """))

    cells.append(md("## 3. One frame is not a photograph"))
    cells.append(md("""
    `DrivingScenario.step` returns a noise image and two cars. The lead car starts near `x = 22` m. The right-lane car starts near `x = 10` m at `y = -3.5` m. The image is `torch.randn`, scaled by `0.1`, not a photo.

    **Predict:** one call of `FullFSDPipeline.step` moves the ego about `v * dt = 12 * 0.1 = 1.2` m forward, and `active_tracks_count` is 2. The network tensors have shapes, and those tensors are absent from the telemetry keys.
    """))
    cells.append(code("""
    def trace_perception(pipe, camera_frames):
        with torch.no_grad():
            B, N_cams, C, H, W = camera_frames.shape
            flat = camera_frames.view(B * N_cams, C, H, W)
            feats = pipe.hydranet.backbone(flat)["p3"]
            fh, fw = feats.shape[-2:]
            feats = feats.view(B, N_cams, 128, fh, fw)
            bev = pipe.lss(feats, pipe.K_tensor, pipe.R_tensor, pipe.T_tensor)
            occ, vel, hidden = pipe.occ_net(bev, None)
        print("camera_frames", tuple(camera_frames.shape))
        print("flat cams", tuple(flat.shape))
        print("p3", (B, N_cams, 128, fh, fw))
        print("bev", tuple(bev.shape))
        print("occ", tuple(occ.shape), "vel", tuple(vel.shape), "hidden", tuple(hidden.shape))
        return occ

    fresh = FullFSDPipeline(device="cpu")
    scenario = DrivingScenario(num_frames=5, dt=DT)
    cam, obstacles = scenario.step()
    print("obstacle names:", [o["name"] for o in obstacles])
    for o in obstacles:
        print(f"  {o['name']}: x={o['x']:.1f} y={o['y']:.1f} v={o['v']:.1f}")
    trace_perception(fresh, cam)

    runner = FullFSDPipeline(device="cpu")
    telemetry = runner.step(cam, obstacles, dt=DT)
    print("telemetry keys:", list(telemetry))
    for key, value in telemetry.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, list):
            print(f"  {key}: list len {len(value)}")
        else:
            print(f"  {key}: {value}")
    print("occ or bev in keys:", any(k in telemetry for k in ("occ", "occ_probs", "bev", "bev_features")))
    zeros = torch.zeros_like(cam)
    tel_zeros = FullFSDPipeline(device="cpu").step(zeros, obstacles, dt=DT)
    print("steer abs diff, noise image vs zeros:", abs(telemetry["steer_angle"] - tel_zeros["steer_angle"]))
    """))
    cells.append(md("""
    The camera tensor is `(1, 3, 3, 128, 256)`. Flattened for the backbone it is `(3, 3, 128, 256)`. `p3` is `(1, 3, 128, 16, 32)`. The bird's-eye map is `(1, 32, 40, 60)`. Occupancy is `(1, 1, 40, 60, 8)`, velocity is `(1, 3, 40, 60, 8)`, and the hidden state is `(1, 32, 40, 60)`. None of those tensors are keys in the telemetry.

    Lead Car sits at `x=23.3` (`22 + 13 * 0.1`). Right Lane Car sits at `x=11.6`, `y=-3.5`. After one step `ego_x` is `1.2000` m and `ego_v` is `11.9977` m/s. Two tracks are alive. Steer is `0.0215` rad, cross-track error is `-0.0239` m, throttle accel is `-0.0234` m/s², and the chosen path costs `320.3105`. Both waypoint lists have length `26`, a 2.5 s horizon at `dt = 0.1`.

    Steer does not change if the noise image is replaced by zeros: the absolute difference is `0.0`. The networks ran. The bicycle moved because of the tracker, the lattice, Stanley, and the PID. A driver should care that the first runner can fall down and the wheel still turns.
    """))

    cells.append(md("## 4. A clean scene hides the missing runner"))
    cells.append(md("""
    `evaluate.py` always calls `step`. It has no switch for occupancy or tracking. The three runs below are that same call sequence, with the switches, and with **no** added noise.

    **Predict:** occupancy off does not move the wheel. Tracker off, on these smooth ground-truth centers, also matches. Section 1 is where the tracker earned its keep. A test that only drives the clean scene will not see the twitch.
    """))
    cells.append(code("""
    # The copy agrees with FullFSDPipeline.step when both switches are on.
    cam_check, gt_check = DrivingScenario(num_frames=1, dt=DT).step()
    real = FullFSDPipeline(device="cpu").step(cam_check, gt_check, dt=DT)
    copied, _timing = step_switched(
        FullFSDPipeline(device="cpu"), cam_check, gt_check, use_occ=True, use_tracker=True
    )
    print("steer abs diff vs step:", abs(real["steer_angle"] - copied["steer"]))
    print("ego_x abs diff vs step:", abs(real["ego_x"] - copied["ego_x"]))
    print("cost abs diff vs step:", abs(real["selected_traj_cost"] - copied["cost"]))

    full_run = rollout("full")
    no_occ_run = rollout("no_occ")
    no_tracker_run = rollout("no_tracker")
    show_run("full", full_run)
    show_run("no occupancy", no_occ_run)
    show_run("no tracker, clean GT", no_tracker_run)

    def steers(run):
        return np.array([r["steer"] for r in run["rows"]])

    print("no occupancy steer matches full:", bool(np.allclose(steers(no_occ_run), steers(full_run))))
    print("no tracker steer matches full:", bool(np.allclose(steers(no_tracker_run), steers(full_run))))
    print("flicker+tracker final ego_x matches full:",
          abs(tracked_run["rows"][-1]["ego_x"] - full_run["rows"][-1]["ego_x"]) < 1e-9)
    """))
    cells.append(md("""
    On one clean step, `step_switched` agrees with `FullFSDPipeline.step`: steer differs by `0.0`, ego `x` differs by `0.0`, cost differs by `0.0`.

    All three clean runs tell the same story. Offset jumps stay `0`. Plan end `y` stays `3.5`. Final ego `x` is `17.9814`, `y` is `0.2789`, `v` is `11.9776`. Mean `|cte|` is `0.0234` m. Minimum true clearance is `10.9731` m. Step-0 cost is `320.3105`. Occupancy off steers like the full loop. Tracker off, on these perfect centers, also steers like the full loop.

    That is the weird part. Drop the map, the wheel does not notice. Drop the tracker, the wheel does not notice, **while the boxes are the true centers**. The twitch in section 1 cost four offset jumps and never shows up here. The filtered flicker run ends at the same ego `x` as this clean run, `17.9814`. The unfiltered flicker run ended at `17.9906`. A driver should care that the bug hides inside the scenario you use to demo the stack.
    """))

    cells.append(md("## 5. How long one lap takes"))
    cells.append(md("""
    `dt` is `0.1` s, so the loop is written as a 10 Hz tick. One `step` is one control update, and the budget for that tick is 100 ms. `evaluate.py` times a whole `step` with `time.time()` over 25 steps. Here each stage of the 15-step full run is timed with `perf_counter`.

    **Predict:** the backbone is the slow runner, the controller is the cheap one, and the total is well under 100 ms on this CPU because the networks are small and untrained. A Hertz figure from that total is this machine's wall clock. It is not a scheduler on a car.
    """))
    cells.append(code("""
    stage_names = ["hydra", "lss", "occ", "track", "plan", "control", "total"]
    means = {
        name: float(np.mean([t[name] for t in full_run["times"]]))
        for name in stage_names
    }
    print(f"steps timed: {len(full_run['times'])}")
    for name in stage_names:
        print(f"  {name:8} {means[name]:.2f} ms")
    slowest = max((n for n in stage_names if n != "total"), key=lambda n: means[n])
    print("slowest stage:", slowest)
    print(f"design tick: {1000 * DT:.1f} ms  ({1 / DT:.0f} Hz)")
    print(f"total under design tick: {means['total'] < 1000 * DT}")
    print(f"implied Hz from mean total: {1000.0 / means['total']:.1f}")
    no_occ_total = float(np.mean([t["total"] for t in no_occ_run["times"]]))
    print(f"no occupancy mean total: {no_occ_total:.2f} ms")
    print(f"occupancy mean: {means['occ']:.2f} ms")
    """))
    cells.append(md("""
    Fifteen steps. Mean milliseconds: hydra `7.75`, LSS `2.79`, occupancy `1.60`, tracker `0.23`, planner `2.38`, control `0.10`, total `14.90`. The slow runner is the backbone. The design tick is `100.0` ms, which is 10 Hz, and the total sits under it. `1000 / 14.90` is `67.1` Hz. That quotient is this CPU timing a tiny untrained forward pass. It is not a scheduler on a car.

The no-occupancy run's mean total is `11.46` ms, next to an occupancy stage of `1.60` ms. Those two totals do not differ by the occupancy line alone, because the other stages also moved between runs. Section 4 already showed the wheel does not move when that stage is removed. A driver should care about the wheel. A millisecond that wanders is not the twitch.
    """))

    cells.append(md("## 6. A green test is not a safety case"))
    cells.append(md("""
    The integration test checks that `step` returns its keys, that steer, accel, and cross-track error are finite, that `ego_v > 0`, and that `ego_x` grows over three steps.

    **Predict:** `pytest` on `modules/08_capstone_fsd/tests/test_integration.py` passes. It does not report a clearance or an offset jump.
    """))
    cells.append(code("""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "modules/08_capstone_fsd", "-q", "--tb=no"],
        cwd=REPO,
    )
    print("pytest exit:", proc.returncode)
    eval_lines = (REPO / "modules" / "08_capstone_fsd" / "evaluate.py").read_text().splitlines()
    for line in eval_lines:
        stripped = line.strip()
        if "range(25)" in stripped or "Collisions" in stripped or "num_frames=25" in stripped:
            print(stripped)
    """))
    cells.append(md("""
    pytest exit is `0`. The quiet report is `2 passed in 1.38s`. Those two tests are the wiring check: the function returns, the numbers are finite, speed stays positive, and ego `x` grows. They do not score a clearance or an offset jump.

    `evaluate.py` builds `DrivingScenario(num_frames=25, dt=0.1)` and loops `for step in range(25)`. The collisions line in that file is the literal string `Collisions:                    0 (ZERO FAILURES)`. It does not compare `min_d` to a radius. The 15-step runs measured a minimum true clearance of `10.9731` m, a large gap in this particular scene. That sentence would still say zero collisions if the gap were small. A driver should care which of those two facts is a measurement.
    """))
    cells.append(md("""
    - The relay is camera, features, map, tracks, path, wheel. Skipping the tracker, with `0.8` m box noise and seed 0, kicks the wheel to `-1.84` deg and `-1.87` deg. Those are the two steps whose plan end `y` flips from `3.5` m to `-1.8` m. The same noise through the tracker has `0` offset jumps. A third box still appears on six steps.
    - Minimum distance to the true cars stays `10.9731` m while the wheel twitches. Mean `|cte|` stays near `0.02` m. The bicycle was not close to anyone.
    - One noise frame is `(1, 3, 3, 128, 256)`. Occupancy is `(1, 1, 40, 60, 8)` and then dropped. Replacing the noise image with zeros changes steer by `0.0`. `ego_x` after one step is `1.2000` m.
    - On clean ground-truth centers, full, no-occupancy, and no-tracker all steer the same way. Final ego `x` is `17.9814` m. The twitch hides until the boxes flicker.
    - Mean full-step time on this CPU is `14.90` ms, under the `100` ms design tick. Hydra is the slow stage at `7.75` ms. The `67.1` Hz figure is `1000 / 14.90` on this run only.
    - `pytest` on `modules/08_capstone_fsd` reports `2 passed in 1.38s`. That means the graph runs and the ego moves forward. It does not mean anyone checked the wheel for a twitch.

    This toy is a Python process and a kinematic bicycle. It is not a car, and it is not a safety case. The cameras are Gaussian noise. The planner does not see occupancy. The tests do not score a collision. Nothing here should be pointed at a motor.

    ### Go deeper
    - [Yihan Hu et al., Planning-oriented Autonomous Driving](https://arxiv.org/abs/2212.10156). The abstract says separate heads "might suffer from accumulative errors or deficient task coordination," and that the framework should be optimized for planning. It states no scalar.
    - [Jiang-Tian Zhai et al., Rethinking the Open-Loop Evaluation of End-to-End Autonomous Driving in nuScenes](https://arxiv.org/abs/2305.10430). The abstract says a method with no camera images reduced average L2 error "by about 20%" on nuScenes, and that perception-based methods still had an advantage on collision rate. That 20% is their sentence, not a number this notebook measured.
    """))

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nb.metadata["colab"] = {"provenance": [], "gpuType": ""}
    nb.metadata["accelerator"] = "CPU"
    return nb


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "notebooks" / "09_full_fsd_system_architecture.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    nbformat.write(nb, out)
    print(f"Wrote {out} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
