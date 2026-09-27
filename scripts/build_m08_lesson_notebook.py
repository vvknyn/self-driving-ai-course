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

    Modules 01–07 each built one piece. This notebook calls those pieces in the order `FullFSDPipeline.step` actually uses, on one synthetic scenario from `DrivingScenario`.

    Each section says what to expect, then runs. **Predict first**, then execute the cell.
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

    cells.append(md("## 1. What end to end means here"))
    cells.append(md("""
    **End to end** in this repo means one function, `FullFSDPipeline.step`, takes a camera tensor and a list of obstacle dicts and returns a telemetry dict with a steer command and an updated ego state.

    It does **not** mean the cameras are the only input. Read the call order in `modules/08_capstone_fsd/pipeline.py`. The cameras go through the networks. The planner scores boxes that come from `MultiObjectTracker.update` on the scenario's ground-truth positions.

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
    The object print matches the diagram. Three camera names, `K` / `R` / `T` each shaped `(1, 3, 3, 3)` except `T`, which is `(1, 3, 3, 1)`. Tracker settings are `max_age 3`, `min_hits 1`, `distance_threshold 4.0`. Initial speed is `12.0` m/s. `vector_lane object: VectorLane`, and `step() mentions vector_lane: False`. The planner call in `step` does contain `lane_centerline_y=0.0`. The return block mentions neither `occ_probs` nor `bev`.
    """))

    cells.append(md("## 2. Run one scenario"))
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
    """))
    cells.append(md("""
    Cameras are `(1, 3, 3, 128, 256)`. Flattened for the backbone they are `(3, 3, 128, 256)`. `p3` is `(1, 3, 128, 16, 32)`. BEV is `(1, 32, 40, 60)`. Occupancy is `(1, 1, 40, 60, 8)`, velocity is `(1, 3, 40, 60, 8)`, and the hidden state is `(1, 32, 40, 60)`.

    The two obstacles are `Lead Car` at `x=23.3` (`22 + 13 * 0.1`) and `Right Lane Car` at `x=11.6`, `y=-3.5`. Telemetry keys are `ego_x`, `ego_y`, `ego_v`, `ego_psi`, `steer_angle`, `throttle_accel`, `cross_track_error`, `active_tracks_count`, `planned_waypoints_x`, `planned_waypoints_y`, `selected_traj_cost`. `occ or bev in keys: False`.

    After this one step, `ego_x` is `1.2000` m and `ego_v` is `11.9977` m/s. `active_tracks_count` is `2`. Steer is `0.0215` rad, cross-track error is `-0.0239` m, throttle accel is `-0.0234` m/s², and `selected_traj_cost` is `320.3105`. Both waypoint lists have length `26` (horizon 2.5 s at `dt = 0.1`). The networks ran. The numbers that moved the bicycle are the tracker, the lattice, Stanley, and the PID.
    """))

    cells.append(md("## 3. First try: skip tracking"))
    cells.append(md("""
    The scenario positions move smoothly. This attempt keeps the same planner, Stanley controller, PID, and bicycle, and feeds each frame's boxes straight in. Every box is the ground-truth center plus independent noise, `sigma = 0.8` m, from `np.random.default_rng(0)`. The tracker is left out: there is no call to `MultiObjectTracker.update`.

    The failure metric is **offset jumps**: how many times the chosen plan's last `y` changes from one step to the next. The lattice offsets inside `step` are `-3.5, -1.8, 0.0, 1.8, 3.5` meters.

    **Predict:** with `0.8` m of independent noise and no filter, that end `y` will change on some steps, and the steer command will kick when it does. Fifteen steps is enough to see it. `evaluate.py` loops 25. One step is about a hundredth of a second, so 25 would still finish quickly. Fifteen keeps the printed lists on one screen, and it still includes the lead car's brake, which starts once `current_step > 10`.
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
    Skipping the tracker fails this metric. `offset jumps: 4`. Plan end `y` is

    `3.5, 3.5, 3.5, 3.5, 3.5, -1.8, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, -1.8, 3.5, 3.5`

    The steer list in degrees is

    `1.23, 0.61, 0.29, 0.15, 0.07, -1.84, 0.94, 0.47, 0.23, 0.11, 0.05, 0.02, -1.87, 0.93, 0.46`

    On the step that flips to `-1.8` m, steer goes to `-1.84` deg, then later to `-1.87` deg. Step-0 cost is `207.2194`, not the clean-run cost `320.3105`, because the boxes are already off the true centers. Final ego `x` is `17.9906` m. Mean `|cte|` is `0.0226` m. Minimum distance to the **true** cars is still `10.9731` m, so this scenario does not turn the jitter into a collision. The failure you can see is the plan thrashing.

    `boxes per step` is all `2`. Every step still has both cars. The positions are what move the plan. Lead speed at the end of the 15 steps is `11.4` m/s: four brake ticks of `0.4` after step 10, down from `13`.
    """))
    cells.append(md("""
    Same noise sequence, now through `MultiObjectTracker` with the pipeline's settings (`max_age=3`, `min_hits=1`, `distance_threshold=4.0`).

    **Predict:** offset jumps drop, and the late steer commands stop kicking to about `-1.8` deg. The tracker can still hold an extra box for a few steps. That is a different number from offset jumps.
    """))
    cells.append(code("""
    tracked_run = rollout("flicker_tracked")
    show_run("same flicker, through the tracker", tracked_run)
    print("offset jumps fell:", tracked_run["jumps"], "<", raw_run["jumps"])
    """))
    cells.append(md("""
    The tracker holds the plan. `offset jumps: 0`. Every plan end `y` is `3.5`. Steer in degrees is

    `1.23, 0.61, 0.30, 0.14, 0.07, 0.03, 0.01, -0.00, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01`

    The `-1.84` and `-1.87` kicks are gone. The cell prints `offset jumps fell: 0 < 4`. Final ego `x` is `17.9814`, `y` is `0.2789`, `v` is `11.9776`. Mean `|cte|` is `0.0234` m. Minimum true clearance is again `10.9731` m.

    `boxes per step` is `2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2, 3, 3, 3`. The filter did not keep a perfect count of two cars. A third box shows up on six of the fifteen steps. The chosen offset still does not jump. Step-0 cost is still `207.2194`: the first track is born on the noisy measurement, so the filter has not helped yet on that step.
    """))
    cells.append(code("""
    plt.figure(figsize=(8, 3.2))
    plt.plot(raw_run["ends"], marker="o", label="no tracker")
    plt.plot(tracked_run["ends"], marker="o", label="tracker")
    plt.xlabel("step")
    plt.ylabel("chosen plan end y (m)")
    plt.title(f"Flicker sigma {FLICKER_SIGMA} m, seed {SEED}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    """))
    cells.append(md("""
    The plot is the same two lists. The no-tracker line dips to `-1.8` twice. The tracker line stays on `3.5`.
    """))

    cells.append(md("## 4. Full pipeline vs ablated"))
    cells.append(md("""
    `evaluate.py` always constructs one `FullFSDPipeline` and calls `step`. It has no switch for occupancy or tracking. The three runs below use `step_switched`, which is the same call sequence, with `use_occ` and `use_tracker`. On the clean scenario there is **no** added noise.

    **Predict:** turning occupancy off will not change steer or ego `x`. The occupancy tensor is computed and then dropped. Turning the tracker off on these smooth ground-truth points will also match, because the tracker is handed the same centers the raw list already has. The flicker run is where the tracker earned its keep.
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
    `step_switched` matches `FullFSDPipeline.step` on the check step: steer abs diff `0.0`, ego `x` abs diff `0.0`, cost abs diff `0.0`.

    All three clean runs print the same summary. `offset jumps: 0`. Plan end `y` stays `3.5`. Final ego `x` is `17.9814`, `y` is `0.2789`, `v` is `11.9776`. Mean `|cte|` is `0.0234` m. Minimum true clearance is `10.9731` m. Step-0 cost is `320.3105`. `no occupancy steer matches full: True`. `no tracker steer matches full: True`.

    So on this scenario, ablating occupancy changes nothing the bicycle does. Ablating the tracker also changes nothing **while the boxes are the ground-truth centers**. That is the honest comparison. The previous section is the one where dropping the tracker costs four offset jumps.

    `flicker+tracker final ego_x matches full: True` (`17.9814`). The filtered flicker run ended on the same ego `x` as the clean full run. The unfiltered flicker run ended at `17.9906`.
    """))

    cells.append(md("## 5. Latency budget"))
    cells.append(md("""
    `dt` is `0.1` s, so the loop is written as a 10 Hz tick: one `step` is one control update, and the time budget for that tick is 100 ms. `evaluate.py` times the whole `step` with `time.time()` over 25 steps. Here each stage of the 15-step full run is timed with `perf_counter`.

    **Predict:** the backbone is the slowest stage, the controller is the cheapest, and the total is well under 100 ms on this CPU because the networks are small and untrained. A printed Hertz value from that total is wall-clock on this machine. It is not a scheduler on a car.
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
    Fifteen full steps, mean milliseconds: hydra `5.44`, LSS `2.38`, occupancy `1.32`, tracker `0.16`, planner `1.94`, control `0.08`, total `11.37`. Slowest stage is `hydra`. The design tick is `100.0` ms (10 Hz). `total under design tick: True`. Dividing 1000 by `11.37` prints `87.9` Hz. That quotient is this CPU timing a tiny forward pass. It is not a claim that the stack runs at 87.9 Hz in a vehicle, and it is not the fixed "80 Hz" label in the browser dashboard.

    The no-occupancy run's mean total is `11.02` ms, next to an occupancy stage of `1.32` ms. Those two totals differ by less than the occupancy line, because the other stages also moved by about a millisecond between runs. The occupancy print is the direct measurement of that stage. Section 4 already showed the steer command does not move when that stage is removed.
    """))

    cells.append(md("## 6. Exercises"))
    cells.append(md("""
    **Exercise — `min_clearance`.** Shortest distance from `(ego_x, ego_y)` to any obstacle `x`, `y`. This is the distance `evaluate.py` appends each step. Return `99.0` when the list is empty. Leave the `TODO` in place to use the reference.
    """))
    cells.append(code("""
    def min_clearance_student(ego_x, ego_y, obstacles):
        # TODO: min hypot to each obstacle, or 99.0 if obstacles is empty
        raise NotImplementedError

    def min_clearance_reference(ego_x, ego_y, obstacles):
        if not obstacles:
            return 99.0
        return float(min(np.hypot(ego_x - o["x"], ego_y - o["y"]) for o in obstacles))

    def get_min_clearance():
        try:
            min_clearance_student(0.0, 0.0, [])
        except NotImplementedError:
            print("Using reference min_clearance (TODO not implemented)")
            return min_clearance_reference
        print("Using your min_clearance")
        return min_clearance_student

    clearance_fn = get_min_clearance()
    toy_obs = [{"x": 10.0, "y": 0.0}, {"x": 0.0, "y": 5.0}]
    got = clearance_fn(0.0, 0.0, toy_obs)
    got_empty = clearance_fn(1.0, 1.0, [])
    assert abs(got - 5.0) < 1e-9
    assert got_empty == 99.0
    scene_clear = clearance_fn(
        full_run["rows"][-1]["ego_x"],
        full_run["rows"][-1]["ego_y"],
        [{"x": 23.3, "y": 0.0}, {"x": 11.6, "y": -3.5}],
    )
    print("toy min clearance:", f"{got:.4f}")
    print("empty list:", got_empty)
    print("distance from final ego to the first-step obstacle snapshot:", f"{scene_clear:.4f}")
    print("✅ correct: min_clearance =", round(got, 4))
    """))
    cells.append(md("""
    The check prints `✅ correct: min_clearance = 5.0`. The line above it is `toy min clearance: 5.0000`. The toy obstacles are 10 m ahead and 5 m to the side, so the minimum is 5 m. An empty list returns `99.0`, the same fallback number `evaluate.py` uses when it has no obstacles. The snapshot print is `5.3259`: final ego against the first step's obstacle positions. That is a different question from the running clearance of `10.9731` m.
    """))
    cells.append(md("""
    <details><summary>Solution</summary>

    ```python
    def min_clearance(ego_x, ego_y, obstacles):
        if not obstacles:
            return 99.0
        return float(min(np.hypot(ego_x - o["x"], ego_y - o["y"]) for o in obstacles))
    ```

    </details>
    """))
    cells.append(md("""
    **Exercise — `count_offset_jumps`.** Count how many times consecutive plan end-`y` values differ. The skip-tracker list from section 3 should score 4. Leave the `TODO` in place to use the reference.
    """))
    cells.append(code("""
    def count_offset_jumps_student(end_y):
        # TODO: number of positions i where end_y[i] != end_y[i + 1]
        raise NotImplementedError

    def count_offset_jumps_reference(end_y):
        return int(sum(a != b for a, b in zip(end_y, end_y[1:])))

    def get_jump_counter():
        try:
            count_offset_jumps_student([0.0, 1.0])
        except NotImplementedError:
            print("Using reference count_offset_jumps (TODO not implemented)")
            return count_offset_jumps_reference
        print("Using your count_offset_jumps")
        return count_offset_jumps_student

    jump_fn = get_jump_counter()
    raw_jumps = jump_fn(raw_run["ends"])
    tracked_jumps = jump_fn(tracked_run["ends"])
    assert raw_jumps == raw_run["jumps"] == 4
    assert tracked_jumps == 0
    print("skip-tracker jumps:", raw_jumps)
    print("tracked jumps:", tracked_jumps)
    print("✅ correct: count_offset_jumps =", raw_jumps)
    """))
    cells.append(md("""
    The check prints `✅`. The skip-tracker ends change 4 times. The tracked ends change 0 times. That is the same pair of numbers section 3 printed as `offset jumps`.
    """))
    cells.append(md("""
    <details><summary>Solution</summary>

    ```python
    def count_offset_jumps(end_y):
        return int(sum(a != b for a, b in zip(end_y, end_y[1:])))
    ```

    </details>
    """))

    cells.append(md("## 7. Recap"))
    cells.append(md("""
    The integration test checks that `step` returns its keys, that steer, accel, and cross-track error are finite, that `ego_v > 0`, and that `ego_x` grows over three steps.

    **Predict:** `pytest` on `modules/08_capstone_fsd/tests/test_integration.py` passes. It does not report a clearance or an offset jump.
    """))
    cells.append(code("""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "modules/08_capstone_fsd/tests/test_integration.py", "-q"],
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
    `pytest` exit is `0`. The quiet report is `2 passed in 1.20s`. Those two tests are the wiring check from the predict cell. They do not assert `min true clearance` or `offset jumps`.

    `evaluate.py` builds `DrivingScenario(num_frames=25, dt=0.1)` and loops `for step in range(25)`. The collisions line in that file is the literal `Collisions: 0 (ZERO FAILURES)`. It does not compare `min_d` to a radius. Our 15-step runs measured a minimum true clearance of `10.9731` m, which is a large gap in this particular scene, and that measurement is the one to cite. A fixed print would still say zero if the gap were small.
    """))
    cells.append(md("""
    - One `step` runs cameras through `HydraNet.backbone`, `LiftSplatShoot`, and `OccupancyNetwork`. The scenario points then go through `MultiObjectTracker`, `LatticePlanner` / `TrajectoryCostEvaluator`, Stanley, the PID, and `KinematicBicycleModel.step`.
    - On one scenario step the camera tensor is `(1, 3, 3, 128, 256)`, BEV is `(1, 32, 40, 60)`, occupancy is `(1, 1, 40, 60, 8)`, and the telemetry dict has no occupancy key. `ego_x` prints `1.2000`.
    - Feeding flickering boxes (`sigma 0.8` m, seed 0) straight to the planner produces `4` offset jumps and steer kicks of `-1.84` deg and `-1.87` deg. The same noise through the tracker produces `0` jumps. The tracker still reports a third box on six steps.
    - With clean ground-truth centers, the full loop, the loop with occupancy skipped, and the loop with the tracker skipped all steer the same way. Final ego `x` is `17.9814`. Occupancy is timed and then ignored by the planner.
    - Mean full-step time on this CPU is `11.37` ms, under the `100` ms design tick. Hydra is the slow stage at `5.44` ms. The `87.9` Hz figure is `1000 / 11.37` on this run only.
    - `pytest` reports `2 passed in 1.20s`. That means the graph runs and the ego moves forward.

    This toy stack is a Python process and a kinematic bicycle. It is not a vehicle, and it does not carry a safety case. The cameras are Gaussian noise. The planner does not see occupancy. The tests do not score collisions. Nothing here should be pointed at a motor.

    ### Go deeper
    - UniAD, a stack trained as one model rather than wired like this one: [arXiv:2212.10156](https://arxiv.org/abs/2212.10156)
    - Duckietown's own lab docs, if you later use their robot: [docs.duckietown.org](https://docs.duckietown.org/)
    - comma.ai's safety page, for reading a real product's engagement rules: [comma.ai/safety](https://comma.ai/safety)
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
