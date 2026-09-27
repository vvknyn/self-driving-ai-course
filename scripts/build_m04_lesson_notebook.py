#!/usr/bin/env python3
"""Regenerate the Module 04 lesson notebook.

Writes ``notebooks/05_3d_occupancy_and_temporal_memory.ipynb`` next to this staging tree.

    python staging/self-driving-ai-course/scripts/build_m04_lesson_notebook.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "05_3d_occupancy_and_temporal_memory.ipynb"


def md(source: str):
    return new_markdown_cell(textwrap.dedent(source).strip() + "\n")


def code(source: str):
    return new_code_cell(textwrap.dedent(source).strip() + "\n")


def build() -> nbformat.NotebookNode:
    cells = []

    cells.append(md("""
    # Module 04 — Is this cube of space occupied?

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/notebooks/05_3d_occupancy_and_temporal_memory.ipynb)

    A bird's-eye grid is a floor plan. It can say that something sits on this square of ground. It cannot say whether that something is a low bumper or a branch hanging over empty air.

    An **occupancy** grid cuts the space around the car into small cubes, called **voxels**, and marks each cube occupied or free. Later frames can **keep the mark when the camera blinks** — memory — so a pillar does not erase a car for one timestep.

    Each section explains one idea, then runs code. **Predict first**, then execute the cell.
    """))

    cells.append(code("""
    import os
    import subprocess
    import sys
    from pathlib import Path

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    def keep_inline():
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

    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    def find_repo(start: Path) -> Path:
        for p in [start, start.parent]:
            if (p / "modules" / "04_occupancy_network").is_dir():
                return p.resolve()
        return start.resolve()

    REPO = find_repo(Path.cwd())
    if not (REPO / "modules" / "04_occupancy_network").is_dir():
        dest = Path.cwd() / "self-driving-ai-course"
        if not (dest / "modules" / "04_occupancy_network").is_dir():
            subprocess.run(
                ["git", "clone", "--depth", "1", "-q",
                 "https://github.com/vvknyn/self-driving-ai-course.git", str(dest)],
                check=True,
            )
        REPO = dest.resolve()

    os.chdir(REPO)

    try:
        import torch as _t
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "torch",
             "--index-url", "https://download.pytorch.org/whl/cpu"],
            check=True,
        )
    try:
        import matplotlib as _m
        import numpy as _n
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy", "matplotlib"], check=True)
    try:
        import pytest  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)

    sys.path.insert(0, str(REPO / "modules" / "04_occupancy_network"))
    from temporal_fusion import ConvGRUCell, OccupancyNetwork
    from voxel_grid import VoxelGridConfig, generate_synthetic_3d_occupancy

    print("Repo:", REPO)
    print("module files:", sorted(p.name for p in (REPO / "modules" / "04_occupancy_network").glob("*.py")))
    keep_inline()
    """))

    cells.append(md("## 1. Cubes, empty air, and a blink"))
    cells.append(md("""
    Picture a tiny room made of cubes: 4 along the road, 4 across, and 4 up. We mark a 2×2×2 block on the bottom two layers, like a car, and one extra cell two layers above a corner of that block, like a branch over empty air.

    A **3D bounding box** wraps every occupied cell. The brick includes the air inside it. **Occupancy** marks only the filled cubes.

    A **pillar** hides the car for one frame. That frame alone would mark the car cell empty. **Memory** means keep the mark when the camera blinks: without memory the car cell goes visible, gone, visible; with memory it stays marked through the blink.

    **Predict:** how many cells are in a 4×4×4 cube? How many are occupied? On three frames with a blind middle frame, what three numbers does the car cell print with no memory, and what three with memory?
    """))

    cells.append(code("""
    side = 4
    cube = np.zeros((side, side, side), dtype=int)
    for x in (1, 2):
        for y in (1, 2):
            for z in (0, 1):
                cube[x, y, z] = 1
    cube[2, 2, 3] = 1

    print("side:", side)
    print("cells:", side * side * side)
    print("occupied:", int(cube.sum()))
    for x, y, z in np.argwhere(cube == 1):
        print(f"occupied ({int(x)}, {int(y)}, {int(z)})")

    fig, axes = plt.subplots(1, side, figsize=(2.2 * side, 2.4))
    for z in range(side):
        layer = cube[:, :, z].T
        axes[z].imshow(layer, origin="lower", cmap="Blues", vmin=0, vmax=1)
        axes[z].set_title(f"z={z}")
        axes[z].set_xticks(range(side))
        axes[z].set_yticks(range(side))
        for iy in range(side):
            for ix in range(side):
                val = cube[ix, iy, z]
                axes[z].text(ix, iy, "#" if val else ".", ha="center", va="center", color="white" if val else "0.4")
    plt.suptitle("horizontal slices (x across, y up)")
    plt.tight_layout()
    plt.show()

    colors = np.zeros(cube.shape + (4,))
    z_index = np.arange(side)[None, None, :]
    colors[(cube == 1) & (z_index <= 1)] = (0.30, 0.47, 0.66, 0.95)
    colors[(cube == 1) & (z_index >= 3)] = (0.96, 0.52, 0.09, 0.95)
    fig = plt.figure(figsize=(4.5, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(cube.astype(bool), facecolors=colors, edgecolor="0.25")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("blue = low block, orange = high cell")
    ax.view_init(elev=22, azim=-58)
    plt.tight_layout()
    plt.show()
    print("drawn low cells: blue")
    print("drawn high cell: orange")

    xs, ys, zs = np.where(cube == 1)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    z0, z1 = int(zs.min()), int(zs.max())
    box_cells = (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1)
    occupied = int(cube.sum())
    print("occupied min corner:", (x0, y0, z0))
    print("occupied max corner:", (x1, y1, z1))
    print("box cells:", box_cells)
    print("occupied cells:", occupied)
    print("empty cells inside the box:", box_cells - occupied)

    car = (2, 1, 1)
    visible = [True, False, True]
    print("car cell:", car)
    no_memory = []
    for t, seen in enumerate(visible):
        grid = np.zeros((4, 4, 4), dtype=int)
        if seen:
            grid[car] = 1
        no_memory.append(int(grid[car]))
        state = "visible" if seen else "behind pillar"
        print(f"frame {t} {state} car cell {no_memory[-1]}")

    with_memory = []
    mark = 0
    for t, seen in enumerate(visible):
        if seen:
            mark = 1
        with_memory.append(mark)
        state = "visible" if seen else "behind pillar"
        print(f"frame {t} {state} kept car cell {with_memory[-1]}")

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))
    titles = ["frame only (1, 0, 1)", "keep the mark (1, 1, 1)"]
    series = [no_memory, with_memory]
    colors_bar = ["#2f4f7a", "#c0392b", "#2f4f7a"]
    for ax, vals, title in zip(axes, series, titles):
        ax.bar([0, 1, 2], vals, color=colors_bar)
        ax.set_ylim(0, 1.15)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["frame 0", "frame 1", "frame 2"])
        ax.set_ylabel("car cell")
        ax.set_title(title)
        for xi, v in zip([0, 1, 2], vals):
            ax.text(xi, v + 0.04, str(v), ha="center")
    plt.suptitle("car cell at (2, 1, 1) — pillar on frame 1")
    plt.tight_layout()
    plt.show()
    print("without memory:", no_memory)
    print("with memory:", with_memory)
    """))

    cells.append(md("""
    The room has `64` cells and `9` occupied. The slice row at `z=2` is all dots: empty air between the blue block and the orange cell at `z=3`. A floor plan would squash orange onto the same ground square as blue and could not show that gap.

    The occupied corners run from `(1, 1, 0)` to `(2, 2, 3)`. One box around them is 2×2×4, so `box cells: 16` while only `occupied cells: 9`. The other `empty cells inside the box: 7` are air the voxels can leave free.

    The car sits at `(2, 1, 1)`. Frame-only marks read `1`, `0`, `1` — visible, gone, visible. Keeping the mark reads `1`, `1`, `1`. The bars use those same integers. For a real stack, a blank middle frame is the difference between trusting only what you see now and trusting what you saw a moment ago.
    """))

    cells.append(md("## 2. Driving-scale voxels"))
    cells.append(md("""
    `VoxelGridConfig` in `voxel_grid.py` is the same cube idea in meters. An index is

    ```
    ix = int((x - x_min) / voxel_size)
    ```

    and the same for `y` and `z`.

    **Predict:** default cell size 0.5 m, x from 0 to 32 m, y from -12 to 12 m, z from -1 to 3 m — how many cells along each axis? On the test grid (1 m cells, x 0–16, y -8–8, z -1–3), where does `(5, 0, 1)` m land?
    """))

    cells.append(code("""
    cfg = VoxelGridConfig()
    print("x meters:", cfg.x_min, cfg.x_max)
    print("y meters:", cfg.y_min, cfg.y_max)
    print("z meters:", cfg.z_min, cfg.z_max)
    print("voxel size meters:", cfg.voxel_size)
    print("nx, ny, nz:", cfg.nx, cfg.ny, cfg.nz)
    print("total voxels:", cfg.nx * cfg.ny * cfg.nz)

    test_cfg = VoxelGridConfig(
        x_range=(0.0, 16.0),
        y_range=(-8.0, 8.0),
        z_range=(-1.0, 3.0),
        voxel_size=1.0,
    )
    print("test nx, ny, nz:", test_cfg.nx, test_cfg.ny, test_cfg.nz)
    print("index of (5, 0, 1):", test_cfg.point_to_voxel_index(5.0, 0.0, 1.0))

    scene = generate_synthetic_3d_occupancy(cfg, lead_x=16.0, lead_y=0.0, lead_vx=2.0, include_barrier=True)
    occ = scene["occupancy"]
    vel = scene["velocity"]
    lead = vel[0, 0] != 0
    static = (occ[0, 0] == 1) & (vel[0, 0] == 0)
    overhang_z0 = cfg.z_min + 6 * cfg.voxel_size
    overhang = int(occ[0, 0, :, :, 6:].sum().item())
    static_count = int(static.sum().item())
    print("occupancy shape:", tuple(occ.shape))
    print("velocity shape:", tuple(vel.shape))
    print("occupancy values:", [float(v) for v in torch.unique(occ).tolist()])
    print("occupied cells:", int(occ.sum().item()))
    print("lead cells:", int(lead.sum().item()))
    print("lead vx:", f"{vel[0, 0][lead].unique().item():.1f}")
    print("static occupied cells:", static_count)
    print("overhang z start meters:", f"{overhang_z0:.1f}")
    print("overhang cells:", overhang)
    print("barrier cells:", static_count - overhang)
    """))

    cells.append(md("""
    Default extents are x `0.0` to `32.0`, y `-12.0` to `12.0`, z `-1.0` to `3.0`, with `voxel size meters: 0.5`. Counts are `nx, ny, nz: 64 48 8`, and `total voxels: 24576`.

    The test grid is `test nx, ny, nz: 16 16 4`. The point `(5, 0, 1)` m is `index of (5, 0, 1): (5, 8, 2)`.

    The synthetic scene is binary: `occupancy values: [0.0, 1.0]`. Shapes are `occupancy shape: (1, 1, 64, 48, 8)` and `velocity shape: (1, 3, 64, 48, 8)`. `occupied cells: 256`. `lead cells: 96` store `lead vx: 2.0`. `static occupied cells: 160`. The overhang starts at `overhang z start meters: 2.0` with `overhang cells: 40`; the rest of the static volume is the barrier at `barrier cells: 120`. Those 40 high cells are the orange-cell idea at road scale.
    """))

    cells.append(md("## 3. ConvGRU — keep the mark when the camera blinks"))
    cells.append(md("""
    The repo's `ConvGRUCell` blends the previous hidden map with a new candidate. One cell updates as

    ```
    h_next = (1 - update_gate) * h_prev + update_gate * candidate
    ```

    `update_gate` near 0 means trust memory. Near 1 means trust the new frame.

    **Predict:** with `h_prev = 0.8`, `candidate = 0.2`, `update_gate = 0.3`, is `h_next` closer to 0.8 or 0.2? On a blank frame with the gate bias pushed to -8, does a hidden value of 1 at the car cell stay near 1 if you pass the previous map in?
    """))

    cells.append(code("""
    h_prev = 0.8
    candidate = 0.2
    update_gate = 0.3
    h_next = (1.0 - update_gate) * h_prev + update_gate * candidate
    print("h_prev:", h_prev)
    print("candidate:", candidate)
    print("update gate:", update_gate)
    print("memory share:", f"{(1.0 - update_gate) * h_prev:.2f}")
    print("new share:", f"{update_gate * candidate:.2f}")
    print("h_next:", f"{h_next:.2f}")

    torch.manual_seed(SEED)
    cell = ConvGRUCell(in_channels=1, hidden_channels=1)
    with torch.no_grad():
        cell.conv_gates.weight.zero_()
        cell.conv_gates.bias.zero_()
        cell.conv_gates.bias[1] = -8.0
        cell.conv_candidate.weight.zero_()
        cell.conv_candidate.bias.zero_()
    print("update-gate bias:", f"{float(cell.conv_gates.bias[1].detach()):.1f}")

    h_prev_map = torch.zeros(1, 1, 4, 4)
    h_prev_map[0, 0, 2, 1] = 1.0
    blank = torch.zeros(1, 1, 4, 4)
    with torch.no_grad():
        forgotten = cell(blank, None)
        kept = cell(blank, h_prev_map)
        combined = torch.cat([blank, h_prev_map], dim=1)
        gates = torch.sigmoid(cell.conv_gates(combined))
        reset_gate, update_gate_map = torch.chunk(gates, 2, dim=1)
        candidate_state = torch.tanh(
            cell.conv_candidate(torch.cat([blank, reset_gate * h_prev_map], dim=1))
        )
    print("update gate at car:", f"{update_gate_map[0, 0, 2, 1].item():.6f}")
    print("candidate at car:", f"{candidate_state[0, 0, 2, 1].item():.6f}")
    print("one minus update gate:", f"{(1.0 - update_gate_map[0, 0, 2, 1]).item():.6f}")
    print("hidden if h_prev is None:", f"{forgotten[0, 0, 2, 1].item():.6f}")
    print("hidden if h_prev is carried:", f"{kept[0, 0, 2, 1].item():.6f}")
    """))

    cells.append(md("""
    The scalar blend gives `h_next: 0.62` from `memory share: 0.56` and `new share: 0.06` — closer to memory `0.8` than to candidate `0.2`.

    With `update-gate bias: -8.0`, a blank frame opens the gate only a little: `update gate at car: 0.000335`. `hidden if h_prev is None: 0.000000` is the section-1 pillar frame with no map to copy from. `hidden if h_prev is carried: 0.999665` matches `one minus update gate: 0.999665`: the mark at the car cell barely moves.

    `OccupancyNetwork.forward` expects the new features and this hidden map on the same grid. Nothing in `temporal_fusion.py` moves the memory when the car turns. If the vehicle yaws and you do not re-align the map first, the mark stays in the old cell.
    """))

    cells.append(md("## 4. Motion in the cube"))
    cells.append(md("""
    The generator stores velocity on lead cells. The network's velocity head predicts one `(vx, vy, vz)` per ground column and copies it onto every height in that column.

    **Predict:** which occupied cells have `vx` of 2 in the synthetic scene? On a blank feature map, will an untrained head's occupancy probabilities sit near 0 and 1, or near one half?
    """))

    cells.append(code("""
    torch.manual_seed(SEED)
    net = OccupancyNetwork(in_channels=4, hidden_channels=8, nz=test_cfg.nz)
    blank_bev = torch.zeros(1, 4, test_cfg.nx, test_cfg.ny)
    with torch.no_grad():
        occ_probs, vel_3d, h_next = net(blank_bev, None)
    print("network occupancy shape:", tuple(occ_probs.shape))
    print("network velocity shape:", tuple(vel_3d.shape))
    print("network hidden shape:", tuple(h_next.shape))
    print("prob min:", f"{occ_probs.min().item():.4f}")
    print("prob max:", f"{occ_probs.max().item():.4f}")
    column = vel_3d[0, :, 3, 4, :]
    print("vx at column (3, 4), every height:", [f"{v:.4f}" for v in column[0].tolist()])
    print("vy at column (3, 4), every height:", [f"{v:.4f}" for v in column[1].tolist()])
    print("vz at column (3, 4), every height:", [f"{v:.4f}" for v in column[2].tolist()])
    print("same velocity at every height:", bool(torch.allclose(vel_3d[..., :1], vel_3d[..., 1:])))
    """))

    cells.append(md("""
    In section 2, the `96` lead cells are the ones with `lead vx: 2.0`. The `160` static cells store vx 0.

    The untrained network on a blank map returns shapes `(1, 1, 16, 16, 4)`, `(1, 3, 16, 16, 4)`, and `(1, 8, 16, 16)`. Probabilities run from `prob min: 0.4498` to `prob max: 0.5501` — one half, not a detector. Do not read 0.55 as "occupied."

    At column `(3, 4)` every height prints the same vector; `same velocity at every height: True`. The numbers are an untrained head on zeros, not the generator's `2.0`.
    """))

    cells.append(md("## 5. Exercises"))
    cells.append(md("""
    **Exercise — `meters_to_index`.** Return the integer voxel index `(ix, iy, iz)` for a point in meters. Same formula as `VoxelGridConfig.point_to_voxel_index`. Leave the `TODO` as it is to use the reference method on the test grid.
    """))

    cells.append(code("""
    def meters_to_index_student(x, y, z, x_min, y_min, z_min, voxel_size):
        # TODO: integer voxel index
        raise NotImplementedError

    def get_index_fn():
        try:
            meters_to_index_student(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        except NotImplementedError:
            print("Using reference point_to_voxel_index (TODO not implemented)")
            def reference(x, y, z, x_min, y_min, z_min, voxel_size):
                return test_cfg.point_to_voxel_index(x, y, z)
            return reference
        print("Using your meters_to_index")
        return meters_to_index_student

    index_fn = get_index_fn()
    ix, iy, iz = index_fn(5.0, 0.0, 1.0, test_cfg.x_min, test_cfg.y_min, test_cfg.z_min, test_cfg.voxel_size)
    print("point meters:", (5.0, 0.0, 1.0))
    print("index:", (ix, iy, iz))
    assert (ix, iy, iz) == test_cfg.point_to_voxel_index(5.0, 0.0, 1.0)
    print("✅ correct: index", (ix, iy, iz))
    """))

    cells.append(md("""
    The check prints `✅`. The point `(5.0, 0.0, 1.0)` m maps to `(5, 8, 2)`, matching section 2. The reference ran because the `TODO` still raises.
    """))

    cells.append(md("""
    <details><summary>Solution</summary>

    ```python
    ix = int((x - x_min) / voxel_size)
    iy = int((y - y_min) / voxel_size)
    iz = int((z - z_min) / voxel_size)
    return ix, iy, iz
    ```

    </details>
    """))

    cells.append(md("""
    **Exercise — `blend_memory`.** One hidden value. Return `(1 - update_gate) * h_prev + update_gate * candidate`. Leave the `TODO` in place to use the reference blend. On `0.8`, `0.2`, and `0.3` the result is the `0.62` from section 3.
    """))

    cells.append(code("""
    def blend_memory_student(h_prev, candidate, update_gate):
        # TODO: mix previous memory with the new candidate
        raise NotImplementedError

    def blend_memory_reference(h_prev, candidate, update_gate):
        return (1.0 - update_gate) * h_prev + update_gate * candidate

    def get_blend_fn():
        try:
            blend_memory_student(0.0, 0.0, 0.0)
        except NotImplementedError:
            print("Using reference blend_memory (TODO not implemented)")
            return blend_memory_reference
        print("Using your blend_memory")
        return blend_memory_student

    blend_fn = get_blend_fn()
    blended = float(blend_fn(0.8, 0.2, 0.3))
    print("h_prev:", 0.8)
    print("candidate:", 0.2)
    print("update gate:", 0.3)
    print("h_next:", f"{blended:.2f}")
    assert abs(blended - 0.62) < 1e-6
    print("✅ correct: h_next =", f"{blended:.2f}")
    """))

    cells.append(md("""
    The check prints `✅`. `h_next: 0.62` matches section 3. The reference ran because the `TODO` still raises.
    """))

    cells.append(md("""
    <details><summary>Solution</summary>

    ```python
    return (1.0 - update_gate) * h_prev + update_gate * candidate
    ```

    </details>
    """))

    cells.append(md("""
    The module tests check this index, tensor shapes, and that probabilities stay between 0 and 1.

    **Predict:** `tests/test_occupancy.py` has three tests. How many pass?
    """))

    cells.append(code("""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "modules/04_occupancy_network/tests/test_occupancy.py", "-q"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    assert completed.returncode == 0
    """))

    cells.append(md("""
    The summary line is `3 passed`. Index math, synthetic shapes, and the temporal forward pass all succeeded.
    """))

    cells.append(md("""
    ## 6. Recap

    - A 4×4×4 cube has 64 cells. Nine were occupied: a low 2×2×2 block and one cell at `(2, 2, 3)`, with empty air between them on the slices.
    - One box around those cells covers 16 cells and would mark 7 empty ones. Voxels leave that air free.
    - The default grid is 64×48×8 = 24576 voxels at 0.5 m. The test grid is 16×16×4 at 1 m. `(5, 0, 1)` m is index `(5, 8, 2)`.
    - The synthetic scene has 256 occupied cells. 96 lead cells store vx 2.0. 120 are barrier. 40 are overhang from z = 2.0 m up.
    - Frame-only marks the car cell 1, 0, 1 behind a pillar; keeping the mark gives 1, 1, 1.
    - The gate blend turns 0.8 and 0.2 at gate 0.3 into 0.62. Inside `ConvGRUCell`, update gate 0.000335 on a blank frame leaves a carried hidden value at 0.999665; `h_prev=None` leaves 0.
    - The velocity head copies one vector onto every height. On blank input, occupancy sits between 0.4498 and 0.5501. The scene speed is the generator's 2.0 until you train the head.
    - This module does not move the hidden map when the car turns; the grid and the memory must already line up.

    ### Go deeper

    - [Tesla AI Day 2022](https://www.youtube.com/watch?v=ODSJsviD_SU) — occupancy from video instead of a box per object.
    - [SurroundOcc (Wei et al., arXiv:2303.09551)](https://arxiv.org/abs/2303.09551) — multi-camera images lifted into a 3D occupancy volume.
    - [Occ3D (Tian et al., arXiv:2304.14365)](https://arxiv.org/abs/2304.14365) — labeled occupancy grids; the synthetic scene here is not that benchmark.
    """))

    nb = new_notebook()
    nb.cells = cells
    nb.metadata["colab"] = {"provenance": []}
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "pygments_lexer": "ipython3",
    }
    return nb


def main() -> None:
    nb = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, OUT)
    print(f"wrote {OUT} cells={len(nb.cells)}")


if __name__ == "__main__":
    main()
