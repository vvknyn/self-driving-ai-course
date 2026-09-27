"""Build the Module 04 Colab lesson notebook.

Writes staging/self-driving-ai-course/notebooks/05_3d_occupancy_and_temporal_memory.ipynb.

    python scripts/build_m04_lesson_notebook.py
    python scripts/build_m04_lesson_notebook.py --execute

The copy checked into staging/ is the executed notebook: every number in the
prose is a value printed by the cell above it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "staging" / "self-driving-ai-course" / "notebooks" / "05_3d_occupancy_and_temporal_memory.ipynb"

CELLS: list[tuple[str, str]] = []


def md(source: str) -> None:
    CELLS.append(("markdown", source.strip("\n") + "\n"))


def code(source: str) -> None:
    CELLS.append(("code", source.strip("\n") + "\n"))


md(
    """
# Module 04 — Is this cube of space occupied?

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/notebooks/05_3d_occupancy_and_temporal_memory.ipynb)

A bird's-eye grid is a floor plan. It can say that something sits on this square of ground. It cannot say whether that something is a low bumper or a branch hanging over empty air.

An **occupancy** grid cuts the space around the car into small cubes, called **voxels**, and marks each cube occupied or free.

Each section explains one idea, then runs code. **Predict first**, then execute the cell.
"""
)

code(
    """
import os
import subprocess
import sys
from pathlib import Path

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
            ["git", "clone", "--depth", "1", "-q", "https://github.com/vvknyn/self-driving-ai-course.git", str(dest)],
            check=True,
        )
    REPO = dest.resolve()

os.chdir(REPO)

try:
    import torch
except ImportError:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "torch",
         "--index-url", "https://download.pytorch.org/whl/cpu"],
        check=True,
    )
try:
    import matplotlib
    import numpy
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy", "matplotlib"], check=True)
try:
    import pytest  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)

import matplotlib.pyplot as plt
import numpy as np
import torch

%matplotlib inline

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

sys.path.insert(0, str(REPO / "modules" / "04_occupancy_network"))
from temporal_fusion import ConvGRUCell, OccupancyNetwork
from voxel_grid import VoxelGridConfig, generate_synthetic_3d_occupancy

print("Repo:", REPO)
print("module files:", sorted(p.name for p in (REPO / "modules" / "04_occupancy_network").glob("*.py")))
"""
)

md(
    """
## 1. A floor plan is flat

Picture a tiny room made of cubes, 4 along the road, 4 across, and 4 up. That is the whole idea of an occupancy grid: the floor plan gains a height.

We will mark a 2×2×2 block on the bottom two layers, like a car, and one extra cell two layers above a corner of that block, like a branch.

**Predict:** how many cells are in a 4×4×4 cube? How many of them will be occupied, and is the high cell sitting on the car or above a gap?
"""
)

code(
    """
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

print("slices, top row is y=3, left column is x=0, # is occupied")
for z in range(side):
    print(f"z={z}")
    for y in range(side - 1, -1, -1):
        print("".join("#" if cube[x, y, z] else "." for x in range(side)))

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
"""
)

md(
    """
The cube has `64` cells. `9` of them are occupied. The list runs from `(1, 1, 0)` through `(2, 2, 3)`.

The slices at `z=0` and `z=1` are the same 2×2 block:

```
....
.##.
.##.
....
```

`z=2` is four rows of `....`. The high cell is alone at `z=3`, on the row `..#.`. There is a gap under it. The picture paints the low block blue and that one cell orange. A floor plan would squash the orange cell onto the same ground square as the blue block and could not say the air in between is empty.
"""
)

md(
    """
## 2. A box fills the empty air

A 3D box is the smallest axis-aligned brick that contains every occupied cell. Anything inside the brick is "the object," including air.

**Predict:** using the corners printed next, how many cells does that brick cover, and how many of those cells are actually empty?
"""
)

code(
    """
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
"""
)

md(
    """
The occupied cells run from `(1, 1, 0)` to `(2, 2, 3)`. The brick is 2 by 2 by 4, so `box cells: 16`. Only `9` of those are occupied. `empty cells inside the box: 7`. Those 7 include the air under the orange cell. A voxel grid can leave that air free. One box cannot, unless you cut it into smaller boxes, which is the grid you already have.
"""
)

md(
    """
## 3. Build the voxel grid

`VoxelGridConfig` in `modules/04_occupancy_network/voxel_grid.py` is the same idea at driving scale. An index is

```
ix = int((x - x_min) / voxel_size)
```

and the same for `y` and `z`.

**Predict:** the default cell is 0.5 m, x runs 0 to 32 m, y runs -12 to 12 m, and z runs -1 to 3 m. How many cells along each axis? On the test grid (1 m cells, x 0 to 16, y -8 to 8, z -1 to 3), where does the point `(5, 0, 1)` m land?
"""
)

code(
    """
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
"""
)

md(
    """
Default extents are x `0.0` to `32.0`, y `-12.0` to `12.0`, z `-1.0` to `3.0`, with `voxel size meters: 0.5`. Counts are `nx, ny, nz: 64 48 8`, and `total voxels: 24576`. That is `32 / 0.5`, `24 / 0.5`, and `4 / 0.5`.

The test grid is `test nx, ny, nz: 16 16 4`. The point `(5, 0, 1)` m is `index of (5, 0, 1): (5, 8, 2)`. Forward is just 5 cells in. Lateral 0 is 8 m above `y = -8`. Height 1 is 2 m above `z = -1`.

The synthetic scene is binary: `occupancy values: [0.0, 1.0]`. Shapes are `occupancy shape: (1, 1, 64, 48, 8)` and `velocity shape: (1, 3, 64, 48, 8)`. `occupied cells: 256`. `lead cells: 96` store `lead vx: 2.0`. `static occupied cells: 160`. The overhang starts at `overhang z start meters: 2.0` and holds `overhang cells: 40`. The other still cells are the barrier: `barrier cells: 120`. The 40 high cells are the orange-cell idea at road scale: occupied volume that does not touch the ground.
"""
)

md(
    """
## 4. Each frame on its own

First try: build occupancy from this frame only. A car sits in one cell of the tiny cube. On the middle frame a pillar blocks the camera, so that frame does not see the car.

**Predict:** what does the car cell print on the three frames if a frame that cannot see the car writes 0?
"""
)

code(
    """
car = (2, 1, 1)
visible = [True, False, True]
print("car cell:", car)
for t, seen in enumerate(visible):
    grid = np.zeros((4, 4, 4), dtype=int)
    if seen:
        grid[car] = 1
    state = "visible" if seen else "behind pillar"
    print(f"frame {t} {state} car cell {int(grid[car])}")
"""
)

md(
    """
The car cell is `(2, 1, 1)`. The three lines are `frame 0 visible car cell 1`, `frame 1 behind pillar car cell 0`, `frame 2 visible car cell 1`. The car did not leave. This rule has nothing to copy from frame 0, so the pillar erases it.
"""
)

md(
    """
## 5. Carry the mark forward

Keep the last mark when the new frame is blank. That is the whole of temporal memory: the hidden cube from the previous frame is an input to this frame.

**Predict:** with that rule, what does the same car cell print on the occluded frame?
"""
)

code(
    """
mark = 0
for t, seen in enumerate(visible):
    if seen:
        mark = 1
    state = "visible" if seen else "behind pillar"
    print(f"frame {t} {state} car cell {mark}")
"""
)

md(
    """
The occluded line is now `frame 1 behind pillar car cell 1`. The sequence is 1, 1, 1. Frame 2 is visible again and still 1. The mark survived because we kept it, not because the pillar frame saw the car.

The repo's `ConvGRUCell` does this with a gate. One cell of the hidden map updates as

```
h_next = (1 - update_gate) * h_prev + update_gate * candidate
```

`update_gate` near 0 means "trust memory." `update_gate` near 1 means "trust the new frame."

**Predict:** `h_prev = 0.8`, `candidate = 0.2`, `update_gate = 0.3`. Which number is `h_next` closer to?
"""
)

code(
    """
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
"""
)

md(
    """
`h_next: 0.62`. That is `memory share: 0.56` plus `new share: 0.06`. Closer to the memory `0.8` than to the new candidate `0.2`.

Now the same blend inside `ConvGRUCell`. The weights are set to zero and the update-gate bias is set to -8, so a blank frame barely opens the gate. This is not a trained detector. It is the update the class actually runs, with the gate held near "keep memory" so you can see the mark.

**Predict:** on a blank frame, what happens to a hidden value of 1 at the car cell if you pass `h_prev=None`, and what happens if you pass the previous hidden state?
"""
)

code(
    """
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
"""
)

md(
    """
`update-gate bias: -8.0`. At the car cell the `update gate at car: 0.000335` and the `candidate at car: 0.000000` (the candidate convolution's weights and bias are 0, so its tanh input is 0).

`hidden if h_prev is None: 0.000000`. The cell starts from zeros when you do not hand it a previous map, which is the pillar frame of section 4: the mark is gone.

`one minus update gate: 0.999665`, and `hidden if h_prev is carried: 0.999665`. Same number: one tiny step from 1 toward the candidate 0. The hidden object stays marked because the previous map was passed in and the gate almost ignored the blank frame.

`OccupancyNetwork.forward` assumes the new features and this hidden map already share a grid. Nothing in `temporal_fusion.py` slides the memory when the car turns. If the vehicle yaws and you do not move the map first, the mark stays in the old cell.
"""
)

md(
    """
## 6. Motion in the cube

The generator also stores a velocity on the lead cells. The network's velocity head predicts one `(vx, vy, vz)` per ground column and copies it onto every height in that column.

**Predict:** which occupied cells in the synthetic scene have `vx` of 2, and which have 0? On a blank feature map, will an untrained head's probabilities sit near 0 and 1, or near one half?
"""
)

code(
    """
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
"""
)

md(
    """
From the scene in section 3, the `96` lead cells are the ones with `lead vx: 2.0`. The `160` static cells, barrier plus overhang, store vx 0.

The untrained network, on a blank map, returns `network occupancy shape: (1, 1, 16, 16, 4)`, `network velocity shape: (1, 3, 16, 16, 4)`, and `network hidden shape: (1, 8, 16, 16)`. Probabilities run from `prob min: 0.4498` to `prob max: 0.5501`. That is one half, not a car detector. Do not read 0.55 as "occupied."

At column `(3, 4)` every height prints the same vector: vx `-0.1438`, vy `0.1333`, vz `-0.0320`. `same velocity at every height: True`. The copy along z is real. The numbers are not a measured speed. They are an untrained head on zeros. The speed that belongs to the scene is the generator's `2.0`.
"""
)

md(
    """
## 7. Exercises

**Exercise — `meters_to_index`.** Return the integer voxel index `(ix, iy, iz)` for a point in meters. Same formula as `VoxelGridConfig.point_to_voxel_index`. Leave the `TODO` as it is to use the reference method on the test grid.
"""
)

code(
    """
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
print("✅ correct: (5, 0, 1) m ->", (ix, iy, iz))
"""
)

md(
    """
The check prints `✅`. The point `(5.0, 0.0, 1.0)` m maps to `(5, 8, 2)`, the same index as section 3. The reference method ran because the `TODO` still raises. Replace the `TODO` and run the cell again if you want the check to call your function.
"""
)

md(
    """
<details><summary>Solution</summary>

```python
ix = int((x - x_min) / voxel_size)
iy = int((y - y_min) / voxel_size)
iz = int((z - z_min) / voxel_size)
return ix, iy, iz
```

</details>
"""
)

md(
    """
**Exercise — `blend_memory`.** One hidden value. Return `(1 - update_gate) * h_prev + update_gate * candidate`. Leave the `TODO` in place to use the reference blend. On `0.8`, `0.2`, and `0.3` the result is the `0.62` from section 5.
"""
)

code(
    """
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
"""
)

md(
    """
The check prints `✅`. `h_next: 0.62`, the same blend as section 5. The reference ran because the `TODO` still raises.
"""
)

md(
    """
<details><summary>Solution</summary>

```python
return (1.0 - update_gate) * h_prev + update_gate * candidate
```

</details>
"""
)

md(
    """
The module tests check this same index, the tensor shapes, and that probabilities stay between 0 and 1.

**Predict:** `tests/test_occupancy.py` has three tests. How many of them pass?
"""
)

code(
    """
completed = subprocess.run(
    [sys.executable, "-m", "pytest", "modules/04_occupancy_network", "-q"],
    cwd=REPO,
    capture_output=True,
    text=True,
)
print(completed.stdout)
print(completed.stderr)
assert completed.returncode == 0
"""
)

md(
    """
The summary line says `3 passed`. The index check, the synthetic shapes, and the temporal forward pass all passed.
"""
)

md(
    """
## 8. Recap

- A 4×4×4 cube has 64 cells. Nine were occupied: a low 2×2×2 block and one cell at `(2, 2, 3)`, with an empty layer between them.
- One box around those cells covers 16 cells and marks 7 empty ones. The voxels leave that air free.
- The default grid is 64×48×8 = 24576 voxels at 0.5 m. The test grid is 16×16×4 at 1 m. `(5, 0, 1)` m is index `(5, 8, 2)`.
- The synthetic scene has 256 occupied cells, values 0 or 1. 96 lead cells store vx 2.0. 120 are the barrier. 40 are the overhang from z = 2.0 m up.
- A per-frame grid prints the car cell as 1, 0, 1 when the middle frame is behind a pillar. Keeping the last mark prints 1, 1, 1.
- The gate blend turns memory 0.8 and candidate 0.2 at gate 0.3 into 0.62. Inside `ConvGRUCell`, update gate 0.000335 and candidate 0 leave a carried hidden value at 0.999665. Passing `h_prev=None` leaves 0.
- The velocity head copies one vector onto every height. On a blank input that vector is `-0.1438, 0.1333, -0.0320`, and occupancy sits between 0.4498 and 0.5501. The scene's speed is the generator's 2.0. The head has not been trained.
- Nothing in this module slides the hidden map when the car turns. The new frame and the memory have to already share a grid.

### Go deeper

- [Tesla AI Day 2022](https://www.youtube.com/watch?v=ODSJsviD_SU) — the occupancy-network segment: a 3D grid of occupied space and its motion, from video, instead of a box around every object.
- [SurroundOcc (Wei et al., arXiv:2303.09551)](https://arxiv.org/abs/2303.09551) — multi-camera images lifted into a 3D occupancy volume. This notebook does not train that model.
- [Occ3D (Tian et al., arXiv:2304.14365)](https://arxiv.org/abs/2304.14365) — a benchmark of labeled occupancy grids. The synthetic scene above is not that benchmark.
"""
)


def build() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(source) if kind == "markdown" else new_code_cell(source)
        for kind, source in CELLS
    ]
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
    if "--execute" in sys.argv:
        from nbclient import execute

        execute(nb, cwd="/tmp/self-driving-ai-course", timeout=180, kernel_name="python3")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, OUT)
    print(f"wrote {OUT} cells={len(nb.cells)} executed={'--execute' in sys.argv}")


if __name__ == "__main__":
    main()
