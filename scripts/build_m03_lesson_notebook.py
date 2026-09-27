#!/usr/bin/env python3
"""Regenerate the Module 03 lesson notebook.

Writes ``notebooks/04_bev_lift_splat_shoot.ipynb`` next to this course staging tree.
The notebook is the lesson: run it top to bottom. This script does not execute it.

    python staging/self-driving-ai-course/scripts/build_m03_lesson_notebook.py
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
    # Module 03 — Lifting cameras into a top-down map

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/notebooks/04_bev_lift_splat_shoot.ipynb)

    Module 01 painted every pixel onto a flat road. A bumper or a torso sits above that plane, so a flat map puts raised objects in the wrong place. Lift–splat–shoot keeps each pixel as a **ray** with many possible depths, splats mass into top-down bins, then mixes neighbors with a small convolution.

    Each section explains one idea, then runs code. **Predict first**, then execute the cell.
    """))

    cells.append(code("""
    import importlib.util
    import os, random, subprocess, sys
    from pathlib import Path

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    def keep_inline():
        # Some course scripts select a file-only backend on import.
        # Put figures back on the notebook backend so plt.show() renders here.
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
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    def find_repo(start: Path) -> Path:
        for p in [start, start.parent]:
            if (p / "modules" / "03_bev_transform").is_dir():
                return p.resolve()
        return start.resolve()

    REPO = find_repo(Path.cwd())
    if not (REPO / "modules" / "03_bev_transform").is_dir():
        dest = Path.cwd() / "self-driving-ai-course"
        if not (dest / "modules" / "03_bev_transform").is_dir():
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/vvknyn/self-driving-ai-course.git", str(dest)],
                check=True,
            )
        REPO = dest.resolve()
        os.chdir(REPO)
    else:
        os.chdir(REPO)

    try:
        import torch as _t
        import cv2
        import matplotlib as _mpl
        import numpy as _np
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision",
             "--index-url", "https://download.pytorch.org/whl/cpu"],
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
            check=True,
        )

    sys.path.insert(0, str(REPO / "modules" / "03_bev_transform"))
    sys.path.insert(0, str(REPO / "modules" / "01_camera_geometry"))
    sys.path.insert(0, str(REPO / "modules"))

    from lift_splat_shoot import DepthFeatureLift, LiftSplatShoot
    from calibrate_rig import build_tesla_style_rig

    def shown_meter(value):
        rounded = round(float(value), 3)
        return 0.0 if rounded == 0 else rounded

    print("Repo:", REPO)
    print("device:", torch.device("cpu"))
    keep_inline()
    """))

    cells.append(md("## 1. Flat ground, smeared rays, wrong squares"))
    cells.append(md("""
    Picture the destination before the formulas. **Inverse perspective** assumes the world is a flat plate. A point on the road and a raised point at the same forward and lateral spot project to different pixels; forcing both pixels onto the ground plane writes the raised object down the road.

    A pixel is also a **ray**: depth is not one number but a pile of bins. Spread probability evenly and the obstacle is painted along the whole ray on the top-down map. Pick the wrong bin and the pillar lands in the **wrong square** on the floor grid.

    **Predict:** the road point returns near 20 m forward on the ground plane. The raised point's pixel, forced onto that plane, lands much farther forward. On the floor preview, true depth and half depth should land on different forward bins.
    """))
    cells.append(code("""
    rig = build_tesla_style_rig()
    front = rig["front"]
    print(f"front camera: {front.width} x {front.height}")

    ground_ego = np.array([[20.0, 0.0, 0.0]])
    raised_ego = np.array([[20.0, 0.0, 1.0]])
    uv_ground, valid_ground = front.project_ego_to_pixel(ground_ego)
    uv_raised, valid_raised = front.project_ego_to_pixel(raised_ego)
    ipm_ground, ipm_ground_ok = front.project_pixels_to_ground(uv_ground, ground_z=0.0)
    ipm_raised, ipm_raised_ok = front.project_pixels_to_ground(uv_raised, ground_z=0.0)

    print(f"ground pixel uv: ({uv_ground[0, 0]:.2f}, {uv_ground[0, 1]:.2f})")
    print(f"raised pixel uv: ({uv_raised[0, 0]:.2f}, {uv_raised[0, 1]:.2f})")
    print(f"IPM ground X: {ipm_ground[0, 0]:.2f}")
    print(f"IPM raised X: {ipm_raised[0, 0]:.2f}")
    print(f"true X ground: {ground_ego[0, 0]:.2f}")
    print(f"true X raised: {raised_ego[0, 0]:.2f}")

    P_ego = np.array([20.0, 0.0, 1.0])
    P_cam = front.R @ P_ego + front.T.ravel()
    d_raised = float(P_cam[2])
    uv, valid_uv = front.project_ego_to_pixel(P_ego.reshape(1, 3))
    u_raised = float(uv[0, 0])
    v_raised = float(uv[0, 1])
    print(f"raised ray u={u_raised:.3f} v={v_raised:.3f} d={d_raised:.3f}")

    lss_geo = LiftSplatShoot()
    K = torch.tensor(front.K, dtype=torch.float32)
    R = torch.tensor(front.R, dtype=torch.float32)
    T = torch.tensor(front.T, dtype=torch.float32)
    packed = torch.tensor([u_raised * d_raised, v_raised * d_raised, d_raised], dtype=torch.float32)
    ego_true = lss_geo.unproject_to_ego(packed.view(1, 1, 1, 3), K, R, T).reshape(3).detach()
    raised_xyz = torch.tensor([shown_meter(ego_true[i]) for i in range(3)])

    d_half = d_raised / 2.0
    packed_half = torch.tensor([u_raised * d_half, v_raised * d_half, d_half], dtype=torch.float32)
    ego_half = lss_geo.unproject_to_ego(packed_half.view(1, 1, 1, 3), K, R, T).reshape(3).detach()
    half_xyz = torch.tensor([shown_meter(ego_half[i]) for i in range(3)])

    xmin, xstep = 0.0, 0.5
    ymin, ystep = -15.0, 0.5
    nx_prev = int(round((40.0 - xmin) / xstep))
    ny_prev = int(round((15.0 - ymin) / ystep))

    def bin_index(x_value, y_value):
        xi = ((torch.tensor(float(x_value)) - xmin) / xstep).long()
        yi = ((torch.tensor(float(y_value)) - ymin) / ystep).long()
        return int(xi), int(yi)

    true_xi, true_yi = bin_index(raised_xyz[0], raised_xyz[1])
    half_xi, half_yi = bin_index(half_xyz[0], half_xyz[1])
    print(f"true bin x_idx={true_xi} y_idx={true_yi}")
    print(f"half bin x_idx={half_xi} y_idx={half_yi}")

    num_bins = 20
    depth_bins = torch.linspace(2.0, 42.0, num_bins)
    flat_logits = torch.zeros(1, num_bins, 1, 1)
    flat_probs = F.softmax(flat_logits, dim=1).squeeze()
    true_bin = 9
    peaked_logits = torch.full((1, num_bins, 1, 1), -5.0)
    peaked_logits[0, true_bin, 0, 0] = 8.0
    peaked_probs = F.softmax(peaked_logits, dim=1).squeeze()
    entropy_flat = -(flat_probs * torch.log(flat_probs + 1e-9)).sum().item()
    entropy_peak = -(peaked_probs * torch.log(peaked_probs + 1e-9)).sum().item()
    spread_m = depth_bins[-1].item() - depth_bins[0].item()
    print(f"flat entropy: {entropy_flat:.2f}")
    print(f"peaked entropy: {entropy_peak:.4f}")
    print(f"ray spread if flat: {spread_m:.1f} m")

    keep_inline()
    labels = ["ground Z=0", "raised Z=1"]
    xpos = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5))

    ax = axes[0, 0]
    ax.bar(xpos - width / 2, [ground_ego[0, 0], raised_ego[0, 0]], width, label="true X")
    ax.bar(xpos + width / 2, [ipm_ground[0, 0], ipm_raised[0, 0]], width, label="IPM X")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("forward X (m)")
    ax.set_title("flat ground warp")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.bar(depth_bins.numpy(), flat_probs.numpy(), width=1.2, alpha=0.55, label="flat")
    ax.bar(depth_bins.numpy(), peaked_probs.numpy(), width=0.8, alpha=0.85, label="peaked")
    ax.set_xlabel("depth bin center (m)")
    ax.set_ylabel("probability")
    ax.set_title("same pixel, two depth guesses")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    xs_flat, xs_peak = [], []
    for prob_f, prob_p, d_c in zip(flat_probs, peaked_probs, depth_bins):
        pack_f = torch.tensor([u_raised * d_c, v_raised * d_c, d_c], dtype=torch.float32)
        pack_p = pack_f
        xf = lss_geo.unproject_to_ego(pack_f.view(1, 1, 1, 3), K, R, T)[0, 0, 0, 0].item()
        xs_flat.append(prob_f.item() * xf)
        xs_peak.append(prob_p.item() * xf)
    x_grid = np.linspace(0, 40, 81)
    ax.fill_between(x_grid, 0, 0.08, color="0.92")
    ax.bar(x_grid[::4], np.zeros(len(x_grid[::4])), width=0.4, color="0.85", edgecolor="0.7")
    ax.plot(xs_flat, np.full(len(xs_flat), 0.04), "o", color="C3", label="flat mass")
    ax.plot(xs_peak, np.full(len(xs_peak), 0.06), "o", color="C0", label="peaked mass")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 0.1)
    ax.set_xlabel("forward X on floor (m)")
    ax.set_yticks([])
    ax.set_title("top-down: obstacle smear vs pin")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1, 1]
    floor = np.zeros((ny_prev, nx_prev))
    floor[true_yi, true_xi] = 2.0
    floor[half_yi, half_xi] = 1.0
    ax.imshow(floor, origin="lower", cmap="Blues", extent=(0, 40, -15, 15), aspect="auto")
    ax.scatter([raised_xyz[0].item()], [raised_xyz[1].item()], c="C2", s=80, marker="*", label="true depth")
    ax.scatter([half_xyz[0].item()], [half_xyz[1].item()], c="C3", s=80, marker="x", label="half depth")
    ax.set_xlabel("forward X (m)")
    ax.set_ylabel("lateral Y (m)")
    ax.set_title("wrong depth → wrong square")
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.show()
    """))
    cells.append(md("""
    The ground pixel `(320.00, 181.44)` maps back to IPM X `20.00`. The raised pixel `(320.00, 171.20)` is only ten rows higher in the image, but the flat warp writes it at IPM X `65.00` while both true forward positions were `20.00`. For the car, that is a phantom obstacle 45 m ahead — emergency braking on empty road.

    The same raised pixel carries depth `17.984`. Unprojecting with the true depth lands bin `x_idx=40`, `y_idx=30`. Halving depth (`8.992`) slides the pillar to `x_idx=22` on the same row. A depth network that is confidently wrong moves occupancy cells the planner treats as real.

    Flat depth entropy is `3.00` over a `40.0` m span along the ray; peaked entropy is `0.0006`. The smear panel paints mass from near the bumper to far down the lane; the peaked panel concentrates on one forward strip. Section 2 names the math that produces those two curves.
    """))

    cells.append(md("## 2. Softmax turns logits into a depth pile"))
    cells.append(md("""
    **Lift** assigns each pixel a vector of logits, one per depth bin. Softmax turns that vector into probabilities that sum to 1. The expected depth is the weighted sum of bin centers.

    **Predict:** the toy three-bin softmax sums to 1. On `DepthFeatureLift`, every spatial location still sums to 1 along the depth axis.
    """))
    cells.append(code("""
    logits = torch.tensor([2.0, 0.0, -2.0])
    depth_p = torch.softmax(logits, dim=0)
    bins = torch.tensor([2.0, 11.0, 20.0])
    expected = (depth_p * bins).sum()
    print("softmax: [" + ", ".join(f"{v.item():.4f}" for v in depth_p) + "]")
    print(f"softmax sum: {depth_p.sum().item():.4f}")
    print(f"expected depth: {expected.item():.4f}")

    torch.manual_seed(0)
    lift = DepthFeatureLift(in_channels=4, out_channels=2, num_depth_bins=3)
    probs, context = lift(torch.randn(1, 4, 2, 2))
    print("probs.shape:", tuple(probs.shape))
    print("context.shape:", tuple(context.shape))
    print(f"probs.sum(dim=1) min: {probs.sum(dim=1).min().item():.4f}")
    print(f"probs.sum(dim=1) max: {probs.sum(dim=1).max().item():.4f}")
    """))
    cells.append(md("""
    Softmax on the toy logits yields `[0.8668, 0.1173, 0.0159]` summing to `1.0000`, with expected depth `3.3416` on bins 2 m, 11 m, and 20 m. The real lift returns `probs.shape` `(1, 3, 2, 2)` and `context.shape` `(1, 2, 2, 2)`; along dim 1 the sums stay at `1.0000`. For driving, context is the appearance feature carried along whichever depth bin wins.
    """))
    cells.append(md("""
    When logits are flat, softmax spreads probability across the whole ray — high entropy, wide smear on the BEV. A sharp peak lowers entropy and pins the obstacle.

    **Predict:** entropy drops when logits spike on one bin. Spread in meters shrinks with a peak.
    """))
    cells.append(code("""
    num_bins = 20
    depth_bins = torch.linspace(2.0, 42.0, num_bins)
    flat_logits = torch.zeros(1, num_bins, 1, 1)
    broken_probs = F.softmax(flat_logits, dim=1).squeeze()
    true_bin = 9
    peaked_logits = torch.full((1, num_bins, 1, 1), -5.0)
    peaked_logits[0, true_bin, 0, 0] = 8.0
    fixed_probs = F.softmax(peaked_logits, dim=1).squeeze()

    entropy_broken = -(broken_probs * torch.log(broken_probs + 1e-9)).sum().item()
    entropy_fixed = -(fixed_probs * torch.log(fixed_probs + 1e-9)).sum().item()
    spread_meters = depth_bins[-1].item() - depth_bins[0].item()
    peak_m = depth_bins[true_bin].item()
    peak_pct = fixed_probs[true_bin].item() * 100.0
    half_bin = (depth_bins[1] - depth_bins[0]).item() / 2.0

    print(f"entropy flat: {entropy_broken:.2f}")
    print(f"entropy peaked: {entropy_fixed:.4f}")
    print(f"smear span: {spread_meters:.1f} m")
    print(f"peak at: {peak_m:.1f} m")
    print(f"peak probability: {peak_pct:.1f}%")
    print(f"localization half-width: {half_bin:.2f} m")

    keep_inline()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    axes[0].plot(depth_bins.numpy(), broken_probs.numpy(), "o-", label="flat logits")
    axes[0].plot(depth_bins.numpy(), fixed_probs.numpy(), "o-", label="peaked logits")
    axes[0].set_xlabel("depth (m)")
    axes[0].set_ylabel("probability")
    axes[0].set_title("probability vs depth")
    axes[0].legend(fontsize=8)

    mass_b = []
    mass_f = []
    x_centers = []
    for pb, pf, d_c in zip(broken_probs, fixed_probs, depth_bins):
        pack = torch.tensor([u_raised * d_c, v_raised * d_c, d_c], dtype=torch.float32)
        xf = lss_geo.unproject_to_ego(pack.view(1, 1, 1, 3), K, R, T)[0, 0, 0, 0].item()
        x_centers.append(xf)
        mass_b.append(pb.item())
        mass_f.append(pf.item())
    axes[1].bar(x_centers, mass_b, width=0.9, alpha=0.45, label="flat → floor")
    axes[1].bar(x_centers, mass_f, width=0.5, alpha=0.9, label="peaked → floor")
    axes[1].set_xlabel("forward X (m)")
    axes[1].set_ylabel("probability mass")
    axes[1].set_title("splat preview along one ray")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    """))
    cells.append(md("""
    Flat logits give entropy `3.00` and smear the obstacle across `40.0` m of depth. The peaked run drops entropy to `0.0006`, puts `100.0`% of mass at `20.9` m, and localizes within `1.05` m. The floor strip shows why: uniform depth paints many forward cells; a peak fills one column the motion planner must respect.
    """))

    cells.append(md("## 3. Unproject (u, v, d) into ego (X, Y, Z)"))
    cells.append(md("""
    A frustum sample packs `[u*d, v*d, d]`. `LiftSplatShoot.unproject_to_ego` inverts the pinhole model with the camera's `K`, `R`, and `T`. Depth is camera-frame Z.

    **Predict:** the raised point at 20 m forward, 1 m up, returns to itself. The flat warp had sent this pixel to 65 m.
    """))
    cells.append(code("""
    print(f"u={u_raised:.3f}")
    print(f"v={v_raised:.3f}")
    print(f"d={d_raised:.3f}")
    print(f"pixel valid: {bool(valid_uv[0])}")

    packed = torch.tensor([u_raised * d_raised, v_raised * d_raised, d_raised], dtype=torch.float32)
    print(
        "packed [u*d, v*d, d]: "
        f"[{packed[0].item():.3f}, {packed[1].item():.3f}, {packed[2].item():.3f}]"
    )

    ego = lss_geo.unproject_to_ego(packed.view(1, 1, 1, 3), K, R, T).reshape(3).detach()
    raised_xyz = torch.tensor([shown_meter(ego[i]) for i in range(3)])
    print(
        "unprojected ego XYZ: "
        f"({raised_xyz[0].item():.3f}, {raised_xyz[1].item():.3f}, {raised_xyz[2].item():.3f})"
    )
    """))
    cells.append(md("""
    The pixel is `u=320.000`, `v=171.200`, depth `d=17.984`. Packed `[5754.898, 3078.873, 17.984]` unprojects to `(20.000, 0.000, 1.000)` — the raised torso, not the ground under it. Occupancy in a BEV stack must come from this ray geometry, not from a single ground intersection.
    """))

    cells.append(md("## 4. Splat into BEV bins"))
    cells.append(md("""
    The top-down map is a grid of bins. `x_bound` is `(x_min, x_max, step)`. Forward index is `((x - xmin) / step)` cast with `.long()`; lateral index uses the same pattern. Out-of-range points drop. Two features that hit the same flat index **add** via `index_add_`.

    **Predict:** two rows splatted into one cell sum feature-wise. A point behind the car fails the bounds check.
    """))
    cells.append(code("""
    x_bound = (0.0, 40.0, 0.5)
    y_bound = (-15.0, 15.0, 0.5)
    nx = int(round((x_bound[1] - x_bound[0]) / x_bound[2]))
    ny = int(round((y_bound[1] - y_bound[0]) / y_bound[2]))
    print(f"nx={nx}")
    print(f"ny={ny}")

    x_m = float(raised_xyz[0])
    y_m = float(raised_xyz[1])
    print(f"splat ego X={x_m:.3f} Y={y_m:.3f}")
    x_idx = ((torch.tensor(x_m) - x_bound[0]) / x_bound[2]).long()
    y_idx = ((torch.tensor(y_m) - y_bound[0]) / y_bound[2]).long()
    print(f"x_idx={int(x_idx)}")
    print(f"y_idx={int(y_idx)}")

    grid = torch.zeros(2, ny * nx)
    rows = torch.tensor([[1.0, 0.0], [0.5, 2.0]])
    flat = int(y_idx) * nx + int(x_idx)
    grid.index_add_(1, torch.tensor([flat, flat]), rows.T)
    cell = grid[:, flat]
    print(f"cell feature: [{cell[0].item():.1f}, {cell[1].item():.1f}]")
    print(f"grid sum: {grid.sum().item():.1f}")

    x_behind = -5.0
    x_idx_behind = ((torch.tensor(x_behind) - x_bound[0]) / x_bound[2]).long()
    valid_behind = bool(
        (int(x_idx_behind) >= 0)
        and (int(x_idx_behind) < nx)
        and (int(y_idx) >= 0)
        and (int(y_idx) < ny)
    )
    print(f"behind X={x_behind:.1f} x_idx={int(x_idx_behind)} valid={valid_behind}")
    print(f"grid sum after discard: {grid.sum().item():.1f}")

    keep_inline()
    heat = grid[0].view(ny, nx)
    r0, r1 = int(y_idx) - 2, int(y_idx) + 3
    c0, c1 = int(x_idx) - 2, int(x_idx) + 3
    window = heat[r0:r1, c0:c1]
    print(f"heatmap window max: {window.max().item():.1f}")
    plt.figure(figsize=(3.2, 3.2))
    plt.imshow(window.detach(), cmap="magma", origin="lower")
    plt.title("splat cell")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    """))
    cells.append(md("""
    The grid is `nx=80` by `ny=60`. The raised point at `X=20.000`, `Y=0.000` lands at `x_idx=40`, `y_idx=30`. Two camera rows adding into that cell give `[1.5, 2.0]`. A point at `X=-5.0` is `valid=False`, so nothing behind the ego pollutes the map — important so reverse lane ghosts do not appear in forward planning.
    """))

    cells.append(md("## 5. Shoot: a small top-down convolution"))
    cells.append(md("""
    **Shoot** is a convolution on the splatted map. Padding keeps height and width fixed so each cell can borrow context from neighbors without resizing the grid.

    **Predict:** a 3×3 conv with padding 1 keeps a 5×5 toy map at 5×5. The module's BEV encoder does the same on a 20×20 grid.
    """))
    cells.append(code("""
    toy = torch.zeros(1, 1, 5, 5)
    toy[0, 0, 2, 2] = 1.0
    conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    torch.nn.init.ones_(conv.weight)
    toy_out = conv(toy)
    print("toy out shape:", tuple(toy_out.shape))
    for row in toy_out.squeeze().detach():
        print(" ".join(f"{v.item():.0f}" for v in row))

    torch.manual_seed(0)
    lss_shoot = LiftSplatShoot(
        in_channels=4,
        out_channels=4,
        num_depth_bins=4,
        d_min=2,
        d_max=20,
        x_bound=(0, 20, 1),
        y_bound=(-10, 10, 1),
    )
    n_conv = sum(isinstance(layer, torch.nn.Conv2d) for layer in lss_shoot.bev_encoder)
    print("bev_encoder convs:", n_conv)
    zeros = torch.zeros(1, 4, 20, 20)
    with torch.no_grad():
        enc = lss_shoot.bev_encoder(zeros)
    print("encoder in shape:", tuple(zeros.shape))
    print("encoder out shape:", tuple(enc.shape))
    """))
    cells.append(md("""
    The toy map stays `(1, 1, 5, 5)`; the center row reads `0 1 1 1 0`, a 3×3 neighborhood from one hot cell. `bev_encoder convs: 2`. Zeros `(1, 4, 20, 20)` encode to the same shape. For the vehicle, shoot is the cheap spatial denoising step after splatting — smoothing phantom speckle without blurring the whole map off-grid.
    """))

    cells.append(md("## 6. Shapes the repo actually returns"))
    cells.append(md("""
    Match the module test: ten depth bins from 2 m to 20 m, 1 m BEV cells. Then run the three-camera script.

    **Predict:** frustum rank is depth × height × width × 3. Probabilities sum to 1 on depth. Forward pass yields `(1, 32, 20, 20)` on the toy bounds.
    """))
    cells.append(code("""
    torch.manual_seed(0)
    lss_shapes = LiftSplatShoot(
        in_channels=16,
        out_channels=32,
        d_min=2,
        d_max=20,
        num_depth_bins=10,
        x_bound=(0, 20, 1),
        y_bound=(-10, 10, 1),
    )
    frustum = lss_shapes.create_camera_frustum(8, 16, torch.device("cpu"))
    print("frustum shape:", tuple(frustum.shape))
    print(f"depth min: {frustum[..., 2].min().item():.1f}")
    print(f"depth max: {frustum[..., 2].max().item():.1f}")

    torch.manual_seed(0)
    lift_test = DepthFeatureLift(16, 32, 10)
    lift_feat = torch.randn(2, 16, 8, 16)
    probs6, context6 = lift_test(lift_feat)
    sums6 = probs6.sum(dim=1)
    print("probs.shape:", tuple(probs6.shape))
    print("context.shape:", tuple(context6.shape))
    print(f"sum dim 1 min: {sums6.min().item():.4f}")
    print(f"sum dim 1 max: {sums6.max().item():.4f}")

    B, N_cams = 1, 2
    torch.manual_seed(0)
    feats6 = torch.randn(B, N_cams, 16, 8, 16)
    K6 = torch.eye(3).view(1, 1, 3, 3).repeat(B, N_cams, 1, 1)
    R6 = torch.eye(3).view(1, 1, 3, 3).repeat(B, N_cams, 1, 1)
    T6 = torch.zeros(3, 1).view(1, 1, 3, 1).repeat(B, N_cams, 1, 1)
    with torch.no_grad():
        bev_out = lss_shapes(feats6, K6, R6, T6)
    print("bev_out.shape:", bev_out.shape)
    print("nan count:", int(torch.isnan(bev_out).sum().item()))
    """))
    cells.append(md("""
    Frustum shape `(10, 8, 16, 3)` spans depth `2.0` to `20.0`. Lift returns probs `(2, 10, 8, 16)` and context `(2, 32, 8, 16)` with depth sums pinned at `1.0000`. One forward pass yields `torch.Size([1, 32, 20, 20])` and `nan count: 0` — the tensor shape the unit tests assert.
    """))
    cells.append(md("""
    The checked-in driver builds front, left, and right rigs and runs lift, splat, and shoot together.

    **Predict:** batch 1, three cameras in, BEV tensor wider in X than a single-camera toy run.
    """))
    cells.append(code("""
    print("cameras: front, left, right")
    subprocess.run(
        [sys.executable, "modules/03_bev_transform/run_bev.py"],
        check=True,
        cwd=REPO,
    )
    """))
    cells.append(md("""
    Input `torch.Size([1, 3, 32, 16, 32])` fuses to output `torch.Size([1, 64, 60, 80])`. Coverage is `0.0m to 40.0m` forward and `-15.0m to 15.0m` lateral at `0.5m` cells. That is the multi-camera BEV feature map a downstream head would read for lanes and obstacles.
    """))

    cells.append(md("## 7. The wrong depth bin lands in the wrong pillar"))
    cells.append(md("""
    Same pixel and camera as above. Halving depth is the deliberate mistake: the network guessed too close. Bin both landings on the half-meter floor grid.

    **Predict:** half depth moves the pillar toward the ego. True depth restores `x_idx=40`.
    """))
    cells.append(code("""
    d_half = d_raised / 2.0
    print(f"half d={d_half:.3f}")
    packed_half = torch.tensor(
        [u_raised * d_half, v_raised * d_half, d_half], dtype=torch.float32
    )
    ego_half = lss_geo.unproject_to_ego(packed_half.view(1, 1, 1, 3), K, R, T).reshape(3).detach()
    half_xyz = torch.tensor([shown_meter(ego_half[i]) for i in range(3)])
    print(
        "half XYZ: "
        f"({half_xyz[0].item():.3f}, {half_xyz[1].item():.3f}, {half_xyz[2].item():.3f})"
    )

    true_xi, true_yi = bin_index(raised_xyz[0], raised_xyz[1])
    half_xi, half_yi = bin_index(half_xyz[0], half_xyz[1])
    print(
        "true XYZ: "
        f"({raised_xyz[0].item():.3f}, {raised_xyz[1].item():.3f}, {raised_xyz[2].item():.3f})"
    )
    print(f"true x_idx={true_xi} y_idx={true_yi}")
    print(f"half x_idx={half_xi} y_idx={half_yi}")

    ego_fix = lss_geo.unproject_to_ego(packed.view(1, 1, 1, 3), K, R, T).reshape(3).detach()
    fix_xyz = torch.tensor([shown_meter(ego_fix[i]) for i in range(3)])
    fix_xi, fix_yi = bin_index(fix_xyz[0], fix_xyz[1])
    print(
        "fixed XYZ: "
        f"({fix_xyz[0].item():.3f}, {fix_xyz[1].item():.3f}, {fix_xyz[2].item():.3f})"
    )
    print(f"fixed x_idx={fix_xi} y_idx={fix_yi}")

    keep_inline()
    floor = np.zeros((ny_prev, nx_prev))
    floor[true_yi, true_xi] = 2.0
    floor[half_yi, half_xi] = 1.5
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.imshow(floor, origin="lower", cmap="Greys", extent=(0, 40, -15, 15), aspect="auto", vmin=0, vmax=2)
    ax.scatter([raised_xyz[0].item()], [raised_xyz[1].item()], c="C2", s=120, marker="*", label=f"true d → bin {true_xi}")
    ax.scatter([half_xyz[0].item()], [half_xyz[1].item()], c="C3", s=120, marker="X", label=f"half d → bin {half_xi}")
    ax.axvline(raised_xyz[0].item(), color="C2", ls="--", alpha=0.4)
    ax.axvline(half_xyz[0].item(), color="C3", ls="--", alpha=0.4)
    ax.set_xlabel("forward X (m)")
    ax.set_ylabel("lateral Y (m)")
    ax.set_title("one pixel, two depth bins")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.show()
    """))
    cells.append(md("""
    True depth `(20.000, 0.000, 1.000)` sits at `x_idx=40`, `y_idx=30`. Half depth `8.992` lands at `(11.000, 0.000, 1.200)` → `x_idx=22`. Lateral index stays `30`; only forward bin moves. Fixing depth returns `fixed x_idx=40`. A single mis-ranked softmax bin shifts occupied cells nine meters forward — inside the braking envelope.
    """))

    cells.append(md("## 8. Exercises"))
    cells.append(md("""
    Three checks. Each stub still raises, so the cell falls back to a reference that calls the real API. Replace the TODO when you want the check to call your function.
    """))
    cells.append(md("""
    **Exercise 1.** Sum the depth probabilities along the depth axis.

    **Predict:** every spatial location sums to 1, and the probability tensor keeps ten depth bins.
    """))
    cells.append(code("""
    def depth_sum_student(features):
        # TODO: return depth probabilities summed along the depth axis
        raise NotImplementedError

    def depth_sum_reference(features):
        lift_ex = DepthFeatureLift(in_channels=16, out_channels=32, num_depth_bins=10)
        probs_ex, _context_ex = lift_ex(features)
        print("probs.shape:", tuple(probs_ex.shape))
        return probs_ex.sum(dim=1)

    def get_depth_sum():
        try:
            depth_sum_student(torch.zeros(1, 16, 2, 2))
        except NotImplementedError:
            print("Using reference depth_sum_student (TODO not implemented)")
            return depth_sum_reference
        print("Using your depth_sum_student")
        return depth_sum_student

    torch.manual_seed(0)
    ex_feat = torch.randn(2, 16, 8, 16)
    depth_fn = get_depth_sum()
    depth_sums = depth_fn(ex_feat)
    assert torch.allclose(depth_sums, torch.ones_like(depth_sums), atol=1e-5, rtol=1e-5)
    print("sum shape:", tuple(depth_sums.shape))
    print("✅ correct: depth probabilities sum to 1")
    """))
    cells.append(md("""
    Fallback line `Using reference depth_sum_student (TODO not implemented)` then `✅`. `probs.shape` is `(2, 10, 8, 16)`. Depth must normalize so splatting does not invent mass from nowhere.
    """))
    cells.append(md("""
    <details><summary>Solution</summary>

    ```python
    def depth_sum_student(features):
        lift_ex = DepthFeatureLift(in_channels=16, out_channels=32, num_depth_bins=10)
        probs_ex, _context_ex = lift_ex(features)
        return probs_ex.sum(dim=1)
    ```

    </details>
    """))

    cells.append(md("""
    **Exercise 2.** Return the camera frustum shape for an 8×16 feature map.

    **Predict:** ten depth bins, then height, width, and 3 coordinates.
    """))
    cells.append(code("""
    def frustum_shape_student(lss, h, w):
        # TODO: return the frustum shape as a tuple
        raise NotImplementedError

    def frustum_shape_reference(lss, h, w):
        return tuple(lss.create_camera_frustum(h, w, torch.device("cpu")).shape)

    def get_frustum_shape():
        try:
            frustum_shape_student(lss_shapes, 8, 16)
        except NotImplementedError:
            print("Using reference frustum_shape_student (TODO not implemented)")
            return frustum_shape_reference
        print("Using your frustum_shape_student")
        return frustum_shape_student

    frustum_fn = get_frustum_shape()
    frustum_shape = frustum_fn(lss_shapes, 8, 16)
    assert frustum_shape == (10, 8, 16, 3)
    print("frustum shape:", frustum_shape)
    print("✅ correct: frustum shape matches")
    """))
    cells.append(md("""
    Reference fallback then `✅`. Frustum shape `(10, 8, 16, 3)` is the lift input geometry the splat step expects.
    """))
    cells.append(md("""
    <details><summary>Solution</summary>

    ```python
    def frustum_shape_student(lss, h, w):
        return tuple(lss.create_camera_frustum(h, w, torch.device("cpu")).shape)
    ```

    </details>
    """))

    cells.append(md("""
    **Exercise 3.** Return the BEV output shape for the same identity calibration as section 6.

    **Predict:** batch 1, 32 channels, 20×20 grid.
    """))
    cells.append(code("""
    def bev_shape_student(lss, feats, K_mat, R_mat, T_mat):
        # TODO: return the BEV output shape as a tuple
        raise NotImplementedError

    def bev_shape_reference(lss, feats, K_mat, R_mat, T_mat):
        return tuple(lss(feats, K_mat, R_mat, T_mat).shape)

    def get_bev_shape():
        try:
            bev_shape_student(lss_shapes, feats6, K6, R6, T6)
        except NotImplementedError:
            print("Using reference bev_shape_student (TODO not implemented)")
            return bev_shape_reference
        print("Using your bev_shape_student")
        return bev_shape_student

    bev_fn = get_bev_shape()
    with torch.no_grad():
        bev_shape = bev_fn(lss_shapes, feats6, K6, R6, T6)
    assert bev_shape == (1, 32, 20, 20)
    print("bev shape:", bev_shape)
    print("✅ correct: BEV shape matches")
    """))
    cells.append(md("""
    Reference fallback then `✅`. BEV shape `(1, 32, 20, 20)` matches the forward pass in section 6.
    """))
    cells.append(md("""
    <details><summary>Solution</summary>

    ```python
    def bev_shape_student(lss, feats, K_mat, R_mat, T_mat):
        return tuple(lss(feats, K_mat, R_mat, T_mat).shape)
    ```

    </details>
    """))

    cells.append(md("## 9. Recap"))
    cells.append(md("""
    - Flat IPM put the road point at X `20.00` and the raised point at `65.00` from pixels `(320.00, 181.44)` and `(320.00, 171.20)`.
    - Flat depth entropy `3.00` smeared `40.0` m; peaked entropy `0.0006` with `100.0`% at `20.9` m localized within `1.05` m.
    - Softmax toy `[0.8668, 0.1173, 0.0159]` summed to `1.0000`; expected depth `3.3416`.
    - Unproject at `d=17.984` restored `(20.000, 0.000, 1.000)` at bin `x_idx=40`, `y_idx=30` on an `nx=80`, `ny=60` grid; two splats summed to `[1.5, 2.0]`.
    - Toy shoot stayed `(1, 1, 5, 5)`; encoder `(1, 4, 20, 20)` → `(1, 4, 20, 20)`.
    - Test forward `torch.Size([1, 32, 20, 20])`; three-camera script `torch.Size([1, 64, 60, 80])`.
    - Half depth moved the pillar to `x_idx=22`; true depth returned `x_idx=40`.

    ### Go deeper
    - [Philion & Fidler, Lift, Splat, Shoot, ECCV 2020](https://arxiv.org/abs/2008.05711)
    - [Official code](https://github.com/nv-tlabs/lift-splat-shoot)
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


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "notebooks" / "04_bev_lift_splat_shoot.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    nbformat.write(nb, out)
    print(f"Wrote {out} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
