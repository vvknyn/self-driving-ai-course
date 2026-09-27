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

    Module 01 already warped pixels onto a flat road. This notebook lifts each pixel along a ray, because a raised object and the ground under it are different places.

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

    print("Repo:", REPO)
    print("device:", torch.device("cpu"))
    keep_inline()
    """))

    cells.append(md("## 1. Flat-ground warp misses a raised point"))
    cells.append(md("""
    Module 01 mapped a pixel onto the road by assuming every pixel sits on the ground plane. A bumper or a torso is not on that plane.

    Use the front camera from `build_tesla_style_rig()`. Take one point on the road and the same forward and lateral point raised by a meter, still under the camera. Project both. Then intersect each pixel with the ground plane. That intersection is the attempt. Watch the raised point miss.

    **Predict:** the ground point should come back near 20 m. The raised point's pixel, forced onto the ground plane, should land much farther forward.
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
    print(f"ground pixel valid: {bool(valid_ground[0])}")
    print(f"raised pixel valid: {bool(valid_raised[0])}")
    print(f"IPM ground valid: {bool(ipm_ground_ok[0])}")
    print(f"IPM raised valid: {bool(ipm_raised_ok[0])}")
    print(f"true X ground: {ground_ego[0, 0]:.2f}")
    print(f"true X raised: {raised_ego[0, 0]:.2f}")

    keep_inline()
    labels = ["ground Z=0", "raised Z=1"]
    xpos = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(xpos - width / 2, [ground_ego[0, 0], raised_ego[0, 0]], width, label="true X")
    ax.bar(xpos + width / 2, [ipm_ground[0, 0], ipm_raised[0, 0]], width, label="IPM X")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("forward X (m)")
    ax.legend()
    plt.tight_layout()
    plt.show()
    """))
    cells.append(md("""
    The ground pixel `(320.00, 181.44)` comes back at IPM X `20.00`. The raised pixel `(320.00, 171.20)` is valid too, but forcing it onto the ground writes it at IPM X `65.00`. Both true X values were `20.00`. The flat-ground warp used one plane, so the raised point was written down the road. The next sections lift that pixel along its ray instead.
    """))

    cells.append(md("## 2. A pixel is a ray with a depth distribution"))
    cells.append(md("""
    Lift treats one pixel as a ray. Depth is a categorical distribution over bins: softmax turns logits into probabilities, and the expected depth is the sum of probability times bin center.

    **Predict:** the three probabilities sum to 1. On a real `DepthFeatureLift`, the probabilities sum to 1 along dim 1, the depth axis.
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
    Softmax prints `[0.8668, 0.1173, 0.0159]` and the sum is `1.0000`. The expected depth on bins 2 m, 11 m, and 20 m is `3.3416`. The real lift returns `probs.shape` `(1, 3, 2, 2)` and `context.shape` `(1, 2, 2, 2)`. Along dim 1 the probabilities run from `1.0000` to `1.0000`.
    """))
    cells.append(md("""
    Flat logits are the obvious attempt: every depth looks equally likely, so the obstacle is painted along the whole ray. A peaked distribution is the fix. The drill script runs both.

    **Predict:** flat logits smear the obstacle across the whole depth range. A peaked distribution pins it.
    """))
    cells.append(code("""
    subprocess.run(
        [sys.executable, "modules/03_bev_transform/break_it_fix_it.py"],
        check=True,
        cwd=REPO,
    )
    """))
    cells.append(md("""
    The flat run prints entropy `3.00` and smears the obstacle across `40.0` meters. The peaked run prints entropy `0.0006`, a peak probability of `100.0`% at `20.9` m, and localization within `1.05` meters. The same ray, two distributions: one fills the range, the other pins the obstacle.
    """))

    cells.append(md("## 3. Unproject (u, v, d) into ego (X, Y, Z)"))
    cells.append(md("""
    A frustum point is the packed triple `[u*d, v*d, d]`. `LiftSplatShoot.unproject_to_ego` turns that triple into ego X, Y, Z with the front camera's K, R, and T. Camera-frame position uses the same convention as `project_ego_to_pixel`: rotate the ego point, then add T. Depth is the camera-frame Z.

    **Predict:** the raised point should come back to itself. The ground warp had sent this same pixel much farther down the road.
    """))
    cells.append(code("""
    def shown_meter(value):
        rounded = round(float(value), 3)
        return 0.0 if rounded == 0 else rounded

    P_ego = np.array([20.0, 0.0, 1.0])
    P_cam = front.R @ P_ego + front.T.ravel()
    d_raised = float(P_cam[2])
    uv, valid_uv = front.project_ego_to_pixel(P_ego.reshape(1, 3))
    u_raised = float(uv[0, 0])
    v_raised = float(uv[0, 1])
    print(f"u={u_raised:.3f}")
    print(f"v={v_raised:.3f}")
    print(f"d={d_raised:.3f}")
    print(f"pixel valid: {bool(valid_uv[0])}")

    packed = torch.tensor([u_raised * d_raised, v_raised * d_raised, d_raised], dtype=torch.float32)
    print(
        "packed [u*d, v*d, d]: "
        f"[{packed[0].item():.3f}, {packed[1].item():.3f}, {packed[2].item():.3f}]"
    )

    lss_geo = LiftSplatShoot()
    K = torch.tensor(front.K, dtype=torch.float32)
    R = torch.tensor(front.R, dtype=torch.float32)
    T = torch.tensor(front.T, dtype=torch.float32)
    ego = lss_geo.unproject_to_ego(packed.view(1, 1, 1, 3), K, R, T).reshape(3).detach()
    raised_xyz = torch.tensor([shown_meter(ego[i]) for i in range(3)])
    print(
        "unprojected ego XYZ: "
        f"({raised_xyz[0].item():.3f}, {raised_xyz[1].item():.3f}, {raised_xyz[2].item():.3f})"
    )
    """))
    cells.append(md("""
    The pixel is `u=320.000`, `v=171.200`, depth `d=17.984`. The packed triple is `[5754.898, 3078.873, 17.984]`. Unproject returns `(20.000, 0.000, 1.000)`, the raised point. The ground warp had sent that pixel to `65.00`.
    """))

    cells.append(md("## 4. Splat into BEV bins"))
    cells.append(md("""
    A BEV grid is bins. `x_bound` is minimum, maximum, and step. The forward index is `((x - xmin) / step)` cast with `.long()`, and the same formula for y. Points outside the grid are dropped. Two feature rows that share one flat index add through `index_add_`.

    **Predict:** two cameras that hit one cell add. A point behind the grid is discarded.
    """))
    cells.append(code("""
    x_bound = (0.0, 40.0, 0.5)
    y_bound = (-15.0, 15.0, 0.5)
    nx = int(round((x_bound[1] - x_bound[0]) / x_bound[2]))
    ny = int(round((y_bound[1] - y_bound[0]) / y_bound[2]))
    print(f"nx={nx}")
    print(f"ny={ny}")

    # Bin the X and Y printed to 3 decimals. A millionth of a meter
    # would otherwise push Y into the neighboring cell.
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
    The grid is `nx=80` by `ny=60`. The unprojected point `X=20.000`, `Y=0.000` lands at `x_idx=40`, `y_idx=30`. The two rows add to `cell feature: [1.5, 2.0]`. The point at `X=-5.0` is `valid=False`, and the grid sum stays `3.5`.
    """))

    cells.append(md("## 5. Shoot: a small top-down convolution"))
    cells.append(md("""
    Shoot is a convolution on the top-down map. Padding keeps the height and width, so a cell can mix with its neighbors without changing the grid size.

    **Predict:** a 3 by 3 kernel with padding 1 leaves the map height and width unchanged, and the single 1 spreads into its neighborhood. The real encoder does not change height or width either.
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
    The toy map stays `(1, 1, 5, 5)`. The center row prints `0 1 1 1 0`: the single 1 spread into a 3 by 3 neighborhood and the corners stayed `0`. `bev_encoder convs: 2`. Zeros of shape `(1, 4, 20, 20)` come back as `(1, 4, 20, 20)`. Shoot does not change height or width.
    """))

    cells.append(md("## 6. Shapes the repo actually returns"))
    cells.append(md("""
    Match the module test: ten depth bins from 2 m to 20 m, and a 1 m BEV grid. Then run the 3-camera script.

    **Predict:** probabilities still sum to 1 along the depth axis. The frustum stores three numbers at every pixel and depth. The BEV map keeps the grid height and width.
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
    The frustum shape is `(10, 8, 16, 3)`, with depth min `2.0` and depth max `20.0`. The lift returns probs `(2, 10, 8, 16)` and context `(2, 32, 8, 16)`, and the sum along dim 1 runs from `1.0000` to `1.0000`. The forward pass prints `torch.Size([1, 32, 20, 20])` with nan count `0`.
    """))
    cells.append(md("""
    The checked-in script builds the front, left, and right cameras and runs lift, splat, and shoot on all three.

    **Predict:** one batch, three cameras in, and a wider BEV grid out. The metric coverage is the default forward and lateral range at half-meter cells.
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
    The script prints input `torch.Size([1, 3, 32, 16, 32])` and output `torch.Size([1, 64, 60, 80])`. Coverage is `0.0m to 40.0m forward` and `-15.0m to 15.0m lateral`, at `0.5m` cells. The cameras are `front, left, right`.
    """))

    cells.append(md("## 7. The wrong depth bin lands in the wrong pillar"))
    cells.append(md("""
    Same pixel and camera as the unproject above. The obvious guess is that the object is closer, so feed in half the camera depth and bin the result with the half-meter grid.

    **Predict:** halving depth slides the pillar toward the car. Putting the true depth back returns the raised point's forward bin.
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

    xmin, xstep = 0.0, 0.5
    ymin, ystep = -15.0, 0.5

    def bin_index(x_value, y_value):
        xi = ((torch.tensor(float(x_value)) - xmin) / xstep).long()
        yi = ((torch.tensor(float(y_value)) - ymin) / ystep).long()
        return int(xi), int(yi)

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
    """))
    cells.append(md("""
    True depth returns `(20.000, 0.000, 1.000)` at `x_idx=40`, `y_idx=30`. Half depth is `8.992` and lands at `(11.000, 0.000, 1.200)`, so `x_idx=22` while `y_idx=30` stays put. Unprojecting again with the true `d` prints `fixed x_idx=40`. Halving depth slid the pillar toward the car.
    """))

    cells.append(md("## 8. Exercises"))
    cells.append(md("""
    Three checks. Each function still raises, so the cell prints a fallback line and calls a reference that uses the real API. Replace the TODO when you want the check to call your function.
    """))
    cells.append(md("""
    **Exercise 1.** Sum the depth probabilities along the depth axis.

    **Predict:** every spatial location sums to 1, and the probability tensor keeps the ten depth bins.
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
    The cell printed `Using reference depth_sum_student (TODO not implemented)` and then `✅`. `probs.shape` is `(2, 10, 8, 16)`.
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
    **Exercise 2.** Return the camera frustum shape for an 8 by 16 feature map, using the module from the shape section.

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
    The cell printed `Using reference frustum_shape_student (TODO not implemented)` and then `✅`. The frustum shape is `(10, 8, 16, 3)`.
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
    **Exercise 3.** Return the BEV output shape for the same identity calibration as the shape section.

    **Predict:** batch 1, 32 channels, and a 20 by 20 grid.
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
    The cell printed `Using reference bev_shape_student (TODO not implemented)` and then `✅`. The BEV shape is `(1, 32, 20, 20)`.
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
    - The flat-ground warp put the road point at IPM X `20.00` and the raised point at `65.00`. The pixels were `(320.00, 181.44)` and `(320.00, 171.20)`.
    - Softmax printed `[0.8668, 0.1173, 0.0159]`, summing to `1.0000`, with expected depth `3.3416`. Flat logits had entropy `3.00` and a `40.0` meter smear. The peaked distribution had entropy `0.0006`, peak probability `100.0`% at `20.9` m, and localization within `1.05` meters.
    - Unproject brought the raised pixel back to `(20.000, 0.000, 1.000)` from depth `17.984`.
    - That point splatted into `x_idx=40`, `y_idx=30` on an `nx=80` by `ny=60` grid. Two rows in that cell summed to `[1.5, 2.0]`.
    - The toy convolution stayed `(1, 1, 5, 5)`. The real encoder stayed `(1, 4, 20, 20)`.
    - The test forward returned `torch.Size([1, 32, 20, 20])`. The 3-camera script returned `torch.Size([1, 64, 60, 80])`.
    - Half the true depth landed at `x_idx=22`. The true depth returned `x_idx=40`. `y_idx=30` did not move.

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
