#!/usr/bin/env python3
"""Regenerate the Module 02 lesson notebook.

Writes ``notebooks/03_hydranet_multitask_learning.ipynb`` next to this course
staging tree. The notebook is the lesson: run it top to bottom. This script
does not execute it.

    python staging/self-driving-ai-course/scripts/build_m02_lesson_notebook.py
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
    # Module 02 — One backbone, several driving jobs

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/notebooks/03_hydranet_multitask_learning.ipynb)

    A camera on a car is asked several questions about the same frame: where the lane markings are, which pixels are drivable, where the vehicles sit, and what the traffic light is doing. Depth comes later. This module keeps a stride-8 feature map for that next step and does not predict depth itself.

    Running a separate network for each question repeats the same early vision. The course code in `modules/02_hydranet/` extracts features once, then branches into four heads. The loss that trains them is `UncertaintyMultiTaskLoss` in `multitask_loss.py`.

    Each section explains one idea, then runs code. **Predict first**, then execute the cell.
    """))

    cells.append(code("""
    import os, subprocess, sys
    from pathlib import Path

    try:
        import torch
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "torch",
             "--index-url", "https://download.pytorch.org/whl/cpu"],
            check=True,
        )
        import torch
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "matplotlib"], check=True)
        import matplotlib
        import matplotlib.pyplot as plt

    import torch.nn as nn
    import torch.nn.functional as F

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
        for p in [start, *start.parents]:
            if (p / "modules" / "02_hydranet" / "heads.py").is_file():
                return p.resolve()
        return start.resolve()

    REPO = find_repo(Path.cwd())
    if not (REPO / "modules" / "02_hydranet" / "heads.py").is_file():
        dest = Path.cwd() / "self-driving-ai-course"
        if not (dest / "modules" / "02_hydranet" / "heads.py").is_file():
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/vvknyn/self-driving-ai-course.git", str(dest)],
                check=True,
            )
        REPO = dest.resolve()

    os.chdir(REPO)
    MOD = REPO / "modules" / "02_hydranet"
    sys.path.insert(0, str(MOD))

    from backbone import HydraNetBackbone
    from heads import (
        FreespaceHead,
        HydraNet,
        LaneSegmentationHead,
        TrafficLightHead,
        VehicleDetectionHead,
    )
    from multitask_loss import UncertaintyMultiTaskLoss

    def nparams(module):
        return sum(p.numel() for p in module.parameters())

    print("Repo:", REPO)
    keep_inline()
    """))

    cells.append(md("## 1. The problem"))
    cells.append(md("""
    One frame has to feed several driving decisions. A lane head wants a full-resolution mask. A vehicle head wants a coarse grid of boxes. A traffic-light head wants a single class for the whole scene. A freespace head wants another mask: which pixels the car may drive on.

    The expensive part is the early convolution, the part that turns RGB into edges and texture. If you clone that trunk once per job, you pay for the same edges three or four times.

    **Predict:** one shared trunk plus the heads should use fewer parameters than a separate trunk for every head. The next sections count them on the repo classes.
    """))

    cells.append(md("## 2. Shared trunk"))
    cells.append(md("""
    `HydraNetBackbone` runs three stride-2 stages and returns a dictionary `p1`, `p2`, `p3`.

    Input for this pass: a synthetic batch `(2, 3, 128, 256)`. There is no image file in this module. The tensor is random, and the shapes do not depend on the pixel values.

    **Predict:** `p1` keeps 32 channels at half resolution, `p2` keeps 64 channels at quarter resolution, and `p3` keeps 128 channels at one eighth. Height 128 divided by 8 is 16. Width 256 divided by 8 is 32.
    """))
    cells.append(code("""
    torch.manual_seed(0)
    images = torch.randn(2, 3, 128, 256)
    backbone = HydraNetBackbone()
    feats = backbone(images)
    print("input", tuple(images.shape))
    for name in ("p1", "p2", "p3"):
        print(name, tuple(feats[name].shape))
    print("p3 height", images.shape[2] // 8, "p3 width", images.shape[3] // 8)
    assert tuple(feats["p1"].shape) == (2, 32, 64, 128)
    assert tuple(feats["p2"].shape) == (2, 64, 32, 64)
    assert tuple(feats["p3"].shape) == (2, 128, 16, 32)
    """))
    cells.append(md("""
    The print says the input is `(2, 3, 128, 256)`. `p1` is `(2, 32, 64, 128)`, `p2` is `(2, 64, 32, 64)`, and `p3` is `(2, 128, 16, 32)`. The last line is `p3 height 16` and `p3 width 32`, which is the input divided by 8. That `p3` tensor is what every head reads, and it is the map stored as `backbone_features` for the next module.
    """))

    cells.append(md("## 3. Three separate models"))
    cells.append(md("""
    Count parameters with the repo classes, not a sketch. Three jobs — lane mask, vehicle grid, traffic light — can each own a full backbone. The shared design keeps one backbone and three heads. The repo then adds a fourth head, freespace, on that same backbone.

    **Predict:** three separate trunks cost about twice one shared trunk plus those three heads, because the backbone is most of the weight and the heads are not free.
    """))
    cells.append(code("""
    backbone_n = nparams(HydraNetBackbone())
    lane_n = nparams(LaneSegmentationHead())
    free_n = nparams(FreespaceHead())
    vehicle_n = nparams(VehicleDetectionHead())
    light_n = nparams(TrafficLightHead())
    separate3 = 3 * backbone_n + lane_n + vehicle_n + light_n
    shared3 = backbone_n + lane_n + vehicle_n + light_n
    separate4 = 4 * backbone_n + lane_n + free_n + vehicle_n + light_n
    hydranet_n = nparams(HydraNet())
    print("backbone", backbone_n)
    print("lane head", lane_n)
    print("freespace head", free_n)
    print("vehicle head", vehicle_n)
    print("traffic light head", light_n)
    print("three separate models", separate3)
    print("one trunk + three heads", shared3)
    print("separate / shared", f"{separate3 / shared3:.4f}")
    print("four separate trunks", separate4)
    print("HydraNet", hydranet_n)
    print("parameters saved", separate4 - hydranet_n)
    assert separate3 > shared3
    assert hydranet_n == shared3 + free_n
    assert separate4 - hydranet_n == 3 * backbone_n
    """))
    cells.append(md("""
    The backbone is `288800` parameters. The lane head is `264417`, the vehicle head is `74117`, and the traffic-light head is `8516`. Three separate models are `1213450` parameters. One trunk plus those three heads is `635850`. The ratio prints as `1.9084`.

    Freespace adds `172161` parameters. Four separate trunks are `1674411`. The real `HydraNet` is `808011`. The line `parameters saved` is `866400`, which is three backbones you do not allocate. The fourth copy of the trunk is the one you keep.
    """))

    cells.append(md("## 4. Heads"))
    cells.append(md("""
    `HydraNet.forward` runs the trunk once, then each head. Lane is the only head that concatenates `p1` and `p2` back in as skip connections. Freespace upsamples `p3` alone. Vehicles stay on the stride-8 grid with 5 channels. Traffic light pools `p3` down to one vector and emits 4 logits. The dict also returns `backbone_features`, which is `p3` itself.

    **Predict:** for the same `(2, 3, 128, 256)` batch, lane and freespace are `(2, 1, 128, 256)`, vehicles are `(2, 5, 16, 32)`, traffic light is `(2, 4)`, and `backbone_features` matches `p3`.
    """))
    cells.append(code("""
    torch.manual_seed(0)
    images = torch.randn(2, 3, 128, 256)
    model = HydraNet()
    preds = model(images)
    for name in ("lane", "freespace", "vehicles", "traffic_light", "backbone_features"):
        print(name, tuple(preds[name].shape))
    assert tuple(preds["lane"].shape) == (2, 1, 128, 256)
    assert tuple(preds["freespace"].shape) == (2, 1, 128, 256)
    assert tuple(preds["vehicles"].shape) == (2, 5, 16, 32)
    assert tuple(preds["traffic_light"].shape) == (2, 4)
    assert tuple(preds["backbone_features"].shape) == (2, 128, 16, 32)
    """))
    cells.append(md("""
    Lane is `(2, 1, 128, 256)` and freespace is the same shape. Vehicles are `(2, 5, 16, 32)`: 5 numbers on each cell of the stride-8 grid. Traffic light is `(2, 4)`. `backbone_features` is `(2, 128, 16, 32)`, the same tensor shape as `p3`. One forward produced all five.
    """))

    cells.append(md("## 5. The conflict"))
    cells.append(md("""
    Two losses on one shared number. The lane target is `w = 1`. The box target is `w = -5`. The losses are `(w - 1)^2` and `(w + 5)^2`. At `w = 0` the lane loss is 1 and the box loss is 25. Sum them and walk downhill with Adam, learning rate `0.05`, for 200 steps.

    **Predict:** the first step moves `w` down, because the box term is larger. The lane loss should rise. The sum's bottom is the midpoint of `1` and `-5`.
    """))
    cells.append(code("""
    torch.manual_seed(0)
    w = nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.Adam([w], lr=0.05)
    print("learning rate", f"{0.05:.2f}")
    print("lane target", f"{1.0:.1f}")
    print("box target", f"{-5.0:.1f}")
    print("closed form w", f"{(1.0 + -5.0) / 2:.4f}")

    def pair_losses(weight):
        return (weight - 1.0) ** 2, (weight + 5.0) ** 2

    lane_hist, box_hist = [], []
    print(f"{'step':>6} {'w':>10} {'lane':>10} {'box':>10}")
    lane, box = pair_losses(w)
    lane_hist.append(lane.item())
    box_hist.append(box.item())
    print(f"{0:6d} {w.item():10.4f} {lane.item():10.4f} {box.item():10.4f}")
    for step in range(1, 201):
        opt.zero_grad()
        lane, box = pair_losses(w)
        (lane + box).backward()
        opt.step()
        lane, box = pair_losses(w)
        lane_hist.append(lane.item())
        box_hist.append(box.item())
        if step in (1, 20, 50, 100, 200):
            print(f"{step:6d} {w.item():10.4f} {lane.item():10.4f} {box.item():10.4f}")

    assert lane_hist[-1] > lane_hist[0]
    assert box_hist[-1] < box_hist[0]
    assert abs(w.item() - (-2.0)) < 0.01

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(lane_hist, label="lane")
    ax.plot(box_hist, label="box")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    plt.tight_layout()
    plt.show()
    """))
    cells.append(md("""
    The targets print as `1.0` and `-5.0`. The closed form for the sum is `w = -2.0000`.

    Step 0 is `w = 0.0000`, lane loss `1.0000`, box loss `25.0000`. After one step, `w` is `-0.0500`, lane loss is `1.1025`, and box loss is `24.5025`. The small loss has already gone up.

    At step 20, `w` is `-0.9598` and the lane loss is `3.8409`. At step 200, `w` is `-1.9999`, lane loss is `8.9996`, and box loss is `9.0004`. The lane loss started at `1.0000` and finished near 9. The box loss started at `25.0000` and finished near 9. The plot shows the lane curve rising and the box curve falling until they meet. The summed loss found the compromise. The lane job is worse than when the knob was still at zero.
    """))
    cells.append(md("""
    The same imbalance shows up on the real stem. One batch, seed 42, the shapes from `break_it_fix_it.py`: images `(4, 3, 128, 256)`, sparse lane targets, vehicle targets scaled by 5. Measure the L2 norm of the gradient of each raw loss on `backbone.stem[0].weight`, and divide by the sum of those two norms.

    **Predict:** the vehicle share of that stem gradient is above 90%.
    """))
    cells.append(code("""
    torch.manual_seed(42)
    drill_images = torch.randn(4, 3, 128, 256)
    drill_lanes = (torch.rand(4, 1, 128, 256) > 0.95).float()
    drill_vehicles = torch.randn(4, 5, 16, 32) * 5.0
    drill_model = HydraNet()
    drill_preds = drill_model(drill_images)
    drill_lane = F.binary_cross_entropy_with_logits(drill_preds["lane"], drill_lanes)
    drill_veh = F.mse_loss(drill_preds["vehicles"], drill_vehicles)
    stem_w = drill_model.backbone.stem[0].weight
    g_lane = torch.autograd.grad(drill_lane, stem_w, retain_graph=True)[0].norm().item()
    g_veh = torch.autograd.grad(drill_veh, stem_w, retain_graph=True)[0].norm().item()
    lane_share = 100.0 * g_lane / (g_lane + g_veh)
    veh_share = 100.0 * g_veh / (g_lane + g_veh)
    print("lane loss", f"{drill_lane.item():.4f}")
    print("vehicle loss", f"{drill_veh.item():.4f}")
    print("lane stem share", f"{lane_share:.2f}")
    print("vehicle stem share", f"{veh_share:.2f}")
    assert veh_share > 90.0
    """))
    cells.append(md("""
    Lane loss is `0.6553`. Vehicle loss is `25.0871`. The lane stem share is `0.34`. The vehicle stem share is `99.66`. Almost the entire update to the first convolution comes from the vehicle loss. The lane head is attached to that convolution, and it is barely moving it.
    """))

    cells.append(md("## 6. Uncertainty weighting"))
    cells.append(md("""
    Kendall, Gal, and Cipolla treat each task residual as Gaussian with one noise level $\\sigma$ for the whole task (homoscedastic: one $\\sigma$ per task, not one per pixel). The negative log likelihood for a residual whose squared size is the task loss $L$ is

    $$\\frac{1}{2\\sigma^2} L + \\log \\sigma$$

    plus a constant. Write $s = \\log \\sigma^2$. Then $\\sigma^2 = e^{s}$, so $1/\\sigma^2 = e^{-s}$ and $\\log \\sigma = s/2$. The same expression becomes

    $$\\frac{1}{2} e^{-s} L + \\frac{1}{2} s.$$

    `UncertaintyMultiTaskLoss` stores `log_vars` as $s$ and adds that term for every task. The weight it reports, `get_task_weights()`, is $\\tfrac{1}{2} e^{-s}$, which is $1/(2\\sigma^2)$.

    If $L$ is huge and $s$ stays 0, the first term dominates and the optimizer increases $s$, which shrinks $e^{-s}$. The $+\\tfrac{1}{2}s$ term grows as $s$ grows, so $s$ cannot run off to infinity. The derivative of one term with respect to $s$ is $\\tfrac{1}{2} - \\tfrac{1}{2} e^{-s} L$. It is zero when $s = \\log L$.

    The derivation is for a squared residual. This repo still passes lane BCE, freespace BCE, vehicle MSE, and traffic-light cross-entropy through that same formula. For the classification heads, $s$ is a learned task weight.

    **Predict:** with lane loss `0.3`, box loss `40`, and both $s = 0$, the box term is most of the total. At box $s = 4$ that term should drop, and `UncertaintyMultiTaskLoss` should print the same total as the hand formula.
    """))
    cells.append(code("""
    lane_L = torch.tensor(0.3)
    box_L = torch.tensor(40.0)
    print(f"{'s':>6} {'L':>8} {'weight':>10} {'penalty':>10} {'term':>12}")
    for s_box in (0.0, 2.0, 4.0):
        s = torch.tensor([0.0, s_box])
        hand_terms = []
        for i, loss_i in enumerate((lane_L, box_L)):
            weight = 0.5 * torch.exp(-s[i])
            penalty = 0.5 * s[i]
            term = weight * loss_i + penalty
            hand_terms.append(term)
            print(
                f"{s[i].item():6.1f} {loss_i.item():8.1f} {weight.item():10.6f} "
                f"{penalty.item():10.4f} {term.item():12.6f}"
            )
        loss_mod = UncertaintyMultiTaskLoss(2)
        with torch.no_grad():
            loss_mod.log_vars.copy_(s)
        repo_total = loss_mod([lane_L, box_L])
        hand_total = hand_terms[0] + hand_terms[1]
        print("repo total", f"{repo_total.item():.6f}", "hand total", f"{hand_total.item():.6f}")
        assert torch.allclose(repo_total, hand_total, atol=1e-5)
        sigma = torch.exp(0.5 * s[1])
        kendall = box_L / (2 * sigma ** 2) + torch.log(sigma)
        repo_box = hand_terms[1]
        print("kendall box", f"{kendall.item():.6f}", "repo box", f"{repo_box.item():.6f}")
        assert torch.allclose(kendall, repo_box, atol=1e-5)

    fresh = UncertaintyMultiTaskLoss(4)
    print("initial weights", [f"{w:.4f}" for w in fresh.get_task_weights()])

    L = 40.0
    s_star = torch.tensor(L).log()
    weight_star = 0.5 * torch.exp(-s_star)
    penalty_star = 0.5 * s_star
    term_star = weight_star * L + penalty_star
    print("s*", f"{s_star.item():.4f}")
    print("weight*", f"{weight_star.item():.6f}")
    print("penalty*", f"{penalty_star.item():.4f}")
    print("term*", f"{term_star.item():.4f}")
    for s_value in (0.0, 4.0, float(s_star)):
        deriv = 0.5 - 0.5 * torch.exp(torch.tensor(-s_value)) * L
        print("d/ds", f"{s_value:.4f}", f"{deriv.item():.4f}")
    """))
    cells.append(md("""
    At $s = 0$ the lane row is weight `0.500000`, penalty `0.0000`, term `0.150000`. The box row is weight `0.500000`, penalty `0.0000`, term `20.000000`. Repo total and hand total both print `20.150000`. The box term is the total.

    At box $s = 2$ the box weight is `0.067668`, the penalty is `1.0000`, and the term is `3.706706`. The repo total is `3.856706`. The line `kendall box` matches `repo box` at `3.706706`: $L/(2\\sigma^2) + \\log \\sigma$ and the repo term are the same number.

    At box $s = 4$ the box weight is `0.009158`, the penalty is `2.0000`, and the term is `2.366313`. The repo total is `2.516313`. Raising $s$ cut the box term from `20.000000` to `2.366313`, and the penalty is the part that climbed to `2.0000`.

    A fresh `UncertaintyMultiTaskLoss(4)` prints initial weights `0.5000` `0.5000` `0.5000` `0.5000`, because $s = 0$ and $\\tfrac{1}{2} e^{0} = 0.5$.

    For $L = 40$ the stationary $s$ prints as `3.6889`, the weight as `0.012500`, the penalty as `1.8444`, and the term as `2.3444`. The derivative is `-19.5000` at $s = 0.0000$ (increase $s$), `0.1337` at $s = 4.0000$ (decrease $s$), and `0.0000` at $s = 3.6889$.
    """))

    cells.append(md("## 7. A run where both losses move"))
    cells.append(md("""
    Put the weights from section 6 on the same stem gradient as section 5. Set `log_vars` to `0` and `3.5` (the fixed setting in `break_it_fix_it.py`) and measure the stem shares again, on the same forward.

    **Predict:** the lane share rises above `0.34`, and the vehicle share falls below `99.66`. The vehicle share can stay well above half.
    """))
    cells.append(code("""
    balancer = UncertaintyMultiTaskLoss(2)
    with torch.no_grad():
        balancer.log_vars.copy_(torch.tensor([0.0, 3.5]))
    weights = balancer.get_task_weights()
    balanced_total = balancer([drill_lane, drill_veh])
    g_lane_w = torch.autograd.grad(weights[0] * drill_lane, stem_w, retain_graph=True)[0].norm().item()
    g_veh_w = torch.autograd.grad(weights[1] * drill_veh, stem_w, retain_graph=True)[0].norm().item()
    lane_share_w = 100.0 * g_lane_w / (g_lane_w + g_veh_w)
    veh_share_w = 100.0 * g_veh_w / (g_lane_w + g_veh_w)
    print("log_vars", [f"{v:.1f}" for v in balancer.log_vars.detach().tolist()])
    print("lane weight", f"{weights[0]:.4f}")
    print("vehicle weight", f"{weights[1]:.4f}")
    print("balanced total", f"{balanced_total.item():.4f}")
    print("weighted lane share", f"{lane_share_w:.2f}")
    print("weighted vehicle share", f"{veh_share_w:.2f}")
    assert lane_share_w > lane_share
    assert veh_share_w < veh_share
    """))
    cells.append(md("""
    `log_vars` prints `0.0` and `3.5`. The lane weight is `0.5000` and the vehicle weight is `0.0151`. The balanced total is `2.4564`. The weighted lane share is `10.11`. The weighted vehicle share is `89.89`.

    Lane's pull on the stem went from `0.34` to `10.11`. Vehicle's pull went from `99.66` to `89.89`. The vehicle term is still most of the stem gradient. The change is that it no longer owns 99 percent of it.
    """))
    cells.append(md("""
    The one-knob model could not represent both targets. Give each job its own last layer and a shared linear trunk of 16 units. Both targets are linear functions of the same 128 inputs. The lane targets have standard deviation near 1. The box targets are scaled by 4, so their squared error starts near 100. Train 80 Adam steps at learning rate `0.01`, once with a plain sum and once with `UncertaintyMultiTaskLoss(2)`.

    **Predict:** under the repo loss, both the lane MSE and the box MSE are smaller at step 80 than at step 1. The plain sum can also reduce both, because the heads are separate. The box weight should end below its start of `0.5`.
    """))
    cells.append(code("""
    torch.manual_seed(0)
    n_rows, n_in, n_hid = 128, 8, 16
    x_toy = torch.randn(n_rows, n_in)
    y_lane = (x_toy @ torch.randn(n_in, 1)) * 0.3
    y_box = (x_toy @ torch.randn(n_in, 1)) * 4.0
    print("learning rate", f"{1e-2:.2f}")
    print("steps", 80)
    print("lane target std", f"{y_lane.std().item():.4f}")
    print("box target std", f"{y_box.std().item():.4f}")

    class TwoHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Linear(n_in, n_hid)
            self.lane = nn.Linear(n_hid, 1)
            self.box = nn.Linear(n_hid, 1)

        def forward(self, z):
            h = torch.relu(self.trunk(z))
            return self.lane(h), self.box(h)

    def run_two_head(mode, steps=80, lr=1e-2):
        torch.manual_seed(1)
        net = TwoHead()
        loss_mod = UncertaintyMultiTaskLoss(2) if mode == "unc" else None
        params = list(net.parameters())
        if loss_mod is not None:
            params = params + list(loss_mod.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        lane_hist, box_hist = [], []
        for step in range(1, steps + 1):
            opt.zero_grad()
            pred_lane, pred_box = net(x_toy)
            loss_lane = F.mse_loss(pred_lane, y_lane)
            loss_box = F.mse_loss(pred_box, y_box)
            lane_hist.append(loss_lane.item())
            box_hist.append(loss_box.item())
            if step in (1, steps):
                if loss_mod is None:
                    g_l = torch.autograd.grad(loss_lane, net.trunk.weight, retain_graph=True)[0].norm().item()
                    g_b = torch.autograd.grad(loss_box, net.trunk.weight, retain_graph=True)[0].norm().item()
                    share = 100.0 * g_l / (g_l + g_b)
                    weight_txt = "none"
                else:
                    wt = loss_mod.get_task_weights()
                    g_l = torch.autograd.grad(wt[0] * loss_lane, net.trunk.weight, retain_graph=True)[0].norm().item()
                    g_b = torch.autograd.grad(wt[1] * loss_box, net.trunk.weight, retain_graph=True)[0].norm().item()
                    share = 100.0 * g_l / (g_l + g_b)
                    weight_txt = [f"{v:.4f}" for v in wt]
                print(
                    mode,
                    "step", step,
                    "lane", f"{loss_lane.item():.4f}",
                    "box", f"{loss_box.item():.4f}",
                    "lane_share", f"{share:.1f}",
                    "weights", weight_txt,
                )
            total = loss_lane + loss_box if loss_mod is None else loss_mod([loss_lane, loss_box])
            total.backward()
            opt.step()
        return lane_hist, box_hist

    naive_lane, naive_box = run_two_head("naive")
    unc_lane, unc_box = run_two_head("unc")
    assert unc_lane[-1] < unc_lane[0]
    assert unc_box[-1] < unc_box[0]
    assert naive_lane[-1] < naive_lane[0]
    assert naive_box[-1] < naive_box[0]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(naive_lane, label="lane")
    axes[0].plot(naive_box, label="box")
    axes[0].set_title("plain sum")
    axes[1].plot(unc_lane, label="lane")
    axes[1].plot(unc_box, label="box")
    axes[1].set_title("UncertaintyMultiTaskLoss")
    for ax in axes:
        ax.set_xlabel("step")
        ax.set_ylabel("MSE")
        ax.legend()
    plt.tight_layout()
    plt.show()
    """))
    cells.append(md("""
    Lane target std is `1.0617`. Box target std is `13.1843`.

    Plain sum, step 1: lane `1.2728`, box `172.8219`, lane share of the trunk gradient `8.8`. Step 80: lane `0.2427`, box `9.3208`, lane share `6.0`. Both losses fell. The trunk gradient stayed mostly the box task.

    Repo loss, step 1: lane `1.2728`, box `172.8219`, weights `0.5000` and `0.5000`, lane share `8.8`. Step 80: lane `0.1066`, box `11.2656`, weights `1.0520` and `0.3058`, lane share `10.3`. Both losses fell. The box weight moved from `0.5000` to `0.3058` because that loss stayed larger. The lane loss ended lower than the plain sum (`0.1066` against `0.2427`). The box loss ended higher (`11.2656` against `9.3208`). Separate heads are why both numbers can fall at all. The one shared knob in section 5 had no such room. The weighting changed the balance. It did not make the trunk gradient a 50/50 split in 80 steps: lane share is still `10.3`.
    """))
    cells.append(md("""
    Now the real `HydraNet`, one fixed synthetic batch, 10 steps. Optimizer is the one in `train_hydranet.py`: AdamW, learning rate `2e-3`, weight decay `1e-4`, on the network and on `log_vars`. Losses are lane BCE, freespace BCE, vehicle MSE times 10, and traffic-light cross-entropy. The batch does not change between steps, so a drop means the weights are fitting these tensors. `train_hydranet.py` draws a new random batch every step, so its printed losses are not a learning curve on one scene.

    **Predict:** lane BCE and traffic-light cross-entropy drop. The vehicle weight moves only a little in 10 steps at this learning rate. Freespace, which starts near $\\log 2$, barely moves.
    """))
    cells.append(code("""
    torch.manual_seed(0)
    images = torch.randn(2, 3, 128, 256)
    gt_lane = (torch.rand(2, 1, 128, 256) > 0.92).float()
    gt_free = (torch.rand(2, 1, 128, 256) > 0.50).float()
    gt_veh = torch.randn(2, 5, 16, 32)
    gt_tl = torch.randint(0, 4, (2,))

    torch.manual_seed(1)
    model = HydraNet()
    loss_balancer = UncertaintyMultiTaskLoss(4)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_balancer.parameters()),
        lr=2e-3,
        weight_decay=1e-4,
    )
    print("steps", 10)
    print("lr", "2e-3")
    print("weight decay", "1e-4")
    print("init weights", [f"{w:.4f}" for w in loss_balancer.get_task_weights()])
    raw_hist = []
    for step in range(1, 11):
        optimizer.zero_grad()
        preds = model(images)
        task = [
            F.binary_cross_entropy_with_logits(preds["lane"], gt_lane),
            F.binary_cross_entropy_with_logits(preds["freespace"], gt_free),
            F.mse_loss(preds["vehicles"], gt_veh) * 10.0,
            F.cross_entropy(preds["traffic_light"], gt_tl),
        ]
        total = loss_balancer(task)
        total.backward()
        optimizer.step()
        raw_hist.append([t.item() for t in task])
        ws = [f"{w:.4f}" for w in loss_balancer.get_task_weights()]
        print(
            f"{step:2d}",
            "total", f"{total.item():.4f}",
            "lane", f"{task[0].item():.4f}",
            "free", f"{task[1].item():.4f}",
            "veh", f"{task[2].item():.4f}",
            "tl", f"{task[3].item():.4f}",
            "w", ws,
        )

    assert raw_hist[-1][0] < raw_hist[0][0]
    assert raw_hist[-1][2] < raw_hist[0][2]
    assert raw_hist[-1][3] < raw_hist[0][3]
    assert abs(raw_hist[0][1] - raw_hist[-1][1]) < 0.1
    """))
    cells.append(md("""
    Initial weights are `0.5000` four times.

    Step 1: total `6.9986`, lane `0.7817`, freespace `0.6935`, vehicle `11.1095`, traffic light `1.4125`, weights `0.5010` `0.5010` `0.4990` `0.4990`.

    Step 2 jumps: total `10.1012`, vehicle `17.6377`. The walk is not monotone.

    Step 10: total `4.6150`, lane `0.3103`, freespace `0.6797`, vehicle `7.7280`, traffic light `0.6408`, weights `0.5101` `0.5101` `0.4907` `0.4978`.

    Lane BCE fell from `0.7817` to `0.3103`. Traffic light fell from `1.4125` to `0.6408`. Vehicle MSE times 10 fell from `11.1095` to `7.7280` after that spike. Freespace moved from `0.6935` to `0.6797`. The vehicle weight moved from `0.5000` at the start to `0.4907` at step 10. Ten steps at learning rate `2e-3` only begin to retune $s$. These targets are synthetic noise. A lower loss here means the batch was memorized, not that the car can see a lane.
    """))

    cells.append(md("## 8. Exercises"))
    cells.append(md("""
    Three checks taken from `modules/02_hydranet/tests/test_hydranet.py`. Leave each `TODO` in place to use the reference. Replace it if you want the check to call your function.
    """))
    cells.append(md("""
    **Exercise — output shapes.** Return the five output shapes for a batch, matching `test_output_head_shapes`. Keys: `lane`, `freespace`, `vehicles`, `traffic_light`, `backbone_features`.
    """))
    cells.append(code("""
    def expected_shapes_student(batch, height, width):
        # TODO: dict of output name -> shape tuple
        raise NotImplementedError

    def expected_shapes_reference(batch, height, width):
        return {
            "lane": (batch, 1, height, width),
            "freespace": (batch, 1, height, width),
            "vehicles": (batch, 5, height // 8, width // 8),
            "traffic_light": (batch, 4),
            "backbone_features": (batch, 128, height // 8, width // 8),
        }

    def get_shapes_fn():
        try:
            expected_shapes_student(2, 128, 256)
        except NotImplementedError:
            print("Using the test's expected shapes (TODO not implemented)")
            return expected_shapes_reference
        print("Using your expected_shapes")
        return expected_shapes_student

    shapes_fn = get_shapes_fn()
    torch.manual_seed(0)
    live = HydraNet()(torch.randn(2, 3, 128, 256))
    shapes = shapes_fn(2, 128, 256)
    for key, expected in shapes.items():
        got = tuple(live[key].shape)
        print(key, got)
        assert got == tuple(expected)
    print("✅ correct: head shapes match the live HydraNet")
    """))
    cells.append(md("""
    The check prints `✅`. The live forward is lane `(2, 1, 128, 256)`, freespace `(2, 1, 128, 256)`, vehicles `(2, 5, 16, 32)`, traffic light `(2, 4)`, and `backbone_features` `(2, 128, 16, 32)`. The reference is used because the `TODO` still raises.
    """))
    cells.append(md("""
    <details><summary>Solution</summary>

    ```python
    return {
        "lane": (batch, 1, height, width),
        "freespace": (batch, 1, height, width),
        "vehicles": (batch, 5, height // 8, width // 8),
        "traffic_light": (batch, 4),
        "backbone_features": (batch, 128, height // 8, width // 8),
    }
    ```

    </details>
    """))
    cells.append(md("""
    **Exercise — stem gradient.** `test_gradient_flow_to_shared_trunk` sums the traffic-light logits, calls `backward`, and checks that `backbone.stem[0].weight.grad` exists and is finite. Return that boolean.
    """))
    cells.append(code("""
    def traffic_light_reaches_stem_student(model, images):
        # TODO: backward on traffic_light only; return True if stem grad is finite
        raise NotImplementedError

    def traffic_light_reaches_stem_reference(model, images):
        preds = model(images)
        model.zero_grad()
        preds["traffic_light"].sum().backward()
        grad = model.backbone.stem[0].weight.grad
        return grad is not None and bool(torch.isfinite(grad).all())

    def get_stem_fn():
        try:
            traffic_light_reaches_stem_student(None, None)
        except NotImplementedError:
            print("Using the test's stem check (TODO not implemented)")
            return traffic_light_reaches_stem_reference
        print("Using your stem check")
        return traffic_light_reaches_stem_student

    stem_fn = get_stem_fn()
    torch.manual_seed(0)
    ex_model = HydraNet()
    ex_images = torch.randn(2, 3, 128, 256)
    ok = stem_fn(ex_model, ex_images)
    grad = ex_model.backbone.stem[0].weight.grad
    print("stem grad finite", ok)
    print("stem grad shape", tuple(grad.shape))
    print("stem grad norm", f"{grad.norm().item():.6f}")
    assert ok is True
    print("✅ correct: traffic-light loss reaches the shared stem")
    """))
    cells.append(md("""
    The check prints `✅`. `stem grad finite` is `True`. The gradient shape is `(32, 3, 3, 3)`, the stem convolution's weight, and its norm is `0.190363`. Traffic-light pooling still lets that loss reach the trunk. The reference is used because the `TODO` still raises.
    """))
    cells.append(md("""
    <details><summary>Solution</summary>

    ```python
    preds = model(images)
    model.zero_grad()
    preds["traffic_light"].sum().backward()
    grad = model.backbone.stem[0].weight.grad
    return grad is not None and bool(torch.isfinite(grad).all())
    ```

    </details>
    """))
    cells.append(md("""
    **Exercise — uncertainty total.** Implement one call of the repo formula: for each task, $\\tfrac{1}{2} e^{-s} L + \\tfrac{1}{2} s$, summed. `test_uncertainty_loss_properties` builds `UncertaintyMultiTaskLoss(4)` on losses `1, 2, 3, 4` and checks that `log_vars.grad` has shape `(4,)`.
    """))
    cells.append(code("""
    def uncertainty_total_student(task_losses, log_vars):
        # TODO: sum over tasks of 0.5 * exp(-s) * L + 0.5 * s
        raise NotImplementedError

    def uncertainty_total_reference(task_losses, log_vars):
        total = task_losses[0] * 0.0
        for i, loss in enumerate(task_losses):
            s = log_vars[i]
            total = total + 0.5 * torch.exp(-s) * loss + 0.5 * s
        return total

    def get_loss_fn():
        try:
            uncertainty_total_student([torch.tensor(1.0)], torch.zeros(1))
        except NotImplementedError:
            print("Using the repo formula (TODO not implemented)")
            return uncertainty_total_reference
        print("Using your uncertainty total")
        return uncertainty_total_student

    loss_fn = get_loss_fn()
    task_losses = [torch.tensor(float(v)) for v in (1.0, 2.0, 3.0, 4.0)]
    hand = loss_fn(task_losses, torch.zeros(4))
    repo = UncertaintyMultiTaskLoss(4)
    repo_total = repo(task_losses)
    print("hand total", f"{hand.item():.4f}")
    print("repo total", f"{repo_total.item():.4f}")
    assert torch.allclose(hand, repo_total, atol=1e-5)
    repo_total.backward()
    print("log_vars.grad", [f"{g:.4f}" for g in repo.log_vars.grad.tolist()])
    print("grad shape", tuple(repo.log_vars.grad.shape))
    assert tuple(repo.log_vars.grad.shape) == (4,)
    print("✅ correct: uncertainty total matches UncertaintyMultiTaskLoss")
    """))
    cells.append(md("""
    The check prints `✅`. Hand total and repo total are both `5.0000` (half of `1+2+3+4`, because every weight starts at `0.5`). `log_vars.grad` is `0.0000` `-0.5000` `-1.0000` `-1.5000`, and `grad shape` is `(4,)`. The zero on the first task is $\\tfrac{1}{2} - \\tfrac{1}{2} \\cdot 1$. The reference is used because the `TODO` still raises.
    """))
    cells.append(md("""
    <details><summary>Solution</summary>

    ```python
    total = task_losses[0] * 0.0
    for i, loss in enumerate(task_losses):
        s = log_vars[i]
        total = total + 0.5 * torch.exp(-s) * loss + 0.5 * s
    return total
    ```

    </details>
    """))

    cells.append(md("## 9. Recap"))
    cells.append(md("""
    - One trunk. On a batch `(2, 3, 128, 256)` the pyramid is `p1` `(2, 32, 64, 128)`, `p2` `(2, 64, 32, 64)`, `p3` `(2, 128, 16, 32)`.
    - Four heads from that forward: lane and freespace `(2, 1, 128, 256)`, vehicles `(2, 5, 16, 32)`, traffic light `(2, 4)`, plus `backbone_features` equal to `p3`.
    - Three separate models are `1213450` parameters. One trunk plus those three heads is `635850`, ratio `1.9084`. The full `HydraNet` is `808011` instead of `1674411`.
    - A shared knob with losses `(w-1)^2` and `(w+5)^2` walks to `w = -1.9999`. Lane loss goes from `1.0000` to `8.9996`. Box loss goes from `25.0000` to `9.0004`.
    - On the real stem, that same kind of scale gap gives lane `0.34` percent of the gradient and vehicle `99.66` percent. Setting the vehicle log-variance to `3.5` moves those shares to `10.11` and `89.89`.
    - The repo loss is $\\tfrac{1}{2} e^{-s} L + \\tfrac{1}{2} s$ with $s = \\log \\sigma^2$. For losses `0.3` and `40` at $s = 0$, the total is `20.150000`, and the box term is `20.000000` of it. At the stationary box noise, $s = 3.6889$.
    - With separate heads, both toy losses fall under that loss: lane `1.2728` to `0.1066`, box `172.8219` to `11.2656`. Ten steps of `HydraNet` on one synthetic batch move lane BCE from `0.7817` to `0.3103` and barely move the vehicle weight, `0.5000` to `0.4907`.

    ### Paper
    Kendall, Gal, and Cipolla, *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics*, CVPR 2018. [arXiv:1705.07115](https://arxiv.org/abs/1705.07115)
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
    out = Path(__file__).resolve().parents[1] / "notebooks" / "03_hydranet_multitask_learning.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    nbformat.write(nb, out)
    print(f"Wrote {out} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
