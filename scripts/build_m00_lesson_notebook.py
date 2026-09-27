#!/usr/bin/env python3
"""Regenerate the Module 00 lesson notebook.

Writes ``notebooks/00_driving_ml_gym.ipynb`` next to this course staging tree.
The notebook is the lesson: run it top to bottom. This script does not execute it.

    python staging/self-driving-ai-course/scripts/build_m00_lesson_notebook.py
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
    # Module 00 — Teaching a computer to recognize road scenes

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/notebooks/00_driving_ml_gym.ipynb)

    A camera on a car records a grid of colored pixels. Your job in this notebook is to turn a small camera **patch** into one of four labels: open road, lead vehicle, pedestrian, or lane marking.

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
        # train.py and the reference gallery select a file-only backend on import.
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
            if (p / "modules" / "00_ml_gym").is_dir():
                return p.resolve()
        return start.resolve()

    REPO = find_repo(Path.cwd())
    if not (REPO / "modules" / "00_ml_gym").is_dir():
        dest = Path.cwd() / "self-driving-ai-course"
        if not (dest / "modules" / "00_ml_gym").is_dir():
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
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision",
                        "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)

    sys.path.insert(0, str(REPO / "modules" / "00_ml_gym"))
    sys.path.insert(0, str(REPO / "modules"))

    from dataset import CLASS_NAMES, DrivingPatchDataset, get_dataloaders
    from config import TrainConfig

    DATA_DIR = REPO / "data" / "m00_sample"
    print("Repo:", REPO)
    print("Classes:", CLASS_NAMES)
    keep_inline()
    """))

    cells.append(md("## 1. The problem"))
    cells.append(md("""
    Real stacks run scene classification under tight latency limits. Here we use a tiny checked-in dataset so you can train on CPU in minutes and inspect every mistake.

    Below: one real training crop per class from `data/m00_sample/`.
    """))
    cells.append(code("""
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.5))
    for ax, name in zip(axes, CLASS_NAMES):
        row = next(r for r in DrivingPatchDataset(DATA_DIR, "train").rows if r["class_name"] == name)
        from PIL import Image
        img = np.array(Image.open(DATA_DIR / row["filename"]).convert("RGB")) / 255.0
        ax.imshow(img)
        ax.set_title(name)
        ax.axis("off")
        print(name, "shape (H,W,C):", img.shape)
    plt.tight_layout()
    plt.show()
    """))

    cells.append(md("## 2. An image is numbers"))
    cells.append(md("""
    An **image** is a table of numbers. A color patch has height, width, and three **channels** (red, green, blue). After scaling 0–255 pixels to floats in `[0, 1]`, how many numbers are in one crop?

    **Predict:** the shape `(H, W, C)` and the total count before you run the next cell.
    """))
    cells.append(code("""
    from PIL import Image

    path = DATA_DIR / "train_clear_road_000.png"
    hwc = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    print("shape (H, W, C):", hwc.shape)
    print("total numbers:", int(np.prod(hwc.shape)))
    print("top-left pixel [R,G,B]:", np.round(hwc[0, 0], 3))
    """))
    cells.append(md("""
    The print says the crop is 64 by 64 with 3 colors, and `total numbers` is 12288. That is `64 × 64 × 3`. The top-left pixel is three of those numbers, one per color, already scaled into 0–1: `[0.278, 0.239, 0.278]`. Nothing in that triple is a class. The class lives in `labels.csv`, beside the file.
    """))
    cells.append(md("""
    PyTorch wants **channels first**: color, then height, then width. A **batch** is a stack of crops along a new leading axis. The next cell draws the same crop and prints the tensor the dataloader actually returns.

    **Predict:** the dataloader shape will list the 3 colors before the 64s, not after.
    """))
    cells.append(code("""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    axes[0].imshow(hwc)
    axes[0].set_title("RGB")
    axes[0].axis("off")
    axes[1].imshow(hwc[:, :, 0], cmap="viridis")
    axes[1].set_title("Red channel heatmap")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

    ds = DrivingPatchDataset(DATA_DIR, "train")
    chw, _ = ds[0]
    print("dataloader tensor shape (C,H,W):", tuple(chw.shape))
    """))
    cells.append(md("""
    The heatmap is only the red table. Bright spots are large red values, not a label. The printed dataloader shape is `(3, 64, 64)`: channels first. A batch of these will look like `(B, 3, 64, 64)`. `B` is how many crops the model scores in one step.
    """))

    cells.append(md("## 3. The dumbest classifier"))
    cells.append(md("""
    Before any neural net: always predict the **most common** class in the training set. That needs no weights and no GPU.

    **Recall** for a class is: of the images that really are that class, what fraction did we name correctly?

    **Predict:** if we always say `clear_road`, what is pedestrian recall on the validation set?
    """))
    cells.append(code("""
    import csv
    from collections import Counter

    train_rows = [r for r in csv.DictReader(open(DATA_DIR / "labels.csv")) if r["split"] == "train"]
    val_rows = [r for r in csv.DictReader(open(DATA_DIR / "labels.csv")) if r["split"] == "val"]
    train_counts = Counter(int(r["label"]) for r in train_rows)
    val_counts = Counter(int(r["label"]) for r in val_rows)
    majority = max(train_counts, key=train_counts.get)

    print("train size:", len(train_rows), "class counts:", {CLASS_NAMES[k]: v for k, v in sorted(train_counts.items())})
    print("val size:", len(val_rows), "class counts:", {CLASS_NAMES[k]: v for k, v in sorted(val_counts.items())})
    print("always predict:", CLASS_NAMES[majority])

    val_labels = [int(r["label"]) for r in val_rows]
    baseline_acc = sum(l == majority for l in val_labels) / len(val_labels)
    ped_support = val_counts[2]
    ped_recall = sum((l == 2 and majority == 2) for l in val_labels) / ped_support
    print("val accuracy:", round(baseline_acc, 4))
    print("pedestrian recall:", round(ped_recall, 4))
    print("training pedestrians:", train_counts[2])
    print("validation pedestrians:", val_counts[2])
    """))
    cells.append(md("""
    Validation accuracy is 0.4762. That is just the 10 clear-road crops out of 21 validation crops. Pedestrian recall is 0.0. The rule never says pedestrian, so it misses all 3 validation pedestrians, and it also misses every lead vehicle and every lane marking. A score near one half can still mean "we do not see people." Hold onto the printed counts: 6 training pedestrians and 3 validation pedestrians. Those two small numbers run the rest of the lesson.
    """))

    cells.append(md("## 4. A model with knobs"))
    cells.append(md("""
    A **classifier** maps the 12288 input numbers to **four scores** (one per class). Those raw scores are **logits**. The simplest learnable map is a linear layer: one weight per pixel per class, plus a bias — a single matrix multiply (see the `total numbers` printed in §2).

    **Predict:** with random weights, will the four scores look similar or wildly different?
    """))
    cells.append(code("""
    import torch.nn as nn

    torch.manual_seed(SEED)
    flat_dim = 3 * 64 * 64
    linear = nn.Linear(flat_dim, 4)
    img, label = ds[0]
    with torch.no_grad():
        scores = linear(img.flatten())
    print("true class:", CLASS_NAMES[int(label)])
    print("scores:", [round(x, 3) for x in scores.tolist()])
    print("weights shape:", tuple(linear.weight.shape))
    """))
    cells.append(md("""
    The four scores are bunched near zero: `0.132`, `0.181`, `-0.021`, `-0.114`. Random weights do not yet prefer a class, including the true one (`clear_road`). The weight matrix is `(4, 12288)`: four classes, one weight for every input number. That is a lot of knobs, and none of them know about edges yet.
    """))

    cells.append(md("## 5. From four scores to a learning signal"))
    cells.append(md("""
    **From four scores to a learning signal**

    The network outputs four numbers per image, say `[2.0, 1.0, 0.1, -1.0]`. Bigger means "looks more like this class." The true label is *pedestrian* (index 2). To learn, we need one number that says how bad this output is.

    **First try: count mistakes.** The highest score is at index 0, not 2, so loss = 1. The problem: nudge any weight a little and the scores shift a little, but index 0 still wins, so the loss stays 1. The loss is flat, and a flat loss gives no hint which way to move. **We need a loss that changes smoothly when scores change.**

    **Second try: turn scores into probabilities.** We need: every value positive, all values summing to 1, bigger score giving bigger probability, and smooth changes. Dividing each score by the total fails because -1.0 would give a negative "probability." Exponentiate first: e^z is always positive, always increasing, and smooth. Then divide by the total. That is **softmax**. It isn't arbitrary; it's the simplest thing that meets all four requirements.
    """))
    cells.append(code("""
    z = torch.tensor([2.0, 1.0, 0.1, -1.0])
    print("example logits z:", [round(x, 1) for x in z.tolist()])
    print("true class index: 2 (pedestrian)")
    """))
    cells.append(md("""
    **Compute softmax** on the scores just printed. The true class is index 2, and it is not the largest score.
    """))
    cells.append(code("""
    import torch.nn.functional as F

    p = z.exp() / z.exp().sum()
    print("softmax p:", [round(x, 3) for x in p.tolist()])
    print("probability on true class (index 2):", round(p[2].item(), 3))
    """))
    cells.append(md("""
    **Try this:** add 10 to every score. `p` doesn't change, because e^(z+10) = e^10 * e^z and the e^10 cancels. **Only differences between scores matter.**
    """))
    cells.append(code("""
    z_shift = z + 10
    p2 = z_shift.exp() / z_shift.exp().sum()
    print("same p?", torch.allclose(p, p2))
    """))
    cells.append(md("""
    The line `same p? True` is that check: adding 10 to every score left the probabilities unchanged.

    **The loss.** Pedestrian got 0.095. Why -log p and not 1-p? Across the whole dataset, the chance the model gets every label right is the product p1*p2*...*pN. Multiplying many numbers below 1 collapses toward 0. The log turns the product into a sum, and the minus turns maximize into minimize. So **cross-entropy** just means "make the true labels as likely as possible", in a form a computer can handle. Feel its shape: for several values of p, compare -log p with 1-p (printed below).
    """))
    cells.append(code("""
    print("p     -log(p)   1-p")
    for p_val in [0.9, 0.5, 0.095, 0.01]:
        p_t = torch.tensor(p_val)
        print(f"{p_val:.3f}  {-torch.log(p_t):.2f}     {1-p_val:.3f}")
    """))
    cells.append(md("""
    The log punishes confident mistakes hard.

    **The payoff.** Ask PyTorch for the gradient: `F.cross_entropy(...).backward()` → gradient ≈ **p minus 1 at the true class**. Read it as instructions: training subtracts the gradient, so the pedestrian score goes up by an amount proportional to how wrong it was, and every other score goes down in proportion to how much probability it stole. **That one line is the whole learning signal of a classifier.**
    """))
    cells.append(code("""
    z_grad = torch.tensor([2.0, 1.0, 0.1, -1.0], requires_grad=True)
    F.cross_entropy(z_grad.unsqueeze(0), torch.tensor([2])).backward()
    print("z.grad:", [round(x, 3) for x in z_grad.grad.tolist()])
    """))

    cells.append(md("## 6. Learning = walking downhill"))
    cells.append(md("""
    **Gradient descent** adjusts one **knob** in the direction that lowers the loss. The **learning rate** is step size. Too large and you overshoot; watch the loss explode.

    **Predict:** we minimize `(w - 3)²` starting at `w = 0`, once with a modest learning rate and once with an oversized one. Which run ends near 3?
    """))
    cells.append(code("""
    def gd_demo(lr, steps=30):
        w = torch.tensor([0.0], requires_grad=True)
        losses = []
        for _ in range(steps):
            loss = (w - 3) ** 2
            losses.append(loss.item())
            loss.backward()
            with torch.no_grad():
                w -= lr * w.grad
            w.grad.zero_()
        return losses, w.item()

    for lr in [0.1, 1.2]:
        losses, w = gd_demo(lr)
        print(f"lr={lr} final w={w:.3f} last loss={losses[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(8, 3))
    for lr in [0.1, 1.2]:
        losses, _ = gd_demo(lr)
        capped = [min(v, 1e4) for v in losses]
        ax.plot(capped, label=f"lr={lr}")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (capped at 10000 for the plot)")
    ax.legend()
    ax.set_title("One-knob loss vs step")
    plt.tight_layout()
    plt.show()
    print("plot y-axis cap:", 10000)
    """))
    cells.append(md("""
    Learning rate 0.1 ends at `w = 2.996` with last loss `0.000`. It walked to the bottom. Learning rate 1.2 ends at `w = -72601.445` with last loss `2689491968.000`. Each step jumped over the valley and landed farther away. The plot cuts that explosion off at 10000 so the calm curve is still visible. The printed last loss is the real one. Step size is the whole story of this picture.
    """))
    cells.append(md("""
    An **epoch** is one full pass through the training crops. Now train the **linear** classifier for a few epochs on real crops. The learning rate is much smaller than the one that blew up.

    **Predict:** will training loss fall on every epoch? Separately: will validation accuracy rise above the always-clear_road accuracy from §3?
    """))
    cells.append(code("""
    import torch.optim as optim
    from losses import cross_entropy_loss
    from metrics import accuracy, per_class_recall

    base_lr = 0.001
    linear_epochs = 12
    print("linear learning rate:", base_lr)
    print("linear epochs:", linear_epochs)
    print("try-this multiplier:", 50)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_loader, val_loader = get_dataloaders(DATA_DIR, batch_size=16, seed=SEED)
    print("batch size:", 16)
    model_lin = nn.Linear(flat_dim, 4)
    opt = optim.SGD(model_lin.parameters(), lr=base_lr)
    lin_losses = []
    for _ in range(linear_epochs):
        total = 0.0
        n = 0
        for imgs, tgts in train_loader:
            opt.zero_grad()
            logits = model_lin(imgs.flatten(1))
            loss = cross_entropy_loss(logits, tgts)
            loss.backward()
            opt.step()
            total += loss.item() * len(tgts)
            n += len(tgts)
        lin_losses.append(total / n)
    print("linear epoch losses:", [f"{x:.3f}" for x in lin_losses])
    print("strictly decreasing?", all(lin_losses[i] > lin_losses[i + 1] for i in range(len(lin_losses) - 1)))

    model_lin.eval()
    lin_preds, lin_tgts = [], []
    with torch.no_grad():
        for imgs, tgts in val_loader:
            lin_preds.append(model_lin(imgs.flatten(1)).argmax(dim=-1))
            lin_tgts.append(tgts)
    lin_preds = torch.cat(lin_preds)
    lin_tgts = torch.cat(lin_tgts)
    print("linear val accuracy:", f"{accuracy(lin_preds, lin_tgts):.4f}")
    print("linear val recall:", {CLASS_NAMES[i]: f"{r:.4f}" for i, r in enumerate(per_class_recall(lin_preds, lin_tgts, 4))})
    print("linear val prediction counts:", {CLASS_NAMES[i]: int(c) for i, c in enumerate(torch.bincount(lin_preds, minlength=4).tolist())})

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(1, len(lin_losses) + 1), lin_losses, marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean cross-entropy")
    ax.set_title("Linear classifier, learning rate 0.001")
    plt.tight_layout()
    plt.show()
    """))
    cells.append(md("""
    The loss falls on every epoch: `strictly decreasing? True`, from `1.254` to `1.011`. The plot is a steady walk downhill. Validation accuracy is still `0.4762`, and the prediction counts are 21 for `clear_road` and 0 for the other three classes. That is the same rule as §3. A lower training loss only means the scores moved in a way these training crops liked. Twelve gentle epochs were not enough for a flat list of pixels to separate a person from asphalt.
    """))
    cells.append(md("""
    **Try this.** Use 50 times the learning rate just printed, for the same number of epochs.

    **Predict:** a smooth decline, or a bounce like learning rate 1.2 on the one-knob curve?
    """))
    cells.append(code("""
    bounce_lr = base_lr * 50
    print("base learning rate:", base_lr)
    print("50x learning rate:", bounce_lr)
    print("epochs:", linear_epochs)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_loader, _ = get_dataloaders(DATA_DIR, batch_size=16, seed=SEED)
    model_bounce = nn.Linear(flat_dim, 4)
    opt = optim.SGD(model_bounce.parameters(), lr=bounce_lr)
    bounce_losses = []
    for _ in range(linear_epochs):
        total = 0.0
        n = 0
        for imgs, tgts in train_loader:
            opt.zero_grad()
            logits = model_bounce(imgs.flatten(1))
            loss = cross_entropy_loss(logits, tgts)
            loss.backward()
            opt.step()
            total += loss.item() * len(tgts)
            n += len(tgts)
        bounce_losses.append(total / n)
    print("bounce epoch losses:", [f"{x:.3f}" for x in bounce_losses])
    print("strictly decreasing?", all(bounce_losses[i] > bounce_losses[i + 1] for i in range(len(bounce_losses) - 1)))
    print("min loss:", f"{min(bounce_losses):.3f}", "max loss:", f"{max(bounce_losses):.3f}")

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(1, len(bounce_losses) + 1), bounce_losses, marker="o", color="C1")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean cross-entropy")
    ax.set_title("Same linear classifier, learning rate 0.05")
    plt.tight_layout()
    plt.show()
    """))
    cells.append(md("""
    `strictly decreasing? False`. The printed losses start at `11.653`, climb as high as `14.579`, and drop as low as `4.524`, then jump again. The first epoch is already huge: the printed loss is an average over the epoch, and a step this large wrecks the weights before the epoch finishes. Same model, same crops, same start. The only change is the step size, and it is the same failure as `lr = 1.2` on `(w - 3)²`: the update overshoots the valley.
    """))

    cells.append(md("## 7. Why a CNN"))
    cells.append(md("""
    A fully connected layer treats every pixel independently. It can memorize brightness, but it does not care whether a bright pixel sits next to a dark one. A **convolution** slides a small **filter** across the image. At each position it multiplies the overlapping pixels by the filter weights and adds them up. One filter can answer "is there an edge here?"

    The filter below is a horizontal-edge detector: negative on the top row, zero in the middle, positive on the bottom row. Its transpose is the vertical-edge detector (left versus right). We will apply both by hand to three tiny 3×3 patches: a flat patch, a patch with a bright bottom row, and a patch with a bright right column.

    **Predict:** the flat patch should score about 0 on both filters. The bright-bottom patch should score high on the horizontal filter and about 0 on the vertical one. The bright-right patch should do the opposite.
    """))
    cells.append(code("""
    horizontal = torch.tensor([
        [-1., -1., -1.],
        [ 0.,  0.,  0.],
        [ 1.,  1.,  1.],
    ])
    vertical = horizontal.T
    print("horizontal filter:")
    print(horizontal)
    print("vertical filter:")
    print(vertical)

    def response(patch, kernel):
        return float((patch * kernel).sum())

    flat = torch.full((3, 3), 0.5)
    edge_h = torch.tensor([
        [0., 0., 0.],
        [0., 0., 0.],
        [1., 1., 1.],
    ])
    edge_v = torch.tensor([
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
    ])
    print("flat patch   horizontal", response(flat, horizontal), "vertical", response(flat, vertical))
    print("bottom edge  horizontal", response(edge_h, horizontal), "vertical", response(edge_h, vertical))
    print("right edge   horizontal", response(edge_v, horizontal), "vertical", response(edge_v, vertical))
    """))
    cells.append(md("""
    Flat scores `0.0` and `0.0`. The positive and negative weights cancel when every pixel is the same. The bright bottom row scores `3.0` on the horizontal filter and `0.0` on the vertical one. The bright right column scores `0.0` and `3.0` the other way. An edge lined up with the filter becomes a big number. A flat neighborhood, or an edge the filter is not looking for, becomes 0. That is the whole reason to use a convolution: the same small detector runs at every position.
    """))
    cells.append(md("""
    Now slide both filters across a real lane-marking crop. **Padding** is a border of zeros so the 3×3 filter still has a neighborhood on the edge pixels, and the response can stay the same height and width as the crop.

    **Predict:** a lane stripe is a thin bright line. Which filter — horizontal or vertical — will have the larger average response?
    """))
    cells.append(code("""
    from PIL import Image

    lane_path = DATA_DIR / "train_lane_marking_001.png"
    lane = np.array(Image.open(lane_path).convert("RGB"), dtype=np.float32) / 255.0
    red = torch.tensor(lane[:, :, 0])[None, None]
    horiz_k = horizontal.view(1, 1, 3, 3)
    vert_k = vertical.view(1, 1, 3, 3)
    resp_h = F.conv2d(red, horiz_k, padding=1)
    resp_v = F.conv2d(red, vert_k, padding=1)
    print("padding:", 1)
    print("file:", lane_path.name)
    print("red shape:", tuple(red.shape))
    print("horizontal response shape:", tuple(resp_h.shape))
    print("vertical response shape:", tuple(resp_v.shape))
    print("mean abs horizontal:", f"{resp_h.abs().mean().item():.3f}")
    print("mean abs vertical:", f"{resp_v.abs().mean().item():.3f}")
    print("brightest column:", int(lane[:, :, 0].mean(axis=0).argmax()))

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(lane)
    axes[0].set_title("original")
    axes[0].axis("off")
    limit = max(resp_h.abs().max().item(), resp_v.abs().max().item())
    axes[1].imshow(resp_h[0, 0].detach(), cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[1].set_title("horizontal edges")
    axes[1].axis("off")
    axes[2].imshow(resp_v[0, 0].detach(), cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[2].set_title("vertical edges")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()
    """))
    cells.append(md("""
    Padding is 1, and both response shapes are `(1, 1, 64, 64)`, the same height and width as the crop. The brightest column is 34: a thin bright stripe, a few pixels wide, in a field of gray asphalt. Mean absolute response is `0.101` for the horizontal filter and `0.216` for the vertical one. The stripe is a left-right jump in brightness, which is what the vertical filter measures, so that panel is stronger. Along the stripe, the pixels above and below are both bright, so the horizontal filter stays quieter except at the ends and on asphalt speckle. Red and blue in the plot are the two signs of the filter (top versus bottom, or left versus right). The middle color is zero, the same canceling you saw on the flat 3×3 patch.

    A driving stack cares about this because lane paint, vehicle edges, and the outline of a person are edges. A filter that answers "edge here" is a better first question than "what is pixel number 4188?"
    """))
    cells.append(md("""
    The repo model `DrivingClassifier` stacks two convolutions, then pooling, then a linear layer.

    **Stride** is how many pixels the filter jumps. A jump of more than one pixel shrinks the map. **Padding** is the zero border from the last cell. **Pooling** replaces a neighborhood with one summary. Average pooling uses the mean. The last pool in this model averages an entire feature map down to one number.

    **Predict:** the stem jumps by more than one pixel, so height and width should shrink. After that pool, one number is left per channel. That channel count is the number of filters in the previous convolution, and those numbers are what the final linear layer turns into four logits.
    """))
    cells.append(code("""
    H, W, k, s, p = 64, 64, 3, 2, 1
    H_out = (H + 2 * p - k) // s + 1
    print("kernel, stride, padding:", k, s, p)
    print("stem output H, W:", H_out, H_out)

    from model import DrivingClassifier

    cnn = DrivingClassifier(num_classes=4)
    cnn.eval()
    batch = torch.stack([ds[i][0] for i in range(2)])
    with torch.no_grad():
        t = batch
        print("input", tuple(t.shape))
        t = cnn.stem(t)
        print("after stem", tuple(t.shape))
        t = cnn.stage2(t)
        print("after stage2", tuple(t.shape))
        t = cnn.pool(t)
        print("after pool", tuple(t.shape))
        flat = torch.flatten(t, 1)
        print("after flatten", tuple(flat.shape))
        logits = cnn.fc(flat)
        print("logits", tuple(logits.shape))
    """))
    cells.append(md("""
    The formula prints stem height and width `32` and `32`. Stride is 2, so the filter jumps two pixels and the map is cut in half: 64 becomes 32. Padding is 1, which is why the arithmetic is `(64 + 2×1 - 3) / 2 + 1 = 32` rather than a cropped edge.

    The printed shapes, and why each one changes:

    - `input (2, 3, 64, 64)` is a batch of 2 crops, channels first.
    - `after stem (2, 32, 32, 32)`. The stem has 32 filters, so 3 color channels become 32 channels. Each channel is one pattern detector. Stride 2 halves height and width.
    - `after stage2 (2, 64, 16, 16)`. This layer has 64 filters, so 32 channels become 64. Stride 2 halves 32 to 16.
    - `after pool (2, 64, 1, 1)`. Average pooling replaces each 16×16 map with its mean, one number. Sixty-four maps become 64 numbers.
    - `after flatten (2, 64)`, then `logits (2, 4)`. A linear layer turns those 64 numbers into one score per class.

    Those 64 numbers are a summary of the image. They are not 12288 independent pixels anymore. That is the bet a convolution makes: local edges, reused everywhere, then a short list of summaries.
    """))

    cells.append(md("## 8. Train the real model"))
    cells.append(md("""
    The training helpers in this module are `train_epoch` (one pass over the training loader, returns mean loss) and `evaluate` (validation loss, accuracy, and per-class recall). The cells below call those two functions. They do not call `train.main`.

    The loader returns each PNG as stored. It does not flip or recolor crops between epochs, so the 6 training pedestrians are the only pedestrians this model will ever see.

    First try: the epoch count stored on `TrainConfig`, with that config's learning rate and weight decay. Weight decay shrinks the weights a little each step so they do not wander off.

    **Predict:** the training loss will fall. Will the confusion matrix still be a single column of `clear_road`?
    """))
    cells.append(code("""
    from model import DrivingClassifier
    from train import evaluate, train_epoch
    keep_inline()

    cfg = TrainConfig()
    print("TrainConfig epochs:", cfg.epochs)
    print("TrainConfig lr:", cfg.lr)
    print("TrainConfig weight decay:", cfg.weight_decay)
    print("batch size:", 16)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_loader, val_loader = get_dataloaders(DATA_DIR, 16, SEED)
    model_short = DrivingClassifier()
    opt = optim.AdamW(model_short.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    device = torch.device("cpu")
    short_losses = []
    for _ in range(cfg.epochs):
        short_losses.append(train_epoch(model_short, train_loader, opt, cross_entropy_loss, device))
    print("short train losses:", [f"{x:.3f}" for x in short_losses])
    print("short first loss:", f"{short_losses[0]:.3f}", "short last loss:", f"{short_losses[-1]:.3f}")

    _, short_acc, short_rec = evaluate(model_short, val_loader, cross_entropy_loss, device, 4)
    print("short val accuracy:", f"{short_acc:.4f}")
    print("always-clear_road val accuracy:", f"{baseline_acc:.4f}")
    print("short per-class recall:", {CLASS_NAMES[i]: f"{short_rec[i]:.4f}" for i in range(4)})

    cm_short = np.zeros((4, 4), dtype=np.int64)
    model_short.eval()
    pred_counts = np.zeros(4, dtype=np.int64)
    with torch.no_grad():
        for images, targets in val_loader:
            preds = model_short(images).argmax(dim=-1).numpy()
            pred_counts += np.bincount(preds, minlength=4)
            for pred, true in zip(preds, targets.numpy()):
                cm_short[int(true), int(pred)] += 1
    print("class order:", CLASS_NAMES)
    print("short prediction counts:", pred_counts.tolist())
    print("short confusion (rows=true, cols=predicted):", cm_short.tolist())

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_short, cmap="Blues")
    ax.set_xticks(range(4), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(4), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Validation counts after {cfg.epochs} epochs")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.show()
    """))
    cells.append(md("""
    Training loss moves from `1.307` to `0.807`. It really did walk downhill. Validation accuracy is `0.4762`, the same number as the always-clear_road rule. Prediction counts are `[21, 0, 0, 0]`. The confusion matrix is a single occupied column: every true class was predicted `clear_road`. Recall is `1.0000` for clear road and `0.0000` for the other three. A falling loss here means the network became a more confident copy of the dumb rule. Six epochs is not enough for this model to leave that rule.
    """))
    cells.append(md("""
    Longer run, same learning rate and weight decay, three random starts (seeds 0, 1, and 2). We record validation accuracy every epoch so a later epoch can be compared with the best one. The curves and the confusion matrix below are seed 0. The mean and the range are there so one lucky start is not the whole story.

    **Predict:** will mean validation accuracy sit above 0.4762? Will pedestrian recall catch the other classes, or stay the weak one?
    """))
    cells.append(code("""
    keep_inline()
    LONG_EPOCHS = 40
    print("long epochs:", LONG_EPOCHS)
    print("long lr:", cfg.lr)
    print("long weight decay:", cfg.weight_decay)
    print("seeds:", [0, 1, 2])

    def train_seeds(criterion, epochs, lr, weight_decay, seeds=(0, 1, 2)):
        runs = []
        device = torch.device("cpu")
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            train_loader, val_loader = get_dataloaders(DATA_DIR, 16, seed)
            model = DrivingClassifier()
            opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            train_losses = []
            val_accs = []
            last = None
            for _ in range(epochs):
                train_losses.append(train_epoch(model, train_loader, opt, criterion, device))
                last = evaluate(model, val_loader, cross_entropy_loss, device, 4)
                val_accs.append(last[1])
            val_loss, val_acc, recalls = last
            best_i = int(np.argmax(val_accs))
            cm = np.zeros((4, 4), dtype=np.int64)
            model.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    preds = model(images).argmax(dim=-1).numpy()
                    for pred, true in zip(preds, targets.numpy()):
                        cm[int(true), int(pred)] += 1
            runs.append({
                "seed": seed,
                "model": model,
                "val_loader": val_loader,
                "train_losses": train_losses,
                "acc": val_acc,
                "recalls": recalls,
                "best_epoch": best_i + 1,
                "best_acc": val_accs[best_i],
                "cm": cm,
            })
        return runs

    def report(title, runs):
        accs = [r["acc"] for r in runs]
        print(title)
        print("per-seed val_accuracy:", [f"{a:.4f}" for a in accs])
        print("mean val_accuracy:", f"{float(np.mean(accs)):.4f}")
        print("val_accuracy range:", f"{min(accs):.4f}", "to", f"{max(accs):.4f}")
        mean_rec = np.mean([r["recalls"] for r in runs], axis=0)
        print("mean per-class recall:", {CLASS_NAMES[i]: f"{float(mean_rec[i]):.4f}" for i in range(4)})
        for r in runs:
            rec = {CLASS_NAMES[i]: f"{r['recalls'][i]:.4f}" for i in range(4)}
            print(
                f"seed {r['seed']}: val_accuracy {r['acc']:.4f} recall {rec} "
                f"train_loss {r['train_losses'][0]:.3f} -> {r['train_losses'][-1]:.3f} "
                f"best_val_accuracy {r['best_acc']:.4f} at epoch {r['best_epoch']}"
            )
        print("class order:", CLASS_NAMES)
        print("seed 0 confusion (rows=true, cols=predicted):", runs[0]["cm"].tolist())

    ce_runs = train_seeds(cross_entropy_loss, LONG_EPOCHS, cfg.lr, cfg.weight_decay)
    report("cross-entropy", ce_runs)
    seed0 = ce_runs[0]
    down = sum(
        seed0["train_losses"][i] > seed0["train_losses"][i + 1]
        for i in range(len(seed0["train_losses"]) - 1)
    )
    print("seed 0 train steps down:", down, "of", len(seed0["train_losses"]) - 1)

    metrics_ce = {
        "val_accuracy": seed0["acc"],
        "per_class_recall": {CLASS_NAMES[i]: seed0["recalls"][i] for i in range(4)},
    }
    model_ce = seed0["model"]
    val_loader = seed0["val_loader"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(range(1, LONG_EPOCHS + 1), seed0["train_losses"], marker="o", ms=3)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train cross-entropy")
    axes[0].set_title("Seed 0 training loss")
    im = axes[1].imshow(seed0["cm"], cmap="Blues")
    axes[1].set_xticks(range(4), CLASS_NAMES, rotation=45, ha="right")
    axes[1].set_yticks(range(4), CLASS_NAMES)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Seed 0 confusion matrix")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    plt.tight_layout()
    plt.show()
    """))
    cells.append(md("""
    Mean validation accuracy is `0.8413`, range `0.7619` to `0.9048`. Every seed is above the always-clear_road accuracy of `0.4762`. The network learned something the dumb rule does not know.

    Mean per-class recall is `clear_road 1.0000`, `lead_vehicle 1.0000`, `pedestrian 0.1111`, `lane_marking 0.8333`. Pedestrian is the weak class on the mean, and on every seed: `0.3333`, `0.0000`, `0.0000`. Lead vehicle is perfect on all three seeds. Lane marking slips only on seed 1 (`0.5000`).

    Seed 0's matrix is `[[10, 0, 0, 0], [0, 4, 0, 0], [1, 0, 1, 1], [0, 0, 0, 4]]`. All 10 clear-road crops, all 4 lead vehicles, and all 4 lane markings are correct. The pedestrian row is one `clear_road`, one correct, and one `lane_marking`. That is 1 out of 3, which is the printed recall `0.3333`. This is not "every non-clear-road image dumped into column 0." The other classes moved off that column. The person is what remains hard, which matches 6 training examples against 40 clear-road examples.

    Seed 0's training loss goes from `1.307` to `0.369`, down on `29` of `39` steps. A batch is 16 crops, not all 74, so the average can tick up even while the trend is down. Seed 0's best validation accuracy is the last epoch (`0.9048` at epoch 40). Seed 2's best and final accuracy are the same (`0.8571`, best already at epoch 16). Seed 1 is different: best validation accuracy is `0.8571` at epoch 16, and the final accuracy is `0.7619`, while training loss kept falling (`1.332` to `0.251`).

    That gap is **overfitting**. The weights keep fitting the training crops after the validation crops have stopped agreeing. With only 6 pedestrians in the training folder, extra epochs can memorize those 6 instead of learning a rule that holds for the 3 validation people. A single seed's last number is not the whole result. The mean and the range are.
    """))

    cells.append(md("## 9. Accuracy lies"))
    cells.append(md("""
    **Accuracy** is the fraction of all guesses that were right. **Recall** for one class ignores the images that are not that class. A model can score well on accuracy by getting the common class right and missing the rare one.

    **Predict:** on seed 0, which gap is larger — accuracy versus the always-clear_road rule, or pedestrian recall versus 1?
    """))
    cells.append(code("""
    print("always-clear_road val accuracy:", f"{baseline_acc:.4f}")
    print("CNN seed 0 val accuracy:", f"{metrics_ce['val_accuracy']:.4f}")
    print("CNN seed 0 pedestrian recall:", f"{metrics_ce['per_class_recall']['pedestrian']:.4f}")
    print("CNN mean val accuracy:", f"{float(np.mean([r['acc'] for r in ce_runs])):.4f}")
    print("CNN mean pedestrian recall:", f"{float(np.mean([r['recalls'][2] for r in ce_runs])):.4f}")
    print("validation pedestrians:", int(val_counts[2]))
    print("possible pedestrian recalls:", [f"{k / val_counts[2]:.4f}" for k in range(int(val_counts[2]) + 1)])
    """))
    cells.append(md("""
    Seed 0 accuracy is `0.9048` against a rule that scores `0.4762`. That gap is real. Pedestrian recall on that same seed is `0.3333`, and the mean across seeds is `0.1111`. The accuracy gap is the one that looks like success. The recall gap is the one that matters if the rare class is a person in the road.

    With 3 validation pedestrians, the only possible recall values are the ones just printed: `0.0000`, `0.3333`, `0.6667`, `1.0000`. One image moves the number by a third. Do not read a difference of `0.3333` between seeds as a precise new talent. It is often one crop.
    """))
    cells.append(md("""
    **Exercise — `minority_recall`.** Same idea as one entry of per-class recall: true positives divided by how many images really are that class. Return 0 when the class is absent. Leave the `TODO` as it is to use the reference implementation.
    """))
    cells.append(code("""
    def minority_recall_student(preds, targets, minority_class):
        # TODO: true positives / support for minority_class
        raise NotImplementedError

    def load_solution_metrics():
        path = REPO / "solutions" / "00_ml_gym" / "metrics.py"
        spec = importlib.util.spec_from_file_location("sol_metrics", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    sol_metrics = load_solution_metrics()

    def get_minority_recall_fn():
        try:
            minority_recall_student(torch.tensor([0]), torch.tensor([2]), 2)
        except NotImplementedError:
            print("Using reference minority_recall (TODO not implemented)")
            return sol_metrics.minority_recall
        return minority_recall_student

    mr_fn = get_minority_recall_fn()
    preds_h = torch.tensor([2, 0, 2, 1])
    tgts_h = torch.tensor([2, 2, 0, 2])
    print("toy preds:", preds_h.tolist())
    print("toy targets:", tgts_h.tolist())
    print("true minority count:", int((tgts_h == 2).sum()))
    print("caught:", int(((preds_h == 2) & (tgts_h == 2)).sum()))
    rec = mr_fn(preds_h, tgts_h, minority_class=2)
    assert abs(rec - 1 / 3) < 1e-5
    print("minority_recall:", f"{rec:.4f}")
    print("✅ correct: minority_recall =", round(rec, 4))
    """))
    cells.append(md("""
    The check prints `✅`. The toy targets contain 3 class-2 labels and the predictions catch 1 of them, so recall is `0.3333`. That is the same definition as the pedestrian entry in the training report. The reference function is used because the `TODO` still raises. Replace the `TODO` and run the cell again if you want the check to call your function instead.
    """))
    cells.append(md("""
    <details><summary>Solution</summary>

    ```python
    support = (targets == minority_class).sum().item()
    if support == 0:
        return 0.0
    tp = ((preds == minority_class) & (targets == minority_class)).sum().item()
    return tp / support
    ```

    </details>
    """))

    cells.append(md("## 10. Focal loss"))
    cells.append(md("""
    Cross-entropy adds up a loss on every image. **Easy** examples (the model is already confident and correct) still contribute a gradient, and there are many more clear-road crops than people.

    **Focal loss** multiplies cross-entropy by `(1 - p_t)^γ`. `p_t` is the probability on the true class. `γ` (gamma) turns that factor down when `p_t` is already near 1, so an easy example stops dominating the update. Gamma 0 leaves the factor at 1, which is ordinary cross-entropy.

    **Predict:** in the table below, the row with `p_t = 0.9` should shrink a lot more than the row with `p_t = 0.2`.
    """))
    cells.append(code("""
    def load_solution_losses():
        path = REPO / "solutions" / "00_ml_gym" / "losses.py"
        spec = importlib.util.spec_from_file_location("sol_losses", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    sol_losses = load_solution_losses()
    print("p_t   CE      focal(g=2)")
    for p_t in [0.9, 0.2]:
        logit0 = torch.log(torch.tensor(p_t / (1 - p_t)))
        logits = torch.tensor([[logit0, 0.0, -8.0, -8.0]])
        ce = F.cross_entropy(logits, torch.tensor([0])).item()
        fl = sol_losses.focal_loss(logits, torch.tensor([0]), gamma=2.0).item()
        print(f"{p_t:.1f}   {ce:.3f}   {fl:.3f}")
    """))
    cells.append(md("""
    At `p_t = 0.9`, cross-entropy is `0.105` and focal loss is `0.001`. The easy example almost vanishes. At `p_t = 0.2`, cross-entropy is `1.610` and focal loss is `1.031`. The hard example stays large. Focal loss does not invent new pedestrians. It turns the volume down on crops the model has already solved.
    """))
    cells.append(md("""
    **Exercise — `focal_loss`.** Same arguments as `modules/00_ml_gym/losses.py`: `logits`, `targets`, `gamma`, optional per-class `alpha`, and `reduction`. Leave the `TODO` in place to use the reference implementation.

    When gamma is 0 and alpha is omitted, the result should match cross-entropy.
    """))
    cells.append(code("""
    def focal_loss_student(logits, targets, gamma=2.0, alpha=None, reduction="mean"):
        # TODO: -alpha_t * (1 - p_t)^gamma * log(p_t)
        raise NotImplementedError

    def get_focal_loss():
        try:
            focal_loss_student(torch.randn(2, 4), torch.tensor([0, 1]), gamma=0.0)
        except NotImplementedError:
            print("Using reference focal_loss (TODO not implemented)")
            return sol_losses.focal_loss
        print("Using your focal_loss")
        return focal_loss_student

    focal_fn = get_focal_loss()
    torch.manual_seed(SEED)
    z_check = torch.randn(4, 4)
    t_check = torch.tensor([0, 1, 2, 3])
    fl0 = focal_fn(z_check, t_check, gamma=0.0)
    ce_check = F.cross_entropy(z_check, t_check)
    assert torch.allclose(fl0, ce_check, atol=1e-5)
    print("gamma 0 focal:", f"{fl0.item():.4f}")
    print("cross-entropy:", f"{ce_check.item():.4f}")
    print("✅ correct: gamma=0 matches cross-entropy")
    """))
    cells.append(md("""
    The check prints `✅`. Gamma 0 produced the same number as cross-entropy on this batch: both prints are `1.5809` when the `TODO` is still the reference. `(1 - p_t)^0` is 1, so the extra factor disappears. The reference implementation is used because the `TODO` still raises.
    """))
    cells.append(md("""
    <details><summary>Solution</summary>

    ```python
    log_probs = F.log_softmax(logits, dim=-1)
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    loss = -((1.0 - pt) ** gamma) * log_pt
    if alpha is not None:
        loss = alpha.to(logits.device)[targets] * loss
    ```

    </details>
    """))
    cells.append(md("""
    Two attempts to help the rare class, with the same 40 epochs, the same learning rate, and the same weight decay as the cross-entropy run. Three seeds again.

    1. Focal loss with gamma 2, and no extra class weight.
    2. Class-weighted cross-entropy. The weight for a class is `total / (4 × count)`, so a rare class pulls harder than a common one. `cross_entropy_loss` in the module does not take those weights, so this cell passes them straight to PyTorch.

    **Predict:** pedestrian recall can only be one of `0.0000`, `0.3333`, `0.6667`, or `1.0000`. Will either attempt raise it on all three seeds without hurting accuracy?
    """))
    cells.append(code("""
    keep_inline()
    counts = torch.tensor([train_counts[i] for i in range(4)], dtype=torch.float)
    class_weights = counts.sum() / (counts.numel() * counts)
    print("training class counts:", {CLASS_NAMES[i]: int(counts[i]) for i in range(4)})
    print("class weights:", {CLASS_NAMES[i]: f"{class_weights[i].item():.3f}" for i in range(4)})
    print("pedestrian weight / clear_road weight:", f"{(class_weights[2] / class_weights[0]).item():.2f}")
    print("focal gamma:", 2.0)
    print("validation pedestrians:", int(val_counts[2]))
    print("possible pedestrian recalls:", [f"{k / val_counts[2]:.4f}" for k in range(int(val_counts[2]) + 1)])

    def focal_crit(logits, targets):
        return focal_fn(logits, targets, gamma=2.0)

    def weighted_ce(logits, targets):
        return F.cross_entropy(logits, targets, weight=class_weights)

    focal_runs = train_seeds(focal_crit, LONG_EPOCHS, cfg.lr, cfg.weight_decay)
    report("focal gamma 2", focal_runs)
    weighted_runs = train_seeds(weighted_ce, LONG_EPOCHS, cfg.lr, cfg.weight_decay)
    report("class-weighted cross-entropy", weighted_runs)
    """))
    cells.append(md("""
    Class weights are `clear_road 0.463`, `lead_vehicle 1.321`, `pedestrian 3.083`, `lane_marking 1.321`. A pedestrian mistake pulls `6.67` times as hard as a clear-road mistake. The training counts behind those weights are 40, 14, 6, and 14.

    Focal loss, mean validation accuracy `0.8730` (range `0.8095` to `0.9524`), against `0.8413` for plain cross-entropy. Mean pedestrian recall is `0.2222` against `0.1111`. The per-seed pedestrian recalls are `0.6667`, `0.0000`, `0.0000`. Two seeds still miss every validation pedestrian. Seed 0's matrix is `[[10, 0, 0, 0], [0, 4, 0, 0], [1, 0, 2, 0], [0, 0, 0, 4]]`: 2 of 3 pedestrians caught, one still called `clear_road`. Focal loss helped a little on the average and did not help on two of the three starts. Turning down easy asphalt does not create a pedestrian pattern the folder barely contains.

    Class-weighted cross-entropy, mean validation accuracy `0.8254` (range `0.5714` to `0.9524`). That mean is a bit *below* plain cross-entropy. Mean pedestrian recall is `0.7778`, which is higher. Seeds 1 and 2 catch 2 of 3 pedestrians and score `0.9524`. Seed 0 is the warning: accuracy `0.5714`, pedestrian recall `1.0000`, and the matrix is `[[1, 0, 9, 0], [0, 4, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]`. Nine clear-road crops were called pedestrian. The rare class got so loud that open asphalt started looking like a person.

    So: plain cross-entropy learns the common classes and stays weak on pedestrians. Focal loss does not reliably fix that. Class weights can raise pedestrian recall and can also invent pedestrians. With 6 training people and 3 validation people, both results are noisy, and a recall printed on this validation set is only ever one of the four fractions above. More pedestrian crops would change this experiment. Reweighting the loss cannot.
    """))

    cells.append(md("## 11. Look at your mistakes"))
    cells.append(md("""
    **Exercise — error gallery.** Save the misclassified validation crops, most confident mistake first. Leave the `TODO` unimplemented to use the reference builder. The gallery should be the actual mistakes of the seed-0 cross-entropy model, not every image that is not clear road.
    """))
    cells.append(code("""
    def build_error_gallery_student(images, targets, logits, class_names, out_path, top_k=16):
        # TODO: save a grid of the top-k confident mistakes
        raise NotImplementedError

    def load_solution_gallery():
        path = REPO / "solutions" / "00_ml_gym" / "error_gallery.py"
        spec = importlib.util.spec_from_file_location("sol_gallery", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    sol_gallery = load_solution_gallery()

    def get_gallery_fn():
        try:
            build_error_gallery_student(
                torch.rand(1, 3, 64, 64), torch.tensor([0]), torch.randn(1, 4), CLASS_NAMES, "/tmp/x.png"
            )
        except NotImplementedError:
            print("Using reference build_error_gallery (TODO not implemented)")
            return sol_gallery.build_error_gallery
        print("Using your build_error_gallery")
        return build_error_gallery_student

    gallery_fn = get_gallery_fn()
    """))
    cells.append(md("""
    The print says the reference builder is in use, because the `TODO` still raises. The next cell runs it on the seed-0 validation crops.
    """))
    cells.append(code("""
    all_imgs, all_tgts, all_logits = [], [], []
    model_ce.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            logits = model_ce(images)
            all_imgs.append(images)
            all_tgts.append(targets)
            all_logits.append(logits)
    images_cat = torch.cat(all_imgs)
    tgts_cat = torch.cat(all_tgts)
    logits_cat = torch.cat(all_logits)
    out_gallery = REPO / "artifacts" / "m00" / "notebook_error_gallery.png"
    result = gallery_fn(images_cat, tgts_cat, logits_cat, CLASS_NAMES, out_gallery, top_k=8)
    assert Path(result["path"]).is_file()
    probs = torch.softmax(logits_cat, dim=-1)
    pred_idx = logits_cat.argmax(dim=-1)
    print("n_errors:", result["n_errors"])
    print("val images:", int(tgts_cat.shape[0]))
    for rank, idx in enumerate(result["order"], start=1):
        true_i = int(tgts_cat[idx])
        pred_i = int(pred_idx[idx])
        print(
            f"{rank}. true {CLASS_NAMES[true_i]} -> predicted {CLASS_NAMES[pred_i]} "
            f"confidence {probs[idx, pred_i].item():.3f}"
        )
    print("✅ correct: gallery saved,", result["n_errors"], "errors")

    keep_inline()
    from PIL import Image as PILImage
    plt.figure(figsize=(6, 3))
    plt.imshow(PILImage.open(out_gallery))
    plt.axis("off")
    plt.title("Mistakes, most confident first")
    plt.show()
    """))
    cells.append(md("""
    `n_errors` is 2, out of 21 validation images. The first line is pedestrian predicted `lane_marking` at confidence `0.367`. The second is pedestrian predicted `clear_road` at confidence `0.321`. The more confident mistake is listed first. The picture titles say the same thing (`T` true, `P` predicted). The pedestrian that was classified correctly is not in the gallery. Lead vehicles and lane markings from seed 0 are not in it either, which matches the confusion matrix: those rows were all on the diagonal. This is two real misses, not a dump of every crop that is not open road.
    """))
    cells.append(md("""
    <details><summary>Solution</summary>

    Sort misclassified indices by the probability of the predicted class, highest first. Draw each crop with its true class, predicted class, and that probability.

    </details>
    """))

    cells.append(md("## 12. Recap"))
    cells.append(md("""
    - A crop is numbers. This loader returns `(3, 64, 64)`, and a batch stacks those crops.
    - Softmax turns four scores into probabilities that add up to 1. Cross-entropy is the negative log of the probability on the true class. Its gradient raises that score and lowers the others.
    - The learning rate is the step size. At 0.1 the one-knob loss walks to the bottom. At 1.2 it explodes. The same thing happens to the linear classifier: learning rate 0.001 walks downhill, and 50 times that rate bounces.
    - A falling loss is not yet a useful driver. The linear model lowered its loss and still predicted `clear_road` every time. Six epochs of the CNN did the same thing: one column in the confusion matrix.
    - A convolution asks a local question, such as "is there an edge here?" The hand filters scored 0 on a flat patch and 3 on an edge they were built for. The network then turns each feature map into one number and scores four classes from that short list.
    - Forty epochs, three seeds, same learning rate as `TrainConfig`: mean validation accuracy `0.8413`, above the always-clear_road accuracy `0.4762`. Pedestrian recall stays the weakest (`0.1111` on average). There are 6 training pedestrians and 3 validation pedestrians, so that recall is only ever `0.0000`, `0.3333`, `0.6667`, or `1.0000`.
    - Focal loss nudged the average and still missed every validation pedestrian on two seeds. Class weights raised pedestrian recall and, on one seed, called nine clear-road crops pedestrians. Reweighting the loss is not a substitute for more examples of the rare class.
    - The error gallery is where you check that story. Seed 0's gallery has the two pedestrian misses, not every other class.

    ### Go deeper
    - [Karpathy — Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
    - [3Blue1Brown — Neural networks](https://www.3blue1brown.com/topics/neural-networks)
    - [CS231n — Convolutional networks](https://cs231n.github.io/convolutional-networks/)
    - [Focal loss paper (Lin et al., arXiv:1708.02002)](https://arxiv.org/abs/1708.02002)
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
    out = Path(__file__).resolve().parents[1] / "notebooks" / "00_driving_ml_gym.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    nbformat.write(nb, out)
    print(f"Wrote {out} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
