"""Cell definitions for Module 00 instruct spine notebook. Loaded by build_notebooks_00_01.build_m00."""

from __future__ import annotations

from typing import Callable

import nbformat as nbf


def build_m00_cells(
    _md: Callable[[str], nbf.NotebookNode],
    _code: Callable[[str], nbf.NotebookNode],
    session_card_text,
    come_back_cue,
) -> list:
    return [
        _md(
            "# Zero2FSD — Week 1 — Driving ML Gym\n\n"
            "This notebook is the **lecture path** for Week 1.\n\n"
            "- **Scaffold** (~60–70%): data loading, `DrivingClassifier`, cross-entropy training loop\n"
            "- **Fill**: implement `focal_loss` and `minority_recall` in `modules/00_ml_gym/*.py`\n"
            "- **From scratch**: implement `build_error_gallery` in `error_gallery.py`\n\n"
            "Crops are **synthetic checked-in PNGs** under `data/m00_sample` (CC0 license), "
            "not random `torch.randn` tensors.\n\n"
            "**Appendix 00b (optional, not this week's critical path):** "
            "`modules/00_nn_scratch` and `notebooks/00_neural_networks_and_autograd.ipynb` "
            "cover autograd from scratch when you want that depth."
        ),
        _md(
            session_card_text("m00").strip()
            + "\n\nXP is not awarded for opening this notebook."
        ),
        _code(
            "import sys\nfrom pathlib import Path\n\n"
            "repo = Path.cwd()\nif not (repo / 'modules' / '00_ml_gym').exists():\n"
            "    repo = repo.parent\n"
            "sys.path.insert(0, str(repo / 'modules'))\n"
            "from common.progress import format_stack\n\n"
            "progress_path = repo / 'artifacts' / 'progress.json'\n"
            "if progress_path.is_file():\n"
            "    import json\n"
            "    with progress_path.open() as f:\n"
            "        progress = json.load(f)\n"
            "    print(format_stack(progress))\n"
            "else:\n"
            "    print('No progress file yet. Opening this notebook awards 0 XP.')"
        ),
        _md(
            "**Governing principles (this week)**\n\n"
            "1. **A model is a function from data to scores;** learning adjusts parameters so a chosen "
            "loss gets small on the training distribution.\n"
            "2. **The loss defines what good means.** If the loss ignores rare classes, the model will too.\n"
            "3. **You cannot improve what you do not inspect.** Headline accuracy hides confident mistakes.\n\n"
            "You will apply (2) and (3) in code, not only read them here."
        ),
        _md("## 0. Cold open"),
        _md(
            "### Theory\n\n"
            "This week you train a function from a **64×64 driving crop** to **four class scores**: "
            "clear road, lead vehicle, pedestrian, lane marking (`CLASS_NAMES`). "
            "The scaffold `DrivingClassifier` already exists in `modules/00_ml_gym/model.py`. "
            "It will not be good yet — weights start random. "
            "The rest of this notebook explains **why** training changes those scores and **which numbers** "
            "you should trust when the dataset is imbalanced.\n\n"
            "Course direction is **top-down** (see a working classifier early, like FastAI). "
            "Inside each section we go **bottom-up**: short theory, tiny demo, check question, then the next idea.\n\n"
            "Pedestrian is class index **2** in `CLASS_NAMES`. The training split is **imbalanced**: "
            "`clear_road` is the majority. That imbalance is deliberate — it mirrors real driving logs where "
            "empty road dominates and rare objects still matter for safety."
        ),
        _code(
            "import sys\nfrom pathlib import Path\nimport torch\nimport matplotlib\n"
            "matplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n\n"
            "repo = Path.cwd()\nif not (repo / 'modules' / '00_ml_gym').exists():\n"
            "    repo = repo.parent\n"
            "sys.path.insert(0, str(repo / 'modules'))\n"
            "sys.path.insert(0, str(repo / 'modules' / '00_ml_gym'))\n"
            "torch.manual_seed(0)\n\n"
            "from config import TrainConfig\nfrom dataset import CLASS_NAMES, get_dataloaders\n"
            "from model import DrivingClassifier\nfrom losses import cross_entropy_loss, focal_loss\n"
            "from metrics import accuracy, per_class_recall, minority_recall\n"
            "print('repo:', repo)"
        ),
        _md("### Demo"),
        _code(
            "cfg = TrainConfig(data_dir=repo / 'data' / 'm00_sample')\n"
            "train_loader, val_loader = get_dataloaders(cfg.data_dir, batch_size=16, seed=0)\n"
            "images, labels = next(iter(train_loader))\n"
            "print('train batches per epoch:', len(train_loader), 'train images:', len(train_loader.dataset))"
        ),
        _code(
            "ds = train_loader.dataset\n"
            "fig, axes = plt.subplots(2, 4, figsize=(8, 4))\n"
            "for ax, i in zip(axes.flat, range(8)):\n"
            "    img, lab = ds[i]\n"
            "    ax.imshow(img.permute(1, 2, 0))\n"
            "    ax.set_title(CLASS_NAMES[lab.item()], fontsize=8)\n"
            "    ax.axis('off')\n"
            "plt.suptitle('Sample crops from PNG files')\nplt.tight_layout()\nplt.show()"
        ),
        _code(
            "model = DrivingClassifier(num_classes=4)\n"
            "model.eval()\n"
            "v_images, v_labels = next(iter(val_loader))\n"
            "with torch.no_grad():\n"
            "    v_logits = model(v_images)\n"
            "v_preds = v_logits.argmax(dim=-1)\n"
            "v_acc = accuracy(v_preds, v_labels)\n"
            "print('logits shape', v_logits.shape, '(B, 4)')\n"
            "print('argmax preds', v_preds.tolist())\n"
            "print('labels       ', v_labels.tolist())\n"
            "print('val accuracy (untrained):', round(v_acc, 3))"
        ),
        _code(
            "counts = train_loader.dataset.class_counts()\n"
            "n = sum(counts.values())\n"
            "count_road = counts[0]\n"
            "always_road_acc = count_road / n\n"
            "plt.bar(CLASS_NAMES, [counts[i] for i in range(4)])\n"
            "plt.title('Train class counts')\nplt.xticks(rotation=20)\nplt.show()\n"
            "print(f'Always-predict-clear_road accuracy: {always_road_acc:.3f} ({count_road}/{n})')\n"
            "print('Pedestrian recall of that dumb rule: 0.0')"
        ),
        _md(
            "### Check\n\n"
            "Which axis of a `(B, 3, 64, 64)` batch is **color**, and why does `Conv2d` care? "
            "The next cell checks pixel range and shape, then prints the axis convention."
        ),
        _code(
            "images, labels = next(iter(train_loader))\n"
            "assert images.shape[1:] == (3, 64, 64), images.shape\n"
            "assert float(images.min()) >= 0.0 and float(images.max()) <= 1.0\n"
            "print('Channel axis is 1 because PyTorch image batches use (B, C, H, W).')"
        ),
        _md("## 1. Images as numbers"),
        _md(
            "### Theory\n\n"
            "A PNG is a grid of pixels. Each pixel stores three numbers **R, G, B**, usually as integers "
            "0–255 on disk. After load we **divide by 255** so each channel sits in **[0, 1]**. "
            "That is a fixed scale, not a learned trick; it keeps activations in a range optimizers can step through. "
            "We do **not** subtract dataset mean/std in this module.\n\n"
            "On disk and in PIL, layout is **(H, W, C)**: height, width, channels. "
            "Our `DrivingPatchDataset` permutes to **(C, H, W)** per image. "
            "A batch stacks images into **(B, C, H, W)** where **B** is batch size, **C** is channels (3 for RGB), "
            "**H** height, **W** width.\n\n"
            "Axis order matters: `nn.Conv2d` expects channels **before** height and width. "
            "If you leave a batch as (H, W, C), PyTorch treats height as the channel dimension and the run fails "
            "or learns nonsense.\n\n"
            "When you `imshow` a tensor, you often `permute(1, 2, 0)` back to HWC because Matplotlib expects "
            "channels last. Inside the network, keep channels first."
        ),
        _md("### Demo"),
        _code(
            "from PIL import Image\nimport numpy as np\n\n"
            "row0 = train_loader.dataset.rows[0]\n"
            "png_path = train_loader.dataset.data_dir / str(row0['filename'])\n"
            "pil_img = Image.open(png_path).convert('RGB')\n"
            "print('PIL size (W,H):', pil_img.size, 'mode:', pil_img.mode)\n"
            "hwc = np.array(pil_img, dtype=np.float32) / 255.0\n"
            "print('array shape (H,W,C):', hwc.shape)\n"
            "print('Red channel 4x4 patch (rounded):\\n', np.round(hwc[:4, :4, 0], 2))"
        ),
        _code(
            "tensor_chw, lab = train_loader.dataset[0]\n"
            "print('dataset tensor shape (C,H,W):', tuple(tensor_chw.shape))\n"
            "hwc_from_tensor = tensor_chw.permute(1, 2, 0).numpy()\n"
            "assert np.allclose(hwc_from_tensor, hwc, atol=1e-5)\n"
            "print('permute(1,2,0) matches PIL array within tolerance')"
        ),
        _code(
            "fig, axes = plt.subplots(1, 2, figsize=(7, 3))\n"
            "axes[0].imshow(hwc)\n"
            "axes[0].set_title('RGB image')\n"
            "axes[0].axis('off')\n"
            "im = axes[1].imshow(hwc[:, :, 0], cmap='viridis')\n"
            "axes[1].set_title('Red channel as numbers')\n"
            "plt.colorbar(im, ax=axes[1], fraction=0.046)\n"
            "plt.tight_layout()\nplt.show()"
        ),
        _md(
            "### Check\n\n"
            "One crop is `(3, 64, 64)`. What shape is a stack of 16? "
            "The next cell builds the stack and asserts the batched shape."
        ),
        _code(
            "stacked = torch.stack([train_loader.dataset[i][0] for i in range(16)])\n"
            "assert stacked.shape == (16, 3, 64, 64), stacked.shape\n"
            "print('Sixteen images batched:', tuple(stacked.shape))"
        ),
        _md("## 2. Classification as scores"),
        _md(
            "### Theory\n\n"
            "The model outputs **logits** \\(z\\): one raw score per class. They are **not** probabilities. "
            "**Softmax** turns one vector of logits into positive numbers that sum to 1:\n\n"
            "$$p_k = \\frac{\\exp(z_k)}{\\sum_j \\exp(z_j)}$$\n\n"
            "The predicted class is **argmax** of \\(z\\) (same as argmax of \\(p\\), because softmax preserves "
            "which coordinate is largest within the same vector).\n\n"
            "If the true class is \\(y\\), **cross-entropy** charges the negative log probability of that class:\n\n"
            "$$\\mathrm{CE} = -\\log p_y$$\n\n"
            "**Hand example:** logits `[2.0, 1.0, 0.5, -1.0]`, true class **0**. "
            "Class 0 has the largest logit, so it gets the largest \\(p_k\\). "
            "CE is small when \\(p_y\\) is near 1 and grows without bound as \\(p_y \\to 0\\) because \\(\\log p\\) goes to \\(-\\infty\\).\n\n"
            "PyTorch `F.cross_entropy` uses a **numerically stable** `log_softmax` internally; the formula above is the definition.\n\n"
            "Optional deepeners (after you can compute CE by hand): "
            "[StatQuest on cross-entropy](https://www.youtube.com/watch?v=6ArSys5qHAU), "
            "[3Blue1Brown neural networks](https://www.3blue1brown.com/lessons/neural-networks).\n\n"
            "The scaffold wraps the same idea in `cross_entropy_loss(logits, targets)` for shape `(B, K)` and `(B,)`. "
            "Training averages CE over the batch; one very wrong crop can still pull the weights if its \\(p_y\\) is tiny."
        ),
        _md("### Demo"),
        _code(
            "import torch.nn.functional as F\n\n"
            "z = torch.tensor([2.0, 1.0, 0.5, -1.0])\n"
            "y = torch.tensor(0)\n"
            "p = F.softmax(z, dim=0)\n"
            "manual_ce = -torch.log(p[y])\n"
            "ce = F.cross_entropy(z.unsqueeze(0), y.unsqueeze(0))\n"
            "print('softmax p', [round(x, 4) for x in p.tolist()])\n"
            "print('manual CE', manual_ce.item(), 'F.cross_entropy', ce.item())\n"
            "assert torch.allclose(manual_ce, ce, atol=1e-5)"
        ),
        _code(
            "fig, ax = plt.subplots(figsize=(5, 3))\n"
            "ax.bar(CLASS_NAMES, p.numpy())\n"
            "ax.set_ylabel('probability')\n"
            "ax.set_title('Softmax of example logits')\nplt.xticks(rotation=20)\nplt.tight_layout()\nplt.show()"
        ),
        _md(
            "### Check\n\n"
            "Same logits `[2, 1, 0.5, -1]`. If the true class is **3** instead of **0**, does CE go **up** or **down**? Why? "
            "The next cell prints both CE values and asserts the harder label costs more."
        ),
        _code(
            "y_wrong = torch.tensor(3)\n"
            "ce_true = F.cross_entropy(z.unsqueeze(0), y.unsqueeze(0)).item()\n"
            "ce_wrong = F.cross_entropy(z.unsqueeze(0), y_wrong.unsqueeze(0)).item()\n"
            "print('CE true class 0:', ce_true)\n"
            "print('CE true class 3 (weak logit):', ce_wrong)\n"
            "assert ce_wrong > ce_true"
        ),
        _md("## 3. Neuron and layer"),
        _md(
            "### Theory\n\n"
            "One **neuron** is an affine map plus a nonlinearity:\n\n"
            "$$z = w \\cdot x + b, \\qquad y = \\mathrm{ReLU}(z), \\qquad \\mathrm{ReLU}(z) = \\max(0, z)$$\n\n"
            "A **layer** is many neurons. With PyTorch `nn.Linear`, batch shape is **(B, F_in)** and "
            "weights have shape **(F_out, F_in)** so \\(Y = \\mathrm{ReLU}(X W^T + b)\\).\n\n"
            "**Why ReLU?** Two linear maps compose into one linear map: \\(W_2(W_1 x) = (W_2 W_1)x\\). "
            "A stack of linear layers **without** a bend does not increase the family of functions you can represent — "
            "it is still one hyperplane decision boundary in feature space. "
            "ReLU **bends** the space: negative pre-activations become 0, so different input regions use different active weights. "
            "That is enough to separate patterns a single plane cannot (XOR is the classic tiny proof).\n\n"
            "This beat is not yet an image model; images come once you trust the train loop on a toy case.\n\n"
            "Batch norm and other blocks in `DrivingClassifier` also add nonlinearity and re-scaling, but the "
            "core lesson is the same: without a bend, depth does not buy expressive power."
        ),
        _md("### Demo"),
        _code(
            "import torch.nn as nn\n"
            "import torch.optim as optim\n\n"
            "X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])\n"
            "y_xor = torch.tensor([[0.], [1.], [1.], [0.]])\n"
            "torch.manual_seed(0)\n\n"
            "def fit_linear(steps=300):\n"
            "    m = nn.Linear(2, 1)\n"
            "    opt = optim.Adam(m.parameters(), lr=0.1)\n"
            "    loss_fn = nn.BCEWithLogitsLoss()\n"
            "    for _ in range(steps):\n"
            "        opt.zero_grad()\n"
            "        loss = loss_fn(m(X).squeeze(-1), y_xor.squeeze(-1))\n"
            "        loss.backward()\n"
            "        opt.step()\n"
            "    with torch.no_grad():\n"
            "        pred = (torch.sigmoid(m(X).squeeze(-1)) > 0.5).float()\n"
            "        acc = (pred == y_xor.squeeze(-1)).float().mean().item()\n"
            "    return acc\n\n"
            "acc_lin = fit_linear()\n"
            "print('Linear-only XOR accuracy:', acc_lin)\n"
            "assert acc_lin <= 0.75"
        ),
        _code(
            "def fit_mlp(steps=2000):\n"
            "    torch.manual_seed(1)\n"
            "    m = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))\n"
            "    opt = optim.Adam(m.parameters(), lr=0.2)\n"
            "    loss_fn = nn.BCEWithLogitsLoss()\n"
            "    for _ in range(steps):\n"
            "        opt.zero_grad()\n"
            "        loss = loss_fn(m(X).squeeze(-1), y_xor.squeeze(-1))\n"
            "        loss.backward()\n"
            "        opt.step()\n"
            "    with torch.no_grad():\n"
            "        pred = (torch.sigmoid(m(X).squeeze(-1)) > 0.5).float()\n"
            "        acc = (pred == y_xor.squeeze(-1)).float().mean().item()\n"
            "    return acc\n\n"
            "acc_mlp = fit_mlp()\n"
            "print('MLP + ReLU XOR accuracy:', acc_mlp)\n"
            "assert acc_mlp == 1.0"
        ),
        _code(
            "fig, ax = plt.subplots(figsize=(4, 4))\n"
            "colors = ['C0' if v == 0 else 'C1' for v in y_xor.squeeze(-1).tolist()]\n"
            "ax.scatter(X[:, 0], X[:, 1], c=colors, s=120)\n"
            "ax.set_xlabel('x1')\nax.set_ylabel('x2')\n"
            "ax.set_title('XOR: a line cannot separate; ReLU MLP can')\nplt.tight_layout()\nplt.show()"
        ),
        _md(
            "### Check\n\n"
            "A stack of linear layers with no ReLU is still one linear map. What does that imply for XOR on the four corners? "
            "Answer first, then compare the demo above: linear accuracy should be at or below 0.75; the ReLU MLP should reach 1.0."
        ),
        _md("## 4. The train loop"),
        _md(
            "### Theory\n\n"
            "Training is the same story on every batch of crops. One **step**:\n\n"
            "1. **Forward:** `logits = model(images)`\n"
            "2. **Loss:** a scalar that is small when scores match targets (`cross_entropy_loss` today)\n"
            "3. **Backward:** `loss.backward()` fills each parameter's `.grad` with \\(\\partial \\mathrm{loss}/\\partial \\mathrm{param}\\). "
            "You do not derive those by hand this week — optional appendix **00b** (`modules/00_nn_scratch`, "
            "`notebooks/00_neural_networks_and_autograd.ipynb`) does.\n"
            "4. **Step:** `optimizer.step()` updates weights using those gradients. "
            "`optimizer.zero_grad()` **before** the next forward so gradients do not accumulate across batches.\n\n"
            "A **batch** is one group of crops (here 16). An **epoch** is one full pass over the training split. "
            "Validation uses `evaluate()` — forward and metrics only, **no** optimizer step.\n\n"
            "**Learning rate** is step size. Too small: loss crawls. Too large: updates overshoot; loss climbs or becomes NaN. "
            "The scaffold treats non-finite loss as a hard failure.\n\n"
            "Optional deepener: [3Blue1Brown backpropagation](https://www.3blue1brown.com/lessons/backpropagation) — "
            "watch after you can narrate the four steps, not before.\n\n"
            "In `train.py`, `train_epoch` loops batches, calls `zero_grad`, forward, loss, `backward`, `step`. "
            "`evaluate` sets `model.eval()`, runs forward only, and aggregates predictions for accuracy and recall."
        ),
        _md("### Demo"),
        _code(
            "def fit_line_lr(lr, steps=40):\n"
            "    torch.manual_seed(0)\n"
            "    x = torch.randn(64, 1)\n"
            "    y = 2.0 * x\n"
            "    m = nn.Linear(1, 1)\n"
            "    opt = optim.SGD(m.parameters(), lr=lr)\n"
            "    losses = []\n"
            "    for _ in range(steps):\n"
            "        opt.zero_grad()\n"
            "        pred = m(x)\n"
            "        loss = ((pred - y) ** 2).mean()\n"
            "        loss.backward()\n"
            "        opt.step()\n"
            "        losses.append(loss.item())\n"
            "    return losses\n\n"
            "losses_small = fit_line_lr(0.05)\n"
            "losses_large = fit_line_lr(2.0)\n"
            "print('final loss lr=0.05', losses_small[-1])\n"
            "print('final loss lr=2.0 ', losses_large[-1])\n"
            "assert losses_small[-1] < 0.05\n"
            "assert losses_large[-1] > losses_small[-1] + 1.0"
        ),
        _code(
            "cap = 1e4\n"
            "fig, ax = plt.subplots(figsize=(5, 3))\n"
            "ax.plot([min(v, cap) for v in losses_small], label='lr=0.05')\n"
            "ax.plot([min(v, cap) for v in losses_large], label='lr=2.0')\n"
            "ax.set_yscale('log')\n"
            "ax.set_xlabel('step')\nax.set_ylabel('MSE (capped for plot)')\n"
            "ax.legend()\nax.set_title('Learning rate: stable vs divergent')\nplt.tight_layout()\nplt.show()"
        ),
        _code(
            "print('Real train_loader: len=', len(train_loader), 'dataset=', len(train_loader.dataset))\n"
            "print('train_epoch in train.py runs forward → loss → backward → step on each batch of crops.')"
        ),
        _md(
            "### Check\n\n"
            "If `backward` runs twice on the same parameter and you **skip** `zero_grad`, what happens to `.grad`? "
            "The next cell runs two backward passes and compares gradient magnitude."
        ),
        _code(
            "tiny = nn.Linear(1, 1)\n"
            "x1 = torch.tensor([[1.0]])\n"
            "y1 = torch.tensor([[2.0]])\n"
            "loss_fn = nn.MSELoss()\n"
            "tiny.zero_grad()\n"
            "loss1 = loss_fn(tiny(x1), y1)\n"
            "loss1.backward()\n"
            "g1 = tiny.weight.grad.abs().item()\n"
            "loss2 = loss_fn(tiny(x1), y1)\n"
            "loss2.backward()\n"
            "g2 = tiny.weight.grad.abs().item()\n"
            "print('grad norm after 1st backward', g1)\n"
            "print('grad norm after 2nd backward without zero_grad', g2)\n"
            "assert g2 > g1\n"
            "print('If you forget zero_grad, gradients accumulate across batches.')"
        ),
        _md("## 5. Why convolutions"),
        _md(
            "### Theory\n\n"
            "A fully connected layer on a 64×64×3 image would use a **separate weight for every pixel–neuron pair**. "
            "Nearby pixels form edges and lane dashes; distant pixels often belong to different objects. "
            "**Convolution** encodes two assumptions:\n\n"
            "1. **Locality:** a 3×3 kernel looks at a neighborhood.\n"
            "2. **Weight sharing:** the same kernel slides over the whole image, so a detector learned on the left also fires on the right.\n\n"
            "Think in **channel stacks:** the stem's **32** output channels are 32 different 3×3 filters applied to RGB — "
            "32 response maps at once. **Stage2** outputs **64** channels by convolving over those 32 maps, so each new channel "
            "mixes lower-level detectors; it is not 64 copies of one edge filter. After pooling, the **fc** layer reads the "
            "64-dimensional channel vector as a feature summary for class scores.\n\n"
            "Each output channel is one learned filter. **Stride 2** (with kernel 3, padding 1) halves height and width so the next layer sees a coarser grid. "
            "**Adaptive average pool** to 1×1 collapses space into one vector per channel so a linear layer can emit class scores. "
            "That pool discards **where** a pattern fired — fine for “what is in this crop”, wrong later for “where is the lane” (we are not doing detection this week).\n\n"
            "**Scaffold `DrivingClassifier` blocks (do not redesign):**\n\n"
            "- **stem:** `Conv2d(3,32,k=3,s=2,p=1,bias=False)` + BN + ReLU → `(B,32,32,32)` from `(B,3,64,64)`\n"
            "- **stage2:** `Conv2d(32,64,...)` → `(B,64,16,16)`\n"
            "- **pool:** `AdaptiveAvgPool2d(1,1)` + flatten → `(B,64)`\n"
            "- **fc:** `Linear(64, num_classes)` → logits `(B, K)`\n\n"
            "Parameter count stays modest because sharing repeats the same 3×3 weights across spatial locations "
            "instead of learning a separate weight per pixel."
        ),
        _md("### Demo"),
        _code(
            "import torch.nn.functional as F_conv\n\n"
            "crop, _ = train_loader.dataset[0]\n"
            "x1 = crop.unsqueeze(0)\n"
            "kernel = torch.tensor([[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]])\n"
            "k = torch.zeros(1, 1, 3, 3)\n"
            "k[0, 0] = kernel\n"
            "green = x1[:, 1:2, :, :]\n"
            "resp = F_conv.conv2d(green, k, padding=1)\n"
            "print('horizontal contrast response shape', tuple(resp.shape))"
        ),
        _code(
            "fig, axes = plt.subplots(1, 2, figsize=(7, 3))\n"
            "axes[0].imshow(crop.permute(1, 2, 0))\n"
            "axes[0].set_title('Input crop')\naxes[0].axis('off')\n"
            "axes[1].imshow(resp[0, 0].detach(), cmap='gray')\n"
            "axes[1].set_title('Green-channel conv response')\naxes[1].axis('off')\n"
            "plt.tight_layout()\nplt.show()"
        ),
        _code(
            "cnn = DrivingClassifier(num_classes=4)\n"
            "cnn.eval()\n"
            "with torch.no_grad():\n"
            "    s = cnn.stem(x1)\n"
            "    t = cnn.stage2(s)\n"
            "    emb = cnn.extract_features(x1)\n"
            "    lg = cnn(x1)\n"
            "print('input', tuple(x1.shape))\n"
            "print('stem ', tuple(s.shape))\n"
            "print('stage2', tuple(t.shape))\n"
            "print('pooled embedding', tuple(emb.shape))\n"
            "print('logits', tuple(lg.shape))\n"
            "assert s.shape == (1, 32, 32, 32)\n"
            "assert t.shape == (1, 64, 16, 16)\n"
            "assert emb.shape == (1, 64)\n"
            "assert lg.shape == (1, 4)"
        ),
        _md(
            "### Check\n\n"
            "Kernel 3, stride 2, padding 1, input height 64. What is `H_out`, and why does **stem** therefore emit spatial size 32? "
            "The next cell evaluates the standard conv output formula."
        ),
        _code(
            "H, W, k, s, p = 64, 64, 3, 2, 1\n"
            "H_out = (H + 2 * p - k) // s + 1\n"
            "print('H_out formula gives', H_out)\n"
            "assert H_out == 32\n"
            "print('Stride 2 with k=3,p=1 halves 64→32 spatial size.')"
        ),
        _md("## 6. Metrics that lie"),
        _md(
            "### Theory\n\n"
            "**Accuracy** = correct / all. If most crops are `clear_road`, a model that **always** predicts class 0 "
            "gets high accuracy and **zero pedestrian recall**.\n\n"
            "A **confusion matrix** has **rows = true class**, **columns = predicted class**. "
            "Diagonal entries are correct; off-diagonal cells are error types (pedestrian called road is the safety-critical cell).\n\n"
            "Driving cares about **missing** a pedestrian (recall). Accuracy hides that trap.\n\n"
            "| | Predicted positive | Predicted negative |\n"
            "|---|---|---|\n"
            "| Actual positive | TP | FN |\n"
            "| Actual negative | FP | TN |\n\n"
            "Recall = TP / (TP + FN). Precision = TP / (TP + FP).\n\n"
            "A four-class matrix is the same idea: one row and one column per class in `CLASS_NAMES`, "
            "with counts of true-vs-predicted pairs.\n\n"
            "`per_class_recall` in the scaffold computes recall for each class id; `minority_recall` (your fill) "
            "is the same formula for one chosen class — pedestrian index 2 in the rubric."
        ),
        _md("### Demo"),
        _code(
            "train_tgts = torch.tensor([train_loader.dataset[i][1].item() for i in range(len(train_loader.dataset))])\n"
            "preds_road = torch.zeros(len(train_tgts), dtype=torch.long)\n"
            "acc_road = accuracy(preds_road, train_tgts)\n"
            "recalls_road = per_class_recall(preds_road, train_tgts, 4)\n"
            "print('always-road accuracy', round(acc_road, 3))\n"
            "print('per-class recalls', [round(r, 2) for r in recalls_road])\n"
            "assert acc_road > 0.4\n"
            "assert recalls_road[2] == 0.0"
        ),
        _code(
            "from train import train_epoch, evaluate\n"
            "import numpy as np\n\n"
            "device = torch.device('cpu')\n"
            "model = DrivingClassifier().to(device)\n"
            "opt = optim.AdamW(model.parameters(), lr=1e-3)\n"
            "for ep in range(4):\n"
            "    train_epoch(model, train_loader, opt, cross_entropy_loss, device)\n"
            "    loss, acc, recalls = evaluate(model, val_loader, cross_entropy_loss, device, 4)\n"
            "    print(f'epoch {ep+1} val_acc={acc:.3f} recalls={[round(r, 2) for r in recalls]}')\n"
            "ce_ped_recall = recalls[2]"
        ),
        _code(
            "num_classes = 4\n"
            "cm = np.zeros((num_classes, num_classes), dtype=np.int64)\n"
            "model.eval()\n"
            "with torch.no_grad():\n"
            "    for imgs, tgts in val_loader:\n"
            "        preds = model(imgs.to(device)).argmax(dim=-1).cpu().numpy()\n"
            "        for p, t in zip(preds, tgts.numpy()):\n"
            "            cm[t, p] += 1\n"
            "fig, ax = plt.subplots(figsize=(5, 4))\n"
            "im = ax.imshow(cm, cmap='Blues')\n"
            "ax.set_xticks(range(num_classes), CLASS_NAMES, rotation=45, ha='right')\n"
            "ax.set_yticks(range(num_classes), CLASS_NAMES)\n"
            "ax.set_xlabel('Predicted')\nax.set_ylabel('True')\n"
            "fig.colorbar(im, ax=ax, fraction=0.046)\n"
            "plt.title('Validation confusion matrix (CE)')\nplt.tight_layout()\nplt.show()"
        ),
        _md(
            "### Check\n\n"
            "Hand batch: preds `[2, 0, 2, 1]`, targets `[2, 2, 0, 2]`, class **2**. "
            "Compute support, TP, recall, and precision before you run the cell — then check against the asserts."
        ),
        _code(
            "preds_h = torch.tensor([2, 0, 2, 1])\n"
            "tgts_h = torch.tensor([2, 2, 0, 2])\n"
            "c = 2\n"
            "support = (tgts_h == c).sum().item()\n"
            "tp = ((preds_h == c) & (tgts_h == c)).sum().item()\n"
            "recall = tp / support\n"
            "pred_pos = (preds_h == c).sum().item()\n"
            "precision = tp / pred_pos\n"
            "batch_acc = (preds_h == tgts_h).float().mean().item()\n"
            "print('support', support, 'TP', tp, 'recall', recall, 'precision', precision)\n"
            "print('batch accuracy', batch_acc)\n"
            "assert abs(recall - 1/3) < 1e-6\n"
            "assert abs(precision - 0.5) < 1e-6\n"
            "print('Accuracy on this tiny batch is not the number you would ship.')"
        ),
        _md("## 7. The loss is the objective"),
        _md(
            "### Theory\n\n"
            "**Principle 2:** the loss is the definition of **good** that the optimizer sees. "
            "Plain cross-entropy charges \\(-\\log p_t\\) on **every** crop, so a pile of easy road examples can outweigh a few hard pedestrians.\n\n"
            "**Class-weighted CE** multiplies each example by a weight that depends on its class (`weight` tensor of shape `(K,)` in PyTorch). "
            "Rare classes cost more per crop; easy examples of that class still pay full CE. We do not implement weighted CE this week — "
            "it sits between plain CE and focal loss.\n\n"
            "**Focal loss** (Lin et al.) multiplies CE by a **focusing factor** that depends on how correct the model already is:\n\n"
            "$$p_t = p_y, \\qquad \\mathrm{FL} = -\\alpha_t (1 - p_t)^\\gamma \\log(p_t)$$\n\n"
            "Derive the factor, do not only memorize the name:\n\n"
            "- If \\(\\gamma = 0\\), \\((1-p_t)^0 = 1\\), so FL matches CE when \\(\\alpha\\) is absent.\n"
            "- If the model is confident and **correct**, \\(p_t \\to 1\\), \\((1-p_t)^\\gamma \\to 0\\) — the example stops dominating the gradient.\n"
            "- If the model is unsure or **wrong**, \\(p_t\\) is small, \\((1-p_t)^\\gamma \\approx 1\\) — CE remains.\n"
            "- \\(\\gamma = 2\\) is the default in `TrainConfig.gamma`.\n"
            "- \\(\\alpha_t\\) is optional per-class weight, shape `(K,)`, applied as `alpha[targets]`; `None` means no extra class weight.\n\n"
            "The easy 0.99 example is **crushed** relative to CE; the hard 0.10 example **barely shrinks**. That is what \\(\\gamma\\) buys you.\n\n"
            "Focal loss does not replace the need for inspection (Principle 3) or good labels — it changes **which gradients** "
            "dominate a minibatch when many crops are easy."
        ),
        _md("### Demo"),
        _code(
            "import math\n\n"
            "print('p_t | (1-p_t)^2 | -log(p_t) | focal (gamma=2)')\n"
            "for p_t in [0.99, 0.90, 0.50, 0.10]:\n"
            "    factor = (1 - p_t) ** 2\n"
            "    ce = -math.log(p_t)\n"
            "    fl = factor * ce\n"
            "    print(f'{p_t:.2f} | {factor:.4f} | {ce:.3f} | {fl:.3f}')\n"
            "assert (1 - 0.99) ** 2 < 0.001\n"
            "assert (1 - 0.10) ** 2 > 0.5"
        ),
        _md(
            "### Check\n\n"
            "From the focal demo table (\\(\\gamma=2\\)): at \\(p_t=0.99\\), is \\((1-p_t)^2\\) below 0.001? "
            "At \\(p_t=0.10\\), is the factor above 0.5? The demo cell above printed the table and asserted both."
        ),
        _md(
            "**FILL — `focal_loss`** in `modules/00_ml_gym/losses.py` only.\n\n"
            "Signature: `focal_loss(logits, targets, gamma=2.0, alpha=None, reduction=\"mean\")`.\n"
            "Do **not** paste a solution into this notebook. "
            "Stable route hint: `log_softmax`, then gather the log prob of the true class. "
            "`gamma=0` and `alpha=None` must match `cross_entropy_loss`."
        ),
        _code(
            "try:\n"
            "    torch.manual_seed(0)\n"
            "    z = torch.randn(4, 4)\n"
            "    t = torch.tensor([0, 1, 2, 3])\n"
            "    fl0 = focal_loss(z, t, gamma=0.0)\n"
            "    ce = cross_entropy_loss(z, t)\n"
            "    print('gamma=0 allclose', torch.allclose(fl0, ce, atol=1e-5))\n"
            "    assert torch.allclose(fl0, ce, atol=1e-5)\n"
            "    z2 = torch.tensor([[8.0, 0.0, 0.0, 0.0]])\n"
            "    t2 = torch.tensor([0])\n"
            "    fl2 = focal_loss(z2, t2, gamma=2.0).item()\n"
            "    ce2 = cross_entropy_loss(z2, t2).item()\n"
            "    print('gamma=2 confident FL', fl2, 'CE', ce2, 'FL < CE', fl2 < ce2)\n"
            "    assert fl2 < ce2\n"
            "except NotImplementedError:\n"
            "    print('STOP: implement focal_loss in modules/00_ml_gym/losses.py')"
        ),
        _code(
            "try:\n"
            "    focal_loss(torch.randn(2, 4), torch.tensor([0, 1]), gamma=2.0)\n"
            "    torch.manual_seed(0)\n"
            "    model_fl = DrivingClassifier().to(device)\n"
            "    opt_fl = optim.AdamW(model_fl.parameters(), lr=1e-3)\n"
            "    focal_fn = lambda lg, tg: focal_loss(lg, tg, gamma=2.0)\n"
            "    for ep in range(4):\n"
            "        train_epoch(model_fl, train_loader, opt_fl, focal_fn, device)\n"
            "        _, acc_f, recalls_f = evaluate(model_fl, val_loader, focal_fn, device, 4)\n"
            "        print(f'focal epoch {ep+1} val_acc={acc_f:.3f} recalls={[round(r, 2) for r in recalls_f]}')\n"
            "    print(f'CE pedestrian recall (index 2): {ce_ped_recall:.3f}')\n"
            "    print(f'Focal pedestrian recall (index 2): {recalls_f[2]:.3f}')\n"
            "    print(f'Delta (focal - CE): {recalls_f[2] - ce_ped_recall:.3f}')\n"
            "except NotImplementedError:\n"
            "    print('Comparison waits until focal_loss fill is done')"
        ),
        _md("## 8. Inspect and ship"),
        _md(
            "### Theory\n\n"
            "**Principle 3:** you cannot improve what you do not inspect. "
            "Accuracy and even one recall number do not show whether mistakes are **confident**. "
            "Confident wrong crops are the ones a data engine would relabel or oversample — "
            "Tesla's data engine is one example of hard-example mining at scale, not a brand claim. "
            "Your **error gallery** is the small version: misclassified crops sorted by the probability of the "
            "**predicted (wrong)** class, highest first.\n\n"
            "Images are **NCHW** in `[0, 1]`. Return dict keys `path`, `n_errors`, `order`; write a figure even when `n_errors=0`.\n\n"
            "After the gallery, you still run **break-it** (engineering hygiene), **pytest** (contracts), and "
            "`train.main` to export `artifacts/m00/<run_id>/metrics.json` — the same artifact path the course tracks for XP."
        ),
        _md(
            "**FILL — `minority_recall`** in `modules/00_ml_gym/metrics.py`.\n\n"
            "Recall = TP / support for one class; **0** if support is 0. "
            "Hand example: preds `[2, 0, 2, 1]`, targets `[2, 2, 0, 2]`, class 2 → **1/3**."
        ),
        _code(
            "try:\n"
            "    preds = torch.tensor([2, 0, 2, 1])\n"
            "    tgts = torch.tensor([2, 2, 0, 2])\n"
            "    mr = minority_recall(preds, tgts, minority_class=2)\n"
            "    print('minority_recall', mr)\n"
            "    assert abs(mr - 1/3) < 1e-6\n"
            "except NotImplementedError:\n"
            "    print('STOP: implement minority_recall in metrics.py')"
        ),
        _md(
            "**FROM SCRATCH — `build_error_gallery`** in `modules/00_ml_gym/error_gallery.py`.\n\n"
            "Contract: sort mistakes by descending confidence of the wrong class. "
            "Do not implement it in this notebook."
        ),
        _code(
            "from error_gallery import build_error_gallery\n"
            "from PIL import Image\n"
            "import numpy as np\n\n"
            "try:\n"
            "    out = repo / 'artifacts' / 'm00' / 'nb_gallery.png'\n"
            "    model.eval()\n"
            "    imgs, tgts = next(iter(val_loader))\n"
            "    with torch.no_grad():\n"
            "        lg = model(imgs)\n"
            "    res = build_error_gallery(imgs, tgts, lg, CLASS_NAMES, out)\n"
            "    print(res)\n"
            "    plt.imshow(np.array(Image.open(out)))\n"
            "    plt.axis('off')\n"
            "    plt.title('Error gallery')\n"
            "    plt.show()\n"
            "except NotImplementedError:\n"
            "    print('STOP: implement build_error_gallery in error_gallery.py')"
        ),
        _md(
            "**Free response (Principle 2):** Which principle did focal loss apply? "
            "Why does \\(\\gamma=0\\) match cross-entropy? Write 3–6 sentences before opening `solutions/00_ml_gym`."
        ),
        _md("_Write 3–6 sentences here._"),
        _md(
            "**Free response (Principle 3):** The always-road classifier had high accuracy and zero pedestrian recall. "
            "Which principle says that metric was the wrong definition of good? Write 3–6 sentences."
        ),
        _md("_Write 3–6 sentences here._"),
        _md("### Demo"),
        _code("from break_it_fix_it import main as break_demo\nbreak_demo()"),
        _md(
            "### Tests\n\n"
            "```bash\npython3 -m pytest modules/00_ml_gym -q\n```\n\n"
            "Scaffold tests in `test_scaffold.py` always run. "
            "`test_assignment_solutions.py` loads reference code from `solutions/00_ml_gym`. "
            "Student fill tests in `test_student_fills.py` skip until you implement the function."
        ),
        _code(
            "from train import main as train_main\n"
            "metrics = train_main(TrainConfig(\n"
            "    data_dir=repo / 'data' / 'm00_sample',\n"
            "    epochs=2,\n"
            "    loss_name='cross_entropy',\n"
            "))\n"
            "metrics_path = repo / 'artifacts' / 'm00' / metrics['run_id'] / 'metrics.json'\n"
            "print('metrics path:', metrics_path)\n"
            "print('minority_recall field:', metrics['minority_recall'])"
        ),
        _md(
            "**Come back cue**\n\n"
            f"{come_back_cue('m00')}\n\n"
            "Suggested slot: 25 minutes. 55 or 90 if you are also writing the principle cells."
        ),
        _code(
            "import sys\nimport json\nfrom pathlib import Path\n\n"
            "repo = Path.cwd()\nif not (repo / 'modules' / '00_ml_gym').exists():\n"
            "    repo = repo.parent\n"
            "sys.path.insert(0, str(repo / 'modules'))\n"
            "from common.progress import come_back_cue\n\n"
            "print(come_back_cue('m00'))\n"
            "progress_path = repo / 'artifacts' / 'progress.json'\n"
            "if progress_path.is_file():\n"
            "    with progress_path.open() as f:\n"
            "        xp = json.load(f).get('xp', 0)\n"
            "    print(f'XP so far: {xp}')"
        ),
    ]
