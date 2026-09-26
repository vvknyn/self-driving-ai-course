#!/usr/bin/env python3
"""Build Module 00 and 01 primary notebooks (nbformat)."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def _md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source)


def _code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source)


def build_m00() -> nbf.NotebookNode:
    cells = [
        _md(
            "# Week 1 — Driving ML Gym\n\n"
            "This notebook is the **lecture path** for Week 1.\n\n"
            "- **Scaffold** (~60–70%): data loading, model, cross-entropy training loop\n"
            "- **Fill**: implement `focal_loss` and `minority_recall` in the module `.py` files\n"
            "- **From scratch**: implement `build_error_gallery` yourself\n\n"
            "Optional autograd appendix: `modules/00_nn_scratch` (treat as **appendix 00b**, not the default chapter).\n\n"
            "The crops are **synthetic checked-in PNGs** under `data/m00_sample` (CC0 license), "
            "not random `torch.randn` tensors."
        ),
        _md(
            "**Principle 1 — Model as function.**\n\n"
            "A classifier is a function from a crop to one score per class. "
            "Those scores are not probabilities until you apply softmax. "
            "Learning changes the weights so a chosen loss gets smaller on the training distribution. "
            "If that distribution is mostly empty road, the function gets good at road unless the loss says otherwise. "
            "This module is a function, a loss, and a dataset — not a claim that the net \"understands driving.\""
        ),
        _md(
            "**Principle 2 — Loss defines good.**\n\n"
            "The loss is the definition of \"good\" that gradient descent sees. "
            "Cross-entropy charges `-log(p_true)` on every crop, so a pile of easy road crops can outweigh a few pedestrians. "
            "Focal loss multiplies that charge by `(1 - p_true)^gamma`. "
            "When the model is already sure and correct, the factor goes to 0 and the easy crop stops dominating. "
            "When gamma is 0 the factor is 1, so focal loss is cross-entropy if there is no extra class weight. "
            "You will check that equality in code, not by trusting the function name."
        ),
        _md(
            "**Principle 3 — Inspect errors.**\n\n"
            "One accuracy number averages over classes. "
            "With many road crops and few pedestrians, a model that never predicts pedestrian can still look mostly right. "
            "A gallery of mistakes, sorted by how confident the wrong answer was, shows whether the model is confused or confidently wrong. "
            "Confident mistakes are the ones you would queue for more labels. "
            "You will build that gallery yourself; the scaffold does not do it for you."
        ),
        _code(
            "import sys\nfrom pathlib import Path\nimport torch\nimport matplotlib\nmatplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n\n"
            "repo = Path.cwd()\nif not (repo / 'modules' / '00_ml_gym').exists():\n"
            "    repo = repo.parent\n"
            "sys.path.insert(0, str(repo / 'modules' / '00_ml_gym'))\n"
            "torch.manual_seed(0)\n\n"
            "from config import TrainConfig\nfrom dataset import CLASS_NAMES, get_dataloaders\n"
            "from model import DrivingClassifier\nfrom losses import cross_entropy_loss, focal_loss\n"
            "from metrics import accuracy, per_class_recall, minority_recall\n"
            "print('repo:', repo)"
        ),
        _md(
            "**Tensor shapes.**\n\n"
            "**B** is the batch size — how many crops are processed together. "
            "**C** is the number of color channels (3 for RGB). "
            "**H** is image height in pixels. "
            "**W** is image width in pixels. "
            "PyTorch expects image batches as `(B, C, H, W)` while PNG files on disk are stored as `(H, W, C)`."
        ),
        _code(
            "cfg = TrainConfig(data_dir=repo / 'data' / 'm00_sample')\n"
            "train_loader, val_loader = get_dataloaders(cfg.data_dir, batch_size=16, seed=0)\n"
            "images, labels = next(iter(train_loader))\n"
            "assert images.shape[1:] == (3, 64, 64), images.shape\n"
            "assert labels.shape == (images.shape[0],), labels.shape\n"
            "print('batch', images.shape, 'labels', labels.shape)\n"
            "print('dtype', images.dtype, labels.dtype)\n"
            "print('pixel min/max', images.min().item(), images.max().item(), '(real PNG pixels in [0,1], not randn)')"
        ),
        _code(
            "from PIL import Image\nimport numpy as np\n"
            "ds = train_loader.dataset\nfig, axes = plt.subplots(2, 4, figsize=(8, 4))\n"
            "for ax, i in zip(axes.flat, range(8)):\n"
            "    img, lab = ds[i]\n    ax.imshow(img.permute(1, 2, 0))\n"
            "    ax.set_title(CLASS_NAMES[lab.item()], fontsize=8)\n    ax.axis('off')\n"
            "plt.suptitle('Sample crops (from PNG files)')\nplt.tight_layout()\nplt.show()"
        ),
        _code(
            "counts = train_loader.dataset.class_counts()\n"
            "n = sum(counts.values())\n"
            "count_road = counts[0]\n"
            "always_road_acc = count_road / n\n"
            "plt.bar(CLASS_NAMES, [counts[i] for i in range(4)])\n"
            "plt.title('Train class counts — accuracy will lie')\nplt.xticks(rotation=20)\nplt.show()\n"
            "print(f'Always-predict-clear_road accuracy: {always_road_acc:.3f} ({count_road}/{n})')\n"
            "print('Pedestrian recall of that classifier: 0.0 — the accuracy lie.')"
        ),
        _md(
            "**Derive cross-entropy.** Softmax turns logits `z_k` into class probabilities:\n\n"
            "$$p_k = \\frac{\\exp(z_k)}{\\sum_j \\exp(z_j)}$$\n\n"
            "Cross-entropy for true class $y$ is $-\\log p_y$. "
            "Minimizing CE is the same as maximizing the probability assigned to the true class."
        ),
        _code(
            "import torch.nn.functional as F\n"
            "z = torch.tensor([2.0, 1.0, 0.5, -1.0])\n"
            "y = torch.tensor(0)\n"
            "manual = -torch.log(F.softmax(z, dim=0)[y])\n"
            "ce = F.cross_entropy(z.unsqueeze(0), y.unsqueeze(0))\n"
            "print('manual CE', manual.item(), 'F.cross_entropy', ce.item())\n"
            "assert torch.allclose(manual, ce, atol=1e-6)"
        ),
        _code(
            "model = DrivingClassifier(num_classes=4)\n"
            "logits = model(images)\n"
            "assert logits.shape == (images.shape[0], 4)\n"
            "n_params = sum(p.numel() for p in model.parameters())\n"
            "print('logits', logits.shape, 'parameter count', n_params)"
        ),
        _md(
            "You are about to fit **cross-entropy** for four epochs. "
            "Watch **per-class recall**, not only accuracy. "
            "Pedestrian is class index **2** (`CLASS_NAMES[2]`)."
        ),
        _code(
            "from train import train_epoch, evaluate\nimport torch.optim as optim\n"
            "import numpy as np\n\n"
            "device = torch.device('cpu')\n"
            "model = DrivingClassifier().to(device)\n"
            "opt = optim.AdamW(model.parameters(), lr=1e-3)\n"
            "for ep in range(4):\n"
            "    train_epoch(model, train_loader, opt, cross_entropy_loss, device)\n"
            "    loss, acc, recalls = evaluate(model, val_loader, cross_entropy_loss, device, 4)\n"
            "    print(f'epoch {ep+1} val_acc={acc:.3f} recalls={[round(r, 2) for r in recalls]}')\n"
            "ce_ped_recall = recalls[2]\n\n"
            "# Confusion matrix on val split\n"
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
            "**Focal loss derivation.** Focal loss modulates cross-entropy with a focusing factor:\n\n"
            "$$\\mathrm{FL} = -(1 - p_t)^\\gamma \\log(p_t)$$\n\n"
            "When $\\gamma = 0$ the factor is 1, so focal loss reduces to cross-entropy. "
            "Numeric sketch: if $p_t = 0.99$ and $\\gamma = 2$, the factor is $0.0001$ — an easy correct crop barely contributes. "
            "If $p_t = 0.3$ and $\\gamma = 2$, the factor is about $0.49$ — a hard example still matters. "
            "$\\alpha$ is an optional per-class weight tensor of shape `(K,)` applied as `alpha[targets]`."
        ),
        _md(
            "**FILL — `focal_loss`** in `modules/00_ml_gym/losses.py` only.\n\n"
            "Inputs: logits `(B, K)` and targets `(B,)`. "
            "Do **not** paste a solution into this notebook. "
            "Hint: the stable route is `log_softmax`, but implement the function body in the `.py` file."
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
        _md(
            "**FILL — `minority_recall`** in `modules/00_ml_gym/metrics.py`.\n\n"
            "Recall = TP / support for one class; if support is 0, return 0. "
            "Hand example: preds `[2, 0, 2, 1]`, targets `[2, 2, 0, 2]`, minority class 2 → recall **1/3**."
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
            "Contract: images NCHW in `[0, 1]`, targets `(N,)`, logits `(N, K)`. "
            "Sort mistakes by descending confidence of the **wrong** class. "
            "Return a dict with keys `path`, `n_errors`, `order`; write a figure even when `n_errors=0`. "
            "Also add a test or un-skip behavior in `tests/test_student_fills.py` by implementing the function. "
            "Do not implement it in this notebook."
        ),
        _code(
            "from error_gallery import build_error_gallery\n"
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
            "Why does $\\gamma=0$ match cross-entropy?"
        ),
        _md("_Write 3–6 sentences here before you open `solutions/00_ml_gym`._"),
        _md(
            "**Free response (Principle 3):** The always-road classifier had high accuracy and zero pedestrian recall. "
            "Which principle says that metric was the wrong definition of good?"
        ),
        _md("_Write 3–6 sentences here._"),
        _code("from break_it_fix_it import main as break_demo\nbreak_demo()"),
        _md(
            "**Tests**\n\n"
            "```bash\npython3 -m pytest modules/00_ml_gym -q\n```\n\n"
            "Scaffold tests in `test_scaffold.py` always run and verify the provided code. "
            "`test_assignment_solutions.py` imports reference code from `solutions/00_ml_gym`; "
            "student fill tests in `test_student_fills.py` skip until you implement the function and fail if the implementation is wrong."
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
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


def build_m01() -> nbf.NotebookNode:
    cells = [
        _md(
            "# Week 2 — Cameras & IPM\n\n"
            "**1-camera IPM is required before the 3-camera stitch.** "
            "Student fills live in the module `.py` files.\n\n"
            "Synthetic calibrated frames are checked in under `data/m01_sample` (CC0), "
            "not captured from a real vehicle rig."
        ),
        _md(
            "**Principle 1 — Pinhole projection.**\n\n"
            "A pinhole maps a 3D ray to a pixel by similar triangles. "
            "A point at camera coordinates `(X, Y, Z)` hits the sensor at `x = f X / Z`, `y = f Y / Z`. "
            "Depth is in the denominator, so far objects shrink. "
            "Pixel coordinates add the principal point: `u = fx X/Z + cx`. "
            "`K` packs `fx`, `fy`, `cx`, `cy` into a 3×3 matrix. "
            "You will build `K` yourself."
        ),
        _md(
            "**Principle 2 — Homography on the road plane.**\n\n"
            "A homography is a 3×3 map between two planes. "
            "The road is the plane `Z=0` in the ego frame (X forward, Y left, Z up). "
            "A camera point is `P_cam = R P_ego + t`. Expanding `R` into columns `r1`, `r2`, `r3`, the `r3` term is multiplied by `Z`. "
            "On the road `Z` is 0, so `r3` drops out and `H = K [r1 r2 t]` maps `(X, Y, 1)` to homogeneous pixels. "
            "You will build that `H`. Using `R = I` on a pitched camera is the wrong plane-to-image map."
        ),
        _md(
            "**Principle 3 — Extrinsics set meters on the ground.**\n\n"
            "Meters on the ground come from extrinsics, not from the network. "
            "If pitch is wrong, the ray from a pixel hits `Z=0` in the wrong place. "
            "The same angular error covers more ground farther away, so the meter error grows with range. "
            "A 2° pitch bias that looks small in the image is a large longitudinal error at 40 m. "
            "You will measure that, not memorize a slogan."
        ),
        _code(
            "import sys\nimport json\nfrom pathlib import Path\nimport numpy as np\nimport cv2\n"
            "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n\n"
            "repo = Path.cwd()\nif not (repo / 'modules' / '01_camera_geometry').exists():\n"
            "    repo = repo.parent\n"
            "sys.path.insert(0, str(repo / 'modules' / '01_camera_geometry'))\n"
            "data = repo / 'data' / 'm01_sample'\n\n"
            "from camera_model import PinholeCamera, build_intrinsic_matrix\n"
            "from calibrate_rig import build_tesla_style_rig\n"
            "from ipm import IPMTransformer, build_ground_homography\n"
            "from stitch import stitch_three_cameras\n"
            "from pitch_sensitivity import pitch_shift_meters\n\n"
            "with open(data / 'calib.json') as f:\n"
            "    calib = json.load(f)\n"
            "frames = {k: cv2.imread(str(data / f'{k}.png')) for k in ('front', 'left', 'right')}\n"
            "for k, v in frames.items():\n"
            "    plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))\n"
            "    plt.title(k)\n    plt.axis('off')\n    plt.show()"
        ),
        _md(
            "**Similar triangles.** Pick focal length `f = 400` px and lateral offset `X = 10` m. "
            "Compute the image coordinate `x = f X / Z` at `Z = 20` m and at `Z = 40` m."
        ),
        _code(
            "f, X = 400.0, 10.0\n"
            "x_20 = f * X / 20.0\n"
            "x_40 = f * X / 40.0\n"
            "print('x at Z=20 m:', x_20)\n"
            "print('x at Z=40 m:', x_40)\n"
            "print('Doubling depth halves the image coordinate — the pinhole principle.')"
        ),
        _md(
            "**FILL — `build_intrinsic_matrix(fx, fy, cx, cy) -> (3, 3)`**\n\n"
            "$$K = \\begin{bmatrix} f_x & 0 & c_x \\\\ 0 & f_y & c_y \\\\ 0 & 0 & 1 \\end{bmatrix}$$\n\n"
            "Implement it in `modules/01_camera_geometry/camera_model.py`, not in this notebook."
        ),
        _code(
            "try:\n"
            "    K = build_intrinsic_matrix(calib['fx'], calib['fy'], calib['cx'], calib['cy'])\n"
            "    assert abs(K[0, 0] - calib['fx']) < 1e-6\n"
            "    assert abs(K[1, 1] - calib['fy']) < 1e-6\n"
            "    assert abs(K[0, 2] - calib['cx']) < 1e-6\n"
            "    assert abs(K[2, 2] - 1.0) < 1e-6\n"
            "    print('K ok\\n', K)\n"
            "except NotImplementedError:\n"
            "    print('STOP: implement build_intrinsic_matrix in camera_model.py')"
        ),
        _code(
            "cams = build_tesla_style_rig(calib['width'], calib['height'])\n"
            "front = cams['front']\n"
            "for pt in [(10, 0, 0), (20, 1.5, 0), (30, -1.5, 0)]:\n"
            "    pix, ok = front.project_ego_to_pixel([pt])\n"
            "    print(pt, '->', pix[0], 'valid', ok[0])"
        ),
        _md(
            "**Extrinsics: ego → camera.** `P_cam = R P_ego + T`. "
            "`R` rotates ego axes into the camera frame; `T` is the translation column (same convention as `PinholeCamera`)."
        ),
        _code(
            "print('front.R (rounded):\\n', np.round(front.R, 3))\n"
            "print('front.T.ravel():', front.T.ravel())\n"
            "print('calib pitch_deg:', calib['pitch_deg'])\n"
            "print('Pitch is not zero, so R is not a pure axis swap.')"
        ),
        _md(
            "**Derive the ground homography.** On the road plane `Z = 0`, the third column of the extrinsic block is unused:\n\n"
            "$$H = K \\,[r_1 \\; r_2 \\; t]$$\n\n"
            "`r3` is the column multiplied by `Z`, hence absent on the road."
        ),
        _md(
            "**FILL — `build_ground_homography(K, R, t)`** in `modules/01_camera_geometry/ipm.py`. "
            "Do not paste the solution here."
        ),
        _code(
            "try:\n"
            "    H = build_ground_homography(front.K, front.R, front.T)\n"
            "    pt = np.array([15.0, 0.0, 1.0])\n"
            "    uv = H @ pt\n    uv = uv[:2] / uv[2]\n"
            "    pix, _ = front.project_ego_to_pixel([[15, 0, 0]])\n"
            "    err_h = np.max(np.abs(uv - pix[0]))\n"
            "    print('H pixel', uv, 'project', pix[0], 'max abs err', err_h)\n"
            "    assert err_h < 1e-2\n"
            "    H_wrong = build_ground_homography(front.K, np.eye(3), front.T)\n"
            "    uv_w = H_wrong @ pt\n    uv_w = uv_w[:2] / uv_w[2]\n"
            "    err_wrong = np.linalg.norm(uv_w - pix[0])\n"
            "    print('R=I error px', err_wrong)\n"
            "    assert err_wrong > 5.0\n"
            "except NotImplementedError:\n"
            "    print('STOP: implement build_ground_homography in ipm.py')"
        ),
        _code(
            "ipm = IPMTransformer(front, x_range=(4, 40), y_range=(-10, 10), bev_resolution=0.1)\n"
            "bev = ipm.warp_to_bev(frames['front'])\n"
            "plt.imshow(cv2.cvtColor(bev, cv2.COLOR_BGR2RGB))\n"
            "plt.title('1-cam IPM')\nplt.axis('off')\nplt.show()"
        ),
        _code(
            "stitched = stitch_three_cameras(frames, cams)\n"
            "plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))\n"
            "plt.title('3-cam stitch')\nplt.axis('off')\nplt.show()"
        ),
        _md(
            "**FROM SCRATCH — `pitch_shift_meters(calib, delta_deg, range_m) -> float`** in "
            "`modules/01_camera_geometry/pitch_sensitivity.py`.\n\n"
            "Add `delta_deg` to the calib pitch, then return estimated forward range minus true range "
            "for the ground point `(range_m, 0, 0)`. "
            "`delta_deg = 0` should be ~0; absolute error should grow with range. "
            "Implement the function in the `.py` file — do not spell out the algorithm in this notebook."
        ),
        _code(
            "from extrinsics import create_euler_rotation, camera_position_to_translation\n\n"
            "ranges = [10, 20, 40]\n"
            "delta_deg = 2.0\n\n"
            "def _scaffold_pitch_shift(rng, delta):\n"
            "    pix, _ = front.project_ego_to_pixel([[rng, 0, 0]])\n"
            "    R2 = create_euler_rotation(calib['pitch_deg'] + delta, 0, 0)\n"
            "    T2 = camera_position_to_translation(R2, np.array(calib['cam_position_ego']))\n"
            "    cam2 = PinholeCamera(\n"
            "        'biased', front.fx, front.fy, front.cx, front.cy,\n"
            "        front.width, front.height, R2, T2,\n"
            "    )\n"
            "    est, _ = cam2.project_pixels_to_ground(pix)\n"
            "    return float(est[0, 0] - rng)\n\n"
            "try:\n"
            "    shifts = [pitch_shift_meters(calib, delta_deg, r) for r in ranges]\n"
            "    for r, s in zip(ranges, shifts):\n"
            "        print(f'range {r} m shift {s:.4f} m')\n"
            "    plt.plot(ranges, shifts, marker='o')\n"
            "    plt.xlabel('range (m)')\n"
            "    plt.ylabel('forward shift (m)')\n"
            "    plt.title('pitch_shift_meters (student fill)')\n"
            "    plt.grid(True, alpha=0.3)\n"
            "    plt.show()\n"
            "    assert abs(shifts[2]) > abs(shifts[0])\n"
            "except NotImplementedError:\n"
            "    print('STOP: implement pitch_shift_meters in pitch_sensitivity.py')\n"
            "    scaffold = [_scaffold_pitch_shift(r, delta_deg) for r in ranges]\n"
            "    for r, s in zip(ranges, scaffold):\n"
            "        print(f'[scaffold] range {r} m shift {s:.4f} m')\n"
            "    plt.plot(ranges, scaffold, marker='o', linestyle='--')\n"
            "    plt.xlabel('range (m)')\n"
            "    plt.ylabel('forward shift (m)')\n"
            "    plt.title('Reference scaffold (not student pitch_shift_meters)')\n"
            "    plt.grid(True, alpha=0.3)\n"
            "    plt.show()"
        ),
        _md(
            "**Free response:** Which principle explains the smear when pitch is wrong? "
            "Why does the meter error grow with range?"
        ),
        _md("_Write 3–6 sentences here._"),
        _md(
            "**Assignment:** The rig's pitch is 2° off the truth. "
            "What would you change so a lane point at 25 m lands back on the correct ground coordinate? "
            "See `python modules/01_camera_geometry/break_it_fix_it.py`."
        ),
        _md("_Write 3–6 sentences here._"),
        _code(
            "from calibrate_rig import main as calib_main\n"
            "m01_metrics = calib_main()\n"
            "print('metrics outputs:', m01_metrics.get('outputs', {}))\n"
            "print('run_id:', m01_metrics.get('run_id'))"
        ),
        _md(
            "**Tests**\n\n"
            "```bash\npython3 -m pytest modules/01_camera_geometry -q\n```\n\n"
            "Scaffold tests always run. "
            "`test_assignment_solutions.py` imports `solutions/01_camera_geometry`; "
            "student fill tests skip until implemented and fail if the implementation is wrong."
        ),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    nbf.write(build_m00(), root / "notebooks" / "00_driving_ml_gym.ipynb")
    nbf.write(build_m01(), root / "notebooks" / "01_cameras_and_ipm.ipynb")
    print("Wrote notebooks/00_driving_ml_gym.ipynb and notebooks/01_cameras_and_ipm.ipynb")
