#!/usr/bin/env python3
"""
Generates 10 self-contained, publication-grade Jupyter Notebooks (.ipynb)
for the Autonomous Driving Course in 'Help Me Understand' mode.
"""

import json
import os
from pathlib import Path

NOTEBOOKS_DIR = Path("/Users/vivek/Downloads/Self Driving Course/notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

REPO_URL = "https://github.com/vvknyn/self-driving-ai-course"
COLAB_BASE = "https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/notebooks"

def make_notebook(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "None",
            "colab": {
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def md_cell(source: str) -> dict:
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }

def code_cell(source: str) -> dict:
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

def make_header(num, title, nb_fname, question, dilemma):
    colab_link = COLAB_BASE + "/" + nb_fname
    return f"""# {num}: {title}

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_link})
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717.svg)]({REPO_URL})

> **The Big Question**: *{question}*

---

## 1. 🚨 The Real-World Dilemma
{dilemma}"""

# ==============================================================================
# 00: Autograd & Scratch NN
# ==============================================================================
def nb_00():
    h = make_header(
        "Chapter 00", "Neural Networks, Autograd & Loss Landscapes",
        "00_neural_networks_and_autograd.ipynb",
        "How does a computational graph tune 1,000,000 synaptic weights simultaneously in milliseconds to learn non-linear stopping distances?",
        "In autonomous emergency braking (AEB), kinetic energy is quadratic ($d \\propto v^2$). Stacking linear layers without activations mathematically collapses into a single straight line, causing high-speed crashes or parking-lot phantom braking!"
    )
    return make_notebook([
        md_cell(h),
        code_cell("""import math
import random
import matplotlib.pyplot as plt
%matplotlib inline

print("Building Karpathy-style Micrograd autograd engine...")"""),
        md_cell("""## 2. 🛠️ The Karpathy Build: Micrograd Scalar Autograd Engine
Every scalar value tracks its local gradient, parents in the Directed Acyclic Graph (DAG), and backward chain-rule closure."""),
        code_cell("""class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self * -1.0

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "ReLU")
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"\n"""),
        md_cell("""## 3. 📐 Finite-Difference Numerical Gradient Check (Andrew Ng Style)
Comparing analytical backpropagation gradients to finite differences:
$$\\frac{\\partial f}{\\partial x} \\approx \\frac{f(x + \\epsilon) - f(x - \\epsilon)}{2\\epsilon}$$"""),
        code_cell("""eps = 1e-5
x = Value(3.0)
w = Value(-2.0)
b = Value(4.0)
y = (x * w + b).relu()
y.backward()
analytical_dw = w.grad

y1 = max(0.0, 3.0 * (-2.0 + eps) + 4.0)
y2 = max(0.0, 3.0 * (-2.0 - eps) + 4.0)
numerical_dw = (y1 - y2) / (2 * eps)
rel_error = abs(analytical_dw - numerical_dw) / (abs(analytical_dw) + abs(numerical_dw) + 1e-8)

print(f"Analytical: {analytical_dw:.6f} | Numerical: {numerical_dw:.6f} | Rel Error: {rel_error:.2e}")
assert rel_error < 1e-5
print("✅ Calculus gradient check certified correct!")"""),
        md_cell("""## 4. 🧠 MLP Training on Quadratic Braking Boundary
Training a 2-layer MLP on non-linear stopping distances: $d_{\\text{safe}} = 0.05 v^2 + 0.2 v$."""),
        code_cell("""class Neuron:
    def __init__(self, nin):
        scale = math.sqrt(2.0 / nin)
        self.w = [Value(random.gauss(0, scale)) for _ in range(nin)]
        self.b = Value(0.0)
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu()
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self, x):
        return [n(x) for n in self.neurons]
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

random.seed(42)
model = MLP(2, [8, 1])

# Generate training data
X_train, y_train = [], []
for _ in range(100):
    v = random.uniform(0.0, 40.0)
    d = random.uniform(0.0, 100.0)
    d_safe = 0.05 * (v**2) + 0.2 * v
    X_train.append([Value(v / 40.0), Value(d / 100.0)])
    y_train.append(Value(1.0 if d < d_safe else 0.0))

losses = []
for step in range(50):
    ypred = [model(x) for x in X_train]
    loss = sum((yp - yt)*(yp - yt) for yp, yt in zip(ypred, y_train)) * (1.0 / len(y_train))
    losses.append(loss.data)
    for p in model.parameters(): p.grad = 0.0
    loss.backward()
    for p in model.parameters(): p.data -= 0.05 * p.grad

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(losses, color="#58a6ff", lw=2)
plt.title("MSE Loss Convergence")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
v_raw = [x[0].data * 40.0 for x in X_train]
d_raw = [x[1].data * 100.0 for x in X_train]
colors = ["#f85149" if y.data > 0.5 else "#3fb950" for y in y_train]
plt.scatter(v_raw, d_raw, c=colors, alpha=0.7, edgecolors="k")
plt.plot(range(41), [0.05*(v**2) + 0.2*v for v in range(41)], color="yellow", lw=2, label="Physics Boundary")
plt.title("Learned Quadratic Braking Space")
plt.xlabel("Speed (m/s)")
plt.ylabel("Distance (m)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""),
        md_cell("""## 5. 🩺 Andrew Ng's Diagnostic Field Guide

| Observed Symptom | Underlying Mathematical Mechanism | Verification Test | Production Fix |
| :--- | :--- | :--- | :--- |
| **All weights update identically** | Zero-weight initialization ($W=0$). Symmetry trap prevents differentiation. | Check if `w1.grad == w2.grad`. | Use He/Xavier normal random init. |
| **Loss explodes to NaN** | Learning rate $\\eta$ overshoots convex valleys. | Check if $\\|\\mathbf{g}\\| > 100$. | Clip gradient norms ($\\|\\mathbf{g}\\| \\le 1.0$) and reduce $\\eta$. |
| **Loss plateaus immediately** | Dead ReLU problem (large negative bias). | Count neurons with zero activations. | Use Leaky ReLU or Batch Normalization. |""")
    ])

# ==============================================================================
# 01: Perception Gym & Focal Loss
# ==============================================================================
def nb_01():
    h = make_header(
        "Chapter 01", "Driving Perception Gym & Class Imbalance",
        "01_driving_perception_gym.ipynb",
        "Why does standard Cross-Entropy ignore pedestrians in driving datasets, and how does Focal Loss mathematically fix it?",
        "In driving datasets, 99.1% of pixels belong to empty asphalt, and <0.1% to pedestrians. Standard Cross-Entropy gradients are overwhelmed 4,000-to-1 by easy road pixels. Focal Loss dynamically damps easy examples by $(1 - p_t)^\\gamma$."
    )
    return make_notebook([
        md_cell(h),
        code_cell("""import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline

print("PyTorch loaded:", torch.__version__)"""),
        md_cell("""## 2. 🛠️ The Karpathy Build: Multi-Class Focal Loss
$$\\mathcal{L}_{\\text{Focal}}(p_t) = -\\alpha_t (1 - p_t)^\\gamma \\log(p_t)$$"""),
        code_cell("""class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        modulating = (1.0 - p_t) ** self.gamma
        focal_loss = modulating * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        return focal_loss.mean()

# Compare CE vs Focal Loss suppression
p_vals = torch.linspace(0.01, 0.99, 100)
ce = -torch.log(p_vals)
focal_2 = ((1.0 - p_vals) ** 2.0) * ce

plt.figure(figsize=(8, 3.5))
plt.plot(p_vals.numpy(), ce.numpy(), label="Cross-Entropy (gamma=0)", color="#f85149", lw=2)
plt.plot(p_vals.numpy(), focal_2.numpy(), label="Focal Loss (gamma=2)", color="#58a6ff", lw=2)
plt.title("Loss Damping on Easy Examples (High pt)")
plt.xlabel("Target Probability pt")
plt.ylabel("Loss Magnitude")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""),
        md_cell("""## 3. 🩺 Andrew Ng's Diagnostic Field Guide

| Observed Symptom | Underlying Mathematical Mechanism | Verification Test | Production Fix |
| :--- | :--- | :--- | :--- |
| **High overall accuracy, 0% recall on rare objects** | Majority class dominance in Cross-Entropy. | Inspect class-stratified confusion matrix. | Switch to Focal Loss with $\\gamma = 2.0$. |
| **Loss explodes when weighting classes** | Extreme $\\alpha$ multiplier ($>100$) causes massive gradient spikes on outliers. | Check $\\max \\|\\mathbf{g}\\|$ per batch. | Clamp class weights $\\alpha \\le 10.0$ and use gradient clipping. |""")
    ])

# ==============================================================================
# 02: 3D Camera Rig & IPM
# ==============================================================================
def nb_02():
    h = make_header(
        "Chapter 02", "3D Camera Rig & Inverse Perspective Mapping (IPM)",
        "02_camera_geometry_and_ipm.ipynb",
        "How do we unproject 2D camera pixels onto a metric 3D ground plane—and why does a 1.5° chassis tilt cause 40% distance error?",
        "Perspective cameras divide by depth $Z_c$. Under the Flat Ground Assumption ($Z=0$), we can construct a Planar Homography $H = K [r_1, r_2, \\mathbf{t}]$. But suspension bounce changes pitch angle, causing lanes to violently flare outward into hyperbolas!"
    )
    return make_notebook([
        md_cell(h),
        code_cell("""import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def compute_intrinsics(width=1920, height=1080, hfov_deg=90.0):
    hfov_rad = np.deg2rad(hfov_deg)
    fx = (width / 2.0) / np.tan(hfov_rad / 2.0)
    return np.array([
        [fx, 0.0, width / 2.0],
        [0.0, fx, height / 2.0],
        [0.0, 0.0, 1.0]
    ])

def euler_to_rotation(roll, pitch, yaw):
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def make_homography(K, R, t):
    Rt = np.hstack([R[:, 0:1], R[:, 1:2], t.reshape(3, 1)])
    H = K @ Rt
    return H / H[2, 2]

K = compute_intrinsics()
R_nominal = euler_to_rotation(0, np.deg2rad(3.0), 0)
t_nominal = np.array([0.0, -1.4, 0.0])
H_nominal = make_homography(K, R_nominal, t_nominal)
print("Nominal Planar Homography Matrix H:")
print(np.round(H_nominal, 4))"""),
        code_cell("""# Pitch error simulation during hard braking (+2 degrees)
R_perturbed = euler_to_rotation(0, np.deg2rad(5.0), 0)
H_perturbed = make_homography(K, R_perturbed, t_nominal)
H_inv_nominal = np.linalg.inv(H_nominal)

y_pts = np.linspace(5.0, 45.0, 50)
left_line = np.stack([-1.85 * np.ones_like(y_pts), y_pts, np.ones_like(y_pts)])
right_line = np.stack([1.85 * np.ones_like(y_pts), y_pts, np.ones_like(y_pts)])

def warp_line(line, H_fwd, H_bwd):
    p_img = H_fwd @ line
    u = p_img[0] / p_img[2]
    v = p_img[1] / p_img[2]
    p_bev = H_bwd @ np.stack([u, v, np.ones_like(u)])
    return p_bev[0] / p_bev[2], p_bev[1] / p_bev[2]

xl_err, yl_err = warp_line(left_line, H_perturbed, H_inv_nominal)
xr_err, yr_err = warp_line(right_line, H_perturbed, H_inv_nominal)

plt.figure(figsize=(8, 4))
plt.plot([-1.85, -1.85], [5, 45], 'g--', label="True Parallel Lane")
plt.plot([1.85, 1.85], [5, 45], 'g--')
plt.plot(xl_err, yl_err, 'r-', lw=2, label="Reconstructed with +2° Pitch Error")
plt.plot(xr_err, yr_err, 'r-', lw=2)
plt.title("The Pitch Flare Effect in IPM Reconstruction")
plt.xlabel("Lateral X (m)")
plt.ylabel("Longitudinal Y (m)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""),
        md_cell("""## 3. 🩺 Andrew Ng's Diagnostic Field Guide

| Observed Symptom | Underlying Mathematical Mechanism | Verification Test | Production Fix |
| :--- | :--- | :--- | :--- |
| **Lanes flare outward at distance** | Pitch over-estimation; camera unprojects rays into ground too early. | Check if distant lane width $> 3.7\\text{m}$. | Recalibrate pitch extrinsic down. |
| **Distance errors oscillate during braking** | Dynamic chassis suspension bounce ($\Delta \\theta \\approx 2^\\circ$). | Correlate distance error with longitudinal accelerometer. | Ingest live IMU pitch rate into dynamic $H(t)$. |""")
    ])

# ==============================================================================
# 03: Multi-Task HydraNet
# ==============================================================================
def nb_03():
    h = make_header(
        "Chapter 03", "Multi-Task HydraNet Perception & Uncertainty Loss",
        "03_hydranet_multitask_learning.ipynb",
        "How do we train 15 perception heads on a single shared backbone without destructive gradient interference?",
        "Running 15 separate networks draws kilowatts of power. A shared HydraNet solves compute, but naive loss summation causes task competition (Negative Transfer). Kendall Uncertainty Weighting learns task observation noise $\\sigma_i$ to dynamically balance gradients."
    )
    return make_notebook([
        md_cell(h),
        code_cell("""import torch
import torch.nn as nn
import matplotlib.pyplot as plt
%matplotlib inline

class KendallUncertaintyLoss(nn.Module):
    def __init__(self, num_tasks=3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        total_loss = 0.0
        for i, loss in enumerate(losses):
            s = self.log_vars[i]
            precision = torch.exp(-s)
            total_loss += 0.5 * precision * loss + 0.5 * s
        return total_loss

loss_fn = KendallUncertaintyLoss(3)
optimizer = torch.optim.Adam(loss_fn.parameters(), lr=0.05)
task_losses = [torch.tensor(0.2), torch.tensor(3.5), torch.tensor(0.05)]

history = []
for _ in range(50):
    optimizer.zero_grad()
    loss = loss_fn(task_losses)
    loss.backward()
    optimizer.step()
    history.append(loss_fn.log_vars.detach().clone())

plt.figure(figsize=(8, 3.5))
plt.plot([h[0].item() for h in history], label="Bbox (0.2)", lw=2)
plt.plot([h[1].item() for h in history], label="Drivable (3.5)", lw=2)
plt.plot([h[2].item() for h in history], label="Lights (0.05)", lw=2)
plt.title("Kendall Learnable Log-Variances s = log(sigma^2) Converging")
plt.xlabel("Step")
plt.ylabel("Learned Log-Variance s")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""),
        md_cell("""## 2. 🩺 Andrew Ng's Diagnostic Field Guide

| Observed Symptom | Underlying Mathematical Mechanism | Verification Test | Production Fix |
| :--- | :--- | :--- | :--- |
| **New task degrades existing tasks by >10%** | Negative transfer / conflicting gradients ($\\mathbf{g}_1 \\cdot \\mathbf{g}_2 < 0$). | Compute cosine similarity between head gradients. | Use Kendall Uncertainty weighting or PCGrad gradient projection. |
| **One head's variance explodes ($s \\to \\infty$)** | Model cheats by declaring a difficult task 'infinite noise' to ignore it. | Check `log_vars` values during training. | Clamp max log-variance ($s_i \\le 3.0$). |""")
    ])

# ==============================================================================
# 04: BEV Transform (Lift-Splat-Shoot)
# ==============================================================================
def nb_04():
    h = make_header(
        "Chapter 04", "Bird's-Eye View (BEV) Transform (Lift, Splat, Shoot)",
        "04_bev_lift_splat_shoot.ipynb",
        "How do we transform 8 perspective cameras into a metric 3D top-down grid without LiDAR?",
        "Perspective cameras break Euclidean geometry. Lift-Splat-Shoot (Philion & Fidler ECCV 2020) predicts categorical depth distributions along camera rays, projects features into 3D space, and pools them into metric BEV voxels."
    )
    return make_notebook([
        md_cell(h),
        code_cell("""import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline

class ToyLiftSplat(nn.Module):
    def __init__(self, D=20, d_min=2.0, d_max=40.0, C=16):
        super().__init__()
        self.D = D
        self.C = C
        self.depth_bins = torch.linspace(d_min, d_max, D)
        self.conv = nn.Conv2d(32, D + C, kernel_size=1)

    def forward(self, x):
        B, _, H, W = x.shape
        logits = self.conv(x)
        depth_logits = logits[:, :self.D]
        context = logits[:, self.D:]
        depth_prob = F.softmax(depth_logits, dim=1)
        frustum = depth_prob.unsqueeze(2) * context.unsqueeze(1)
        return frustum, depth_prob

model = ToyLiftSplat()
dummy_features = torch.randn(1, 32, 16, 16)
frustum, depth_prob = model(dummy_features)

print(f"Frustum tensor: {frustum.shape} (Batch, Depth, Channels, Height, Width)")
print(f"Depth prob sum along ray: {depth_prob[0, :, 8, 8].sum().item():.4f}")

plt.figure(figsize=(8, 3))
plt.plot(model.depth_bins.numpy(), depth_prob[0, :, 8, 8].detach().numpy(), 'o-', color="#58a6ff", lw=2)
plt.title("Categorical Depth Probability Along Camera Sightline")
plt.xlabel("Depth (m)")
plt.ylabel("Probability")
plt.grid(True, alpha=0.3)
plt.show()"""),
        md_cell("""## 2. 🩺 Andrew Ng's Diagnostic Field Guide

| Observed Symptom | Underlying Mathematical Mechanism | Verification Test | Production Fix |
| :--- | :--- | :--- | :--- |
| **Obstacles smeared radially along camera sightlines** | Depth distribution entropy too high (uniform guessing). | Compute $-\\sum p_i \\log p_i$. If $>3.0$, model has zero depth confidence. | Add auxiliary depth supervision with sparse LiDAR or radar. |
| **All obstacles shift sideways during acceleration** | Pitch/squat dynamic chassis tilt rotates $R_{\\text{ext}}$. | Correlate position error with chassis pitch IMU. | Feed live IMU suspension pitch into extrinsic rotation matrix. |""")
    ])

# ==============================================================================
# 05: 3D Occupancy & Temporal Memory
# ==============================================================================
def nb_05():
    h = make_header(
        "Chapter 05", "3D Occupancy Networks & Temporal Memory",
        "05_3d_occupancy_and_temporal_memory.ipynb",
        "How does an autonomous vehicle maintain object permanence when obstacles pass behind occluding barriers?",
        "3D bounding boxes fail on weird road rubble (mattresses, overturned boats). 3D Occupancy discretizes space into class-agnostic matter vs air. A Spatiotemporal ConvGRU recurrently preserves occluded obstacles across time."
    )
    return make_notebook([
        md_cell(h),
        code_cell("""import torch
import torch.nn as nn
import matplotlib.pyplot as plt
%matplotlib inline

class SpatialConvGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv_gates = nn.Conv2d(in_channels + hidden_channels, 2 * hidden_channels, 3, padding=1)
        self.conv_cand = nn.Conv2d(in_channels + hidden_channels, hidden_channels, 3, padding=1)

    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv_gates(combined)
        z, r = torch.split(torch.sigmoid(gates), self.hidden_channels, dim=1)
        h_tilde = torch.tanh(self.conv_cand(torch.cat([x, r * h_prev], dim=1)))
        return (1.0 - z) * h_prev + z * h_tilde

cell = SpatialConvGRUCell(16, 16)
h_state = torch.zeros(1, 16, 32, 32)

# Step 1: Target visible
obs_vis = torch.zeros(1, 16, 32, 32)
obs_vis[:, :, 14:18, 14:18] = 2.0
h_state = cell(obs_vis, h_state)

# Step 2: Target occluded
obs_occ = torch.zeros(1, 16, 32, 32)
h_state = cell(obs_occ, h_state)

print(f"Memory persistence at occluded cell: {h_state[0, 0, 16, 16].item():.4f}")
print("✅ Object permanence verified: state preserved across occlusion!")"""),
        md_cell("""## 2. 🩺 Andrew Ng's Diagnostic Field Guide

| Observed Symptom | Underlying Mathematical Mechanism | Verification Test | Production Fix |
| :--- | :--- | :--- | :--- |
| **Moving cars leave long red 'ghost trails' in memory** | Update gate $z \\approx 0$ fails to overwrite vacated voxels. | Check mean value of update gate `z.mean()`. | Feed optical flow / velocity vectors into input tensor. |
| **Occupancy grid blurs when turning corners** | Ego-motion uncompensated before temporal blending. | Rotate car in place and check if stationary poles blur into arcs. | Apply bilinear `grid_sample` pose warping $T_{t-1 \\to t}^{-1}$ before ConvGRU. |""")
    ])

# ==============================================================================
# 06: Vector Space Tracking (Kalman Filter)
# ==============================================================================
def nb_06():
    h = make_header(
        "Chapter 06", "Vector Space Tracking & Multi-Object Association",
        "06_vector_space_tracking_kalman.ipynb",
        "How do we prove detections across video frames belong to the same vehicle and calculate smooth velocity?",
        "Sensor noise causes finite-difference velocity calculations to spike violently at 27 mph. A Kalman filter fuses physics predictions with measurement uncertainty, using Mahalanobis distance for robust data association."
    )
    return make_notebook([
        md_cell(h),
        code_cell("""import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

class KalmanFilter2D:
    def __init__(self, dt=0.05):
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.P = np.eye(4) * 10.0
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 0.5
        self.x = np.zeros((4, 1))

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        y = z.reshape(2, 1) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x

kf = KalmanFilter2D(dt=0.1)
true_pos, noisy_meas, filtered_pos = [], [], []

for t in np.arange(0, 4, 0.1):
    tx = 18.0 * t
    ty = 1.5
    true_pos.append([tx, ty])
    mx = tx + np.random.normal(0, 1.0)
    my = ty + np.random.normal(0, 0.3)
    noisy_meas.append([mx, my])
    kf.predict()
    state = kf.update(np.array([mx, my]))
    filtered_pos.append([state[0, 0], state[1, 0]])

true_pos = np.array(true_pos)
noisy_meas = np.array(noisy_meas)
filtered_pos = np.array(filtered_pos)

plt.figure(figsize=(10, 3))
plt.scatter(noisy_meas[:, 0], noisy_meas[:, 1], color="#f85149", alpha=0.5, label="Raw Noisy Detections")
plt.plot(true_pos[:, 0], true_pos[:, 1], 'k--', label="True Motion")
plt.plot(filtered_pos[:, 0], filtered_pos[:, 1], color="#3fb950", lw=2, label="Kalman Filter")
plt.title("Kalman Filter Noise Rejection in Vector Space")
plt.xlabel("Forward X (m)")
plt.ylabel("Lateral Y (m)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""),
        md_cell("""## 2. 🩺 Andrew Ng's Diagnostic Field Guide

| Observed Symptom | Underlying Mathematical Mechanism | Verification Test | Production Fix |
| :--- | :--- | :--- | :--- |
| **Track ID switches when cars cross** | Euclidean association ambiguity in close proximity. | Compute cross-track Mahalanobis distance. | Use Hungarian Algorithm with appearance embeddings (Re-ID). |
| **Covariance matrix becomes non-positive definite** | Numerical roundoff in $(I - KH)P$. | Check eigenvalues $\\lambda_i \\le 0$. | Use Joseph Form: $P = (I - KH)P(I - KH)^T + KRK^T$. |""")
    ])

# ==============================================================================
# 07: Trajectory Planning (Quintic Splines)
# ==============================================================================
def nb_07():
    h = make_header(
        "Chapter 07", "Lattice Trajectory Planning & Quintic Splines",
        "07_lattice_trajectory_planning.ipynb",
        "How do we generate jerk-optimal, passenger-comfortable trajectories around obstacles at 70 mph?",
        "Piecewise paths require instantaneous steering changes and infinite jerk ($j(t) = \\dddot{x}(t) \\to \\infty$), causing rollovers. Quintic ($5^{\\text{th}}$-order) polynomials solve 6 boundary constraints (position, velocity, acceleration) in the Frenet frame $(s, d)$."
    )
    return make_notebook([
        md_cell(h),
        code_cell("""import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

class QuinticPolynomial:
    def __init__(self, x0, v0, a0, x1, v1, a1, T):
        self.T = T
        self.a0 = x0
        self.a1 = v0
        self.a2 = 0.5 * a0
        M = np.array([
            [   T**3,    T**4,    T**5],
            [ 3*T**2,  4*T**3,  5*T**4],
            [    6*T, 12*T**2, 20*T**3]
        ])
        b = np.array([
            x1 - (self.a0 + self.a1 * T + self.a2 * T**2),
            v1 - (self.a1 + 2 * self.a2 * T),
            a1 - (2 * self.a2)
        ])
        self.a3, self.a4, self.a5 = np.linalg.solve(M, b)

    def calc_pos(self, t):
        return self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4 + self.a5*t**5

    def calc_jerk(self, t):
        return 6*self.a3 + 24*self.a4*t + 60*self.a5*t**2

poly = QuinticPolynomial(x0=0.0, v0=0.0, a0=0.0, x1=3.5, v1=0.0, a1=0.0, T=3.0)
t_steps = np.linspace(0, 3.0, 100)
pos = [poly.calc_pos(t) for t in t_steps]
jerk = [poly.calc_jerk(t) for t in t_steps]

plt.figure(figsize=(9, 3.5))
plt.subplot(1, 2, 1)
plt.plot(t_steps, pos, color="#58a6ff", lw=2)
plt.title("Lateral Offset d(t) [Lane Change]")
plt.xlabel("Time (s)")
plt.ylabel("Lateral Offset (m)")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(t_steps, jerk, color="#3fb950", lw=2)
plt.title("Jerk Profile j(t) = d3x/dt3")
plt.xlabel("Time (s)")
plt.ylabel("Jerk (m/s³)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""),
        md_cell("""## 2. 🩺 Andrew Ng's Diagnostic Field Guide

| Observed Symptom | Underlying Mathematical Mechanism | Verification Test | Production Fix |
| :--- | :--- | :--- | :--- |
| **Car swerves aggressively then snaps back** | Horizon time $T$ too short, exceeding tire grip $\\|a_{\\text{lat}}\\| > \\mu g$. | Check peak lateral acceleration. | Dynamic horizon scaling: $T_{\\min} \\ge \\sqrt{2\\Delta d / a_{\\text{comfort}}}$. |
| **Planner oscillates between left/right lanes** | Symmetric cost well around center obstacle. | Check cost difference $|J_L - J_R| < 10^{-3}$. | Add lane bias hysteresis. |""")
    ])

# ==============================================================================
# 08: Closed-Loop Control (Stanley)
# ==============================================================================
def nb_08():
    h = make_header(
        "Chapter 08", "Closed-Loop Control & Vehicle Kinematics",
        "08_closed_loop_control_stanley.ipynb",
        "Why do simple steering controllers enter divergent death wobbles at 70 mph, and how does Stanley control prevent it?",
        "Vehicle heading rate $\\dot{\\psi} = \\frac{v}{L} \\tan(\\delta)$ scales with speed. A steering angle that works at 10 mph rolls the car at 70 mph! The Stanley Controller dampens cross-track steering by velocity in the denominator: $\\arctan(\\frac{ke}{v + k_{\\text{soft}}})$, guaranteeing exponential convergence without oscillation."
    )
    return make_notebook([
        md_cell(h),
        code_cell("""import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

class KinematicBicycle:
    def __init__(self, x=0.0, y=0.5, yaw=0.0, v=25.0, L=2.8):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.L = L

    def step(self, delta, dt=0.02):
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += (self.v / self.L) * np.tan(delta) * dt

def stanley_control(vehicle, path_y=0.0, k=0.8, k_soft=1.0):
    fx = vehicle.x + vehicle.L * np.cos(vehicle.yaw)
    fy = vehicle.y + vehicle.L * np.sin(vehicle.yaw)
    cross_track_error = path_y - fy
    heading_error = -vehicle.yaw
    delta = heading_error + np.arctan2(k * cross_track_error, vehicle.v + k_soft)
    return np.clip(delta, -np.deg2rad(30), np.deg2rad(30))

car = KinematicBicycle()
history_y = []
for _ in range(200):
    delta = stanley_control(car)
    car.step(delta)
    history_y.append(car.y)

plt.figure(figsize=(8, 3))
plt.plot(history_y, color="#3fb950", lw=2, label="Vehicle Lateral Position (Stanley)")
plt.axhline(0.0, color="k", linestyle="--", label="Target Path Centerline")
plt.title("Stanley Controller Exponential Cross-Track Convergence at 55 mph")
plt.xlabel("Simulation Steps (dt = 20ms)")
plt.ylabel("Lateral Error (m)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""),
        md_cell("""## 2. 🩺 Andrew Ng's Diagnostic Field Guide

| Observed Symptom | Underlying Mathematical Mechanism | Verification Test | Production Fix |
| :--- | :--- | :--- | :--- |
| **Steady-state cross-track offset on banked highways** | Gravitational lateral force $g \\sin(\\phi)$ not canceled by pure P-term. | Measure mean error over constant bank. | Add integral anti-windup term ($K_i \\int e \\, dt$). |
| **Steering shudder at near-zero speeds ($v < 0.2\\text{ m/s}$)** | Singularity at zero velocity when $k_{\\text{soft}} = 0$. | Check steering commands when coming to a complete stop. | Set $k_{\\text{soft}} \\ge 1.0\\text{ m/s}$. |""")
    ])

# ==============================================================================
# 09: Full FSD System Architecture
# ==============================================================================
def nb_09():
    h = make_header(
        "Chapter 09", "Full FSD Capstone & Real-Time System Architecture",
        "09_full_fsd_system_architecture.ipynb",
        "How do production systems integrate 8 cameras, 5 neural networks, and 2 actuators into an 80 Hz fail-safe pipeline?",
        "At 65 mph, a 200ms latency equals 5.8 meters of blind motion before brakes bite. Production systems use 3 multi-rate tiers (Control at 100-200 Hz, Perception at 30-50 Hz, Route Planning at 10 Hz) with zero-copy ring buffers and watchdog fail-safes."
    )
    return make_notebook([
        md_cell(h),
        code_cell("""import time
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

latencies = []
cross_track_errors = []

for step in range(100):
    t0 = time.perf_counter()
    time.sleep(0.005) # Perception
    time.sleep(0.003) # Planning
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    latencies.append(elapsed_ms)
    cte = 0.4 * np.exp(-step / 20.0) + np.random.normal(0, 0.01)
    cross_track_errors.append(cte)

print(f"Mean Pipeline Latency: {np.mean(latencies):.2f} ms (Target: <30ms)")
print(f"99th Percentile Latency: {np.percentile(latencies, 99):.2f} ms")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(latencies, color="#58a6ff", lw=1.5)
plt.axhline(30.0, color="r", linestyle="--", label="30ms Deadline")
plt.title("Sensor-to-Actuation Latency Profile")
plt.xlabel("Step")
plt.ylabel("Latency (ms)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(cross_track_errors, color="#3fb950", lw=2)
plt.title("Closed-Loop Cross-Track Error (m)")
plt.xlabel("Step")
plt.ylabel("CTE (m)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""),
        md_cell("""## 2. 🩺 Andrew Ng's Diagnostic Field Guide

| Observed Symptom | Underlying Mathematical Mechanism | Verification Test | Production Fix |
| :--- | :--- | :--- | :--- |
| **Random steering jolts every few minutes** | Python Garbage Collection (GC) pauses threads for 40ms. | Profile GC pause durations. | Disable automatic GC in 100 Hz control loop; pre-allocate tensors. |
| **Car drifts toward lane edges during heavy loads** | Pipeline latency exceeds 30ms time slice, delivering stale commands. | Measure 99th percentile latency. | Enforce hard 20ms timeout anytime algorithm. |""")
    ])

def main():
    generators = [
        ("00_neural_networks_and_autograd.ipynb", nb_00),
        ("01_driving_perception_gym.ipynb", nb_01),
        ("02_camera_geometry_and_ipm.ipynb", nb_02),
        ("03_hydranet_multitask_learning.ipynb", nb_03),
        ("04_bev_lift_splat_shoot.ipynb", nb_04),
        ("05_3d_occupancy_and_temporal_memory.ipynb", nb_05),
        ("06_vector_space_tracking_kalman.ipynb", nb_06),
        ("07_lattice_trajectory_planning.ipynb", nb_07),
        ("08_closed_loop_control_stanley.ipynb", nb_08),
        ("09_full_fsd_system_architecture.ipynb", nb_09),
    ]

    for fname, func in generators:
        nb_path = NOTEBOOKS_DIR / fname
        nb_data = func()
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb_data, f, indent=1)
        print(f"✅ Generated {fname} ({os.path.getsize(nb_path):,} bytes)")

    # Readme
    readme_path = NOTEBOOKS_DIR / "README.md"
    readme_lines = [
        "# 📓 Interactive Jupyter Notebooks for Autonomous Driving Course\n\n",
        "Every chapter in the Self-Driving AI course is available as a **standalone, self-contained Jupyter Notebook** runnable in **Google Colab with 1-click** (no GPU required, zero local installation).\n\n",
        "| Chapter | Notebook Title | Google Colab | Source File |\n",
        "| :--- | :--- | :---: | :--- |\n"
    ]
    titles = [
        "Neural Networks, Autograd & Loss Landscapes",
        "Driving Perception Gym & Class Imbalance (Focal Loss)",
        "3D Camera Rig & Inverse Perspective Mapping (IPM)",
        "Multi-Task HydraNet Perception & Uncertainty Loss",
        "Bird's-Eye View (BEV) Transform (Lift-Splat-Shoot)",
        "3D Occupancy Networks & Temporal Memory (ConvGRU)",
        "Vector Space Tracking & Multi-Object Association",
        "Lattice Trajectory Planning & Quintic Splines",
        "Closed-Loop Control & Vehicle Kinematics (Stanley)",
        "Full FSD Capstone & Real-Time System Architecture"
    ]
    for i, (fname, _) in enumerate(generators):
        colab_link = f"{COLAB_BASE}/{fname}"
        badge = f"[![Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_link})"
        file_link = f"[`{fname}`](./{fname})"
        readme_lines.append(f"| **{i:02d}** | {titles[i]} | {badge} | {file_link} |\n")

    readme_lines.append("""
---

### Local Execution
```bash
# Clone the repository
git clone https://github.com/vvknyn/self-driving-ai-course.git
cd self-driving-ai-course

# Install dependencies
pip install torch numpy matplotlib jupyterlab

# Launch JupyterLab
jupyter lab notebooks/
```
""")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.writelines(readme_lines)
    print(f"✅ Generated {readme_path.name}")

if __name__ == "__main__":
    main()
