# Chapter 00: Neural Networks, Autograd & Gradient Descent from Scratch

> *"What I cannot create, I do not understand."* — Richard Feynman

Welcome to the foundation of autonomous driving intelligence. Before we process camera pixels, build Bird's-Eye View transformers, or plan trajectories, we must understand the fundamental unit of computation: **the artificial neuron**, the **computational graph**, and **reverse-mode automatic differentiation (backpropagation)**.

---

## 🎯 JIT Depth Navigation
- **Tier 1: Intuition** — Biological vs Artificial Neurons, the non-linear braking manifold ($d > 0.75 v^2$).
- **Tier 2: Code From Scratch** — 100-line pure Python `Value` autograd engine and 2-layer MLP without PyTorch.
- **Tier 3: Calculus & Math** — Multivariate chain rule proof, finite-difference numerical verification ($\Delta < 10^{-5}$).
- **Tier 4: SOTA Research** — Loss landscape geometry (Li et al., NeurIPS 2018), vanishing gradients.
- **Tier 5: Hardware & Compute** — FLOPs accounting ($2\times$ backward vs forward), GPU activation caching.

---

## 🚀 How to Run

```bash
# 1. Run the interactive training script (trains MLP in 0.1s on toy driving states)
python train_toy_driving.py

# 2. Run Andrew Ng's diagnostic Break-It-Fix-It drills (zero init, exploding LR, linear collapse)
python break_it_fix_it.py

# 3. Run the unit test suite (including finite-difference numerical gradient checking)
python -m unittest tests/test_nn_scratch.py
```

---

## 🧠 Diagnostic Error Analysis Table (Andrew Ng Standard)

| Symptom | Root Cause | Engineering Fix |
| :--- | :--- | :--- |
| **All neuron weights stay identical** | Zero weight initialization ($W = 0.0$). Symmetry trap prevents differentiation. | Xavier / He Normal random initialization ($w \sim \mathcal{N}(0, \sqrt{2/n_{\text{in}}})$). |
| **Loss explodes to `NaN` or `inf`** | Learning rate $\eta$ too large; gradient descent overshoots convex valleys. | Lower $\eta$ by $10\times$; apply gradient clipping ($\|\mathbf{g}\| \le 1.0$). |
| **Gradients vanish to exactly 0.0** | Saturating Sigmoid/Tanh on extreme inputs ($|z| > 5$), or dead ReLUs ($z < 0$). | Use LeakyReLU ($\alpha = 0.01$), ELU, or Batch Normalization. |
| **Model cannot separate braking curve** | Missing non-linear activation. Deep linear layers collapse into a single matrix. | Interleave affine projections with ReLU / Tanh activations. |
