# Module 00: The Driving ML Gym (FastAI Primer for Autonomous Driving)

> "To understand Full Self-Driving, you must first master the atomic unit of neural networks: the flow of tensors, loss computation, and backpropagation on driving data."

---

## 🎯 Purpose & Learning Goals
Before diving into multi-camera rigs and 3D Bird's-Eye View (BEV) spaces, you need rock-solid intuition for how vision neural networks actually learn from driving scenes. In this module, you will:
1. Understand image tensors: `(B, C, H, W)` where $B$ is batch size, $C=3$ (RGB), and $H, W$ are spatial resolution.
2. Build a custom multi-class driving scene classifier (detecting **Clear Road**, **Lead Vehicle**, **Pedestrian**, and **Lane Marking**).
3. Implement **Focal Loss** to handle the severe class imbalance inherent to autonomous driving (roads are 90% of pixels; pedestrians are <1%).
4. Practice FastAI-style iterative engineering: run a baseline, inspect predictions, break something on purpose, and fix it.

---

## 📐 Mathematical Formulation

### 1. Cross-Entropy vs. Focal Loss
Standard Cross-Entropy Loss treats all examples equally:
$$\mathcal{L}_{\text{CE}}(p_t) = -\log(p_t)$$

In driving, background road pixels dominate the gradient, drowning out rare critical safety objects (e.g., pedestrians). **Focal Loss** adds a modulating factor $(1 - p_t)^\gamma$:
$$\mathcal{L}_{\text{Focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
Where:
- $p_t$ is the model's estimated probability for the ground-truth class.
- $\gamma \ge 0$ is the focusing parameter (typically $\gamma = 2.0$). When an easy road patch has $p_t = 0.99$, $(1 - 0.99)^2 = 0.0001$, scaling down its loss by $10,000\times$!
- $\alpha_t \in [0, 1]$ addresses class frequency weighting.

### 2. Backpropagation & Gradient Descent
Weights are updated using the chain rule:
$$\theta_{t+1} = \theta_t - \eta \frac{\partial \mathcal{L}}{\partial \theta_t}$$
Where $\eta$ is the learning rate.

---

## 🏎️ Why Tesla Does It This Way
Tesla's Autopilot and FSD neural networks ingest millions of video clips daily.
- Andrei Karpathy described the **"Data Engine"**: models fail not because of model depth, but because of data distribution skew and gradient starvation.
- Focal loss and active sample weighting allow FSD networks to train on billions of common highway miles without forgetting rare edge cases like overturned trucks or pedestrians wearing dark clothing at night.

---

## 🛠️ Hands-On Files in This Module
- [`dataset.py`](dataset.py): Deterministic driving crop generator providing multi-class road patches.
- [`model.py`](model.py): Modern convolutional driving backbone with batch norm and residual connections.
- [`train.py`](train.py): **Run this first!** FastAI-style training script with visual progress and evaluation.
- [`break_it_fix_it.py`](break_it_fix_it.py): **The Drill.** Exploding learning rate and class imbalance failure drill.
- [`tests/test_gym.py`](tests/test_gym.py): Unit tests verifying loss functions, tensor dimensions, and backward gradients.

---

## 🧪 Quick Run
```bash
# 1. Run the training script
python modules/00_ml_gym/train.py

# 2. Run the Break-It & Fix-It drill
python modules/00_ml_gym/break_it_fix_it.py

# 3. Verify unit tests
pytest modules/00_ml_gym/tests/test_gym.py
```
