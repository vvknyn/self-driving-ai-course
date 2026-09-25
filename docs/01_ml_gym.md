# Chapter 01: Driving Perception Gym & Class Imbalance

> **The Big Question**: *In autonomous driving, 99.1% of camera pixels belong to empty asphalt, crash barriers, and sky, while less than 0.1% belong to pedestrians or bicycles. If a neural network simply guesses "empty road" 100% of the time, it achieves 99.1% accuracy—yet it will kill every pedestrian on the road. How do we mathematically prevent the neural network from taking this lazy shortcut?*

---

## 1. 🚨 The Real-World Dilemma: The Asphalt Dominance Trap

Imagine training a neural network on 100,000 highway video clips recorded by Tesla or Waymo. 
- Over an average highway drive, the front camera views smooth gray pavement for miles on end.
- Suddenly, an overturned semi-truck or a deer leaps across the lane.

If you train a standard neural network with Cross-Entropy loss on raw driving footage:

$$\mathcal{L}_{\text{CE}} = -\sum_{i=1}^N \log(p_{t, i})$$

- For 1,000,000 road pixels where the network is already 99% confident ($p_t = 0.99$):
  $$\text{Loss per pixel} = -\log(0.99) \approx 0.01$$
  $$\text{Total Road Gradient Contribution} = 1,000,000 \times 0.01 = \mathbf{10,000}$$
- For 1 rare pedestrian pixel where the network is confused ($p_t = 0.10$):
  $$\text{Loss per pixel} = -\log(0.10) \approx 2.30$$
  $$\text{Total Pedestrian Gradient Contribution} = 1 \times 2.30 = \mathbf{2.30}$$

```
Gradient Contribution to Optimizer:
Road Pixels (Easy):       ████████████████████████████████████████ (10,000)
Pedestrian Pixels (Hard): ▌ (2.3)
```

> [!CAUTION]
> **The Gradient Starvation Catastrophe**  
> The optimizer updates its weights in the direction of the largest gradient. Even though each individual road pixel has a tiny loss, the sheer **quantity** of boring asphalt overwhelms the gradient by **4,300 to 1**! The network spends all its learning capacity making the road 99.99% confident while completely ignoring vulnerable road users.

---

## 2. 💡 The Mental Model: The Dynamic Focusing Damper

How does a human driving instructor teach a student?
- If the student drives straight in an empty lane 50 times in a row, the instructor doesn't shout advice on every single straight meter.
- The instructor stays quiet during the easy, repetitive parts and **focuses 100% of attention on the near-miss mistakes**.

### The Mathematical Mechanism: Focal Loss (Lin et al., ICCV 2017)
To mimic this in calculus, we attach a **dynamic modulating factor** $(1 - p_t)^\gamma$ to the loss:

$$\mathcal{L}_{\text{Focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Look at how the term $(1 - p_t)^\gamma$ acts as an automatic volume knob when $\gamma = 2$:
- **Easy Road Pixel ($p_t = 0.99$)**:
  $$(1 - 0.99)^2 = (0.01)^2 = \mathbf{0.0001}$$
  The loss and gradient are suppressed by **$10,000\times$**! The road pixels are effectively muted.
- **Hard Pedestrian Pixel ($p_t = 0.10$)**:
  $$(1 - 0.10)^2 = (0.90)^2 = \mathbf{0.81}$$
  The loss and gradient are virtually uninhibited ($81\%$ of full strength)!

```
Confidence (pt) │ Standard CE Loss │ Focal Loss (γ = 2) │ Suppression Ratio
────────────────┼──────────────────┼────────────────────┼──────────────────
0.10 (Hard)     │ 2.302            │ 1.865              │ 1.2x
0.50 (Medium)   │ 0.693            │ 0.173              │ 4.0x
0.90 (Easy)     │ 0.105            │ 0.00105            │ 100.0x
0.99 (Trivial)  │ 0.010            │ 0.000001           │ 10,000.0x
```

---

## 3. 🧪 Lab Mission: Hands-On Simulator Experiments

Scroll up to the **Interactive Perception Gym** at the top of this chapter:

1. **Experiment 1 (The Naive Cross-Entropy Failure)**:
   - Slide **Class Imbalance** to `98% Road / 2% Pedestrian`.
   - Set **Focal Parameter ($\gamma$)** to `0.0` (standard Cross-Entropy).
   - Click **Train 50 Epochs**.
   - *Observation*: Look at the Confusion Matrix. The model predicts 0 pedestrians! Pedestrian recall is 0%, even though overall validation accuracy is 98.2%.
2. **Experiment 2 (Activating Focal Modulation)**:
   - Keep the same 98:2 imbalance, but move the slider **Focal Parameter ($\gamma$)** to `2.0`.
   - Click **Train 50 Epochs**.
   - *Observation*: The Pedestrian Recall jumps from 0% to >92%! Notice the live gradient bar chart: the blue asphalt gradient shrinks to a hairline, allowing pedestrian gradients to guide weight updates.

---

## 4. 🛠️ The Karpathy Build: Custom Focal Loss from Scratch

Here is the exact PyTorch implementation of multi-class Focal Loss and the perception classifier, tracking tensor shapes at each step:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for severe automotive class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # (C,) tensor of per-class weights or None
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Input:
            logits: (B, C) unnormalized raw network outputs
            targets: (B,) ground-truth class integer indices in [0, C-1]
        Returns:
            loss: scalar tensor
        """
        # 1. Compute cross-entropy without reduction: -log(p_t)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # -> (B,)
        
        # 2. Get p_t = exp(-ce_loss)
        p_t = torch.exp(-ce_loss)                                    # -> (B,)
        
        # 3. Compute modulating factor: (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma                # -> (B,)
        
        # 4. Focal loss per sample
        focal_loss = modulating_factor * ce_loss                     # -> (B,)
        
        # 5. Apply class balancing alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]                            # -> (B,)
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()
```

---

## 5. 📐 Mathematical Rigor: Cross-Entropy & Gradients

### From KL Divergence to Cross-Entropy
Let $p(y)$ be the true distribution ($p(t)=1$ for true class, $0$ otherwise) and $q(y)$ be the predicted softmax probabilities. The Kullback-Leibler (KL) divergence is:

$$D_{\text{KL}}(p \parallel q) = \sum_{k} p(y_k) \log\left(\frac{p(y_k)}{q(y_k)}\right) = \sum_k p(y_k) \log p(y_k) - \sum_k p(y_k) \log q(y_k)$$

Since the ground truth $p$ is constant with respect to model parameters $\theta$, minimizing information divergence is identical to minimizing:

$$\mathcal{L}_{\text{CE}} = -\sum_k p(y_k) \log q(y_k) = -\log(p_t)$$

### Derivative with Respect to Logits $z_k$
For standard Cross-Entropy with Softmax $p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$:

$$\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z_k} = p_k - y_k$$

For Focal Loss with $\gamma > 0$, the gradient incorporates the derivative of the modulating factor:

$$\frac{\partial \mathcal{L}_{\text{FL}}}{\partial z_k} = (1 - p_t)^\gamma (p_k - y_k) + \gamma (1 - p_t)^{\gamma - 1} p_t \log(p_t) (p_k - y_k)$$

Notice that as $p_t \to 1$ (easy sample), the first term decays as $(1 - p_t)^\gamma$ and the second term decays as $p_t \log(p_t) \to 0$. **Both gradient terms vanish to 0!**

---

## 6. 🩺 Andrew Ng's Diagnostic Field Guide

| Observed Symptom | Underlying Mathematical Mechanism | Verification Test | Production Fix |
| :--- | :--- | :--- | :--- |
| **High overall accuracy (>95%), but 0% recall on rare objects** | **Majority Class Dominance**: The model minimizes total dataset loss by predicting the prior probability of empty road. | Print class-stratified confusion matrix; check recall per class. | Switch from Cross-Entropy to Focal Loss with $\gamma \in [1.5, 2.5]$ and $\alpha$-weighting. |
| **Training loss diverges or oscillates violently** | **Extreme Class Weighting ($\alpha \gg 100$)**: Multiplying small classes by excessive $\alpha$ causes massive gradients when an outlier occurs. | Monitor $\max \|\mathbf{g}\|$ per mini-batch. | Clamp $\alpha$ weights ($\alpha_{\max} \le 10.0$) and use gradient clipping. |
| **Model outputs overconfident wrong probabilities (Calibration Drift)** | **Cross-Entropy Overconfidence**: Standard CE pushes logits $z \to \pm \infty$ to minimize $-\log(p_t) \to 0$. | Compute Expected Calibration Error (ECE) and reliability diagrams. | Apply temperature scaling $T$ at inference ($p = \text{Softmax}(z / T)$) or use label smoothing ($\epsilon = 0.05$). |

---

## 7. 🎯 Self-Check: Test Your Mental Model

<details>
<summary><b>Q1: If you set $\gamma = 0$ in the Focal Loss formula, what mathematical function do you recover?</b></summary>

<br>

**Answer**: You recover standard **Cross-Entropy Loss**. When $\gamma = 0$, the modulating factor $(1 - p_t)^0 = 1$ for all values of $p_t \in (0, 1]$, leaving $\mathcal{L} = -\alpha_t \log(p_t)$.
</details>

<details>
<summary><b>Q2: Why not just discard 99% of empty highway frames from your dataset instead of using Focal Loss?</b></summary>

<br>

**Answer**: Naive undersampling discards crucial negative examples. If you delete 99% of empty roads, the network never learns the vast diversity of visual pavement textures (wet asphalt, oil slicks, shadows under overpasses, tar repair lines). When deployed in the real world, the network will hallucinate obstacles on road textures it was never trained to reject. Focal loss retains all the rich negative visual data, but dynamically turns down its gradient volume.
</details>
