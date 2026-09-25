# Chapter 03: Multi-Task HydraNet Perception

> **The Big Question**: *An autonomous vehicle must simultaneously identify pedestrians, track lane lines, read speed limit signs, classify traffic light colors, and predict drivable asphalt. If you train a separate neural network for every task, your car's computer draws 5,000 Watts and runs at 3 FPS. But when you combine them into a single shared network, training the car to see pedestrians causes its traffic light accuracy to plunge by 14%! Why do neural network tasks fight each other, and how do we mathematically force them to cooperate?*

---

## 1. 🚨 The Real-World Dilemma: The Compute & Gradient Tug-of-War

Consider the sensory workload of a modern autonomous vehicle:
- 8 high-resolution camera streams at 36 Hz.
- At least 15 distinct perception outputs:
  - 3D Obstacle Bounding Boxes (cars, trucks, bikes, pedestrians).
  - Drivable Free Space Segmentation.
  - Lane Boundary Polylines (solid, dashed, double-yellow, curbs).
  - Traffic Light Status (red, yellow, green, protected left arrow).
  - Dynamic Road Signs & Variable Speed Limits.

### The Naive Architecture: 15 Separate Deep Networks
If each task uses an independent EfficientNet-B4 or ResNet-50 backbone:
- Total Compute: $15 \times 25\text{ GFLOPs} = \mathbf{375\text{ GFLOPs}}$ per frame!
- Across 8 cameras at 36 Hz: **$108\text{ TeraFLOPs/sec}$**.
- The car's trunk fills with liquid-cooled server racks consuming several kilowatts, draining the vehicle's battery and causing latency to exceed 200 ms.

### The Multi-Task Trap: Negative Transfer
To fix this, autonomous engineers build a **HydraNet**: a single massive shared "trunk" (backbone) that computes general visual features, followed by lightweight specialized "heads" for each task.

```
                           ┌──► Head 1: 3D Object Detection (Cars, Trucks)
                           ├──► Head 2: Drivable Free Space Segmentation
Shared Trunk (Backbone) ───┼──► Head 3: Lane Line Polyline Extrusion
                           └──► Head 4: Traffic Light State (Red/Green)
```

> [!CAUTION]
> **The Tug-of-War: Destructive Gradient Interference**  
> When you train all heads together with a naive sum of losses $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}} + \mathcal{L}_{\text{lanes}} + \mathcal{L}_{\text{lights}}$, disaster strikes:
> - Head 1 computes gradient vector $\mathbf{g}_{\text{det}}$ pushing trunk weights in direction $\vec{u}$.
> - Head 4 computes gradient vector $\mathbf{g}_{\text{lights}}$ pushing trunk weights in the opposite direction $-\vec{u}$.
> If $\mathbf{g}_1 \cdot \mathbf{g}_2 < 0$, the tasks actively cancel and sabotage each other. This is known as **Negative Transfer**.

---

## 2. 💡 The Mental Model: The Master Architect & Specialist Apprentices

Think of a construction company:
- You don't train 15 separate master architects to understand structural physics, geometry, and material strength.
- You train **one world-class master architect** (the shared backbone) who understands 3D spatial reality.
- You then hire **specialized apprentices** (the task heads): one focuses on electrical wiring, one on plumbing, one on drywall.

### The Loss Balancing Conundrum
Different tasks operate on completely different numerical scales:
- Bounding box regression loss is continuous Smooth-L1: $\mathcal{L}_{\text{box}} \approx 0.12$.
- Drivable segmentation loss is pixel binary cross-entropy: $\mathcal{L}_{\text{seg}} \approx 4.85$.
- Traffic light classification loss: $\mathcal{L}_{\text{lights}} \approx 0.03$.

If you sum them naively, the segmentation loss ($4.85$) is **$160\times$ larger** than the traffic light loss ($0.03$). The shared backbone will devote all its learning capacity to asphalt textures and completely ignore whether the light ahead is red or green!

### The Solution: Kendall Homoscedastic Uncertainty Weighting
Instead of hand-tuning arbitrary loss multipliers ($w_1, w_2, w_3$) through endless trial-and-error, Kendall, Gal & Cipolla (CVPR 2018) derived a principled Bayesian loss. We let the neural network **learn its own uncertainty $\sigma_i$ for each task**:

$$\mathcal{L}(\mathbf{W}, \sigma_1, \sigma_2) = \frac{1}{2 \sigma_1^2} \mathcal{L}_1(\mathbf{W}) + \frac{1}{2 \sigma_2^2} \mathcal{L}_2(\mathbf{W}) + \log \sigma_1 + \log \sigma_2$$

Notice the brilliant mathematical tension:
- If a task is noisy or difficult, the network can increase $\sigma_i$, which shrinks $\frac{1}{2 \sigma_i^2}$ and discounts that task's gradient.
- But the network cannot cheat by setting $\sigma_i \to \infty$, because the penalty term **$\log \sigma_i$** will explode!

---

## 3. 🧪 Lab Mission: Hands-On Simulator Experiments

Scroll to the **Interactive HydraNet Architecture Studio** at the top of this chapter:

1. **Experiment 1 (The Hand-Tuned Disaster)**:
   - Toggle loss mode to `Manual Weights`.
   - Set **Weight 1 (Detection)** to `1.0` and **Weight 2 (Lanes)** to `0.005`.
   - Click **Train 100 Steps**.
   - *Observation*: The lane detection head performance collapses. The shared backbone strips away edge features to satisfy the bounding box head.
2. **Experiment 2 (Kendall Bayesian Auto-Tuning)**:
   - Switch loss mode to `Kendall Uncertainty Weighting`.
   - Click **Train 100 Steps**.
   - *Observation*: Watch the live log-variance bars $\log(\sigma^2)$ dynamically calibrate. As each head learns, the gradients automatically normalize, allowing all 4 heads to converge simultaneously!

---

## 4. 🛠️ The Karpathy Build: HydraNet & Uncertainty Loss from Scratch

Here is the complete multi-task architecture and Bayesian loss module in pure PyTorch:

```python
import torch
import torch.nn as nn

class KendallUncertaintyLoss(nn.Module):
    """
    Multi-Task Loss with Homoscedastic Task Uncertainty.
    Ref: Kendall, Gal & Cipolla (CVPR 2018)
    """
    def __init__(self, num_tasks=3):
        super().__init__()
        # Learnable log-variance parameter for each task: s = log(sigma^2)
        # Using s instead of sigma ensures numerical stability and sigma^2 > 0.
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, task_losses: list[torch.Tensor]) -> torch.Tensor:
        """
        Input:
            task_losses: List of scalar tensors [L_det, L_seg, L_lights]
        Returns:
            total_loss: scalar tensor
        """
        total_loss = 0.0
        for i, loss in enumerate(task_losses):
            s = self.log_vars[i]
            # L_weighted = exp(-s) * loss + 0.5 * s
            precision = torch.exp(-s)
            total_loss += 0.5 * precision * loss + 0.5 * s
        return total_loss

class HydraNet(nn.Module):
    """
    Unified multi-task perception network for autonomous vehicles.
    """
    def __init__(self, shared_dim=256):
        super().__init__()
        # 1. Shared Trunk (Feature Extractor)
        self.trunk = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, shared_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU()
        )
        
        # 2. Specialized Task Heads
        # Head A: 3D Object Detection Bounding Boxes (x, y, z, w, l, h, yaw)
        self.head_bbox = nn.Conv2d(shared_dim, 7, kernel_size=1)
        
        # Head B: Drivable Free Space Segmentation (binary road mask)
        self.head_seg = nn.Conv2d(shared_dim, 1, kernel_size=1)
        
        # Head C: Traffic Light Classifier (4 classes: Red, Yellow, Green, Off)
        self.head_lights = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(shared_dim, 4)
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        features = self.trunk(x)                     # -> (B, shared_dim, H/4, W/4)
        out_bbox = self.head_bbox(features)          # -> (B, 7, H/4, W/4)
        out_seg = self.head_seg(features)            # -> (B, 1, H/4, W/4)
        out_lights = self.head_lights(features)      # -> (B, 4)
        return out_bbox, out_seg, out_lights
```

---

## 5. 📐 Mathematical Rigor: Derivation from Likelihood

Consider a multi-output model with shared parameters $\mathbf{W}$. For task 1 (regression with Gaussian noise $\sigma_1$) and task 2 (classification with softmax):

The likelihood for regression with observation variance $\sigma_1^2$ is:

$$p(\mathbf{y}_1 \mid \mathbf{f}^{\mathbf{W}}(\mathbf{x}), \sigma_1) = \frac{1}{\sqrt{2\pi}\sigma_1} \exp\left( -\frac{\|\mathbf{y}_1 - \mathbf{f}^{\mathbf{W}}(\mathbf{x})\|^2}{2\sigma_1^2} \right)$$

Taking the negative log-likelihood:

$$-\log p(\mathbf{y}_1 \mid \mathbf{f}^{\mathbf{W}}(\mathbf{x}), \sigma_1) = \frac{1}{2\sigma_1^2} \|\mathbf{y}_1 - \mathbf{f}^{\mathbf{W}}(\mathbf{x})\|^2 + \log \sigma_1 + \text{const}$$

Setting parameter $s_1 = \log \sigma_1^2 \implies \sigma_1 = \exp(s_1 / 2)$:

$$\mathcal{L}(\mathbf{W}, s_1) = \exp(-s_1) \mathcal{L}_1(\mathbf{W}) + \frac{1}{2} s_1$$

### Gradient Surgery: PCGrad (Yu et al., NeurIPS 2020)
When two head gradients conflict ($\mathbf{g}_i \cdot \mathbf{g}_j < 0$), Projecting Conflicting Gradients (PCGrad) projects $\mathbf{g}_i$ onto the normal plane of $\mathbf{g}_j$:

$$\mathbf{g}_i \leftarrow \mathbf{g}_i - \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_j\|^2} \mathbf{g}_j$$

This mathematically guarantees that task $i$'s update never harms task $j$'s performance!

---

## 6. 🩺 Andrew Ng's Diagnostic Field Guide

| Observed Symptom | Underlying Mathematical Mechanism | Verification Test | Production Fix |
| :--- | :--- | :--- | :--- |
| **Adding a new task degrades existing tasks by >10%** | **Negative Transfer / Gradient Conflict**: Gradients from the new task head have negative inner product with the shared trunk gradients. | Compute cosine similarity $\cos(\theta) = \frac{\mathbf{g}_a \cdot \mathbf{g}_b}{\|\mathbf{g}_a\|\|\mathbf{g}_b\|}$. | Use Kendall Uncertainty weighting or apply PCGrad gradient projection. |
| **One head's learnable variance explodes ($s_i \to \infty$)** | **Task Cheating**: The model discovers it can minimize overall loss by declaring a difficult task "infinite noise" and ignoring it. | Print `model.log_vars.data` values during training. | Clamp maximum log-variance ($s_i \le 3.0$) or provide higher initial learning rate for that head. |
| **Trunk features overfit to low-resolution textures** | **Capacity Bottleneck**: The shared feature dimension is too narrow to hold representations for all downstream tasks. | Measure feature rank via Singular Value Decomposition (SVD). | Increase trunk channel width or incorporate Feature Pyramid Networks (FPN). |

---

## 7. 🎯 Self-Check: Test Your Mental Model

<details>
<summary><b>Q1: Why do we optimize $s = \log(\sigma^2)$ instead of directly optimizing $\sigma$ in Kendall uncertainty loss?</b></summary>

<br>

**Answer**: For two critical numerical reasons:
1. **Positivity Constraint**: Standard deviation $\sigma$ must strictly be $> 0$. If we optimize $\sigma$ directly with gradient descent, a single large step can push $\sigma \le 0$, causing division by zero or negative variance. Because $\exp(s) > 0$ for all $s \in (-\infty, \infty)$, optimizing $s$ guarantees mathematical validity.
2. **Gradient Stability**: When $\sigma \to 0$, $\frac{1}{\sigma^2}$ produces explosive gradients. The term $\exp(-s)$ provides smoother, more stable gradient curves.
</details>

<details>
<summary><b>Q2: If Task A has 100x lower loss than Task B, does that mean Task A is 100x better learned?</b></summary>

<br>

**Answer**: **No!** Loss magnitudes cannot be compared across different tasks. Smooth-L1 regression loss is denominated in meters (e.g. $0.05\text{ m}$), while Cross-Entropy loss is denominated in nats (information theory). A bounding box regression loss of $0.05$ might be quite poor for parking precision, while a cross-entropy loss of $0.5$ might represent a highly accurate classifier. Kendall uncertainty weighting normalizes away these arbitrary scale differences.
</details>
