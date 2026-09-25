# Module 02: HydraNet Architecture & Uncertainty Loss Balancing

> "Multi-task learning allows representations to reinforce each other—lane detection informs drivable freespace, which informs vehicle trajectory bounds."

---

## 🟢 Tier 1: Intuition & Diagnostics (Andrew Ng Style)

### The Multi-Task Computational Bottleneck
Imagine an autonomous car equipped with 8 surround cameras. For each camera, we want the AI to perform 4 crucial perception tasks:
1. Detect 2D bounding boxes of dynamic vehicles.
2. Segment drivable road freespace.
3. Trace lane line markings.
4. Estimate monocular depth.

If you train a separate neural network (e.g. ResNet-50) for each task:
$$8 \text{ cameras} \times 4 \text{ tasks} = 32 \text{ separate neural network forward passes per frame!}$$
At 36 frames per second, the car must compute over **1,150 heavy deep learning passes every second**. Even a massive liquid-cooled GPU cluster will melt.

### The HydraNet Solution: One Body, Many Heads
In biology, the mythical Hydra has one shared body and multiple heads. In deep learning:
- **Shared Trunk (Backbone)**: A deep convolutional or transformer backbone extracts high-level spatial visual representations (edges, textures, 3D context) **once per image**. The trunk accounts for $\sim 85\%$ of all computational FLOPs.
- **Task Heads**: Lightweight convolutional or MLP modules branch off the shared trunk. Each head accounts for only $\sim 3\text{--}5\%$ of FLOPs.
- Total latency drops by over **$75\%$**!

```
                       ┌─────────────────────────┐
                       │  Camera Image (B,3,H,W) │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │   Shared Trunk (85%)    │
                       │   ConvNeXt / ResNet-50  │
                       └────────────┬────────────┘
                                    │ Feature Map (B, C, H/16, W/16)
         ┌──────────────────┬───────┴──────────┬──────────────────┐
         ▼                  ▼                  ▼                  ▼
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ Drivable Head│   │  Lane Head   │   │ BBox 2D Head │   │  Depth Head  │
  │  (Freespace) │   │ (Boundary)   │   │  (Vehicles)  │   │  (Logits)    │
  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

### Andrew Ng Diagnostic Table: Multi-Task Training Failures

| Symptom | Root Cause | Diagnostic Test | Solution |
|---|---|---|---|
| Task A converges to near-zero loss, but Task B fails to learn | Gradient Starvation (Loss Magnitude Disparity) | Log loss magnitudes $\mathcal{L}_A$ vs $\mathcal{L}_B$ during Epoch 1 | Replace static loss weights with Homoscedastic Uncertainty Balancing |
| Overall performance on all tasks is worse than single-task baselines | Negative Transfer / Gradient Conflict | Calculate cosine similarity $\langle g_A, g_B \rangle$ between task gradients | Apply PCGrad (Projected Conflicting Gradients) or separate early trunk layers |
| Loss explodes to `NaN` during multi-task training | Division by zero in uncertainty variance $\sigma_i^2 \to 0$ | Print raw $\sigma_i$ values per iteration | Parametrize using log-variance $s_i = \ln(\sigma_i^2)$ |
| Depth head oscillates wildly while lane head is stable | Task learning rate mismatch | Inspect gradient norms $\|g_{\text{depth}}\|$ vs $\|g_{\text{lane}}\|$ | Add task-specific learning rate multipliers or LayerNorm |

---

## 🟡 Tier 2: Code From Scratch (Andrej Karpathy Style)

Let's inspect the exact PyTorch implementation of Kendall & Gal's Homoscedastic Uncertainty Loss balancing. Notice how the network learns its own loss weights without human hyperparameter tuning!

```python
import torch
import torch.nn as nn

class HomoscedasticUncertaintyLoss(nn.Module):
    """
    Multi-Task Loss Balancing via Automatic Homoscedastic Task Uncertainty.
    Reference: Kendall, Gal, & Cipolla (CVPR 2018).
    
    Instead of manually guessing weights w_1, w_2, w_3, w_4:
    L_total = sum( 0.5 * exp(-s_i) * L_i + 0.5 * s_i )
    where s_i = ln(sigma_i^2) is a learnable parameter.
    """
    def __init__(self, num_tasks: int = 4):
        super().__init__()
        # Andrej Karpathy note: Initialize s_i to 0.0, so initial sigma_i^2 = exp(0) = 1.0.
        # This means all initial loss multipliers 0.5 * exp(-s_i) start at exactly 0.5!
        self.log_vars = nn.Parameter(torch.zeros(num_tasks, dtype=torch.float32))

    def forward(self, losses: list[torch.Tensor]) -> tuple[torch.Tensor, dict]:
        total_loss = torch.tensor(0.0, device=losses[0].device)
        diagnostic_weights = {}

        for i, loss in enumerate(losses):
            s_i = self.log_vars[i]
            # Precision factor: exp(-s_i) = 1 / sigma_i^2
            precision = torch.exp(-s_i)
            
            # Weighted loss term + logarithmic regularizer penalty
            weighted_task_loss = 0.5 * precision * loss + 0.5 * s_i
            total_loss = total_loss + weighted_task_loss
            
            # Diagnostic: effective weight multiplied onto raw task loss
            diagnostic_weights[f"task_{i}_weight"] = float(0.5 * precision.detach())
            diagnostic_weights[f"task_{i}_sigma"] = float(torch.exp(0.5 * s_i).detach())

        return total_loss, diagnostic_weights
```

---

## 🔴 Tier 3: Mathematical Derivations & Proofs (MIT 6.041x / Kendall & Gal)

### Maximum Likelihood Derivation with Heteroscedastic Observation Noise
- **Starting Point**: Consider a multi-output model $f(x; W)$ with parameters $W$. For each task $i \in \{1, \dots, M\}$, the ground truth observation $y_i$ is corrupted by zero-mean Gaussian observation noise with variance $\sigma_i^2$:
  $$y_i = f_i(x; W) + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma_i^2)$$
- **Likelihood Formulation**:
  The conditional probability density of observing $y_i$ given model prediction $f_i(x; W)$ is:
  $$p(y_i \mid f_i(x; W), \sigma_i) = \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\left( -\frac{\|y_i - f_i(x; W)\|^2}{2\sigma_i^2} \right)$$
- **Joint Negative Log-Likelihood (NLL)**:
  Assuming conditionally independent observation noise across tasks:
  $$p(y_1, \dots, y_M \mid W, \sigma_1, \dots, \sigma_M) = \prod_{i=1}^M p(y_i \mid f_i(x; W), \sigma_i)$$
  Taking the negative natural logarithm ($-\ln$):
  $$\mathcal{L}(W, \sigma_1, \dots, \sigma_M) = -\sum_{i=1}^M \ln p(y_i \mid \dots) = \sum_{i=1}^M \left[ \frac{\|y_i - f_i(x; W)\|^2}{2\sigma_i^2} + \ln \sigma_i + \frac{1}{2}\ln(2\pi) \right]$$
  Let $\mathcal{L}_i(W) = \|y_i - f_i(x; W)\|^2$ (the raw task loss). Dropping constants:
  $$\mathcal{L}(W, \sigma_1, \dots, \sigma_M) = \sum_{i=1}^M \left( \frac{1}{2\sigma_i^2} \mathcal{L}_i(W) + \ln \sigma_i \right)$$

### Reparametrization for Numerical Stability
During gradient descent, directly optimizing $\sigma_i$ can cause two fatal numerical bugs:
1. $\sigma_i \to 0$, leading to division by zero (`NaN` error).
2. $\sigma_i < 0$, which is mathematically undefined for standard deviations.

We reparametrize using log-variance $s_i = \ln(\sigma_i^2)$:
$$\sigma_i^2 = \exp(s_i) \implies \frac{1}{\sigma_i^2} = \exp(-s_i)$$
$$\ln \sigma_i = \ln\left((\sigma_i^2)^{1/2}\right) = \frac{1}{2} \ln(\sigma_i^2) = \frac{1}{2} s_i$$
Substituting back into the objective yields the smooth, unbounded loss function:
$$\mathcal{L}_{\text{total}}(W, s_1, \dots, s_M) = \sum_{i=1}^M \left( \frac{1}{2}\exp(-s_i) \mathcal{L}_i(W) + \frac{1}{2} s_i \right)$$
- When a task is noisy or difficult ($\mathcal{L}_i$ large), the gradient $\frac{\partial \mathcal{L}}{\partial s_i} = -\frac{1}{2}\exp(-s_i)\mathcal{L}_i + \frac{1}{2}$ pushes $s_i$ higher, which automatically diminishes the effective weight $\frac{1}{2}\exp(-s_i)$.
- However, the regularizer $+\frac{1}{2} s_i$ penalizes arbitrarily large uncertainty, preventing the network from ignoring all tasks!

---

## 🎓 Tier 4: Cutting-Edge Research & PhD Track

### 1. Gradient Surgery: PCGrad (NeurIPS 2020)
Even with uncertainty loss balancing, shared trunk parameters can experience **gradient conflict**.
If task 1's gradient $g_1 = \nabla_W \mathcal{L}_1$ and task 2's gradient $g_2 = \nabla_W \mathcal{L}_2$ have negative cosine similarity ($\langle g_1, g_2 \rangle < 0$), an optimizer step in the direction of $g_1$ actively degrades performance on task 2!

**PCGrad Algorithm** (Yu et al., NeurIPS 2020):
If $\langle g_i, g_j \rangle < 0$, project $g_i$ onto the normal plane of $g_j$:
$$g_i^{\text{proj}} = g_i - \frac{\langle g_i, g_j \rangle}{\|g_j\|^2} g_j$$
This removes the conflicting component, guaranteeing that parameter updates never hurt competing driving tasks.

### 2. Open PhD Research Questions
- *How can foundation driving models determine task affinity dynamically during pre-training to decide whether representation sharing helps or harms downstream planning?*
- *Can we formulate multi-task uncertainty balancing under heavy out-of-distribution epistemic uncertainty, rather than purely homoscedastic aleatoric observation noise?*

---

## 🟣 Tier 5: Real-World Hardware & Practical Robotics

### Edge Compute Benchmarks (Jetson Orin Nano vs Apple Silicon MPS)
On embedded automotive silicon, memory bandwidth is the primary bottleneck.

| Backbone Model | FP32 Latency (M3 Max MPS) | FP32 Latency (Jetson Orin 8GB) | VRAM Footprint | Recommendation |
|---|---|---|---|---|
| ResNet-50 | 4.8 ms | 14.2 ms | 380 MB | Balanced baseline |
| ConvNeXt-Tiny | 5.2 ms | 16.1 ms | 410 MB | Superior semantic features |
| MobileNetV4-Conv | **1.8 ms** | **5.4 ms** | **120 MB** | Best for low-power edge robots |

**Pro Tip for Robotics**: In `modules/02_hydranet/train_hydranet.py`, set `pin_memory=True` in your PyTorch DataLoader and utilize PyTorch `torch.compile(model, mode="reduce-overhead")` on Linux systems for an instant $20\text{--}35\%$ FPS speedup!
