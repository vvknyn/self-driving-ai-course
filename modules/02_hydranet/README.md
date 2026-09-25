# Module 02: Semantic Perception with HydraNet

> "HydraNet is how Tesla runs 50+ perception tasks in real-time on one chip: one shared visual trunk, branching into dozens of specialized heads." — Andrei Karpathy

---

## 🎯 Purpose & Learning Goals
Running separate neural networks for every driving task (detecting lanes, detecting vehicles, segmenting drivable space, recognizing stop signs) is impossible inside a moving car—it would require 10 GPUs in the trunk.

In this module, you will:
1. Build a **HydraNet multi-task architecture**: a shared convolutional trunk that extracts universal visual features, branching into four specialized perception heads:
   - **Lane Segmentation Head**: Identifies lane boundaries.
   - **Drivable Freespace Head**: Binary mask of drivable road surface.
   - **Vehicle Detection Head**: Predicts bounding box centroids & confidence heatmaps.
   - **Traffic Light Head**: Classifies traffic light states (Green / Yellow / Red / None).
2. Connect to **MITx Probability Foundations**:
   - Solve multi-task loss balancing using **Homoscedastic Task Uncertainty** (Kendall, Gal & Cipolla). Instead of guessing arbitrary weights by trial and error, the network learns the measurement noise variance $\sigma_i^2$ for each task using Maximum Likelihood Estimation!
3. FastAI "Break It & Fix It": Witness **gradient starvation** where one task drowns out another, and cure it using uncertainty-weighted loss.

---

## 📐 Mathematical Formulation (MITx Probability Connection)

### Why naive loss summation fails:
If we simply add task losses:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{lane}} + \mathcal{L}_{\text{vehicle}} + \mathcal{L}_{\text{freespace}} + \mathcal{L}_{\text{traffic\_light}}$$
If $\mathcal{L}_{\text{vehicle}} = 50.0$ and $\mathcal{L}_{\text{lane}} = 0.05$, the gradients of the vehicle task are $1000\times$ larger! The shared trunk updates only to please the vehicle task, starving the lane task of any learning.

### The Probabilistic Fix: Gaussian Observation Likelihood
In MITx Probability, when a model makes a prediction $y$ with Gaussian observation noise $\sigma^2$:
$$P(y \mid f(x), \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{\|y - f(x)\|^2}{2\sigma^2}\right)$$
Taking the negative log-likelihood ($-\log P$):
$$-\log P(y \mid f(x), \sigma) \propto \frac{1}{2\sigma^2} \mathcal{L}(y, f(x)) + \log \sigma$$

For $M$ simultaneous tasks, our total loss becomes:
$$\mathcal{L}_{\text{total}}(W, \sigma_1, \dots, \sigma_M) = \sum_{i=1}^M \left( \frac{1}{2\sigma_i^2} \mathcal{L}_i(W) + \log \sigma_i \right)$$
- If a task is noisy or difficult, the network increases $\sigma_i$, automatically scaling down that task's gradient so it doesn't destabilize the shared trunk.
- The $\log \sigma_i$ term acts as a regularizer—the network cannot simply set $\sigma_i \to \infty$ to make loss zero!
- In practice, we parametrize $s_i = \log(\sigma_i^2)$ for numerical stability:
$$\mathcal{L}_i = \frac{1}{2} \exp(-s_i) \mathcal{L}_i + \frac{1}{2} s_i$$

---

## 🏎️ Why Tesla Does It This Way
- Tesla's FSD computer (HW3/HW4) has a strict latency budget: **15 milliseconds per frame**.
- HydraNet caches the heavy shared trunk activations once per camera frame.
- Different teams at Tesla can train and fine-tune different heads independently by freezing the shared trunk, preventing regressions across perception capabilities.

---

## 🧪 Quick Run
```bash
# 1. Train HydraNet on all 4 tasks simultaneously
python modules/02_hydranet/train_hydranet.py

# 2. Run the Break-It & Fix-It drill (gradient starvation vs uncertainty weighting)
python modules/02_hydranet/break_it_fix_it.py

# 3. Verify unit tests
pytest modules/02_hydranet/tests/test_hydranet.py
```

🎨 **Interactive Visual Explainer**: Open [`visual_explainers/02_hydranet_architecture.html`](../../visual_explainers/02_hydranet_architecture.html) in your browser to inspect interactive trunk-to-head feature routing!
