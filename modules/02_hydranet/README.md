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

## 📐 How the Formulas Are Obtained (MITx Probability Proof)

### 1. Step-by-Step Derivation of Homoscedastic Multi-Task Loss
In MITx Probability, when an estimator makes a prediction $f(x; W)$ corrupted by additive Gaussian observation noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$, the observed target $y$ follows:
$$p(y \mid f(x; W), \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{\|y - f(x; W)\|^2}{2\sigma^2}\right)$$

Taking the natural logarithm ($\ln$):
$$\ln p(y \mid f(x; W), \sigma) = -\frac{\|y - f(x; W)\|^2}{2\sigma^2} - \ln(\sqrt{2\pi\sigma^2}) = -\frac{1}{2\sigma^2} \|y - f(x; W)\|^2 - \ln \sigma - \frac{1}{2}\ln(2\pi)$$

To maximize likelihood, we minimize the Negative Log-Likelihood (NLL). Let $\mathcal{L}_i(W) = \|y_i - f_i(x; W)\|^2$ be the task loss for task $i$.
For $M$ independent tasks, the total negative log-likelihood is:
$$\mathcal{L}_{\text{total}}(W, \sigma_1, \dots, \sigma_M) = \sum_{i=1}^M \left( \frac{1}{2\sigma_i^2} \mathcal{L}_i(W) + \ln \sigma_i \right)$$

### 2. Numerical Reparametrization for SGD
If we optimize $\sigma_i$ directly, standard gradient descent could push $\sigma_i \le 0$ or cause numerical instability near $\sigma_i = 0$.
**The trick**: Define $s_i = \ln(\sigma_i^2)$.
Then:
$$\sigma_i^2 = \exp(s_i) \implies \frac{1}{\sigma_i^2} = \exp(-s_i)$$
$$\ln \sigma_i = \ln\left((\sigma_i^2)^{1/2}\right) = \frac{1}{2}\ln(\sigma_i^2) = \frac{1}{2} s_i$$

Substituting these into the loss yields the exact equation implemented in our code:
$$\mathcal{L}_{\text{total}}(W, s_1, \dots, s_M) = \sum_{i=1}^M \left( \frac{1}{2}\exp(-s_i) \mathcal{L}_i(W) + \frac{1}{2} s_i \right)$$
- If task $i$ has high loss, the optimizer increases $s_i$, which scales down that task's gradient via $\exp(-s_i)$.
- However, the regularizer $+ \frac{1}{2} s_i$ penalizes excessive variance, preventing the network from trivializing all losses!

---

## 📺 Recommended Free Video Tutorials to Learn the Math
- **Maximum Likelihood Estimation**: [StatQuest: Maximum Likelihood Clearly Explained](https://www.youtube.com/watch?v=XepXtl9YKwc)
- **Gaussian Probability Density**: [MIT 6.041x: Normal Random Variables](https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/)
- **Multi-Task Gradient Dynamics**: [Stanford CS231n: Multi-Task Learning](https://cs231n.github.io/)

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
