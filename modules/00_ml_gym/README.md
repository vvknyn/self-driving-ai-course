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

## 📐 How the Formulas Are Obtained

### 1. Cross-Entropy Loss from KL Divergence
Let $p(y)$ be the true ground-truth distribution (one-hot vector where $p(y_{\text{true}}) = 1$ and $p(y \neq y_{\text{true}}) = 0$), and $q(y)$ be the model's predicted probability distribution from $\text{Softmax}(z)$:
$$q(y_k) = \frac{\exp(z_k)}{\sum_j \exp(z_j)}$$

In information theory, the **Kullback-Leibler (KL) Divergence** measures the relative entropy or information lost when approximating $p$ with $q$:
$$D_{\text{KL}}(p \parallel q) = \sum_{k} p(y_k) \log\left(\frac{p(y_k)}{q(y_k)}\right) = \underbrace{\sum_k p(y_k) \log p(y_k)}_{-H(p) \text{ (Entropy of Ground Truth)}} - \underbrace{\sum_k p(y_k) \log q(y_k)}_{H(p, q) \text{ (Cross Entropy)}}$$

Because the ground-truth distribution $p$ is fixed, minimizing $D_{\text{KL}}$ is mathematically identical to minimizing the Cross-Entropy:
$$\mathcal{L}_{\text{CE}} = -\sum_k p(y_k) \log q(y_k)$$
Since $p(y_k) = 1$ only for the true class $t$, this simplifies directly to:
$$\mathcal{L}_{\text{CE}}(p_t) = -\log(p_t)$$

### 2. Derivation of Focal Loss (Lin et al., ICCV 2017)
In driving datasets, 90%+ of crops are clear road or sky. For these easy examples, the model outputs high confidence $p_t \approx 0.99$.
Under standard cross-entropy:
$$\mathcal{L}_{\text{CE}}(0.99) = -\log(0.99) \approx 0.01$$
While $0.01$ is small, summing across millions of easy road pixels yields a massive gradient that overwhelms rare pedestrians!

**How the formula is obtained**:
We introduce a modulating factor $(1 - p_t)^\gamma$ with focusing parameter $\gamma \ge 0$:
$$\mathcal{L}_{\text{Focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
- When an example is misclassified and hard ($p_t = 0.1$), $(1 - 0.1)^2 = 0.81$ (loss is barely affected).
- When an example is well-classified ($p_t = 0.99$), $(1 - 0.99)^2 = 0.0001$ (**loss is scaled down by 10,000x**!).

### 3. Backpropagation via the Multivariable Chain Rule
For a weight matrix $W$ in layer $l$, the gradient of scalar loss $\mathcal{L}$ with respect to $W$ is:
$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}} = \delta^{(l)} \cdot (a^{(l-1)})^T$$
Where error $\delta^{(l)}$ propagates backward recursively:
$$\delta^{(l)} = \left( (W^{(l+1)})^T \delta^{(l+1)} \right) \odot \sigma'(z^{(l)})$$

---

## 📺 Recommended Free Video Tutorials to Learn the Math
- **Neural Networks & Backprop**: [3Blue1Brown: What is Backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
- **Cross Entropy & Loss Intuition**: [StatQuest: Cross Entropy Clearly Explained](https://www.youtube.com/watch?v=6ArSys5qHAU)
- **Calculus of Gradients**: [3Blue1Brown: Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)

---

## 🏎️ Why Tesla Does It This Way
Tesla's Autopilot neural networks ingest millions of video clips daily.
- Andrei Karpathy described the **"Data Engine"**: models fail not because of model depth, but because of data distribution skew and gradient starvation.
- Focal loss and active sample weighting allow FSD networks to train on billions of common highway miles without forgetting rare edge cases like overturned trucks or pedestrians wearing dark clothing at night.

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
