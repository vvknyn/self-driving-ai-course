# Deep Dive: HydraNet Multi-Task Architecture & Uncertainty Loss Balancing

> "Multi-task learning allows representations to reinforce each other—lane detection informs drivable freespace, which informs vehicle trajectory bounds."

---

## 1. The Multi-Task Computational Bottleneck
Running $M$ independent neural networks for $M$ driving tasks has time complexity $\mathcal{O}(M \cdot C_{\text{backbone}})$.
- For 8 surround cameras running 20 distinct perception tasks, computing separate backbones would require over 160 forward passes per frame!
- At 36 FPS, this would exceed 5,700 passes per second—impossible on embedded automotive silicon.

**HydraNet Solution**:
Compute a single shared convolutional/transformer backbone once per camera image. The backbone accounts for $\sim 85\%$ of FLOPs.
Lightweight task heads account for only $\sim 15\%$ of FLOPs:
$$\text{Latency} = T_{\text{backbone}} + \sum_{i=1}^M T_{\text{head}, i} \ll M \cdot T_{\text{backbone}}$$

---

## 2. Multi-Task Gradient Conflict & Starvation
When training a multi-task network with shared trunk weights $W_{\text{trunk}}$:
$$\mathcal{L}_{\text{total}} = \sum_{i=1}^M w_i \mathcal{L}_i$$
The gradient update to the trunk is:
$$\nabla_{W_{\text{trunk}}} \mathcal{L}_{\text{total}} = \sum_{i=1}^M w_i \nabla_{W_{\text{trunk}}} \mathcal{L}_i$$

Two fatal problems occur if weights $w_i$ are fixed manually:
1. **Magnitude Disparity (Gradient Starvation)**: If task A has loss magnitude $100.0$ and task B has loss magnitude $0.1$, task A's gradient vector completely dominates the optimizer step. Task B never learns.
2. **Gradient Conflict**: If $\langle \nabla \mathcal{L}_i, \nabla \mathcal{L}_j \rangle < 0$, optimizing task $i$ actively degrades performance on task $j$.

---

## 3. Homoscedastic Uncertainty Balancing (MITx Probability Proof)
Kendall, Gal, & Cipolla (CVPR 2018) showed that multi-task loss weighting can be derived purely from Maximum Likelihood Estimation with observation noise.

Let the model output $f_i(x; W)$ for task $i$. We assume the ground-truth observation $y_i$ is corrupted by homoscedastic Gaussian task noise with variance $\sigma_i^2$:
$$p(y_i \mid f_i(x; W), \sigma_i) = \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\left( -\frac{\|y_i - f_i(x; W)\|^2}{2\sigma_i^2} \right)$$

The joint log-likelihood across $M$ mutually independent tasks is:
$$\log p(y_1, \dots, y_M \mid W, \sigma_1, \dots, \sigma_M) = \sum_{i=1}^M \left[ -\frac{1}{2\sigma_i^2} \|y_i - f_i(x; W)\|^2 - \log \sigma_i - \frac{1}{2}\log(2\pi) \right]$$

To maximize likelihood, we minimize the Negative Log-Likelihood (NLL):
$$\mathcal{L}(W, \sigma_1, \dots, \sigma_M) = \sum_{i=1}^M \left( \frac{1}{2\sigma_i^2} \mathcal{L}_i(W) + \log \sigma_i \right)$$

### Numerical Parametrization
To ensure $\sigma_i > 0$ and avoid dividing by zero during gradient descent, we parametrize:
$$s_i = \log(\sigma_i^2) \implies \sigma_i^2 = \exp(s_i), \quad \log \sigma_i = \frac{1}{2} s_i$$
The objective becomes smooth and convex with respect to $s_i$:
$$\mathcal{L}(W, s_1, \dots, s_M) = \sum_{i=1}^M \left( \frac{1}{2} \exp(-s_i) \mathcal{L}_i(W) + \frac{1}{2} s_i \right)$$
- If $\mathcal{L}_i$ is large, the network increases $s_i$, which scales down the gradient by $\exp(-s_i)$.
- However, the regularizer $+ \frac{1}{2} s_i$ penalizes excessive uncertainty, preventing the network from trivializing all losses to zero!
