# Chapter 00: Neural Networks, Autograd & Gradient Descent from Scratch

> *"What I cannot create, I do not understand."* — Richard Feynman
>
> *"Backpropagation is just the multivariable chain rule from calculus dressed up in efficient graph data structures."* — Andrej Karpathy

---

## 🎯 JIT Depth Selector
Choose your depth:
- **[Tier 1: Intuition](#tier-1-core-intuition)** (5 min read) — Biological vs artificial neurons, non-linear boundaries, and error surfaces.
- **[Tier 2: Karpathy Scratch Code](#tier-2-code-from-scratch)** (15 min drill) — Build an autograd DAG engine and 2-layer MLP in 100 lines of pure Python.
- **[Tier 3: Sebastian Thrun & MIT Calculus](#tier-3-deep-math--academic-lineage)** (30 min derivation) — Multivariate chain rule proof and numerical finite-difference gradient checking.
- **[Tier 4: Cutting-Edge SOTA Research](#tier-4-cutting-edge-research--phd-frontiers)** (1 hour deep dive) — Loss landscape geometry (Li et al., NeurIPS 2018), saddle points, and gradient flow.
- **[Tier 5: Edge Hardware & Compute](#tier-5-practical-hardware--compute-budgets)** (Production engineering) — FLOPs accounting, activation caching memory footprint, and FP16 vs INT8 quantization.

---

## Tier 1: Core Intuition

### The Biological Inspiration vs Matrix Reality
A biological neuron receives electrical signals along dendrites, integrates them in the soma, and fires an action potential down the axon if a threshold is reached.

In artificial intelligence, an **artificial neuron** is an affine linear transformation followed by a non-linear squashing function:

$$z = \sum_{i=1}^n w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$

$$a = \sigma(z)$$

Where:
- $\mathbf{x} = [x_1, x_2, \dots, x_n]^T$ is the input feature vector (e.g. ego speed and obstacle distance).
- $\mathbf{w} = [w_1, w_2, \dots, w_n]^T$ are learnable synaptic weights determining feature importance.
- $b \in \mathbb{R}$ is the bias offset determining how easily the neuron activates.
- $\sigma(\cdot)$ is an activation function (ReLU, Tanh, Sigmoid).

```
   x1 ────[ w1 ]────┐
                    │
   x2 ────[ w2 ]────┼───► Σ (w·x + b) ───► σ(z) ───► Activation a
                    │         z
    1 ────[  b ]────┘
```

### Why Do We Need Non-Linearity?
Imagine driving a car toward an obstacle. The physics of braking distance is governed by kinetic energy:

$$d_{\text{stop}} = \frac{v^2}{2 \mu g}$$

The safety boundary is **quadratic ($v^2$)**, not linear. If a neural network only contains linear transformations $W_2(W_1 x + b_1) + b_2$, the entire deep network algebraically collapses into a single matrix multiplication:

$$y = (W_2 W_1) x + (W_2 b_1 + b_2) = W_{\text{eff}} x + b_{\text{eff}}$$

A 100-layer linear network has the exact same representational power as a 1-layer perceptron. **Without non-linear activations ($\text{ReLU}(z) = \max(0, z)$), deep learning cannot separate curved or quadratic real-world decision manifolds.**

### Diagnostic Failure Modes (Andrew Ng Style)

| Symptom | Root Cause | Engineering Fix |
| :--- | :--- | :--- |
| **All neuron weights stay identical** | Zero weight initialization ($W = 0.0$). Symmetry trap prevents differentiation. | Xavier / He Normal random initialization ($w \sim \mathcal{N}(0, \sqrt{2/n_{\text{in}}})$). |
| **Loss explodes to `NaN` or `inf`** | Learning rate $\eta$ too large; gradient descent overshoots convex valleys. | Lower $\eta$ by $10\times$; apply gradient clipping ($\|\mathbf{g}\| \le 1.0$). |
| **Gradients vanish to exactly 0.0** | Saturating Sigmoid/Tanh on extreme inputs ($|z| > 5$), or dead ReLUs ($z < 0$). | Use LeakyReLU ($\alpha = 0.01$), ELU, or Batch Normalization. |

---

## Tier 2: Code from Scratch (Karpathy Micrograd Style)

The heart of automatic differentiation is the **computational graph**. Every arithmetic operation produces a new node that remembers its parents and stores a lambda function for local derivative propagation:

```python
class Value:
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

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "ReLU")
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # Topological sort of the DAG
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev: build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
```

---

## Tier 3: Deep Math & Academic Lineage

### The Multivariate Chain Rule
Consider an objective scalar loss $L$. If a variable $u$ influences $L$ through multiple downstream pathways $v_1, v_2, \dots, v_k$:

$$\frac{\partial L}{\partial u} = \sum_{j=1}^k \frac{\partial L}{\partial v_j} \cdot \frac{\partial v_j}{\partial u}$$

In backpropagation, this accumulation is why gradients **add**:
```python
self.grad += local_derivative * out.grad
```

### Finite-Difference Numerical Verification (Calculus Verification)
To verify that backpropagation is mathematically sound without trusting the code, we compute the two-sided numerical quotient:

$$\frac{\partial f}{\partial x} = \lim_{\epsilon \to 0} \frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon} + \mathcal{O}(\epsilon^2)$$

When evaluated with $\epsilon = 10^{-5}$, the relative error between analytical `x.grad` and numerical quotient must satisfy:

$$\frac{|\nabla_{\text{analytical}} - \nabla_{\text{numerical}}|}{\max(|\nabla_{\text{analytical}}|, |\nabla_{\text{numerical}}|) + 10^{-8}} < 10^{-5}$$

### Curated Online Lectures:
- **MIT OpenCourseWare 18.02**: *Multivariable Calculus — Partial Derivatives & Chain Rule* ([Link](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/))
- **MIT OpenCourseWare 6.036**: *Introduction to Machine Learning — Neural Networks & Gradient Descent* ([Link](https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/course/))
- **Andrej Karpathy**: *The spelled-out intro to neural networks and backpropagation: building micrograd* ([YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0))

---

## Tier 4: Cutting-Edge Research & PhD Frontiers

1. **Loss Landscape Geometry**:
   - Li et al. (NeurIPS 2018), *Visualizing the Loss Landscape of Neural Nets*. Deep networks with skip-connections (ResNet) create smooth convex-like funnels, whereas deep networks without skip connections exhibit chaotic, non-convex fractured terrain with exponential saddle points.
2. **The Vanishing & Exploding Gradient Problem**:
   - For an $L$-layer network, the gradient of the loss with respect to early weights is a product of $L$ Jacobian matrices:
     $$\frac{\partial L}{\partial \mathbf{W}_1} = \left( \prod_{l=2}^L \mathbf{W}_l^T \operatorname{diag}(\sigma'(z_l)) \right) \frac{\partial L}{\partial \mathbf{a}_L}$$
   - If the spectral norm $\|\mathbf{W}\| < 1$, gradients decay exponentially to zero (learning halts). If $\|\mathbf{W}\| > 1$, gradients explode exponentially (divergence).

---

## Tier 5: Practical Hardware & Compute Budgets

| Operation | Forward Pass FLOPs | Backward Pass FLOPs | Activation Memory Stored |
| :--- | :--- | :--- | :--- |
| **Linear Layer ($N \times M$)** | $2 \cdot N \cdot M$ | $4 \cdot N \cdot M$ | Vector $\mathbf{x} \in \mathbb{R}^N$ ($4N$ bytes) |
| **ReLU Activation** | $1 \cdot M$ | $1 \cdot M$ | Bitmask $\mathbf{1}_{\{z > 0\}}$ ($M/8$ bytes) |
| **Total per Parameter** | **2 FLOPs** | **4 FLOPs** | **Total Backward is $2\times$ Forward compute** |

**Hardware Rule of Thumb**: The backward pass requires **$2\times$ more FLOPs** than the forward pass, and requires storing all intermediate pre-activation tensors in GPU SRAM/HBM until backward execution completes.
