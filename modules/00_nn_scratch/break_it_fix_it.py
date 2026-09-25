"""
Chapter 00: Break-It-Fix-It Diagnostic Drills
Author: Vivek Narayanan / LearnFSD

Hands-on diagnostic drills modeled on Andrew Ng's systematic error analysis:
Diagnose the failure mode, inspect the symptoms, and apply the engineering fix.
"""

import os
import sys
import random

_curr_dir = os.path.dirname(os.path.abspath(__file__))
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

from engine import Value
from nn import Neuron, Layer, MLP


def drill_zero_initialization():
    """
    DRILL 1: The Symmetry Trap (Zero Weight Initialization).
    If all weights are initialized to 0.0, all neurons in a layer compute
    the exact same output and receive the exact same gradient.
    They remain permanently symmetric and can never learn diverse features!
    """
    print("--- DRILL 1: Zero Weight Initialization Symmetry ---")
    layer = Layer(nin=2, nout=3, nonlin="relu")
    # Defect: zero out all weights
    for p in layer.parameters():
        p.data = 0.0

    x = [Value(1.5), Value(-2.0)]
    out = layer(x)
    print(f"Outputs with zero weights: {[f'{v.data:.2f}' for v in out]}")
    
    # Run backprop on sum of outputs
    loss = sum(out, Value(0.0))
    loss.backward()
    grads = [f"{p.grad:.2f}" for p in layer.parameters()]
    print(f"Gradients across all 3 neurons: {grads}")
    print("Symptom: All neuron gradients are identical! The network is trapped in symmetry.")
    print("Fix: Use Xavier / He normal random initialization (e.g. w ~ N(0, 1/sqrt(nin))).\n")


def drill_exploding_learning_rate():
    """
    DRILL 2: The Learning Rate Divergence Catastrophe.
    When the learning rate is too large, gradient descent steps overshoot the valley
    and bounce higher and higher up the loss surface.
    """
    print("--- DRILL 2: Exploding Learning Rate ---")
    w = Value(2.0)
    # Objective: Minimize L = w^2 (Minimum is at w = 0)
    lr_safe = 0.1
    lr_exploding = 1.5  # Overshoots: w_new = w - 1.5 * 2w = w - 3w = -2w (oscillates with magnitude 2x!)

    w_val = 2.0
    print(f"Initial w = {w_val:.2f}, Goal: Minimize L = w^2")
    for step in range(1, 4):
        # Gradient of w^2 is 2w
        grad = 2.0 * w_val
        w_val -= lr_exploding * grad
        loss = w_val ** 2
        print(f"Step {step}: w = {w_val:.2f}, Loss = {loss:.2f}")

    print("Symptom: Loss explodes exponentially (2 -> 16 -> 256...)!")
    print("Fix: Reduce learning rate or apply gradient clipping: clip(grad, -1.0, 1.0).\n")


def drill_missing_nonlinearity():
    """
    DRILL 3: The Missing Non-Linearity Illusion.
    Without non-linear activations, stacking 10 deep layers is mathematically
    equivalent to a single linear layer: W3 * (W2 * (W1 * x)) = (W3 * W2 * W1) * x = W_eff * x.
    """
    print("--- DRILL 3: The Linear Collapse Illusion ---")
    mlp_linear = MLP(nin=2, nouts=[4, 4, 1], activations=["linear", "linear", "linear"])
    mlp_nonlinear = MLP(nin=2, nouts=[4, 4, 1], activations=["tanh", "tanh", "linear"])
    print("Without non-linear activations (ReLU / Tanh), a deep network CANNOT learn non-linear boundaries.")
    print("Symptom: Model fails to separate quadratic braking distance boundaries (d > v^2).")
    print("Fix: Always interleave linear affine projections with non-linear activation functions.\n")


if __name__ == "__main__":
    drill_zero_initialization()
    drill_exploding_learning_rate()
    drill_missing_nonlinearity()
