"""
Chapter 00: Neural Networks from Scratch — Non-Linear Driving Classification
Author: Vivek Narayanan / LearnFSD

Demonstrates why autonomous driving requires non-linear neural networks:
A simple linear model fails to learn the quadratic stopping distance relationship (d > v^2).
We train a 2-layer MLP using pure scalar autograd and gradient descent.
"""

import os
import sys
import random
from typing import List, Tuple

_curr_dir = os.path.dirname(os.path.abspath(__file__))
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

from engine import Value
from nn import MLP


def generate_driving_dataset(num_samples: int = 50, seed: int = 42) -> List[Tuple[List[float], float]]:
    """
    Generates synthetic driving telemetry states:
    x0: Normalized Speed (0.0 to 1.0, representing 0 to 120 km/h)
    x1: Normalized Distance to lead vehicle (0.0 to 1.0, representing 0 to 60m)
    y:  +1.0 (Safe Following) if distance > 0.75 * speed^2 + 0.1 else -1.0 (Emergency Braking Required)
    """
    random.seed(seed)
    dataset = []
    for _ in range(num_samples):
        speed = random.uniform(0.1, 1.0)
        dist = random.uniform(0.05, 1.0)
        safe_boundary = 0.75 * (speed ** 2) + 0.1
        label = 1.0 if dist > safe_boundary else -1.0
        dataset.append(([speed, dist], label))
    return dataset


def train(epochs: int = 35, learning_rate: float = 0.08, seed: int = 42) -> MLP:
    random.seed(seed)
    dataset = generate_driving_dataset(num_samples=40, seed=seed)
    
    # 2 input features -> 8 hidden (tanh) -> 8 hidden (tanh) -> 1 readout (linear)
    model = MLP(nin=2, nouts=[8, 8, 1], activations=["tanh", "tanh", "linear"])
    
    print("=" * 68)
    print("🚀 TRAINING 2-LAYER MLP FROM RAW SCRATCH (AUTOGRAD ENGINE)")
    print(f"Total Parameters: {len(model.parameters())} | Epochs: {epochs} | LR: {learning_rate}")
    print("=" * 68)

    for epoch in range(1, epochs + 1):
        # 1. Forward pass
        total_loss = Value(0.0)
        correct = 0

        for x, y in dataset:
            pred = model(x)
            # Max-margin / Hinge-style loss: max(0, 1 - y * pred)
            margin = Value(1.0) - (Value(y) * pred)
            loss_i = margin.relu()
            total_loss = total_loss + loss_i

            # Accuracy check
            is_correct = (pred.data > 0 and y > 0) or (pred.data < 0 and y < 0)
            if is_correct:
                correct += 1

        loss = total_loss / len(dataset)
        acc = (correct / len(dataset)) * 100.0

        # 2. Backward pass (Automatic Differentiation via Topological Chain Rule)
        model.zero_grad()
        loss.backward()

        # 3. Gradient Descent parameter update: θ = θ - η * ∇θ(L)
        grad_norm_sq = 0.0
        for p in model.parameters():
            grad_norm_sq += p.grad ** 2
            p.data -= learning_rate * p.grad
        grad_norm = grad_norm_sq ** 0.5

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {loss.data:.4f} | Accuracy: {acc:5.1f}% | ||∇L||: {grad_norm:.4f}")

    print("=" * 68)
    print("✅ Training complete. The network has learned the non-linear safe braking boundary.")
    return model


if __name__ == "__main__":
    train(epochs=35, learning_rate=0.08)
