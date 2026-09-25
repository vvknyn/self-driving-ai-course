"""
Unit Tests for Chapter 00: Neural Networks & Autograd from Scratch
Author: Vivek Narayanan / LearnFSD

Verifies mathematical correctness of forward passes, reverse-mode autodiff,
and finite-difference numerical gradient checking (Andrew Ng / Karpathy standard).
"""

import os
import sys
import unittest

_curr_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.abspath(os.path.join(_curr_dir, ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from engine import Value
from nn import Neuron, Layer, MLP


class TestNNScratch(unittest.TestCase):

    def test_forward_arithmetic(self):
        """Verifies fundamental forward arithmetic operators."""
        a = Value(3.0)
        b = Value(-2.0)
        c = a + b       # 1.0
        d = a * b       # -6.0
        e = a / 2.0     # 1.5
        f = (a ** 2)    # 9.0
        self.assertAlmostEqual(c.data, 1.0)
        self.assertAlmostEqual(d.data, -6.0)
        self.assertAlmostEqual(e.data, 1.5)
        self.assertAlmostEqual(f.data, 9.0)

    def test_finite_difference_gradient_check(self):
        """
        Calculus verification: compares reverse-mode analytical gradients
        against finite-difference numerical approximations:
        df/dx ≈ (f(x + ε) - f(x - ε)) / (2ε)
        """
        eps = 1e-5

        def func(x_val, y_val):
            # Non-linear expression: f(x, y) = tanh(x * y) + relu(x^2 - y)
            x = Value(x_val)
            y = Value(y_val)
            term1 = (x * y).tanh()
            term2 = ((x ** 2) - y).relu()
            return term1 + term2, x, y

        x0, y0 = 1.2, -0.8

        # 1. Analytical gradient via backpropagation
        out, x_var, y_var = func(x0, y0)
        out.backward()
        grad_x_analytical = x_var.grad
        grad_y_analytical = y_var.grad

        # 2. Numerical gradient for x
        out_plus, _, _ = func(x0 + eps, y0)
        out_minus, _, _ = func(x0 - eps, y0)
        grad_x_numerical = (out_plus.data - out_minus.data) / (2.0 * eps)

        # 3. Numerical gradient for y
        out_plus_y, _, _ = func(x0, y0 + eps)
        out_minus_y, _, _ = func(x0, y0 - eps)
        grad_y_numerical = (out_plus_y.data - out_minus_y.data) / (2.0 * eps)

        # Assert relative tolerance < 1e-4
        self.assertAlmostEqual(grad_x_analytical, grad_x_numerical, places=4)
        self.assertAlmostEqual(grad_y_analytical, grad_y_numerical, places=4)

    def test_mlp_convergence_single_step(self):
        """Verifies that an MLP parameter update strictly decreases loss."""
        mlp = MLP(nin=2, nouts=[4, 1], activations=["tanh", "linear"])
        x = [Value(0.5), Value(-0.2)]
        target = Value(1.0)

        # Forward pass 1
        out1 = mlp(x)
        loss1 = (out1 - target) ** 2

        # Backward & Update
        mlp.zero_grad()
        loss1.backward()

        for p in mlp.parameters():
            p.data -= 0.1 * p.grad

        # Forward pass 2 after step
        out2 = mlp(x)
        loss2 = (out2 - target) ** 2

        self.assertLess(loss2.data, loss1.data, "Gradient descent step must reduce loss on single sample.")

    def test_zero_grad_resets_buffers(self):
        """Verifies that zero_grad() clears accumulated gradients."""
        mlp = MLP(nin=2, nouts=[3, 1])
        x = [Value(1.0), Value(2.0)]
        out = mlp(x)
        out.backward()

        has_nonzero_grad = any(abs(p.grad) > 0.0 for p in mlp.parameters())
        self.assertTrue(has_nonzero_grad, "Gradients should be non-zero after backward()")

        mlp.zero_grad()
        all_zero = all(p.grad == 0.0 for p in mlp.parameters())
        self.assertTrue(all_zero, "zero_grad() must reset all parameter gradients to 0.0")


if __name__ == "__main__":
    unittest.main()
