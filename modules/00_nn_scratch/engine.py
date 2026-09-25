"""
Chapter 00: Neural Networks from Scratch — Autograd Engine
Author: Vivek Narayanan / LearnFSD

A pure Python, zero-dependency scalar autograd engine implementing a directed acyclic graph (DAG)
for reverse-mode automatic differentiation (backpropagation).

Theoretical Lineage:
- Andrej Karpathy's Micrograd (2020)
- Rumelhart, Hinton, & Williams (1986): Learning representations by back-propagating errors.
- MIT OCW 6.036 / 18.02 Multivariable Calculus: Multivariate Chain Rule
"""

from __future__ import annotations
import math
from typing import Set, Tuple, Union, List


class Value:
    """
    Stores a scalar value and its accumulated gradient with respect to a scalar objective (Loss).
    Maintains a computational graph of parents (_children) and the local derivative (_backward).
    """

    def __init__(self, data: float, _children: Tuple[Value, ...] = (), _op: str = "", label: str = ""):
        self.data: float = float(data)
        self.grad: float = 0.0
        self._backward = lambda: None
        self._prev: Set[Value] = set(_children)
        self._op: str = _op
        self.label: str = label

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other: Union[Value, float, int]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(x + y)/dx = 1.0, d(x + y)/dy = 1.0
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Union[Value, float, int]) -> Value:
        return self + other

    def __sub__(self, other: Union[Value, float, int]) -> Value:
        return self + (-other)

    def __rsub__(self, other: Union[Value, float, int]) -> Value:
        return Value(other) - self

    def __mul__(self, other: Union[Value, float, int]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # Product rule: d(x * y)/dx = y, d(x * y)/dy = x
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Union[Value, float, int]) -> Value:
        return self * other

    def __truediv__(self, other: Union[Value, float, int]) -> Value:
        # x / y = x * (y ** -1)
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)

    def __rtruediv__(self, other: Union[Value, float, int]) -> Value:
        return Value(other) / self

    def __pow__(self, power: Union[float, int]) -> Value:
        assert isinstance(power, (int, float)), "Power must be int or float"
        out = Value(self.data ** power, (self,), f"**{power}")

        def _backward():
            # Power rule: d(x^n)/dx = n * x^(n - 1)
            self.grad += (power * (self.data ** (power - 1))) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> Value:
        return self * -1.0

    def relu(self) -> Value:
        """
        Rectified Linear Unit: f(x) = max(0, x)
        Derivative: f'(x) = 1 if x > 0 else 0
        """
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        """
        Hyperbolic Tangent: f(x) = (e^(2x) - 1) / (e^(2x) + 1)
        Derivative: f'(x) = 1 - tanh^2(x)
        """
        # Clamp to avoid numerical overflow with exp
        x = max(-20.0, min(20.0, self.data))
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        """
        Logistic Sigmoid: f(x) = 1 / (1 + e^(-x))
        Derivative: f'(x) = f(x) * (1 - f(x))
        """
        x = max(-20.0, min(20.0, self.data))
        s = 1.0 / (1.0 + math.exp(-x))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += (s * (1.0 - s)) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """
        Executes reverse-mode automatic differentiation across the entire DAG.
        Uses depth-first topological sorting to ensure gradients propagate in exact causal order.
        """
        topo: List[Value] = []
        visited: Set[Value] = set()

        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Root node seed: d(Loss)/d(Loss) = 1.0
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
