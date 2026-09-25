"""
Chapter 00: Neural Networks from Scratch — Layer and MLP Abstractions
Author: Vivek Narayanan / LearnFSD

First-principles Neural Network abstractions built purely on the scalar Value engine.
Implements Xavier/Glorot weight initialization, forward matrix-vector propagation,
and parameter management without PyTorch.
"""

import os
import sys
import random
from typing import List, Union

_curr_dir = os.path.dirname(os.path.abspath(__file__))
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

from engine import Value


class Neuron:
    """
    An artificial neuron performing affine transformation z = sum(w_i * x_i) + b
    followed by an optional non-linear activation function (ReLU or Tanh).
    """

    def __init__(self, nin: int, nonlin: str = "relu"):
        # Xavier/He normal-style initialization scaled by 1/sqrt(nin)
        scale = (2.0 / nin) ** 0.5 if nonlin == "relu" else (1.0 / nin) ** 0.5
        self.w: List[Value] = [Value(random.gauss(0.0, scale), label=f"w{i}") for i in range(nin)]
        self.b: Value = Value(0.0, label="b")
        self.nonlin: str = nonlin

    def __call__(self, x: List[Union[Value, float]]) -> Value:
        # z = w · x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.nonlin == "relu":
            return act.relu()
        elif self.nonlin == "tanh":
            return act.tanh()
        elif self.nonlin == "sigmoid":
            return act.sigmoid()
        return act

    def parameters(self) -> List[Value]:
        return self.w + [self.b]


class Layer:
    """A single neural layer consisting of a collection of parallel neurons."""

    def __init__(self, nin: int, nout: int, nonlin: str = "relu"):
        self.neurons: List[Neuron] = [Neuron(nin, nonlin=nonlin) for _ in range(nout)]

    def __call__(self, x: List[Union[Value, float]]) -> Union[Value, List[Value]]:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> List[Value]:
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params


class MLP:
    """
    Multi-Layer Perceptron (Deep Neural Network).
    Passes activations sequentially through each layer.
    """

    def __init__(self, nin: int, nouts: List[int], activations: List[str] = None):
        sz = [nin] + nouts
        if activations is None:
            # Default to ReLU for hidden layers, linear for final readout layer
            activations = ["relu"] * (len(nouts) - 1) + ["linear"]
        self.layers: List[Layer] = [
            Layer(sz[i], sz[i + 1], nonlin=activations[i]) for i in range(len(nouts))
        ]

    def __call__(self, x: List[Union[Value, float]]) -> Union[Value, List[Value]]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self) -> None:
        """Resets gradient accumulation buffers to 0.0 prior to backprop."""
        for p in self.parameters():
            p.grad = 0.0
