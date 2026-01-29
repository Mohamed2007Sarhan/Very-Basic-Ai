from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ReLU:
    """
    Element-wise ReLU activation: f(x) = max(0, x)

    Backprop:
      df/dx = 1 for x > 0, else 0
    """

    mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return np.where(self.mask, x, 0.0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.mask is None:
            raise RuntimeError("ReLU.backward called before forward")
        return grad_output * self.mask


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax.

    softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_shifted = np.exp(x - x_max)
    sum_exp = np.sum(exp_shifted, axis=axis, keepdims=True)
    return exp_shifted / sum_exp


def softmax_backward(grad_output: np.ndarray, softmax_output: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Backprop for softmax given upstream gradient and the softmax output.

    For each vector s = softmax(z), the Jacobian J has:
      J_ij = s_i * (delta_ij - s_j)
    We compute J^T * grad_output efficiently without forming J explicitly.
    """
    # grad w.r.t. pre-softmax logits z:
    # dL/dz_i = sum_j J_ji * dL/ds_j
    dot = np.sum(grad_output * softmax_output, axis=axis, keepdims=True)
    grad_input = softmax_output * (grad_output - dot)
    return grad_input


if __name__ == "__main__":
    # Quick numerical sanity check: softmax outputs sum to 1
    x = np.array([[1.0, 2.0, 3.0]])
    s = softmax(x, axis=-1)
    print("Softmax:", s)
    print("Row sums:", s.sum(axis=-1))

