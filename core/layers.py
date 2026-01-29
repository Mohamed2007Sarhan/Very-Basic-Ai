from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


def xavier_init(fan_in: int, fan_out: int) -> np.ndarray:
    """Xavier/Glorot uniform initialization."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)


@dataclass
class Dense:
    """
    Fully-connected layer: y = x W + b

    x: (batch, seq, in_dim)
    W: (in_dim, out_dim)
    b: (out_dim,)
    """

    in_dim: int
    out_dim: int
    W: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    dW: np.ndarray = field(init=False)
    db: np.ndarray = field(init=False)
    _x_cache: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.W = xavier_init(self.in_dim, self.out_dim)
        self.b = np.zeros(self.out_dim, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x_cache = x
        y = x @ self.W + self.b
        return y

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._x_cache is None:
            raise RuntimeError("Dense.backward called before forward")
        x = self._x_cache

        # Gradients for parameters
        # We sum over batch and sequence dimensions
        x_2d = x.reshape(-1, self.in_dim)
        grad_2d = grad_output.reshape(-1, self.out_dim)
        self.dW = x_2d.T @ grad_2d
        self.db = grad_2d.sum(axis=0)

        # Gradient wrt input
        grad_input = grad_output @ self.W.T
        return grad_input


@dataclass
class LayerNorm:
    """
    Layer normalization over the last dimension.

    For each token vector x:
      mean = sum_i x_i / D
      var  = sum_i (x_i - mean)^2 / D
      x_hat = (x - mean) / sqrt(var + eps)
      y = gamma * x_hat + beta
    """

    dim: int
    eps: float = 1e-5
    gamma: np.ndarray = field(init=False)
    beta: np.ndarray = field(init=False)
    dgamma: np.ndarray = field(init=False)
    dbeta: np.ndarray = field(init=False)

    _x_hat: np.ndarray | None = field(default=None, init=False, repr=False)
    _mean: np.ndarray | None = field(default=None, init=False, repr=False)
    _var: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.gamma = np.ones(self.dim, dtype=np.float32)
        self.beta = np.zeros(self.dim, dtype=np.float32)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # mean, var over last dim
        self._mean = x.mean(axis=-1, keepdims=True)
        self._var = x.var(axis=-1, keepdims=True)
        x_hat = (x - self._mean) / np.sqrt(self._var + self.eps)
        self._x_hat = x_hat
        return self.gamma * x_hat + self.beta

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._x_hat is None or self._mean is None or self._var is None:
            raise RuntimeError("LayerNorm.backward called before forward")

        x_hat = self._x_hat
        var = self._var

        # gradients for gamma, beta
        self.dgamma = np.sum(grad_output * x_hat, axis=(0, 1), keepdims=False)
        self.dbeta = np.sum(grad_output, axis=(0, 1), keepdims=False)

        N = x_hat.shape[-1]
        # derivative wrt x_hat
        dx_hat = grad_output * self.gamma

        # Backprop through normalization.
        # See: https://arxiv.org/abs/1607.06450 and standard LN backward derivation.
        std_inv = 1.0 / np.sqrt(var + self.eps)

        dx = (
            (1.0 / N)
            * std_inv
            * (
                N * dx_hat
                - dx_hat.sum(axis=-1, keepdims=True)
                - x_hat * (dx_hat * x_hat).sum(axis=-1, keepdims=True)
            )
        )
        return dx


if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    dense = Dense(4, 5)
    y = dense.forward(x)
    gy = np.ones_like(y)
    gx = dense.backward(gy)
    print("Dense output shape:", y.shape, "grad input shape:", gx.shape)

    ln = LayerNorm(5)
    y2 = ln.forward(y)
    gy2 = np.ones_like(y2)
    gx2 = ln.backward(gy2)
    print("LayerNorm output shape:", y2.shape, "grad input shape:", gx2.shape)

