from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from .layers import Dense
from .activations import softmax, softmax_backward


@dataclass
class SelfAttention:
    """
    Single-head scaled dot-product self-attention.

    Input:  x of shape (batch, seq, d_model)
    Output: y of shape (batch, seq, d_model)

    We learn four linear projections:
      Q = x W_q
      K = x W_k
      V = x W_v
      out = (softmax(Q K^T / sqrt(d_k)) V) W_o

    Causal masking ensures each position attends only to itself and previous positions.
    """

    d_model: int
    causal: bool = True

    # Projections
    W_q: np.ndarray = field(init=False)
    W_k: np.ndarray = field(init=False)
    W_v: np.ndarray = field(init=False)
    W_o: np.ndarray = field(init=False)

    dW_q: np.ndarray = field(init=False)
    dW_k: np.ndarray = field(init=False)
    dW_v: np.ndarray = field(init=False)
    dW_o: np.ndarray = field(init=False)

    # Caches for backward
    _x: np.ndarray | None = field(default=None, init=False, repr=False)
    _Q: np.ndarray | None = field(default=None, init=False, repr=False)
    _K: np.ndarray | None = field(default=None, init=False, repr=False)
    _V: np.ndarray | None = field(default=None, init=False, repr=False)
    _scores: np.ndarray | None = field(default=None, init=False, repr=False)
    _attn: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        limit = 1.0 / np.sqrt(self.d_model)
        def init_w() -> np.ndarray:
            return np.random.uniform(-limit, limit, size=(self.d_model, self.d_model)).astype(
                np.float32
            )

        self.W_q = init_w()
        self.W_k = init_w()
        self.W_v = init_w()
        self.W_o = init_w()

        self.dW_q = np.zeros_like(self.W_q)
        self.dW_k = np.zeros_like(self.W_k)
        self.dW_v = np.zeros_like(self.W_v)
        self.dW_o = np.zeros_like(self.W_o)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (B, T, D)
        returns: (B, T, D)
        """
        B, T, D = x.shape
        assert D == self.d_model
        self._x = x

        # Linear projections
        Q = x @ self.W_q  # (B, T, D)
        K = x @ self.W_k  # (B, T, D)
        V = x @ self.W_v  # (B, T, D)
        self._Q, self._K, self._V = Q, K, V

        # Scaled dot-product scores
        scale = 1.0 / np.sqrt(self.d_model)
        scores = (Q @ K.transpose(0, 2, 1)) * scale  # (B, T, T)

        if self.causal:
            # Mask out future positions: set them to large negative value before softmax
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            scores = scores.copy()
            scores[:, mask] = -1e9

        # Attention weights
        attn = softmax(scores, axis=-1)  # (B, T, T)

        # Weighted sum of values
        context = attn @ V  # (B, T, D)

        # Final projection
        out = context @ self.W_o  # (B, T, D)

        self._scores = scores
        self._attn = attn
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        grad_output: (B, T, D) gradient wrt attention output
        returns: gradient wrt input x, same shape as x
        """
        if any(v is None for v in (self._x, self._Q, self._K, self._V, self._attn, self._scores)):
            raise RuntimeError("SelfAttention.backward called before forward")

        x = self._x
        Q = self._Q
        K = self._K
        V = self._V
        attn = self._attn
        scores = self._scores

        B, T, D = x.shape
        scale = 1.0 / np.sqrt(self.d_model)

        # grad wrt W_o and context
        context = attn @ V  # recompute or cache; safe to recompute from cached tensors
        self.dW_o = (context.reshape(-1, D).T @ grad_output.reshape(-1, D))
        grad_context = grad_output @ self.W_o.T  # (B, T, D)

        # grad wrt attn and V from context = attn @ V
        # context_{bt} = sum_j attn_{btj} * V_{bj}
        grad_attn = grad_context @ V.transpose(0, 2, 1)  # (B, T, T)
        grad_V = attn.transpose(0, 2, 1) @ grad_context  # (B, T, D)

        # Backprop through softmax over scores
        grad_scores = softmax_backward(grad_attn, attn, axis=-1)  # (B, T, T)

        # Causal mask has zero gradient because masked scores are constant (-1e9)

        # scores = (Q K^T) * scale
        dQ_from_scores = grad_scores @ K  # (B, T, D)
        dK_from_scores = grad_scores.transpose(0, 2, 1) @ Q  # (B, T, D)
        dQ_from_scores *= scale
        dK_from_scores *= scale

        # Combine gradients for Q, K, V
        dQ = dQ_from_scores  # (B, T, D)
        dK = dK_from_scores  # (B, T, D)
        dV = grad_V          # (B, T, D)

        # Gradients wrt projection weights and input x
        x_flat = x.reshape(-1, D)
        dQ_flat = dQ.reshape(-1, D)
        dK_flat = dK.reshape(-1, D)
        dV_flat = dV.reshape(-1, D)
        self.dW_q = x_flat.T @ dQ_flat
        self.dW_k = x_flat.T @ dK_flat
        self.dW_v = x_flat.T @ dV_flat

        # Gradient wrt x from each path
        dX_q = dQ @ self.W_q.T
        dX_k = dK @ self.W_k.T
        dX_v = dV @ self.W_v.T
        grad_x = dX_q + dX_k + dX_v
        return grad_x

    def zero_grad(self) -> None:
        self.dW_q.fill(0.0)
        self.dW_k.fill(0.0)
        self.dW_v.fill(0.0)
        self.dW_o.fill(0.0)


if __name__ == "__main__":
    np.random.seed(0)
    attn = SelfAttention(d_model=8, causal=True)
    x = np.random.randn(2, 4, 8).astype(np.float32)
    out = attn.forward(x)
    print("Attention output shape:", out.shape)
    grad_out = np.ones_like(out)
    grad_x = attn.backward(grad_out)
    print("Grad input shape:", grad_x.shape)

