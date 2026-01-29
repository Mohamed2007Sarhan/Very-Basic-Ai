from dataclasses import dataclass, field

import numpy as np

from .attention import SelfAttention
from .layers import Dense, LayerNorm
from .activations import ReLU


@dataclass
class TransformerBlock:
    """
    Single transformer block with:
      - LayerNorm
      - Single-head self-attention
      - Residual connection
      - Feed-forward network (Dense -> ReLU -> Dense)
      - Second LayerNorm and residual

    Shapes:
      Input x: (B, T, D)
      Output:  (B, T, D)
    """

    d_model: int
    d_ff: int

    attn: SelfAttention = field(init=False)
    ln1: LayerNorm = field(init=False)
    ln2: LayerNorm = field(init=False)
    ff1: Dense = field(init=False)
    ff2: Dense = field(init=False)
    act: ReLU = field(init=False)

    # Caches for backward
    _x: np.ndarray | None = field(default=None, init=False, repr=False)
    _x_attn_in: np.ndarray | None = field(default=None, init=False, repr=False)
    _x_ff_in: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.attn = SelfAttention(d_model=self.d_model, causal=True)
        self.ln1 = LayerNorm(dim=self.d_model)
        self.ln2 = LayerNorm(dim=self.d_model)
        self.ff1 = Dense(self.d_model, self.d_ff)
        self.ff2 = Dense(self.d_ff, self.d_model)
        self.act = ReLU()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through transformer block.
        """
        self._x = x

        # Self-attention sub-layer with pre-norm
        x_norm1 = self.ln1.forward(x)
        self._x_attn_in = x_norm1
        attn_out = self.attn.forward(x_norm1)
        x_res1 = x + attn_out  # residual connection

        # Feed-forward sub-layer with pre-norm
        x_norm2 = self.ln2.forward(x_res1)
        self._x_ff_in = x_norm2
        ff_hidden = self.ff1.forward(x_norm2)
        ff_activated = self.act.forward(ff_hidden)
        ff_out = self.ff2.forward(ff_activated)
        x_out = x_res1 + ff_out  # second residual
        return x_out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through transformer block.

        grad_output: gradient wrt block output (B, T, D)
        returns: gradient wrt block input (B, T, D)
        """
        if self._x is None or self._x_attn_in is None or self._x_ff_in is None:
            raise RuntimeError("TransformerBlock.backward called before forward")

        x = self._x
        x_attn_in = self._x_attn_in
        x_ff_in = self._x_ff_in

        # Backprop through second residual: x_out = x_res1 + ff_out
        grad_x_res1 = grad_output.copy()
        grad_ff_out = grad_output

        # Feed-forward network backward
        grad_ff2_in = self.ff2.backward(grad_ff_out)
        grad_act_in = self.act.backward(grad_ff2_in)
        grad_ff1_in = self.ff1.backward(grad_act_in)

        # Backprop through LayerNorm 2 (pre-norm)
        grad_x_norm2 = self.ln2.backward(grad_ff1_in)

        # x_norm2 input is x_res1
        grad_x_res1 += grad_x_norm2

        # x_res1 = x + attn_out
        grad_x_from_res = grad_x_res1
        grad_attn_out = grad_x_res1

        # Attention backward
        grad_x_norm1 = self.attn.backward(grad_attn_out)

        # Backprop through LayerNorm 1 (pre-norm)
        grad_x_from_ln1 = self.ln1.backward(grad_x_norm1)

        # Combine gradients wrt x from residual path and LN1 path
        grad_x_input = grad_x_from_res + grad_x_from_ln1
        return grad_x_input

    def zero_grad(self) -> None:
        self.attn.zero_grad()
        self.ff1.dW.fill(0.0)
        self.ff1.db.fill(0.0)
        self.ff2.dW.fill(0.0)
        self.ff2.db.fill(0.0)
        self.ln1.dgamma.fill(0.0)
        self.ln1.dbeta.fill(0.0)
        self.ln2.dgamma.fill(0.0)
        self.ln2.dbeta.fill(0.0)


if __name__ == "__main__":
    np.random.seed(0)
    block = TransformerBlock(d_model=16, d_ff=32)
    x = np.random.randn(2, 4, 16).astype(np.float32)
    y = block.forward(x)
    print("Block output shape:", y.shape)
    grad_y = np.ones_like(y)
    grad_x = block.backward(grad_y)
    print("Grad input shape:", grad_x.shape)

