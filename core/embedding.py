from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class Embedding:
    """
    Token embedding layer.

    Maps integer token ids to dense vectors.
    We keep explicit gradients and update only rows that were used.

    weight: (vocab_size, d_model)
    """

    vocab_size: int
    d_model: int
    weight: np.ndarray = field(init=False)
    grad_weight: np.ndarray = field(init=False)
    _last_tokens: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        limit = 1.0 / np.sqrt(self.d_model)
        self.weight = np.random.uniform(-limit, limit, size=(self.vocab_size, self.d_model)).astype(
            np.float32
        )
        self.grad_weight = np.zeros_like(self.weight)

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """
        tokens: (batch, seq) integer ids
        returns: (batch, seq, d_model)
        """
        self._last_tokens = tokens
        return self.weight[tokens]

    def backward(self, grad_output: np.ndarray) -> None:
        """
        Accumulate gradients into grad_weight for used token rows only.

        grad_output: (batch, seq, d_model) gradient wrt embeddings.
        """
        if self._last_tokens is None:
            raise RuntimeError("Embedding.backward called before forward")
        tokens = self._last_tokens

        # Reset gradients
        self.grad_weight.fill(0.0)

        # Flatten batch and seq
        flat_tokens = tokens.reshape(-1)
        flat_grad = grad_output.reshape(-1, self.d_model)

        # For each occurrence of token id i, add its gradient to row i.
        np.add.at(self.grad_weight, flat_tokens, flat_grad)


if __name__ == "__main__":
    np.random.seed(0)
    emb = Embedding(vocab_size=10, d_model=4)
    tokens = np.array([[1, 2, 3], [3, 2, 1]], dtype=np.int64)
    out = emb.forward(tokens)
    print("Embedding output shape:", out.shape)
    grad_out = np.ones_like(out)
    emb.backward(grad_out)
    # Only token rows 1,2,3 should have non-zero gradients.
    print("Non-zero grad rows:", np.where(np.abs(emb.grad_weight).sum(axis=1) > 0)[0])

