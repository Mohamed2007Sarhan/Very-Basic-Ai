from dataclasses import dataclass

import numpy as np

from .activations import softmax


@dataclass
class CrossEntropyLoss:
    """
    Cross-entropy loss for next-token prediction.

    We take logits of shape (B, T, V) and integer targets of shape (B, T),
    ignoring positions where target == ignore_index.

    For each valid position (b, t):
      p = softmax(logits[b, t])
      loss = -log p[target]
    """

    ignore_index: int = -100

    # cache for backward
    _probs: np.ndarray | None = None
    _targets: np.ndarray | None = None
    _mask: np.ndarray | None = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        logits: (B, T, V)
        targets: (B, T) int64
        returns: scalar loss (float)
        """
        assert logits.ndim == 3
        assert targets.shape == logits.shape[:2]

        B, T, V = logits.shape
        probs = softmax(logits, axis=-1)

        mask = targets != self.ignore_index
        self._probs = probs
        self._targets = targets
        self._mask = mask

        # Gather probabilities of the correct classes
        flat_probs = probs.reshape(-1, V)
        flat_targets = targets.reshape(-1)
        flat_mask = mask.reshape(-1)

        idx = np.arange(flat_probs.shape[0])
        # To avoid indexing invalid positions, temporarily set their target to 0
        safe_targets = flat_targets.copy()
        safe_targets[~flat_mask] = 0
        correct_probs = flat_probs[idx, safe_targets]

        # For masked positions, skip contribution
        log_probs = np.zeros_like(correct_probs)
        valid = flat_mask
        log_probs[valid] = -np.log(correct_probs[valid] + 1e-12)

        loss = log_probs[valid].mean() if valid.any() else 0.0
        return float(loss)

    def backward(self) -> np.ndarray:
        """
        Returns gradient of loss w.r.t logits, same shape as logits.
        """
        if self._probs is None or self._targets is None or self._mask is None:
            raise RuntimeError("CrossEntropyLoss.backward called before forward")

        probs = self._probs
        targets = self._targets
        mask = self._mask

        B, T, V = probs.shape

        grad = probs.copy()
        flat_grad = grad.reshape(-1, V)
        flat_targets = targets.reshape(-1)
        flat_mask = mask.reshape(-1)
        idx = np.arange(flat_grad.shape[0])

        safe_targets = flat_targets.copy()
        safe_targets[~flat_mask] = 0
        flat_grad[idx, safe_targets] -= 1.0

        # Zero-out masked positions
        flat_grad[~flat_mask] = 0.0

        # Normalize by number of valid positions (to match mean loss)
        num_valid = flat_mask.sum()
        if num_valid > 0:
            flat_grad /= num_valid

        return grad


if __name__ == "__main__":
    np.random.seed(0)
    loss_fn = CrossEntropyLoss(ignore_index=-100)
    logits = np.random.randn(2, 3, 5).astype(np.float32)
    targets = np.array([[1, 2, -100], [0, 4, 3]], dtype=np.int64)
    loss = loss_fn.forward(logits, targets)
    print("Loss:", loss)
    grad = loss_fn.backward()
    print("Grad shape:", grad.shape)

