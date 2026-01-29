from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class LongTermMemory:
    """
    Simple vector-based long-term memory using cosine similarity.

    Each memory item is a pair:
      - vector: fixed-size numpy array (embedding)
      - text:   associated raw text (e.g., interaction snippet)
    """

    dim: int
    vectors: List[np.ndarray] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)

    def add(self, vector: np.ndarray, text: str) -> None:
        assert vector.ndim == 1 and vector.shape[0] == self.dim
        self.vectors.append(vector.astype(np.float32))
        self.texts.append(text)

    def _stack_vectors(self) -> np.ndarray:
        if not self.vectors:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack(self.vectors, axis=0)

    def query(self, vector: np.ndarray, top_k: int = 3) -> List[Tuple[float, str]]:
        """
        Retrieve top_k most similar memories by cosine similarity.
        Returns list of (similarity, text) sorted descending.
        """
        assert vector.ndim == 1 and vector.shape[0] == self.dim
        if not self.vectors:
            return []

        vecs = self._stack_vectors()  # (N, D)
        q = vector.astype(np.float32)

        # Normalize
        vecs_norm = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        q_norm = np.linalg.norm(q) + 1e-12
        vecs_unit = vecs / vecs_norm
        q_unit = q / q_norm

        sims = vecs_unit @ q_unit  # (N,)
        idx = np.argsort(-sims)[:top_k]
        results: List[Tuple[float, str]] = []
        for i in idx:
            results.append((float(sims[i]), self.texts[i]))
        return results


if __name__ == "__main__":
    np.random.seed(0)
    ltm = LongTermMemory(dim=4)
    ltm.add(np.array([1.0, 0.0, 0.0, 0.0]), "x-axis")
    ltm.add(np.array([0.0, 1.0, 0.0, 0.0]), "y-axis")
    q = np.array([0.9, 0.1, 0.0, 0.0])
    results = ltm.query(q, top_k=2)
    print("Results:", results)

