from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np

from core.embedding import Embedding
from core.transformer import TransformerBlock
from core.layers import Dense, LayerNorm


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 64
    n_layers: int = 2
    d_ff: int = 128
    max_seq_len: int = 128


class LanguageModel:
    """
    Minimal transformer-based character-level language model.

    Components:
      - Token embedding
      - Positional embedding
      - N transformer blocks
      - Final layer norm
      - Linear projection to vocab logits
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.token_embedding = Embedding(config.vocab_size, config.d_model)
        # Positional embeddings for positions [0, max_seq_len)
        self.pos_embedding = Embedding(config.max_seq_len, config.d_model)
        self.blocks: List[TransformerBlock] = [
            TransformerBlock(d_model=config.d_model, d_ff=config.d_ff)
            for _ in range(config.n_layers)
        ]
        self.ln_f = LayerNorm(dim=config.d_model)
        self.fc_out = Dense(config.d_model, config.vocab_size)

        # cache for backward
        self._pos_ids: np.ndarray | None = None

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """
        tokens: (B, T) integer ids
        returns: logits of shape (B, T, V)
        """
        B, T = tokens.shape
        assert T <= self.config.max_seq_len, "Sequence too long for configured max_seq_len"

        # Token + positional embeddings
        tok_emb = self.token_embedding.forward(tokens)  # (B, T, D)
        pos_ids = np.arange(T, dtype=np.int64)[None, :].repeat(B, axis=0)
        pos_emb = self.pos_embedding.forward(pos_ids)  # (B, T, D)
        x = tok_emb + pos_emb
        self._pos_ids = pos_ids

        # Transformer blocks
        for block in self.blocks:
            x = block.forward(x)

        # Final layer norm and projection
        x = self.ln_f.forward(x)
        logits = self.fc_out.forward(x)
        return logits

    def backward(self, grad_logits: np.ndarray) -> None:
        """
        Backpropagate gradient from logits to all parameters.
        """
        # Backward through final projection and layer norm
        grad_x = self.fc_out.backward(grad_logits)
        grad_x = self.ln_f.backward(grad_x)

        # Backward through transformer blocks (reverse order)
        for block in reversed(self.blocks):
            grad_x = block.backward(grad_x)

        # Split gradient into token and positional parts
        # Both received the same gradient when added
        grad_tok_emb = grad_x
        grad_pos_emb = grad_x

        # Backward into embeddings
        self.token_embedding.backward(grad_tok_emb)
        self.pos_embedding.backward(grad_pos_emb)

    def zero_grad(self) -> None:
        """
        Reset all gradient buffers to zero.
        """
        self.token_embedding.grad_weight.fill(0.0)
        self.pos_embedding.grad_weight.fill(0.0)
        for block in self.blocks:
            block.zero_grad()
        self.fc_out.dW.fill(0.0)
        self.fc_out.db.fill(0.0)
        self.ln_f.dgamma.fill(0.0)
        self.ln_f.dbeta.fill(0.0)

    def step(self, lr: float) -> None:
        """
        Simple SGD update with learning rate lr.
        """
        self.token_embedding.weight -= lr * self.token_embedding.grad_weight
        self.pos_embedding.weight -= lr * self.pos_embedding.grad_weight
        for block in self.blocks:
            # attention weights
            block.attn.W_q -= lr * block.attn.dW_q
            block.attn.W_k -= lr * block.attn.dW_k
            block.attn.W_v -= lr * block.attn.dW_v
            block.attn.W_o -= lr * block.attn.dW_o
            # feed-forward
            block.ff1.W -= lr * block.ff1.dW
            block.ff1.b -= lr * block.ff1.db
            block.ff2.W -= lr * block.ff2.dW
            block.ff2.b -= lr * block.ff2.db
            # layer norms
            block.ln1.gamma -= lr * block.ln1.dgamma
            block.ln1.beta -= lr * block.ln1.dbeta
            block.ln2.gamma -= lr * block.ln2.dgamma
            block.ln2.beta -= lr * block.ln2.dbeta
        # final layer norm and output
        self.ln_f.gamma -= lr * self.ln_f.dgamma
        self.ln_f.beta -= lr * self.ln_f.dbeta
        self.fc_out.W -= lr * self.fc_out.dW
        self.fc_out.b -= lr * self.fc_out.db

    # ---------- Serialization ----------

    def state_dict(self) -> Dict[str, np.ndarray]:
        """
        Flatten all parameters into a dict of numpy arrays.
        Keys are stable and versioned by design.
        """
        state: Dict[str, np.ndarray] = {}
        state["token_embedding.weight"] = self.token_embedding.weight
        state["pos_embedding.weight"] = self.pos_embedding.weight
        for i, block in enumerate(self.blocks):
            prefix = f"blocks.{i}."
            state[prefix + "attn.W_q"] = block.attn.W_q
            state[prefix + "attn.W_k"] = block.attn.W_k
            state[prefix + "attn.W_v"] = block.attn.W_v
            state[prefix + "attn.W_o"] = block.attn.W_o
            state[prefix + "ff1.W"] = block.ff1.W
            state[prefix + "ff1.b"] = block.ff1.b
            state[prefix + "ff2.W"] = block.ff2.W
            state[prefix + "ff2.b"] = block.ff2.b
            state[prefix + "ln1.gamma"] = block.ln1.gamma
            state[prefix + "ln1.beta"] = block.ln1.beta
            state[prefix + "ln2.gamma"] = block.ln2.gamma
            state[prefix + "ln2.beta"] = block.ln2.beta
        state["ln_f.gamma"] = self.ln_f.gamma
        state["ln_f.beta"] = self.ln_f.beta
        state["fc_out.W"] = self.fc_out.W
        state["fc_out.b"] = self.fc_out.b
        return state

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        """
        Load parameters from a dict created by state_dict().
        """
        self.token_embedding.weight[...] = state["token_embedding.weight"]
        self.pos_embedding.weight[...] = state["pos_embedding.weight"]
        for i, block in enumerate(self.blocks):
            prefix = f"blocks.{i}."
            block.attn.W_q[...] = state[prefix + "attn.W_q"]
            block.attn.W_k[...] = state[prefix + "attn.W_k"]
            block.attn.W_v[...] = state[prefix + "attn.W_v"]
            block.attn.W_o[...] = state[prefix + "attn.W_o"]
            block.ff1.W[...] = state[prefix + "ff1.W"]
            block.ff1.b[...] = state[prefix + "ff1.b"]
            block.ff2.W[...] = state[prefix + "ff2.W"]
            block.ff2.b[...] = state[prefix + "ff2.b"]
            block.ln1.gamma[...] = state[prefix + "ln1.gamma"]
            block.ln1.beta[...] = state[prefix + "ln1.beta"]
            block.ln2.gamma[...] = state[prefix + "ln2.gamma"]
            block.ln2.beta[...] = state[prefix + "ln2.beta"]
        self.ln_f.gamma[...] = state["ln_f.gamma"]
        self.ln_f.beta[...] = state["ln_f.beta"]
        self.fc_out.W[...] = state["fc_out.W"]
        self.fc_out.b[...] = state["fc_out.b"]


if __name__ == "__main__":
    # Simple smoke test for forward/backward and step.
    np.random.seed(0)
    config = ModelConfig(vocab_size=20, d_model=16, n_layers=1, d_ff=32, max_seq_len=8)
    model = LanguageModel(config)
    tokens = np.random.randint(0, config.vocab_size, size=(2, 8), dtype=np.int64)
    logits = model.forward(tokens)
    print("Logits shape:", logits.shape)
    grad_logits = np.random.randn(*logits.shape).astype(np.float32)
    model.zero_grad()
    model.backward(grad_logits)
    model.step(lr=0.01)
    print("Smoke test completed.")

