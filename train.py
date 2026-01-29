from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Tuple, List, Dict, Any, Optional

import numpy as np

from core.tokenizer import CharVocab
from core.loss import CrossEntropyLoss
from model import LanguageModel, ModelConfig


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_corpora(paths: List[str]) -> List[Tuple[str, str]]:
    """
    Load multiple corpus files.
    Returns list of (path, text).
    If no valid files are provided, returns a single fallback corpus.
    """
    corpora: List[Tuple[str, str]] = []
    for p in paths:
        if os.path.isfile(p):
            corpora.append((p, load_text_file(p)))
    if corpora:
        return corpora
    # Fallback tiny English text to keep system runnable without external files.
    return [(
        "<built-in>",
        "This is a tiny training corpus for a zero-knowledge character model.\n"
        "It only learns local English-like statistics, not real-world facts.\n",
    )]


def load_vocab_if_exists(vocab_path: str) -> Optional[CharVocab]:
    if not os.path.isfile(vocab_path):
        return None
    with open(vocab_path, "r", encoding="utf-8") as f:
        return CharVocab.from_json(f.read())


def extend_vocab(vocab: CharVocab, new_text: str) -> Tuple[CharVocab, int]:
    """
    Extend an existing CharVocab with any new characters found in new_text.
    Returns (vocab, num_new_chars_added).
    """
    existing = set(vocab.stoi.keys())
    chars = sorted(set(new_text))
    added = 0
    # Keep deterministic insertion order by sorted chars.
    next_id = max(vocab.itos.keys()) + 1 if vocab.itos else 0
    for ch in chars:
        if ch in existing:
            continue
        # never overwrite reserved pad token
        if ch == "<pad>":
            continue
        vocab.stoi[ch] = next_id
        vocab.itos[next_id] = ch
        next_id += 1
        added += 1
    # Ensure pad stays correct
    vocab.stoi["<pad>"] = 0
    vocab.itos[0] = "<pad>"
    return vocab, added


def expand_rows(old: np.ndarray, new_rows: int, init_scale: float) -> np.ndarray:
    """
    Expand a 2D weight matrix by adding new rows at the bottom.
    Used for embeddings and output projection when vocab grows.
    """
    assert old.ndim == 2
    old_rows, cols = old.shape
    assert new_rows >= old_rows
    if new_rows == old_rows:
        return old
    extra = np.random.uniform(-init_scale, init_scale, size=(new_rows - old_rows, cols)).astype(old.dtype)
    return np.vstack([old, extra])


def expand_cols(old: np.ndarray, new_cols: int, init_scale: float) -> np.ndarray:
    """
    Expand a 2D weight matrix by adding new columns at the right.
    Used for output projection if needed (not used in this project currently).
    """
    assert old.ndim == 2
    rows, old_cols = old.shape
    assert new_cols >= old_cols
    if new_cols == old_cols:
        return old
    extra = np.random.uniform(-init_scale, init_scale, size=(rows, new_cols - old_cols)).astype(old.dtype)
    return np.hstack([old, extra])


def expand_vector(old: np.ndarray, new_size: int) -> np.ndarray:
    """
    Expand a 1D vector by padding zeros at the end.
    Used for output bias when vocab grows.
    """
    assert old.ndim == 1
    old_size = old.shape[0]
    assert new_size >= old_size
    if new_size == old_size:
        return old
    out = np.zeros((new_size,), dtype=old.dtype)
    out[:old_size] = old
    return out


def make_batches(
    encoded: np.ndarray, block_size: int, batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a mini-batch of sequences and targets using random starting positions.
    encoded: 1D array of token ids.
    Returns:
      x: (B, T)
      y: (B, T)
    """
    assert encoded.ndim == 1
    max_start = len(encoded) - block_size - 1
    if max_start <= 0:
        raise ValueError("Corpus is too small for the chosen block_size.")
    starts = np.random.randint(0, max_start, size=(batch_size,))
    x = np.stack([encoded[s : s + block_size] for s in starts], axis=0)
    y = np.stack([encoded[s + 1 : s + 1 + block_size] for s in starts], axis=0)
    return x.astype(np.int64), y.astype(np.int64)


def _load_config_if_exists(config_path: str) -> Optional[ModelConfig]:
    if not os.path.isfile(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ModelConfig(**data)


def _load_weights_if_exists(weights_path: str) -> Optional[Dict[str, np.ndarray]]:
    if not os.path.isfile(weights_path):
        return None
    return {k: v for k, v in dict(np.load(weights_path)).items()}


def _initialize_or_load_model(
    config: ModelConfig,
    weights_path: str,
    allow_vocab_resize: bool = True,
) -> Tuple[LanguageModel, Optional[Dict[str, np.ndarray]]]:
    """
    Instantiate a model and optionally return loaded weight dict.
    """
    model = LanguageModel(config)
    weights = _load_weights_if_exists(weights_path)
    if weights is None:
        return model, None
    # load_state_dict requires exact keys/shapes. We may resize below in train().
    return model, weights


def _apply_weights_with_vocab_resize(
    model: LanguageModel,
    loaded: Dict[str, np.ndarray],
    old_vocab_size: int,
    new_vocab_size: int,
) -> None:
    """
    Load weights into `model`, expanding vocab-dependent tensors if vocab grew.

    Vocab-dependent tensors:
      - token_embedding.weight: (V, D)
      - fc_out.W: (D, V)
      - fc_out.b: (V,)
    """
    state = loaded.copy()

    if old_vocab_size != new_vocab_size:
        # token embeddings (rows)
        if "token_embedding.weight" in state:
            init_scale = 1.0 / np.sqrt(model.config.d_model)
            state["token_embedding.weight"] = expand_rows(
                state["token_embedding.weight"], new_vocab_size, init_scale=init_scale
            )
        # output projection (columns are vocab in Dense? In our Dense, W is (in_dim, out_dim) => (D, V)
        if "fc_out.W" in state:
            init_scale = np.sqrt(6.0 / (model.config.d_model + new_vocab_size))
            state["fc_out.W"] = expand_cols(
                state["fc_out.W"], new_vocab_size, init_scale=init_scale
            )
        if "fc_out.b" in state:
            state["fc_out.b"] = expand_vector(state["fc_out.b"], new_vocab_size)

    # Positional embeddings depend on max_seq_len (rows).
    # If config.max_seq_len changed upward, expand rows; if it shrank, we keep existing rows and just use a smaller window.
    if "pos_embedding.weight" in state:
        old_max_seq = state["pos_embedding.weight"].shape[0]
        new_max_seq = model.config.max_seq_len
        if new_max_seq > old_max_seq:
            init_scale = 1.0 / np.sqrt(model.config.d_model)
            state["pos_embedding.weight"] = expand_rows(
                state["pos_embedding.weight"], new_max_seq, init_scale=init_scale
            )

    # Now we expect exact keys/shapes.
    model.load_state_dict(state)


def _encode_with_oov_handling(vocab: CharVocab, text: str) -> np.ndarray:
    """
    Encode text using vocab. If a char is not in vocab, this will raise KeyError.
    We keep it strict (security/validation) and expect vocab to be extended before calling.
    """
    return np.array(vocab.encode(text), dtype=np.int64)


def train(
    corpora: List[str],
    model_out: str = "model_weights.npz",
    config_out: str = "model_config.json",
    vocab_out: str = "vocab.json",
    block_size: int = 64,
    batch_size: int = 32,
    steps: int = 2000,
    learning_rate: float = 1e-2,
    seed: int = 0,
    enable_vector_memory: bool = False,
    memory_out: str = "vector_memory.npz",
) -> None:
    """
    Incremental training:
      - If model_out exists: load weights and continue training.
      - If vocab_out exists: load vocab and extend it with new corpora chars.
      - If vocab grows: expand vocab-dependent weights instead of restarting.

    Multiple corpora:
      - Trains sequentially over each corpus file for `steps` steps (per corpus).
      - All training updates accumulate into the same weights file.
    """
    np.random.seed(seed)

    corpora_loaded = load_corpora(corpora)
    print("Corpora:")
    for p, t in corpora_loaded:
        print(f"  - {p} ({len(t)} chars)")

    # Load existing vocab if present; otherwise start from first corpus text.
    vocab = load_vocab_if_exists(vocab_out)
    if vocab is None:
        vocab = CharVocab.from_text(corpora_loaded[0][1])
        print(f"No existing vocab found. Initialized new vocab with size {len(vocab.stoi)}.")
    else:
        print(f"Loaded existing vocab with size {len(vocab.stoi)} from {vocab_out}.")

    # Extend vocab with all corpora (so encoding won’t fail mid-run)
    total_added = 0
    for _, text in corpora_loaded:
        vocab, added = extend_vocab(vocab, text)
        total_added += added
    if total_added > 0:
        print(f"Extended vocab with {total_added} new characters. New vocab size: {len(vocab.stoi)}.")
    else:
        print("No new characters found; vocab unchanged.")

    # Load config if present; otherwise create a new one.
    existing_cfg = _load_config_if_exists(config_out)
    if existing_cfg is None:
        config = ModelConfig(
            vocab_size=len(vocab.stoi),
            d_model=64,
            n_layers=2,
            d_ff=128,
            max_seq_len=block_size,
        )
        print("No existing config found; created a new config.")
    else:
        # Keep architecture stable to avoid destructive mismatch.
        config = existing_cfg
        # Update vocab_size and optionally max_seq_len (we support expanding positional embeddings).
        config.vocab_size = len(vocab.stoi)
        if block_size > config.max_seq_len:
            print(f"Expanding max_seq_len from {config.max_seq_len} to {block_size}.")
            config.max_seq_len = block_size
        else:
            # We can train with a smaller block_size than the model max_seq_len.
            print(f"Using block_size={block_size} with model max_seq_len={config.max_seq_len}.")

    # Initialize model and load weights if present (with resizing if needed)
    model, loaded_weights = _initialize_or_load_model(config, weights_path=model_out)
    if loaded_weights is None:
        print("No existing weights found. Training from scratch.")
    else:
        # Determine old vocab size from stored weights
        if "token_embedding.weight" in loaded_weights:
            old_vocab_size = int(loaded_weights["token_embedding.weight"].shape[0])
        else:
            raise ValueError("Existing weights file is missing token_embedding.weight (corrupt or wrong file).")
        new_vocab_size = config.vocab_size
        print(f"Loaded existing weights from {model_out}. Old vocab size={old_vocab_size}, new vocab size={new_vocab_size}.")
        _apply_weights_with_vocab_resize(model, loaded_weights, old_vocab_size=old_vocab_size, new_vocab_size=new_vocab_size)
        print("Weights loaded successfully (with resizing if needed).")

    # Loss
    loss_fn = CrossEntropyLoss(ignore_index=-100)

    # Optional: very simple persistent vector memory of "chunks" (long-term recall independent of corpus files).
    # We store vectors as numpy and texts as a parallel array of strings.
    mem_vectors: List[np.ndarray] = []
    mem_texts: List[str] = []
    if enable_vector_memory and os.path.isfile(memory_out):
        loaded_mem = dict(np.load(memory_out, allow_pickle=True))
        mem_vectors = [v for v in loaded_mem.get("vectors", [])]
        mem_texts = [t for t in loaded_mem.get("texts", [])]
        print(f"Loaded vector memory with {len(mem_texts)} items from {memory_out}.")

    # Train sequentially per corpus
    for corpus_idx, (path, text) in enumerate(corpora_loaded, start=1):
        encoded = _encode_with_oov_handling(vocab, text)
        print(f"\n=== Training on corpus {corpus_idx}/{len(corpora_loaded)}: {path} ===")

        # If corpus is huge, you still get random windows via make_batches.
        for step in range(1, steps + 1):
            x, y = make_batches(encoded, block_size=block_size, batch_size=batch_size)
            model.zero_grad()
            logits = model.forward(x)
            loss = loss_fn.forward(logits, y)
            grad_logits = loss_fn.backward()
            model.backward(grad_logits)
            model.step(lr=learning_rate)

            if step % 10 == 0 or step == 1 or step == steps:
                print(f"[{path}] step {step}/{steps} - loss: {loss:.4f}")

        # Optional: store a few vector memories for this corpus (small, stable).
        if enable_vector_memory:
            # Take up to 5 small snippets from the corpus and store their final-token hidden embedding.
            # This "memory" persists even if the corpus file changes/deletes later.
            snippets = []
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            for ln in lines[:5]:
                snippets.append(ln[:200])
            for snip in snippets:
                ids = vocab.encode(snip)
                if not ids:
                    continue
                ids = ids[-config.max_seq_len :]
                tokens = np.array(ids, dtype=np.int64)[None, :]
                # Compute last hidden vector (pre-logits) by mirroring model.forward up to ln_f.
                tok_emb = model.token_embedding.forward(tokens)
                T = tokens.shape[1]
                pos_ids = np.arange(T, dtype=np.int64)[None, :]
                pos_emb = model.pos_embedding.forward(pos_ids)
                h = tok_emb + pos_emb
                for block in model.blocks:
                    h = block.forward(h)
                h = model.ln_f.forward(h)
                vec = h[0, -1, :].astype(np.float32)
                mem_vectors.append(vec)
                mem_texts.append(snip)
            print(f"Vector memory size is now {len(mem_texts)} items.")

    # Save vocab, config, weights (IMPORTANT: we overwrite with updated, cumulative weights)
    with open(vocab_out, "w", encoding="utf-8") as f:
        f.write(vocab.to_json())
    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f)

    state = model.state_dict()
    np.savez(model_out, **state)

    if enable_vector_memory:
        # Store texts as object array (pickle is needed for variable-length strings).
        np.savez(memory_out, vectors=np.array(mem_vectors, dtype=np.float32), texts=np.array(mem_texts, dtype=object))
        print(f"Saved vector memory to {memory_out}")

    print(f"\nSaved vocab to {vocab_out}")
    print(f"Saved config to {config_out}")
    print(f"Saved cumulative model weights to {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incremental trainer for character-level transformer LM (NumPy only).")
    parser.add_argument(
        "--corpora",
        nargs="*",
        default=["corpus.txt"],
        help="One or more corpus text files. Trains sequentially over them.",
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable_vector_memory", action="store_true")
    args = parser.parse_args()

    train(
        corpora=args.corpora,
        steps=args.steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        enable_vector_memory=args.enable_vector_memory,
    )

