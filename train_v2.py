from __future__ import annotations

"""
train_v2.py

Multi-stage incremental training pipeline for a transformer LM (NumPy only).

Stages (in order) and default training effort weights:
  - char     : 0.2
  - word     : 0.3
  - sentence : 0.3
  - bpe      : 0.2

Key design choice (to preserve old weights across tokenization stages):
  - We use ONE shared/global vocabulary for the model.
  - Tokens from different stages are namespaced:
      "C:<char>", "W:<word>", "S:<sentence>", "B:<subword>"
  - This allows incremental vocab expansion without reinterpreting existing ids.

This file is self-contained (does NOT depend on train.py), but it does reuse the
core model implementation from model.py and core/* (still NumPy-only).
"""

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import asdict
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from model import LanguageModel, ModelConfig
from core.loss import CrossEntropyLoss


# ---------------------------
# Utilities: persistence
# ---------------------------


def read_text(path: str, max_chars: Optional[int] = None) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        if max_chars is None:
            return f.read()
        return f.read(max_chars)


def ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def load_json_if_exists(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------
# Global vocabulary
# ---------------------------


class GlobalVocab:
    """
    Single shared vocab for all training stages.

    Reserved ids:
      0: <pad>

    Note on backward compatibility:
      - Earlier project phases used a CharVocab with only <pad> at id 0.
      - To support incremental training without breaking existing weights,
        this vocab does NOT force <unk> to be id 1.
      - If <unk> is missing, we add it at the next available id.
    """

    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, stoi: Optional[Dict[str, int]] = None) -> None:
        if stoi is None:
            self.stoi = {self.PAD: 0}
        else:
            self.stoi = stoi
            # enforce reserved
            self.stoi[self.PAD] = 0
            # do NOT force UNK to id 1 (keeps legacy ids stable)
        self.itos = {i: t for t, i in self.stoi.items()}
        self.itos[0] = self.PAD
        # ensure UNK exists somewhere
        if self.UNK not in self.stoi:
            new_id = max(self.itos.keys()) + 1 if self.itos else 1
            self.stoi[self.UNK] = new_id
            self.itos[new_id] = self.UNK

        self.unk_id = int(self.stoi[self.UNK])

    @property
    def size(self) -> int:
        return len(self.stoi)

    def add_token(self, token: str) -> int:
        if token in self.stoi:
            return self.stoi[token]
        new_id = max(self.itos.keys()) + 1
        self.stoi[token] = new_id
        self.itos[new_id] = token
        return new_id

    def encode_tokens(self, tokens: Sequence[str], add_new: bool = True) -> List[int]:
        ids: List[int] = []
        for t in tokens:
            if add_new:
                ids.append(self.add_token(t))
            else:
                ids.append(self.stoi.get(t, self.unk_id))
        return ids

    def to_json(self) -> str:
        return json.dumps({"stoi": self.stoi}, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "GlobalVocab":
        data = json.loads(s)
        return cls(stoi=data["stoi"])


def load_vocab(path: str) -> Optional[GlobalVocab]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return GlobalVocab.from_json(f.read())


def save_vocab(path: str, vocab: GlobalVocab) -> None:
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(vocab.to_json())


# ---------------------------
# Weight loading + resizing
# ---------------------------


def load_weights_npz(path: str) -> Optional[Dict[str, np.ndarray]]:
    if not os.path.isfile(path):
        return None
    return {k: v for k, v in dict(np.load(path)).items()}


def expand_rows(old: np.ndarray, new_rows: int, init_scale: float) -> np.ndarray:
    assert old.ndim == 2
    old_rows, cols = old.shape
    assert new_rows >= old_rows
    if new_rows == old_rows:
        return old
    extra = np.random.uniform(-init_scale, init_scale, size=(new_rows - old_rows, cols)).astype(old.dtype)
    return np.vstack([old, extra])


def expand_cols(old: np.ndarray, new_cols: int, init_scale: float) -> np.ndarray:
    assert old.ndim == 2
    rows, old_cols = old.shape
    assert new_cols >= old_cols
    if new_cols == old_cols:
        return old
    extra = np.random.uniform(-init_scale, init_scale, size=(rows, new_cols - old_cols)).astype(old.dtype)
    return np.hstack([old, extra])


def expand_vector(old: np.ndarray, new_size: int) -> np.ndarray:
    assert old.ndim == 1
    old_size = old.shape[0]
    assert new_size >= old_size
    if new_size == old_size:
        return old
    out = np.zeros((new_size,), dtype=old.dtype)
    out[:old_size] = old
    return out


def apply_weights_with_resize(model: LanguageModel, state: Dict[str, np.ndarray], new_vocab_size: int) -> None:
    """
    Load weights into `model`. If vocab grew, expand:
      - token_embedding.weight (V, D) rows
      - fc_out.W (D, V) cols
      - fc_out.b (V,) length
    If max_seq_len grew, expand pos_embedding rows.
    """
    s = state.copy()

    if "token_embedding.weight" not in s:
        raise ValueError("Weights missing token_embedding.weight; wrong/corrupt .npz?")

    old_vocab_size = int(s["token_embedding.weight"].shape[0])
    if new_vocab_size < old_vocab_size:
        raise ValueError(
            "New vocab_size is smaller than old vocab_size; cannot load existing weights into smaller vocab."
        )

    if new_vocab_size != old_vocab_size:
        init_scale = 1.0 / np.sqrt(model.config.d_model)
        s["token_embedding.weight"] = expand_rows(s["token_embedding.weight"], new_vocab_size, init_scale=init_scale)

        init_scale_out = np.sqrt(6.0 / (model.config.d_model + new_vocab_size))
        s["fc_out.W"] = expand_cols(s["fc_out.W"], new_vocab_size, init_scale=init_scale_out)
        s["fc_out.b"] = expand_vector(s["fc_out.b"], new_vocab_size)

    # pos embeddings if needed
    if "pos_embedding.weight" in s:
        old_max = int(s["pos_embedding.weight"].shape[0])
        new_max = int(model.config.max_seq_len)
        if new_max > old_max:
            init_scale = 1.0 / np.sqrt(model.config.d_model)
            s["pos_embedding.weight"] = expand_rows(s["pos_embedding.weight"], new_max, init_scale=init_scale)

    model.load_state_dict(s)


# ---------------------------
# Tokenization stages
# ---------------------------


def iter_chars(text: str) -> Iterator[str]:
    for ch in text:
        yield ch


_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\s]")


def iter_words(text: str) -> Iterator[str]:
    # Keeps punctuation as its own tokens.
    for m in _WORD_RE.finditer(text):
        yield m.group(0)


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def iter_sentences(text: str) -> Iterator[str]:
    # Very simple sentence splitter: punctuation boundary or newlines.
    parts = _SENT_SPLIT_RE.split(text)
    for p in parts:
        s = p.strip()
        if s:
            yield s


class BPETokenizer:
    """
    Minimal BPE trainer/tokenizer (classic word-level BPE).

    - Trains merges on a SAMPLE of text (for robustness on huge corpora).
    - Tokenizes by applying merges to each word, producing subword tokens.

    This is intentionally simple and readable, not fast.
    """

    def __init__(self, merges: Optional[List[Tuple[str, str]]] = None) -> None:
        self.merges: List[Tuple[str, str]] = merges or []
        # mapping pair->rank for fast application
        self.ranks: Dict[Tuple[str, str], int] = {pair: i for i, pair in enumerate(self.merges)}

    def to_json(self) -> dict:
        return {"merges": self.merges}

    @classmethod
    def from_json(cls, data: dict) -> "BPETokenizer":
        merges = [tuple(x) for x in data.get("merges", [])]
        return cls(merges=merges)

    @staticmethod
    def _word_to_symbols(word: str) -> List[str]:
        # Add end-of-word marker to keep boundaries.
        return list(word) + ["</w>"]

    @staticmethod
    def _get_pair_counts(vocab: Dict[Tuple[str, ...], int]) -> Counter:
        counts: Counter = Counter()
        for symbols, freq in vocab.items():
            for i in range(len(symbols) - 1):
                counts[(symbols[i], symbols[i + 1])] += freq
        return counts

    @staticmethod
    def _merge_pair_in_symbols(symbols: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
        a, b = pair
        out: List[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return tuple(out)

    def train(self, text: str, num_merges: int = 200, max_words: int = 20000) -> None:
        """
        Train additional merges on the given text sample.
        """
        # Build a word frequency vocab from sample.
        words = []
        for w in iter_words(text):
            if w.isspace():
                continue
            words.append(w)
            if len(words) >= max_words:
                break
        wf = Counter(words)
        bpe_vocab: Dict[Tuple[str, ...], int] = {}
        for w, f in wf.items():
            bpe_vocab[tuple(self._word_to_symbols(w))] = int(f)

        merges_added = 0
        for _ in range(num_merges):
            pair_counts = self._get_pair_counts(bpe_vocab)
            if not pair_counts:
                break
            (a, b), _freq = pair_counts.most_common(1)[0]
            pair = (a, b)
            if pair in self.ranks:
                # already known
                continue
            # apply merge to all words
            new_vocab: Dict[Tuple[str, ...], int] = {}
            for sym, f in bpe_vocab.items():
                new_vocab[self._merge_pair_in_symbols(sym, pair)] = f
            bpe_vocab = new_vocab
            self.merges.append(pair)
            self.ranks[pair] = len(self.merges) - 1
            merges_added += 1
        return

    def encode_word(self, word: str) -> List[str]:
        symbols = tuple(self._word_to_symbols(word))
        # Apply merges greedily by rank until no merges apply.
        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            ranked = [(self.ranks[p], p) for p in pairs if p in self.ranks]
            if not ranked:
                break
            _, best = min(ranked, key=lambda x: x[0])
            symbols = self._merge_pair_in_symbols(symbols, best)
        # drop end-of-word marker token; keep it as boundary implied by word splitting
        out = [s for s in symbols if s != "</w>"]
        return out

    def iter_bpe_tokens(self, text: str) -> Iterator[str]:
        for w in iter_words(text):
            # Keep punctuation as whole tokens (don’t BPE-merge punctuation)
            if len(w) == 1 and not w.isalnum() and w != "'":
                yield w
                continue
            for sub in self.encode_word(w):
                yield sub


# ---------------------------
# Training core: stream buffer sampler
# ---------------------------


class TokenBufferSampler:
    """
    For huge corpora: keep a rolling buffer of token ids and sample random windows.

    - Feeds on an iterator of ids (streamed).
    - Keeps `max_buffer_tokens` ids.
    - Provides random (x, y) batches for next-token prediction.
    """

    def __init__(self, block_size: int, batch_size: int, max_buffer_tokens: int = 500_000) -> None:
        self.block_size = block_size
        self.batch_size = batch_size
        self.max_buffer_tokens = max_buffer_tokens
        self.buffer: List[int] = []

    def feed(self, ids: Sequence[int]) -> None:
        self.buffer.extend(ids)
        if len(self.buffer) > self.max_buffer_tokens:
            # keep most recent tokens
            self.buffer = self.buffer[-self.max_buffer_tokens :]

    def can_sample(self) -> bool:
        return len(self.buffer) >= (self.block_size + 2)

    def sample_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns x,y as (B,T) each.
        """
        if not self.can_sample():
            raise RuntimeError("Not enough tokens buffered to sample a batch.")
        buf = self.buffer
        max_start = len(buf) - self.block_size - 1
        starts = np.random.randint(0, max_start, size=(self.batch_size,))
        x = np.stack([np.array(buf[s : s + self.block_size], dtype=np.int64) for s in starts], axis=0)
        y = np.stack([np.array(buf[s + 1 : s + 1 + self.block_size], dtype=np.int64) for s in starts], axis=0)
        return x, y


def train_on_stream(
    model: LanguageModel,
    loss_fn: CrossEntropyLoss,
    id_stream: Iterator[int],
    steps: int,
    block_size: int,
    batch_size: int,
    learning_rate: float,
    label: str,
    max_buffer_tokens: int = 500_000,
) -> None:
    sampler = TokenBufferSampler(block_size=block_size, batch_size=batch_size, max_buffer_tokens=max_buffer_tokens)

    # Fill buffer initially
    chunk: List[int] = []
    for _ in range(50_000):  # up to 50k tokens warmup
        try:
            chunk.append(next(id_stream))
        except StopIteration:
            break
    sampler.feed(chunk)

    if not sampler.can_sample():
        raise ValueError(f"Not enough tokens to train for stage {label}.")

    for step in range(1, steps + 1):
        # Keep feeding a bit each iteration to keep buffer fresh
        feed_chunk: List[int] = []
        for _ in range(2048):
            try:
                feed_chunk.append(next(id_stream))
            except StopIteration:
                break
        if feed_chunk:
            sampler.feed(feed_chunk)

        x, y = sampler.sample_batch()
        model.zero_grad()
        logits = model.forward(x)
        loss = loss_fn.forward(logits, y)
        grad_logits = loss_fn.backward()
        model.backward(grad_logits)
        model.step(lr=learning_rate)

        if step == 1 or step == steps or step % 10 == 0:
            print(f"{label} step {step}/{steps} - loss: {loss:.4f}")


# ---------------------------
# Stage implementations (separate functions)
# ---------------------------


def stage_char(
    model: LanguageModel,
    vocab: GlobalVocab,
    file_path: str,
    steps: int,
    block_size: int,
    batch_size: int,
    learning_rate: float,
    max_chars_for_vocab: int = 2_000_000,
    read_chunk_chars: int = 1_000_000,
) -> Tuple[LanguageModel, List[str]]:
    """
    Character-level stage.
    Returns the list of newly added token strings (for vocab export).
    """
    text_sample = read_text(file_path, max_chars=max_chars_for_vocab)
    before = vocab.size
    # expand vocab from sample (good enough; full streaming expansion is supported via add_token too)
    for ch in set(text_sample):
        vocab.add_token(f"C:{ch}")
    added = vocab.size - before
    print(f"[char] vocab +{added} tokens (size={vocab.size})")
    # If vocab grew, resize model BEFORE training to avoid OOB embedding indices.
    if model.config.vocab_size != vocab.size:
        old_state = model.state_dict()
        cfg = model.config
        cfg.vocab_size = vocab.size
        model_new = LanguageModel(cfg)
        apply_weights_with_resize(model_new, old_state, new_vocab_size=vocab.size)
        model = model_new
        print(f"[char] resized model vocab -> {vocab.size}")

    def id_stream() -> Iterator[int]:
        # Stream file in chunks to avoid loading huge files.
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                chunk = f.read(read_chunk_chars)
                if not chunk:
                    break
                for ch in iter_chars(chunk):
                    yield vocab.stoi.get(f"C:{ch}", vocab.unk_id)

    loss_fn = CrossEntropyLoss(ignore_index=-100)
    train_on_stream(
        model,
        loss_fn,
        id_stream(),
        steps=steps,
        block_size=block_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        label=f"[char:{os.path.basename(file_path)}]",
    )
    return model, [t for t, i in vocab.stoi.items() if i >= before and t.startswith("C:")]


def stage_word(
    model: LanguageModel,
    vocab: GlobalVocab,
    file_path: str,
    steps: int,
    block_size: int,
    batch_size: int,
    learning_rate: float,
    max_words_for_vocab: int = 200_000,
    read_chunk_chars: int = 1_000_000,
) -> Tuple[LanguageModel, List[str]]:
    """
    Word-level stage.
    """
    text_sample = read_text(file_path, max_chars=2_000_000)
    words = []
    for w in iter_words(text_sample):
        words.append(w)
        if len(words) >= max_words_for_vocab:
            break
    before = vocab.size
    for w in set(words):
        vocab.add_token(f"W:{w}")
    added = vocab.size - before
    print(f"[word] vocab +{added} tokens (size={vocab.size})")
    if model.config.vocab_size != vocab.size:
        old_state = model.state_dict()
        cfg = model.config
        cfg.vocab_size = vocab.size
        model_new = LanguageModel(cfg)
        apply_weights_with_resize(model_new, old_state, new_vocab_size=vocab.size)
        model = model_new
        print(f"[word] resized model vocab -> {vocab.size}")

    def id_stream() -> Iterator[int]:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                chunk = f.read(read_chunk_chars)
                if not chunk:
                    break
                for w in iter_words(chunk):
                    yield vocab.stoi.get(f"W:{w}", vocab.unk_id)

    loss_fn = CrossEntropyLoss(ignore_index=-100)
    train_on_stream(
        model,
        loss_fn,
        id_stream(),
        steps=steps,
        block_size=block_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        label=f"[word:{os.path.basename(file_path)}]",
    )
    return model, [t for t, i in vocab.stoi.items() if i >= before and t.startswith("W:")]


def stage_sentence(
    model: LanguageModel,
    vocab: GlobalVocab,
    file_path: str,
    steps: int,
    block_size: int,
    batch_size: int,
    learning_rate: float,
    max_sentences_for_vocab: int = 50_000,
    max_sentence_chars: int = 240,
    read_chunk_chars: int = 2_000_000,
) -> Tuple[LanguageModel, List[str]]:
    """
    Sentence-level stage (sentence tokens).

    NOTE: Sentence vocab can explode on huge corpora.
    We cap collection from sample for practicality; unknown sentences map to <unk>.
    """
    text_sample = read_text(file_path, max_chars=3_000_000)
    sents = []
    for s in iter_sentences(text_sample):
        s = s[:max_sentence_chars]
        sents.append(s)
        if len(sents) >= max_sentences_for_vocab:
            break

    before = vocab.size
    for s in set(sents):
        vocab.add_token(f"S:{s}")
    added = vocab.size - before
    print(f"[sentence] vocab +{added} tokens (size={vocab.size})")
    if model.config.vocab_size != vocab.size:
        old_state = model.state_dict()
        cfg = model.config
        cfg.vocab_size = vocab.size
        model_new = LanguageModel(cfg)
        apply_weights_with_resize(model_new, old_state, new_vocab_size=vocab.size)
        model = model_new
        print(f"[sentence] resized model vocab -> {vocab.size}")

    def id_stream() -> Iterator[int]:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                chunk = f.read(read_chunk_chars)
                if not chunk:
                    break
                for s in iter_sentences(chunk):
                    s = s[:max_sentence_chars]
                    yield vocab.stoi.get(f"S:{s}", vocab.unk_id)

    loss_fn = CrossEntropyLoss(ignore_index=-100)
    train_on_stream(
        model,
        loss_fn,
        id_stream(),
        steps=steps,
        block_size=block_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        label=f"[sent:{os.path.basename(file_path)}]",
    )
    return model, [t for t, i in vocab.stoi.items() if i >= before and t.startswith("S:")]


def stage_bpe(
    model: LanguageModel,
    vocab: GlobalVocab,
    bpe: BPETokenizer,
    file_path: str,
    steps: int,
    block_size: int,
    batch_size: int,
    learning_rate: float,
    bpe_merges_add: int = 200,
    bpe_sample_chars: int = 2_000_000,
    read_chunk_chars: int = 1_000_000,
) -> Tuple[LanguageModel, List[str]]:
    """
    Subword/BPE stage.
    """
    sample = read_text(file_path, max_chars=bpe_sample_chars)
    before_merges = len(bpe.merges)
    bpe.train(sample, num_merges=bpe_merges_add, max_words=50_000)
    print(f"[bpe] merges +{len(bpe.merges) - before_merges} (total={len(bpe.merges)})")

    # Expand vocab with BPE tokens observed in sample (namespaced)
    before_vocab = vocab.size
    bpe_tokens = []
    for t in bpe.iter_bpe_tokens(sample):
        bpe_tokens.append(t)
        if len(bpe_tokens) >= 200_000:
            break
    for t in set(bpe_tokens):
        vocab.add_token(f"B:{t}")
    added = vocab.size - before_vocab
    print(f"[bpe] vocab +{added} tokens (size={vocab.size})")
    if model.config.vocab_size != vocab.size:
        old_state = model.state_dict()
        cfg = model.config
        cfg.vocab_size = vocab.size
        model_new = LanguageModel(cfg)
        apply_weights_with_resize(model_new, old_state, new_vocab_size=vocab.size)
        model = model_new
        print(f"[bpe] resized model vocab -> {vocab.size}")

    def id_stream() -> Iterator[int]:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                chunk = f.read(read_chunk_chars)
                if not chunk:
                    break
                for t in bpe.iter_bpe_tokens(chunk):
                    yield vocab.stoi.get(f"B:{t}", vocab.unk_id)

    loss_fn = CrossEntropyLoss(ignore_index=-100)
    train_on_stream(
        model,
        loss_fn,
        id_stream(),
        steps=steps,
        block_size=block_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        label=f"[bpe:{os.path.basename(file_path)}]",
    )
    return model, [t for t, i in vocab.stoi.items() if i >= before_vocab and t.startswith("B:")]


# ---------------------------
# Step distribution
# ---------------------------


def distribute_steps(total_steps: int, weights: Dict[str, float]) -> Dict[str, int]:
    # deterministic rounding with remainder distribution
    stages = list(weights.keys())
    raw = {k: total_steps * float(weights[k]) for k in stages}
    base = {k: int(raw[k]) for k in stages}
    used = sum(base.values())
    rem = total_steps - used
    # distribute remaining steps to stages with largest fractional parts
    fracs = sorted(((raw[k] - base[k], k) for k in stages), reverse=True)
    i = 0
    while rem > 0 and i < len(fracs):
        _, k = fracs[i]
        base[k] += 1
        rem -= 1
        i += 1
        if i >= len(fracs):
            i = 0
    # ensure at least 1 step for non-zero weights if total_steps allows
    for k, w in weights.items():
        if w > 0 and total_steps > 0 and base[k] == 0:
            base[k] = 1
    return base


def split_steps_across_files(stage_steps: int, n_files: int) -> List[int]:
    # Even split with remainder to early files.
    base = stage_steps // n_files
    rem = stage_steps % n_files
    out = [base + (1 if i < rem else 0) for i in range(n_files)]
    return out


# ---------------------------
# Main
# ---------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-stage incremental trainer (char/word/sent/bpe) for NumPy LM.")
    parser.add_argument("--corpora", nargs="+", required=True, help="One or more text files (processed sequentially).")
    parser.add_argument("--steps", type=int, default=2000, help="Total steps across all stages and files.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--weights_out", type=str, default="model_weights.npz")
    parser.add_argument("--config_out", type=str, default="model_config.json")
    parser.add_argument("--vocab_out", type=str, default="vocab_v2.json")
    parser.add_argument("--vocab_char_out", type=str, default="vocab_char.json")
    parser.add_argument("--vocab_word_out", type=str, default="vocab_word.json")
    parser.add_argument("--vocab_sentence_out", type=str, default="vocab_sentence.json")
    parser.add_argument("--vocab_bpe_out", type=str, default="vocab_bpe.json")
    parser.add_argument("--bpe_out", type=str, default="bpe_merges.json")

    args = parser.parse_args()

    np.random.seed(args.seed)

    # Stage weights
    stage_weights = {"char": 0.2, "word": 0.3, "sentence": 0.3, "bpe": 0.2}
    steps_by_stage = distribute_steps(args.steps, stage_weights)
    print("Total steps:", args.steps)
    print("Steps by stage:", steps_by_stage)

    # Load or initialize global vocab + bpe merges.
    # If vocab_v2.json is missing but legacy weights exist, bootstrap from legacy vocab.json
    # to keep token ids aligned with existing weights.
    vocab = load_vocab(args.vocab_out)
    if vocab is None and os.path.isfile(args.weights_out) and os.path.isfile("vocab.json"):
        # Legacy bootstrap: map old char ids to "C:<char>" tokens using the legacy ids.
        legacy = load_json_if_exists("vocab.json")
        if not legacy or "stoi" not in legacy:
            vocab = GlobalVocab()
            print(f"Initialized new global vocab size={vocab.size}")
        else:
            legacy_stoi: Dict[str, int] = legacy["stoi"]
            # build itos for legacy ids
            legacy_itos = {int(i): ch for ch, i in legacy_stoi.items()}
            # Create a global stoi preserving ids:
            stoi: Dict[str, int] = {GlobalVocab.PAD: 0}
            # Fill ids 1..max with C: tokens
            max_id = max(legacy_itos.keys())
            for idx in range(1, max_id + 1):
                ch = legacy_itos.get(idx)
                if ch is None:
                    continue
                if ch == GlobalVocab.PAD:
                    continue
                stoi[f"C:{ch}"] = idx
            vocab = GlobalVocab(stoi=stoi)
            print(f"Bootstrapped global vocab from legacy vocab.json (size={vocab.size}).")
    if vocab is None:
        vocab = GlobalVocab()
        print(f"Initialized new global vocab size={vocab.size}")
    else:
        print(f"Loaded global vocab size={vocab.size} from {args.vocab_out}")

    bpe_data = load_json_if_exists(args.bpe_out)
    bpe = BPETokenizer.from_json(bpe_data) if bpe_data else BPETokenizer()
    if bpe_data:
        print(f"Loaded BPE merges: {len(bpe.merges)}")

    # Load or initialize config (keep architecture stable)
    cfg_data = load_json_if_exists(args.config_out)
    if cfg_data:
        config = ModelConfig(**cfg_data)
        # Expand max_seq_len if requested block_size is larger
        if args.block_size > config.max_seq_len:
            config.max_seq_len = args.block_size
        # Update vocab_size to current vocab
        config.vocab_size = vocab.size
        print("Loaded config from", args.config_out)
    else:
        config = ModelConfig(vocab_size=vocab.size, d_model=64, n_layers=2, d_ff=128, max_seq_len=args.block_size)
        print("Initialized new config")

    # Initialize model
    model = LanguageModel(config)

    # If weights exist, load and resize to current vocab_size
    loaded = load_weights_npz(args.weights_out)
    if loaded is not None:
        print("Loading existing weights from", args.weights_out)
        apply_weights_with_resize(model, loaded, new_vocab_size=vocab.size)
        print("Weights loaded.")
    else:
        print("No existing weights found; starting from scratch.")

    # Process files sequentially, applying each stage in order.
    files = args.corpora
    n_files = len(files)

    # Track stage tokens for exporting separate vocab JSONs
    char_tokens: List[str] = []
    word_tokens: List[str] = []
    sent_tokens: List[str] = []
    bpe_tokens: List[str] = []

    # Pre-resize model vocab to include anything already in vocab file
    # (already done by config.vocab_size assignment above)

    # Helper: if vocab grew, rebuild model with new vocab_size and load old weights with expansion.
    def sync_model_to_vocab() -> None:
        nonlocal model, config
        if config.vocab_size == vocab.size:
            return
        old_state = model.state_dict()
        old_vocab_size = config.vocab_size
        config.vocab_size = vocab.size
        model_new = LanguageModel(config)
        apply_weights_with_resize(model_new, old_state, new_vocab_size=vocab.size)
        model = model_new
        print(f"Resized model vocab {old_vocab_size} -> {vocab.size}")

    # Char stage per file
    per_file_steps = split_steps_across_files(steps_by_stage["char"], n_files)
    for fp, st in zip(files, per_file_steps):
        if st <= 0:
            continue
        model, new_tokens = stage_char(model, vocab, fp, st, args.block_size, args.batch_size, args.learning_rate)
        char_tokens.extend(new_tokens)

    # Word stage per file
    per_file_steps = split_steps_across_files(steps_by_stage["word"], n_files)
    for fp, st in zip(files, per_file_steps):
        if st <= 0:
            continue
        model, new_tokens = stage_word(model, vocab, fp, st, args.block_size, args.batch_size, args.learning_rate)
        word_tokens.extend(new_tokens)

    # Sentence stage per file
    per_file_steps = split_steps_across_files(steps_by_stage["sentence"], n_files)
    for fp, st in zip(files, per_file_steps):
        if st <= 0:
            continue
        model, new_tokens = stage_sentence(model, vocab, fp, st, args.block_size, args.batch_size, args.learning_rate)
        sent_tokens.extend(new_tokens)

    # BPE stage per file
    per_file_steps = split_steps_across_files(steps_by_stage["bpe"], n_files)
    for fp, st in zip(files, per_file_steps):
        if st <= 0:
            continue
        model, new_tokens = stage_bpe(model, vocab, bpe, fp, st, args.block_size, args.batch_size, args.learning_rate)
        bpe_tokens.extend(new_tokens)

    # Save final artifacts
    save_vocab(args.vocab_out, vocab)
    with open(args.config_out, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f)
    np.savez(args.weights_out, **model.state_dict())
    with open(args.bpe_out, "w", encoding="utf-8") as f:
        json.dump(bpe.to_json(), f)

    # Save stage-specific vocab lists (human-readable)
    def save_stage_vocab(path: str, prefix: str) -> None:
        toks = [t[2:] for t in vocab.stoi.keys() if t.startswith(prefix)]
        ensure_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"tokens": toks}, f, ensure_ascii=False)

    save_stage_vocab(args.vocab_char_out, "C:")
    save_stage_vocab(args.vocab_word_out, "W:")
    save_stage_vocab(args.vocab_sentence_out, "S:")
    save_stage_vocab(args.vocab_bpe_out, "B:")

    print("\nSaved:")
    print(" - weights:", args.weights_out)
    print(" - config :", args.config_out)
    print(" - vocab  :", args.vocab_out)
    print(" - char vocab:", args.vocab_char_out)
    print(" - word vocab:", args.vocab_word_out)
    print(" - sentence vocab:", args.vocab_sentence_out)
    print(" - bpe vocab:", args.vocab_bpe_out)
    print(" - bpe merges:", args.bpe_out)


if __name__ == "__main__":
    main()

