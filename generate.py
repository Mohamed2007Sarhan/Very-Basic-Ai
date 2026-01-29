from __future__ import annotations

import json
import os
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from core.tokenizer import CharVocab
from model import LanguageModel, ModelConfig


def _safe_load_json(path: str) -> Dict[str, Any]:
    """
    Load JSON safely:
      - If missing or empty -> {}
      - If invalid -> {}
    """
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return {}
        return json.loads(raw)
    except Exception:
        return {}


class UnifiedVocabV2:
    """
    Internal unified representation that merges:
      - vocab.json (legacy char stoi)
      - vocab_v2.json (global stoi if present)
      - vocab_char/word/sentence/bpe.json (token lists)
      - bpe_merges.json (for better BPE decoding)

    IMPORTANT:
      - We do NOT rename files on disk.
      - We keep backwards compatibility: if only legacy vocab exists, we behave like old generate.py.

    Token naming convention (v2):
      - "C:<char>"     character token
      - "W:<word>"     word token
      - "S:<sentence>" sentence token (capped during training)
      - "B:<subword>"  BPE/subword token
    """

    def __init__(self) -> None:
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.unk_id: int = 1  # default fallback; updated when <unk> exists
        self.has_v2_tokens: bool = False
        self.bpe_merges: List[Tuple[str, str]] = []

    @property
    def size(self) -> int:
        return len(self.stoi)

    def _rebuild_itos(self) -> None:
        self.itos = {int(i): t for t, i in self.stoi.items()}

    def add_token_preserve_id(self, token: str, idx: int) -> None:
        self.stoi[token] = int(idx)

    def add_token_next(self, token: str) -> int:
        if token in self.stoi:
            return self.stoi[token]
        new_id = max(self.itos.keys()) + 1 if self.itos else 0
        self.stoi[token] = int(new_id)
        self.itos[int(new_id)] = token
        return int(new_id)

    def encode_prompt_as_chars(self, prompt: str) -> List[int]:
        # For v2 models we always condition on character tokens: "C:<ch>"
        ids: List[int] = []
        for ch in prompt:
            ids.append(self.stoi.get(f"C:{ch}", self.unk_id))
        return ids

    def decode_ids(self, ids: List[int]) -> str:
        """
        Stage-aware decoding:
          - C: emit raw char
          - W: emit words with spacing rules
          - S: emit sentence with sentence boundary spacing
          - B: try to reconstruct wordpieces using </w> marker if present
        """
        out: List[str] = []
        pending_bpe = ""  # build up current word for BPE tokens

        def flush_bpe() -> None:
            nonlocal pending_bpe
            if pending_bpe:
                out.append(pending_bpe)
                pending_bpe = ""

        def append_wordlike(w: str) -> None:
            # Attach punctuation to previous token if appropriate.
            if w in {".", ",", "!", "?", ":", ";", ")", "]", "}", "”", "’"}:
                # punctuation: no preceding space if we already have something
                if out:
                    out[-1] = out[-1] + w
                else:
                    out.append(w)
                return
            if w in {"(", "[", "{", "“", "‘"}:
                # opening punctuation: attach directly, but keep following token separated naturally
                out.append(w)
                return
            # normal token: ensure space separation if needed
            if not out:
                out.append(w)
            else:
                # If last is an opening bracket/quote, don't insert an extra space after it.
                if out[-1] in {"(", "[", "{", "“", "‘"}:
                    out.append(w)
                else:
                    out.append(" " + w)

        for idx in ids:
            tok = self.itos.get(int(idx))
            if tok is None:
                continue
            if tok == "<pad>":
                continue

            if tok.startswith("C:"):
                flush_bpe()
                out.append(tok[2:])
                continue

            if tok.startswith("W:"):
                flush_bpe()
                append_wordlike(tok[2:])
                continue

            if tok.startswith("S:"):
                flush_bpe()
                s = tok[2:].strip()
                if not s:
                    continue
                if out and not out[-1].endswith((" ", "\n")):
                    out.append("\n")
                out.append(s)
                out.append("\n")
                continue

            if tok.startswith("B:"):
                # BPE/subword reconstruction:
                # - if token ends with </w>, treat it as end-of-word and add a space.
                piece = tok[2:]
                if piece.endswith("</w>"):
                    piece_clean = piece[: -len("</w>")]
                    pending_bpe += piece_clean
                    flush_bpe()
                    # add a space after a completed word unless next token is punctuation
                    if out and not out[-1].endswith((" ", "\n")):
                        out.append(" ")
                else:
                    pending_bpe += piece
                continue

            # Unknown/un-namespaced token: fall back to printing it safely.
            flush_bpe()
            append_wordlike(tok)

        flush_bpe()
        text = "".join(out)
        # Clean up excessive whitespace/newlines.
        text = "\n".join([ln.rstrip() for ln in text.splitlines()]).strip()
        return text


def _build_unified_vocab_v2(
    vocab_json_path: str = "vocab.json",
    vocab_v2_path: str = "vocab_v2.json",
    vocab_char_path: str = "vocab_char.json",
    vocab_word_path: str = "vocab_word.json",
    vocab_sentence_path: str = "vocab_sentence.json",
    vocab_bpe_path: str = "vocab_bpe.json",
    bpe_merges_path: str = "bpe_merges.json",
) -> UnifiedVocabV2:
    v2 = UnifiedVocabV2()

    # 1) Prefer explicit v2 stoi if present (this aligns with train_v2.py outputs).
    data_v2 = _safe_load_json(vocab_v2_path)
    if "stoi" in data_v2 and isinstance(data_v2["stoi"], dict) and data_v2["stoi"]:
        # Use it as the base mapping.
        v2.stoi = {str(k): int(v) for k, v in data_v2["stoi"].items()}
        v2._rebuild_itos()
        v2.has_v2_tokens = True
    else:
        # 2) Otherwise, bootstrap from legacy vocab.json by mapping char ids to C:<ch> tokens.
        legacy = _safe_load_json(vocab_json_path)
        legacy_stoi = legacy.get("stoi", {})
        if isinstance(legacy_stoi, dict) and legacy_stoi:
            # preserve ids exactly
            for ch, idx in legacy_stoi.items():
                if ch == "<pad>":
                    v2.add_token_preserve_id("<pad>", int(idx))
                else:
                    v2.add_token_preserve_id(f"C:{ch}", int(idx))
            v2._rebuild_itos()
            v2.has_v2_tokens = True

    # Ensure <pad> exists at 0 if possible; ensure <unk> exists somewhere.
    if "<pad>" not in v2.stoi:
        v2.stoi["<pad>"] = 0
        v2._rebuild_itos()
    if "<unk>" not in v2.stoi:
        # Add <unk> at next id (do not force to 1 because existing weights may already use id 1 for something else).
        v2.add_token_next("<unk>")
    v2.unk_id = int(v2.stoi.get("<unk>", 1))

    # 3) Merge stage vocab token lists (if they exist). These are token strings only.
    #    We add them as next ids; this does not overwrite existing ids (preserves backward compatibility).
    def merge_token_list(path: str, prefix: str) -> int:
        data = _safe_load_json(path)
        toks = data.get("tokens", [])
        if not isinstance(toks, list):
            return 0
        added = 0
        for t in toks:
            if not isinstance(t, str) or not t:
                continue
            name = f"{prefix}{t}"
            if name in v2.stoi:
                continue
            v2.add_token_next(name)
            added += 1
        return added

    merge_token_list(vocab_char_path, "C:")
    merge_token_list(vocab_word_path, "W:")
    merge_token_list(vocab_sentence_path, "S:")
    merge_token_list(vocab_bpe_path, "B:")

    # 4) Load BPE merges (for future use / debugging; decoding uses </w> markers primarily).
    merges_data = _safe_load_json(bpe_merges_path)
    merges = merges_data.get("merges", [])
    if isinstance(merges, list):
        out_merges: List[Tuple[str, str]] = []
        for m in merges:
            if isinstance(m, list) and len(m) == 2 and all(isinstance(x, str) for x in m):
                out_merges.append((m[0], m[1]))
        v2.bpe_merges = out_merges

    return v2


def _load_model_and_any_vocab(
    config_path: str = "model_config.json",
    weights_path: str = "model_weights.npz",
    legacy_vocab_path: str = "vocab.json",
) -> tuple[LanguageModel, Optional[CharVocab], UnifiedVocabV2]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    config = ModelConfig(**cfg_dict)
    model = LanguageModel(config)

    weights = dict(np.load(weights_path))
    model.load_state_dict(weights)

    legacy_vocab: Optional[CharVocab] = None
    if os.path.isfile(legacy_vocab_path):
        try:
            with open(legacy_vocab_path, "r", encoding="utf-8") as f:
                legacy_vocab = CharVocab.from_json(f.read())
        except Exception:
            legacy_vocab = None

    v2_vocab = _build_unified_vocab_v2()
    return model, legacy_vocab, v2_vocab


def generate(
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    config_path: str = "model_config.json",
    weights_path: str = "model_weights.npz",
    vocab_path: str = "vocab.json",
) -> str:
    model, legacy_vocab, v2_vocab = _load_model_and_any_vocab(config_path, weights_path, vocab_path)
    model.zero_grad()

    # Decide whether we're using legacy (pure char) or v2 (namespaced tokens).
    # Heuristic: if config vocab_size is much larger than legacy vocab, we must be v2.
    use_v2 = True
    if legacy_vocab is not None and model.config.vocab_size == len(legacy_vocab.stoi):
        use_v2 = False

    # Sanity: weight shape should match config.vocab_size
    w_vocab = int(model.token_embedding.weight.shape[0])
    if w_vocab != int(model.config.vocab_size):
        raise ValueError(
            f"Embedding vocab size mismatch: weights have {w_vocab}, config has {model.config.vocab_size}"
        )
    if int(model.fc_out.W.shape[1]) != int(model.config.vocab_size):
        raise ValueError(
            f"Output projection mismatch: fc_out.W has out_dim={model.fc_out.W.shape[1]}, "
            f"config has vocab_size={model.config.vocab_size}"
        )

    if use_v2:
        context_ids = v2_vocab.encode_prompt_as_chars(prompt)
    else:
        if legacy_vocab is None:
            raise ValueError("Legacy vocab.json is missing or invalid, cannot run legacy generation.")
        context_ids = legacy_vocab.encode(prompt)

    prompt_len = len(context_ids)
    for _ in range(max_new_tokens):
        if len(context_ids) > model.config.max_seq_len:
            context_ids = context_ids[-model.config.max_seq_len :]
        tokens = np.array(context_ids, dtype=np.int64)[None, :]
        logits = model.forward(tokens)
        last_logits = logits[0, -1, :]

        if temperature <= 0:
            probs = np.zeros_like(last_logits)
            probs[int(np.argmax(last_logits))] = 1.0
        else:
            scaled = last_logits / float(temperature)
            exp = np.exp(scaled - np.max(scaled))
            probs = exp / exp.sum()

        next_id = int(np.random.choice(len(probs), p=probs))
        context_ids.append(next_id)
        # stopping heuristic: if we generate a newline (legacy or v2 char token), stop early
        if use_v2:
            tok = v2_vocab.itos.get(next_id, "")
            if tok == "C:\n":
                break
        else:
            ch = legacy_vocab.itos.get(next_id, "") if legacy_vocab is not None else ""
            if ch == "\n":
                break

    generated_ids = context_ids[prompt_len:]
    if use_v2:
        return v2_vocab.decode_ids(generated_ids)
    assert legacy_vocab is not None
    return legacy_vocab.decode(generated_ids)


if __name__ == "__main__":
    # Startup sanity check:
    # - Print model architecture + vocab sizes.
    # - Do a short test generation.
    try:
        model, legacy_vocab, v2_vocab = _load_model_and_any_vocab()
        print("=== Sanity check ===")
        print(f"Model: d_model={model.config.d_model}, n_layers={model.config.n_layers}, d_ff={model.config.d_ff}, max_seq_len={model.config.max_seq_len}")
        print(f"Weights vocab_size (embedding rows): {model.token_embedding.weight.shape[0]}")
        if legacy_vocab is not None:
            print(f"Legacy vocab.json size: {len(legacy_vocab.stoi)}")
        print(f"Unified v2 vocab size (internal): {v2_vocab.size}")
        print(f"BPE merges loaded: {len(v2_vocab.bpe_merges)}")
        print("Generating test text...")
        text = generate(prompt="The hello worlds", max_new_tokens=200, temperature=0.8)
        print(text if text else "(empty output)")
    except Exception as e:
        print("Startup sanity check failed:", str(e))

