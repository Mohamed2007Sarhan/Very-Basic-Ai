# AI From Scratch

From-scratch character-level transformer language model and agent using only **Python 3.10+**, **NumPy**, and the standard library. No PyTorch, TensorFlow, JAX, or agent frameworks.

---

## Setup

### Install the package

```bash
pip install -e .
```

### Reinstall weights and JSON files

If you have deleted `model_weights.npz` and the vocab/config JSON files, regenerate them with:

```bash
python setup.py install_artifacts
```

This runs `train.py` and creates:

- `model_weights.npz` — model parameters  
- `model_config.json` — model architecture config  
- `vocab.json` — character vocabulary  

Training uses `corpus.txt` in the project root (or a built-in tiny corpus if the file is missing).

For the full v2 pipeline (multi-stage tokenization and vocabs), run after `install_artifacts`:

```bash
python train_v2.py
```

That produces `vocab_v2.json`, `vocab_char.json`, `vocab_word.json`, `vocab_sentence.json`, `vocab_bpe.json`, and `bpe_merges.json`.

---

## Project layout

- **`core/`** — tokenizer, embeddings, attention, transformer block, activations, loss  
- **`memory/`** — short-term (rolling buffer) and long-term (vector store)  
- **`agent/`** — planner and agent loop  
- **`tools/`** — sandboxed command tool  
- **`model.py`** — language model  
- **`train.py`** — character-level training  
- **`train_v2.py`** — multi-stage (char → word → sentence → BPE) training  
- **`generate.py`** — text generation  
- **`run_agent.py`** — interactive agent  

---

## Quick start

1. **Train** (creates weights and JSON if missing):

   ```bash
   python train.py
   ```

2. **Generate text**:

   ```bash
   python generate.py
   ```

3. **Run the agent**:

   ```bash
   python run_agent.py
   ```

---

## Core components

- **Tokenizer** (`core/tokenizer.py`) — character-level vocab; `encode` / `decode` reversible; index 0 reserved for `<pad>`.  
- **Model** (`model.py`) — `ModelConfig` + `LanguageModel` (embeddings, transformer blocks, LayerNorm, linear head); manual `backward`, `zero_grad`, `step`; `state_dict` / `load_state_dict`.  
- **Training** (`train.py`) — loads corpus, builds vocab, runs SGD on cross-entropy; saves `model_weights.npz`, `model_config.json`, `vocab.json`.  
- **Generation** (`generate.py`) — loads config, weights, and vocab; autoregressive sampling with temperature and sliding context.  
- **Agent** (`run_agent.py`) — short-term + long-term memory, planner, optional tool calls (`tool:list_files`, `tool:read_file`).  

---

## Zero-knowledge disclaimer

The model is trained only on the provided corpus and has **no factual knowledge** beyond character-level statistics. Do not treat it as a source of facts.
