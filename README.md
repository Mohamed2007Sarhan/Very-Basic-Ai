# 🧠 AI From Scratch

> *No frameworks. No shortcuts. Just NumPy and a lot of matrix multiplication.*

Build a **transformer language model** and a **conversational agent** from the ground up—using only **Python 3.10+**, **NumPy**, and the standard library. No PyTorch. No TensorFlow. No JAX. No LangChain. Just you, backprop, and a corpus.

---

## ⚡ Get Going

### 1. Install

```bash
pip install -e .
```

### 2. Bring Back the Weights & JSON

Deleted `model_weights.npz` or the vocab/config JSONs? No sweat. Regenerate everything with one command:

```bash
python setup.py install_artifacts
```

This fires up `train.py` and writes:

| File | What it is |
|------|------------|
| `model_weights.npz` | The model’s learned parameters |
| `model_config.json` | Architecture (d_model, layers, etc.) |
| `vocab.json` | Character vocabulary |

Training reads from `corpus.txt` in the project root—or uses a tiny built-in corpus if the file’s missing.

**Want the full v2 pipeline?** (char → word → sentence → BPE) After `install_artifacts`, run:

```bash
python train_v2.py
```

You’ll get `vocab_v2.json`, stage vocabs (`vocab_char`, `vocab_word`, `vocab_sentence`, `vocab_bpe`), and `bpe_merges.json`.

---

## 📁 What’s in the Box

```
core/          → tokenizer, embeddings, attention, transformer blocks, loss
memory/        → short-term buffer + long-term vector store
agent/         → planner + agent loop
tools/         → sandboxed command tool (list_files, read_file)
model.py       → the language model
train.py       → character-level training
train_v2.py    → multi-stage (char → word → sentence → BPE) training
generate.py    → text generation
run_agent.py   → interactive agent
```

---

## 🚀 Three Commands to Rule Them All

| Step | Command | What happens |
|------|---------|----------------|
| **Train** | `python train.py` | Trains on corpus, writes weights + config + vocab (or continues from existing). |
| **Generate** | `python generate.py` | Loads the model and samples text. |
| **Chat** | `python run_agent.py` | Starts the agent: memory, tools, and generated replies. |

---

## 🔧 Under the Hood

- **Tokenizer** — Character-level vocab; reversible encode/decode; `<pad>` at id 0.
- **Model** — Embeddings → transformer blocks → LayerNorm → linear head. Manual `backward`, `zero_grad`, `step`, plus `state_dict` / `load_state_dict`.
- **Training** — Corpus → vocab → SGD on cross-entropy → save weights, config, vocab.
- **Generation** — Autoregressive sampling with temperature and a sliding context window.
- **Agent** — Short-term + long-term memory, a planner, and optional tools: `tool:list_files`, `tool:read_file`.

---

## ⚠️ The Fine Print

This model is **zero-knowledge**. It learns only from the corpus you give it—character-level statistics, not facts about the world. Treat any “knowledge” it seems to have as statistical pattern, not truth.

---

*Built from scratch. Run from the command line. Understand every line.*
