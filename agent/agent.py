from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np

from core.tokenizer import CharVocab
from model import LanguageModel, ModelConfig
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from agent.planner import Planner
from tools.cmd_tool import CmdTool


def _text_embedding_from_model(model: LanguageModel, vocab: CharVocab, text: str) -> np.ndarray:
    """
    Build a simple fixed-size embedding for a piece of text using the LM.

    Procedure:
      - Encode characters to token ids.
      - Truncate or pad to the model's max_seq_len.
      - Run a forward pass.
      - Take the last hidden vector (before logits) by reusing embeddings and blocks.
        For simplicity, we re-run a manual forward that stops before the final projection.
    """
    encoded = vocab.encode(text)
    if not encoded:
        return np.zeros(model.config.d_model, dtype=np.float32)

    max_len = model.config.max_seq_len
    if len(encoded) > max_len:
        encoded = encoded[-max_len:]
    else:
        # left-pad with zeros (pad token index is 0 in our tokenizer)
        encoded = [0] * (max_len - len(encoded)) + encoded
    tokens = np.array(encoded, dtype=np.int64)[None, :]  # (1, T)

    # This mirrors LanguageModel.forward but stops before final projection.
    tok_emb = model.token_embedding.forward(tokens)
    T = tokens.shape[1]
    pos_ids = np.arange(T, dtype=np.int64)[None, :]
    pos_emb = model.pos_embedding.forward(pos_ids)
    x = tok_emb + pos_emb
    for block in model.blocks:
        x = block.forward(x)
    x = model.ln_f.forward(x)  # (1, T, D)
    # Use the last token's representation.
    return x[0, -1, :].astype(np.float32)


@dataclass
class Agent:
    """
    Minimal autonomous agent built around the language model.

    Loop:
      Observe:  accept user input.
      Think:    run LM on short-term context.
      Recall:   query long-term memory using a vector embedding.
      Act:      either respond with generated text or invoke tools.
      Reflect:  store the interaction in long-term memory.
    """

    model: LanguageModel
    vocab: CharVocab
    short_term: ShortTermMemory
    long_term: LongTermMemory
    planner: Planner
    cmd_tool: CmdTool

    def _build_prompt(self, user_input: str, recalls: str) -> str:
        """
        Construct a plain-text prompt for the character-level model.

        The model has zero world knowledge; we explicitly remind the user of this.
        """
        context = self.short_term.get_context()
        parts = []
        if context:
            parts.append(context)
        if recalls:
            parts.append(f"[memory]\n{recalls}")
        parts.append("agent: I am a zero-knowledge character-level model. "
                     "I only know statistics from training text, not facts about the world.")
        parts.append(f"user: {user_input}")
        parts.append("agent:")
        return "\n".join(parts)

    def _generate_text(self, prompt: str, max_new_tokens: int = 128, temperature: float = 1.0) -> str:
        """
        Simple autoregressive character-level generation using greedy/temperature sampling.
        """
        self.model.zero_grad()  # ensure no stale gradients
        context_ids = self.vocab.encode(prompt)
        for _ in range(max_new_tokens):
            if len(context_ids) > self.model.config.max_seq_len:
                # Keep only the most recent window
                context_ids = context_ids[-self.model.config.max_seq_len :]
            tokens = np.array(context_ids, dtype=np.int64)[None, :]
            logits = self.model.forward(tokens)  # (1, T, V)
            last_logits = logits[0, -1, :]  # (V,)

            # Temperature scaling
            if temperature <= 0:
                # degenerate case: pure argmax
                probs = np.zeros_like(last_logits)
                probs[np.argmax(last_logits)] = 1.0
            else:
                scaled = last_logits / float(temperature)
                exp = np.exp(scaled - np.max(scaled))
                probs = exp / exp.sum()

            next_id = int(np.random.choice(len(probs), p=probs))
            context_ids.append(next_id)
            # Stop on newline as a simple heuristic
            if self.vocab.itos.get(next_id, "") == "\n":
                break

        # Decode only the tokens generated after the original prompt.
        generated_ids = context_ids[len(self.vocab.encode(prompt)) :]
        return self.vocab.decode(generated_ids)

    def step(self, user_input: str, temperature: float = 0.9) -> Dict[str, Any]:
        """
        Run a single agent step given user input.
        """
        # Observe
        self.short_term.add(f"user: {user_input}")

        # Planner decides high-level action
        decision = self.planner.decide(user_input)

        # Build query embedding for long-term memory
        q_vec = _text_embedding_from_model(self.model, self.vocab, user_input)
        recalls = self.long_term.query(q_vec, top_k=3)
        recalls_text = "\n".join(f"{score:.3f}: {text}" for score, text in recalls)

        tool_result: Dict[str, Any] | None = None
        if decision.get("type") == "use_tool" and decision.get("tool") == "cmd":
            action = decision.get("action")
            arg = decision.get("arg")
            if action == "list_files":
                tool_result = self.cmd_tool.list_files()
            elif action == "read_file" and isinstance(arg, str):
                try:
                    tool_result = self.cmd_tool.read_file(arg)
                except Exception as e:  # sandbox/validation error
                    tool_result = {"action": "read_file", "error": str(e)}

        # Think / Act: generate response text
        prompt = self._build_prompt(user_input, recalls_text)
        model_output = self._generate_text(prompt, temperature=temperature)

        if tool_result is not None:
            # Append a textual summary of tool output to the answer
            model_output = (
                "Tool result:\n"
                + str(tool_result)
                + "\n\n"
                + "Agent (zero-knowledge) reply:\n"
                + model_output
            )

        # Reflect: store combined interaction in long-term memory
        interaction_text = f"user: {user_input}\nagent: {model_output}"
        interaction_vec = _text_embedding_from_model(self.model, self.vocab, interaction_text)
        self.long_term.add(interaction_vec, interaction_text)
        self.short_term.add(f"agent: {model_output}")

        return {
            "decision": decision,
            "tool_result": tool_result,
            "response": model_output,
        }


if __name__ == "__main__":
    # This is only a structural smoke test; real usage is via run_agent.py
    # with a trained model and non-trivial vocab.
    dummy_vocab = CharVocab.from_text("abc\n ")
    cfg = ModelConfig(vocab_size=len(dummy_vocab.stoi), d_model=16, n_layers=1, d_ff=32, max_seq_len=32)
    from model import LanguageModel  # noqa: E402  (already imported above; repeated for clarity)

    lm = LanguageModel(cfg)
    stm = ShortTermMemory(max_utterances=5)
    ltm = LongTermMemory(dim=cfg.d_model)
    planner = Planner()
    import os

    cmd = CmdTool(base_dir=os.getcwd())
    agent = Agent(model=lm, vocab=dummy_vocab, short_term=stm, long_term=ltm, planner=planner, cmd_tool=cmd)
    result = agent.step("hello")
    print("Agent step result keys:", list(result.keys()))

