from __future__ import annotations

import json
import os

import numpy as np

from core.tokenizer import CharVocab
from model import LanguageModel, ModelConfig
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from agent.planner import Planner
from agent.agent import Agent
from tools.cmd_tool import CmdTool


def load_model_and_vocab(
    base_dir: str,
    config_path: str = "model_config.json",
    weights_path: str = "model_weights.npz",
    vocab_path: str = "vocab.json",
) -> tuple[LanguageModel, CharVocab]:
    config_full = os.path.join(base_dir, config_path)
    weights_full = os.path.join(base_dir, weights_path)
    vocab_full = os.path.join(base_dir, vocab_path)

    with open(config_full, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    config = ModelConfig(**cfg_dict)
    model = LanguageModel(config)

    weights = dict(np.load(weights_full))
    model.load_state_dict(weights)

    with open(vocab_full, "r", encoding="utf-8") as f:
        vocab_json = f.read()
    vocab = CharVocab.from_json(vocab_json)
    return model, vocab


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model, vocab = load_model_and_vocab(base_dir=base_dir)

    stm = ShortTermMemory(max_utterances=10)
    ltm = LongTermMemory(dim=model.config.d_model)
    planner = Planner()
    cmd_tool = CmdTool(base_dir=base_dir)

    agent = Agent(
        model=model,
        vocab=vocab,
        short_term=stm,
        long_term=ltm,
        planner=planner,
        cmd_tool=cmd_tool,
    )

    print("Zero-knowledge character-level agent.")
    print("Type 'quit' to exit. Commands: 'tool:list_files', 'tool:read_file README.md'.")
    while True:
        try:
            user_input = input("you> ")
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            break
        result = agent.step(user_input)
        print("agent>", result["response"])


if __name__ == "__main__":
    main()

