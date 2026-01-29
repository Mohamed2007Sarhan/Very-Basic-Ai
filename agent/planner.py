from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Dict, Any


ActionType = Literal["respond", "use_tool"]


@dataclass
class Planner:
    """
    Very simple decision logic for the agent.

    This planner does NOT use the model weights; it is explicit and rule-based.
    Policy:
      - If the user input starts with "tool:" or "cmd:", attempt a tool call.
      - Otherwise, respond with plain language using the LLM.
    """

    def decide(self, user_input: str) -> Dict[str, Any]:
        user_input = user_input.strip()
        lowered = user_input.lower()

        if lowered.startswith("tool:") or lowered.startswith("cmd:"):
            # Extract the remainder as tool command string.
            payload = user_input.split(":", 1)[1].strip()
            # Supported tool "commands" are high-level verbs (see CmdTool).
            if payload.startswith("list_files"):
                return {"type": "use_tool", "tool": "cmd", "action": "list_files", "arg": None}
            if payload.startswith("read_file"):
                # expected format: read_file path/to/file.txt
                parts = payload.split(maxsplit=1)
                rel_path = parts[1].strip() if len(parts) > 1 else ""
                return {
                    "type": "use_tool",
                    "tool": "cmd",
                    "action": "read_file",
                    "arg": rel_path,
                }
            # Unsupported or malformed -> fall back to language response.
            return {"type": "respond"}

        return {"type": "respond"}


if __name__ == "__main__":
    planner = Planner()
    print(planner.decide("hello"))
    print(planner.decide("tool:list_files"))
    print(planner.decide("cmd:read_file README.md"))

