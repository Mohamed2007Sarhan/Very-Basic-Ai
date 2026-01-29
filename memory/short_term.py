from dataclasses import dataclass, field
from typing import List


@dataclass
class ShortTermMemory:
    """
    Rolling text-based context buffer.

    This is deliberately simple and transparent:
      - Stores the last N utterances as strings.
      - Provides a combined context string for conditioning the LLM.
    """

    max_utterances: int = 10
    _buffer: List[str] = field(default_factory=list)

    def add(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._buffer.append(text)
        if len(self._buffer) > self.max_utterances:
            self._buffer = self._buffer[-self.max_utterances :]

    def clear(self) -> None:
        self._buffer.clear()

    def get_context(self) -> str:
        """
        Returns the concatenated recent conversation as a single string.
        """
        return "\n".join(self._buffer)


if __name__ == "__main__":
    stm = ShortTermMemory(max_utterances=3)
    stm.add("user: hi")
    stm.add("agent: hello")
    stm.add("user: how are you?")
    stm.add("agent: I am a zero-knowledge model.")
    print("Context:")
    print(stm.get_context())

