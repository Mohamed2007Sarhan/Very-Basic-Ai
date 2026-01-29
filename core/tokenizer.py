import json
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class CharVocab:
    """
    Simple character-level vocabulary.

    Builds a reversible mapping between characters and integer ids.
    Index 0 is reserved for padding; all real characters start from 1.
    """

    stoi: Dict[str, int]
    itos: Dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharVocab":
        # Collect sorted unique characters for deterministic ids
        chars = sorted(set(text))
        stoi = {"<pad>": 0}
        itos = {0: "<pad>"}
        for i, ch in enumerate(chars, start=1):
            stoi[ch] = i
            itos[i] = ch
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str) -> List[int]:
        """Convert string to list of token ids."""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: List[int], skip_pad: bool = True) -> str:
        """Convert list of token ids back to string."""
        chars: List[str] = []
        for idx in ids:
            if skip_pad and idx == 0:
                continue
            ch = self.itos.get(idx, "")
            # ignore unknown ids silently (should not happen if trained properly)
            if ch and ch != "<pad>":
                chars.append(ch)
        return "".join(chars)

    def to_json(self) -> str:
        return json.dumps({"stoi": self.stoi}, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "CharVocab":
        data = json.loads(s)
        stoi = data["stoi"]
        itos = {int(i): ch for ch, i in ((ch, int(idx)) for ch, idx in stoi.items())}
        # Ensure padding token mapping is present and correct
        stoi["<pad>"] = 0
        itos[0] = "<pad>"
        return cls(stoi=stoi, itos=itos)


def build_vocab_from_file(path: str) -> CharVocab:
    """Utility to build a vocabulary from a text file."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return CharVocab.from_text(text)


def encode_with_vocab(vocab: CharVocab, text: str) -> List[int]:
    return vocab.encode(text)


def decode_with_vocab(vocab: CharVocab, ids: List[int]) -> str:
    return vocab.decode(ids)


if __name__ == "__main__":
    # Basic sanity tests for tokenizer reversibility.
    sample_text = "hello world!"
    vocab = CharVocab.from_text(sample_text)
    encoded = vocab.encode(sample_text)
    decoded = vocab.decode(encoded)
    print("Sample text:", sample_text)
    print("Encoded:", encoded)
    print("Decoded:", decoded)
    assert decoded == sample_text

    # Test persistence round-trip
    json_str = vocab.to_json()
    vocab2 = CharVocab.from_json(json_str)
    encoded2 = vocab2.encode(sample_text)
    decoded2 = vocab2.decode(encoded2)
    assert decoded2 == sample_text
    print("Tokenizer tests passed.")

