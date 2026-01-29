from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class CmdTool:
    """
    Sandboxed command execution tool.

    Security model:
      - No raw shell from user input.
      - Only a small set of high-level actions are supported.
      - All paths are validated to stay inside a base directory.

    Supported actions:
      - list_files: list files in the current project directory.
      - read_file:  read a small text file relative to the project root.
    """

    base_dir: str
    max_output_bytes: int = 8192

    def _safe_path(self, rel_path: str) -> str:
        rel_path = rel_path.replace("\\", "/")
        if ".." in rel_path:
            raise ValueError("Parent directory segments are not allowed in paths.")
        abs_path = os.path.abspath(os.path.join(self.base_dir, rel_path))
        base = os.path.abspath(self.base_dir)
        if not abs_path.startswith(base):
            raise ValueError("Access outside sandbox base directory is forbidden.")
        return abs_path

    def list_files(self) -> Dict[str, Any]:
        entries: List[str] = []
        for name in os.listdir(self.base_dir):
            entries.append(name)
        return {"action": "list_files", "entries": entries}

    def read_file(self, rel_path: str) -> Dict[str, Any]:
        path = self._safe_path(rel_path)
        if not os.path.isfile(path):
            return {"action": "read_file", "error": "File does not exist."}
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(self.max_output_bytes)
        truncated = os.path.getsize(path) > self.max_output_bytes
        return {
            "action": "read_file",
            "path": rel_path,
            "content": content,
            "truncated": truncated,
        }


if __name__ == "__main__":
    # Example usage in isolation. Adjust base_dir when running manually.
    base = os.getcwd()
    tool = CmdTool(base_dir=base)
    print(tool.list_files())

