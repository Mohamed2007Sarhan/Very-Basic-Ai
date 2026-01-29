"""
setup.py — Install package and optionally regenerate model weights and JSON artifacts.

After deleting model_weights.npz and the various vocab/config JSON files, run:

  pip install -e .
  python setup.py install_artifacts

to reinstall dependencies and regenerate:
  - model_weights.npz
  - model_config.json
  - vocab.json

For the full v2 pipeline (vocab_v2.json, vocab_char/word/sentence/bpe.json, bpe_merges.json),
run train_v2.py after install_artifacts.
"""

from setuptools import setup
import os
import subprocess
import sys


def _project_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _run_train() -> None:
    """Run train.py to regenerate model_weights.npz, model_config.json, vocab.json."""
    project = _project_dir()
    train_py = os.path.join(project, "train.py")
    if not os.path.isfile(train_py):
        raise FileNotFoundError(f"train.py not found in {project}")
    subprocess.check_call(
        [sys.executable, train_py],
        cwd=project,
        env={**os.environ},
    )


try:
    from setuptools import Command
except ImportError:
    from distutils.core import Command


class InstallArtifactsCommand(Command):
    """Custom command: python setup.py install_artifacts — regenerate weights and JSON."""

    description = "Regenerate model_weights.npz, model_config.json, and vocab.json by running train.py"
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        print("Running train.py to regenerate weights and JSON...")
        _run_train()
        print("install_artifacts done.")


setup(
    name="ai_from_scratch",
    version="0.1.0",
    description="From-scratch transformer language model and agent (NumPy only)",
    python_requires=">=3.10",
    install_requires=["numpy>=1.20"],
    packages=["core", "agent", "memory", "tools"],
    py_modules=["model", "train", "train_v2", "generate", "init_weights", "run_agent"],
    cmdclass={"install_artifacts": InstallArtifactsCommand},
)
