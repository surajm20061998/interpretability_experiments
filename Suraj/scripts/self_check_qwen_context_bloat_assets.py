from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "qwen_context_bloat_experiments.ipynb"
GENERATOR = ROOT / "scripts" / "generate_qwen_context_bloat_notebook.py"
SETUP_SCRIPT = ROOT / "scripts" / "setup_qwen_context_bloat_env.sh"
ENV_FILE = ROOT / "environment.yml"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_command(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def main() -> None:
    require(NOTEBOOK.exists(), f"Notebook not found: {NOTEBOOK}")
    require(GENERATOR.exists(), f"Generator not found: {GENERATOR}")
    require(SETUP_SCRIPT.exists(), f"Setup script not found: {SETUP_SCRIPT}")
    require(ENV_FILE.exists(), f"Environment file not found: {ENV_FILE}")

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    require("cells" in notebook and notebook["cells"], "Notebook JSON is missing cells.")

    notebook_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    env_text = ENV_FILE.read_text(encoding="utf-8")

    require("from tqdm import tqdm" in notebook_text, "Notebook should import tqdm without auto-detection.")
    require("tqdm.auto" not in notebook_text, "Notebook still contains tqdm.auto.")
    require("token_exact_match" in notebook_text, "Notebook is missing token_exact_match.")
    require(
        "normalized_text_exact_match" in notebook_text,
        "Notebook is missing normalized_text_exact_match.",
    )
    require(
        '"exact_match": normalized_text_exact_match' in notebook_text,
        "Notebook should alias exact_match to normalized_text_exact_match.",
    )
    require(
        'accuracy=("normalized_text_exact_match", "mean")' in notebook_text,
        "Notebook plots should aggregate normalized_text_exact_match.",
    )
    require("front_mean_token_logprob" in notebook_text, "Competition scoring fields are missing.")
    require("ipywidgets==8.1.7" in env_text, "environment.yml should include ipywidgets.")

    run_command([sys.executable, "-m", "py_compile", str(GENERATOR)])
    run_command(["bash", "-n", str(SETUP_SCRIPT)])

    print("Self-check passed.")
    print(f"Notebook: {NOTEBOOK}")
    print(f"Generator: {GENERATOR}")
    print(f"Setup script: {SETUP_SCRIPT}")


if __name__ == "__main__":
    main()
