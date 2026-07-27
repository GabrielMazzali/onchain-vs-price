"""Launcher script to execute the LR and XGBoost notebooks in parallel.

Usage (from the project root):

    python utils/run_models_parallel.py

Each notebook is re-executed with `jupyter nbconvert` and the result is written
next to it as `<name>_output.ipynb`. The LSTM notebook is intentionally left
out of the parallel launcher: it is a separate capacity-ceiling stage with
optional heavy dependencies (torch/poutyne) that are not installed by default.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Project-root-relative notebook paths. Update this list when new notebooks
# join the pipeline (or when the LSTM notebook is ready for batch execution).
NOTEBOOKS: list[Path] = [
    PROJECT_ROOT / "pipeline" / "2a_lr_model_pipeline.ipynb",
    PROJECT_ROOT / "pipeline" / "2b_xgboost_model_pipeline.ipynb",
]


def run_notebook_async(notebook_path: Path) -> subprocess.Popen:
    """Launch `jupyter nbconvert --execute` in a subprocess and return the handle."""
    output_path = notebook_path.with_name(notebook_path.stem + "_output.ipynb")
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=3600",
        f"--output={output_path.name}",
        f"--output-dir={notebook_path.parent}",
        str(notebook_path),
    ]
    print(f"[START] {notebook_path.relative_to(PROJECT_ROOT)}")
    return subprocess.Popen(cmd)


def main() -> None:
    missing = [nb for nb in NOTEBOOKS if not nb.exists()]
    if missing:
        print("[ERROR] Notebooks not found:")
        for nb in missing:
            print(f"  - {nb}")
        sys.exit(1)

    print("=" * 60)
    print(f"Running {len(NOTEBOOKS)} model pipelines in PARALLEL")
    print("=" * 60)

    processes = [(nb, run_notebook_async(nb)) for nb in NOTEBOOKS]
    print(f"\n[INFO] {len(processes)} processes started. Waiting for completion...\n")

    exit_code = 0
    for nb, proc in processes:
        rc = proc.wait()
        status = "DONE" if rc == 0 else "FAILED"
        print(f"[{status}] {nb.relative_to(PROJECT_ROOT)} (rc={rc})")
        if rc != 0:
            exit_code = rc

    print("\n" + "=" * 60)
    print("All pipelines complete.")
    print("=" * 60)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
