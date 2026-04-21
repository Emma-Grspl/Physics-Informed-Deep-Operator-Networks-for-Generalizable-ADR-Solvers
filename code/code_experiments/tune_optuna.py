"""Inspect the best Optuna trial available in the local comparison workspace."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "src" / "utils" / "get_best_trial_optuna.py"
    runpy.run_path(str(target), run_name="__main__")
