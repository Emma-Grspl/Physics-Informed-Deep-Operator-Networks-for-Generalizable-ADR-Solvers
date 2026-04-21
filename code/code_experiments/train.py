"""Run the default PyTorch side of the strict JAX vs PyTorch comparison."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "benchmarks" / "pytorch" / "train_fulltrainer_benchmark.py"
    runpy.run_path(str(target), run_name="__main__")
