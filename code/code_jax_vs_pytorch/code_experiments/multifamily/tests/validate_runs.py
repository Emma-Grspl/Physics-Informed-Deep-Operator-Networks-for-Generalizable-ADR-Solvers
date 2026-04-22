from __future__ import annotations

from pathlib import Path


REQUIRED = {
    "pytorch/benchmark_fulltrainer_t1/seed_0": [
        "train_metrics.json",
        "evaluation.json",
        "inference.json",
    ],
    "jax/benchmark_fulltrainer_t1_equal/seed_0": [
        "train_metrics.json",
        "evaluation.json",
        "inference.json",
    ],
}


def main() -> None:
    root = Path(__file__).resolve().parents[3] / "benchmarks"
    missing: list[str] = []
    for rel_dir, filenames in REQUIRED.items():
        run_dir = root / rel_dir
        for filename in filenames:
            path = run_dir / filename
            if not path.exists():
                missing.append(str(path))
    if missing:
        print("Missing benchmark artifacts:")
        for item in missing:
            print(f"- {item}")
        raise SystemExit(1)
    print("Multifamily benchmark artifacts: OK")


if __name__ == "__main__":
    main()
