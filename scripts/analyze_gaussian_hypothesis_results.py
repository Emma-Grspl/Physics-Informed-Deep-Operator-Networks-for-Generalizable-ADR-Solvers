#!/usr/bin/env python3
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUT_DIR = ROOT / "plot" / "gaussian_hypothesis"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_pytorch_runs():
    variants = [
        ("benchmark_gaussian_hyp_free_lbfgs_off", "PyTorch free / no LBFGS", "pytorch", "free", "off"),
        ("benchmark_gaussian_hyp_free_lbfgs_on", "PyTorch free / LBFGS", "pytorch", "free", "on"),
        ("benchmark_gaussian_hyp_ansatz_lbfgs_off", "PyTorch ansatz / no LBFGS", "pytorch", "ansatz", "off"),
        ("benchmark_gaussian_hyp_ansatz_lbfgs_on", "PyTorch ansatz / LBFGS", "pytorch", "ansatz", "on"),
    ]
    rows = []
    for folder, label, backend, ansatz, lbfgs in variants:
        run_dir = RESULTS_DIR / backend / folder
        seed_values = []
        train_times = []
        for seed_dir in sorted(run_dir.glob("seed_*")):
            eval_path = seed_dir / "evaluation.json"
            train_path = seed_dir / "train_metrics.json"
            if eval_path.exists():
                eval_data = load_json(eval_path)
                seed_values.append(eval_data["global_l2_mean"])
            if train_path.exists():
                train_data = load_json(train_path)
                train_times.append(train_data.get("total_time_sec"))

        rows.append(
            {
                "label": label,
                "backend": backend,
                "ansatz": ansatz,
                "lbfgs": lbfgs,
                "status": "complete" if seed_values else "missing",
                "n_seeds": len(seed_values),
                "global_l2_mean": mean(seed_values) if seed_values else None,
                "global_l2_std": pstdev(seed_values) if len(seed_values) > 1 else 0.0 if seed_values else None,
                "train_time_sec_mean": mean([x for x in train_times if x is not None]) if train_times else None,
                "source": str(run_dir.relative_to(ROOT)),
            }
        )
    return rows


def summarize_jax_legacy_runs():
    rows = []

    free_dir = RESULTS_DIR / "jax" / "benchmark_fulltrainer_t1_gaussian_only" / "seed_0"
    free_eval = load_json(free_dir / "evaluation.json")
    free_train = load_json(free_dir / "train_metrics.json")
    rows.append(
        {
            "label": "JAX free legacy",
            "backend": "jax",
            "ansatz": "free",
            "lbfgs": "unknown",
            "status": "complete_legacy",
            "n_seeds": 1,
            "global_l2_mean": free_eval["global_l2_mean"],
            "global_l2_std": 0.0,
            "train_time_sec_mean": free_train.get("total_time_sec"),
            "source": str(free_dir.parent.relative_to(ROOT)),
        }
    )

    ansatz_dir = RESULTS_DIR / "jax" / "benchmark_gaussian_ansatz_t1" / "seed_0"
    ansatz_train = load_json(ansatz_dir / "train_metrics.json")
    time_steps = ansatz_train.get("trainer_metrics", {}).get("time_steps", [])
    successful_steps = [step for step in time_steps if step.get("success")]
    best_err = min((step["err"] for step in successful_steps), default=None)
    last_t = max((step["t_step"] for step in successful_steps), default=None)
    rows.append(
        {
            "label": "JAX ansatz legacy",
            "backend": "jax",
            "ansatz": "ansatz",
            "lbfgs": "unknown",
            "status": "partial_legacy",
            "n_seeds": 1,
            "global_l2_mean": None,
            "global_l2_std": None,
            "train_time_sec_mean": ansatz_train.get("total_time_sec"),
            "best_step_err": best_err,
            "last_success_t": last_t,
            "source": str(ansatz_dir.parent.relative_to(ROOT)),
        }
    )
    return rows, time_steps


def summarize_jax_gaussian_hyp_runs():
    variants = [
        ("benchmark_gaussian_hyp_free_lbfgs_off", "JAX free / no LBFGS", "jax", "free", "off"),
        ("benchmark_gaussian_hyp_free_lbfgs_on", "JAX free / LBFGS", "jax", "free", "on"),
        ("benchmark_gaussian_hyp_ansatz_lbfgs_off", "JAX ansatz / no LBFGS", "jax", "ansatz", "off"),
        ("benchmark_gaussian_hyp_ansatz_lbfgs_on", "JAX ansatz / LBFGS", "jax", "ansatz", "on"),
    ]
    rows = []
    for folder, label, backend, ansatz, lbfgs in variants:
        run_dir = ROOT / backend / folder
        seed_values = []
        train_times = []
        for seed_dir in sorted(run_dir.glob("seed_*")):
            eval_path = seed_dir / "evaluation.json"
            train_path = seed_dir / "train_metrics.json"
            if eval_path.exists():
                eval_data = load_json(eval_path)
                seed_values.append(eval_data["global_l2_mean"])
            if train_path.exists():
                train_data = load_json(train_path)
                train_times.append(train_data.get("total_time_sec"))

        rows.append(
            {
                "label": label,
                "backend": backend,
                "ansatz": ansatz,
                "lbfgs": lbfgs,
                "status": "complete" if seed_values else "missing",
                "n_seeds": len(seed_values),
                "global_l2_mean": mean(seed_values) if seed_values else None,
                "global_l2_std": pstdev(seed_values) if len(seed_values) > 1 else 0.0 if seed_values else None,
                "train_time_sec_mean": mean([x for x in train_times if x is not None]) if train_times else None,
                "source": str(run_dir.relative_to(ROOT)),
            }
        )
    return rows


def write_summary_csv(rows, out_path: Path):
    fieldnames = [
        "label",
        "backend",
        "ansatz",
        "lbfgs",
        "status",
        "n_seeds",
        "global_l2_mean",
        "global_l2_std",
        "train_time_sec_mean",
        "best_step_err",
        "last_success_t",
        "source",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_final_error_plot(rows, out_path: Path):
    plot_rows = [row for row in rows if row.get("global_l2_mean") is not None]
    labels = [row["label"] for row in plot_rows]
    values = [row["global_l2_mean"] for row in plot_rows]
    errors = [row["global_l2_std"] or 0.0 for row in plot_rows]
    colors = []
    for row in plot_rows:
        if row["backend"] == "pytorch" and row["ansatz"] == "free":
            colors.append("#c44e52")
        elif row["backend"] == "pytorch":
            colors.append("#55a868")
        else:
            colors.append("#4c72b0")

    plt.figure(figsize=(10, 5.5))
    plt.bar(range(len(values)), values, yerr=errors, color=colors, capsize=6)
    plt.xticks(range(len(values)), labels, rotation=20, ha="right")
    plt.ylabel("Global relative L2")
    plt.title("Gaussian mono-family: available final errors")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_jax_progress_plot(time_steps, out_path: Path):
    t_vals = [step["t_step"] for step in time_steps]
    errs = [step["err"] for step in time_steps]
    colors = ["#55a868" if step["success"] else "#c44e52" for step in time_steps]

    plt.figure(figsize=(8, 4.5))
    plt.plot(t_vals, errs, color="#4c72b0", linewidth=2, alpha=0.9)
    plt.scatter(t_vals, errs, c=colors, s=70, zorder=3)
    plt.axvline(0.5, color="#c44e52", linestyle="--", linewidth=1, alpha=0.8)
    plt.ylim(0.0, max(errs) * 1.05 if errs else 1.0)
    plt.xlabel("Reached t_step")
    plt.ylabel("Audit error")
    plt.title("JAX Gaussian ansatz legacy run progression")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_summary_json(rows, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pytorch_rows = summarize_pytorch_runs()
    jax_rows = summarize_jax_gaussian_hyp_runs()
    rows = pytorch_rows + jax_rows

    write_summary_csv(rows, OUT_DIR / "gaussian_hypothesis_summary.csv")
    write_summary_json(rows, OUT_DIR / "gaussian_hypothesis_summary.json")
    make_final_error_plot(rows, OUT_DIR / "gaussian_hypothesis_final_errors.png")

    legacy_jax_rows, jax_time_steps = summarize_jax_legacy_runs()
    write_summary_json(legacy_jax_rows, OUT_DIR / "gaussian_hypothesis_legacy_jax_summary.json")
    make_jax_progress_plot(jax_time_steps, OUT_DIR / "jax_gaussian_ansatz_progress.png")

    print(f"Wrote analysis artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
