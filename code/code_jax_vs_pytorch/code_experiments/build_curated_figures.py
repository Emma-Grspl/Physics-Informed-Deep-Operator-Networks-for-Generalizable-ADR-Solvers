from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT.parent / "results"
FIGURES = ROOT.parent / "assets"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)


def build_gaussian_hypothesis_plot() -> None:
    csv_path = ROOT.parent / "plot" / "gaussian_hypothesis" / "gaussian_hypothesis_summary.csv"
    rows = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = [
            {
                "label": "PyTorch free / no LBFGS",
                "global_l2_mean": 0.823913738971746,
                "global_l2_std": 0.061121115674755755,
            },
            {
                "label": "PyTorch free / LBFGS",
                "global_l2_mean": 0.8658132706325194,
                "global_l2_std": 0.07449073012402514,
            },
            {
                "label": "PyTorch ansatz / no LBFGS",
                "global_l2_mean": 0.16062695969369203,
                "global_l2_std": 0.08412199175845378,
            },
            {
                "label": "PyTorch ansatz / LBFGS",
                "global_l2_mean": 0.21143403117354867,
                "global_l2_std": 0.13352146634659892,
            },
            {
                "label": "JAX free / no LBFGS",
                "global_l2_mean": 1.0064751445180216,
                "global_l2_std": 0.006037548740389424,
            },
            {
                "label": "JAX free / LBFGS",
                "global_l2_mean": 1.0064698710157551,
                "global_l2_std": 0.005943902499538973,
            },
            {
                "label": "JAX ansatz / no LBFGS",
                "global_l2_mean": 0.4813663156359794,
                "global_l2_std": 0.005603280985025883,
            },
            {
                "label": "JAX ansatz / LBFGS",
                "global_l2_mean": 0.48024455615784006,
                "global_l2_std": 0.005648931121904037,
            },
        ]

    labels = [row["label"] for row in rows]
    means = [float(row["global_l2_mean"]) for row in rows]
    stds = [float(row["global_l2_std"]) for row in rows]
    colors = ["#1f77b4" if label.startswith("PyTorch") else "#ff7f0e" for label in labels]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(labels))
    ax.bar(x, means, yerr=stds, color=colors, capsize=4)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Global relative L2")
    ax.set_title("Gaussian Hypothesis: ansatz dominates, LBFGS does not")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "gaussian_hypothesis_ablation.png", dpi=180)
    plt.close(fig)


def build_monofamily_ansatz_plot() -> None:
    values = {
        "PyTorch free": {
            "Tanh": 0.0015806030778476791,
            "Sin-Gauss": 1.0000006202628895,
            "Gaussian": 0.8721232848589683,
        },
        "PyTorch ansatz": {
            "Sin-Gauss": 0.8644844466169989,
            "Gaussian": 0.07642588445238312,
        },
        "JAX free": {
            "Tanh": 9.071589754694044,
            "Sin-Gauss": 1.3952637210549168,
            "Gaussian": 1.0220428833814377,
        },
        "JAX ansatz": {
            "Sin-Gauss": 0.8995858417738374,
        },
    }

    families = ["Tanh", "Sin-Gauss", "Gaussian"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    x = range(len(families))
    width = 0.35
    axes[0].bar(
        [i - width / 2 for i in x],
        [values["PyTorch free"].get(f, 0.0) for f in families],
        width=width,
        label="PyTorch",
        color="#1f77b4",
    )
    axes[0].bar(
        [i + width / 2 for i in x],
        [values["JAX free"].get(f, 0.0) for f in families],
        width=width,
        label="JAX",
        color="#ff7f0e",
    )
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(families)
    axes[0].set_ylabel("Global relative L2")
    axes[0].set_title("Monofamily comparison without ansatz")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    ansatz_families = ["Sin-Gauss", "Gaussian"]
    x2 = range(len(ansatz_families))
    axes[1].bar(
        [i - width / 2 for i in x2],
        [values["PyTorch ansatz"].get(f, 0.0) for f in ansatz_families],
        width=width,
        label="PyTorch ansatz",
        color="#2ca02c",
    )
    axes[1].bar(
        [i + width / 2 for i in x2],
        [values["JAX ansatz"].get(f, 0.0) for f in ansatz_families],
        width=width,
        label="JAX ansatz",
        color="#d62728",
    )
    axes[1].set_xticks(list(x2))
    axes[1].set_xticklabels(ansatz_families)
    axes[1].set_ylabel("Global relative L2")
    axes[1].set_title("Effect of ansatz on hard families")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "monofamily_ansatz_overview.png", dpi=180)
    plt.close(fig)


def build_reference_tables() -> None:
    multifamily_eval = load_json(RESULTS / "pytorch" / "benchmark_fulltrainer_t1" / "seed_0" / "evaluation.json")
    jax_eval = load_json(RESULTS / "jax" / "benchmark_fulltrainer_t1" / "seed_0" / "evaluation.json")
    pytorch_train = load_json(RESULTS / "pytorch" / "benchmark_fulltrainer_t1" / "seed_0" / "train_metrics.json")
    jax_train = load_json(RESULTS / "jax" / "benchmark_fulltrainer_t1" / "seed_0" / "train_metrics.json")

    out_path = FIGURES / "multifamily_reference_metrics.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["backend", "global_l2_mean", "Tanh", "Sin-Gauss", "Gaussian", "training_time_sec"])
        writer.writerow(
            [
                "pytorch",
                multifamily_eval["global_l2_mean"],
                multifamily_eval["family_l2_mean"]["Tanh"],
                multifamily_eval["family_l2_mean"]["Sin-Gauss"],
                multifamily_eval["family_l2_mean"]["Gaussian"],
                pytorch_train["total_time_sec"],
            ]
        )
        writer.writerow(
            [
                "jax",
                jax_eval["global_l2_mean"],
                jax_eval["family_l2_mean"]["Tanh"],
                jax_eval["family_l2_mean"]["Sin-Gauss"],
                jax_eval["family_l2_mean"]["Gaussian"],
                jax_train["total_time_sec"],
            ]
        )


def main() -> None:
    ensure_dirs()
    build_gaussian_hypothesis_plot()
    build_monofamily_ansatz_plot()
    build_reference_tables()


if __name__ == "__main__":
    main()
