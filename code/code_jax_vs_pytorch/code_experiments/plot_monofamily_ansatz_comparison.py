from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = CODE_ROOT.parent / "assets"
RESULT_ROOT = CODE_ROOT.parent / "results"

PT_COLOR = "deepskyblue"
JAX_COLOR = "deeppink"
CN_COLOR = "black"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def setup_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "#faf7f2",
            "axes.facecolor": "#fffdf9",
            "axes.edgecolor": "#2b251f",
            "axes.labelcolor": "#2b251f",
            "xtick.color": "#2b251f",
            "ytick.color": "#2b251f",
            "text.color": "#2b251f",
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.color": "#8d7f73",
            "font.size": 11,
        }
    )


def savefig(fig: plt.Figure, name: str) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_bundle() -> dict:
    return {
        "Sin-Gauss": {
            "pytorch": {
                "train": load_json(RESULT_ROOT / "pytorch" / "benchmark_singauss_ansatz_t1" / "seed_0" / "train_metrics.json"),
                "eval": load_json(RESULT_ROOT / "pytorch" / "benchmark_singauss_ansatz_t1" / "seed_0" / "evaluation.json"),
                "infer": load_json(RESULT_ROOT / "pytorch" / "benchmark_singauss_ansatz_t1" / "seed_0" / "inference.json"),
            },
            "jax": {
                "train": load_json(RESULT_ROOT / "jax" / "benchmark_singauss_ansatz_t1" / "seed_0" / "train_metrics.json"),
                "eval": load_json(RESULT_ROOT / "jax" / "benchmark_singauss_ansatz_t1" / "seed_0" / "evaluation.json"),
                "infer": load_json(RESULT_ROOT / "jax" / "benchmark_singauss_ansatz_t1" / "seed_0" / "inference.json"),
            },
        },
        "Gaussian": {
            "pytorch": {
                "train": load_json(RESULT_ROOT / "pytorch" / "benchmark_gaussian_ansatz_t1" / "seed_0" / "train_metrics.json"),
                "eval": load_json(RESULT_ROOT / "pytorch" / "benchmark_gaussian_ansatz_t1" / "seed_0" / "evaluation.json"),
                "infer": load_json(RESULT_ROOT / "pytorch" / "benchmark_gaussian_ansatz_t1" / "seed_0" / "inference.json"),
            },
            "jax": None,
        },
    }


def plot_l2(bundle: dict) -> None:
    families = list(bundle.keys())
    x = np.arange(len(families))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    pt_means = [bundle[f]["pytorch"]["eval"]["global_l2_mean"] for f in families]
    pt_stds = [bundle[f]["pytorch"]["eval"]["global_l2_std"] for f in families]
    jax_means = [bundle[f]["jax"]["eval"]["global_l2_mean"] if bundle[f]["jax"] else np.nan for f in families]
    jax_stds = [bundle[f]["jax"]["eval"]["global_l2_std"] if bundle[f]["jax"] else np.nan for f in families]

    pt_bars = ax.bar(x - width / 2, pt_means, yerr=pt_stds, capsize=6, width=width, color=PT_COLOR, alpha=0.9, label="Pytorch")
    jax_vals = np.nan_to_num(jax_means, nan=0.0)
    jax_bars = ax.bar(x + width / 2, jax_vals, yerr=np.nan_to_num(jax_stds, nan=0.0), capsize=6, width=width, color=JAX_COLOR, alpha=0.9, label="JAX")

    for bar, val in zip(pt_bars, pt_means):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.08, f"{val:.2%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for idx, (bar, val) in enumerate(zip(jax_bars, jax_means)):
        if np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, 0.015, "N/A", ha="center", va="bottom", fontsize=9, fontweight="bold", color=JAX_COLOR)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, val * 1.08, f"{val:.2%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(0.05, color=CN_COLOR, linestyle="--", linewidth=1.2, label="5% threshold")
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 2e0)
    ax.set_xticks(x, families)
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Ansatz Monofamily Relative L2 Error")
    ax.legend(frameon=True, loc="upper left")
    ax.text(0.99, 0.02, "Gaussian JAX final evaluation missing locally", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    savefig(fig, "1_ansatz_family_l2.png")


def plot_training_time(bundle: dict) -> None:
    families = list(bundle.keys())
    x = np.arange(len(families))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    pt_vals = [bundle[f]["pytorch"]["train"]["total_time_sec"] / 3600.0 for f in families]
    jax_vals = [bundle[f]["jax"]["train"]["total_time_sec"] / 3600.0 if bundle[f]["jax"] else np.nan for f in families]

    pt_bars = ax.bar(x - width / 2, pt_vals, width=width, color=PT_COLOR, alpha=0.9, label="Pytorch")
    jax_plot = np.nan_to_num(jax_vals, nan=0.0)
    jax_bars = ax.bar(x + width / 2, jax_plot, width=width, color=JAX_COLOR, alpha=0.9, label="JAX")

    for bar, val in zip(pt_bars, pt_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f} h", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for bar, val in zip(jax_bars, jax_vals):
        if np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, 0.05, "N/A", ha="center", va="bottom", fontsize=9, fontweight="bold", color=JAX_COLOR)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f} h", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x, families)
    ax.set_ylabel("Training time (hours)")
    ax.set_title("Ansatz Monofamily Training Time")
    ax.legend(frameon=True, loc="upper left")
    ax.text(0.99, 0.02, "Gaussian JAX final JSON missing locally", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    savefig(fig, "2_ansatz_training_time.png")


def plot_inference(bundle: dict) -> None:
    families = list(bundle.keys())
    x = np.arange(len(families))
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), constrained_layout=True)
    panels = [
        ("inference_full_grid_sec", "Full-grid inference"),
        ("time_jump_sec", "Time-jump inference"),
    ]

    for ax, (key, title) in zip(axes, panels):
        pt_vals = [bundle[f]["pytorch"]["infer"][key] for f in families]
        jax_vals = [bundle[f]["jax"]["infer"][key] if bundle[f]["jax"] else np.nan for f in families]
        cn_vals = []
        for family in families:
            if bundle[family]["jax"]:
                cn = 0.5 * (
                    bundle[family]["pytorch"]["infer"]["cn_reference_sec"]
                    + bundle[family]["jax"]["infer"]["cn_reference_sec"]
                )
            else:
                cn = bundle[family]["pytorch"]["infer"]["cn_reference_sec"]
            cn_vals.append(cn)

        ax.bar(x - width, pt_vals, width=width, color=PT_COLOR, alpha=0.9, label="Pytorch")
        ax.bar(x, np.nan_to_num(jax_vals, nan=0.0), width=width, color=JAX_COLOR, alpha=0.9, label="JAX")
        ax.bar(x + width, cn_vals, width=width, color=CN_COLOR, alpha=0.9, label="CN")

        for idx, val in enumerate(jax_vals):
            if np.isnan(val):
                ax.text(x[idx], 0.01 if key == "time_jump_sec" else 0.02, "N/A", ha="center", va="bottom", fontsize=9, fontweight="bold", color=JAX_COLOR)

        ax.set_xticks(x, families)
        ax.set_ylabel("Time (s)")
        ax.set_title(title)

    axes[0].legend(frameon=True, loc="upper left")
    axes[1].text(0.98, 0.02, "Gaussian JAX final inference missing locally", transform=axes[1].transAxes, ha="right", va="bottom", fontsize=9)
    savefig(fig, "3_ansatz_inference.png")


def plot_summary(bundle: dict) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    ax.axis("off")
    rows = [
        ["Sin-Gauss", f"{bundle['Sin-Gauss']['pytorch']['eval']['global_l2_mean']:.2%}", f"{bundle['Sin-Gauss']['jax']['eval']['global_l2_mean']:.2%}", f"{bundle['Sin-Gauss']['pytorch']['train']['total_time_sec']/3600:.2f} h", f"{bundle['Sin-Gauss']['jax']['train']['total_time_sec']/3600:.2f} h"],
        ["Gaussian", f"{bundle['Gaussian']['pytorch']['eval']['global_l2_mean']:.2%}", "N/A", f"{bundle['Gaussian']['pytorch']['train']['total_time_sec']/3600:.2f} h", "N/A"],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["Family", "Pytorch L2", "JAX L2", "Pytorch train", "JAX train"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    ax.set_title("Ansatz Monofamily Summary", pad=14, fontweight="bold")
    ax.text(0.99, 0.04, "Gaussian JAX run stopped before final evaluation/inference export", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    savefig(fig, "4_ansatz_summary.png")


def main() -> None:
    setup_style()
    bundle = load_bundle()
    plot_l2(bundle)
    plot_training_time(bundle)
    plot_inference(bundle)
    plot_summary(bundle)


if __name__ == "__main__":
    main()
