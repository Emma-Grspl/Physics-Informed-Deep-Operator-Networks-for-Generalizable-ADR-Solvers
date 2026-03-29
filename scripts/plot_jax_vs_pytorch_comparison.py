from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generators import get_ic_value
from src.models.PI_DeepOnet_ADR import PI_DeepONet_ADR
from src.utils.CN_ADR import crank_nicolson_adr
from src_jax.models.pi_deeponet_adr import apply_model


ASSET_DIR = PROJECT_ROOT / "assets_jax"
BENCH_ROOTS = [PROJECT_ROOT / "benchmarks", PROJECT_ROOT / "outputs" / "benchmarks"]

JAX_NAME = "benchmark_fulltrainer_t1_equal"
PT_NAME = "benchmark_fulltrainer_t1"
SEED = "seed_0"


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jax_pickle_compat(path: Path):
    from jax._src import core

    original_init = core.ShapedArray.__init__

    def patched_init(self, shape, dtype, weak_type=False, **kwargs):
        kwargs.pop("named_shape", None)
        kwargs.pop("memory_space", None)
        kwargs.pop("sharding", None)
        kwargs.pop("vma", None)
        return original_init(self, shape, dtype, weak_type)

    core.ShapedArray.__init__ = patched_init
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    finally:
        core.ShapedArray.__init__ = original_init


def resolve_run_dir(backend: str, benchmark_name: str) -> Path:
    for root in BENCH_ROOTS:
        candidate = root / backend / benchmark_name / SEED
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Run directory not found for {backend}/{benchmark_name}")


def family_order() -> list[str]:
    return ["Tanh", "Sin-Gauss", "Gaussian"]


def load_benchmark_bundle() -> dict:
    jax_dir = resolve_run_dir("jax", JAX_NAME)
    pt_dir = resolve_run_dir("pytorch", PT_NAME)
    return {
        "jax": {
            "run_dir": jax_dir,
            "train": load_json(jax_dir / "train_metrics.json"),
            "eval": load_json(jax_dir / "evaluation.json"),
            "infer": load_json(jax_dir / "inference.json"),
        },
        "pytorch": {
            "run_dir": pt_dir,
            "train": load_json(pt_dir / "train_metrics.json"),
            "eval": load_json(pt_dir / "evaluation.json"),
            "infer": load_json(pt_dir / "inference.json"),
        },
    }


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
    ASSET_DIR.mkdir(exist_ok=True)
    fig.savefig(ASSET_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_family_l2(bundle: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6))
    fams = family_order()
    x = np.arange(len(fams))
    width = 0.34
    colors = {"pytorch": "#d1495b", "jax": "#00798c"}

    for idx, backend in enumerate(["pytorch", "jax"]):
        means = [bundle[backend]["eval"]["family_l2_mean"][fam] for fam in fams]
        stds = [bundle[backend]["eval"]["family_l2_std"][fam] for fam in fams]
        offset = -width / 2 if backend == "pytorch" else width / 2
        bars = ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=8,
            color=colors[backend],
            alpha=0.88,
            label="PyTorch" if backend == "pytorch" else "JAX",
        )
        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.12,
                f"{value:.2%}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                rotation=0,
            )

    ax.axhline(0.05, color="#222222", linestyle="--", linewidth=1.3, label="5% threshold")
    ax.set_xticks(x, fams)
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Family-wise L2 Error")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 5.0)
    ax.legend(frameon=True)
    savefig(fig, "1_family_l2_comparison.png")


def plot_final_l2_vs_training_time(bundle: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6))
    colors = {"pytorch": "#d1495b", "jax": "#00798c"}
    markers = {"Global": "X", "Tanh": "o", "Sin-Gauss": "s", "Gaussian": "D"}

    for backend in ["pytorch", "jax"]:
        train_hours = bundle[backend]["train"]["total_time_sec"] / 3600.0
        metrics = {"Global": bundle[backend]["eval"]["global_l2_mean"]}
        metrics.update(bundle[backend]["eval"]["family_l2_mean"])
        for metric_name, value in metrics.items():
            ax.scatter(
                train_hours,
                value,
                s=120,
                marker=markers[metric_name],
                color=colors[backend],
                edgecolors="#1f1f1f",
                linewidths=0.8,
                alpha=0.95,
            )
            ax.annotate(
                f"{backend[:2].upper()} {metric_name}",
                (train_hours, value),
                textcoords="offset points",
                xytext=(8, 6),
                fontsize=9,
            )

    ax.axhline(0.05, color="#222222", linestyle="--", linewidth=1.3)
    ax.set_yscale("log")
    ax.set_xlabel("Training time (hours)")
    ax.set_ylabel("Final relative L2 error")
    ax.set_title("Final L2 Error vs Training Time")
    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d1495b", markeredgecolor="#1f1f1f", markersize=10, label="PyTorch"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#00798c", markeredgecolor="#1f1f1f", markersize=10, label="JAX"),
        Line2D([0], [0], color="#222222", linestyle="--", label="5% threshold"),
    ]
    ax.legend(handles=legend_items, frameon=True)
    savefig(fig, "2_final_l2_vs_training_time.png")


def plot_training_time_hist(bundle: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    labels = ["PyTorch", "JAX"]
    values = [bundle["pytorch"]["train"]["total_time_sec"] / 60.0, bundle["jax"]["train"]["total_time_sec"] / 60.0]
    colors = ["#d1495b", "#00798c"]
    bars = ax.bar(labels, values, color=colors, alpha=0.9)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f} min", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Total training time (minutes)")
    ax.set_title("Raw Training Time")
    savefig(fig, "3_training_time_histogram.png")


def plot_inference(bundle: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    colors = {"PyTorch": "#d1495b", "JAX": "#00798c", "CN": "#2f2f2f"}

    full_vals = [
        bundle["pytorch"]["infer"]["inference_full_grid_sec"],
        bundle["jax"]["infer"]["inference_full_grid_sec"],
        float(np.mean([bundle["pytorch"]["infer"]["cn_reference_sec"], bundle["jax"]["infer"]["cn_reference_sec"]])),
    ]
    jump_vals = [
        bundle["pytorch"]["infer"]["time_jump_sec"],
        bundle["jax"]["infer"]["time_jump_sec"],
        float(np.mean([bundle["pytorch"]["infer"]["cn_reference_sec"], bundle["jax"]["infer"]["cn_reference_sec"]])),
    ]

    for ax, vals, title in zip(
        axes,
        [full_vals, jump_vals],
        ["Full-grid inference", "Time-jump inference"],
    ):
        bars = ax.bar(["PyTorch", "JAX", "CN"], vals, color=[colors["PyTorch"], colors["JAX"], colors["CN"]], alpha=0.9)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(title)
        ax.set_ylabel("Time (s)")

    savefig(fig, "4_inference_vs_cn.png")


def plot_threshold_frontier(bundle: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6))
    colors = {"pytorch": "#d1495b", "jax": "#00798c"}

    for backend in ["pytorch", "jax"]:
        x = bundle[backend]["train"]["total_time_sec"] / 3600.0
        y = bundle[backend]["eval"]["global_l2_mean"]
        ax.scatter(x, y, s=240, color=colors[backend], edgecolors="#1f1f1f", linewidths=1.0)
        ax.annotate(
            f"{backend.capitalize()}\n{y:.2%}",
            (x, y),
            textcoords="offset points",
            xytext=(8, -4),
            fontsize=10,
            fontweight="bold",
        )

    ax.axhline(0.05, color="#222222", linestyle="--", linewidth=1.4, label="5% threshold")
    ax.set_yscale("log")
    ax.set_xlabel("Training time (hours)")
    ax.set_ylabel("Global L2 error")
    ax.set_title("Quality vs Time Frontier")
    ax.legend(frameon=True)
    savefig(fig, "5_quality_time_frontier_threshold5.png")


def load_models_and_cfg():
    cfg_pt = load_yaml(PROJECT_ROOT / "configs" / "config_ADR_t1_compare.yaml")
    cfg_jax = load_yaml(PROJECT_ROOT / "configs_jax" / "config_ADR_jax_t1_equal_pipeline.yaml")

    pt_dir = resolve_run_dir("pytorch", PT_NAME)
    jax_dir = resolve_run_dir("jax", JAX_NAME)

    device = torch.device("cpu")
    pt_model = PI_DeepONet_ADR(cfg_pt).to(device)
    checkpoint = torch.load(pt_dir / "model.pt", map_location=device)
    pt_model.load_state_dict(checkpoint["model_state_dict"])
    pt_model.eval()

    jax_params = load_jax_pickle_compat(jax_dir / "params.pkl")

    return cfg_pt, pt_model, cfg_jax, jax_params


def predict_pytorch_grid(model, physics: dict, cfg: dict, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    x_grid, t_grid = np.meshgrid(x, t, indexing="ij")
    xt = np.stack([x_grid.flatten(), t_grid.flatten()], axis=1).astype(np.float32)
    p_vec = np.array([physics["v"], physics["D"], physics["mu"], physics["type"], physics["A"], 0.0, physics["sigma"], physics["k"]], dtype=np.float32)
    p_batch = np.repeat(p_vec[None, :], xt.shape[0], axis=0)
    with torch.no_grad():
        pred = model(torch.tensor(p_batch, dtype=torch.float32), torch.tensor(xt, dtype=torch.float32)).numpy().reshape(len(x), len(t))
    return pred


def predict_jax_grid(params, physics: dict, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    x_grid, t_grid = np.meshgrid(x, t, indexing="ij")
    xt = np.stack([x_grid.flatten(), t_grid.flatten()], axis=1).astype(np.float32)
    p_vec = np.array([physics["v"], physics["D"], physics["mu"], physics["type"], physics["A"], 0.0, physics["sigma"], physics["k"]], dtype=np.float32)
    p_batch = np.repeat(p_vec[None, :], xt.shape[0], axis=0)
    pred = apply_model(params, jnp.asarray(p_batch), jnp.asarray(xt))
    return np.asarray(jax.device_get(pred)).reshape(len(x), len(t))


def compute_cn_grid(physics: dict, cfg: dict, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    ic_params = {"type": physics["type"], "A": physics["A"], "x0": 0.0, "sigma": physics["sigma"], "k": physics["k"]}
    u0 = get_ic_value(x, "mixed", ic_params)
    bc_kind = "tanh_pm1" if physics["type"] == 0 else "zero_zero"
    _, u_cn, _ = crank_nicolson_adr(
        v=physics["v"],
        D=physics["D"],
        mu=physics["mu"],
        xL=cfg["geometry"]["x_min"],
        xR=cfg["geometry"]["x_max"],
        Nx=len(x),
        Tmax=t[-1],
        Nt=len(t),
        bc_kind=bc_kind,
        x0=x,
        u0=u0,
    )
    if u_cn.shape == (len(t), len(x)):
        u_cn = u_cn.T
    return u_cn


def plot_snapshots() -> None:
    cfg_pt, pt_model, _, jax_params = load_models_and_cfg()
    x = np.linspace(cfg_pt["geometry"]["x_min"], cfg_pt["geometry"]["x_max"], 400)
    t = np.linspace(0.0, 1.0, 200)
    t_indices = [0, len(t) // 8, len(t) // 4, len(t) // 2, -1]
    time_colors = ["#f6d0b1", "#f0a06b", "#d1495b", "#8f2d56", "#5e1742"]

    cases = [
        ("Tanh", {"type": 0, "v": 0.8, "D": 0.1, "mu": 0.5, "A": 1.0, "sigma": 0.5, "k": 2.0}),
        ("Sin-Gauss", {"type": 1, "v": 0.8, "D": 0.1, "mu": 0.5, "A": 1.0, "sigma": 0.5, "k": 2.0}),
        ("Gaussian", {"type": 3, "v": 0.8, "D": 0.1, "mu": 0.5, "A": 1.0, "sigma": 0.5, "k": 2.0}),
    ]

    fig, axes = plt.subplots(len(cases), 1, figsize=(12.5, 14.2), sharex=True)
    model_styles = {
        "CN": {"linestyle": "--", "linewidth": 2.6, "alpha": 0.98, "zorder": 6},
        "PyTorch": {"linestyle": "-", "linewidth": 1.9, "alpha": 0.92, "zorder": 3},
        "JAX": {"linestyle": ":", "linewidth": 2.1, "alpha": 0.95, "zorder": 4},
    }

    for row, (family, physics) in enumerate(cases):
        u_cn = compute_cn_grid(physics, cfg_pt, x, t)
        u_pt = predict_pytorch_grid(pt_model, physics, cfg_pt, x, t)
        u_jax = predict_jax_grid(jax_params, physics, x, t)
        ax = axes[row]
        for color, t_idx in zip(time_colors, t_indices):
            label_time = f"t = {t[t_idx]:.2f}"
            ax.plot(x, u_pt[:, t_idx], color=color, label=label_time, **model_styles["PyTorch"])
            ax.plot(x, u_jax[:, t_idx], color=color, **model_styles["JAX"])
            ax.plot(x, u_cn[:, t_idx], color=color, **model_styles["CN"])
        ax.set_title(f"{family} Initial Condition", pad=10)
        ax.set_ylabel("u(x, t)")
        ax.grid(True, alpha=0.22)
        ax.set_ylim(-1.4, 1.4)

    axes[-1].set_xlabel("Space x")
    model_handles = [
        Line2D([0], [0], color="#2b251f", linestyle="--", linewidth=2.6, label="CN"),
        Line2D([0], [0], color="#2b251f", linestyle="-", linewidth=1.9, label="PyTorch"),
        Line2D([0], [0], color="#2b251f", linestyle=":", linewidth=2.1, label="JAX"),
    ]
    time_handles = [
        Line2D([0], [0], color=color, linestyle="-", linewidth=3.0, label=f"t = {t[t_idx]:.2f}")
        for color, t_idx in zip(time_colors, t_indices)
    ]
    fig.subplots_adjust(top=0.93, bottom=0.16, hspace=0.24)
    fig.legend(
        handles=model_handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 0.065),
        columnspacing=2.4,
        handlelength=2.8,
        title="Line style",
    )
    fig.legend(
        handles=time_handles,
        loc="lower center",
        ncol=5,
        frameon=True,
        bbox_to_anchor=(0.5, 0.015),
        columnspacing=1.8,
        handlelength=2.8,
        title="Snapshot times",
    )
    fig.suptitle("Reference and Network Reconstructions for Five Time Snapshots", fontsize=15, fontweight="bold")
    savefig(fig, "6_snapshots_reconstruction.png")


def plot_summary_table(bundle: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 2.8))
    ax.axis("off")
    rows = [
        ["PyTorch", f"{bundle['pytorch']['train']['total_time_sec']/3600:.2f} h", f"{bundle['pytorch']['eval']['global_l2_mean']:.2%}", f"{bundle['pytorch']['infer']['time_jump_sec']:.4f} s"],
        ["JAX", f"{bundle['jax']['train']['total_time_sec']/3600:.2f} h", f"{bundle['jax']['eval']['global_l2_mean']:.2%}", f"{bundle['jax']['infer']['time_jump_sec']:.4f} s"],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["Backend", "Training time", "Global L2", "Time-jump"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.7)
    ax.set_title("Global Summary", pad=14, fontweight="bold")
    savefig(fig, "7_summary_table.png")


def main() -> None:
    setup_style()
    bundle = load_benchmark_bundle()
    plot_family_l2(bundle)
    plot_final_l2_vs_training_time(bundle)
    plot_training_time_hist(bundle)
    plot_inference(bundle)
    plot_threshold_frontier(bundle)
    plot_snapshots()
    plot_summary_table(bundle)
    print(f"Plots written to {ASSET_DIR}")


if __name__ == "__main__":
    main()
