from __future__ import annotations

import json
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


ASSET_DIR = PROJECT_ROOT / "assets_monofamille"
BENCH_ROOT = PROJECT_ROOT / "benchmarks"
SEED = "seed_0"

FAMILY_META = {
    "Tanh": {
        "bench_name": "tanh_only",
        "type_id": 0,
        "physics": {"type": 0, "v": 0.8, "D": 0.1, "mu": 0.5, "A": 1.0, "sigma": 0.5, "k": 2.0},
        "pt_cfg": PROJECT_ROOT / "configs" / "config_ADR_t1_tanh_only.yaml",
        "jax_cfg": PROJECT_ROOT / "configs_jax" / "config_ADR_jax_t1_tanh_only.yaml",
    },
    "Sin-Gauss": {
        "bench_name": "singauss_only",
        "type_id": 1,
        "physics": {"type": 1, "v": 0.8, "D": 0.1, "mu": 0.5, "A": 1.0, "sigma": 0.5, "k": 2.0},
        "pt_cfg": PROJECT_ROOT / "configs" / "config_ADR_t1_singauss_only.yaml",
        "jax_cfg": PROJECT_ROOT / "configs_jax" / "config_ADR_jax_t1_singauss_only.yaml",
    },
    "Gaussian": {
        "bench_name": "gaussian_only",
        "type_id": 3,
        "physics": {"type": 3, "v": 0.8, "D": 0.1, "mu": 0.5, "A": 1.0, "sigma": 0.5, "k": 2.0},
        "pt_cfg": PROJECT_ROOT / "configs" / "config_ADR_t1_gaussian_only.yaml",
        "jax_cfg": PROJECT_ROOT / "configs_jax" / "config_ADR_jax_t1_gaussian_only.yaml",
    },
}


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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


def resolve_run_dir(backend: str, suffix: str) -> Path:
    path = BENCH_ROOT / backend / f"benchmark_fulltrainer_t1_{suffix}" / SEED
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_bundle() -> dict:
    bundle = {}
    for family, meta in FAMILY_META.items():
        suffix = meta["bench_name"]
        bundle[family] = {
            "pytorch": {
                "run_dir": resolve_run_dir("pytorch", suffix),
                "train": load_json(resolve_run_dir("pytorch", suffix) / "train_metrics.json"),
                "eval": load_json(resolve_run_dir("pytorch", suffix) / "evaluation.json"),
                "infer": load_json(resolve_run_dir("pytorch", suffix) / "inference.json"),
            },
            "jax": {
                "run_dir": resolve_run_dir("jax", suffix),
                "train": load_json(resolve_run_dir("jax", suffix) / "train_metrics.json"),
                "eval": load_json(resolve_run_dir("jax", suffix) / "evaluation.json"),
                "infer": load_json(resolve_run_dir("jax", suffix) / "inference.json"),
            },
        }
    return bundle


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
    families = list(FAMILY_META.keys())
    x = np.arange(len(families))
    width = 0.36
    colors = {"pytorch": "#d1495b", "jax": "#00798c"}

    fig, ax = plt.subplots(figsize=(10.5, 6))
    for backend, offset in [("pytorch", -width / 2), ("jax", width / 2)]:
        means = [bundle[family][backend]["eval"]["global_l2_mean"] for family in families]
        stds = [bundle[family][backend]["eval"]["global_l2_std"] for family in families]
        bars = ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=8,
            alpha=0.9,
            color=colors[backend],
            label="PyTorch" if backend == "pytorch" else "JAX",
        )
        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.10,
                f"{value:.2%}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.axhline(0.05, color="#222222", linestyle="--", linewidth=1.2, label="5% threshold")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 2e1)
    ax.set_xticks(x, families)
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Monofamily Relative L2 Error by Backend")
    ax.legend(frameon=True)
    savefig(fig, "1_monofamily_l2.png")


def plot_training_time(bundle: dict) -> None:
    families = list(FAMILY_META.keys())
    x = np.arange(len(families))
    width = 0.36
    colors = {"pytorch": "#d1495b", "jax": "#00798c"}

    fig, ax = plt.subplots(figsize=(10.5, 6))
    pt_vals = [bundle[family]["pytorch"]["train"]["total_time_sec"] / 60.0 for family in families]
    jax_vals = [bundle[family]["jax"]["train"]["total_time_sec"] / 60.0 for family in families]

    pt_bars = ax.bar(x - width / 2, pt_vals, width=width, color=colors["pytorch"], alpha=0.9, label="PyTorch")
    jax_bars = ax.bar(x + width / 2, jax_vals, width=width, color=colors["jax"], alpha=0.9, label="JAX")

    for idx, (bar, value) in enumerate(zip(pt_bars, pt_vals)):
        label = f"{value:.1f} min"
        if families[idx] == "Tanh":
            label += " *"
        ax.text(bar.get_x() + bar.get_width() / 2, value, label, ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, value in zip(jax_bars, jax_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f} min", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x, families)
    ax.set_ylabel("Training time (minutes)")
    ax.set_title("Monofamily Training Time by Backend")
    ax.legend(frameon=True)
    ax.text(0.99, 0.98, "* resumed PyTorch run", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    savefig(fig, "2_monofamily_training_time.png")


def plot_inference(bundle: dict) -> None:
    families = list(FAMILY_META.keys())
    x = np.arange(len(families))
    width = 0.24
    colors = {"pytorch": "#d1495b", "jax": "#00798c", "cn": "#2f2f2f"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    for ax, key, title in [
        (axes[0], "inference_full_grid_sec", "Full-grid inference"),
        (axes[1], "time_jump_sec", "Time-jump inference"),
    ]:
        pt_vals = [bundle[family]["pytorch"]["infer"][key] for family in families]
        jax_vals = [bundle[family]["jax"]["infer"][key] for family in families]
        cn_vals = [0.5 * (bundle[family]["pytorch"]["infer"]["cn_reference_sec"] + bundle[family]["jax"]["infer"]["cn_reference_sec"]) for family in families]

        ax.bar(x - width, pt_vals, width=width, color=colors["pytorch"], alpha=0.9, label="PyTorch")
        ax.bar(x, jax_vals, width=width, color=colors["jax"], alpha=0.9, label="JAX")
        ax.bar(x + width, cn_vals, width=width, color=colors["cn"], alpha=0.9, label="CN")
        ax.set_xticks(x, families)
        ax.set_ylabel("Time (s)")
        ax.set_title(title)

    axes[0].legend(frameon=True)
    savefig(fig, "3_monofamily_inference.png")


def plot_frontier(bundle: dict) -> None:
    colors = {"pytorch": "#d1495b", "jax": "#00798c"}
    markers = {"Tanh": "o", "Sin-Gauss": "s", "Gaussian": "D"}

    fig, ax = plt.subplots(figsize=(9.5, 6))
    for family in FAMILY_META:
        for backend in ["pytorch", "jax"]:
            x = bundle[family][backend]["train"]["total_time_sec"] / 3600.0
            y = bundle[family][backend]["eval"]["global_l2_mean"]
            ax.scatter(
                x,
                y,
                s=180,
                marker=markers[family],
                color=colors[backend],
                edgecolors="#1f1f1f",
                linewidths=0.8,
                alpha=0.95,
            )
            ax.annotate(
                f"{backend[:2].upper()} {family}",
                (x, y),
                textcoords="offset points",
                xytext=(8, 6),
                fontsize=9,
            )

    ax.axhline(0.05, color="#222222", linestyle="--", linewidth=1.3, label="5% threshold")
    ax.set_yscale("log")
    ax.set_xlabel("Training time (hours)")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Monofamily Quality vs Training Time")
    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d1495b", markeredgecolor="#1f1f1f", markersize=10, label="PyTorch"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#00798c", markeredgecolor="#1f1f1f", markersize=10, label="JAX"),
        Line2D([0], [0], color="#222222", linestyle="--", label="5% threshold"),
    ]
    ax.legend(handles=legend_items, frameon=True)
    savefig(fig, "4_monofamily_frontier.png")


def load_models():
    models = {}
    for family, meta in FAMILY_META.items():
        cfg_pt = load_yaml(meta["pt_cfg"])
        cfg_jax = load_yaml(meta["jax_cfg"])

        pt_model = PI_DeepONet_ADR(cfg_pt).to(torch.device("cpu"))
        pt_ckpt = torch.load(resolve_run_dir("pytorch", meta["bench_name"]) / "model.pt", map_location="cpu")
        pt_model.load_state_dict(pt_ckpt["model_state_dict"])
        pt_model.eval()

        jax_params = load_jax_pickle_compat(resolve_run_dir("jax", meta["bench_name"]) / "params.pkl")
        models[family] = {"pt_cfg": cfg_pt, "pt_model": pt_model, "jax_cfg": cfg_jax, "jax_params": jax_params}
    return models


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


def plot_snapshots(bundle: dict) -> None:
    models = load_models()
    time_colors = ["#f6d0b1", "#f0a06b", "#d1495b", "#8f2d56", "#5e1742"]
    t = np.linspace(0.0, 1.0, 200)
    t_indices = [0, len(t) // 8, len(t) // 4, len(t) // 2, -1]

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 14.2), sharex=True)
    model_styles = {
        "CN": {"linestyle": "--", "linewidth": 2.6, "alpha": 0.98, "zorder": 6},
        "PyTorch": {"linestyle": "-", "linewidth": 1.9, "alpha": 0.92, "zorder": 3},
        "JAX": {"linestyle": ":", "linewidth": 2.1, "alpha": 0.95, "zorder": 4},
    }

    for ax, family in zip(axes, FAMILY_META):
        meta = FAMILY_META[family]
        cfg_pt = models[family]["pt_cfg"]
        pt_model = models[family]["pt_model"]
        jax_params = models[family]["jax_params"]
        x = np.linspace(cfg_pt["geometry"]["x_min"], cfg_pt["geometry"]["x_max"], 400)
        physics = meta["physics"]
        u_cn = compute_cn_grid(physics, cfg_pt, x, t)
        u_pt = predict_pytorch_grid(pt_model, physics, cfg_pt, x, t)
        u_jax = predict_jax_grid(jax_params, physics, x, t)

        for color, t_idx in zip(time_colors, t_indices):
            ax.plot(x, u_pt[:, t_idx], color=color, label=f"t = {t[t_idx]:.2f}", **model_styles["PyTorch"])
            ax.plot(x, u_jax[:, t_idx], color=color, **model_styles["JAX"])
            ax.plot(x, u_cn[:, t_idx], color=color, **model_styles["CN"])

        ax.set_title(f"{family} Reconstruction vs Reference")
        ax.set_ylabel("u(x, t)")
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
    fig.suptitle("Monofamily Reconstructions for Five Time Snapshots", fontsize=15, fontweight="bold")
    savefig(fig, "5_monofamily_snapshots.png")


def plot_summary_table(bundle: dict) -> None:
    rows = []
    for family in FAMILY_META:
        rows.append(
            [
                family,
                f"{bundle[family]['pytorch']['eval']['global_l2_mean']:.2%}",
                f"{bundle[family]['jax']['eval']['global_l2_mean']:.2%}",
                f"{bundle[family]['pytorch']['train']['total_time_sec']/60.0:.1f}",
                f"{bundle[family]['jax']['train']['total_time_sec']/60.0:.1f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Family", "PyTorch L2", "JAX L2", "PyTorch train (min)", "JAX train (min)"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.7)
    ax.set_title("Monofamily Summary", pad=14, fontweight="bold")
    savefig(fig, "6_monofamily_summary.png")


def main() -> None:
    setup_style()
    bundle = load_bundle()
    plot_family_l2(bundle)
    plot_training_time(bundle)
    plot_inference(bundle)
    plot_frontier(bundle)
    plot_snapshots(bundle)
    plot_summary_table(bundle)
    print(f"Plots written to {ASSET_DIR}")


if __name__ == "__main__":
    main()
