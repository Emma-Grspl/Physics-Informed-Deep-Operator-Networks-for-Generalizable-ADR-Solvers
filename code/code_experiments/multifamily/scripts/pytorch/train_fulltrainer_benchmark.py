"""Comparison-layer module `jax_comparison.multifamily.scripts.pytorch.train_fulltrainer_benchmark`.

This file supports the PyTorch versus JAX comparison workflows, including replicated implementations, scripts, or validation helpers.
"""

import argparse
import json
import os
import sys
import time

import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmarks.common.config import build_run_dir, load_yaml
from benchmarks.common.io import ensure_dir, save_json
import src.training.trainer_ADR_benchmark as trainer_module
from src.models.PI_DeepOnet_ADR import PI_DeepONet_ADR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-config", default=os.path.join(PROJECT_ROOT, "benchmarks", "configs", "benchmark_fulltrainer_t1.yaml"))
    parser.add_argument("--model-config", default=os.path.join(PROJECT_ROOT, "configs", "config_ADR_t1_compare.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark_cfg = load_yaml(args.benchmark_config)
    with open(args.model_config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    seed = benchmark_cfg["seed"] if args.seed is None else args.seed
    run_dir = build_run_dir(benchmark_cfg["outputs"]["root_dir"], "pytorch", benchmark_cfg["name"], seed)
    ensure_dir(run_dir)

    torch.manual_seed(seed)
    trainer_module.cfg = cfg
    trainer_module.cfg["audit"]["save_dir"] = run_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PI_DeepONet_ADR(cfg).to(device)
    train_start = time.perf_counter()
    try:
        model = trainer_module.train_smart_time_marching(model, bounds=cfg["physics_ranges"])
    except Exception:
        torch.save(model.state_dict(), os.path.join(run_dir, "model_CRASHED.pth"))
        raise
    total_time = time.perf_counter() - train_start

    torch.save({"model_state_dict": model.state_dict()}, os.path.join(run_dir, "model.pt"))
    payload = {
        "backend": "pytorch",
        "seed": seed,
        "device": str(device),
        "t_max": cfg["geometry"]["T_max"],
        "config_path": args.model_config,
        "total_time_sec": total_time,
    }
    save_json(os.path.join(run_dir, "train_metrics.json"), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
