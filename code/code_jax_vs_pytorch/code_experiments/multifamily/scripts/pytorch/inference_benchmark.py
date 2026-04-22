import argparse
import os
import sys

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmarks.common.config import build_run_dir, load_yaml
from benchmarks.common.eval import benchmark_inference
from benchmarks.common.io import save_json
from src.models.PI_DeepOnet_ADR import PI_DeepONet_ADR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-config", default=os.path.join(PROJECT_ROOT, "benchmarks", "configs", "benchmark_t1.yaml"))
    parser.add_argument("--model-config", default=os.path.join(PROJECT_ROOT, "configs", "config_ADR.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark_cfg = load_yaml(args.benchmark_config)
    with open(args.model_config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    seed = benchmark_cfg["seed"] if args.seed is None else args.seed
    run_dir = build_run_dir(benchmark_cfg["outputs"]["root_dir"], "pytorch", benchmark_cfg["name"], seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PI_DeepONet_ADR(cfg).to(device)
    checkpoint = torch.load(os.path.join(run_dir, "model.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def build_inputs_fn(scenarios, t_max, nx, nt, full_grid):
        x = np.linspace(cfg["geometry"]["x_min"], cfg["geometry"]["x_max"], nx)
        if full_grid:
            t = np.linspace(0.0, t_max, nt)
            x_grid, t_grid = np.meshgrid(x, t, indexing="ij")
            xt_single = np.stack([x_grid.flatten(), t_grid.flatten()], axis=1).astype(np.float32)
            repeat = xt_single.shape[0]
        else:
            xt_single = np.stack([x, np.full_like(x, t_max)], axis=1).astype(np.float32)
            repeat = xt_single.shape[0]

        xt_batch = np.tile(xt_single, (len(scenarios), 1))
        p_rows = []
        for p in scenarios:
            p_vec = np.array([p["v"], p["D"], p["mu"], p["type"], p["A"], 0.0, p["sigma"], p["k"]], dtype=np.float32)
            p_rows.append(np.tile(p_vec, (repeat, 1)))
        p_batch = np.concatenate(p_rows, axis=0)
        return p_batch, xt_batch

    def predict_fn(p_batch, xt_batch):
        p_tensor = torch.tensor(p_batch, dtype=torch.float32).to(device)
        xt_tensor = torch.tensor(xt_batch, dtype=torch.float32).to(device)
        with torch.no_grad():
            return model(p_tensor, xt_tensor)

    def sync_fn(_result):
        if device.type == "cuda":
            torch.cuda.synchronize()

    result = benchmark_inference(cfg, benchmark_cfg, build_inputs_fn, predict_fn, sync_fn)
    save_json(os.path.join(run_dir, "inference.json"), result)
    print(result)


if __name__ == "__main__":
    main()
