import argparse
import os
import sys

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmarks.common.cases import generate_eval_cases
from benchmarks.common.config import build_run_dir, load_yaml
from benchmarks.common.eval import evaluate_cases
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

    def predict_grid_fn(p_batch, xt, nx, nt):
        p_tensor = torch.tensor(p_batch, dtype=torch.float32).to(device)
        xt_tensor = torch.tensor(xt, dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = model(p_tensor, xt_tensor).cpu().numpy().reshape(nx, nt)
        return pred

    cases = generate_eval_cases(cfg, benchmark_cfg)
    result = evaluate_cases(cfg, benchmark_cfg, cases, predict_grid_fn)
    save_json(os.path.join(run_dir, "evaluation.json"), result)
    print(result)


if __name__ == "__main__":
    main()
