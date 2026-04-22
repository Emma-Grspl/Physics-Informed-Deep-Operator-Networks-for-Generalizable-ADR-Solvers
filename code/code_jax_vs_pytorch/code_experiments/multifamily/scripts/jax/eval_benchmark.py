import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmarks.common.cases import generate_eval_cases
from benchmarks.common.config import build_run_dir, load_yaml
from benchmarks.common.eval import evaluate_cases
from benchmarks.common.io import load_pickle, save_json
from src_jax.models.pi_deeponet_adr import apply_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-config", default=os.path.join(PROJECT_ROOT, "benchmarks", "configs", "benchmark_t1.yaml"))
    parser.add_argument("--model-config", default=os.path.join(PROJECT_ROOT, "configs_jax", "config_ADR_jax.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark_cfg = load_yaml(args.benchmark_config)
    with open(args.model_config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    seed = benchmark_cfg["seed"] if args.seed is None else args.seed
    run_dir = build_run_dir(benchmark_cfg["outputs"]["root_dir"], "jax", benchmark_cfg["name"], seed)
    params = load_pickle(os.path.join(run_dir, "params.pkl"))

    def predict_grid_fn(p_batch, xt, nx, nt):
        pred = apply_model(params, jnp.asarray(p_batch), jnp.asarray(xt))
        return np.asarray(jax.device_get(pred)).reshape(nx, nt)

    cases = generate_eval_cases(cfg, benchmark_cfg)
    result = evaluate_cases(cfg, benchmark_cfg, cases, predict_grid_fn)
    save_json(os.path.join(run_dir, "evaluation.json"), result)
    print(result)


if __name__ == "__main__":
    main()
