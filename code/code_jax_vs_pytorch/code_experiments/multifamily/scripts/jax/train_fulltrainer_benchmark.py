import argparse
import json
import os
import sys
import time

import jax
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmarks.common.config import build_run_dir, load_yaml
from benchmarks.common.io import ensure_dir, save_json, save_pickle
from src_jax.models.pi_deeponet_adr import init_model_params
from src_jax.training.trainer_ADR_JAX import train_smart_time_marching


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-config", default=os.path.join(PROJECT_ROOT, "benchmarks", "configs", "benchmark_fulltrainer_t1.yaml"))
    parser.add_argument("--model-config", default=os.path.join(PROJECT_ROOT, "configs_jax", "config_ADR_jax_t1_equal_pipeline.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark_cfg = load_yaml(args.benchmark_config)
    with open(args.model_config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    seed = benchmark_cfg["seed"] if args.seed is None else args.seed
    run_dir = build_run_dir(benchmark_cfg["outputs"]["root_dir"], "jax", benchmark_cfg["name"], seed)
    ensure_dir(run_dir)
    cfg["audit"]["save_dir"] = run_dir

    key = jax.random.PRNGKey(seed)
    params = init_model_params(key, cfg)
    train_start = time.perf_counter()
    params, trainer_metrics = train_smart_time_marching(params, cfg, cfg["physics_ranges"])
    total_time = time.perf_counter() - train_start

    save_pickle(os.path.join(run_dir, "params.pkl"), params)
    payload = {
        "backend": "jax",
        "seed": seed,
        "device": str(jax.devices()[0]),
        "t_max": cfg["geometry"]["T_max"],
        "config_path": args.model_config,
        "total_time_sec": total_time,
        "trainer_metrics": trainer_metrics,
    }
    save_json(os.path.join(run_dir, "train_metrics.json"), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
