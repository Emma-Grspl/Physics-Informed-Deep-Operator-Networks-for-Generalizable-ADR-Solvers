"""JAX benchmark entry point `benchmarks.jax.train_short_benchmark`.

This script configures the JAX benchmark workflow and runs the train short benchmark stage for a selected ADR protocol.
"""

import argparse
import os
import sys
import time

import jax
import yaml
import optax

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmarks.common.config import build_run_dir, load_yaml
from benchmarks.common.io import ensure_dir, save_json, save_pickle
from src_jax.data.generators import generate_mixed_batch
from src_jax.models.pi_deeponet_adr import init_model_params
from src_jax.training.step import make_train_step


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-config", default=os.path.join(PROJECT_ROOT, "benchmarks", "configs", "benchmark_t1.yaml"))
    parser.add_argument("--model-config", default=os.path.join(PROJECT_ROOT, "configs_jax", "config_ADR_jax.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark_cfg = load_yaml(args.benchmark_config)
    with open(args.model_config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    if args.iters is not None:
        benchmark_cfg["training"]["iters"] = args.iters
    if args.batch_size is not None:
        benchmark_cfg["training"]["batch_size"] = args.batch_size
    if args.log_every is not None:
        benchmark_cfg["training"]["log_every"] = args.log_every

    seed = benchmark_cfg["seed"] if args.seed is None else args.seed
    run_dir = build_run_dir(benchmark_cfg["outputs"]["root_dir"], "jax", benchmark_cfg["name"], seed)
    ensure_dir(run_dir)

    key = jax.random.PRNGKey(seed)
    model_key, key = jax.random.split(key)
    params = init_model_params(model_key, cfg)
    optimizer = optax.adam(benchmark_cfg["training"]["learning_rate"])
    opt_state = optimizer.init(params)
    train_step = make_train_step(optimizer)

    weights = benchmark_cfg["training"]["loss_weights"]
    metrics = []
    train_start = time.perf_counter()
    window_start = train_start

    for step in range(1, benchmark_cfg["training"]["iters"] + 1):
        key, batch_key = jax.random.split(key)
        batch = generate_mixed_batch(
            batch_key,
            benchmark_cfg["training"]["batch_size"],
            cfg["physics_ranges"],
            cfg["geometry"]["x_min"],
            cfg["geometry"]["x_max"],
            benchmark_cfg["training"]["t_max"],
        )
        step_start = time.perf_counter()
        params, opt_state, loss = train_step(params, opt_state, batch, weights["wr"], weights["wi"], weights["wb"])
        loss = float(jax.device_get(loss))
        step_duration = time.perf_counter() - step_start

        if step == 1 or step % benchmark_cfg["training"]["log_every"] == 0 or step == benchmark_cfg["training"]["iters"]:
            elapsed = time.perf_counter() - train_start
            avg_window = (time.perf_counter() - window_start) / (benchmark_cfg["training"]["log_every"] if step > 1 else 1)
            metrics.append(
                {
                    "step": step,
                    "loss": loss,
                    "elapsed_sec": elapsed,
                    "avg_iter_sec_window": avg_window,
                    "last_step_sec": step_duration,
                }
            )
            print(
                "[step {0:6d}] loss={1:.6e} | elapsed={2:.2f}s | avg_iter_window={3:.4f}s".format(
                    step, loss, elapsed, avg_window
                )
            )
            window_start = time.perf_counter()

    total_time = time.perf_counter() - train_start
    metrics_payload = {
        "backend": "jax",
        "device": str(jax.devices()[0]),
        "seed": seed,
        "t_max": benchmark_cfg["training"]["t_max"],
        "iters": benchmark_cfg["training"]["iters"],
        "batch_size": benchmark_cfg["training"]["batch_size"],
        "total_time_sec": total_time,
        "avg_iter_sec_total": total_time / benchmark_cfg["training"]["iters"],
        "metrics": metrics,
    }
    save_json(os.path.join(run_dir, "train_metrics.json"), metrics_payload)
    save_pickle(os.path.join(run_dir, "params.pkl"), params)
    print("Saved JAX benchmark run to {0}".format(run_dir))


if __name__ == "__main__":
    main()
