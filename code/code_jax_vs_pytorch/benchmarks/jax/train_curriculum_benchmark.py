import argparse
import os
import sys
import time

import jax
import optax
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmarks.common.config import build_run_dir, load_yaml
from benchmarks.common.io import ensure_dir, save_json, save_pickle
from src_jax.data.generators import generate_mixed_batch
from src_jax.models.pi_deeponet_adr import init_model_params
from src_jax.training.step import make_ic_train_step, make_train_step


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-config", default=os.path.join(PROJECT_ROOT, "benchmarks", "configs", "benchmark_curriculum_t1.yaml"))
    parser.add_argument("--model-config", default=os.path.join(PROJECT_ROOT, "configs_jax", "config_ADR_jax.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def maybe_log(metrics, section, step, total_steps, loss_value, train_start, window_start, log_every):
    if step == 1 or step % log_every == 0 or step == total_steps:
        elapsed = time.perf_counter() - train_start
        avg_window = (time.perf_counter() - window_start) / (log_every if step > 1 else 1)
        metrics.append(
            {
                "section": section,
                "step": step,
                "loss": float(loss_value),
                "elapsed_sec": elapsed,
                "avg_iter_sec_window": avg_window,
            }
        )
        print(
            "[{0} step {1:6d}] loss={2:.6e} | elapsed={3:.2f}s | avg_iter_window={4:.4f}s".format(
                section, step, loss_value, elapsed, avg_window
            )
        )
        return time.perf_counter()
    return window_start


def main():
    args = parse_args()
    benchmark_cfg = load_yaml(args.benchmark_config)
    with open(args.model_config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    seed = benchmark_cfg["seed"] if args.seed is None else args.seed
    run_dir = build_run_dir(benchmark_cfg["outputs"]["root_dir"], "jax", benchmark_cfg["name"], seed)
    ensure_dir(run_dir)

    key = jax.random.PRNGKey(seed)
    model_key, key = jax.random.split(key)
    params = init_model_params(model_key, cfg)
    optimizer = optax.adam(benchmark_cfg["training"]["learning_rate"])
    opt_state = optimizer.init(params)
    ic_train_step = make_ic_train_step(optimizer)
    train_step = make_train_step(optimizer)
    weights = benchmark_cfg["training"]["loss_weights"]

    metrics = []
    train_start = time.perf_counter()
    window_start = train_start
    batch_size = benchmark_cfg["training"]["batch_size"]

    warmup_cfg = benchmark_cfg["curriculum"]["warmup"]
    for step in range(1, warmup_cfg["iters"] + 1):
        key, batch_key = jax.random.split(key)
        batch = generate_mixed_batch(
            batch_key,
            batch_size,
            cfg["physics_ranges"],
            cfg["geometry"]["x_min"],
            cfg["geometry"]["x_max"],
            0.0,
        )
        params, opt_state, loss = ic_train_step(params, opt_state, batch)
        loss = float(jax.device_get(loss))
        window_start = maybe_log(
            metrics,
            "warmup_ic",
            step,
            warmup_cfg["iters"],
            loss,
            train_start,
            window_start,
            warmup_cfg["log_every"],
        )

    for phase in benchmark_cfg["curriculum"]["phases"]:
        phase_name = phase["name"]
        for step in range(1, phase["iters"] + 1):
            key, batch_key = jax.random.split(key)
            batch = generate_mixed_batch(
                batch_key,
                batch_size,
                cfg["physics_ranges"],
                cfg["geometry"]["x_min"],
                cfg["geometry"]["x_max"],
                phase["t_max"],
            )
            params, opt_state, loss = train_step(params, opt_state, batch, weights["wr"], weights["wi"], weights["wb"])
            loss = float(jax.device_get(loss))
            window_start = maybe_log(
                metrics,
                phase_name,
                step,
                phase["iters"],
                loss,
                train_start,
                window_start,
                phase["log_every"],
            )

    total_time = time.perf_counter() - train_start
    payload = {
        "backend": "jax",
        "device": str(jax.devices()[0]),
        "seed": seed,
        "t_max": benchmark_cfg["training"]["t_max"],
        "batch_size": batch_size,
        "total_time_sec": total_time,
        "curriculum": benchmark_cfg["curriculum"],
        "metrics": metrics,
    }
    save_json(os.path.join(run_dir, "train_metrics.json"), payload)
    save_pickle(os.path.join(run_dir, "params.pkl"), params)
    print("Saved JAX curriculum benchmark run to {0}".format(run_dir))


if __name__ == "__main__":
    main()
