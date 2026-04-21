"""PyTorch benchmark entry point `benchmarks.pytorch.train_curriculum_benchmark`.

This script configures the PyTorch benchmark workflow and runs the train curriculum benchmark stage for a selected ADR protocol.
"""

import argparse
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
from src.data.generators import generate_mixed_batch
from src.models.PI_DeepOnet_ADR import PI_DeepONet_ADR
from src.training.trainer_ADR import get_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-config", default=os.path.join(PROJECT_ROOT, "benchmarks", "configs", "benchmark_curriculum_t1.yaml"))
    parser.add_argument("--model-config", default=os.path.join(PROJECT_ROOT, "configs", "config_ADR.yaml"))
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
    run_dir = build_run_dir(benchmark_cfg["outputs"]["root_dir"], "pytorch", benchmark_cfg["name"], seed)
    ensure_dir(run_dir)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PI_DeepONet_ADR(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=benchmark_cfg["training"]["learning_rate"])
    weights = benchmark_cfg["training"]["loss_weights"]

    metrics = []
    train_start = time.perf_counter()
    window_start = train_start
    batch_size = benchmark_cfg["training"]["batch_size"]

    warmup_cfg = benchmark_cfg["curriculum"]["warmup"]
    for step in range(1, warmup_cfg["iters"] + 1):
        batch = generate_mixed_batch(
            batch_size,
            cfg["physics_ranges"],
            cfg["geometry"]["x_min"],
            cfg["geometry"]["x_max"],
            0.0,
            device=device,
        )
        params, _, xt_ic, u_true_ic, _, _, _, _ = batch
        optimizer.zero_grad()
        loss = torch.mean((model(params, xt_ic) - u_true_ic) ** 2)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        window_start = maybe_log(
            metrics,
            "warmup_ic",
            step,
            warmup_cfg["iters"],
            loss.item(),
            train_start,
            window_start,
            warmup_cfg["log_every"],
        )

    for phase in benchmark_cfg["curriculum"]["phases"]:
        phase_name = phase["name"]
        for step in range(1, phase["iters"] + 1):
            batch = generate_mixed_batch(
                batch_size,
                cfg["physics_ranges"],
                cfg["geometry"]["x_min"],
                cfg["geometry"]["x_max"],
                phase["t_max"],
                device=device,
            )
            optimizer.zero_grad()
            loss = get_loss(model, batch, weights["wr"], weights["wi"], weights["wb"])
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            window_start = maybe_log(
                metrics,
                phase_name,
                step,
                phase["iters"],
                loss.item(),
                train_start,
                window_start,
                phase["log_every"],
            )

    total_time = time.perf_counter() - train_start
    payload = {
        "backend": "pytorch",
        "device": str(device),
        "seed": seed,
        "t_max": benchmark_cfg["training"]["t_max"],
        "batch_size": batch_size,
        "total_time_sec": total_time,
        "curriculum": benchmark_cfg["curriculum"],
        "metrics": metrics,
    }
    save_json(os.path.join(run_dir, "train_metrics.json"), payload)
    torch.save({"model_state_dict": model.state_dict()}, os.path.join(run_dir, "model.pt"))
    print("Saved PyTorch curriculum benchmark run to {0}".format(run_dir))


if __name__ == "__main__":
    main()
