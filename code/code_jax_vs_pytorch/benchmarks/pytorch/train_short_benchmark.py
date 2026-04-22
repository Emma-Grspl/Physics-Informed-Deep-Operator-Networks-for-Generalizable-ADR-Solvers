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
    parser.add_argument("--benchmark-config", default=os.path.join(PROJECT_ROOT, "benchmarks", "configs", "benchmark_t1.yaml"))
    parser.add_argument("--model-config", default=os.path.join(PROJECT_ROOT, "configs", "config_ADR.yaml"))
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

    for step in range(1, benchmark_cfg["training"]["iters"] + 1):
        batch = generate_mixed_batch(
            benchmark_cfg["training"]["batch_size"],
            cfg["physics_ranges"],
            cfg["geometry"]["x_min"],
            cfg["geometry"]["x_max"],
            benchmark_cfg["training"]["t_max"],
            device=device,
        )
        optimizer.zero_grad()
        step_start = time.perf_counter()
        loss = get_loss(model, batch, weights["wr"], weights["wi"], weights["wb"])
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_duration = time.perf_counter() - step_start

        if step == 1 or step % benchmark_cfg["training"]["log_every"] == 0 or step == benchmark_cfg["training"]["iters"]:
            elapsed = time.perf_counter() - train_start
            avg_window = (time.perf_counter() - window_start) / (benchmark_cfg["training"]["log_every"] if step > 1 else 1)
            metrics.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "elapsed_sec": elapsed,
                    "avg_iter_sec_window": avg_window,
                    "last_step_sec": step_duration,
                }
            )
            print(
                "[step {0:6d}] loss={1:.6e} | elapsed={2:.2f}s | avg_iter_window={3:.4f}s".format(
                    step, loss.item(), elapsed, avg_window
                )
            )
            window_start = time.perf_counter()

    total_time = time.perf_counter() - train_start
    metrics_payload = {
        "backend": "pytorch",
        "device": str(device),
        "seed": seed,
        "t_max": benchmark_cfg["training"]["t_max"],
        "iters": benchmark_cfg["training"]["iters"],
        "batch_size": benchmark_cfg["training"]["batch_size"],
        "total_time_sec": total_time,
        "avg_iter_sec_total": total_time / benchmark_cfg["training"]["iters"],
        "metrics": metrics,
    }
    save_json(os.path.join(run_dir, "train_metrics.json"), metrics_payload)
    torch.save({"model_state_dict": model.state_dict()}, os.path.join(run_dir, "model.pt"))
    print("Saved PyTorch benchmark run to {0}".format(run_dir))


if __name__ == "__main__":
    main()
