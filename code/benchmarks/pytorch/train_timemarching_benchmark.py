"""PyTorch benchmark entry point `benchmarks.pytorch.train_timemarching_benchmark`.

This script configures the PyTorch benchmark workflow and runs the train timemarching benchmark stage for a selected ADR protocol.
"""

import argparse
import copy
import os
import sys
import time

import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmarks.common.cases import generate_eval_cases
from benchmarks.common.config import build_run_dir, load_yaml
from benchmarks.common.eval import evaluate_cases
from benchmarks.common.io import ensure_dir, save_json
from src.data.generators import generate_mixed_batch
from src.models.PI_DeepOnet_ADR import PI_DeepONet_ADR
from src.training.trainer_ADR import get_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-config", default=os.path.join(PROJECT_ROOT, "benchmarks", "configs", "benchmark_timemarch_t1.yaml"))
    parser.add_argument("--model-config", default=os.path.join(PROJECT_ROOT, "configs", "config_ADR.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def maybe_log(metrics, section, step, total_steps, loss_value, train_start, window_start, log_every, extra=None):
    if step == 1 or step % log_every == 0 or step == total_steps:
        elapsed = time.perf_counter() - train_start
        avg_window = (time.perf_counter() - window_start) / (log_every if step > 1 else 1)
        payload = {
            "section": section,
            "step": step,
            "loss": float(loss_value),
            "elapsed_sec": elapsed,
            "avg_iter_sec_window": avg_window,
        }
        if extra:
            payload.update(extra)
        metrics.append(payload)
        print(
            "[{0} step {1:6d}] loss={2:.6e} | elapsed={3:.2f}s | avg_iter_window={4:.4f}s".format(
                section, step, loss_value, elapsed, avg_window
            )
        )
        return time.perf_counter()
    return window_start


def get_phase_weights(benchmark_cfg, phase_t_max):
    target_w_res = benchmark_cfg["loss_weights"]["first_w_res"]
    if phase_t_max <= 0.3:
        ratio = phase_t_max / 0.3
        w_res = 0.1 + (target_w_res - 0.1) * (ratio * ratio)
        w_ic = benchmark_cfg["loss_weights"]["weight_ic_init"]
    else:
        w_res = target_w_res
        w_ic = benchmark_cfg["loss_weights"]["weight_ic_final"]
    w_bc = benchmark_cfg["loss_weights"]["weight_bc"]
    return w_res, w_ic, w_bc


def quick_audit(model, cfg, benchmark_cfg, phase_t_max, device):
    audit_cfg = copy.deepcopy(benchmark_cfg)
    audit_cfg["training"] = dict(benchmark_cfg["training"])
    audit_cfg["training"]["t_max"] = phase_t_max
    audit_cfg["evaluation"] = dict(benchmark_cfg["evaluation"])
    audit_cfg["evaluation"]["n_cases_per_family"] = benchmark_cfg["timemarch"]["audit_cases_per_family"]

    def predict_grid_fn(p_batch, xt, nx, nt):
        p_tensor = torch.tensor(p_batch, dtype=torch.float32).to(device)
        xt_tensor = torch.tensor(xt, dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = model(p_tensor, xt_tensor).cpu().numpy().reshape(nx, nt)
        return pred

    cases = generate_eval_cases(cfg, audit_cfg)
    return evaluate_cases(cfg, audit_cfg, cases, predict_grid_fn)


def run_phase(model, cfg, benchmark_cfg, phase, device, train_start, metrics):
    phase_name = phase["name"]
    phase_t_max = phase["t_max"]
    retry_cfg = benchmark_cfg["timemarch"]["retry"]
    base_lr = benchmark_cfg["training"]["learning_rate"]
    lr = phase.get("learning_rate", base_lr)
    w_res, w_ic, w_bc = get_phase_weights(benchmark_cfg, phase_t_max)
    best_overall_state = copy.deepcopy(model.state_dict())
    phase_audits = []

    for attempt in range(1, retry_cfg["max_retry"] + 1):
        print("Phase {0} | attempt {1}/{2} | t_max={3} | lr={4:.2e} | wr={5:.2f} wi={6:.2f} wb={7:.2f}".format(
            phase_name, attempt, retry_cfg["max_retry"], phase_t_max, lr, w_res, w_ic, w_bc
        ))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_attempt_state = copy.deepcopy(model.state_dict())
        best_attempt_loss = float("inf")
        window_start = time.perf_counter()

        for step in range(1, phase["iters"] + 1):
            batch = generate_mixed_batch(
                benchmark_cfg["training"]["batch_size"],
                cfg["physics_ranges"],
                cfg["geometry"]["x_min"],
                cfg["geometry"]["x_max"],
                phase_t_max,
                device=device,
            )
            optimizer.zero_grad()
            loss = get_loss(model, batch, w_res, w_ic, w_bc)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            loss_value = loss.item()
            if loss_value < best_attempt_loss:
                best_attempt_loss = loss_value
                best_attempt_state = copy.deepcopy(model.state_dict())
            window_start = maybe_log(
                metrics,
                phase_name,
                step,
                phase["iters"],
                loss_value,
                train_start,
                window_start,
                phase["log_every"],
                extra={"attempt": attempt, "t_max": phase_t_max, "lr": lr, "wr": w_res, "wi": w_ic, "wb": w_bc},
            )

        model.load_state_dict(best_attempt_state)
        audit = quick_audit(model, cfg, benchmark_cfg, phase_t_max, device)
        audit["phase"] = phase_name
        audit["attempt"] = attempt
        audit["lr"] = lr
        phase_audits.append(audit)
        print("Audit {0} attempt {1}: global_l2_mean={2:.4f} target<={3:.4f}".format(
            phase_name, attempt, audit["global_l2_mean"], phase["threshold"]
        ))

        if audit["global_l2_mean"] <= phase["threshold"]:
            return phase_audits

        model.load_state_dict(best_overall_state)
        lr *= retry_cfg["lr_decay"]

    return phase_audits


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
    metrics = []
    phase_audits = []
    train_start = time.perf_counter()
    batch_size = benchmark_cfg["training"]["batch_size"]

    warmup_cfg = benchmark_cfg["timemarch"]["warmup"]
    optimizer = torch.optim.Adam(model.parameters(), lr=benchmark_cfg["training"]["learning_rate"])
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    window_start = train_start

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
        loss_value = loss.item()
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = copy.deepcopy(model.state_dict())
        window_start = maybe_log(
            metrics,
            "warmup_ic",
            step,
            warmup_cfg["iters"],
            loss_value,
            train_start,
            window_start,
            warmup_cfg["log_every"],
        )

    model.load_state_dict(best_state)

    for phase in benchmark_cfg["timemarch"]["phases"]:
        phase_audits.extend(run_phase(model, cfg, benchmark_cfg, phase, device, train_start, metrics))

    total_time = time.perf_counter() - train_start
    payload = {
        "backend": "pytorch",
        "device": str(device),
        "seed": seed,
        "t_max": benchmark_cfg["training"]["t_max"],
        "batch_size": batch_size,
        "total_time_sec": total_time,
        "timemarch": benchmark_cfg["timemarch"],
        "metrics": metrics,
        "phase_audits": phase_audits,
    }
    save_json(os.path.join(run_dir, "train_metrics.json"), payload)
    torch.save({"model_state_dict": model.state_dict()}, os.path.join(run_dir, "model.pt"))
    print("Saved PyTorch timemarch benchmark run to {0}".format(run_dir))


if __name__ == "__main__":
    main()
