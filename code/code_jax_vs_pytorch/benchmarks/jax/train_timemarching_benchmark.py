import argparse
import copy
import os
import sys
import time

import jax
import optax
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmarks.common.cases import generate_eval_cases
from benchmarks.common.config import build_run_dir, load_yaml
from benchmarks.common.eval import evaluate_cases
from benchmarks.common.io import ensure_dir, save_json, save_pickle
from src_jax.data.generators import generate_mixed_batch
from src_jax.models.pi_deeponet_adr import apply_model, init_model_params
from src_jax.training.step import make_ic_train_step, make_train_step


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-config", default=os.path.join(PROJECT_ROOT, "benchmarks", "configs", "benchmark_timemarch_t1.yaml"))
    parser.add_argument("--model-config", default=os.path.join(PROJECT_ROOT, "configs_jax", "config_ADR_jax.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def clone_params(params):
    return jax.tree_util.tree_map(lambda x: x.copy(), params)


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


def quick_audit(params, cfg, benchmark_cfg, phase_t_max):
    audit_cfg = copy.deepcopy(benchmark_cfg)
    audit_cfg["training"] = dict(benchmark_cfg["training"])
    audit_cfg["training"]["t_max"] = phase_t_max
    audit_cfg["evaluation"] = dict(benchmark_cfg["evaluation"])
    audit_cfg["evaluation"]["n_cases_per_family"] = benchmark_cfg["timemarch"]["audit_cases_per_family"]

    def predict_grid_fn(p_batch, xt, nx, nt):
        pred = apply_model(params, jax.numpy.asarray(p_batch), jax.numpy.asarray(xt))
        return jax.device_get(pred).reshape(nx, nt)

    cases = generate_eval_cases(cfg, audit_cfg)
    return evaluate_cases(cfg, audit_cfg, cases, predict_grid_fn)


def run_phase(params, key, cfg, benchmark_cfg, phase, train_start, metrics):
    phase_name = phase["name"]
    phase_t_max = phase["t_max"]
    retry_cfg = benchmark_cfg["timemarch"]["retry"]
    base_lr = benchmark_cfg["training"]["learning_rate"]
    lr = phase.get("learning_rate", base_lr)
    w_res, w_ic, w_bc = get_phase_weights(benchmark_cfg, phase_t_max)
    best_overall_params = clone_params(params)
    phase_audits = []

    for attempt in range(1, retry_cfg["max_retry"] + 1):
        print("Phase {0} | attempt {1}/{2} | t_max={3} | lr={4:.2e} | wr={5:.2f} wi={6:.2f} wb={7:.2f}".format(
            phase_name, attempt, retry_cfg["max_retry"], phase_t_max, lr, w_res, w_ic, w_bc
        ))
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)
        train_step = make_train_step(optimizer)
        best_attempt_params = clone_params(params)
        best_attempt_loss = float("inf")
        window_start = time.perf_counter()

        for step in range(1, phase["iters"] + 1):
            key, batch_key = jax.random.split(key)
            batch = generate_mixed_batch(
                batch_key,
                benchmark_cfg["training"]["batch_size"],
                cfg["physics_ranges"],
                cfg["geometry"]["x_min"],
                cfg["geometry"]["x_max"],
                phase_t_max,
            )
            params, opt_state, loss = train_step(params, opt_state, batch, w_res, w_ic, w_bc)
            loss_value = float(jax.device_get(loss))
            if loss_value < best_attempt_loss:
                best_attempt_loss = loss_value
                best_attempt_params = clone_params(params)
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

        params = clone_params(best_attempt_params)
        audit = quick_audit(params, cfg, benchmark_cfg, phase_t_max)
        audit["phase"] = phase_name
        audit["attempt"] = attempt
        audit["lr"] = lr
        phase_audits.append(audit)
        print("Audit {0} attempt {1}: global_l2_mean={2:.4f} target<={3:.4f}".format(
            phase_name, attempt, audit["global_l2_mean"], phase["threshold"]
        ))

        if audit["global_l2_mean"] <= phase["threshold"]:
            return params, key, phase_audits

        params = clone_params(best_overall_params)
        lr *= retry_cfg["lr_decay"]

    return params, key, phase_audits


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
    metrics = []
    phase_audits = []
    train_start = time.perf_counter()
    batch_size = benchmark_cfg["training"]["batch_size"]

    warmup_cfg = benchmark_cfg["timemarch"]["warmup"]
    optimizer = optax.adam(benchmark_cfg["training"]["learning_rate"])
    opt_state = optimizer.init(params)
    ic_train_step = make_ic_train_step(optimizer)
    best_params = clone_params(params)
    best_loss = float("inf")
    window_start = train_start

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
        loss_value = float(jax.device_get(loss))
        if loss_value < best_loss:
            best_loss = loss_value
            best_params = clone_params(params)
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

    params = clone_params(best_params)

    for phase in benchmark_cfg["timemarch"]["phases"]:
        params, key, audits = run_phase(params, key, cfg, benchmark_cfg, phase, train_start, metrics)
        phase_audits.extend(audits)

    total_time = time.perf_counter() - train_start
    payload = {
        "backend": "jax",
        "device": str(jax.devices()[0]),
        "seed": seed,
        "t_max": benchmark_cfg["training"]["t_max"],
        "batch_size": batch_size,
        "total_time_sec": total_time,
        "timemarch": benchmark_cfg["timemarch"],
        "metrics": metrics,
        "phase_audits": phase_audits,
    }
    save_json(os.path.join(run_dir, "train_metrics.json"), payload)
    save_pickle(os.path.join(run_dir, "params.pkl"), params)
    print("Saved JAX timemarch benchmark run to {0}".format(run_dir))


if __name__ == "__main__":
    main()
