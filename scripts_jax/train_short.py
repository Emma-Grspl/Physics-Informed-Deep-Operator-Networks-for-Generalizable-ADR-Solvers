from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_jax.config import load_config
from src_jax.data.generators import generate_mixed_batch
from src_jax.models.pi_deeponet_adr import init_model_params
from src_jax.training.step import make_train_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short JAX ADR training loop for performance benchmarking.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs_jax" / "config_ADR_jax.yaml"))
    parser.add_argument("--t-max", type=float, default=1.0)
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "outputs" / "JAX" / "short_runs"))
    return parser.parse_args()


def _tree_to_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    run_name = (
        f"tmax_{str(args.t_max).replace('.', 'p')}"
        f"_iters_{args.iters}"
        f"_bs_{args.batch_size}"
        f"_seed_{args.seed}"
    )
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(args.seed)
    model_key, key = jax.random.split(key)

    params = init_model_params(model_key, cfg)
    optimizer = optax.adam(cfg["training"]["learning_rate"])
    opt_state = optimizer.init(params)
    train_step = make_train_step(optimizer)

    print("JAX short training run")
    print(f"device={jax.devices()[0]}")
    print(f"t_max={args.t_max}")
    print(f"iters={args.iters}")
    print(f"batch_size={args.batch_size}")
    print(f"log_every={args.log_every}")

    metrics: list[dict[str, float]] = []
    compile_time = None
    train_start = time.perf_counter()
    window_start = train_start

    for step in range(1, args.iters + 1):
        key, batch_key = jax.random.split(key)
        batch = generate_mixed_batch(
            batch_key,
            n_samples=args.batch_size,
            bounds_phy=cfg["physics_ranges"],
            x_min=cfg["geometry"]["x_min"],
            x_max=cfg["geometry"]["x_max"],
            t_max=args.t_max,
        )

        step_start = time.perf_counter()
        params, opt_state, loss = train_step(
            params,
            opt_state,
            batch,
            wr=cfg["loss_weights"]["first_w_res"],
            wi=cfg["loss_weights"]["weight_ic_final"],
            wb=cfg["loss_weights"]["weight_bc"],
        )
        loss = float(jax.device_get(loss))
        step_duration = time.perf_counter() - step_start

        if step == 1:
            compile_time = step_duration

        if step % args.log_every == 0 or step == 1 or step == args.iters:
            now = time.perf_counter()
            elapsed = now - train_start
            window_elapsed = now - window_start
            avg_window = window_elapsed / max(1, args.log_every if step > 1 else 1)
            record = {
                "step": step,
                "loss": loss,
                "elapsed_sec": elapsed,
                "avg_iter_sec_window": avg_window,
            }
            metrics.append(record)
            print(
                f"[step {step:6d}] loss={loss:.6e} | "
                f"elapsed={elapsed:.2f}s | avg_iter_window={avg_window:.4f}s"
            )
            window_start = now

    total_time = time.perf_counter() - train_start

    metrics_payload = {
        "device": str(jax.devices()[0]),
        "t_max": args.t_max,
        "iters": args.iters,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "compile_first_step_sec": compile_time,
        "total_time_sec": total_time,
        "avg_iter_sec_total": total_time / max(1, args.iters),
        "metrics": metrics,
    }

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    params_path = out_dir / "params_step_last.npz"
    np.savez_compressed(params_path, **_tree_to_numpy(params))

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved parameters to {params_path}")
    print(
        f"Training finished | total={total_time:.2f}s | "
        f"avg_iter={total_time / max(1, args.iters):.4f}s"
    )


if __name__ == "__main__":
    main()
