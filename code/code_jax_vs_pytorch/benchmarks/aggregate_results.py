import argparse
import csv
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmarks.common.config import load_yaml
from benchmarks.common.io import load_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-config", default=os.path.join("benchmarks", "configs", "benchmark_t1.yaml"))
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark_cfg = load_yaml(args.benchmark_config)
    root_dir = benchmark_cfg["outputs"]["root_dir"]
    benchmark_name = benchmark_cfg["name"]
    seeds = benchmark_cfg["seeds"]
    rows = []

    for backend in ["pytorch", "jax"]:
        for seed in seeds:
            run_dir = os.path.join(root_dir, backend, benchmark_name, "seed_{0}".format(seed))
            train = load_json(os.path.join(run_dir, "train_metrics.json"))
            evaluation = load_json(os.path.join(run_dir, "evaluation.json"))
            inference = load_json(os.path.join(run_dir, "inference.json"))
            rows.append(
                {
                    "backend": backend,
                    "seed": seed,
                    "total_time_sec": train["total_time_sec"],
                    "avg_iter_sec_total": train["avg_iter_sec_total"],
                    "global_l2_mean": evaluation["global_l2_mean"],
                    "tanh_l2": evaluation["family_l2_mean"].get("Tanh"),
                    "sin_gauss_l2": evaluation["family_l2_mean"].get("Sin-Gauss"),
                    "gaussian_l2": evaluation["family_l2_mean"].get("Gaussian"),
                    "inference_full_grid_sec": inference["inference_full_grid_sec"],
                    "time_jump_sec": inference["time_jump_sec"],
                    "cn_reference_sec": inference["cn_reference_sec"],
                    "time_jump_speedup_vs_cn": inference["time_jump_speedup_vs_cn"],
                }
            )

    summary_dir = os.path.join(root_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    output_path = os.path.join(summary_dir, "{0}_summary.csv".format(benchmark_name))
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved summary to {0}".format(output_path))


if __name__ == "__main__":
    main()
