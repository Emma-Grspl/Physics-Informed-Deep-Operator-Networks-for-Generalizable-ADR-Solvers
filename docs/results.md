# Results

This page separates the reported results into three groups:

- canonical baseline results from `base/`
- strict matched PyTorch vs JAX benchmark results from `jax_vs_pytorch/`
- diagnostic monofamily and ablation results

## Baseline

This page follows the numerical values reported in the root [README.md](/Users/emma.grospellier/Thèse/Projet_These_ADR/README.md).

The canonical `base/` PyTorch baseline is reported there as:

| Study | Horizon | Reported metric |
| --- | --- | --- |
| Baseline global relative L2 | `T_max = 3.0` | `0.0170 ± 0.00392` |
| Tanh family | `T_max = 3.0` | `0.0044 ± 0.0015` |
| Sin-Gauss family | `T_max = 3.0` | `0.0320 ± 0.0200` |
| Gaussian family | `T_max = 3.0` | `0.0148 ± 0.0100` |
| Full-grid inference time | baseline timing summary | `3.79 s` |
| Time-jump inference time | baseline timing summary | `0.034 s` |
| Crank-Nicolson reference time | baseline timing summary | `0.7 s` |
| Time-jump speedup | baseline timing summary | `×21` |

## Strict matched multifamily benchmark

For the strict PyTorch vs JAX benchmark at `T_max = 1.0`, the same `README.md` reports:

| Backend | Global L2 | Tanh | Sin-Gauss | Gaussian | Training time |
| --- | --- | --- | --- | --- | --- |
| PyTorch | `0.005072` | `0.001390` | `0.009777` | `0.004048` | `5329.21 s` |
| JAX | `1.668841` | `1.236420` | `2.639372` | `1.130731` | `349.13 s` |

Interpretation:

- PyTorch is slower by roughly one order of magnitude in training time
- PyTorch is dramatically better in final multifamily accuracy
- `Sin-Gauss` remains the hardest family in both backends

## Monofamily diagnostics

The same `README.md` reports the following monofamily results:

| Protocol | Reported global L2 |
| --- | --- |
| PyTorch `Tanh` only | `0.00158 ± 0.00048` |
| PyTorch `Sin-Gauss` only | `1.00000 ± 0.00000` |
| PyTorch `Gaussian` only | `0.87212 ± 0.01769` |
| JAX `Tanh` only | `9.07159 ± 15.67342` |
| JAX `Sin-Gauss` only | `1.39526 ± 0.41086` |
| JAX `Gaussian` only | `1.02204 ± 0.05012` |

These diagnostics suggest:

- `Tanh` is easy for PyTorch and unstable for JAX in the reported run set
- `Sin-Gauss` is structurally hard
- `Gaussian` benefits strongly from ansatz-based modeling choices

## Ansatz diagnostics

The same `README.md` reports the following ansatz runs:

| Protocol | Reported global L2 |
| --- | --- |
| PyTorch `Sin-Gauss` ansatz | `0.86448 ± 0.13245` |
| PyTorch `Gaussian` ansatz | `0.07643 ± 0.02998` |
| JAX `Sin-Gauss` ansatz | `0.89959 ± 0.13354` |

The strongest qualitative effect is on the Gaussian family, where the ansatz is a large improvement over free learning.

## Gaussian-hypothesis ablation

The same `README.md` reports the following Gaussian-family ablation values:

| Variant | Global L2 |
| --- | --- |
| PyTorch free, no L-BFGS | `0.82391 ± 0.06112` |
| PyTorch free, with L-BFGS | `0.86581 ± 0.07449` |
| PyTorch ansatz, no L-BFGS | `0.16063 ± 0.08412` |
| PyTorch ansatz, with L-BFGS | `0.21143 ± 0.13352` |
| JAX free, no L-BFGS | `1.00648 ± 0.00604` |
| JAX free, with L-BFGS | `1.00647 ± 0.00594` |
| JAX ansatz, no L-BFGS | `0.48137 ± 0.00560` |
| JAX ansatz, with L-BFGS | `0.48024 ± 0.00565` |

Interpretation:

- the ansatz is the main lever
- the L-BFGS finisher is secondary in this ablation
- this is true for both backends, even though PyTorch still reaches better absolute quality

## Seeds

The repository exposes two seed policies in config:

| Protocol family | Seed coverage in config |
| --- | --- |
| short benchmarks, curriculum, timemarch, `t1`, `t05`, Gaussian-hypothesis ablations | `seeds: [0, 1, 2]` |
| full matched multifamily `benchmark_fulltrainer_t1`, equal-pipeline multifamily, monofamily fulltrainer runs, ansatz monofamily runs | `seeds: [0]` |

Examples:

- [jax_vs_pytorch/code/benchmarks/configs/benchmark_t1.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/benchmarks/configs/benchmark_t1.yaml) uses `seeds: [0, 1, 2]`
- [jax_vs_pytorch/code/benchmarks/configs/benchmark_gaussian_hyp_ansatz_lbfgs_on.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/benchmarks/configs/benchmark_gaussian_hyp_ansatz_lbfgs_on.yaml) uses `seeds: [0, 1, 2]`
- [jax_vs_pytorch/code/benchmarks/configs/benchmark_fulltrainer_t1.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/benchmarks/configs/benchmark_fulltrainer_t1.yaml) uses `seeds: [0]`
- [jax_vs_pytorch/code/benchmarks/configs/benchmark_fulltrainer_t1_equal.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/benchmarks/configs/benchmark_fulltrainer_t1_equal.yaml) uses `seeds: [0]`

So the current repo layout supports both multi-seed diagnostics and single-seed flagship runs, but they should not be interpreted with the same statistical strength.
