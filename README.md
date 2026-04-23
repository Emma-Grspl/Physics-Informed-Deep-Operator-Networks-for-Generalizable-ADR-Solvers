# Physics-Informed DeepONets for Generalizable ADR Solvers

[![CI](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml/badge.svg)](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml)

This repository studies physics-informed DeepONets for a parametric 1D advection-diffusion-reaction (ADR) equation. It contains:

- a canonical PyTorch baseline in `base/`
- a strict PyTorch vs JAX comparison in `jax_vs_pytorch/`
- reproducible benchmark runners, figures, and experiment configs

## 1. What is this project?

The core question is whether a PI-DeepONet can learn an ADR solution operator that remains accurate across:

- varying physical coefficients `(v, D, mu)`
- multiple initial-condition families
- different inference modes, including time jumping

The repository therefore has two roles:

- `base/` answers whether the surrogate works as a scientific baseline
- `jax_vs_pytorch/` answers how closely matched PyTorch and JAX workflows compare

## 2. Why it matters

Fast PDE surrogates matter when classical solvers are too expensive inside larger loops such as optimization, inverse problems, uncertainty quantification, or repeated scenario screening.

This project focuses on a nonlinear ADR equation, where the surrogate must be both fast and robust across families of initial conditions rather than only one curated case.

## 3. Key results

Two result layers matter and should not be mixed:

| Study | Scope | Headline result |
| --- | --- | --- |
| `base/` baseline | Canonical PyTorch study at `T_max = 3.0` | Reported global relative `L2 = 0.0170 ± 0.00392`, with time-jump speedup `×21` vs Crank-Nicolson |
| `jax_vs_pytorch/` matched benchmark | Strict PyTorch vs JAX comparison at `T_max = 1.0` | Curated multifamily metrics report PyTorch `0.00507` vs JAX `1.66884` global relative `L2` |

Matched multifamily benchmark summary:

| Backend | Global relative L2 | Tanh | Sin-Gauss | Gaussian | Training time |
| --- | --- | --- | --- | --- | --- |
| PyTorch | `0.005072` | `0.001390` | `0.009777` | `0.004048` | `5329.21 s` |
| JAX | `1.668841` | `1.236420` | `2.639372` | `1.130731` | `349.13 s` |

The main takeaway is simple: in this repository, JAX is much faster to train, but PyTorch is the reliable backend for final multifamily accuracy.

## 4. Main figures

### Baseline

![Baseline global mean L2](assets/base/base_global_mean_L2.png)

![Baseline time-jump speedup](assets/base/base_time_jump_speedup.png)

### Framework comparison

![PyTorch vs JAX family comparison](assets/jax_vs_pytorch/multifamily_family_l2.png)

![PyTorch vs JAX snapshots](assets/jax_vs_pytorch/multifamily_snapshots.png)

## 5. Quickstart

Install the baseline dependencies:

```bash
pip install -r requirements-base.txt
python base/code/scripts/train.py
```

Run the canonical PyTorch tuning entry point:

```bash
python base/code/scripts/tune_optuna.py
```

Run the matched PyTorch vs JAX full benchmark:

```bash
python jax_vs_pytorch/code/benchmarks/pytorch/train_fulltrainer_benchmark.py \
  --benchmark-config jax_vs_pytorch/code/benchmarks/configs/benchmark_fulltrainer_t1.yaml \
  --model-config jax_vs_pytorch/code/configs/config_ADR_t1_compare.yaml

python jax_vs_pytorch/code/benchmarks/jax/train_fulltrainer_benchmark.py \
  --benchmark-config jax_vs_pytorch/code/benchmarks/configs/benchmark_fulltrainer_t1.yaml \
  --model-config jax_vs_pytorch/code/configs_jax/config_ADR_jax_t1_compare.yaml
```

## 6. Links to detailed docs

- [docs/science.md](docs/science.md): equation, architecture, loss, curriculum, and generalization setup
- [docs/benchmark.md](docs/benchmark.md): strict PyTorch vs JAX benchmark design and interpretation
- [docs/results.md](docs/results.md): headline tables, ablations, and seed coverage
- [base/README.md](base/README.md): baseline subtree overview
- [jax_vs_pytorch/README.md](jax_vs_pytorch/README.md): framework-comparison subtree overview
- For full reproducibility, see docs/experimental_protocol.md
