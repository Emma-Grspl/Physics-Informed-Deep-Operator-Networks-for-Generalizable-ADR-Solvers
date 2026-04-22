# Benchmark Notes

## PyTorch vs JAX

The `jax_vs_pytorch/` subtree is the full framework-comparison layer of the repository. It does not only contain one headline benchmark. It contains the complete comparison protocol used to answer four different questions:

- does PyTorch or JAX perform better on the full multifamily ADR task?
- what changes when each family is isolated?
- does an initial-condition ansatz reduce the difficulty?
- does an L-BFGS finisher materially help?

The public-facing numerical values in this document follow the root [README.md](/Users/emma.grospellier/Thèse/Projet_These_ADR/README.md), which remains the source of truth for reported results.

## Protocol map

The comparison protocol is split into four experiment blocks under `jax_vs_pytorch/code/experiments/`:

- `multifamily/`: main PyTorch vs JAX comparison on the full three-family task
- `monofamily/`: family-isolated diagnostics
- `ablations/gaussian_hypothesis/`: ansatz vs free-learning and L-BFGS vs no-L-BFGS
- `base/`: placeholder for the baseline protocol inside the comparison namespace

Useful registry files:

- [jax_vs_pytorch/code/experiments/README.md](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/experiments/README.md)
- [jax_vs_pytorch/code/experiments/multifamily/README.md](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/experiments/multifamily/README.md)
- [jax_vs_pytorch/code/experiments/monofamily/README.md](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/experiments/monofamily/README.md)
- [jax_vs_pytorch/code/experiments/ablations/gaussian_hypothesis/README.md](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/experiments/ablations/gaussian_hypothesis/README.md)

## Matched setup

Across the comparison layer, PyTorch and JAX are kept as aligned as possible:

- same ADR equation
- same physical ranges
- same three main families: `Tanh`, `Sin-Gauss`, `Gaussian`
- same branch/trunk depth and width in the main matched runs
- same Fourier-feature setup
- same benchmark evaluation and inference logic

Main matched model configs:

- PyTorch: [jax_vs_pytorch/code/configs/config_ADR_t1_compare.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/configs/config_ADR_t1_compare.yaml)
- JAX: [jax_vs_pytorch/code/configs_jax/config_ADR_jax_t1_compare.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/configs_jax/config_ADR_jax_t1_compare.yaml)

Shared headline settings in the main comparison:

- `T_max = 1.0`
- `x in [-5, 8]`
- audit grid `Nx = 400`, `Nt = 200`
- branch depth `5`
- trunk depth `4`
- width `256`
- latent dimension `256`
- Fourier features `20`
- learning rate `6.078744921577277e-05`
- batch size `4096`
- warmup `5000` iterations
- per-step training `2500` iterations
- correction `4000` iterations

## Multifamily

This is the primary comparison block. If one result should be treated as the main framework conclusion, it is this one.

Main multifamily benchmarks:

- [jax_vs_pytorch/code/benchmarks/configs/benchmark_fulltrainer_t1.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/benchmarks/configs/benchmark_fulltrainer_t1.yaml)
- [jax_vs_pytorch/code/benchmarks/configs/benchmark_fulltrainer_t1_equal.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/benchmarks/configs/benchmark_fulltrainer_t1_equal.yaml)
- [jax_vs_pytorch/code/benchmarks/configs/benchmark_curriculum_t1.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/benchmarks/configs/benchmark_curriculum_t1.yaml)
- [jax_vs_pytorch/code/benchmarks/configs/benchmark_timemarch_t1.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/benchmarks/configs/benchmark_timemarch_t1.yaml)
- [jax_vs_pytorch/code/benchmarks/configs/benchmark_t1.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/benchmarks/configs/benchmark_t1.yaml)
- [jax_vs_pytorch/code/benchmarks/configs/benchmark_t1_5k.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/benchmarks/configs/benchmark_t1_5k.yaml)
- [jax_vs_pytorch/code/benchmarks/configs/benchmark_t1_20k.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/benchmarks/configs/benchmark_t1_20k.yaml)

Role of these runs:

- `benchmark_fulltrainer_t1`: flagship strict comparison used in the `README`
- `benchmark_fulltrainer_t1_equal`: equal-pipeline variant
- `benchmark_curriculum_t1` and `benchmark_timemarch_t1`: curriculum and time-marching diagnostics
- `benchmark_t1`, `benchmark_t1_5k`, `benchmark_t1_20k`: shorter or longer training-budget comparisons

Headline result reported in [README.md](/Users/emma.grospellier/Thèse/Projet_These_ADR/README.md):

| Backend | Global L2 | Tanh | Sin-Gauss | Gaussian | Training time |
| --- | --- | --- | --- | --- | --- |
| PyTorch | `0.005072` | `0.001390` | `0.009777` | `0.004048` | `5329.21 s` |
| JAX | `1.668841` | `1.236420` | `2.639372` | `1.130731` | `349.13 s` |

## Monofamily

The monofamily block asks whether the framework gap remains once each family is isolated.

Main monofamily benchmark configs:

- [jax_vs_pytorch/code/benchmarks/configs/benchmark_fulltrainer_t1_tanh_only.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/benchmarks/configs/benchmark_fulltrainer_t1_tanh_only.yaml)
- [jax_vs_pytorch/code/benchmarks/configs/benchmark_fulltrainer_t1_singauss_only.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/benchmarks/configs/benchmark_fulltrainer_t1_singauss_only.yaml)
- [jax_vs_pytorch/code/benchmarks/configs/benchmark_fulltrainer_t1_gaussian_only.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/benchmarks/configs/benchmark_fulltrainer_t1_gaussian_only.yaml)
- [jax_vs_pytorch/code/benchmarks/configs/benchmark_singauss_focused_t1.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/benchmarks/configs/benchmark_singauss_focused_t1.yaml)

These runs are explanatory, not primary. They are used to locate which family drives failure.

Values reported in [README.md](/Users/emma.grospellier/Thèse/Projet_These_ADR/README.md):

- PyTorch `Tanh` only: `0.00158 ± 0.00048`
- PyTorch `Sin-Gauss` only: `1.00000 ± 0.00000`
- PyTorch `Gaussian` only: `0.87212 ± 0.01769`
- JAX `Tanh` only: `9.07159 ± 15.67342`
- JAX `Sin-Gauss` only: `1.39526 ± 0.41086`
- JAX `Gaussian` only: `1.02204 ± 0.05012`

## Ansatz

The ansatz studies test whether part of the difficulty comes from forcing the network to learn the initial-condition structure freely.

Main ansatz configs:

- [jax_vs_pytorch/code/benchmarks/configs/benchmark_singauss_ansatz_t1.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/benchmarks/configs/benchmark_singauss_ansatz_t1.yaml)
- [jax_vs_pytorch/code/benchmarks/configs/benchmark_gaussian_ansatz_t1.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/benchmarks/configs/benchmark_gaussian_ansatz_t1.yaml)

These are still monofamily diagnostics, but they play a separate interpretive role because they change the modeling assumption rather than only the family composition.

Values reported in [README.md](/Users/emma.grospellier/Thèse/Projet_These_ADR/README.md):

- PyTorch `Sin-Gauss` ansatz: `0.86448 ± 0.13245`
- PyTorch `Gaussian` ansatz: `0.07643 ± 0.02998`
- JAX `Sin-Gauss` ansatz: `0.89959 ± 0.13354`

Interpretation:

- the ansatz helps little on `Sin-Gauss`
- the ansatz helps strongly on `Gaussian`
- a large part of the difficulty comes from how the initial condition is represented

## L-BFGS and Gaussian Hypothesis

The Gaussian-hypothesis ablation isolates two axes:

- free learning vs ansatz
- no L-BFGS vs with L-BFGS

Main ablation configs:

- [jax_vs_pytorch/code/benchmarks/configs/benchmark_gaussian_hyp_free_lbfgs_off.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/benchmarks/configs/benchmark_gaussian_hyp_free_lbfgs_off.yaml)
- [jax_vs_pytorch/code/benchmarks/configs/benchmark_gaussian_hyp_free_lbfgs_on.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/benchmarks/configs/benchmark_gaussian_hyp_free_lbfgs_on.yaml)
- [jax_vs_pytorch/code/benchmarks/configs/benchmark_gaussian_hyp_ansatz_lbfgs_off.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/benchmarks/configs/benchmark_gaussian_hyp_ansatz_lbfgs_off.yaml)
- [jax_vs_pytorch/code/benchmarks/configs/benchmark_gaussian_hyp_ansatz_lbfgs_on.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/benchmarks/configs/benchmark_gaussian_hyp_ansatz_lbfgs_on.yaml)

Reported values in [README.md](/Users/emma.grospellier/Thèse/Projet_These_ADR/README.md):

- free learning, no L-BFGS: `0.82391 ± 0.06112`
- free learning, with L-BFGS: `0.86581 ± 0.07449`
- ansatz, no L-BFGS: `0.16063 ± 0.08412`
- ansatz, with L-BFGS: `0.21143 ± 0.13352`

This block is important because it separates two different ideas:

- whether injecting structure helps
- whether a stronger finisher helps

The conclusion reported by the repo is that the ansatz is the stronger lever, while L-BFGS is secondary.

## Metrics

The evaluation and timing logic is implemented in [jax_vs_pytorch/code/benchmarks/common/eval.py](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/benchmarks/common/eval.py).

For each run, the comparison layer can record:

- global relative `L2`
- family-wise relative `L2`
- full-grid inference time
- time-jump inference time
- Crank-Nicolson reference time
- time-jump speedup vs Crank-Nicolson
- total training time

The headline time-jump speedups reported in [README.md](/Users/emma.grospellier/Thèse/Projet_These_ADR/README.md) are:

- baseline PyTorch: `×21`
- matched JAX benchmark: `×45.38`

## Interpretation

The full `jax_vs_pytorch` protocol should be read as a layered argument:

1. multifamily establishes the main framework conclusion
2. monofamily localizes which families are hard
3. ansatz tests whether structured initial-condition modeling reduces that difficulty
4. Gaussian-hypothesis isolates the effect of L-BFGS from the effect of the ansatz

The overall conclusion stays the same across the package:

- JAX is much faster in raw training time
- PyTorch is much better in final scientific quality on the main ADR task
- `Sin-Gauss` is the hardest family
- ansatz-based structure matters more than the L-BFGS finisher
