# Physics-Informed DeepONets for Generalizable ADR Solvers

[![CI](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml/badge.svg)](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml)

Learning physics-informed neural operators for fast and generalizable ADR PDE solving.

PI-DeepONet benchmark for ADR systems.

This project combines scientific ML, operator learning, benchmarking, and reproducible engineering.

### Key Results
- Relative L2 error: **0.00507**
- Inference speedup: **×175** vs Crank-Nicolson
- Strong generalization across 3 initial-condition families
- Full PyTorch vs JAX benchmark included

![Snapshots](assets/base/base_snapshots.png)

![Speedup](assets/base/base_time_jump_speedup.png)

## Contents

- [Quickstart](#quickstart)
- [Why It Matters](#why-it-matters)
- [Method Overview](#method-overview)
- [Main Results](#main-results)
- [PyTorch vs JAX](#pytorch-vs-jax)
- [Experimental Setup](#experimental-setup)
- [Detailed Results](#detailed-results)
- [Repository Structure](#repository-structure)
- [Author](#author)
- [Contact](#contact)

## Quickstart

```bash
pip install -r requirements-base.txt
python base/code/scripts/train.py
```

Additional entry points:

```bash
python base/code/scripts/tune_optuna.py
python jax_vs_pytorch/code/benchmarks/pytorch/train_fulltrainer_benchmark.py \
  --config jax_vs_pytorch/code/benchmarks/configs/benchmark_fulltrainer_t1.yaml
python jax_vs_pytorch/code/benchmarks/jax/train_fulltrainer_benchmark.py \
  --config jax_vs_pytorch/code/benchmarks/configs/benchmark_fulltrainer_t1.yaml
```

## Why It Matters

Fast PDE surrogates can accelerate:

- transport simulations
- reactive systems
- optimization loops
- uncertainty quantification
- scientific computing pipelines

This repository studies a one-dimensional advection-diffusion-reaction equation:

\[
u_t + v\,u_x - D\,u_{xx} = \mu (u-u^3).
\]

ADR combines:

- advection: transport driven by velocity \(v\)
- diffusion: spatial smoothing controlled by \(D\)
- reaction: nonlinear local dynamics controlled by \(\mu\)

The goal is not to solve one trajectory, but to learn a surrogate that generalizes across physical coefficients and initial-condition families.

## Method Overview

The project uses a physics-informed DeepONet with a branch-trunk decomposition:

- branch network: ADR parameters plus initial-condition descriptors
- trunk network: query coordinates \((x,t)\)
- fusion: FiLM-style conditional modulation

Matched architecture in the baseline and the PyTorch vs JAX benchmark:

- branch depth: 5
- trunk depth: 4
- branch width: 256
- trunk width: 256
- latent dimension: 256
- activation: SiLU
- multiscale Fourier features: 20
- Fourier scales: \(0, 1, 2, 3, 4, 5, 6, 8, 10, 12\)

Training objective:

- PDE residual loss
- initial-condition loss
- boundary-condition loss

The surrogate is benchmarked against a Crank-Nicolson reference solver for both accuracy and inference speed.

## Main Results

### Baseline PI-DeepONet

- Global relative \(L^2\): `0.00507 ± 0.00392`
- Speedup: `×175.03`
- Strongest family: `Tanh` with `0.00139 ± 0.00035`
- Hardest family: `Sin-Gauss` with `0.00978 ± 0.00286`

Full multifamily benchmark with 20 evaluation cases per family:

- `Tanh`: `0.00139 ± 0.00035`
- `Sin-Gauss`: `0.00978 ± 0.00286`
- `Gaussian`: `0.00405 ± 0.00100`

Inference benchmark:

- full-grid inference time: `0.210 s`
- time-jump inference time: `0.00285 s`
- Crank-Nicolson reference time: `0.499 s`
- speedup on time-jump inference: `×175.03`

Training time for the short multifamily baseline protocol:

- total training time: `5329.21 s`

![Generalization](assets/base/base_global_mean_L2.png)

Scientific takeaway:

- the baseline PyTorch PI-DeepONet is accurate enough to act as a surrogate in the tested regime
- most of the remaining error is concentrated in the `Sin-Gauss` family
- inference is substantially faster than the classical solver

## PyTorch vs JAX

PyTorch trains slower but reaches significantly better final accuracy.

JAX trains faster but underperformed on the matched multifamily benchmark used in this repository.

Strict multifamily benchmark:

- PyTorch global relative \(L^2\): `0.00507 ± 0.00392`
- JAX global relative \(L^2\): `1.66884 ± 1.62812`
- PyTorch time-jump speedup: `×175.03`
- JAX time-jump speedup: `×45.38`
- PyTorch training time: `5329.21 s`
- JAX training time: `349.13 s`

JAX family-wise results:

- `Tanh`: `1.23642 ± 0.15997`
- `Sin-Gauss`: `2.63937 ± 2.54170`
- `Gaussian`: `1.13073 ± 0.21905`

PyTorch monofamily runs:

- `Tanh` only: `0.00158 ± 0.00048`
- `Sin-Gauss` only: `1.00000 ± 0.00000`
- `Gaussian` only: `0.87212 ± 0.01769`

JAX monofamily runs:

- `Tanh` only: `9.07159 ± 15.67342`
- `Sin-Gauss` only: `1.39526 ± 0.41086`
- `Gaussian` only: `1.02204 ± 0.05012`

Ansatz diagnostics:

- PyTorch `Sin-Gauss` ansatz: `0.86448 ± 0.13245`
- PyTorch `Gaussian` ansatz: `0.07643 ± 0.02998`
- JAX `Sin-Gauss` ansatz: `0.89959 ± 0.13354`

Gaussian-hypothesis ablation over three PyTorch seeds:

- free learning, no L-BFGS: `0.82391 ± 0.06112`
- free learning, with L-BFGS: `0.86581 ± 0.07449`
- ansatz, no L-BFGS: `0.16063 ± 0.08412`
- ansatz, with L-BFGS: `0.21143 ± 0.13352`

Interpretation:

- the ansatz is a stronger lever than the L-BFGS finisher for the Gaussian family
- training speed and final scientific quality must be evaluated separately
- in this repository, PyTorch is the reliable framework for the main ADR conclusion

![Framework Comparison](assets/jax_vs_pytorch/multifamily_family_l2.png)

## Experimental Setup

The repository studies a parametric ADR problem rather than one fixed equation instance.

Physical coefficient ranges:

- \(v \in [0.5, 1.0]\)
- \(D \in [0.01, 0.2]\)
- \(\mu \in [0.0, 1.0]\)

Initial-condition parameters:

- \(A \in [0.7, 1.0]\)
- \(\sigma \in [0.4, 0.8]\)
- \(k \in [1.0, 3.0]\)
- \(x_0 = 0\)

Domain and horizons:

- spatial domain: \(x \in [-5, 8]\)
- baseline horizon: \(T_{\max} = 3.0\)
- PyTorch vs JAX horizon: \(T_{\max} = 1.0\)

Audit grids:

- baseline audits: \(N_x = 500\), \(N_t = 200\)
- JAX vs PyTorch comparison: \(N_x = 400\), \(N_t = 200\)

Initial-condition families:

- `Tanh`
- `Sin-Gauss`
- `Gaussian`

### Baseline Protocol

The baseline training protocol uses:

- a warmup phase on the initial condition
- a progressive temporal curriculum
- king-of-the-hill checkpoint selection
- rollback and retry logic
- NTK-inspired adaptive PDE weighting
- targeted correction on hard families
- a final stricter polishing phase

Baseline time curriculum:

- from \(t=0\) to \(t=0.05\): step size `0.01`
- from \(t=0.05\) to \(t=0.30\): step size `0.05`
- from \(t=0.30\) to \(t=3.0\): step size `0.10`

Main baseline hyperparameters:

- batch size: `8192`
- sampled training points per batch generation: `12288`
- warmup iterations: `7000`
- iterations per time window: `8000`
- correction iterations: `9000`
- number of outer loops: `3`
- rolling window: `2000`
- maximum retries: `4`

Validation thresholds:

- initial-condition threshold: `0.02`
- time-step threshold: `0.03`

Loss weights:

- initial condition weight at start: `2000`
- initial condition weight in final regime: `100`
- boundary-condition weight: `200`
- initial PDE weight: `500`

### PyTorch vs JAX Protocol

The comparison keeps the following matched:

- geometry
- parameter ranges
- width, depth, and latent dimension
- multiscale Fourier encoding
- branch-trunk logic
- evaluation families
- audit philosophy

Comparison protocol:

- \(T_{\max} = 1.0\)
- batch size: `4096`
- sampled training points: `4096`
- warmup iterations: `5000`
- iterations per time window: `2500`
- correction iterations: `4000`
- number of outer loops: `1`
- maximum retries: `2`
- global audit cases: `40`
- family audit cases: `12`

Time curriculum:

- from \(t=0\) to \(t=0.30\): step size `0.05`
- from \(t=0.30\) to \(t=1.0\): step size `0.10`

Benchmark reporting:

- 20 test cases per family
- three reported families: `Tanh`, `Sin-Gauss`, `Gaussian`
- seed `0` for the main public runs

Tracked metrics:

- global relative \(L^2\) error
- family-wise relative \(L^2\) error
- full-grid inference time
- time-jump inference time
- speedup versus Crank-Nicolson
- total training time

## Detailed Results

### Baseline Conclusion

The baseline result is the main scientific success of the repository: a physics-informed PyTorch DeepONet can learn a reliable multifamily surrogate for the tested ADR regime.

### JAX Conclusion

The JAX result is valuable methodologically: it exposes a strong speed-quality trade-off under a matched benchmark, and shows that faster training alone is not sufficient for scientific deployment.

### Overall Assessment

Three conclusions emerge:

- the baseline PyTorch PI-DeepONet is a useful surrogate for the parametric ADR problem
- PyTorch vs JAX must be compared on both speed and final surrogate quality
- family structure matters strongly, and the initial-condition representation has more impact than the L-BFGS finisher in the tested setting

## Repository Structure

The `main` branch is the integration branch of the project.

Top-level layout:

- `base/`: canonical PyTorch baseline study
- `jax_vs_pytorch/`: strict framework-comparison study and related ablations
- `figures/`: curated scientific figures grouped by theme
- `assets/`: representative visuals for quick inspection
- `models/`: root-level reference models exposed at repository level
- `PROJECT_STRUCTURE.md`: detailed map of the repository structure

Main branches:

- `main`: global presentation branch
- `base`: PyTorch baseline branch
- `jax-comparison`: framework-comparison branch

Internal structure of `main`:

- `base/code/`: baseline PyTorch code, configs, launchers, and tests
- `base/README.md`: baseline-specific scientific summary
- `jax_vs_pytorch/code/`: JAX code, benchmark runners, comparison protocols, and experiment code
- `jax_vs_pytorch/figures/`: multifamily and monofamily comparison figures
- `jax_vs_pytorch/models/`: benchmark checkpoints and serialized trained parameters

If you want the stable scientific baseline, start with `base/`. If you want the framework comparison and ablations, go to `jax_vs_pytorch/`.

## Author

Emma Grospellier

## Contact

- GitHub: [Emma-Grspl](https://github.com/Emma-Grspl)
- Email: `emma.grospellier@gmail.com`
