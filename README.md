# Physics-Informed DeepONets for the 1D ADR Equation

[![CI](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml/badge.svg)](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml)

This repository contains two closely related but distinct tracks:

1. `base/`: the canonical PyTorch implementation of the ADR PI-DeepONet.
2. `jax_comparison/`: the experimental comparison layer used to compare PyTorch and JAX under controlled protocols.

Human-facing experiment protocols are organized under `experiments/`, which acts as the public experiment registry for reproducible runs and ablations.

## Scientific Scope

The repository studies operator learning for the one-dimensional advection-diffusion-reaction (ADR) equation with parametric initial conditions.

The central scientific goals are:

- learn a surrogate that maps an initial condition and space-time query points to the corresponding ADR solution
- enforce PDE structure through a physics-informed DeepONet (PI-DeepONet) training objective
- compare surrogate predictions against a classical Crank-Nicolson reference solver
- evaluate whether the PyTorch and JAX implementations behave differently under matched training protocols

The main target problem is the multifamily ADR setting, where the model must generalize across several initial-condition families rather than a single restricted family.

## Intended Branching Model

The recommended git organization is:

- `base`: stable branch for the canonical PyTorch ADR pipeline.
- `jax-comparison`: branch built on top of `base`, containing the JAX implementation and all comparison-specific material.

Current `main` still contains both tracks because it is the integration branch used during the cleanup.

Practical rule:

- changes that improve the canonical solver, data generation, training loop, or analysis belong to `base`
- changes that only exist to compare PyTorch and JAX belong to `jax-comparison`

## Repository Map

- [base/](base): canonical PyTorch ADR workflow
- [jax_comparison/](jax_comparison): comparison workspace layered on top of the base workflow
- [experiments/](experiments): official experiment registry for reproducible protocols, configs, and launchers
- [benchmarks/](benchmarks): benchmark helpers and shared benchmark configs
- [plot/](plot): generated figures and summaries
- `results/`: runtime outputs used by analyses

Legacy top-level folders such as `src/`, `src_jax/`, `configs/`, `configs_jax/`, `launch/`, and `scripts/` remain active because some training and benchmark entry points still depend on them directly.

## Installation

### Base PyTorch environment

Use this for the canonical ADR pipeline:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### JAX comparison environment

Use this on top of the base environment for local CPU experiments:

```bash
pip install -r requirements-jax.txt
```

For GPU environments, especially HPC systems, do not assume `requirements-jax.txt` is sufficient. Install the platform-specific `jax` and `jaxlib` wheels first, then install the remaining comparison dependencies from `requirements-jax.txt`.

Jean Zay example:

- use the cluster-provided CUDA stack
- install the matching archived `jaxlib` wheel explicitly
- then install `optax`, `scipy`, `pyyaml`, `tqdm`, and plotting dependencies

## Documentation Entry Points

- [base/README.md](base/README.md): canonical PyTorch workflow, scope, and entry points
- [jax_comparison/README.md](jax_comparison/README.md): overall comparison workspace
- [jax_comparison/multifamily/README.md](jax_comparison/multifamily/README.md): strict full-task comparison
- [jax_comparison/monofamily/README.md](jax_comparison/monofamily/README.md): mono-family diagnostics and ablations
- [experiments/README.md](experiments/README.md): official experiment registry and protocol layout

## Key Results Snapshot

### Base PyTorch ADR result

The canonical PyTorch pipeline is the stable scientific baseline of the repository:

- accurate operator learning on the ADR task
- usable surrogate quality
- substantial inference speedup relative to the classical Crank-Nicolson solver

Reference multifamily benchmark (`benchmark_fulltrainer_t1`, 20 evaluation cases per family):

- global relative L2: `0.00507 +- 0.00392`
- Tanh family: `0.00139 +- 0.00035`
- Sin-Gauss family: `0.00978 +- 0.00286`
- Gaussian family: `0.00405 +- 0.00100`

### Strict multifamily PyTorch vs JAX comparison

On the full three-family task, PyTorch is the reliable framework in this repository.

Interpretation:

- JAX is faster in raw training time
- PyTorch is decisively better in final solution quality under the matched pipeline

### Gaussian hypothesis ablation

The Gaussian-family `ansatz` / `LBFGS` 2x2x2 ablation is now complete for both frameworks.

Final global relative L2 means:

- PyTorch free / no LBFGS: `0.8239 +- 0.0611`
- PyTorch free / LBFGS: `0.8658 +- 0.0745`
- PyTorch ansatz / no LBFGS: `0.1606 +- 0.0841`
- PyTorch ansatz / LBFGS: `0.2114 +- 0.1335`
- JAX free / no LBFGS: `1.0065 +- 0.0060`
- JAX free / LBFGS: `1.0065 +- 0.0059`
- JAX ansatz / no LBFGS: `0.4814 +- 0.0056`
- JAX ansatz / LBFGS: `0.4802 +- 0.0056`

Takeaway:

- the `ansatz` is the dominant factor for both frameworks
- `LBFGS` does not materially help on this Gaussian ablation
- JAX is faster, but PyTorch remains clearly more accurate on the final error

## What Is Stable vs Experimental

Stable:

- PyTorch ADR model and training pipeline
- canonical configs and launchers for the PyTorch workflow
- base analyses used for the main ADR conclusions

Experimental:

- JAX implementation under `src_jax/`
- benchmark harness under `benchmarks/`
- equal-pipeline PyTorch vs JAX benchmark material
- mono-family diagnostics
- ansatz and LBFGS ablations

## Cleanup Direction

The target end state is:

- `base/` remains the reference package for the stable PyTorch solver
- `jax_comparison/` remains a clear add-on package for framework comparison
- `experiments/` becomes the single place for human-facing experiment definitions
- legacy duplicated configs and launchers are removed once all active paths point to the cleaned layout
