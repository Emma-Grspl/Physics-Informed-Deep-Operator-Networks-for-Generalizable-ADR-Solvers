# Physics-Informed DeepONets for Generalizable ADR Solvers

[![CI](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml/badge.svg)](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml)

This repository studies operator learning for a one-dimensional advection-diffusion-reaction problem with parametric initial conditions.

The central equation is

\[
u_t + v\,u_x - D\,u_{xx} = \mu (u-u^3),
\]

where \(v\) is the advection velocity, \(D\) the diffusion coefficient, and \(\mu\) the nonlinear reaction coefficient.

The repository has two scientific goals:

- build a reliable PI-DeepONet surrogate for the ADR problem
- compare PyTorch and JAX under matched protocols on the same task

## What This Repository Contains

This repository intentionally contains two related but distinct tracks.

### 1. `base/`

`base/` is the canonical PyTorch ADR pipeline.

Use it if your question is:

- does the PI-DeepONet work on the ADR problem?
- what is the stable reference implementation?
- what should be cited or reused as the main PyTorch baseline?

### 2. `jax_comparison/`

`jax_comparison/` is the comparison layer built on top of the PyTorch baseline.

Use it if your question is:

- how does JAX behave relative to PyTorch?
- what happens on strict multifamily comparison?
- what do monofamily diagnostics and Gaussian ablations show?

### 3. `experiments/`

`experiments/` is the human-facing registry of reproducible protocols.

It groups the configs, launchers, and protocol notes for:

- the base PyTorch study
- the strict multifamily PyTorch vs JAX comparison
- monofamily diagnostics
- Gaussian ansatz / LBFGS ablations

## Reading Order

If you want the repository to feel clear quickly, read it in this order:

1. this root `README.md`
2. [base/README.md](base/README.md)
3. [jax_comparison/README.md](jax_comparison/README.md)
4. [experiments/README.md](experiments/README.md)

Then, depending on your interest:

- [jax_comparison/multifamily/README.md](jax_comparison/multifamily/README.md) for the main framework comparison
- [jax_comparison/monofamily/README.md](jax_comparison/monofamily/README.md) for diagnostics
- [experiments/ablations/gaussian_hypothesis/README.md](experiments/ablations/gaussian_hypothesis/README.md) for the Gaussian ablation

## Scientific Scope

The model learns an operator that maps:

- physical parameters
- parameters describing the initial condition family
- a query point \((x,t)\)

to the corresponding ADR solution value \(u(x,t)\).

The study covers several initial-condition families, not a single fixed profile. This is why the repository is centered on generalization rather than on one deterministic simulation.

The reference numerical target is a Crank-Nicolson solver. The neural model is trained in a physics-informed way through a loss that combines:

- PDE residual
- initial-condition fit
- boundary-condition fit

## Main Conclusions

### Base PyTorch result

The PyTorch PI-DeepONet is the stable scientific baseline of the repository.

On the reference multifamily benchmark with 20 evaluation cases per family:

- global relative L2: `0.00507 +- 0.00392`
- Tanh: `0.00139 +- 0.00035`
- Sin-Gauss: `0.00978 +- 0.00286`
- Gaussian: `0.00405 +- 0.00100`

Interpretation:

- the surrogate is accurate on the target ADR task
- the model is usable as a fast replacement for the reference solver in this regime

### PyTorch vs JAX

On the strict three-family comparison:

- JAX is much faster in raw training time
- PyTorch is much better in final solution quality

In this repository, PyTorch is therefore the reliable framework for the main ADR conclusions.

### Gaussian Hypothesis Ablation

The Gaussian-family 2x2x2 ablation compares:

- free learning versus ansatz for the initial condition
- with and without an L-BFGS finisher

The main conclusion is:

- the ansatz is the dominant improvement
- L-BFGS does not provide a robust gain in the tested setting

## Repository Map

### Scientific Entry Points

- [base/](base): canonical PyTorch ADR workflow
- [jax_comparison/](jax_comparison): comparison workspace layered on top of the base pipeline
- [experiments/](experiments): reproducible experiment registry
- [benchmarks/](benchmarks): shared benchmark runners and utilities

### Outputs And Assets

- `results/`: active benchmark outputs and run artifacts
- [plot/](plot): curated figures and visual summaries
- `assets/`: presentation assets used across the repository

### Legacy Compatibility Layer

The following top-level folders still exist because some active scripts and benchmark runners depend on them directly:

- `src/`
- `src_jax/`
- `configs/`
- `configs_jax/`
- `launch/`
- `scripts/`

They are runtime infrastructure, not the best human entry points.

## Branching Model

The intended logical separation is:

- `base`: stable PyTorch ADR branch
- `jax-comparison`: comparison branch layered on top of `base`

Practical interpretation:

- a change that would still matter if all JAX material were removed belongs conceptually to `base`
- a change that only exists because of the framework comparison belongs conceptually to `jax-comparison`

The current repository still contains both layers together because it is also used as an integration workspace.

## Installation

### Base PyTorch environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### JAX comparison environment

Install this on top of the base environment:

```bash
pip install -r requirements-jax.txt
```

For GPU machines and HPC systems, install the platform-compatible `jax` and `jaxlib` build first, then install the remaining comparison dependencies.

## Reproducibility

The public experiment definitions live under `experiments/`.

Typical benchmark artifacts include:

- training metrics
- saved checkpoints or serialized parameters
- evaluation against the Crank-Nicolson reference
- inference timing summaries

Benchmark outputs are organized by:

- backend
- benchmark name
- seed

The Gaussian Hypothesis ablation is the main study in this repository that is explicitly aggregated across multiple seeds.

## Which README Should Answer What

- [base/README.md](base/README.md): what the stable PyTorch pipeline is and how to interpret it
- [jax_comparison/README.md](jax_comparison/README.md): what the comparison layer is for
- [experiments/README.md](experiments/README.md): where reproducible protocols are defined
- [benchmarks/README.md](benchmarks/README.md): how the benchmark execution layer is organized

## About The Other `.md` Files At The Repository Root

The other root-level Markdown files are internal maintenance notes from the repository cleanup and branch-splitting process.

They are not the recommended entry points for readers, users, or reviewers of the scientific work.
