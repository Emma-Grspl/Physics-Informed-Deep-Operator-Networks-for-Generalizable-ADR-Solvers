# Base PyTorch ADR Pipeline

`base/` is the canonical scientific baseline of the repository.

If someone wants to understand the main ADR surrogate without the framework-comparison layer, this is the correct place to start.

## Role

This subtree contains the reference PyTorch workflow used to support the main conclusions of the project:

- data generation for the parametric ADR problem
- PI-DeepONet model definition
- PDE residual computation
- time-marching training
- comparison with the Crank-Nicolson reference solver
- saved checkpoints, plots, and baseline artifacts

In other words:

- `base/` answers whether the PI-DeepONet works on the ADR task
- `jax_comparison/` answers how JAX behaves relative to PyTorch

## What A Reader Should Expect Here

Someone reading `base/` should be able to answer four questions quickly:

1. what physical problem is being solved?
2. what neural architecture is used?
3. what is the training logic?
4. what should be considered the stable PyTorch reference result?

This subtree is therefore meant to stay as readable as possible and should not absorb comparison-only material.

## Executive Summary

This baseline shows that a PyTorch PI-DeepONet can learn a reliable surrogate for the one-dimensional ADR problem with parametric initial conditions.

What this subtree demonstrates:

- the model reaches low relative error on the target multifamily task
- the surrogate is much faster at inference than the Crank-Nicolson reference solver
- the PyTorch pipeline is stable enough to serve as the scientific reference of the project

If someone only reads one conclusion from `base/`, it should be:

- the baseline works, and it works well enough to justify the whole project

## Main Baseline Results

### Reference Multifamily Result

On the reference multifamily benchmark with 20 evaluation cases per family:

- global relative L2: `0.00507 +- 0.00392`
- Tanh: `0.00139 +- 0.00035`
- Sin-Gauss: `0.00978 +- 0.00286`
- Gaussian: `0.00405 +- 0.00100`

Interpretation:

- the global error is low on the actual target task
- the model generalizes across several initial-condition families rather than a single restricted setting
- `Tanh` is very well captured
- `Sin-Gauss` is the hardest family inside the successful multifamily baseline

### Inference Value

The baseline is not only accurate. It is also useful as a surrogate.

In the benchmark setting used in the project, the trained PI-DeepONet provides a substantial speedup relative to the Crank-Nicolson reference in direct inference and time-jump usage.

This is the practical reason the branch matters:

- the baseline is not only numerically interesting
- it is also computationally valuable

## What This Subtree Establishes Scientifically

This branch supports the following claims:

- a physics-informed DeepONet can learn the ADR operator with good fidelity
- the surrogate remains accurate on a multifamily benchmark, not only on a toy single-family case
- the baseline PyTorch implementation is robust enough to be the reference point for all later comparisons

## What This Subtree Does Not Claim

This subtree is deliberately not trying to answer:

- whether JAX is better than PyTorch
- whether the matched comparison protocol is fairer or faster
- how much of the difficulty comes specifically from family-wise failure modes

Those questions belong to the comparison layer under `jax_comparison/`.

## Directory Layout

- `src/`: main PyTorch codebase for model, data, physics, training, and utilities
- `scripts/`: training and tuning entry points for the base workflow
- `configs/`: canonical PyTorch configurations
- `launch/`: SLURM launchers for the base workflow
- `tests/`: regression and numerical sanity checks
- `models_saved/`: saved reference artifacts
- `plots/`: base-only figures and analysis outputs
- `assets_pytorch/`: curated visual assets for the PyTorch baseline

## How To Read This Subtree

Recommended reading order:

1. this README
2. the main configuration files in `configs/`
3. the training entry points in `scripts/`
4. the saved figures and analysis outputs

For an interview-style quick read:

1. this README
2. the result summary above
3. the figures under `plots/` and `assets_pytorch/`

## Environment

Recommended installation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is intentionally the base environment. It should remain usable without JAX.

## What Belongs In `base/`

Belongs here:

- improvements to the canonical PyTorch model
- fixes to the ADR residual or the classical solver
- changes to the stable training pipeline
- analyses that support the reference ADR study

Does not belong here:

- JAX-only code
- benchmark infrastructure created only for framework comparison
- PyTorch wrappers whose sole purpose is to match JAX protocols
- monofamily diagnostics used only to compare backends

Those belong under [jax_comparison/](../jax_comparison).

## Main Limitations

Even though this branch is the stable baseline, it still has limitations that matter scientifically and practically.

### Scientific limitations

- the problem becomes significantly harder on some families than on others, especially `Sin-Gauss`
- the repository demonstrates a strong empirical baseline, but not a full theoretical analysis of why each family behaves differently
- the most detailed failure-mode analysis is outside this subtree, in the comparison and ablation branches

### Repository limitations

- some active runtime infrastructure still lives in legacy top-level folders
- benchmark and comparison utilities are not fully isolated from the rest of the repository
- the subtree is the baseline source of truth, but the repository as a whole is still partly an integration workspace

These limitations do not invalidate the baseline result, but they explain why the repository is documented through several layers rather than through a single perfectly isolated package.

## Branching Guidance

The intended long-term mapping is:

- `base` branch: stable PyTorch project
- `jax-comparison` branch: comparison layer added on top of `base`

If a change would still make sense after removing all JAX-related material, it most likely belongs conceptually to `base`.
