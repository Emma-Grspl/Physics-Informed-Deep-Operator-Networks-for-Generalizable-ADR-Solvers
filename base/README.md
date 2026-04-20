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

## Branching Guidance

The intended long-term mapping is:

- `base` branch: stable PyTorch project
- `jax-comparison` branch: comparison layer added on top of `base`

If a change would still make sense after removing all JAX-related material, it most likely belongs conceptually to `base`.
