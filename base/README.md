# Base PyTorch ADR Pipeline

`base/` is the canonical implementation of the project.

If you want the stable ADR solver without the framework-comparison overhead, start here.

## Scope

This subtree contains the reference PyTorch workflow for:

- data generation for the parametric ADR problem
- PI-DeepONet model definition
- PDE residual computation
- time-marching training
- classical Crank-Nicolson comparison
- baseline analysis plots and saved checkpoints

In manuscript terms, `base/` is the stable scientific baseline of the repository: it contains the reference implementation used to support the main ADR surrogate-learning claims independently of the PyTorch versus JAX comparison layer.

## Directory Layout

- `src/`: main PyTorch codebase for models, data, physics, training, and utilities
- `scripts/`: training and tuning entry points
- `configs/`: canonical PyTorch configs
- `launch/`: generic SLURM launchers for the base workflow
- `tests/`: regression and numerical sanity checks
- `models_saved/`: reference saved model artifacts
- `plots/`: copied base-only plots and figure outputs
- `assets_pytorch/`: selected assets for base-only presentation

## Environment

Recommended installation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is intentionally the base PyTorch environment. It should remain usable without JAX.

## What Belongs Here

Belongs in `base/`:

- improvements to the canonical PyTorch model
- fixes to the classical solver or ADR residual
- changes to the stable training pipeline
- analyses that support the reference ADR study

Does not belong here:

- JAX-only code
- benchmark harness used only for framework comparison
- framework-comparison-only launchers
- mono-family comparison diagnostics that exist only to compare backends

Those belong under [jax_comparison/](../jax_comparison).

## Branching Guidance

The intended long-term git mapping is:

- `base` branch: contents of this subtree as the stable project
- `jax-comparison` branch: adds JAX and comparison material on top of it

If a change would still be valuable after deleting all JAX-related folders, it probably belongs to `base`.
