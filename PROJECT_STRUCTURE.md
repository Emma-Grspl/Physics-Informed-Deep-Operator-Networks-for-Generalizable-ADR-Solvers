# Project Structure

This file documents the actual structure of the `base` branch.

The `base` branch is the PyTorch-only scientific baseline of the repository. It is intentionally narrower than `main` and `jax-comparison`: it focuses on the canonical ADR PI-DeepONet workflow without the additional framework-comparison layer.

## Repository Root

- `README.md`
  Main scientific entry point for the branch.
- `Project_structure.md`
  This file. It explains the actual tree layout.
- `Makefile`
  Convenience commands for installation, testing, compilation checks, training, and analysis.
- `requirements-base.txt`
  Reference Python dependencies for the PyTorch baseline.
- `.github/workflows/ci.yml`
  CI workflow aligned with the baseline layout.
- `code/`
  All baseline source code and utilities.
- `figures/`
  Full baseline figures.
- `assets/`
  Curated branch-level visuals for fast reading.

## `code/`

This directory contains the executable PyTorch baseline.

- `code/configs/`
  Canonical baseline configuration files.
- `code/scripts/`
  Main training and tuning entry points.
- `code/src/`
  Main PyTorch source tree.
- `code/src/data/`
  ADR data-generation utilities.
- `code/src/models/`
  PI-DeepONet model definition.
- `code/src/physics/`
  ADR residual and physics operators.
- `code/src/training/`
  Time-marching training logic and benchmark-compatible training helpers.
- `code/src/utils/`
  Classical solver utilities, metrics, and auxiliary helpers.
- `code/src/analyse/`
  Baseline analysis and figure-generation helpers.
- `code/launch/`
  SLURM launchers for the baseline workflow.
- `code/tests/`
  Small regression tests and numerical sanity checks.

## `figures/`

This directory contains the full baseline visual outputs.

- `figures/crank_nicolson/`
  Figures for the reference classical solver.
- `figures/pi_deeponet/`
  Figures for the PI-DeepONet baseline alone.
- `figures/pi_deeponet_vs_cn/`
  Direct comparison figures between the PI-DeepONet and the Crank-Nicolson reference.

## `assets/`

This directory contains the small curated set of visuals that summarize the baseline quickly.

Typical branch-level assets include the main error summary, snapshots, and the time-jump speedup view.

## Reading Path

For a scientific read:

1. start with `README.md`
2. inspect `assets/`
3. inspect `figures/`
4. inspect `code/configs/`
5. inspect `code/scripts/` and `code/src/training/`

For an implementation read:

1. inspect `code/src/models/`
2. inspect `code/src/physics/`
3. inspect `code/src/training/`
4. inspect `code/src/utils/`
