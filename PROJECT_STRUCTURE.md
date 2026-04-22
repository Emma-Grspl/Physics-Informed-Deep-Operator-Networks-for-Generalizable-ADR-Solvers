# Project Structure

This file documents the actual structure of the `main` branch.

The `main` branch is the presentation and integration branch of the repository. It is meant to expose the whole project clearly: the PyTorch baseline, the JAX versus PyTorch comparison, the curated visuals, and the top-level reference artifacts.

## Repository Root

- `README.md`
  Main scientific entry point for the repository.
- `Project_structure.md`
  This file. It describes the tree and the role of each major directory.
- `Makefile`
  Convenience commands for installation, tests, compilation checks, and the main entry points.
- `requirements-base.txt`
  Dependencies for the PyTorch baseline.
- `requirements-jax.txt`
  Additional dependencies for the JAX comparison.
- `.github/workflows/ci.yml`
  CI workflow aligned with the current `main` branch layout.
- `base/`
  Canonical PyTorch baseline subtree.
- `jax_vs_pytorch/`
  Comparison subtree dedicated to framework evaluation and ablations.
- `assets/`
  Short visual selection for quick inspection at repository level.
- `code/`
  Legacy or staging workspace kept at repository level during the reorganization.
- `model_pi_deeponet/`
  Root-level reference model artifact exposed at repository level.

## `base/`

This subtree is the PyTorch baseline packaged inside `main`.

- `base/README.md`
  Baseline-specific summary.
- `base/code/`
  Baseline code.
- `base/code/configs/`
  Baseline configurations.
- `base/code/scripts/`
  Main training and tuning entry points for the baseline.
- `base/code/src/`
  PyTorch implementation of the ADR pipeline.
- `base/code/launch/`
  Baseline SLURM launchers.
- `base/code/tests/`
  Baseline regression tests.

`base/` should be read as the clean PyTorch-only reference block embedded inside the integration branch.

## `jax_vs_pytorch/`

This subtree is the comparison block packaged inside `main`.

- `jax_vs_pytorch/README.md`
  Comparison-specific summary.
- `jax_vs_pytorch/code/`
  Executable comparison code.
- `jax_vs_pytorch/code/benchmarks/`
  Standardized benchmark runners for training, evaluation, and inference.
- `jax_vs_pytorch/code/configs/`
  PyTorch-side comparison configs.
- `jax_vs_pytorch/code/configs_jax/`
  JAX-side comparison configs.
- `jax_vs_pytorch/code/src_jax/`
  JAX implementation used in the comparison studies.
- `jax_vs_pytorch/code/experiments/`
  Registry of multifamily, monofamily, and ablation protocols.
- `jax_vs_pytorch/code/code_experiments/`
  Plotting and synthesis scripts specific to the comparison branch.
- `jax_vs_pytorch/code/launch/`
  Comparison SLURM launchers.
- `jax_vs_pytorch/figures/`
  Full comparison figures.
- `jax_vs_pytorch/models/`
  Serialized benchmark model artifacts grouped by protocol.

## `assets/`

This root-level directory contains the most representative visuals surfaced at repository level.

- `assets/base/`
  Curated visuals from the baseline side.
- `assets/jax_vs_pytorch/`
  Curated visuals from the comparison side.

## `code/`

This root-level directory is currently a reorganization workspace that mirrors or stages code extracted from the scientific subtrees.

## `model_pi_deeponet/`

This root-level directory contains the small number of reference model artifacts intentionally exposed at repository level.

The detailed benchmark outputs remain inside the relevant scientific subtree rather than at root.

## Reading Path

For a first visit:

1. read `README.md`
2. inspect `assets/`
3. inspect `base/README.md`
4. inspect `jax_vs_pytorch/README.md`

For implementation details:

1. inspect `base/code/`
2. inspect `jax_vs_pytorch/code/`
3. inspect the corresponding `figures/` and `models/`
