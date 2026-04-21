# Project Structure

This file documents the actual structure of the `jax-comparison` branch.

The branch is intentionally organized around one goal: comparing PyTorch and JAX on the same ADR operator-learning problem while keeping the code, figures, assets, and models clearly separated.

## Repository Root

- `README.md`
  Main scientific overview of the branch. It explains the physical problem, the comparison protocol, the main results, and how to run the branch.
- `Project_structure.md`
  This file. It is meant to describe the actual tree rather than the scientific narrative.
- `Makefile`
  Convenience commands for installation, tests, compilation checks, benchmark aggregation, and default comparison entry points.
- `requirements-base.txt`
  Python dependencies required for the PyTorch side of the comparison and shared utilities.
- `requirements-jax.txt`
  Additional dependencies required for the JAX side of the comparison.
- `.github/workflows/ci.yml`
  CI workflow for syntax checks and the small regression test suite.
- `models/`
  Root-level versioned reference checkpoint exposed at branch level.
- `assets/`
  Curated branch-level visuals for fast reading.
- `figures/`
  Full comparison figures.
- `code/`
  All executable comparison code.

## `code/`

This directory contains everything needed to run, evaluate, and analyze the comparison.

- `code/src/`
  Shared PyTorch implementation used on the comparison side.
  Contains data generation, the PI-DeepONet model, the PDE residual, training utilities, and analysis helpers.
- `code/src_jax/`
  JAX implementation of the ADR PI-DeepONet workflow.
  Contains JAX data utilities, the model, the residual, and training logic.
- `code/benchmarks/`
  Standardized runners used to train, evaluate, and benchmark both backends.
  Includes backend-specific entry points under `pytorch/` and `jax/`, shared utilities under `common/`, and benchmark config files under `configs/`.
- `code/configs/`
  PyTorch model configurations used in the comparison branch.
  Includes the main multifamily comparison config and the monofamily or ansatz variants.
- `code/configs_jax/`
  JAX model configurations parallel to the PyTorch ones.
- `code/experiments/`
  Human-readable experiment registry.
  Organizes the scientific protocols into `multifamily/`, `monofamily/`, and `ablations/`.
- `code/code_experiments/`
  Comparison-specific plotting and synthesis scripts.
  This includes the figure builders for multifamily comparison, monofamily comparison, ansatz studies, and Gaussian-hypothesis summaries.
- `code/launch/`
  SLURM launchers for HPC execution of the benchmarked protocols.
- `code/tests/`
  Small regression checks for the ADR residual and the Crank-Nicolson reference solver.

## `assets/`

This directory contains a compact selection of visuals that summarize the comparison without requiring a reader to inspect the full figure sets.

- `assets/multifamily/`
  Curated visuals for the main strict comparison.
- `assets/monofamily/`
  Curated visuals for monofamily and ansatz-oriented comparisons.

## `figures/`

This directory contains the fuller figure outputs produced for the comparison branch.

- `figures/multifamily/`
  Family-wise error, training-time, inference, frontier, snapshot, and summary figures for the strict multifamily comparison.
- `figures/monofamily/`
  Monofamily comparison plots and ansatz-related comparison plots.

## `models/`

- `models/pytorch_reference.pth`
  Root-level versioned PyTorch reference checkpoint exposed directly at branch level.

This branch does not expose a similarly curated JAX root checkpoint because the main branch-level model artifact is the PyTorch reference, while detailed JAX serialized outputs are produced through the benchmark runners and their result directories.

## Reading Path

For a scientific read:

1. start with `README.md`
2. inspect `assets/`
3. inspect `figures/`
4. inspect `code/experiments/`
5. inspect `code/benchmarks/` and the backend implementations under `code/src/` and `code/src_jax/`

For an implementation read:

1. start with `code/benchmarks/`
2. inspect `code/configs/` and `code/configs_jax/`
3. inspect `code/src/` and `code/src_jax/`
4. inspect `code/code_experiments/` for the reporting layer
