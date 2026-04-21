# Monofamily Experiments

This package groups mono-family diagnostic protocols.

These studies are not the main benchmark of the repository. They exist to explain behavior observed in the strict multifamily comparison.

## Scope

This package includes:

- tanh-only studies
- Sin-Gauss-only studies
- Gaussian-only studies
- ansatz-based monofamily diagnostics
- targeted focused runs on hard families

## Contents

- `configs/pytorch/`: PyTorch model configs
- `configs/jax/`: JAX model configs
- `configs/benchmarks/`: benchmark definitions shared by the launchers
- `launch/`: Jean Zay SLURM launchers

## Interpretation

Use this package to answer questions such as:

- which family is intrinsically hard?
- does a family remain hard when isolated?
- does an ansatz reduce the difficulty?

These experiments are explanatory, not primary.

## Execution Model

- launch from the repository root
- benchmark scripts are resolved from `benchmarks/`
- for local or non-SLURM usage, reuse the configs in this directory and call the benchmark runners manually
