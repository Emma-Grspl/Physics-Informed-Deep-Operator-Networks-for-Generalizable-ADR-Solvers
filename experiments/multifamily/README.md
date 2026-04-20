# Multifamily Experiments

This package groups the main full-task comparison protocols.

This is the most important experiment package for the framework comparison because it evaluates the real three-family ADR task rather than a simplified diagnostic subset.

## Scope

This package covers:

- strict same-protocol PyTorch vs JAX comparison
- equal-pipeline comparison variants

## Contents

- `configs/pytorch/`: PyTorch model configs
- `configs/jax/`: JAX model configs
- `configs/benchmarks/`: benchmark definitions used by both backends
- `launch/`: Jean Zay SLURM launchers

## Interpretation

If one single comparison result must be treated as the main framework conclusion of the repository, it should come from this package.

## Execution Model

- launch from the repository root
- benchmark scripts are resolved from `benchmarks/`
- for local or non-SLURM usage, reuse the configs in this directory and call the benchmark runners manually
