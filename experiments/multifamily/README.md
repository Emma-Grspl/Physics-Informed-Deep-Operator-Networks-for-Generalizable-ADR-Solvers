# Multifamily Experiments

This package groups strict multi-family comparison protocols.

Contents:

- `configs/pytorch/`: PyTorch model configs
- `configs/jax/`: JAX model configs
- `configs/benchmarks/`: benchmark configs shared by the launchers
- `launch/`: Jean Zay SLURM launchers

Scope:

- strict same-pipeline PyTorch vs JAX comparison
- equal-pipeline variants

Execution model:

- launch from the repository root
- benchmark scripts are resolved from `benchmarks/`
- local or non-SLURM runs should reuse the configs in this directory and invoke the benchmark scripts manually
