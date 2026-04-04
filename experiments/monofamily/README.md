# Monofamily Experiments

This namespace groups mono-family comparison protocols.

Current contents:

- `configs/pytorch/`: copied monofamily PyTorch configs
- `configs/jax/`: copied monofamily JAX configs
- `configs/benchmarks/`: copied benchmark configs
- `launch/`: copied SLURM launchers

Transition status:

- files were copied here from `jax_comparison/monofamily/`
- legacy paths remain valid for compatibility
- this directory should become the primary home for mono-family experiment definitions in later stages

Scope:

- tanh-only
- singauss-only
- gaussian-only
- ansatz and focused monofamily diagnostics
