# Multifamily Experiments

This namespace groups strict multi-family comparison protocols.

Current contents:

- `configs/pytorch/`: copied multifamily PyTorch configs
- `configs/jax/`: copied multifamily JAX configs
- `configs/benchmarks/`: copied benchmark configs
- `launch/`: copied SLURM launchers

Transition status:

- files were copied here from `jax_comparison/multifamily/`
- legacy paths remain valid for compatibility
- this directory should become the primary home for multi-family experiment definitions in later stages

Scope:

- strict same-pipeline PyTorch vs JAX comparison
- equal-pipeline variants
