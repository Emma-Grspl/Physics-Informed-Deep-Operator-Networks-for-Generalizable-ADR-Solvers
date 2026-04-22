# Multifamily

Strict PyTorch vs JAX comparison on the full ADR multi-family problem.

Contents:
- `assets_multifamily/`: curated assets for the strict multi-family comparison.
- `src/pytorch/`: PyTorch-side benchmark training dependencies used for comparison.
- `src/jax/`: JAX implementation used in the comparison runs.
- `scripts/`: benchmark launch scripts for training, evaluation, and inference.
- `configs/`: PyTorch, JAX, and benchmark configs for the multifamily runs.
- `launch/`: SLURM launchers for fulltrainer equal-pipeline runs.
- `tests/`: lightweight validators for benchmark artifacts.
- `plots/`: multifamily comparison figures.

Scope:
- Main result: equal-pipeline PyTorch vs JAX on the reference multi-family task.
