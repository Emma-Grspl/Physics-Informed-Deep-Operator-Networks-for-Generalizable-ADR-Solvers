# Monofamily

Exploratory PyTorch vs JAX comparison on mono-family ADR experiments.

Contents:
- `assets_monofamily/`: curated assets for mono-family comparison experiments.
- `src/pytorch/`: PyTorch-side benchmark training dependencies used for monofamily runs.
- `src/jax/`: JAX implementation used in monofamily runs.
- `scripts/`: benchmark launch scripts for training, evaluation, and inference.
- `configs/`: PyTorch, JAX, and benchmark configs for tanh, sin-gauss, gaussian, focused, and ansatz runs.
- `launch/`: SLURM launchers for monofamily experiments.
- `tests/`: lightweight validators for benchmark artifacts.
- `plots/`: monofamily comparison figures.

Scope:
- These runs are exploratory and should be interpreted with more caution than the multifamily main result.
