# Experiments

This directory groups experiment definitions independently from the active source code.

Rules:

- source code stays in `src/` and `src_jax/` during the current transition
- `experiments/` contains configs, launchers, and short protocol notes
- new experiment setup should be added here first, not into duplicated legacy trees
- results must still be written to `results/` or the configured benchmark output directory

Subdirectories:

- `base/`: canonical PyTorch baseline protocol
- `multifamily/`: strict PyTorch vs JAX comparison protocols
- `monofamily/`: mono-family comparison protocols
- `ablations/`: bounded ablation studies such as ansatz/LBFGS sweeps

This namespace is introduced in Stage 2 of the repository reorganization and is intentionally non-destructive.
