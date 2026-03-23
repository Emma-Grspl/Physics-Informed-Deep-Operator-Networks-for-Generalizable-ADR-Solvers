# JAX Porting Notes

This directory layout is intentionally isolated from the PyTorch baseline.

Current scope:
- `configs_jax/`: dedicated JAX configuration files
- `src_jax/`: standalone JAX implementation
- `scripts_jax/`: JAX-only entry points
- `outputs/JAX/`: isolated artifacts, metrics, and checkpoints for JAX runs

Initial target:
- reproduce the minimal ADR PI-DeepONet kernel
- validate one compiled training step
- keep the PyTorch baseline untouched

Next implementation steps:
1. validate the JAX residual numerically against the PyTorch version on a tiny batch
2. add short-horizon training (`T_max = 0.5`)
3. add evaluation against the Crank-Nicolson reference solver
4. port the time-marching logic only after the minimal core is stable
