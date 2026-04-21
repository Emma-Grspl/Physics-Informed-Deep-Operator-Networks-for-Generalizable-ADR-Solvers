# Monofamily Comparison

This subtree contains diagnostic comparisons where the ADR task is restricted to a single family or a targeted ablation.

Examples:

- Gaussian-only
- Tanh-only
- Sin-Gauss-only
- ansatz-based diagnostics
- LBFGS on/off ablations

## Purpose

This track is not the main scientific benchmark. It exists to explain failure modes and isolate mechanisms.

Typical questions answered here:

- is the difficulty mainly in multi-family generalization?
- does a hard family remain hard when isolated?
- does an ansatz reduce the burden of learning the initial condition?
- does LBFGS help or hurt within a constrained setting?

## Contents

- `configs/`: PyTorch, JAX, and benchmark configs for mono-family diagnostics
- `launch/`: SLURM launchers for mono-family runs and targeted sweeps

## Interpretation

Use these experiments diagnostically.

They are valuable for explaining behavior, but they should not replace the multifamily comparison when stating the main PyTorch vs JAX conclusion.
