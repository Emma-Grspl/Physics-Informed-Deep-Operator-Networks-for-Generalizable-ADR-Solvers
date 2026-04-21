# Gaussian Hypothesis

This package isolates the Gaussian-family ablation used to test two ideas:

- is a large part of the difficulty caused by free learning of the initial condition?
- does an L-BFGS finisher materially improve the result?

## Variants

The package spans the four standard variants:

- `free_lbfgs_off`
- `free_lbfgs_on`
- `ansatz_lbfgs_off`
- `ansatz_lbfgs_on`

Each variant exists for both PyTorch and JAX.

## Contents

- `configs/pytorch/`: PyTorch model configs for the four variants
- `configs/jax/`: JAX model configs for the four variants
- `configs/benchmarks/`: benchmark definitions used by both backends
- `launch/`: Jean Zay array launchers

## Why This Package Matters

This is the cleanest ablation in the repository for separating:

- architecture-side help from the ansatz
- optimizer-side help from the L-BFGS finisher

It is therefore one of the key interpretation packages of the comparison layer.

## Execution Model

- configs and launchers in this directory are self-contained
- launchers resolve benchmark and model configs from this package
- for local or non-SLURM usage, reuse the configs in this directory and call the benchmark runners manually

Launch from repository root:

- `sbatch experiments/ablations/gaussian_hypothesis/launch/launch_gaussian_hypothesis_pytorch_array.slurm`
- `sbatch experiments/ablations/gaussian_hypothesis/launch/launch_gaussian_hypothesis_jax_array.slurm`
