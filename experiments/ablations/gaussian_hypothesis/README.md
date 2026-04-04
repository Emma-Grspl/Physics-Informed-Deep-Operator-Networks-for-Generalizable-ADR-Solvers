# Gaussian Hypothesis

This experiment package isolates the mono-family Gaussian ablation used to test the hypothesis that performance differences are driven mainly by:

- free learning of the initial condition versus ansatz
- presence or absence of an L-BFGS finisher

Contents:

- `configs/pytorch/`: PyTorch model configs for the four variants
- `configs/jax/`: JAX model configs for the four variants
- `configs/benchmarks/`: benchmark configs used by both backends
- `launch/`: Jean Zay array launchers for PyTorch and JAX

Package properties:

- configs and launchers in this directory are self-contained
- launchers resolve benchmark and model configs from `experiments/ablations/gaussian_hypothesis/...`
- local or non-SLURM runs should reuse these configs and invoke the benchmark scripts manually

Variants:

- `free_lbfgs_off`
- `free_lbfgs_on`
- `ansatz_lbfgs_off`
- `ansatz_lbfgs_on`

Launch from repository root:

- `sbatch experiments/ablations/gaussian_hypothesis/launch/launch_gaussian_hypothesis_pytorch_array.slurm`
- `sbatch experiments/ablations/gaussian_hypothesis/launch/launch_gaussian_hypothesis_jax_array.slurm`
