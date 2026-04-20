# Experiments

`experiments/` is the public registry of reproducible protocols.

Its role is simple:

- source code lives in the implementation trees
- reproducible protocol definitions live here

Each experiment package documents a bounded study through:

- model configs
- benchmark configs
- launchers
- short protocol notes

## Why This Directory Exists

Without `experiments/`, the repository would mix two different concerns:

- how the code works
- which scientific protocol was run

This directory is meant to keep those two concerns separate.

## Packages

- `base/`: canonical PyTorch baseline protocol
- `multifamily/`: strict PyTorch vs JAX comparison on the full task
- `monofamily/`: mono-family diagnostics
- `ablations/`: focused studies such as Gaussian ansatz / LBFGS sweeps

## How To Use This Directory

Use `experiments/` when you want to know:

- which config defines a given named study
- which launcher belongs to that study
- which benchmark protocol was intended

Do not use it as the primary place to understand implementation logic. For that, start with:

- [../base/README.md](../base/README.md)
- [../jax_comparison/README.md](../jax_comparison/README.md)

## Conventions

- configs live inside the corresponding experiment package
- launchers are grouped with the protocol they belong to
- benchmark scripts are invoked from the repository root
- generated outputs should be written to `results/` or to the configured benchmark output directory
- `.slurm` launchers in this tree target Jean Zay and are not meant to be cluster-agnostic
