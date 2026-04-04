# JAX Comparison Layer

`jax_comparison/` is not the canonical project root. It is the comparison layer built on top of the stable PyTorch ADR pipeline.

Its purpose is to answer one question cleanly:

- how does JAX behave relative to PyTorch under matched ADR training protocols?

## Scope

This subtree groups:

- JAX implementations of the ADR model and training logic
- comparison-specific PyTorch wrappers used for matched benchmarks
- the benchmark harness under `benchmarks/`
- benchmark configs and launchers
- figure-generation material for framework comparison

It is intentionally separated from `base/` so the stable PyTorch workflow can remain readable.

## Structure

- [multifamily/](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_comparison/multifamily): strict full-task comparison on the three-family ADR problem
- [monofamily/](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_comparison/monofamily): mono-family diagnostics and targeted ablations

## Environment

Install the base environment first:

```bash
pip install -r requirements.txt
```

Then add JAX comparison dependencies:

```bash
pip install -r requirements-jax.txt
```

For GPU clusters, especially Jean Zay, install the correct `jaxlib` wheel for the cluster CUDA stack before installing the rest of the JAX comparison dependencies.

## Branching Guidance

The intended git model is:

- `base` branch: stable PyTorch ADR repository
- `jax-comparison` branch: comparison layer on top of `base`

If a change only exists because the repository compares two frameworks, it belongs conceptually here.
