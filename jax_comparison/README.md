# JAX Comparison Layer

`jax_comparison/` is the comparison workspace built on top of the stable PyTorch ADR baseline.

It is not the canonical project root. Its job is narrower and clearer:

- compare PyTorch and JAX on the same ADR problem
- identify where JAX is faster
- identify where PyTorch remains more reliable
- isolate failure modes through targeted diagnostics

## Main Question

This subtree exists to answer one scientific question:

- under matched ADR protocols, which framework is better for this problem?

That question is different from the one answered by `base/`.

- `base/` asks whether the PyTorch PI-DeepONet works well
- `jax_comparison/` asks how JAX compares to that PyTorch reference

## What Is Included Here

This layer groups:

- the JAX ADR implementation
- comparison-specific PyTorch wrappers used for matched protocols
- benchmark harnesses and benchmark-oriented configs
- launchers for comparison studies
- figures and assets tied specifically to the framework comparison

It is intentionally separated from `base/` so that the stable PyTorch workflow does not get buried under comparison material.

## Structure

- [multifamily/](multifamily): main full-task comparison on the three-family ADR problem
- [monofamily/](monofamily): diagnostic studies and targeted ablations

Interpretation:

- `multifamily/` carries the main framework conclusion
- `monofamily/` exists to explain mechanisms, not to replace the main benchmark

## Relationship With The Rest Of The Repository

- `base/` is the stable PyTorch scientific baseline
- `jax_comparison/` is the framework-comparison layer
- `experiments/` is the public registry of reproducible protocols
- `benchmarks/` is the execution layer used by those protocols

## Environment

Install the base environment first:

```bash
pip install -r requirements.txt
```

Then add the JAX comparison dependencies:

```bash
pip install -r requirements-jax.txt
```

For GPU systems and HPC clusters, install the compatible `jax` and `jaxlib` build first, then install the remaining comparison dependencies.

## What Belongs Here

Belongs here:

- JAX implementation details
- benchmark infrastructure used to compare frameworks
- equal-pipeline comparison material
- monofamily comparison diagnostics
- ansatz and LBFGS ablations used to interpret framework behavior

Does not belong here:

- stable PyTorch baseline logic that would remain valuable without any JAX work

That stable material belongs under [base/](../base).

## Branching Guidance

The intended git model is:

- `base` branch: stable PyTorch ADR repository
- `jax-comparison` branch: comparison layer added on top of `base`

If a change only exists because the project compares two frameworks, it belongs conceptually here.
