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

## Executive Summary

This subtree exists because the answer to "does the model work?" is not the same as the answer to "which framework is better?".

The central outcome of the comparison is:

- JAX is much faster in raw training time
- PyTorch is much better in final solution quality on the real multifamily ADR task

So this subtree is not a showcase of JAX beating PyTorch. It is a structured investigation of why a faster framework can still be the worse scientific choice for this problem.

## Main Results

### Strict Multifamily Comparison

This is the most important result in the entire comparison layer.

PyTorch on the strict three-family benchmark:

- global relative L2: `0.00507 +- 0.00392`
- Tanh: `0.00139 +- 0.00035`
- Sin-Gauss: `0.00978 +- 0.00286`
- Gaussian: `0.00405 +- 0.00100`
- training time: about `5329 s`

JAX on the matched strict three-family benchmark:

- global relative L2: about `1.67`
- Tanh: about `1.24`
- Sin-Gauss: about `2.64`
- Gaussian: about `1.13`
- training time: about `349 s`

Interpretation:

- JAX is dramatically faster in raw wall-clock training time
- that speed does not translate into a usable final surrogate under the matched protocol
- PyTorch is therefore the framework that carries the main scientific conclusion of the project

### Monofamily Diagnostics

The monofamily studies are not the main benchmark, but they are crucial for interpretation.

They show that:

- difficulty is not distributed uniformly across families
- `Tanh` is much easier than the other families
- `Sin-Gauss` and `Gaussian` remain genuinely difficult even when isolated
- the gap between frameworks is not only a multifamily effect

This matters because it prevents an overly simple interpretation such as "JAX only failed because the task was too broad".

### Gaussian Hypothesis Ablation

The Gaussian ablation isolates two factors:

- free learning versus ansatz for the initial condition
- with and without an L-BFGS finisher

Main aggregated results:

- PyTorch free / no LBFGS: `0.8239 +- 0.0611`
- PyTorch ansatz / no LBFGS: `0.1606 +- 0.0841`
- JAX free / no LBFGS: `1.0065 +- 0.0060`
- JAX ansatz / no LBFGS: `0.4814 +- 0.0056`

Interpretation:

- the ansatz is the dominant helpful factor
- L-BFGS does not provide a robust gain in the tested setting
- PyTorch still remains clearly ahead in final error

## Structure

- [multifamily/](multifamily): main full-task comparison on the three-family ADR problem
- [monofamily/](monofamily): diagnostic studies and targeted ablations

Interpretation:

- `multifamily/` carries the main framework conclusion
- `monofamily/` exists to explain mechanisms, not to replace the main benchmark

## What A Recruiter Or External Reader Should Take Away

If someone scans this branch quickly, the message should be:

- the project does not stop at "we trained a model"
- it evaluates framework choice seriously
- it measures both speed and quality
- it investigates failure modes rather than hiding them

That is the professional value of this branch.

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

## Main Limitations

This subtree is informative, but it is also where the main limitations of the project become visible.

### Scientific limitations

- the matched JAX pipeline is not competitive in final quality in the current study
- some hard families remain difficult even after narrowing the task
- the comparison is strong empirically, but it does not by itself prove a framework-level theorem

### Practical limitations

- this comparison layer still depends on parts of the broader repository infrastructure
- some comparison logic is layered on top of compatibility-era folders rather than a perfectly minimal codebase
- results are clear, but the repository still reflects an active research workflow rather than a polished product package

These limitations are important to state explicitly. They make the branch more credible, not less.

## Branching Guidance

The intended git model is:

- `base` branch: stable PyTorch ADR repository
- `jax-comparison` branch: comparison layer added on top of `base`

If a change only exists because the project compares two frameworks, it belongs conceptually here.
