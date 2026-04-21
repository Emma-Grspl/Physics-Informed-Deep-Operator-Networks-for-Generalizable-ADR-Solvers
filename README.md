# JAX vs PyTorch Comparison for a Physics-Informed DeepONet on the 1D ADR Equation

[![CI](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml/badge.svg)](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml)

This branch is the dedicated comparison branch of the project. Its purpose is not to establish whether a PI-DeepONet works on the ADR problem in absolute terms. That question is addressed in the `base` branch. The purpose here is narrower and more empirical: compare two closely matched implementations of the same scientific workflow, one in PyTorch and one in JAX, and determine which framework delivers the better practical outcome on the parametric ADR task.

## Introduction

The physical problem studied in this branch is the one-dimensional advection-diffusion-reaction equation

\[
u_t + v\,u_x - D\,u_{xx} = \mu (u-u^3).
\]

This equation describes the evolution of a quantity \(u(x,t)\) under the combined action of three physical mechanisms.

- \(u_t\) represents the time evolution of the solution.
- \(v\,u_x\) is the advective transport term, controlled by the velocity \(v\).
- \(D\,u_{xx}\) is the diffusion term, controlled by the diffusion coefficient \(D\).
- \(\mu (u-u^3)\) is a cubic nonlinear reaction term, controlled by \(\mu\).

ADR equations arise in many modeling settings involving simultaneous transport, diffusion, and local transformation, including concentration transport, simplified reaction systems, and broader parametric PDE benchmark problems. In this project, the goal is not to solve one fixed instance, but to learn a surrogate that generalizes across a family of physical coefficients and initial conditions.

## Branch Structure

This branch is organized so that the comparison can be understood from the repository root.

- `code/` contains the source code, benchmark runners, configurations, launch scripts, and tests required for the comparison.
- `code/src/` contains the PyTorch implementation used on the comparison side.
- `code/src_jax/` contains the JAX implementation.
- `code/benchmarks/` contains standardized training, evaluation, and inference runners.
- `code/configs/` and `code/configs_jax/` contain the PyTorch and JAX model configurations.
- `code/experiments/` contains the experiment registry and protocol structure.
- `code/code_experiments/` contains comparison-specific plotting and synthesis scripts.
- `code/launch/` contains the SLURM launchers.
- `figures/` contains the full comparison figure sets.
- `assets/` contains a shorter curated selection of the most representative visuals.
- `models/` contains the versioned PyTorch reference checkpoint stored at branch root.

The organizing principle is straightforward: `code/` runs the comparison, `figures/` stores the detailed outputs, `assets/` surfaces the most useful visuals, and `models/` holds the root-level versioned checkpoint.

## Parametric ADR Problem Formulation

The problem is parametric rather than single-instance. The model is therefore asked to learn a family of solutions that depends on both physical coefficients and the shape of the initial condition.

The physical coefficients vary over the following ranges:

- \(v \in [0.5, 1.0]\)
- \(D \in [0.01, 0.2]\)
- \(\mu \in [0.0, 1.0]\)

The initial conditions are also parameterized:

- amplitude \(A \in [0.7, 1.0]\)
- center \(x_0\), used to position the profile
- width \(\sigma \in [0.4, 0.8]\)
- frequency or slope \(k \in [1.0, 3.0]\), depending on the family

Three main families of initial conditions are considered:

- `Tanh`
- `Sin-Gauss`
- `Gaussian`

The domain is one-dimensional in space, and the time horizon depends on the protocol. The main strict comparison is carried out on a short horizon with \(T_{\max}=1.0\), so that the comparison remains closely matched between frameworks. The numerical reference discretization is chosen to produce a reliable ground truth against which both neural implementations can be evaluated.

This makes the task a genuine operator-learning problem: the goal is not just to predict one solution, but to learn a mapping from physical parameters, initial-condition parameters, and space-time coordinates \((x,t)\) to the value of the solution.

## Reference Numerical Solver: Crank-Nicolson

The reference numerical solver used throughout this branch is a Crank-Nicolson solver. It provides the numerical ground truth against which the networks are evaluated.

Crank-Nicolson is a classical implicit time discretization scheme for time-dependent PDEs. Its main advantage is that it offers a good balance between numerical stability, temporal accuracy, and robustness. In this branch, it serves three essential purposes:

- generating the reference solutions used to measure the network error,
- providing a timing baseline for inference comparisons,
- anchoring the quality assessment of the learned surrogate to a standard numerical method.

This reference is essential because a framework comparison only makes sense when both implementations are assessed against the same numerical target. The goal is therefore not to replace Crank-Nicolson as a reference method, but to compare how well PyTorch and JAX can approximate its solutions quickly and accurately.

## Neural Network Description

The model used in this branch is a Physics-Informed Deep Operator Network, or PI-DeepONet. It is designed to learn an operator that maps problem parameters and space-time coordinates to the solution value.

The architecture relies on a branch/trunk decomposition.

- The `branch` network encodes the physical parameters and the parameters describing the initial condition.
- The `trunk` network encodes the evaluation coordinates \((x,t)\).

The final output is obtained through the interaction of both representations.

In the main comparison, the architecture is intentionally aligned between PyTorch and JAX:

- branch depth: `5`
- trunk depth: `4`
- branch width: `256`
- trunk width: `256`
- latent dimension: `256`
- number of Fourier features: `20`

The training logic remains physics-informed in both frameworks. The loss combines:

- a PDE residual term,
- an initial-condition term,
- a boundary-condition term.

In the main protocol, the collocation and batch settings are matched across frameworks:

- `batch_size = 4096`
- `n_sample = 4096`

The core optimizer is Adam-like in both workflows, combined with controlled temporal progression, audits, and targeted correction. In some ablations, an additional L-BFGS refinement stage is tested, especially in the `Gaussian Hypothesis` study.

## JAX vs PyTorch Protocol

The protocol in this branch is designed to answer one specific question: under closely matched architecture and training logic, which framework produces the better scientific result on the parametric ADR problem?

The comparison targets the following quantities:

- final accuracy on the main multifamily task,
- family-wise error across initial-condition types,
- total training time,
- inference time relative to Crank-Nicolson,
- robustness of the conclusion in monofamily and ansatz-based settings.

To make the comparison credible, the two frameworks share the same high-level training philosophy:

1. an initial-condition warmup phase,
2. progressive time training,
3. repeated audits against the Crank-Nicolson reference,
4. targeted correction on hard regimes,
5. optional final refinement.

The main multifamily protocol uses in particular:

- `T_max = 1.0`
- `n_warmup = 5000`
- `n_iters_per_step = 2500`
- `n_iters_correction = 4000`
- `nb_loop = 1`
- `40` global audit cases
- `12` audit cases per family
- `20` evaluation cases per family
- `seed = 0` for the primary benchmark

The retained comparison metrics are:

- global relative \(L^2\) error,
- family-wise relative \(L^2\) error,
- total training time,
- full-grid inference time,
- time-jump inference time,
- speedup relative to Crank-Nicolson.

This choice of metrics is deliberate. A faster framework is not automatically better if the final surrogate is not scientifically usable. Likewise, a single global error is not enough to explain failure modes, which is why family-wise analysis, monofamily runs, and ansatz experiments are included.

The comparison is reproducible through:

- explicit configuration files for each backend,
- standardized benchmark runners,
- fixed benchmark seeds,
- separated training, evaluation, and inference outputs.

## Numerical Results

The main branch-level result is unambiguous: in the tested protocol, JAX is much faster, but PyTorch is far better in final accuracy.

### Main Multifamily Result

On the strict multifamily benchmark:

PyTorch obtains:

- global relative \(L^2\): `0.00507 +- 0.00392`
- `Tanh`: `0.00139 +- 0.00035`
- `Sin-Gauss`: `0.00978 +- 0.00286`
- `Gaussian`: `0.00405 +- 0.00100`
- training time: about `5329 s`
- time-jump speedup relative to Crank-Nicolson: about `x175`

JAX obtains:

- global relative \(L^2\): `1.66884 +- 1.62812`
- `Tanh`: `1.23642 +- 0.15997`
- `Sin-Gauss`: `2.63937 +- 2.54170`
- `Gaussian`: `1.13073 +- 0.21905`
- training time: about `349 s`
- time-jump speedup relative to Crank-Nicolson: about `x45`

The interpretation is direct:

- JAX is highly advantageous in raw training time,
- this gain does not translate into competitive final solution quality,
- in this branch, PyTorch is the only backend that delivers a genuinely reliable multifamily surrogate.

### Family-Wise Analysis

The monofamily experiments help distinguish easy cases from hard cases.

PyTorch results:

- `Tanh` only: `0.00158 +- 0.00048`
- `Sin-Gauss` only: about `1.00000`
- `Gaussian` only: `0.87212 +- 0.01769`

JAX results:

- `Tanh` only: `9.07159 +- 15.67342`
- `Sin-Gauss` only: `1.39526 +- 0.41086`
- `Gaussian` only: `1.02204 +- 0.05012`

This shows that:

- `Tanh` is learned very well by PyTorch but is unstable in the tested JAX setup,
- `Sin-Gauss` is difficult for both frameworks,
- freely learned `Gaussian` remains hard,
- the framework gap is not only a multifamily generalization effect, but also appears on isolated families.

### Ansatz Results

The ansatz experiments are designed to enforce more structure on the initial condition and test the effect of that structure on learning.

PyTorch ansatz results:

- `Sin-Gauss` with ansatz: `0.86448 +- 0.13245`
- `Gaussian` with ansatz: `0.07643 +- 0.02998`

JAX ansatz results:

- `Sin-Gauss` with ansatz: `0.89959 +- 0.13354`

The interpretation is the following:

- the ansatz helps little on `Sin-Gauss`,
- the ansatz helps strongly on `Gaussian`,
- a substantial part of the difficulty comes from how the initial condition is represented inside the learning problem.

### Gaussian Hypothesis Ablation

The `Gaussian Hypothesis` ablation isolates two factors:

- free learning versus ansatz on the initial condition,
- with and without an L-BFGS finisher.

Aggregated results over three seeds:

- PyTorch free / no LBFGS: `0.8239 +- 0.0611`
- PyTorch free / LBFGS: `0.8658 +- 0.0745`
- PyTorch ansatz / no LBFGS: `0.1606 +- 0.0841`
- PyTorch ansatz / LBFGS: `0.2114 +- 0.1335`
- JAX free / no LBFGS: `1.0065 +- 0.0060`
- JAX free / LBFGS: `1.0065 +- 0.0059`
- JAX ansatz / no LBFGS: `0.4814 +- 0.0056`
- JAX ansatz / LBFGS: `0.4802 +- 0.0056`

This ablation shows that:

- the ansatz is the dominant helpful factor,
- L-BFGS does not provide a robust gain in this setting,
- PyTorch remains clearly ahead of JAX in final error.

### Global Result Summary

At the branch level, the main scientific message is simple:

- JAX accelerates training substantially,
- that acceleration comes with a major loss in final quality,
- PyTorch is the reference backend for obtaining a credible surrogate on the multifamily task,
- monofamily and ansatz experiments show that initial-condition structure plays a central role in the difficulty of the problem.

## Conclusion

This branch shows that framework choice is not a minor implementation detail. In this project, it strongly affects the final scientific quality of the results.

The conclusion can be summarized as follows:

- PyTorch is slower, but much more reliable,
- JAX is faster, but does not reach the required level of accuracy on the main task in the tested setup,
- the families of initial conditions do not have uniform difficulty,
- ansatz-based structure is a more meaningful lever than L-BFGS in the hardest regimes studied here.

In other words, this branch does not simply rank two frameworks. It also identifies which parts of the parametric ADR problem drive the difficulty of learning, and why some modeling choices are much more consequential than others.

## Branch Usage

The branch is organized so that the comparison workflow can be run from a small set of clear entry points.

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-base.txt
pip install -r requirements-jax.txt
```

### Run tests

```bash
pytest -q code/tests
```

### Run the default PyTorch comparison benchmark

```bash
python code/benchmarks/pytorch/train_fulltrainer_benchmark.py
```

### Run the default JAX comparison benchmark

```bash
python code/benchmarks/jax/train_fulltrainer_benchmark.py
```

### Run evaluation

```bash
python code/benchmarks/pytorch/eval_benchmark.py
python code/benchmarks/jax/eval_benchmark.py
```

### Run inference timing

```bash
python code/benchmarks/pytorch/inference_benchmark.py
python code/benchmarks/jax/inference_benchmark.py
```

### Generate comparison plots

```bash
python code/code_experiments/plot_jax_vs_pytorch_comparison.py
python code/code_experiments/plot_monofamily_comparison.py
python code/code_experiments/plot_monofamily_ansatz_comparison.py
python code/code_experiments/analyze_gaussian_hypothesis_results.py
```

### Useful directories

- `code/benchmarks/` contains the benchmark runners.
- `code/configs/` and `code/configs_jax/` contain the model configurations.
- `code/experiments/` contains the experiment registry.
- `figures/` contains the full comparison figure sets.
- `assets/` contains a compact visual selection.
- `models/` contains the main versioned reference checkpoint at branch root.

This branch is meant to be read as a self-contained comparison branch: runnable code, curated figures, and a root README exposing the main scientific conclusions directly.
