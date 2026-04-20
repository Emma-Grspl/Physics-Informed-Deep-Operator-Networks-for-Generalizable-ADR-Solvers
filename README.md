# PyTorch Baseline for Physics-Informed DeepONets on the 1D ADR Equation

[![CI](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml/badge.svg)](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml)

This branch is the stable PyTorch baseline of the project.

Its role is simple:

- establish that the PI-DeepONet approach works on the one-dimensional advection-diffusion-reaction problem
- provide the main scientific reference implementation
- expose the key baseline results directly at branch root

If someone opens only this branch during an interview or a technical review, they should be able to understand the problem, the method, the protocol, the results, and the limitations without searching elsewhere.

## Executive Summary

This branch shows that a PyTorch physics-informed DeepONet can learn a reliable surrogate for a parametric 1D advection-diffusion-reaction equation.

Main takeaways:

- the baseline works on the real multifamily ADR task, not only on a toy case
- the best reference result is strong enough to justify the overall project
- the surrogate is much faster at inference than the Crank-Nicolson reference solver
- difficulty is not uniform across initial-condition families, and this matters for interpretation

Reference multifamily result:

- global relative L2: `0.00507 +- 0.00392`
- Tanh: `0.00139 +- 0.00035`
- Sin-Gauss: `0.00978 +- 0.00286`
- Gaussian: `0.00405 +- 0.00100`
- training time: about `5329 s`
- time-jump speedup vs Crank-Nicolson: about `x175`

## Problem Setting

The physical problem is the one-dimensional advection-diffusion-reaction equation

\[
u_t + v\,u_x - D\,u_{xx} = \mu (u-u^3),
\]

where:

- \(v\) is the advection velocity
- \(D\) is the diffusion coefficient
- \(\mu\) controls the nonlinear reaction term

This equation combines three mechanisms:

- transport through advection
- spatial smoothing through diffusion
- local nonlinear dynamics through reaction

The task is parametric. The model must not solve a single fixed PDE instance. It must learn a surrogate over varying physical coefficients and varying initial conditions.

The initial conditions are drawn from several families:

- `Tanh`
- `Sin-Gauss`
- `Gaussian`

That multifamily setting is essential. It is what makes the baseline scientifically interesting and what separates it from a narrow proof-of-concept.

## What This Branch Is Meant To Prove

This branch answers one question:

- can a PyTorch PI-DeepONet serve as a scientifically usable baseline for the parametric ADR problem?

It does not try to answer:

- whether JAX is faster or better
- whether framework-level conclusions hold under matched protocols
- which backend is preferable overall

Those questions belong to the `jax-comparison` branch.

## Model And Training Logic

The baseline model is a physics-informed DeepONet.

It learns an operator that maps:

- physical parameters
- initial-condition parameters
- a query point \((x,t)\)

to the solution value \(u(x,t)\).

The network follows the standard branch/trunk decomposition:

- the branch processes the ADR coefficients and the parameters describing the initial condition
- the trunk processes the space-time coordinates
- the final prediction combines both representations

In this project, the baseline is strengthened by two design choices:

- a multiscale Fourier encoding on the space-time input
- conditional modulation so that the space-time representation adapts to the physical regime

The training objective is physics-informed and combines:

- PDE residual loss
- initial-condition loss
- boundary-condition loss

The reference numerical target used for evaluation is a Crank-Nicolson solver.

## Baseline Protocol

The baseline protocol is not a single flat training loop. It uses a staged logic designed to stabilize learning on a hard parametric PDE.

The main steps are:

1. warm up the model on the initial condition
2. train progressively in time rather than solving the full horizon at once
3. audit the model regularly against the numerical reference
4. identify hard cases and hard families
5. run targeted correction phases when needed
6. finish with a stricter polishing stage

This matters because the branch is meant to demonstrate not only that the model can fit, but that a reproducible and robust training procedure exists.

## Reference Configuration

The canonical baseline uses the following overall structure:

- branch depth: `5`
- trunk depth: `4`
- branch width: `256`
- trunk width: `256`
- latent dimension: `256`
- Fourier features: `20`

Representative training parameters for the stable baseline pipeline:

- batch size: `8192`
- sampled collocation points per draw: `12288`
- warmup iterations: `7000`
- iterations per time step: `8000`
- correction iterations: `9000`
- number of correction loops: `3`
- initial-condition threshold: `0.02`
- per-step threshold: `0.03`
- target horizon in the long baseline workflow: `T_max = 3.0`

The comparison-oriented short benchmark used as the visible reference result in this repository is stricter and smaller:

- target horizon: `T_max = 1.0`
- batch size: `4096`
- sampled collocation points per draw: `4096`
- warmup iterations: `5000`
- iterations per step: `2500`
- correction iterations: `4000`
- number of correction loops: `1`
- evaluation set: `20` cases per family
- benchmark seed: `0`

This distinction is important:

- the long baseline workflow shows the full training philosophy
- the short benchmark provides the clean reference number used for branch-level communication

## Main Results

### Multifamily Reference Result

This is the main result of the branch.

On the strict three-family benchmark with 20 evaluation cases per family:

- global relative L2: `0.00507 +- 0.00392`
- Tanh: `0.00139 +- 0.00035`
- Sin-Gauss: `0.00978 +- 0.00286`
- Gaussian: `0.00405 +- 0.00100`

Interpretation:

- the model is accurate on the actual target task
- the baseline is not limited to a single easy family
- `Sin-Gauss` is the hardest family within the successful multifamily setting

### Inference Value

The baseline is useful not only because it is accurate, but because it is fast once trained.

On the reference benchmark:

- PyTorch training time: about `5329 s`
- full-grid inference time: about `0.210 s`
- time-jump inference time: about `0.00285 s`
- Crank-Nicolson reference time: about `0.499 s`
- time-jump speedup vs Crank-Nicolson: about `x175`

This is the practical value of the branch:

- a reliable surrogate is obtained
- inference becomes dramatically cheaper than the classical solver

## Family-Wise Analysis

The baseline branch is strongest on the multifamily reference task, but family-wise diagnostics are important to understand what is easy and what is hard.

Monofamily PyTorch results:

- `Tanh` only: `0.00158 +- 0.00048`
- `Sin-Gauss` only: about `1.00000`
- `Gaussian` only: `0.87212 +- 0.01769`

Interpretation:

- `Tanh` is easy for the model
- `Sin-Gauss` is intrinsically difficult
- `Gaussian` is also hard when learned freely in isolation

These monofamily numbers are not the main headline of the branch, but they explain why the multifamily result is nontrivial.

## Ansatz-Focused Results

The repository also contains targeted experiments showing that explicit structure on the initial condition can strongly change performance.

Key PyTorch ansatz results:

- `Sin-Gauss` with ansatz: `0.86448 +- 0.13245`
- `Gaussian` with ansatz: `0.07643 +- 0.02998`

Interpretation:

- the ansatz brings only a limited gain on `Sin-Gauss`
- the ansatz changes the Gaussian case dramatically
- a large part of the Gaussian difficulty comes from how the initial condition is represented and enforced

This is an important scientific message of the repository:

- performance depends not only on optimizer choice or raw model size
- it also depends strongly on how the initial condition is embedded into the learning problem

## What This Branch Establishes

This branch supports the following conclusions:

- a PI-DeepONet can learn the ADR operator with strong accuracy
- the PyTorch implementation is reliable enough to serve as the reference baseline
- multifamily generalization is achievable on the target task
- inference speedup is one of the practical strengths of the surrogate

## Limitations

The branch is strong as a baseline, but it still has limitations that should be stated clearly.

Scientific limitations:

- difficulty varies sharply across initial-condition families
- the baseline demonstrates empirical success, not a theoretical explanation of generalization
- some targeted hard cases remain far less satisfactory than the main multifamily result

Repository limitations:

- the repository still contains integration-era folders inherited from broader experimental work
- the branch is readable, but not yet a perfectly minimal standalone package
- some supporting material remains in the repository because the project evolved through active research rather than through a clean-slate product design

These limitations do not weaken the main baseline conclusion, but they help explain the current repository shape.

## What To Read In This Branch

If you want a fast scientific overview:

1. this README
2. the result figures under `assets/`
3. the baseline subtree under `base/`

If you want the implementation details:

1. `base/`
2. `base/configs/`
3. `base/scripts/`
4. `src/` and `configs/` for shared runtime infrastructure

The intended reading rule is simple:

- this root README is the complete branch-level summary
- the rest of the branch exists to support or reproduce it

## Environment

Use the PyTorch environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This baseline branch is intended to remain usable without JAX-specific dependencies.
