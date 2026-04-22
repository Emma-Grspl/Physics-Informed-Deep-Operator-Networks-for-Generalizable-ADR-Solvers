# Base PyTorch ADR Pipeline

This branch is the PyTorch reference branch of the project. It presents the canonical PI-DeepONet pipeline for the parametric advection-diffusion-reaction equation, without the additional organizational layer dedicated to the full JAX versus PyTorch comparison.

## Introduction

The physical problem studied in this branch is a one-dimensional advection-diffusion-reaction equation of the form

$$
u_t + v\,u_x - D\,u_{xx} = \mu (u-u^3).
$$

The quantity \(u(x,t)\) depends on space \(x\) and time \(t\). The different terms of the equation have a straightforward physical interpretation.

- $\(u_t\)$ describes the temporal evolution of the solution.
- $\(v\,u_x\)$ represents advection, that is, transport of the quantity of interest.
- $\(D\,u_{xx}\)$ represents diffusion, which tends to smooth the solution spatially.
- $\(\mu (u-u^3)\)$ represents a cubic nonlinear reaction term.

This type of equation appears in problems where a quantity is simultaneously transported, diffused, and locally transformed. In this project, the goal is not to solve a single isolated case, but to construct a neural surrogate able to generalize across a family of parametric ADR problems.

## Branch Structure

This branch is organized as a baseline-only branch. It is not intended to carry the full cross-framework comparison layer. Its purpose is to make the reference PyTorch pipeline readable and reproducible.

- `code/`: source code, configurations, launch scripts, and tests
- `figures/`: reference figures for the classical solver, the PI-DeepONet alone, and the PI-DeepONet versus Crank-Nicolson comparisons
- `assets/`: the most representative visuals for quick reading
- `models/`: reference saved weights
- `README.md`: the full scientific presentation of the branch
- `Makefile`: the usual utility commands
- `requirements-base.txt`: the reference Python environment for this branch

A reader landing on this branch should quickly understand:

1. which physical problem is being studied,
2. which architecture is used,
3. how the training is organized,
4. which results should be treated as the reference PyTorch results.

## Parametric ADR Problem Formulation

The problem studied here is parametric. The network therefore does not learn a single solution, but an operator able to predict the solution for different combinations of physical parameters and initial conditions.

The physical coefficients vary over the following intervals:

- $\(v \in [0.5, 1.0]\)$
- $\(D \in [0.01, 0.2]\)$
- $\(\mu \in [0.0, 1.0]\)$

The initial conditions are also parameterized:

- $\(A \in [0.7, 1.0]\)$
- $\(\sigma \in [0.4, 0.8]\)$
- $\(k \in [1.0, 3.0]\)$
- $\(x_0 = 0\)$

The spatial domain is

$$
x \in [-5, 8],
$$

and the full baseline time horizon is

$$
T_{\max} = 3.0.
$$

The reference discretization used for the baseline audits is:

- $\(N_x = 500\)$
- $\(N_t = 200\)$

Three main families of initial conditions are considered:

- `Tanh`
- `Sin-Gauss`
- `Gaussian`

This diversity of families is essential. It allows the model to be evaluated on qualitatively different regimes rather than on a single initial shape.

## Reference Numerical Solver: Crank-Nicolson

The network predictions are evaluated against a reference Crank-Nicolson solver.

Crank-Nicolson is a classical implicit time discretization scheme for time-dependent partial differential equations. It offers a strong balance between numerical stability and accuracy, which makes it a relevant reference for evaluating a neural surrogate.

In this branch, the Crank-Nicolson solver plays two roles:

- it provides the numerical ground truth against which the network predictions are compared,
- it serves as the temporal baseline used to measure the inference gain of the PI-DeepONet.

This choice matters scientifically because the network is not judged only through its training loss, but against a well-established numerical solution procedure.

## Neural Network Description

The model used in this branch is a PI-DeepONet, that is, a Deep Operator Network trained with an explicit physical constraint through the ADR residual.

The architecture relies on a branch/trunk decomposition.

- The `branch` network encodes the physical parameters and the parameters describing the initial condition.
- The `trunk` network encodes the coordinates of the evaluation point \((x,t)\).
- Both representations are fused through FiLM-style conditional transformations.

The input dimensions are:

- branch: 8 variables \((v, D, \mu, type, A, x_0, \sigma, k)\)
- trunk: 2 variables \((x,t)\)

The reference architecture is:

- branch depth: 5
- trunk depth: 4
- branch width: 256
- trunk width: 256
- latent dimension: 256
- number of Fourier features: 20
- Fourier scales: \(0, 1, 2, 3, 4, 5, 6, 8, 10, 12\)
- activation: SiLU

The trunk uses a multiscale Fourier encoding in order to better represent oscillatory or localized structures, especially for the `Sin-Gauss` and `Gaussian` families.

The loss combines three terms:

- PDE residual loss,
- initial-condition loss,
- boundary-condition loss.

The main optimization is performed with Adam. The branch also includes an L-BFGS finisher in some polishing phases. The reference learning rate is approximately \(6.08 \times 10^{-5}\), with explicit decay during the longer training phases.

The main baseline hyperparameters are:

- batch size: 8192
- number of sampled points: 12288
- warmup: 7000 iterations
- iterations per temporal window: 8000
- correction iterations: 9000
- number of outer loops: 3
- rolling window: 2000
- maximum number of retries: 4

## Base Model Experimental Protocol

The purpose of the baseline training is to build a reliable PyTorch surrogate for the full parametric ADR problem, not just for a simple case or a single family.

The training strategy is progressive. It relies on a temporal curriculum and on several control mechanisms intended to stabilize learning.

The main mechanisms are:

- an initial warmup on the initial condition,
- training over successive temporal windows,
- a `king of the hill` strategy to preserve the best model state,
- rollback and retry when a temporal window does not satisfy the validation criteria,
- adaptive PDE weight adjustment through an NTK-inspired heuristic,
- targeted correction on hard families detected during audits,
- a stricter final polishing phase.

The temporal curriculum is defined through three zones:

- from $\(t=0\)$ to $\(t=0.05\)$: time step $\(0.01\)$
- from $\(t=0.05\)$ to $\(t=0.30\)$: time step $\(0.05\)$
- from $\(t=0.30\)$ to $\(t=3.0\)$: time step $\(0.10\)$

The initial and final loss weights also play an important role:

- initial initial-condition weight: 2000
- final initial-condition weight: 100
- boundary weight: 200
- initial PDE weight: 500

The validation criteria used during audits are:

- initial-condition threshold: 0.02
- temporal-step threshold: 0.03

The central idea behind this protocol is that convergence is defined by the effective quality of the produced solution, not only by the decrease of a global loss. This is what makes the `base` branch a scientific baseline rather than just a training script.

## Numerical Results of the Base Model

The main conclusion of this branch is positive: the PyTorch PI-DeepONet learns a good surrogate for the multifamily ADR problem.

Full multifamily benchmark with 1000 evaluation cases per family:

- `Tanh`: `0.0044 ± 0.0015`
- `Sin-Gauss`: `0.0320 ± 0.0200`
- `Gaussian`: `0.0148 ± 0.0100`

Time jump benchmark:

- full-grid inference time: `3.79 s`
- time-jump inference time: `0.034 s`
- Crank-Nicolson reference time: `0.7 s`
- speedup on time-jump inference: `×21`

The total training time for the corresponding short multifamily benchmark is:

- total time: `5329.21 s`

Scientific interpretation:

- the global accuracy is low enough for the model to be treated as a credible surrogate,
- the model generalizes well across several initial-condition families,
- the `Sin-Gauss` family remains the most difficult one,
- the inference gain justifies the practical interest of the surrogate.

This section contains the main result of the `base` branch.

## Conclusion

The `base` branch supports three main conclusions.

First, a properly structured PyTorch PI-DeepONet can learn an accurate surrogate for the parametric ADR equation. Second, that accuracy comes with a significant inference gain relative to the Crank-Nicolson reference solver. Third, the PyTorch baseline forms the solid scientific foundation of the project, on top of which methodological comparisons and ablations can later be built.

In other words, this branch answers the fundamental question: does the approach truly work on the target problem? The answer is yes.

## Branch Usage

The environment for this branch is intentionally limited to the PyTorch baseline.

Recommended installation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-base.txt
```

Useful commands:

Install dependencies:

```bash
make install
```

Run tests:

```bash
make test
```

Verify source compilation:

```bash
make check
```

Run the main training:

```bash
make train
```

Run the global PI-DeepONet versus Crank-Nicolson analysis:

```bash
make analysis
```

Run the inference benchmark:

```bash
make benchmark
```

Direct entry points:

- training: `python code/scripts/train.py`
- tuning: `python code/scripts/tune_optuna.py`
- tests: `python -m pytest -q code/tests`

Recommended reading path:

1. read this `README.md`,
2. inspect `code/configs/`,
3. read `code/scripts/` and `code/src/training/`,
4. inspect `figures/`, `assets/`, and `models/`.
