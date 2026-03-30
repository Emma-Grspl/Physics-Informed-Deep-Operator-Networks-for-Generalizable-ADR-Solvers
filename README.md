# Physics-Informed DeepONet for the 1D ADR Equation

[![CI](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml/badge.svg)](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Framework](https://img.shields.io/badge/framework-Pytorch-ee4c2c)

This repository studies whether a Physics-Informed Deep Operator Network can learn a reliable surrogate for a one-dimensional advection-diffusion-reaction equation, and whether Pytorch or JAX is the better framework for that task.

## Physical Problem

The target PDE is

$$
u_t + v u_x = D u_{xx} + \mu u(1-u),
$$

with variable physical parameters:
- advection velocity `v`
- diffusion coefficient `D`
- reaction coefficient `mu`
- parametric initial conditions spanning three families:
  - `Tanh`
  - `Sin-Gauss`
  - `Gaussian`

The goal is not to solve a single trajectory, but to learn an operator

$$
(p, x, t) \mapsto u(x,t;p),
$$

where `p` collects the PDE and initial-condition parameters.

## PI-DeepONet Architecture in Pytorch

The baseline model is a Physics-Informed DeepONet implemented in Pytorch.

- The **branch network** encodes the physical and initial-condition parameters.
- The **trunk network** encodes the query coordinates `(x, t)`.
- Their latent representations are merged through a DeepONet inner-product style coupling.
- The trunk uses **multi-scale Fourier features** to reduce spectral bias and better represent oscillatory patterns.
- The training loss combines:
  - PDE residual loss
  - initial-condition loss
  - boundary-condition loss

The training pipeline also includes:
- time marching across increasing temporal windows
- adaptive loss balancing
- Adam optimization, with optional L-BFGS finishing phases
- rollback / best-state selection
- targeted correction on difficult families

## Research Questions

This repository addresses two concrete questions:

1. **Can a PI-DeepONet learn the ADR operator with sufficient accuracy to replace or accelerate a classical Crank-Nicolson solver?**
2. **For this problem, is Pytorch or JAX the better framework in practice?**

The second question was studied through:
- a strict **same-pipeline** comparison
- exploratory **mono-family** tests
- focused **ansatz-based** tests to isolate the role of the initial-condition learning

## Repository Structure

The repository is organized around two main scientific tracks.

- [base/](base): canonical Pytorch pipeline, Optuna tuning, saved model, and base analysis
- [jax_comparison/multifamily/](jax_comparison/multifamily): strict JAX vs Pytorch comparison on the three-family task
- [jax_comparison/monofamily/](jax_comparison/monofamily): mono-family and ansatz diagnostics
- [plot/](plot): organized figure gallery
- [assets/](assets): short-list showcase figures for the project landing page

Result folders used in the conclusions are now grouped in:
- [jax_comparison/multifamily/results/](jax_comparison/multifamily/results)
- [jax_comparison/monofamily/results/](jax_comparison/monofamily/results)

Legacy runtime folders such as `src/`, `src_jax/`, `configs/`, `configs_jax/`, `launch/`, `benchmarks/`, `scripts/`, and `test/` are still present because some training and benchmark entry points still import them directly.

## Most Important Figures

The main showcase figures copied to [assets/](assets) are:

- [1_base_snapshots.png](assets/1_base_snapshots.png): baseline Pytorch reconstructions against the classical solver
- [2_base_time_jump_speedup.png](assets/2_base_time_jump_speedup.png): baseline time-jump speedup against Crank-Nicolson
- [3_multifamily_family_l2.png](assets/3_multifamily_family_l2.png): strict JAX vs Pytorch family-wise error comparison
- [4_multifamily_snapshots.png](assets/4_multifamily_snapshots.png): strict JAX vs Pytorch reconstructions against CN
- [5_monofamily_same_pipeline_summary.png](assets/5_monofamily_same_pipeline_summary.png): mono-family same-pipeline summary
- [6_monofamily_ansatz_summary.png](assets/6_monofamily_ansatz_summary.png): mono-family ansatz summary

## Baseline Pytorch Results

The baseline Pytorch workflow demonstrates that the PI-DeepONet approach is viable for the ADR equation.

Reference base results:
- `Tanh`: about `0.3%` relative L2
- `Sin-Gauss`: about `2.9%`
- `Gaussian`: about `1.4%`

Reference inference benchmark:
- Crank-Nicolson full reference: about `0.726 s`
- PI-DeepONet direct time jump: about `0.034 s`
- speedup: about `21x`

### Intermediate Conclusion on the Baseline

For the canonical Pytorch setup, the answer to the first scientific question is **yes**:
- the ADR operator can be learned with a physics-informed DeepONet
- the surrogate is accurate enough to be useful
- inference is substantially faster than the classical solver in the direct time-jump setting

## Strict JAX vs Pytorch Comparison on the Three-Family Problem

The strict comparison used the same overall training pipeline for both frameworks at `T_max = 1.0`.

### Pytorch

- training time: `5329.2 s` (`~1.48 h`)
- global relative L2: `0.507%`
- family-wise L2:
  - `Tanh`: `0.139%`
  - `Sin-Gauss`: `0.978%`
  - `Gaussian`: `0.405%`
- inference:
  - full grid: `0.210 s`
  - time jump: `0.00285 s`
  - speedup vs CN: `x175`

### JAX

- training time: `351.5 s` (`~5.9 min`)
- global relative L2: `164.78%`
- family-wise L2:
  - `Tanh`: `122.75%`
  - `Sin-Gauss`: `253.95%`
  - `Gaussian`: `117.65%`
- inference:
  - full grid: `0.248 s`
  - time jump: `0.00973 s`
  - speedup vs CN: `x49.8`

### Intermediate Conclusion on the Strict Comparison

This comparison is the main answer to the second scientific question.

- **Pytorch is clearly the best framework on the actual three-family ADR problem.**
- JAX is much faster in raw training time, but the learned solution is not usable under the same pipeline.
- Pytorch is also faster at inference in this repository.

## Mono-Family Same-Pipeline Diagnostics

To understand whether the JAX failure came only from multi-family generalization, mono-family runs were tested with the same training pipeline.

### Tanh-only

- Pytorch: `0.158%`
- JAX: `907.16%`

### Sin-Gauss-only

- Pytorch: `100.00%`
- JAX: `139.53%`

### Gaussian-only

- Pytorch: `87.21%`
- JAX: `102.20%`

### Intermediate Conclusion on Mono-Family Same-Pipeline Runs

These runs show that the mono-family setting is **not automatically easier** under the same pipeline.

- Pytorch remains excellent only on `Tanh`.
- `Sin-Gauss` and `Gaussian` remain difficult even for Pytorch in this forced mono-family setup.
- JAX remains worse overall.

So these runs are useful diagnostically, but they are not as decisive as the strict multi-family comparison.

## Mono-Family Ansatz Diagnostics

To isolate the role of initial-condition learning, ansatz tests of the form

$$
u(x,t) = u_0(x) + (1-e^{-t})\,NN(x,t,p)
$$

were run.

### Sin-Gauss + Ansatz

- Pytorch:
  - training time: `8331.9 s`
  - relative L2: `86.45%`
  - time jump: `0.00251 s`
- JAX:
  - training time: `2611.6 s`
  - relative L2: `89.96%`
  - time jump: `0.01558 s`

Interpretation:
- both frameworks become much closer
- JAX is no longer catastrophically worse
- this strongly suggests that free initial-condition learning was a major source of difficulty

### Gaussian + Ansatz

- Pytorch:
  - training time: `14448.8 s`
  - relative L2: `7.64%`
  - time jump: `0.00251 s`
- JAX:
  - the run improved substantially and held until `t = 0.4`
  - it failed later around `t = 0.5`
  - local final exported evaluation/inference JSON is missing
  - the Jean Zay training log indicates a partial recovery but still a large final error

Interpretation:
- the ansatz clearly helps `Gaussian`
- Pytorch benefits much more strongly than JAX

### Intermediate Conclusion on the Ansatz Tests

The ansatz experiments refine the interpretation:

- JAX is **not** simply incapable of learning ADR-related dynamics
- removing the burden of freely reconstructing the initial condition makes JAX much more competitive on `Sin-Gauss`
- however, Pytorch remains more robust and more accurate overall

## General Conclusion

The conclusions of the project are:

1. **A PI-DeepONet can solve the 1D ADR operator-learning problem accurately in Pytorch.**
2. **Pytorch is the most reliable framework for this problem in the present repository.**
3. **JAX can be faster in raw optimization time, but under the same pipeline it fails on the main multi-family benchmark.**
4. **Mono-family and ansatz diagnostics show that part of the JAX difficulty is linked to the initial-condition learning stage, not only to the PDE evolution itself.**
5. **Even after these diagnostics, Pytorch remains the better practical choice for this ADR study.**

## How to Run the Repository

### Installation

```bash
git clone https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers.git
cd Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Base Pytorch workflow

```bash
python scripts/train.py
python src/analyse/global_analyse_PI_DeepOnet_vs_CN.py
python src/analyse/inference.py
```

The static base model used by the analysis scripts is:
- [models_saved/PI-DeepOnet.pth](models_saved/PI-DeepOnet.pth)

### Checks

```bash
make test
make check
```

### Benchmark and comparison figures

```bash
/opt/anaconda3/bin/python scripts/plot_jax_vs_pytorch_comparison.py
/opt/anaconda3/bin/python scripts/plot_monofamily_comparison.py
/opt/anaconda3/bin/python scripts/plot_monofamily_ansatz_comparison.py
```

### HPC launchers

Pytorch and JAX launch scripts are kept in:
- [launch/](launch)
- [base/launch/](base/launch)
- [jax_comparison/multifamily/launch/](jax_comparison/multifamily/launch)
- [jax_comparison/monofamily/launch/](jax_comparison/monofamily/launch)

## Author

Emma Grospellier  
PhD research project
