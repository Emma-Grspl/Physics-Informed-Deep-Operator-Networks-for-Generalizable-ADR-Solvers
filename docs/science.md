# Science Notes

## Equation ADR

The repository studies a parametric 1D advection-diffusion-reaction equation of the form

$$
u_t + v u_x - D u_{xx} = \mu (u - u^3).
$$

The physical parameters vary over the ranges configured in [base/code/configs/config_ADR.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/base/code/configs/config_ADR.yaml):

- `v in [0.5, 1.0]`
- `D in [0.01, 0.2]`
- `mu in [0.0, 1.0]`
- `A in [0.7, 1.0]`
- `sigma in [0.4, 0.8]`
- `k in [1.0, 3.0]`
- `x in [-5, 8]`

Three initial-condition families are used throughout the baseline and benchmark code:

- `Tanh`
- `Sin-Gauss`
- `Gaussian`

The numerical reference is a Crank-Nicolson ADR solver implemented in [base/code/src/utils/CN_ADR.py](/Users/emma.grospellier/Thèse/Projet_These_ADR/base/code/src/utils/CN_ADR.py).

## Method

The model is a physics-informed DeepONet implemented in [base/code/src/models/PI_DeepONet_ADR.py](/Users/emma.grospellier/Thèse/Projet_These_ADR/base/code/src/models/PI_DeepONet_ADR.py).

Its structure is:

- branch input: physical parameters plus initial-condition descriptors `(v, D, mu, type, A, x0, sigma, k)`
- trunk input: query coordinates `(x, t)`
- trunk encoding: multiscale Fourier features
- fusion: FiLM-style conditional modulation between branch context and trunk activations
- output: scalar prediction `u(x, t)`

Canonical model hyperparameters in the baseline config:

- branch depth: `5`
- trunk depth: `4`
- branch width: `256`
- trunk width: `256`
- latent dimension: `256`
- activation: `SiLU`
- Fourier features: `20`
- Fourier scales: `[0, 1, 2, 3, 4, 5, 6, 8, 10, 12]`

Some comparison experiments also enable an initial-condition ansatz directly in the model, so the network predicts a correction on top of the known initial profile instead of learning the whole field from scratch.

## Loss

The training objective is assembled in [base/code/src/training/trainer_ADR.py](/Users/emma.grospellier/Thèse/Projet_These_ADR/base/code/src/training/trainer_ADR.py) and uses three terms:

- PDE residual loss
- initial-condition loss
- boundary-condition loss

In compact form:

$$
\mathcal{L} = w_r \mathcal{L}_{PDE} + w_i \mathcal{L}_{IC} + w_b \mathcal{L}_{BC}.
$$

The PDE residual is computed in [base/code/src/physics/residual_ADR.py](/Users/emma.grospellier/Thèse/Projet_These_ADR/base/code/src/physics/residual_ADR.py):

$$
r(x,t) = u_t + v u_x - D u_{xx} - \mu (u-u^3).
$$

The baseline trainer also includes an NTK-inspired heuristic to rebalance the PDE term against the IC term during training.

## Training curriculum

The baseline PyTorch workflow is not a single flat training run. It is a staged curriculum:

1. warmup on the initial condition at `t = 0`
2. progressive time marching over larger windows
3. repeated audits against the Crank-Nicolson reference
4. retry logic with reduced learning rate if a phase fails
5. optional L-BFGS finisher
6. targeted correction on failing initial-condition families

The canonical baseline time curriculum from [base/code/configs/config_ADR.yaml](/Users/emma.grospellier/Thèse/Projet_These_ADR/base/code/configs/config_ADR.yaml) is:

- `0.00 -> 0.05` with `dt = 0.01`
- `0.05 -> 0.30` with `dt = 0.05`
- `0.30 -> 3.00` with `dt = 0.10`

The matched PyTorch vs JAX benchmark uses a shorter horizon and a lighter curriculum:

- `0.00 -> 0.30` with `dt = 0.05`
- `0.30 -> 1.00` with `dt = 0.10`

## Generalization setup

Generalization in this repository means more than interpolating one trajectory. The model is asked to generalize across:

- continuous physical coefficients `(v, D, mu)`
- different amplitudes and shape parameters `(A, sigma, k)`
- multiple initial-condition families
- full-grid inference and time-jump inference

Two study regimes are exposed:

- baseline regime in `base/`: canonical PyTorch study with `T_max = 3.0`
- matched comparison regime in `jax_vs_pytorch/`: PyTorch vs JAX study with `T_max = 1.0`

The benchmark evaluation logic is centralized in [jax_vs_pytorch/code/benchmarks/common/eval.py](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_vs_pytorch/code/benchmarks/common/eval.py). It computes:

- global relative `L2`
- family-wise relative `L2`
- inference time on the full grid
- inference time for time jumping
- speedup relative to Crank-Nicolson
