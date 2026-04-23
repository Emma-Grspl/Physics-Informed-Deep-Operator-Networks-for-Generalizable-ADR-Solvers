
# Reproducibility Guide

## Overview

This document provides the exact implementation conditions used to reproduce the main experiments of this repository.

It complements:

- `README.md` for project overview
- `docs/experimental_protocol.md` for scientific logic
- `docs/results.md` for final reported metrics

The goal of this file is practical reproducibility.

---

# 1. Compute Environment

## Primary HPC Platform

Main experiments were conducted on **Jean Zay** (IDRIS / GENCI, France).

## GPU Configuration

Typical training runs used:

- **1 × NVIDIA V100 32 GB**
- CUDA-enabled PyTorch environment
- identical GPU class for matched JAX comparisons whenever possible

## CPU Resources

Typical Slurm allocation:

- **10 CPU cores per task**

used for:

- data preparation
- batch generation
- Python runtime overhead

## Wall-Time Constraints

Jean Zay queue policy imposed:

- **20h maximum wall-clock time per submitted run**

This practical constraint strongly influenced experiment design.

---

# 2. Numerical Precision

All scientific experiments were executed in:

- **float64 / double precision**

for both:

- PyTorch
- JAX

This was chosen to improve stability for PDE residuals, second derivatives, and scientific accuracy.

---

# 3. Random Seeds

## Default Seed

Unless explicitly stated:

```text
seed = 0
````

## Multi-Seed Protocols

Some robustness studies used:

```text
seeds = [0]
```

or:

```text
seeds = [0, 1, 2]
```

depending on experiment cost and objective.

## Optuna Note

Optuna studies were created **without an explicitly fixed sampler seed**.

Therefore:

* trial ordering may vary
* sampled hyperparameter sequences may differ across reruns

while remaining statistically comparable.

---

# 4. Physical Problem Configuration

## Geometry

```yaml
T_max = 3.0
x_min = -5.0
x_max = 8.0
```

## Audit Resolution

```yaml
Nt_audit = 200
Nx_audit = 500
Nx_solver = 500
```

These grids were used for:

* benchmark reporting
* final solution comparison
* relative L2 evaluation

---

# 5. Parameter Ranges

The PI-DeepONet was trained on a parametric family of ADR systems.

## PDE Coefficients

```yaml
v     ∈ [0.5, 1.0]
D     ∈ [0.01, 0.2]
mu    ∈ [0.0, 1.0]
```

## Initial Condition Parameters

```yaml
A      ∈ [0.7, 1.0]
sigma  ∈ [0.4, 0.8]
k      ∈ [1.0, 3.0]
x0     = 0.0
```

---

# 6. Model Architecture

## PI-DeepONet Core

```yaml
branch_depth = 5
branch_width = 256

trunk_depth  = 4
trunk_width  = 256

latent_dim   = 256
```

## Fourier Features

```yaml
nFourier = 20
```

Scales:

```yaml
[0,1,2,3,4,5,6,8,10,12]
```

These were used for multiscale spatial encoding.

---

# 7. Loss Weights

```yaml
first_w_res     = 500.0
weight_bc       = 200.0
weight_ic_init  = 2000.0
weight_ic_final = 100.0
```

These values reflect the curriculum strategy:

* strong early enforcement of initial conditions
* later rebalance toward PDE residual quality

---

# 8. Time Curriculum

Training did not directly start on the full horizon.

A progressive temporal curriculum was used.

## Time Zones

```yaml
Zone 1: t ∈ [0, 0.05], dt = 0.01
Zone 2: t ∈ [0.05, 0.30], dt = 0.05
Zone 3: t ∈ [0.30, 3.00], dt = 0.10
```

## Motivation

This curriculum improves:

* optimization stability
* early learning of short-time dynamics
* gradual extension toward harder horizons

---

# 9. Training Hyperparameters

## Main Baseline Setup

```yaml
batch_size          = 8192
n_sample            = 12288

learning_rate       = 6.078744921577277e-05

n_warmup            = 7000
n_iters_per_step    = 8000
n_iters_correction  = 9000

nb_loop             = 3
rolling_window      = 2000
max_retry           = 4

threshold_ic        = 0.02
threshold_step      = 0.03
```

---

# 10. Optimizers

## Main Optimizer

Used everywhere unless stated otherwise:

* **PyTorch:** `torch.optim.Adam`
* **JAX:** `optax.adam`

## Scheduler Policy

No standard built-in scheduler such as:

* StepLR
* CosineAnnealingLR
* ReduceLROnPlateau

was used.

Instead, learning-rate control relied on custom logic.

## Manual LR Logic

### Retry Strategy

When convergence failed:

```text
lr ← lr × 0.5
```

### Final Polishing

Some protocols used manual exponential decay toward:

```text
1e-6
```

---

# 11. Optional L-BFGS Finisher

Some experiments included a second optimization stage:

```text
use_lbfgs_finisher = true
```

Purpose:

* final local refinement
* sharper convergence after Adam pretraining

This was tested as an ablation rather than always enabled.

---

# 12. Optuna Hyperparameter Search

## Number of Trials

```text
50 trials
```

## Runtime Policy

Individual trials were capped in practice by queue limits:

* typically never beyond 20h

## Parallelism

Trials were frequently launched in parallel across available allocations.

## Objective

Searches balanced:

* relative L2 accuracy
* training robustness
* training time

---

# 13. Typical Runtime

## Full PyTorch Long-Horizon Baseline

Approximate full training cost:

```text
~70 hours
```

when trained over the complete target horizon.

## HPC Submitted Runs

Per run limit:

```text
20 hours max
```

## JAX Matched Benchmarks

Substantially faster in training wall-clock time on the matched benchmark protocols.

---

# 14. Matched PyTorch vs JAX Comparison Rules

Framework comparisons attempted to preserve:

* same geometry
* same parameter ranges
* same precision (float64)
* same architecture
* same loss structure
* same evaluation metrics
* same GPU class when possible

This benchmark is therefore intended as a **controlled engineering comparison**, not a universal statement about either framework.

---

# 15. Reproduction Checklist

To reproduce the main baseline:

1. Use float64 precision
2. Use V100-class GPU or similar
3. Apply the temporal curriculum
4. Use Adam with fixed initial LR
5. Apply retry logic and final polishing
6. Evaluate on 500 × 200 audit grids
7. Compare against Crank-Nicolson reference solutions

---

# 16. Important Practical Note

The strongest results in this repository come not from one isolated hyperparameter, but from the combination of:

* curriculum learning
* physics loss balancing
* retry logic
* progressive horizon extension
* carefully tuned architecture

Reproducing only the architecture without the training protocol may lead to significantly weaker results.
