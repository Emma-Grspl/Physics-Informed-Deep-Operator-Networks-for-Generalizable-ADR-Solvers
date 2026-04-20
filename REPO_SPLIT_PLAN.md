# Internal Repo Split Plan

This file is an internal maintenance note about branch separation and cleanup history.

It is not part of the main user-facing repository documentation.

## Goal

Produce two clean logical branches from the current mixed repository:

- `base`: stable PyTorch ADR pipeline
- `jax-comparison`: JAX comparison layer built on top of `base`

This document is the working inventory used to decide what belongs where.

## Current Situation

The repository currently contains three overlapping layouts:

1. active top-level runtime layout
   - `src/`, `src_jax/`, `configs/`, `configs_jax/`, `launch/`, `scripts/`, `benchmarks/`
2. a self-contained stable PyTorch copy
   - `base/`
3. comparison-oriented organization
   - `jax_comparison/`
   - `experiments/`

The main source of confusion is duplication between the top-level runtime layout and `base/` / `jax_comparison/`.

## Keep / Move / Legacy Matrix

### Keep in `base`

These are the stable PyTorch ADR assets and should define the future `base` branch:

- `base/src/`
- `base/scripts/`
- `base/configs/`
- `base/launch/`
- `base/tests/`
- `base/models_saved/`
- `base/assets_pytorch/`
- `base/plots/`
- `requirements.txt`
- `base/README.md`

Reason:

- this subtree is already internally coherent
- it represents the canonical PyTorch workflow
- it can become the stable public-facing branch with minimal ambiguity

### Keep in `jax-comparison`

These are comparison-specific and should define the future `jax-comparison` branch on top of `base`:

- `src_jax/`
- `jax_comparison/`
- `requirements-jax.txt`
- `benchmarks/`
- JAX benchmark launchers and configs
- comparison plots and benchmark aggregation scripts
- `scripts/plot_jax_vs_pytorch_comparison.py`
- `scripts/plot_monofamily_comparison.py`
- `scripts/plot_monofamily_ansatz_comparison.py`
- `scripts/analyze_gaussian_hypothesis_results.py`

Reason:

- these files exist only because the repository compares frameworks or benchmark protocols

### Keep as integration-only for now

These top-level folders still matter because active scripts and launchers refer to them directly:

- `src/`
- `configs/`
- `launch/`
- `scripts/train.py`
- `scripts/tune_optuna.py`

Reason:

- they are still the compatibility layer used by current entry points
- removing them immediately would break existing commands

Target action:

- in `base`, replace these with thin wrappers or delete them after all paths are switched to `base/`
- in `jax-comparison`, keep only what is required by benchmark entry points

### Legacy / duplicate candidates

These are structurally useful during the cleanup but should not remain duplicated forever:

- `configs/` vs `base/configs/`
- `launch/` vs `base/launch/` and `jax_comparison/*/launch/`
- `src/` vs `base/src/`
- `jax_comparison/*/configs/` vs `experiments/*/configs/`
- `jax_comparison/*/launch/` vs `experiments/*/launch/`

Reason:

- same information exists in multiple places
- maintenance cost is too high
- branch boundaries stay blurry while those duplicates coexist

## Proposed Branch Construction

### Branch `base`

Contents to expose:

- `base/` as the canonical package
- root `README.md` oriented toward the PyTorch solver first
- `requirements.txt`

Short-term compatibility option:

- keep minimal top-level wrappers pointing into `base/`
- example: top-level `scripts/train.py` can delegate to `base/scripts/train.py`

### Branch `jax-comparison`

Start from `base`, then add:

- `src_jax/`
- `jax_comparison/`
- `benchmarks/`
- comparison plotting/aggregation scripts
- `requirements-jax.txt`

This branch should not pretend to be the canonical solver branch. It is the experimental comparison branch.

## Immediate Practical Next Steps

1. freeze documentation and dependency split
2. decide whether `base/` becomes the only source of truth for the PyTorch workflow
3. add compatibility wrappers at the top level if needed
4. stop adding new experiment definitions into duplicated legacy trees
5. create git branches only after step 2 is accepted

## Recommendation

Do not create the final public `base` and `jax-comparison` branches until the repository has one clear source of truth for:

- PyTorch source code
- configs
- launchers

Right now, the correct source-of-truth candidate is:

- `base/` for the PyTorch branch
- `jax_comparison/` plus `src_jax/` for the comparison branch

## Wrapper Candidates

### Safe now

These top-level files are exact duplicates of their `base/` counterparts and can be turned into wrappers immediately:

- `scripts/train.py` -> `base/scripts/train.py`
- `scripts/tune_optuna.py` -> `base/scripts/tune_optuna.py`
- `launch/launch.slurm` -> `base/scripts/train.py`
- `launch/launch_optuna.slurm` -> `base/scripts/tune_optuna.py`

Status:

- converted to compatibility wrappers

### Not safe yet

These paths are still used too broadly or contain comparison-specific material mixed with legacy runtime assumptions:

- `src/`
- `configs/`
- most of `launch/`

They need a second-pass refactor rather than a blind wrapper conversion.

### Partially migrated benchmark scripts
Benchmark migration to `base/src` was reverted.

Reason:

- the benchmark harness belongs conceptually to the comparison layer, not to the stable base pipeline
- `trainer_ADR.py` is the canonical PyTorch training workflow
- `trainer_ADR_benchmark.py` and benchmark-oriented PyTorch scripts are comparison artifacts
