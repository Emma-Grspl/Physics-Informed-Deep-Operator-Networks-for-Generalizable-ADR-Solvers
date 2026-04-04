# Reorganization Plan - Stage 1

This document defines the target structure and the migration map for the repository.
Stage 1 is intentionally non-destructive: it clarifies ownership, freezes the active source of truth, and prepares later moves without breaking current runs.

## Goals

- keep one active implementation per backend
- separate source code from experiment definitions
- separate experiment definitions from results and publication assets
- stop adding new logic into duplicated legacy trees
- make the benchmark and comparison stack easier to maintain

## Current Problems

The repository currently mixes four concerns:

1. source code
2. experiment protocols
3. run outputs
4. publication figures and curated assets

The main structural issue is duplicated code with hidden coupling:

- `base/src/` duplicates the root PyTorch code
- `jax_comparison/monofamily/src/` duplicates comparison code
- `jax_comparison/multifamily/src/` duplicates comparison code
- benchmark runners often import the root `src/` or `src_jax/`, so the duplicated trees are not truly independent

This creates ambiguity about which code is authoritative.

## Stage 1 Decision

During Stage 1, the active source of truth is:

- `src/` for PyTorch
- `src_jax/` for JAX
- `benchmarks/` for standard benchmark runners

The following trees are considered legacy or presentation-oriented and should not receive new core logic:

- `base/src/`
- `jax_comparison/monofamily/src/`
- `jax_comparison/multifamily/src/`

## Target Structure

The long-term target is:

```text
project/
  src/
    adr/
      common/
      pytorch/
      jax/

  experiments/
    base/
    multifamily/
    monofamily/
    ablations/

  benchmarks/
    common/
    runners/
      pytorch/
      jax/

  reports/
    figures/
    scripts/

  results/
  assets/
```

Stage 1 does not yet rename `src/` and `src_jax/`; it prepares for that move.

## Ownership Rules

After Stage 1, each area should have one role only.

### Source Code

Purpose:
- model definitions
- physics residuals
- data generation
- trainers
- reusable utilities

Active today:
- `src/`
- `src_jax/`

### Experiment Definitions

Purpose:
- configs
- launchers
- protocol notes

Target destination:
- `experiments/`

### Benchmarks

Purpose:
- standard training/eval/inference runners
- shared benchmark utilities

Active today:
- `benchmarks/`

### Results and Reports

Purpose:
- run outputs
- figures
- curated assets

Active today:
- `results/`
- `plot/`
- `assets/`
- parts of `base/plots/`
- parts of `jax_comparison/*/plots/`

## Migration Map

This table defines the intended destination or role of each top-level path.

| Current Path | Current Role | Stage 1 Status | Target Role / Target Path |
| --- | --- | --- | --- |
| `src/` | active PyTorch source | keep active | future `src/adr/pytorch/` |
| `src_jax/` | active JAX source | keep active | future `src/adr/jax/` |
| `benchmarks/common/` | shared benchmark utilities | keep active | future `benchmarks/common/` |
| `benchmarks/pytorch/` | benchmark runners | keep active | future `benchmarks/runners/pytorch/` |
| `benchmarks/jax/` | benchmark runners | keep active | future `benchmarks/runners/jax/` |
| `configs/` | mixed experiment configs | freeze as compatibility layer | move into `experiments/*/configs/` |
| `configs_jax/` | mixed JAX experiment configs | freeze as compatibility layer | move into `experiments/*/configs/` |
| `launch/` | mixed launchers | freeze as compatibility layer | move into `experiments/*/launch/` |
| `scripts/` | mixed training and plotting scripts | split by function | training runners stay under `benchmarks/`; plotting moves to `reports/scripts/` |
| `base/` | mixed code, configs, plots, docs | convert to experiment/archive | future `experiments/base/` plus archive notes |
| `base/src/` | duplicated PyTorch code | freeze legacy | remove after import migration |
| `base/configs/` | duplicated configs | freeze legacy | merge into `experiments/base/configs/` |
| `base/launch/` | duplicated launchers | freeze legacy | merge into `experiments/base/launch/` |
| `base/scripts/` | duplicated runners | freeze legacy | replace with benchmark runners or experiment launchers |
| `base/plots/` | figures | archive | move to `reports/figures/base/` if still needed |
| `base/assets_pytorch/` | curated publication assets | keep for now | consolidate into `assets/` or `reports/figures/base/` |
| `jax_comparison/monofamily/` | mixed code, configs, launch, plots, results | convert to experiment/report area | future `experiments/monofamily/` + `reports/figures/monofamily/` |
| `jax_comparison/monofamily/src/` | duplicated comparison code | freeze legacy | remove after import migration |
| `jax_comparison/monofamily/configs/` | monofamily experiment configs | keep as source for move | future `experiments/monofamily/configs/` |
| `jax_comparison/monofamily/launch/` | monofamily launchers | keep as source for move | future `experiments/monofamily/launch/` |
| `jax_comparison/monofamily/plots/` | monofamily figures | archive/output | future `reports/figures/monofamily/` |
| `jax_comparison/monofamily/assets_monofamily/` | curated assets | keep for now | future `assets/` or `reports/figures/monofamily/` |
| `jax_comparison/monofamily/results/` | monofamily results snapshot | archive | keep under `results/` or archive |
| `jax_comparison/multifamily/` | mixed code, configs, launch, plots, results | convert to experiment/report area | future `experiments/multifamily/` + `reports/figures/multifamily/` |
| `jax_comparison/multifamily/src/` | duplicated comparison code | freeze legacy | remove after import migration |
| `jax_comparison/multifamily/configs/` | multifamily experiment configs | keep as source for move | future `experiments/multifamily/configs/` |
| `jax_comparison/multifamily/launch/` | multifamily launchers | keep as source for move | future `experiments/multifamily/launch/` |
| `jax_comparison/multifamily/plots/` | multifamily figures | archive/output | future `reports/figures/multifamily/` |
| `jax_comparison/multifamily/assets_multifamily/` | curated assets | keep for now | future `assets/` or `reports/figures/multifamily/` |
| `jax_comparison/multifamily/results/` | multifamily results snapshot | archive | keep under `results/` or archive |
| `plot/` | central figure gallery | keep temporarily | future `reports/figures/` |
| `assets/` | top-level curated assets | keep active | stay `assets/` |
| `results/` | active benchmark outputs | keep active | stay `results/` |
| `models_saved/` | checkpoint artifacts | keep active | future `results/models/` or `models/` |
| `test/` | root tests | keep active | future `tests/` |
| `base/tests/` | duplicated tests | freeze legacy | merge or remove after consolidation |

## Phase Plan

### Phase 1: Freeze and Label

Actions:

- declare `src/`, `src_jax/`, and `benchmarks/` as active
- declare `base/src/` and `jax_comparison/*/src/` as legacy
- stop adding new trainer/model logic into legacy trees
- keep old paths working for now

Expected result:
- one clear source of truth for code

### Phase 2: Create Experiment Namespace

Actions:

- create `experiments/base/`
- create `experiments/multifamily/`
- create `experiments/monofamily/`
- create `experiments/ablations/gaussian_hypothesis/`
- move configs and launchers there

Expected result:
- experiment protocols are separated from code

### Phase 3: Consolidate Reports

Actions:

- create `reports/figures/`
- create `reports/scripts/`
- move plotting scripts out of `scripts/`
- migrate figure hubs from `plot/`, `base/plots/`, and `jax_comparison/*/plots/`

Expected result:
- figures and publication assets are no longer mixed with runtime code

### Phase 4: Rename Source Package

Actions:

- move `src/` to `src/adr/pytorch/`
- move `src_jax/` to `src/adr/jax/`
- extract shared utilities to `src/adr/common/`
- update benchmark imports

Expected result:
- cleaner package boundary with explicit backend separation

### Phase 5: Remove Legacy Trees

Actions:

- delete `base/src/`
- delete `jax_comparison/monofamily/src/`
- delete `jax_comparison/multifamily/src/`
- remove duplicated tests and duplicate configs no longer used

Expected result:
- no more duplicate code trees

## Immediate Rules For New Work

Until the refactor is complete:

1. new model, trainer, physics, or data logic goes only into `src/` or `src_jax/`
2. new benchmark runners go only into `benchmarks/`
3. new configs should be written with their future experiment group in mind
4. no new code should be added inside `base/src/` or `jax_comparison/*/src/`
5. `plot/` and `jax_comparison/*/plots/` should be treated as outputs, not sources

## Risks

The main risks of the migration are:

- import breakage in launch scripts
- configs still pointing to old paths
- benchmark scripts depending on duplicated files implicitly
- confusion between archived historical results and active outputs

This is why Stage 1 is non-destructive.

## Recommended Next Move

The next implementation step should be:

1. create the `experiments/` tree
2. move one protocol end-to-end first, preferably `monofamily/gaussian_hypothesis`
3. keep compatibility wrappers or symlinks only if strictly necessary
4. verify that benchmark runners still work

This limits the blast radius and tests the reorganization pattern on one bounded slice before touching the full repository.
