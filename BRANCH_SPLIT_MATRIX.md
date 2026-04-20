# Internal Branch Split Matrix

This file is an internal maintenance note about ownership and branch separation.

It is not part of the main user-facing documentation of the repository.

This matrix defines the intended ownership of each top-level namespace.

## Rules

- `base`: belongs to the stable PyTorch branch
- `jax-comparison`: belongs to the comparison branch layered on top of `base`
- `legacy`: keep temporarily for compatibility, but do not treat as the long-term source of truth

## Top-Level Matrix

| Path | Target branch | Status | Rationale |
|---|---|---|---|
| `base/` | `base` | keep | Canonical PyTorch ADR pipeline |
| `requirements.txt` | `base` | keep | Base PyTorch environment |
| `models_saved/` | `base` | legacy or fold into `base/` | Historical base artifacts at repo root |
| `assets/` | `base` | keep | Presentation assets mainly tied to project conclusions |
| `src_jax/` | `jax-comparison` | keep | JAX implementation |
| `jax_comparison/` | `jax-comparison` | keep | Main comparison workspace |
| `requirements-jax.txt` | `jax-comparison` | keep | Comparison-layer dependencies |
| `benchmarks/` | `jax-comparison` | keep | Comparison harness, including PyTorch benchmark wrappers |
| `jax/` | `jax-comparison` | legacy results | Local copied JAX benchmark outputs, not source code |
| `experiments/` | `jax-comparison` | transitional keep | Better human-facing organization of comparison protocols |
| `plot/gaussian_hypothesis/` | `jax-comparison` | keep | Comparison-specific analysis outputs |
| `scripts/plot_jax_vs_pytorch_comparison.py` | `jax-comparison` | keep | Comparison-only plotting |
| `scripts/plot_monofamily_comparison.py` | `jax-comparison` | keep | Comparison-only plotting |
| `scripts/plot_monofamily_ansatz_comparison.py` | `jax-comparison` | keep | Comparison-only plotting |
| `scripts/analyze_gaussian_hypothesis_results.py` | `jax-comparison` | keep | Comparison-only analysis |
| `src/` | `legacy` | transitional keep | Active compatibility layer still used by benchmarks and legacy launchers |
| `configs/` | `legacy` | transitional keep | Mixed PyTorch base and comparison-era configs at repo root |
| `configs_jax/` | `jax-comparison` | transitional keep | JAX comparison configs still referenced by legacy launchers |
| `launch/` | `legacy` | transitional keep | Mixed base and comparison launchers |
| `scripts/train.py` | `base` | wrapper keep | Compatibility wrapper to `base/scripts/train.py` |
| `scripts/tune_optuna.py` | `base` | wrapper keep | Compatibility wrapper to `base/scripts/tune_optuna.py` |
| `test/` | `legacy` | inspect later | Unclear ownership until test coverage is consolidated |
| `results/` | neither | runtime data | Generated outputs, shared by both tracks |
| `README.md` | shared | keep | Repository-level orientation |

## Practical Interpretation

### What should define branch `base`

Primary source of truth:

- `base/`
- `requirements.txt`
- root documentation

Allowed compatibility layer:

- `scripts/train.py`
- `scripts/tune_optuna.py`
- `launch/launch.slurm`
- `launch/launch_optuna.slurm`

Those wrappers may remain for convenience, but they should point into `base/`.

### What should define branch `jax-comparison`

Primary source of truth:

- `src_jax/`
- `jax_comparison/`
- `benchmarks/`
- `requirements-jax.txt`
- comparison analysis scripts

Optional transitional support:

- `configs_jax/`
- `experiments/`
- selected comparison outputs if you want the branch to remain reproducible locally

## Legacy Cleanup Order

1. Stop adding new logic into root `src/`, `configs/`, and `launch/`
2. Keep `base/` as the PyTorch source of truth
3. Keep `benchmarks/` and `src_jax/` on the comparison side
4. Replace only safe top-level entry points with wrappers
5. Remove or archive root duplicates after branch creation

## Branch Creation Guidance

Create `base` only after you accept that `base/` is the canonical PyTorch source tree.

Create `jax-comparison` from that `base` branch, then add:

- `src_jax/`
- `jax_comparison/`
- `benchmarks/`
- `requirements-jax.txt`
- comparison docs and plots

Do not build either branch directly from the current mixed root without this ownership model, otherwise the ambiguity just gets frozen into git history.
