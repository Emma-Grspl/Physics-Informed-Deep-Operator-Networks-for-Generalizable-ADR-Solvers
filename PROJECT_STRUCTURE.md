# Internal Project Structure Note

This file is an internal maintenance note.

It is not a primary entry point for readers of the repository. Start with the root `README.md`, then `base/README.md` or `jax_comparison/README.md` instead.

The repository is organized around three main entry points:

- `base/`
  - canonical PyTorch ADR implementation and base analysis workflow
- `jax_comparison/multifamily/`
  - strict PyTorch vs JAX comparison on the main multi-family task
  - includes dedicated `results/`, `plots/`, and `assets_multifamily/`
- `jax_comparison/monofamily/`
  - exploratory mono-family comparison runs and focused diagnostics
  - includes dedicated `results/`, `plots/`, and `assets_monofamily/`

Central plot hub:
- `plot/`
  - `Pytorch/`
  - `PI_DeepOnet_Base_Analyse/`
  - `Classical_Solver/`
  - `Jax_Vs_Pytorch_Comparison/Multifamily/`
  - `Jax_Vs_Pytorch_Comparison/Monofamily/`

Current navigation rule:
- `base/` is the human-facing home for the PyTorch baseline, Optuna tuning, saved model, and base plots.
- `jax_comparison/` is the human-facing home for all framework comparisons.
- `plot/` is the central visual gallery.
- benchmark result folders are now organized under `jax_comparison/multifamily/results/` and `jax_comparison/monofamily/results/`

Legacy runtime note:
- The historical root-level folders (`src/`, `src_jax/`, `configs/`, `configs_jax/`, `launch/`, `benchmarks/`, `scripts/`, `test/`) are still present because several benchmark scripts still import them directly at runtime.
- They should now be treated as compatibility infrastructure, not as the main navigation entry points.
- A deeper refactor is still required before those root runtime folders can be safely removed.
