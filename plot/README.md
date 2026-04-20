# Plot Hub

Centralized plot directory for the repository.

This tree stores generated figures and curated visual artifacts. It is an output-facing namespace, not the source of truth for experiment definitions.

Subfolders:
- `Pytorch/`
  - `assets/`: selected PyTorch showcase plots
  - `DeepONet_vs_CN/`: detailed PyTorch vs classical-solver comparison plots
  - `Inference_Time/`: PyTorch inference timing figures
- `PI_DeepOnet_Base_Analyse/`
  - base PyTorch analysis outputs
- `Classical_Solver/`
  - classical solver visual outputs
- `Jax_Vs_Pytorch_Comparison/Multifamily/`
  - main strict multi-family PyTorch vs JAX comparison figures
- `Jax_Vs_Pytorch_Comparison/Monofamily/`
  - exploratory mono-family comparison figures

Color code:
- `PyTorch`: `deepskyblue`
- `JAX`: `deeppink`
- `CN`: `black`
- snapshot times: `RdPu`

Related asset hubs:
- `base/assets_pytorch/`
- `jax_comparison/multifamily/assets_multifamily/`
- `jax_comparison/monofamily/assets_monofamily/`

For the authoritative experiment definitions and launchers, see [experiments/](../experiments).
