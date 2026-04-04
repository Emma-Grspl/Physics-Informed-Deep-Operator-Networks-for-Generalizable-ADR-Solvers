# Experiments

This directory is the public registry for reproducible experiment definitions.

Each experiment package contains the configuration files, launchers, and short protocol notes required to reproduce a bounded study from the repository root. Source code remains in the main implementation trees, while `experiments/` defines how that code is exercised for a named protocol.

Subdirectories:

- `base/`: canonical PyTorch baseline protocol
- `multifamily/`: strict PyTorch vs JAX comparison protocols
- `monofamily/`: mono-family comparison protocols
- `ablations/`: bounded ablation studies such as ansatz/LBFGS sweeps

Conventions:

- configs live inside the corresponding experiment package
- benchmark scripts are invoked from the repository root
- generated outputs should be written to `results/` or the configured benchmark output directory
- `.slurm` launchers in this tree target Jean Zay specifically; they are not intended to be cluster-agnostic job scripts
