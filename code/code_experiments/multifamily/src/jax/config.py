"""Comparison-layer module `jax_comparison.multifamily.src.jax.config`.

This file supports the PyTorch versus JAX comparison workflows, including replicated implementations, scripts, or validation helpers.
"""

from pathlib import Path
from typing import Dict, Optional

import yaml


def load_config(path: Optional[str] = None) -> Dict:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "configs_jax" / "config_ADR_jax.yaml"
    else:
        path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
