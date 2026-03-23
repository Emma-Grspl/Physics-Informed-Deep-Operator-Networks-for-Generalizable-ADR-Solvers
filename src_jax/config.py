from pathlib import Path

import yaml


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "configs_jax" / "config_ADR_jax.yaml"
    else:
        path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
