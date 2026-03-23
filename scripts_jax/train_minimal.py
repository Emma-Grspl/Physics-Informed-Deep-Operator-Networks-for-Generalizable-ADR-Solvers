from __future__ import annotations

import sys
from pathlib import Path

import jax
import optax

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_jax.config import load_config
from src_jax.data.generators import generate_mixed_batch
from src_jax.models.pi_deeponet_adr import init_model_params
from src_jax.training.step import make_train_step


def main() -> None:
    cfg = load_config()
    key = jax.random.PRNGKey(0)
    model_key, batch_key = jax.random.split(key)

    params = init_model_params(model_key, cfg)
    optimizer = optax.adam(cfg["training"]["learning_rate"])
    opt_state = optimizer.init(params)
    train_step = make_train_step(optimizer)

    batch = generate_mixed_batch(
        batch_key,
        n_samples=256,
        bounds_phy=cfg["physics_ranges"],
        x_min=cfg["geometry"]["x_min"],
        x_max=cfg["geometry"]["x_max"],
        t_max=0.5,
    )

    params, opt_state, loss = train_step(
        params,
        opt_state,
        batch,
        wr=cfg["loss_weights"]["first_w_res"],
        wi=cfg["loss_weights"]["weight_ic_final"],
        wb=cfg["loss_weights"]["weight_bc"],
    )

    print(f"Minimal JAX training step OK | loss={float(loss):.6e}")


if __name__ == "__main__":
    main()
