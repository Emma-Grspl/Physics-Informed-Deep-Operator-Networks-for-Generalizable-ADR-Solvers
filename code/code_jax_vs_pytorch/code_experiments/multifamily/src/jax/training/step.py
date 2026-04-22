from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax

from src_jax.physics.residual_adr import pde_residual_adr


def get_ic_loss(params: Dict, batch: Tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    batch_params, _, xt_ic, u_true_ic, _, _, _, _ = batch

    from src_jax.models.pi_deeponet_adr import apply_model

    return jnp.mean((apply_model(params, batch_params, xt_ic) - u_true_ic) ** 2)


def get_loss(params: Dict, batch: Tuple[jnp.ndarray, ...], wr: float, wi: float, wb: float) -> jnp.ndarray:
    batch_params, xt, xt_ic, u_true_ic, xt_bc_l, xt_bc_r, u_true_bc_l, u_true_bc_r = batch

    from src_jax.models.pi_deeponet_adr import apply_model

    l_pde = jnp.mean(pde_residual_adr(params, batch_params, xt) ** 2)
    l_ic = jnp.mean((apply_model(params, batch_params, xt_ic) - u_true_ic) ** 2)
    l_bc = jnp.mean((apply_model(params, batch_params, xt_bc_l) - u_true_bc_l) ** 2)
    l_bc += jnp.mean((apply_model(params, batch_params, xt_bc_r) - u_true_bc_r) ** 2)
    return wr * l_pde + wi * l_ic + wb * l_bc


def make_train_step(optimizer: optax.GradientTransformation):
    @jax.jit
    def train_step(
        params: Dict,
        opt_state: optax.OptState,
        batch: Tuple[jnp.ndarray, ...],
        wr: float,
        wi: float,
        wb: float,
    ) -> Tuple[Dict, optax.OptState, jnp.ndarray]:
        loss_fn = lambda current_params: get_loss(current_params, batch, wr, wi, wb)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, next_opt_state = optimizer.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        return next_params, next_opt_state, loss

    return train_step


def make_ic_train_step(optimizer: optax.GradientTransformation):
    @jax.jit
    def train_step(
        params: Dict,
        opt_state: optax.OptState,
        batch: Tuple[jnp.ndarray, ...],
    ) -> Tuple[Dict, optax.OptState, jnp.ndarray]:
        loss, grads = jax.value_and_grad(get_ic_loss)(params, batch)
        updates, next_opt_state = optimizer.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        return next_params, next_opt_state, loss

    return train_step
