"""Physics module `jax_comparison.multifamily.src.jax.physics.residual_adr`.

This file implements ADR residual terms or other PDE-side computations used by the physics-informed losses.
"""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp

from src_jax.models.pi_deeponet_adr import apply_model


def _scalar_prediction(params: Dict, p_row: jnp.ndarray, xt_row: jnp.ndarray) -> jnp.ndarray:
    return apply_model(params, p_row[None, :], xt_row[None, :])[0, 0]


def pde_residual_adr(params: Dict, batch_params: jnp.ndarray, xt: jnp.ndarray) -> jnp.ndarray:
    grad_fn = jax.grad(_scalar_prediction, argnums=2)
    hess_fn = jax.hessian(_scalar_prediction, argnums=2)

    grads = jax.vmap(lambda p_row, xt_row: grad_fn(params, p_row, xt_row))(batch_params, xt)
    hess = jax.vmap(lambda p_row, xt_row: hess_fn(params, p_row, xt_row))(batch_params, xt)

    u = apply_model(params, batch_params, xt)
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]
    u_xx = hess[:, 0:1, 0]

    v_in = batch_params[:, 0:1]
    d_in = batch_params[:, 1:2]
    mu_in = batch_params[:, 2:3]

    u_safe = jnp.clip(u, -1.2, 1.2)
    reaction = mu_in * (u_safe - u_safe**3)
    return u_t + v_in * u_x - d_in * u_xx - reaction
