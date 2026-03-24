from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp


def get_ic_value(x: jnp.ndarray, ic_params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    types = ic_params["type"]
    amplitude = ic_params["A"]
    x0 = ic_params["x0"]
    sigma = ic_params["sigma"]
    k_val = ic_params["k"]

    tanh_part = jnp.tanh((x - x0) / (sigma + 1e-8))
    gauss_env = jnp.exp(-((x - x0) ** 2) / (2.0 * sigma**2 + 1e-8))
    sin_gauss = amplitude * gauss_env * jnp.sin(k_val * x)
    gaussian = amplitude * gauss_env

    is_tanh = (types == 0).astype(jnp.float32)
    is_sin = jnp.isin(types, jnp.array([1, 2], dtype=types.dtype)).astype(jnp.float32)
    is_gauss = jnp.isin(types, jnp.array([3, 4], dtype=types.dtype)).astype(jnp.float32)

    return is_tanh * tanh_part + is_sin * sin_gauss + is_gauss * gaussian


def _sample_uniform(key, bounds: Tuple[float, float], shape: Tuple[int, ...]) -> jnp.ndarray:
    return jax.random.uniform(key, shape=shape, minval=bounds[0], maxval=bounds[1])


def generate_mixed_batch(
    key,
    n_samples: int,
    bounds_phy: Dict,
    x_min: float,
    x_max: float,
    t_max: float,
    allowed_types: Optional[List[int]] = None,
) -> Tuple[jnp.ndarray, ...]:
    keys = jax.random.split(key, 12)

    v = _sample_uniform(keys[0], tuple(bounds_phy["v"]), (n_samples, 1))
    d_val = _sample_uniform(keys[1], tuple(bounds_phy["D"]), (n_samples, 1))
    mu = _sample_uniform(keys[2], tuple(bounds_phy["mu"]), (n_samples, 1))
    amplitude = _sample_uniform(keys[3], tuple(bounds_phy["A"]), (n_samples, 1))
    x0 = _sample_uniform(keys[4], tuple(bounds_phy["x0"]), (n_samples, 1))
    sigma = _sample_uniform(keys[5], tuple(bounds_phy["sigma"]), (n_samples, 1))
    k_val = _sample_uniform(keys[6], tuple(bounds_phy["k"]), (n_samples, 1))

    if allowed_types:
        type_choices = jnp.array(allowed_types, dtype=jnp.int32)
    else:
        type_choices = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
    types = jax.random.choice(keys[7], type_choices, shape=(n_samples, 1)).astype(jnp.float32)

    n_action = int(n_samples * 0.8) if x_min < 0.0 < x_max else 0
    if n_action > 0:
        n_rest = n_samples - n_action
        x_rest = jax.random.uniform(keys[8], shape=(n_rest, 1), minval=x_min, maxval=0.0)
        x_action = jax.random.uniform(keys[9], shape=(n_action, 1), minval=0.0, maxval=x_max)
        x = jnp.concatenate([x_rest, x_action], axis=0)
        x = jax.random.permutation(keys[10], x, axis=0)
    else:
        x = jax.random.uniform(keys[8], shape=(n_samples, 1), minval=x_min, maxval=x_max)

    t = jax.random.uniform(keys[11], shape=(n_samples, 1), minval=0.0, maxval=t_max)
    xt = jnp.concatenate([x, t], axis=1)

    x_ic = jax.random.uniform(keys[0], shape=(n_samples, 1), minval=x_min, maxval=x_max)
    xt_ic = jnp.concatenate([x_ic, jnp.zeros_like(x_ic)], axis=1)

    ic_params = {
        "type": types,
        "A": amplitude,
        "x0": x0,
        "sigma": sigma,
        "k": k_val,
    }
    u_true_ic = get_ic_value(x_ic, ic_params)

    t_bc = jax.random.uniform(keys[1], shape=(n_samples, 1), minval=0.0, maxval=t_max)
    xt_bc_left = jnp.concatenate([jnp.full((n_samples, 1), x_min), t_bc], axis=1)
    xt_bc_right = jnp.concatenate([jnp.full((n_samples, 1), x_max), t_bc], axis=1)

    is_tanh = (types == 0).astype(jnp.float32)
    u_true_bc_l = -1.0 * is_tanh
    u_true_bc_r = 1.0 * is_tanh

    params = jnp.concatenate([v, d_val, mu, types, amplitude, x0, sigma, k_val], axis=1)
    return params, xt, xt_ic, u_true_ic, xt_bc_left, xt_bc_right, u_true_bc_l, u_true_bc_r
