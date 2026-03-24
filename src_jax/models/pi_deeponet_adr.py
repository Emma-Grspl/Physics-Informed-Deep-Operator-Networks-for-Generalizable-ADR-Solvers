from __future__ import annotations

from typing import Dict, List

import jax
import jax.numpy as jnp


def init_linear_params(key, in_dim: int, out_dim: int, scale: float = 1.0) -> Dict[str, jnp.ndarray]:
    w_key, b_key = jax.random.split(key)
    limit = jnp.sqrt(2.0 / (in_dim + out_dim))
    weight = jax.random.normal(w_key, (in_dim, out_dim)) * limit * scale
    bias = jnp.zeros((out_dim,), dtype=jnp.float32)
    return {"weight": weight, "bias": bias}


def linear(params: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    return x @ params["weight"] + params["bias"]


def init_mlp_params(key, dims: List[int]) -> List[Dict[str, jnp.ndarray]]:
    keys = jax.random.split(key, len(dims) - 1)
    return [init_linear_params(k, dims[i], dims[i + 1]) for i, k in enumerate(keys)]


def apply_mlp(params: List[Dict[str, jnp.ndarray]], x: jnp.ndarray, activation=jax.nn.silu) -> jnp.ndarray:
    h = x
    for layer in params[:-1]:
        h = activation(linear(layer, h))
    return linear(params[-1], h)


def init_model_params(key, cfg: Dict) -> Dict:
    m_cfg = cfg["model"]
    g_cfg = cfg["geometry"]
    p_cfg = cfg["physics_ranges"]
    latent_dim = m_cfg["latent_dim"]

    branch_dims = [8] + [m_cfg["branch_width"]] * m_cfg["branch_depth"] + [latent_dim]
    trunk_in_dim = 2 * m_cfg["nFourier"] if m_cfg["nFourier"] > 0 else 2

    keys = jax.random.split(key, 5 + m_cfg["trunk_depth"])
    branch_params = init_mlp_params(keys[0], branch_dims)
    trunk_input_map = init_linear_params(keys[1], trunk_in_dim, latent_dim)
    trunk_layers = [init_linear_params(keys[2 + i], latent_dim, latent_dim) for i in range(m_cfg["trunk_depth"])]
    branch_transforms = [
        init_linear_params(keys[2 + m_cfg["trunk_depth"] + i], latent_dim, 2 * latent_dim)
        for i in range(m_cfg["trunk_depth"])
    ]
    final_layer = init_linear_params(keys[-1], latent_dim, 1, scale=0.01)

    n_scales = len(m_cfg["sFourier"])
    freqs_per_scale = m_cfg["nFourier"] // n_scales
    fourier_keys = jax.random.split(jax.random.PRNGKey(1234), n_scales + 1)
    blocks = [
        jax.random.normal(fourier_keys[i], (2, freqs_per_scale)) * scale
        for i, scale in enumerate(m_cfg["sFourier"])
    ]
    remainder = m_cfg["nFourier"] - freqs_per_scale * n_scales
    if remainder > 0:
        blocks.append(
            jax.random.normal(fourier_keys[-1], (2, remainder)) * jnp.median(jnp.array(m_cfg["sFourier"]))
        )
    fourier_b = jnp.concatenate(blocks, axis=1)

    return {
        "branch_net": branch_params,
        "trunk_input_map": trunk_input_map,
        "trunk_layers": trunk_layers,
        "branch_transforms": branch_transforms,
        "final_layer": final_layer,
        "fourier_B": fourier_b,
        "norm": {
            "lb_geom": jnp.array([g_cfg["x_min"], 0.0], dtype=jnp.float32),
            "ub_geom": jnp.array([g_cfg["x_max"], g_cfg["T_max"]], dtype=jnp.float32),
            "v_min": jnp.array(p_cfg["v"][0], dtype=jnp.float32),
            "v_max": jnp.array(p_cfg["v"][1], dtype=jnp.float32),
            "D_min": jnp.array(p_cfg["D"][0], dtype=jnp.float32),
            "D_max": jnp.array(p_cfg["D"][1], dtype=jnp.float32),
            "mu_min": jnp.array(p_cfg["mu"][0], dtype=jnp.float32),
            "mu_max": jnp.array(p_cfg["mu"][1], dtype=jnp.float32),
            "A_scale": jnp.array(max(abs(p_cfg["A"][0]), abs(p_cfg["A"][1])), dtype=jnp.float32),
            "sigma_scale": jnp.array(max(abs(p_cfg["sigma"][0]), abs(p_cfg["sigma"][1])), dtype=jnp.float32),
            "k_scale": jnp.array(max(abs(p_cfg["k"][0]), abs(p_cfg["k"][1])), dtype=jnp.float32),
        },
    }


def normalize_tensor(x: jnp.ndarray, min_val: jnp.ndarray, max_val: jnp.ndarray) -> jnp.ndarray:
    return 2.0 * (x - min_val) / (max_val - min_val + 1e-8) - 1.0


def fourier_encode(x: jnp.ndarray, b_matrix: jnp.ndarray) -> jnp.ndarray:
    x_proj = 2.0 * jnp.pi * (x @ b_matrix)
    return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


def apply_model(params: dict, inputs_params: jnp.ndarray, xt: jnp.ndarray) -> jnp.ndarray:
    norm = params["norm"]
    p_v = normalize_tensor(inputs_params[:, 0:1], norm["v_min"], norm["v_max"])
    p_d = normalize_tensor(inputs_params[:, 1:2], norm["D_min"], norm["D_max"])
    p_mu = normalize_tensor(inputs_params[:, 2:3], norm["mu_min"], norm["mu_max"])
    p_type = inputs_params[:, 3:4]
    p_a = inputs_params[:, 4:5] / norm["A_scale"]
    p_x0 = jnp.zeros_like(inputs_params[:, 5:6])
    p_sigma = inputs_params[:, 6:7] / norm["sigma_scale"]
    p_k = inputs_params[:, 7:8] / norm["k_scale"]

    params_norm = jnp.concatenate([p_v, p_d, p_mu, p_type, p_a, p_x0, p_sigma, p_k], axis=1)
    xt_norm = normalize_tensor(xt, norm["lb_geom"], norm["ub_geom"])

    xt_embed = fourier_encode(xt_norm, params["fourier_B"])
    context_b = apply_mlp(params["branch_net"], params_norm)
    z_val = jax.nn.silu(linear(params["trunk_input_map"], xt_embed))

    for trunk_layer, branch_transform in zip(params["trunk_layers"], params["branch_transforms"]):
        z_trunk = linear(trunk_layer, z_val)
        uv = linear(branch_transform, context_b)
        u_val, v_val = jnp.split(uv, 2, axis=1)
        z_val = jax.nn.silu((1.0 - z_trunk) * u_val + z_trunk * v_val)

    return linear(params["final_layer"], z_val)
