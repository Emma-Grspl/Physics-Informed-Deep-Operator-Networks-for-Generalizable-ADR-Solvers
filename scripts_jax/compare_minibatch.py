from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generators import get_ic_value as torch_get_ic_value
from src.physics.residual_ADR import pde_residual_adr as torch_pde_residual
from src_jax.config import load_config
from src_jax.data.generators import generate_mixed_batch, get_ic_value as jax_get_ic_value


def compare_ic_values() -> None:
    x = np.linspace(-2.0, 2.0, 16, dtype=np.float32).reshape(-1, 1)
    test_cases = [
        {"type": 0, "A": 1.0, "x0": 0.0, "sigma": 0.5, "k": 2.0},
        {"type": 1, "A": 0.9, "x0": 0.0, "sigma": 0.6, "k": 1.5},
        {"type": 3, "A": 0.8, "x0": 0.0, "sigma": 0.7, "k": 2.5},
    ]

    for case in test_cases:
        torch_val = torch_get_ic_value(x, "mixed", case)
        jax_case = {
            "type": jnp.full_like(jnp.asarray(x), float(case["type"])),
            "A": jnp.full_like(jnp.asarray(x), case["A"]),
            "x0": jnp.full_like(jnp.asarray(x), case["x0"]),
            "sigma": jnp.full_like(jnp.asarray(x), case["sigma"]),
            "k": jnp.full_like(jnp.asarray(x), case["k"]),
        }
        jax_val = np.asarray(jax_get_ic_value(jnp.asarray(x), jax_case))
        max_diff = float(np.max(np.abs(torch_val - jax_val)))
        print(f"IC type={case['type']} | max abs diff = {max_diff:.3e}")
        assert max_diff < 1e-6


def compare_analytic_residual() -> None:
    xt = np.array([[1.0, 0.5], [0.0, 2.0], [-1.0, 1.0], [0.4, 0.1]], dtype=np.float32)
    params = np.array(
        [
            [2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            [2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            [2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            [2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    class TorchMockModel(torch.nn.Module):
        def forward(self, params_tensor, xt_tensor):
            v_val = params_tensor[:, 0:1]
            x_val = xt_tensor[:, 0:1]
            t_val = xt_tensor[:, 1:2]
            return torch.sin(x_val - v_val * t_val)

    torch_model = TorchMockModel()
    torch_res = torch_pde_residual(
        torch_model,
        torch.tensor(params),
        torch.tensor(xt, requires_grad=True),
    )
    torch_mean = float(torch.mean(torch.abs(torch_res)).item())

    def scalar_solution(p_row: jnp.ndarray, xt_row: jnp.ndarray) -> jnp.ndarray:
        return jnp.sin(xt_row[0] - p_row[0] * xt_row[1])

    grad_fn = jax.grad(scalar_solution, argnums=1)
    hess_fn = jax.hessian(scalar_solution, argnums=1)

    def one_residual(p_row: jnp.ndarray, xt_row: jnp.ndarray) -> jnp.ndarray:
        grads = grad_fn(p_row, xt_row)
        hess = hess_fn(p_row, xt_row)
        u_val = scalar_solution(p_row, xt_row)
        u_safe = jnp.clip(u_val, -1.2, 1.2)
        reaction = p_row[2] * (u_safe - u_safe**3)
        return grads[1] + p_row[0] * grads[0] - p_row[1] * hess[0, 0] - reaction

    jax_res = jax.vmap(one_residual)(jnp.asarray(params), jnp.asarray(xt))
    jax_mean = float(jnp.mean(jnp.abs(jax_res)))

    print(f"Analytic residual | torch mean = {torch_mean:.3e} | jax mean = {jax_mean:.3e}")
    assert torch_mean < 1e-4
    assert jax_mean < 1e-4


def smoke_test_jax_batch() -> None:
    cfg = load_config()
    batch = generate_mixed_batch(
        key=jax.random.PRNGKey(0),
        n_samples=32,
        bounds_phy=cfg["physics_ranges"],
        x_min=cfg["geometry"]["x_min"],
        x_max=cfg["geometry"]["x_max"],
        t_max=1.0,
    )
    shapes = [tuple(arr.shape) for arr in batch]
    print(f"JAX batch shapes = {shapes}")
    assert shapes == [
        (32, 8),
        (32, 2),
        (32, 2),
        (32, 1),
        (32, 2),
        (32, 2),
        (32, 1),
        (32, 1),
    ]


def main() -> None:
    compare_ic_values()
    compare_analytic_residual()
    smoke_test_jax_batch()
    print("Minibatch comparison OK")


if __name__ == "__main__":
    main()
