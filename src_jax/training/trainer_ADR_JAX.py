from __future__ import annotations

import copy
import os
import pickle
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from benchmarks.common.eval import compute_cn_solution
from src_jax.data.generators import generate_mixed_batch, get_ic_value
from src_jax.models.pi_deeponet_adr import apply_model
from src_jax.training.step import get_ic_loss, get_loss, make_ic_train_step, make_train_step


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def generate_time_steps(cfg: Dict) -> List[float]:
    steps = []
    current_t = 0.0
    t_limit = cfg["geometry"]["T_max"]
    for zone in cfg["time_stepping"]["zones"]:
        t_end = t_limit if zone["t_end"] == -1 else zone["t_end"]
        dt = zone["dt"]
        while current_t < t_end - 1e-5:
            current_t = round(current_t + dt, 3)
            if current_t > t_limit + 1e-5:
                break
            steps.append(current_t)
    return steps


def save_pickle(path: str, payload) -> None:
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path: str):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def clone_params(params):
    return jax.tree_util.tree_map(lambda x: x.copy(), params)


def tree_l2_norm(tree) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves if leaf is not None) + 1e-12)


def compute_ntk_weights(params, batch, w_ic_ref: float) -> float:
    def loss_pde(current_params):
        batch_params, xt, _, _, _, _, _, _ = batch
        from src_jax.physics.residual_adr import pde_residual_adr

        return jnp.mean(pde_residual_adr(current_params, batch_params, xt) ** 2)

    def loss_ic(current_params):
        return get_ic_loss(current_params, batch)

    grad_pde = jax.grad(loss_pde)(params)
    grad_ic = jax.grad(loss_ic)(params)
    norm_pde = tree_l2_norm(grad_pde)
    norm_ic = tree_l2_norm(grad_ic)
    new_w_pde = float(norm_ic / (norm_pde + 1e-8) * w_ic_ref)
    return min(max(new_w_pde, 10.0), 500.0)


def monitor_gradients(params, batch) -> Tuple[float, float]:
    def loss_pde(current_params):
        batch_params, xt, _, _, _, _, _, _ = batch
        from src_jax.physics.residual_adr import pde_residual_adr

        return jnp.mean(pde_residual_adr(current_params, batch_params, xt) ** 2)

    def loss_ic(current_params):
        return get_ic_loss(current_params, batch)

    grad_pde = jax.grad(loss_pde)(params)
    grad_ic = jax.grad(loss_ic)(params)
    flat_pde, _ = jax.flatten_util.ravel_pytree(grad_pde)
    flat_ic, _ = jax.flatten_util.ravel_pytree(grad_ic)
    ratio = float(jnp.linalg.norm(flat_ic) / (jnp.linalg.norm(flat_pde) + 1e-8))
    cos_sim = float(jnp.dot(flat_pde, flat_ic) / (jnp.linalg.norm(flat_pde) * jnp.linalg.norm(flat_ic) + 1e-8))
    return ratio, cos_sim


class KingOfTheHill:
    def __init__(self, params):
        self.best_state = clone_params(params)
        self.best_loss = float("inf")

    def update(self, params, current_loss: float) -> bool:
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_state = clone_params(params)
            return True
        return False


def _predict_grid(params, params_dict: Dict, t_max: float, nx: int, nt: int, cfg: Dict) -> np.ndarray:
    x_min = cfg["geometry"]["x_min"]
    x_max = cfg["geometry"]["x_max"]
    x = np.linspace(x_min, x_max, nx)
    t = np.linspace(0.0, t_max, nt)
    x_grid, t_grid = np.meshgrid(x, t, indexing="ij")
    xt = np.stack([x_grid.flatten(), t_grid.flatten()], axis=1).astype(np.float32)
    p_vec = np.array(
        [
            params_dict["v"],
            params_dict["D"],
            params_dict["mu"],
            params_dict["type"],
            params_dict["A"],
            0.0,
            params_dict["sigma"],
            params_dict["k"],
        ],
        dtype=np.float32,
    )
    p_batch = np.repeat(p_vec[None, :], xt.shape[0], axis=0)
    pred = apply_model(params, jnp.asarray(p_batch), jnp.asarray(xt))
    return np.asarray(jax.device_get(pred)).reshape(nx, nt)


def _predict_ic(params, params_dict: Dict, x: np.ndarray) -> np.ndarray:
    xt = np.stack([x, np.zeros_like(x)], axis=1).astype(np.float32)
    p_vec = np.array(
        [
            params_dict["v"],
            params_dict["D"],
            params_dict["mu"],
            params_dict["type"],
            params_dict["A"],
            0.0,
            params_dict["sigma"],
            params_dict["k"],
        ],
        dtype=np.float32,
    )
    p_batch = np.repeat(p_vec[None, :], xt.shape[0], axis=0)
    pred = apply_model(params, jnp.asarray(p_batch), jnp.asarray(xt))
    return np.asarray(jax.device_get(pred)).reshape(-1)


def _true_ic(params_dict: Dict, x: np.ndarray) -> np.ndarray:
    ic_params = {
        "type": jnp.asarray(params_dict["type"], dtype=jnp.float32),
        "A": jnp.asarray(params_dict["A"], dtype=jnp.float32),
        "x0": jnp.asarray(params_dict.get("x0", 0.0), dtype=jnp.float32),
        "sigma": jnp.asarray(params_dict["sigma"], dtype=jnp.float32),
        "k": jnp.asarray(params_dict["k"], dtype=jnp.float32),
    }
    return np.asarray(get_ic_value(jnp.asarray(x, dtype=jnp.float32), ic_params)).reshape(-1)


def _true_ic_np(x: np.ndarray, params_dict: Dict) -> np.ndarray:
    type_id = int(params_dict["type"])
    amplitude = float(params_dict["A"])
    x0 = float(params_dict.get("x0", 0.0))
    sigma = float(params_dict["sigma"])
    k_val = float(params_dict["k"])

    tanh_part = np.tanh((x - x0) / (sigma + 1e-8))
    gauss_env = np.exp(-((x - x0) ** 2) / (2.0 * sigma**2 + 1e-8))
    if type_id == 0:
        return tanh_part
    if type_id in (1, 2):
        return amplitude * gauss_env * np.sin(k_val * x)
    return amplitude * gauss_env


def _ic_holdout_metrics(params, cfg: Dict, bounds: Dict, batch_size: int, key) -> Tuple[float, float]:
    batch = generate_mixed_batch(
        key,
        batch_size,
        bounds,
        cfg["geometry"]["x_min"],
        cfg["geometry"]["x_max"],
        0.0,
    )
    held_mse = float(jax.device_get(get_ic_loss(params, batch)))
    params_batch, _, xt_ic, u_true_ic, _, _, _, _ = batch
    pred = apply_model(params, params_batch, xt_ic)
    rel_l2 = float(
        np.linalg.norm(np.asarray(jax.device_get(pred - u_true_ic)))
        / (np.linalg.norm(np.asarray(jax.device_get(u_true_ic))) + 1e-8)
    )
    return held_mse, rel_l2


def _audit_ic_case(params, params_dict: Dict, x: np.ndarray) -> float:
    u_true = _true_ic_np(x, params_dict)
    u_pred = _predict_ic(params, params_dict, x)
    return float(np.linalg.norm(u_true - u_pred) / (np.linalg.norm(u_true) + 1e-8))


def _balanced_warmup_types() -> List[int]:
    # 1/3 Tanh, 1/3 Sin-Gauss, 1/3 Gaussian at the family level.
    return [0, 0, 1, 2, 3, 4]


def audit_global_fast(params, cfg: Dict, t_max: float) -> Tuple[bool, float]:
    rng = np.random.default_rng(42)
    errors = []
    if t_max == 0.0:
        target_threshold = cfg["training"].get("threshold_ic", cfg["training"].get("threshold", 0.01))
        mode_str = "IC (Strict)"
    else:
        target_threshold = cfg["training"].get("threshold_step", cfg["training"].get("threshold", 0.05))
        mode_str = "Step (Relaxed)"

    nx = cfg["audit"]["Nx_audit"]
    nt = cfg["audit"]["Nt_audit"]
    for _ in range(cfg["audit"].get("n_global_cases", 40)):
        p_dict = {k: rng.uniform(v[0], v[1]) for k, v in cfg["physics_ranges"].items()}
        p_dict["type"] = int(rng.integers(0, 5))
        try:
            if t_max == 0.0:
                x = np.linspace(cfg["geometry"]["x_min"], cfg["geometry"]["x_max"], nx)
                err = _audit_ic_case(params, p_dict, x)
            else:
                u_true = compute_cn_solution(cfg, p_dict, t_max, nx, nt)
                u_pred = _predict_grid(params, p_dict, t_max, nx, nt, cfg)
                err = np.linalg.norm(u_true - u_pred) / (np.linalg.norm(u_true) + 1e-8)
            errors.append(float(err))
        except Exception:
            continue

    if not errors:
        return False, 1.0
    avg_err = float(np.mean(errors))
    print("Audit {0} | Avg Rel L2: {1:.2%} (Target: < {2:.2%})".format(mode_str, avg_err, target_threshold))
    return avg_err < target_threshold, avg_err


def diagnose_model(params, cfg: Dict, threshold: float | None = None, t_max: float | None = None) -> List[int]:
    if t_max is None:
        t_max = cfg["geometry"]["T_max"]
    if threshold is None:
        if t_max == 0.0:
            threshold = cfg["training"].get("threshold_ic", cfg["training"].get("threshold", 0.01))
        else:
            threshold = cfg["training"].get("threshold_step", cfg["training"].get("threshold", 0.05))

    rng = np.random.default_rng(123)
    families_map = {"Gaussian": [3, 4], "Sin-Gauss": [1, 2], "Tanh": [0]}
    failed_ids = []
    nx = cfg["audit"]["Nx_audit"]
    nt = cfg["audit"]["Nt_audit"]
    print("Diag (t_max={0}, Threshold: {1:.1%})...".format(t_max, threshold))

    for fam_name, type_ids in families_map.items():
        errors = []
        for _ in range(cfg["audit"].get("n_family_cases", 12)):
            p_dict = {k: rng.uniform(v[0], v[1]) for k, v in cfg["physics_ranges"].items()}
            p_dict["type"] = int(rng.choice(type_ids))
            try:
                if t_max == 0.0:
                    x = np.linspace(cfg["geometry"]["x_min"], cfg["geometry"]["x_max"], nx)
                    err = _audit_ic_case(params, p_dict, x)
                else:
                    u_true = compute_cn_solution(cfg, p_dict, t_max, nx, nt)
                    u_pred = _predict_grid(params, p_dict, t_max, nx, nt, cfg)
                    err = np.linalg.norm(u_true - u_pred) / (np.linalg.norm(u_true) + 1e-8)
                errors.append(float(err))
            except Exception:
                continue
        if not errors:
            print("  - {0:<12} : err solver".format(fam_name))
            continue
        mean_err = float(np.mean(errors))
        status = "OK" if mean_err < threshold else "FAIL"
        print("  - {0:<12} : {1:.2%} {2}".format(fam_name, mean_err, status))
        if mean_err > threshold:
            failed_ids.extend(type_ids)
    return failed_ids


def get_t_failed(params, cfg: Dict, threshold: float = 0.04) -> float:
    print("Recherche du point de bascule temporel (t_failed)...")
    t_evals = [0.0, 0.5, 1.0]
    rng = np.random.default_rng(7)
    nx = cfg["audit"]["Nx_audit"]
    nt = cfg["audit"]["Nt_audit"]
    for t_val in t_evals:
        errors = []
        for _ in range(10):
            p_dict = {k: rng.uniform(v[0], v[1]) for k, v in cfg["physics_ranges"].items()}
            p_dict["type"] = int(rng.choice([0, 1, 3]))
            try:
                u_true = compute_cn_solution(cfg, p_dict, t_val, nx, nt)
                u_pred = _predict_grid(params, p_dict, t_val, nx, nt, cfg)
                err = np.linalg.norm(u_true - u_pred) / (np.linalg.norm(u_true) + 1e-8)
                errors.append(float(err))
            except Exception:
                continue
        if errors and np.mean(errors) > threshold:
            print("Decrochage detecte a t={0} (Erreur: {1:.2%})".format(t_val, np.mean(errors)))
            return max(0.1, t_val)
    print("Modele robuste partout, fallback t_failed=0.7")
    return 0.7


def targeted_correction(params, cfg: Dict, bounds: Dict, t_max: float, initial_failed_ids: List[int], n_iters_base: int, start_lr: float, target_threshold: float | None = None, apply_80_20: bool = False):
    print("Demarrage du Dynamic Soft Polishing...")
    key = jax.random.PRNGKey(2026)
    king_corr = KingOfTheHill(params)

    w_bc_loc = cfg["loss_weights"]["weight_bc"]
    if t_max == 0.0:
        w_res_loc, w_ic_loc = 0.0, 100.0
    else:
        w_res_loc = cfg["loss_weights"]["first_w_res"]
        w_ic_loc = cfg["loss_weights"]["weight_ic_final"]

    end_lr = 1e-6
    gamma = (end_lr / start_lr) ** (1 / max(n_iters_base / 1000, 1)) if start_lr > end_lr else 1.0
    current_lr = start_lr
    current_failed_ids = list(initial_failed_ids)
    t_failed = None
    if apply_80_20 and target_threshold is not None:
        t_failed = get_t_failed(params, cfg, threshold=target_threshold)

    optimizer = optax.adam(current_lr)
    opt_state = optimizer.init(params)
    train_step = make_train_step(optimizer)

    for i in range(n_iters_base):
        if i % 1000 == 0:
            if i > 0:
                current_failed_ids = diagnose_model(params, cfg, threshold=target_threshold, t_max=t_max)
                current_lr = max(end_lr, current_lr * gamma)
                optimizer = optax.adam(current_lr)
                opt_state = optimizer.init(params)
                train_step = make_train_step(optimizer)
                if len(current_failed_ids) == 0:
                    print("Objectif atteint a l'iteration {0}".format(i))
                    return params, True

            all_types = [0, 1, 2, 3, 4]
            weighted_types = []
            for tid in all_types:
                weighted_types.extend([tid] * (2 if tid in current_failed_ids else 1))
            print("Polishing | LR: {0:.1e} | Fails: {1}".format(current_lr, current_failed_ids))

        key, batch_key = jax.random.split(key)
        batch = generate_mixed_batch(
            batch_key,
            cfg["training"]["n_sample"],
            bounds,
            cfg["geometry"]["x_min"],
            cfg["geometry"]["x_max"],
            t_max,
            allowed_types=weighted_types,
        )
        params_batch, xt, xt_ic, u_true_ic, xt_bc_l, xt_bc_r, u_true_bc_l, u_true_bc_r = batch

        if apply_80_20 and t_failed is not None and t_max > t_failed:
            n_samples = xt.shape[0]
            n_fail = int(0.65 * n_samples)
            n_pass = n_samples - n_fail
            key, k1, k2, k3 = jax.random.split(key, 4)
            t_pass = jax.random.uniform(k1, (n_pass, 1), minval=0.0, maxval=t_failed)
            t_fail = jax.random.uniform(k2, (n_fail, 1), minval=t_failed, maxval=t_max)
            new_t = jnp.concatenate([t_pass, t_fail], axis=0)
            new_t = jax.random.permutation(k3, new_t, axis=0)
            xt = xt.at[:, 1:2].set(new_t)
            batch = (params_batch, xt, xt_ic, u_true_ic, xt_bc_l, xt_bc_r, u_true_bc_l, u_true_bc_r)

        if t_max > 0.0 and i % 100 == 0:
            w_res_loc = compute_ntk_weights(params, batch, w_ic_loc)

        params, opt_state, loss = train_step(params, opt_state, batch, w_res_loc, w_ic_loc, w_bc_loc)
        loss_value = float(jax.device_get(loss))
        king_corr.update(params, loss_value)
        if i % 500 == 0:
            print("Polishing iter {0} | loss={1:.2e}".format(i, loss_value))

    params = clone_params(king_corr.best_state)
    final_fails = diagnose_model(params, cfg, threshold=target_threshold, t_max=t_max)
    return params, len(final_fails) == 0


def train_step_time_window(params, cfg: Dict, bounds: Dict, t_max: float, n_iters_main: int):
    king = KingOfTheHill(params)
    w_bc = cfg["loss_weights"]["weight_bc"]
    t_ramp_end = 0.3
    if t_max <= t_ramp_end:
        start_w_res = 0.1
        target_w_res = cfg["loss_weights"]["first_w_res"]
        ratio = t_max / t_ramp_end
        w_res = start_w_res + (target_w_res - start_w_res) * (ratio * ratio)
        w_ic = cfg["loss_weights"]["weight_ic_init"]
        mode = "RAMPE DOUCE"
    else:
        w_res = cfg["loss_weights"]["first_w_res"]
        w_ic = cfg["loss_weights"]["weight_ic_final"]
        mode = "NTK"

    print("Level t={0} | Mode: {1}".format(t_max, mode))
    current_lr = cfg["training"]["learning_rate"]
    global_success = False
    key = jax.random.PRNGKey(int(t_max * 1000) + 17)

    for macro in range(cfg["training"]["nb_loop"]):
        print("Macro {0}/{1}".format(macro + 1, cfg["training"]["nb_loop"]))

        for retry in range(cfg["training"]["max_retry"]):
            params = clone_params(king.best_state)
            optimizer = optax.adam(current_lr)
            opt_state = optimizer.init(params)
            train_step = make_train_step(optimizer)

            for i in range(n_iters_main):
                key, batch_key = jax.random.split(key)
                batch = generate_mixed_batch(
                    batch_key,
                    cfg["training"]["n_sample"],
                    bounds,
                    cfg["geometry"]["x_min"],
                    cfg["geometry"]["x_max"],
                    t_max,
                )
                if mode == "NTK" and i % 100 == 0:
                    w_res = compute_ntk_weights(params, batch, w_ic)
                params, opt_state, loss = train_step(params, opt_state, batch, w_res, w_ic, w_bc)
                loss_value = float(jax.device_get(loss))
                king.update(params, loss_value)
                if i % 1000 == 0:
                    r, c = monitor_gradients(params, batch)
                    print("It {0} | Loss: {1:.2e} | ForceRatio: {2:.2f} | CosSim: {3:.2f}".format(i, loss_value, r, c))

            success_adam, err = audit_global_fast(params, cfg, t_max)
            if success_adam:
                print("Global success (Adam).")
                global_success = True
                break
            current_lr *= 0.5

        if global_success:
            break

    if not global_success:
        print("Global failure")
        return params, False, 1.0

    params = clone_params(king.best_state)
    print("Final audit")
    failed_ids = diagnose_model(params, cfg, t_max=t_max)
    if len(failed_ids) == 0:
        _, final_err = audit_global_fast(params, cfg, t_max)
        return params, True, final_err

    print("Specific failure. Correction.")
    params, success = targeted_correction(
        params,
        cfg,
        bounds,
        t_max,
        failed_ids,
        cfg["training"]["n_iters_correction"],
        start_lr=current_lr,
    )
    if success:
        _, final_err = audit_global_fast(params, cfg, t_max)
        return params, True, final_err
    return params, False, 1.0


def train_smart_time_marching(params, cfg: Dict, bounds: Dict):
    save_dir = cfg["audit"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    metrics = {"time_steps": [], "final_polish": None}

    n_warmup = cfg["training"].get("n_warmup", 0)
    if n_warmup > 0:
        print("Warmup ({0} iters)".format(n_warmup))
        warmup_lr = cfg["training"].get("warmup_learning_rate", cfg["training"]["learning_rate"])
        warmup_batch_size = cfg["training"].get("warmup_batch_size", cfg["training"]["batch_size"])
        holdout_every = cfg["training"].get("warmup_holdout_every", 0)
        holdout_batch_size = cfg["training"].get("warmup_holdout_n", 2048)
        warmup_audit_every = cfg["training"].get("warmup_audit_every", 0)
        warmup_allowed_types = cfg["training"].get("warmup_allowed_types")

        optimizer = optax.adam(warmup_lr)
        opt_state = optimizer.init(params)
        train_step = make_ic_train_step(optimizer)
        king = KingOfTheHill(params)
        best_warmup_params = clone_params(params)
        best_warmup_err = float("inf")
        key = jax.random.PRNGKey(11)
        holdout_key = jax.random.PRNGKey(111)
        for i in range(n_warmup):
            key, batch_key = jax.random.split(key)
            batch = generate_mixed_batch(
                batch_key,
                warmup_batch_size,
                bounds,
                cfg["geometry"]["x_min"],
                cfg["geometry"]["x_max"],
                0.0,
                allowed_types=warmup_allowed_types,
            )
            params, opt_state, loss = train_step(params, opt_state, batch)
            loss_value = float(jax.device_get(loss))
            king.update(params, loss_value)
            if (i + 1) % 1000 == 0:
                print("[Warmup] Iter {0} | Loss: {1:.2e}".format(i + 1, loss_value))
            if holdout_every > 0 and (i + 1) % holdout_every == 0:
                holdout_key, batch_key = jax.random.split(holdout_key)
                held_mse, held_rel_l2 = _ic_holdout_metrics(params, cfg, bounds, holdout_batch_size, batch_key)
                print(
                    "[Warmup Holdout] Iter {0} | MSE: {1:.2e} | RelL2: {2:.2%}".format(
                        i + 1, held_mse, held_rel_l2
                    )
                )
            if warmup_audit_every > 0 and (i + 1) % warmup_audit_every == 0:
                _, warmup_err = audit_global_fast(params, cfg, 0.0)
                if warmup_err < best_warmup_err:
                    best_warmup_err = warmup_err
                    best_warmup_params = clone_params(params)
                    print("[Warmup Audit] New best strict IC audit: {0:.2%}".format(best_warmup_err))

        if warmup_audit_every > 0 and best_warmup_err < float("inf"):
            params = clone_params(best_warmup_params)
            print("Warmup best checkpoint selected from strict audit: {0:.2%}".format(best_warmup_err))
        else:
            params = clone_params(king.best_state)

        audit_global_fast(params, cfg, 0.0)
        failed_warmup = diagnose_model(params, cfg, t_max=0.0)
        if failed_warmup:
            print("Failed Warmup on {0}. Correction.".format(failed_warmup))
            params, success = targeted_correction(
                params,
                cfg,
                bounds,
                0.0,
                failed_warmup,
                min(5000, cfg["training"]["n_iters_correction"]),
                start_lr=cfg["training"]["learning_rate"],
            )
            if not success:
                print("Critical failure.")
                return params, metrics
            audit_global_fast(params, cfg, 0.0)

    time_steps = generate_time_steps(cfg)
    print("Multi zone training : {0}".format(time_steps))
    for t_step in time_steps:
        step_start = os.times()[4]
        params, success, err = train_step_time_window(params, cfg, bounds, t_step, cfg["training"]["n_iters_per_step"])
        elapsed = os.times()[4] - step_start
        metrics["time_steps"].append({"t_step": t_step, "success": success, "err": err, "elapsed_sec": elapsed})
        if success:
            save_pickle(os.path.join(save_dir, "model_checkpoint_t{0}.pkl".format(t_step)), {"t_max": t_step, "params": params})
            print("Level t={0} OK.".format(t_step))
        else:
            print("Stop at t={0}.".format(t_step))
            break

    print("Start of final polishing (Target: < 2% everywhere)")
    final_t = cfg["geometry"]["T_max"]
    target_strict = 0.02
    failed_ids = diagnose_model(params, cfg, threshold=target_strict, t_max=final_t)
    if failed_ids:
        print("IC above {0:.1%}: {1}".format(target_strict, failed_ids))
        polish_start = os.times()[4]
        params, success_polish = targeted_correction(
            params,
            cfg,
            bounds,
            t_max=final_t,
            initial_failed_ids=failed_ids,
            n_iters_base=min(30000, cfg["training"]["n_iters_correction"] * 3),
            start_lr=1e-5,
            target_threshold=target_strict,
            apply_80_20=True,
        )
        metrics["final_polish"] = {"success": success_polish, "elapsed_sec": os.times()[4] - polish_start}
        if success_polish:
            save_pickle(os.path.join(save_dir, "model_final_elite_2percent.pkl"), {"t_max": final_t, "params": params, "note": "Polished < 2%"})
            diagnose_model(params, cfg, threshold=target_strict, t_max=final_t)
    else:
        print("Everything under 2%")

    return params, metrics
