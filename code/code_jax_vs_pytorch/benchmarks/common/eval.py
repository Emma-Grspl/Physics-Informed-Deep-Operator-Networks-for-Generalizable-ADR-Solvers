import time

import numpy as np

from src.utils.CN_ADR import crank_nicolson_adr


def get_ic_value_numpy(x, ic_params):
    x = np.asarray(x)
    types = ic_params.get("type")
    A = ic_params.get("A", 1.0)
    x0 = ic_params.get("x0", 0.0)
    sigma = ic_params.get("sigma", 0.5)
    k = ic_params.get("k", 2.0)

    u0 = np.zeros_like(x, dtype=float)

    if types == 0:
        u0 += np.tanh((x - x0) / (sigma + 1e-8))
    elif types in [1, 2]:
        u0 += A * np.exp(-((x - x0) ** 2) / (2 * sigma**2 + 1e-8)) * np.sin(k * x)
    elif types in [3, 4]:
        u0 += A * np.exp(-((x - x0) ** 2) / (2 * sigma**2 + 1e-8))

    return u0


def _predict_grid_common(cfg, params_dict, t_max, nx, nt):
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
    return x, t, xt, p_batch


def compute_cn_solution(cfg, params_dict, t_max, nx, nt):
    x_min = cfg["geometry"]["x_min"]
    x_max = cfg["geometry"]["x_max"]
    x = np.linspace(x_min, x_max, nx)
    u0 = get_ic_value_numpy(x, params_dict)
    bc_kind = "zero_zero" if params_dict["type"] in [1, 3, 2, 4] else "tanh_pm1"
    _, u_cn, _ = crank_nicolson_adr(
        v=params_dict["v"],
        D=params_dict["D"],
        mu=params_dict["mu"],
        xL=x_min,
        xR=x_max,
        Nx=nx,
        Tmax=t_max,
        Nt=nt,
        bc_kind=bc_kind,
        x0=x,
        u0=u0,
    )
    if u_cn.shape == (nt, nx):
        u_cn = u_cn.T
    return u_cn


def evaluate_cases(cfg, benchmark_cfg, cases, predict_grid_fn):
    nx = benchmark_cfg["inference"]["nx"]
    nt = benchmark_cfg["inference"]["nt"]
    t_max = benchmark_cfg["training"]["t_max"]
    summary = {"global_l2": [], "families": {}}

    for case in cases:
        family = case["family"]
        params_dict = case["params"]
        _, _, xt, p_batch = _predict_grid_common(cfg, params_dict, t_max, nx, nt)
        u_pred = predict_grid_fn(p_batch, xt, nx, nt)
        u_cn = compute_cn_solution(cfg, params_dict, t_max, nx, nt)
        err = np.linalg.norm(u_cn - u_pred) / (np.linalg.norm(u_cn) + 1e-8)
        summary["global_l2"].append(float(err))
        summary["families"].setdefault(family, []).append(float(err))

    family_means = {k: float(np.mean(v)) for k, v in summary["families"].items()}
    family_stds = {k: float(np.std(v)) for k, v in summary["families"].items()}
    return {
        "global_l2_mean": float(np.mean(summary["global_l2"])),
        "global_l2_std": float(np.std(summary["global_l2"])),
        "global_l2_values": summary["global_l2"],
        "family_l2_mean": family_means,
        "family_l2_std": family_stds,
    }


def benchmark_inference(cfg, benchmark_cfg, build_inputs_fn, predict_fn, sync_fn):
    nx = benchmark_cfg["inference"]["nx"]
    nt = benchmark_cfg["inference"]["nt"]
    batch_size = benchmark_cfg["inference"]["batch_size"]
    warmup_iters = benchmark_cfg["inference"]["warmup_iters"]
    t_max = benchmark_cfg["training"]["t_max"]

    rng = np.random.default_rng(benchmark_cfg["seed"])
    allowed_types = benchmark_cfg["inference"].get("allowed_types", [0, 1, 3])
    scenarios = []
    for _ in range(batch_size):
        p_dict = {k: rng.uniform(v[0], v[1]) for k, v in cfg["physics_ranges"].items()}
        p_dict["type"] = int(rng.choice(allowed_types))
        scenarios.append(p_dict)

    inference_inputs = build_inputs_fn(scenarios, t_max, nx, nt, full_grid=True)
    timejump_inputs = build_inputs_fn(scenarios, t_max, nx, nt, full_grid=False)

    for _ in range(warmup_iters):
        sync_fn(predict_fn(*inference_inputs))
        sync_fn(predict_fn(*timejump_inputs))

    start = time.perf_counter()
    inference_result = predict_fn(*inference_inputs)
    sync_fn(inference_result)
    inference_time = time.perf_counter() - start

    start = time.perf_counter()
    timejump_result = predict_fn(*timejump_inputs)
    sync_fn(timejump_result)
    timejump_time = time.perf_counter() - start

    start = time.perf_counter()
    for params_dict in scenarios:
        compute_cn_solution(cfg, params_dict, t_max, nx, nt)
    cn_time = time.perf_counter() - start

    return {
        "batch_size": batch_size,
        "nx": nx,
        "nt": nt,
        "t_max": t_max,
        "inference_full_grid_sec": inference_time,
        "time_jump_sec": timejump_time,
        "cn_reference_sec": cn_time,
        "time_jump_speedup_vs_cn": cn_time / max(timejump_time, 1e-8),
    }
