import numpy as np


def generate_eval_cases(cfg, benchmark_cfg):
    rng = np.random.default_rng(benchmark_cfg["seed"])
    families_cfg = benchmark_cfg["evaluation"]["families"]
    n_cases = benchmark_cfg["evaluation"]["n_cases_per_family"]
    cases = []
    for family_name, type_ids in families_cfg.items():
        for case_idx in range(n_cases):
            p_dict = {k: rng.uniform(v[0], v[1]) for k, v in cfg["physics_ranges"].items()}
            p_dict["type"] = int(type_ids[case_idx % len(type_ids)])
            cases.append({"family": family_name, "params": p_dict})
    return cases
