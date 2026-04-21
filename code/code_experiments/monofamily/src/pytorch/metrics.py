"""Comparison-layer module `jax_comparison.monofamily.src.pytorch.metrics`.

This file supports the PyTorch versus JAX comparison workflows, including replicated implementations, scripts, or validation helpers.
"""

import torch
import numpy as np
import sys
import os

file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.utils.CN_ADR import get_ground_truth_CN
except ImportError as e:
    print(f"err: {e}")
    sys.exit(1)

def diagnose_model(model, device, cfg, threshold=None, t_max=None):
    """
    Diagnoses the model accuracy for each IC family over a given time window.This function performs a quality audit. It generates 20 random test cases for each family (Gaussian, Sin-Gauss, Tanh), calculates the exact solution using the Crank-Nicolson solver, and measures the relative L2 error. It allows you to decide if a targeted correction is necessary for certain families.
    Args:
        model(nn.Module): The PI-DeepONet model to diagnose.
        device(torch.device): The device (CPU/GPU) for the calculations.
        cfg(dict): Complete configuration dictionary (YAML)
        threshold(float, optional): Relative L2 error threshold. If None, it is chosen dynamically in cfg (threshold_ic or threshold_step).
        t_max(float, optional): Upper bound of the time window to be tested. If None, uses the geometry's T_max.
    Outputs:
        list[int]: List of type IDs (0 to 4) that failed the accuracy test. An empty list means the model is valid everywhere.
    """
    model.eval()
    
    if t_max is None: t_max = cfg['geometry']['T_max']
    
    if threshold is None:
        if t_max == 0.0:
            threshold = cfg['training'].get('threshold_ic', cfg['training'].get('threshold', 0.01))
        else:
            threshold = cfg['training'].get('threshold_step', cfg['training'].get('threshold', 0.05))
    
    families_map = {
        "Gaussian": [3, 4], 
        "Sin-Gauss": [1, 2], 
        "Tanh": [0]
    }
    allowed_types = set(cfg.get('training', {}).get('allowed_types', [0, 1, 2, 3, 4]))
    families_map = {
        fam_name: [tid for tid in type_ids if tid in allowed_types]
        for fam_name, type_ids in families_map.items()
    }
    families_map = {fam_name: type_ids for fam_name, type_ids in families_map.items() if type_ids}
    
    failed_ids = []
    
    print(f"Diag (t_max={t_max}, Threshold: {threshold:.1%})...")
    
    for fam_name, type_ids in families_map.items():
        errors = []
        for _ in range(20): 
            p_dict = {k: np.random.uniform(v[0], v[1]) for k, v in cfg['physics_ranges'].items()}
            current_id = np.random.choice(type_ids)
            p_dict['type'] = current_id 
            try:
                X_grid, T_grid, U_true_np = get_ground_truth_CN(
                    p_dict, cfg, t_step_max=t_max
                )
            except Exception as e:
                continue
            X_flat = X_grid.flatten()
            T_flat = T_grid.flatten()
            U_true = U_true_np.flatten()
            
            xt_tensor = torch.tensor(np.stack([X_flat, T_flat], axis=1), dtype=torch.float32).to(device)
            p_vec = np.array([p_dict['v'], p_dict['D'], p_dict['mu'], p_dict['type'], 
                              p_dict['A'], 0.0, p_dict['sigma'], p_dict['k']])
            p_tensor = torch.tensor(p_vec, dtype=torch.float32).repeat(len(X_flat), 1).to(device)

            with torch.no_grad():
                u_pred = model(p_tensor, xt_tensor).cpu().numpy().flatten()
            norm = np.linalg.norm(U_true) + 1e-7
            err = np.linalg.norm(U_true - u_pred) / norm
            errors.append(err)
            
        if not errors:
            print(f"  - {fam_name:<12} : err solver")
            continue

        mean_err = np.mean(errors)
        status = "✅" if mean_err < threshold else "❌"
        print(f"  - {fam_name:<12} : {mean_err:.2%} {status}")
        
        if mean_err > threshold:
            failed_ids.extend(type_ids)
            
    return failed_ids
