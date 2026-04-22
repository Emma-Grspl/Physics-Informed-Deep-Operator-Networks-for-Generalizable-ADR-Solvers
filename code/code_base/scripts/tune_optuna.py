"""
Allows launching Optuna trials. The hyperparameters that vary are the architecture, learning rate, Fourier features, and loss weights.
"""
import os
import sys
import yaml
import copy
import torch
import numpy as np
import optuna
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import src.training.trainer_ADR as trainer_module
from src.models.PI_DeepONet_ADR import PI_DeepONet_ADR
from src.utils.CN_ADR import get_ground_truth_CN
optuna.logging.set_verbosity(optuna.logging.INFO)

#validation function
def evaluate_model(model, cfg, device):
    """
    Calculate the mean error L2 for the three standard cases (Tanh, Sin-Gauss, Gaussian).
    This is the final score for the trial.
    """
    model.eval()
    errors = []
    target_types = [0, 1, 3] # Tanh, Sin-Gauss, Gaussian
    p_base = {k: np.mean(v) for k, v in cfg['physics_ranges'].items()}
    
    print("Optuna : Validation in progress")
    for tid in target_types:
        p = p_base.copy()
        p['type'] = tid
        
        try:
            # CN
            X, T, U_true = get_ground_truth_CN(p, cfg, t_step_max=3.0)
            
            # Inputs
            x_flat = X.flatten()
            t_flat = T.flatten()
            p_vec = np.array([p['v'], p['D'], p['mu'], p['type'], p['A'], 0.0, p['sigma'], p['k']])
            
            p_tensor = torch.tensor(p_vec, dtype=torch.float32).repeat(len(x_flat), 1).to(device)
            xt_tensor = torch.tensor(np.stack([x_flat, t_flat], axis=1), dtype=torch.float32).to(device)
            
            with torch.no_grad():
                u_pred = model(p_tensor, xt_tensor).cpu().numpy().flatten()
            
            # Relative L2
            u_true_flat = U_true.flatten()
            err = np.linalg.norm(u_true_flat - u_pred) / (np.linalg.norm(u_true_flat) + 1e-8)
            errors.append(err)
            
        except Exception as e:
            print(f"Err validation IC type {tid}: {e}")
            return 1.0 

    mean_error = np.mean(errors)
    print(f"Mean L2: {mean_error:.4%}")
    return mean_error

#objective function
def objective(trial):
    base_cfg = copy.deepcopy(trainer_module.load_config())
    trial_cfg = base_cfg
    
    trial_cfg['training']['n_warmup'] = 4000
    trial_cfg['training']['n_iters_per_step'] = 3000
    trial_cfg['training']['n_iters_correction'] = 5000
    trial_cfg['training']['n_sample'] = 12288
    trial_cfg['training']['batch_size'] = 4096
    trial_cfg['training']['nb_loop'] = 2
    trial_cfg['training']['max_retry'] = 3
    trial_cfg['training']['threshold'] = 0.03 
    trial_cfg['training']['rolling_window'] = 2000

    #Optuna hyperparameters    
    width = trial.suggest_categorical("width", [128, 256, 512])
    depth = trial.suggest_int("depth", 4, 6)
    
    trial_cfg['model']['branch_width'] = width
    trial_cfg['model']['trunk_width'] = width
    trial_cfg['model']['latent_dim'] = width 
    trial_cfg['model']['branch_depth'] = depth
    trial_cfg['model']['trunk_depth'] = depth - 1 
    
    # Fourier Features 
    fourier_presets = {
        "standard": [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 24.0],
        "high_freq": [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
        "dense": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
    }
    fourier_mode = trial.suggest_categorical("fourier_mode", ["standard", "high_freq", "dense"])
    trial_cfg['model']['sFourier'] = fourier_presets[fourier_mode]
    trial_cfg['model']['nFourier'] = len(trial_cfg['model']['sFourier']) * 2

    # LR
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    trial_cfg['training']['learning_rate'] = lr
    
    # Loss Weights
    w_ic_init = trial.suggest_categorical("w_ic_init", [200.0, 500.0, 1000.0, 2000.0])
    w_bc = trial.suggest_categorical("w_bc", [100.0, 200.0, 500.0])
    w_res_target = trial.suggest_categorical("w_res_target", [200.0, 500.0, 1000.0])
    
    trial_cfg['loss_weights']['weight_ic_init'] = w_ic_init
    trial_cfg['loss_weights']['weight_bc'] = w_bc
    trial_cfg['loss_weights']['first_w_res'] = w_res_target

    trial_dir = os.path.join(project_root, "results", "optuna", f"trial_{trial.number:04d}")
    os.makedirs(trial_dir, exist_ok=True)
    trial_cfg['audit']['save_dir'] = trial_dir
    
    trainer_module.cfg = trial_cfg
    
    real_finder = trainer_module.find_latest_checkpoint
    def smart_finder_redirect(ignored_path):
        return real_finder(trial_dir)
    trainer_module.find_latest_checkpoint = smart_finder_redirect
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        print(f"Trial = {trial.number}] Start | W={width} D={depth} | LR={lr:.2e} | Mode={fourier_mode} | Weights: IC={w_ic_init} BC={w_bc} PDE={w_res_target}")
        
        model = PI_DeepONet_ADR(trial_cfg).to(device)
        model = trainer_module.train_smart_time_marching(model, bounds=trial_cfg['physics_ranges'])
        
        # Validation
        score = evaluate_model(model, trial_cfg, device)
        trial.set_user_attr("score_final", score)
        return score

    except Exception as e:
        print(f"Trial {trial.number}] CRASH: {e}")
        return float('inf') 
    
    finally:
        trainer_module.find_latest_checkpoint = real_finder
        if 'model' in locals(): del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    db_path = os.path.join(project_root, "optuna_study.db")
    storage_url = f"sqlite:///{db_path}"
    
    print(f"Launching optuna")
    
    study = optuna.create_study(
        study_name="ADR_Final_Optimization", 
        storage=storage_url,
        load_if_exists=True,
        direction="minimize"
    )
    
    study.optimize(objective, n_trials=50)

    print("Best results : ")
    print(f" Error: {study.best_value:.4%} | Params: {study.best_params}")
    
    # Save config
    best = study.best_params
    cfg = trainer_module.load_config()
    
    # updated with the winner
    cfg['model']['branch_width'] = best['width']
    cfg['model']['trunk_width'] = best['width']
    cfg['model']['latent_dim'] = best['width']
    cfg['model']['branch_depth'] = best['depth']
    cfg['model']['trunk_depth'] = best['depth'] - 1
    
    fourier_presets = {
        "standard": [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 24.0],
        "high_freq": [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
        "dense": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
    }
    cfg['model']['sFourier'] = fourier_presets[best['fourier_mode']]
    cfg['model']['nFourier'] = len(cfg['model']['sFourier']) * 2
    
    cfg['training']['learning_rate'] = best['lr']
    cfg['loss_weights']['weight_ic_init'] = best['w_ic_init']
    cfg['loss_weights']['weight_bc'] = best['w_bc']
    cfg['loss_weights']['first_w_res'] = best['w_res_target']
    
    cfg['training']['n_warmup'] = 7000
    cfg['training']['n_iters_per_step'] = 6000
    cfg['training']['threshold'] = 0.02

    with open("best_config_optuna.yaml", "w") as f:
        yaml.dump(cfg, f)
    print("Final config : best_config_optuna.yaml")