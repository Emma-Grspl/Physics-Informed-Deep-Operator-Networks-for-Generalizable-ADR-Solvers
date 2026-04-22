import optuna
import yaml
import os
import sys
from pathlib import Path

# --- CONFIGURATION ---
DB_URL = "sqlite:///optuna_study.db"
STUDY_NAME = "ADR_Final_Optimization" 

def main():
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=DB_URL)
        print(f"{len(study.trials)} complete trial.")
    except Exception as e:
        return

    best_trial = study.best_trial
    print(f"Best trial : {best_trial.number}| L2 : {best_trial.value:.4%} ")
    print(" Hyperparameters :")
    for key, value in best_trial.params.items():
        print(f"- {key}: {value}")
    
    fourier_presets = {
        "standard": [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 24.0],
        "high_freq": [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
        "dense": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
    }

    base_config_path = Path(__file__).resolve().parents[2] / "configs" / "config_ADR.yaml"
    with open(base_config_path, 'r') as f:
        final_cfg = yaml.safe_load(f)

    params = best_trial.params
    width = params['width']
    depth = params['depth']
    final_cfg['model']['branch_width'] = width
    final_cfg['model']['trunk_width'] = width
    final_cfg['model']['latent_dim'] = width
    final_cfg['model']['branch_depth'] = depth
    final_cfg['model']['trunk_depth'] = depth - 1
    
    # Fourier
    mode = params['fourier_mode']
    final_cfg['model']['sFourier'] = fourier_presets[mode]
    final_cfg['model']['nFourier'] = len(final_cfg['model']['sFourier']) * 2

    # Training Params
    final_cfg['training']['learning_rate'] = params['lr']
    
    # Loss Weights
    final_cfg['loss_weights']['weight_ic_init'] = params['w_ic_init']
    final_cfg['loss_weights']['weight_bc'] = params['w_bc']
    final_cfg['loss_weights']['first_w_res'] = params['w_res_target']

    print("Production training")
    final_cfg['training']['n_warmup'] = 7000          
    final_cfg['training']['n_iters_per_step'] = 6000
    final_cfg['training']['n_iters_correction'] = 8000
    final_cfg['training']['nb_loop'] = 3              
    final_cfg['training']['threshold'] = 0.02

    # Save
    output_filename = "final_optimized_config.yaml"
    with open(output_filename, 'w') as f:
        yaml.dump(final_cfg, f)
    
    print(f"Config saved in : {output_filename}")

if __name__ == "__main__":
    main()
