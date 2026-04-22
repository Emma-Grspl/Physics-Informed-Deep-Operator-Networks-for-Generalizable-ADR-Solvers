"""
Folder with a main function allowing you to organize and launch all network training
"""
import sys
import os
import torch
import yaml
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    import src.training.trainer_ADR as trainer_module 
    from src.models.PI_DeepOnet_ADR import PI_DeepONet_ADR
except ImportError as e:
    print(f"Import err: {e}")
    sys.exit(1)

def main():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"run_{timestamp}"
    run_dir = os.path.join(project_root, "results", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Updated exit file")
    trainer_module.cfg['audit']['save_dir'] = run_dir
    cfg = trainer_module.cfg 

    print(f"outputs file : {run_dir}")
    print(f"Key parameters : Tmax: {cfg['geometry']['T_max']}, Samples : {cfg['training']['n_sample']}, Macro loops : : {cfg['training']['nb_loop']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    
    model = PI_DeepONet_ADR(cfg).to(device)

    # Training
    try:
        model = trainer_module.train_smart_time_marching(model, bounds=cfg['physics_ranges'])
    except Exception as e:
        print(f"Critical err : {e}")
        emergency_path = os.path.join(run_dir, "model_CRASHED.pth")
        torch.save(model.state_dict(), emergency_path)
        print(f"Crash saved: {emergency_path}")
        raise e

    # 6. SAUVEGARDE FINALE
    final_model_path = os.path.join(run_dir, "model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved : {final_model_path}")

if __name__ == "__main__":
    main()