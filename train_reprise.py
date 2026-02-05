import sys
import os
import torch
from datetime import datetime

# --- GESTION DES CHEMINS ---
project_root = os.getcwd()
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path: sys.path.append(src_path)

# --- IMPORTS DE REPRISE ---
from config_reprise import Config  # On importe la nouvelle config
from src.models.adr import PI_DeepONet_ADR
from smart_trainer_reprise import train_smart_time_marching_reprise

# CHEMIN DU CHECKPOINT (À adapter selon ton dossier)
CHECKPOINT_PATH = "results/run_20260204-083040/model_checkpoint_t1.6.pth"

def main():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(project_root, "results", f"reprise_from_1.6_{timestamp}")
    Config.save_dir = run_dir
    os.makedirs(run_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔄 REPRISE DU RUN DEPUIS t=1.6")
    print(f"📁 Nouveau dossier : {run_dir}")

    # 1. Initialisation Modèle
    model = PI_DeepONet_ADR().to(device)

    # 2. Chargement des poids
    print(f"📂 Chargement du checkpoint : {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    # Si ton checkpoint contient un dictionnaire {'model_state_dict': ...}
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        start_t = checkpoint['t_max']
    else:
        model.load_state_dict(checkpoint)
        start_t = 1.6
    
    print(f"✅ Modèle chargé avec succès (Ancien t_max = {start_t})")

    # 3. Lancement de l'entraînement de reprise
    try:
        model = train_smart_time_marching_reprise(
            model,
            bounds=Config.ranges,
            start_t=start_t, # On passe le point de départ
            n_iters_per_step=Config.n_iters_per_step
        )
    except Exception as e:
        torch.save(model.state_dict(), os.path.join(run_dir, "model_REPRISE_CRASH.pth"))
        raise e

    torch.save(model.state_dict(), os.path.join(run_dir, "model_final_reprise.pth"))
    print(f"✅ Reprise terminée avec succès.")

if __name__ == "__main__":
    main()