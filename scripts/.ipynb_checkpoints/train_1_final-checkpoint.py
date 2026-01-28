import sys
import os
import torch
import numpy as np
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.adr import PI_DeepONet_ADR
from src.training.smart_trainer import train_smart_adptive
from src.visualization.plots import (
    compare_model_vs_cn_snapshots, 
    plot_error_heatmaps,
    evaluate_global_accuracy
)

# --- CONFIGURATION PRODUCTION ---
CONFIG = {
    "bounds_phy": {'v': (0.5, 2.0), 'D': (0.01, 0.2), 'mu': (0.0, 1.0)},
    "x_min": -7.0, "x_max": 7.0, "Tmax": 1.0,

    # Architecture "Standard" (Pas Mini !)
    "branch_dim": 8, "trunk_dim": 2, "latent_dim": 64, 
    "branch_layers": [256, 256, 256, 256, 256], 
    "trunk_layers": [256, 256, 256, 256],
    "nFourier": 50, "sFourier": [0.0, 1.0],

    # Paramètres d'entraînement
    # Note: Pour un très gros résultat final, on pourra monter à 50k ou 100k itérations plus tard
    "n_warmup": 5000,     
    "n_physics": 15000,   
    "save_dir": "results_train_1_final" # Dossier de sortie clair
}

def run_production():
    print("🚀 DÉMARRAGE TRAINING PRODUCTION (Jean Zay)...")

    # 1. Device (Cuda forcé si dispo, sinon CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device utilisé: {device}")
    
    if device.type == 'cpu':
        print("⚠️ ATTENTION: Tu tournes sur CPU ! Vérifie ton allocation Slurm.")

    # 2. Modèle
    lb_geom = np.array([CONFIG["x_min"], 0.0])
    ub_geom = np.array([CONFIG["x_max"], CONFIG["Tmax"]])

    model = PI_DeepONet_ADR(
        branch_dim=CONFIG["branch_dim"],
        trunk_dim=CONFIG["trunk_dim"],
        latent_dim=CONFIG["latent_dim"],
        branch_layers=CONFIG["branch_layers"],
        trunk_layers=CONFIG["trunk_layers"],
        num_fourier_features=CONFIG["nFourier"],
        fourier_scales=CONFIG["sFourier"],
        lb_geom=lb_geom, ub_geom=ub_geom,
        phy_bounds=CONFIG["bounds_phy"]
    ).to(device)

    print(f"   Modèle initialisé (Layers: {CONFIG['branch_layers']})")

    # 3. Entraînement
    print(f"   Lancement Smart Trainer ({CONFIG['n_warmup']} warmup + {CONFIG['n_physics']} physics)...")
    start = time.time()

    model = train_smart_adptive(
        model, 
        bounds=CONFIG["bounds_phy"], 
        n_warmup=CONFIG["n_warmup"], 
        n_physics=CONFIG["n_physics"]
    )

    duration = (time.time()-start)/60
    print(f"✅ Fin entraînement en {duration:.1f} min.")

    # 4. Visualisation & Sauvegarde
    save_dir = CONFIG["save_dir"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"   Génération des résultats dans '{save_dir}'...")

    try:
        # Sauvegarde du modèle complet (IMPORTANT pour ne pas perdre le calcul)
        torch.save(model.state_dict(), os.path.join(save_dir, "model_final.pth"))
        print("   Modèle .pth sauvegardé.")

        compare_model_vs_cn_snapshots(model, CONFIG["x_min"], CONFIG["x_max"], CONFIG["Tmax"], save_dir)
        plot_error_heatmaps(model, CONFIG["x_min"], CONFIG["x_max"], CONFIG["Tmax"], save_dir)
        evaluate_global_accuracy(model, 10, CONFIG["bounds_phy"], CONFIG["x_min"], CONFIG["x_max"], CONFIG["Tmax"], save_dir)
    except Exception as e:
        print(f"❌ Erreur pendant la visualisation: {e}")

    print("Terminé.")

if __name__ == "__main__":
    run_production()