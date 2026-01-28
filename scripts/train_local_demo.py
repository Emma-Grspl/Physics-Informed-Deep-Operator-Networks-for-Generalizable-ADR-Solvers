import sys
import os
import torch
import numpy as np
import time

# Ajout du path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# On importe les modules
from src.models.adr import PI_DeepONet_ADR
from src.training.smart_trainer import train_smart_adptive
from src.visualization.plots import (
    compare_model_vs_cn_snapshots, 
    plot_error_heatmaps,
    evaluate_global_accuracy
)

# --- CONFIGURATION LOCALE "LITE" ---
# On écrase la config globale pour une version légère adaptée au Mac
LOCAL_CONFIG = {
    "bounds_phy": {'v': (0.5, 2.0), 'D': (0.01, 0.2), 'mu': (0.0, 1.0)},
    "x_min": -7.0, "x_max": 7.0, "Tmax": 1.0,

    # Réseau "Mini" (64 neurones au lieu de 256)
    "branch_dim": 8, "trunk_dim": 2, "latent_dim": 64, 
    "branch_layers": [64, 64, 64], 
    "trunk_layers": [64, 64, 64],
    "nFourier": 20, "sFourier": [0.0, 1.0], # Moins de Fourier pour aller vite

    # Entraînement Court mais significatif
    "n_warmup": 500,     # Assez pour apprendre la forme initiale
    "n_physics": 2000,   # Assez pour commencer à voir la physique
    "save_dir": "results_local_demo"
}

def run_local_demo():
    print("🍎 DÉMARRAGE DÉMO LOCALE (MAC/PC)...")

    # 1. Device (MPS pour Mac M1/M2/M3 ou CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): 
        device = torch.device("mps")
        print("   ✅ Utilisation de l'accélération Apple Metal (MPS)")
    else:
        print(f"   Device: {device}")

    # 2. Modèle Léger
    lb_geom = np.array([LOCAL_CONFIG["x_min"], 0.0])
    ub_geom = np.array([LOCAL_CONFIG["x_max"], LOCAL_CONFIG["Tmax"]])

    model = PI_DeepONet_ADR(
        branch_dim=LOCAL_CONFIG["branch_dim"],
        trunk_dim=LOCAL_CONFIG["trunk_dim"],
        latent_dim=LOCAL_CONFIG["latent_dim"],
        branch_layers=LOCAL_CONFIG["branch_layers"],
        trunk_layers=LOCAL_CONFIG["trunk_layers"],
        num_fourier_features=LOCAL_CONFIG["nFourier"],
        fourier_scales=LOCAL_CONFIG["sFourier"],
        lb_geom=lb_geom, ub_geom=ub_geom,
        phy_bounds=LOCAL_CONFIG["bounds_phy"]
    ).to(device)

    print(f"   Modèle 'Lite' initialisé.")

    # 3. Entraînement
    print("   Lancement Smart Trainer (Warmup: 500, Physics: 2000)...")
    start = time.time()

    model = train_smart_adptive(
        model, 
        bounds=LOCAL_CONFIG["bounds_phy"], 
        n_warmup=LOCAL_CONFIG["n_warmup"], 
        n_physics=LOCAL_CONFIG["n_physics"]
    )

    print(f"   ✅ Fin entraînement ({ (time.time()-start)/60:.1f} min).")

    # 4. Visualisation
    save_dir = LOCAL_CONFIG["save_dir"]
    print(f"   Génération des résultats dans '{save_dir}'...")

    # On compare juste les snapshots et les heatmaps
    try:
        compare_model_vs_cn_snapshots(model, LOCAL_CONFIG["x_min"], LOCAL_CONFIG["x_max"], LOCAL_CONFIG["Tmax"], save_dir)
        plot_error_heatmaps(model, LOCAL_CONFIG["x_min"], LOCAL_CONFIG["x_max"], LOCAL_CONFIG["Tmax"], save_dir)

        # Petit audit sur 10 cas seulement
        evaluate_global_accuracy(model, 10, LOCAL_CONFIG["bounds_phy"], LOCAL_CONFIG["x_min"], LOCAL_CONFIG["x_max"], LOCAL_CONFIG["Tmax"], save_dir)
    except Exception as e:
        print(f"   ⚠️ Erreur Plotting: {e}")

    print("\n👋 Démo terminée. Regarde les images !")

if __name__ == "__main__":
    run_local_demo()
