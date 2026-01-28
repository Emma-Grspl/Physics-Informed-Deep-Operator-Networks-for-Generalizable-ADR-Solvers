import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ajout du dossier parent au path pour voir les modules 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Imports de tes modules
from src.config import CONFIG
from src.models.adr import PI_DeepONet_ADR
from src.training.smart_trainer import train_smart_adptive
from src.visualization.plots import (
    compare_model_vs_cn_snapshots, 
    animate_model_vs_cn, 
    plot_error_heatmaps, 
    evaluate_global_accuracy
)

def run_smoke_test():
    print("🚀 DÉMARRAGE DU CRASH TEST (Mode Rapide)...")

    # 1. SETUP DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    print(f"   Device: {device}")

    # 2. INIT MODEL
    print("   Initialisation du modèle...")
    # On prépare les bornes géométriques
    lb_geom = np.array([CONFIG["x_min"], 0.0])
    ub_geom = np.array([CONFIG["x_max"], CONFIG["Tmax"]])

    model = PI_DeepONet_ADR(
        branch_dim=CONFIG["network"]["branch_dim"],
        trunk_dim=CONFIG["network"]["trunk_dim"],
        latent_dim=CONFIG["network"]["latent_dim"],
        branch_layers=CONFIG["network"]["branch_layers"],
        trunk_layers=CONFIG["network"]["trunk_layers"],
        num_fourier_features=CONFIG["network"]["nFourier"],
        fourier_scales=CONFIG["network"]["sFourier"],
        lb_geom=lb_geom,
        ub_geom=ub_geom,
        phy_bounds=CONFIG["bounds_phy"]
    ).to(device)

    # 3. TRAIN (VERSION MINIATURE)
    print("   Test de la boucle d'entraînement (Smart Trainer)...")
    # On force des valeurs très petites pour tester vite
    model = train_smart_adptive(
        model, 
        bounds=CONFIG["bounds_phy"], 
        n_warmup=10,    # Seulement 10 itérations !
        n_physics=10    # Seulement 10 itérations !
    )
    print("   ✅ Entraînement terminé sans crash.")

    # 4. VISUALIZATION
    save_dir = "test_results_local"
    os.makedirs(save_dir, exist_ok=True)

    print("   Test des Plots...")
    try:
        compare_model_vs_cn_snapshots(model, CONFIG["x_min"], CONFIG["x_max"], CONFIG["Tmax"], save_dir)
        print("     - Snapshots: OK")
    except Exception as e:
        print(f"     - Snapshots: ÉCHEC ({e})")

    try:
        plot_error_heatmaps(model, CONFIG["x_min"], CONFIG["x_max"], CONFIG["Tmax"], save_dir)
        print("     - Heatmaps: OK")
    except Exception as e:
        print(f"     - Heatmaps: ÉCHEC ({e})")

    try:
        # On teste l'audit global sur seulement 2 cas
        evaluate_global_accuracy(model, 2, CONFIG["bounds_phy"], CONFIG["x_min"], CONFIG["x_max"], CONFIG["Tmax"], save_dir)
        print("     - Audit CSV: OK")
    except Exception as e:
        print(f"     - Audit CSV: ÉCHEC ({e})")

    print(f"\n🎉 TEST COMPLET TERMINÉ. Vérifie le dossier '{save_dir}' pour les images.")

if __name__ == "__main__":
    run_smoke_test()
