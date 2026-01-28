import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION DU CHEMIN ---
# Ajout du dossier courant au path pour trouver les modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# --- 2. IMPORTS MIS À JOUR ---
from config import Config  # Nouvelle source de vérité

from src.models.models import PI_DeepONet_ADR
from src.train.smart_trainer import train_smart_time_marching
from src.visualization.plots import (
    compare_model_vs_cn_snapshots, 
    animate_model_vs_cn, 
    plot_error_heatmaps, 
    evaluate_global_accuracy
)

def run_smoke_test():
    print("🚀 DÉMARRAGE DU SMOKE TEST (Validation Rapide)...")

    # --- 3. OVERRIDE CONFIG (POUR LE TEST SEULEMENT) ---
    # On force des valeurs minuscules pour vérifier que ça ne plante pas
    print("   🔧 Modification temporaire de la Config pour le test...")
    Config.n_warmup = 10           # 10 itérations au lieu de 10000
    Config.n_iters_per_step = 10   # 10 itérations par palier
    Config.T_max = 0.2             # Juste 2 pas de temps (0.1 et 0.2)
    Config.n_sample = 50           # Petits samples pour l'audit
    Config.batch_size = 16         # Petit batch
    Config.max_retry = 1           # Pas d'acharnement si ça rate
    Config.save_dir = "./test_results_local"
    
    # SETUP DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    print(f"   📱 Device utilisé : {device}")

    # --- 4. INIT MODEL ---
    print("   🏗️ Initialisation du modèle...")
    # Plus besoin de passer d'arguments, tout est lu dans Config !
    model = PI_DeepONet_ADR().to(device)

    # --- 5. TRAIN LOOP ---
    print("   🏋️ Test de la boucle d'entraînement (Smart Time Marching)...")
    try:
        model = train_smart_time_marching(
            model, 
            bounds=Config.ranges, 
            n_warmup=Config.n_warmup, 
            n_iters_per_step=Config.n_iters_per_step
        )
        print("   ✅ Entraînement terminé sans erreur.")
    except Exception as e:
        print(f"   ❌ CRASH PENDANT L'ENTRAÎNEMENT : {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 6. VISUALIZATION ---
    save_dir = Config.save_dir
    os.makedirs(save_dir, exist_ok=True)

    print("\n   🎨 Test des Plots & Audits...")
    
    # Test Snapshots
    try:
        print("     - Génération Snapshots...", end="")
        compare_model_vs_cn_snapshots(model, save_dir=save_dir)
        print(" OK")
    except Exception as e:
        print(f" ÉCHEC ({e})")

    # Test Animation
    try:
        print("     - Génération Animation...", end="")
        animate_model_vs_cn(model, save_dir=save_dir)
        print(" OK")
    except Exception as e:
        print(f" ÉCHEC ({e})")

    # Test Heatmaps
    try:
        print("     - Génération Heatmaps...", end="")
        plot_error_heatmaps(model, Config.x_min, Config.x_max, Config.T_max, save_dir=save_dir)
        print(" OK")
    except Exception as e:
        print(f" ÉCHEC ({e})")

    # Test Audit CSV
    try:
        print("     - Audit Statistique (sur 5 cas)...", end="")
        evaluate_global_accuracy(model, 5, Config.ranges, Config.x_min, Config.x_max, Config.T_max, save_dir)
        print(" OK")
    except Exception as e:
        print(f" ÉCHEC ({e})")

    print(f"\n🎉 TEST TERMINÉ. Vérifiez le dossier '{save_dir}'.")

if __name__ == "__main__":
    run_smoke_test()