import sys
import os
import torch
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import CONFIG 
from src.models.adr import PI_DeepONet_ADR
from src.training.smart_trainer import train_smart_time_marching
from src.visualization.plots import (
    compare_model_vs_cn_snapshots, 
    plot_error_heatmaps,
    evaluate_global_accuracy
)

def run_production_v2():
    print("🚀 DÉMARRAGE TRAINING V2 (Optuna Friendly)...")
    SAVE_DIR = "results_train_2_final"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device utilisé: {device}")

    # Modèle
    net_cfg = CONFIG["network"]
    lb_geom = np.array([CONFIG["x_min"], 0.0])
    ub_geom = np.array([CONFIG["x_max"], CONFIG["Tmax"]])

    model = PI_DeepONet_ADR(
        branch_dim=net_cfg["branch_dim"],
        trunk_dim=net_cfg["trunk_dim"],
        latent_dim=net_cfg["latent_dim"],
        branch_layers=net_cfg["branch_layers"],
        trunk_layers=net_cfg["trunk_layers"],
        num_fourier_features=net_cfg["nFourier"],
        fourier_scales=net_cfg["sFourier"],
        lb_geom=lb_geom, ub_geom=ub_geom,
        phy_bounds=CONFIG["bounds_phy"]
    ).to(device)

    # --- C'est ici que la magie Optuna opérera plus tard ---
    # On récupère les valeurs depuis le Config file
    train_cfg = CONFIG["training"]
    
    start = time.time()

    model = train_smart_time_marching(
        model, 
        bounds=CONFIG["bounds_phy"], 
        n_warmup=train_cfg["n_warmup"],       # <--- Pilotable par Config/Optuna
        n_iters_per_step=train_cfg["n_physics"], # <--- Pilotable par Config/Optuna
        save_dir=SAVE_DIR
    )

    print(f"✅ Fin totale en {(time.time()-start)/60:.1f} min.")

    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    try:
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model_v2_final.pth"))
        compare_model_vs_cn_snapshots(model, CONFIG["x_min"], CONFIG["x_max"], CONFIG["Tmax"], SAVE_DIR)
        plot_error_heatmaps(model, CONFIG["x_min"], CONFIG["x_max"], CONFIG["Tmax"], SAVE_DIR)
        
        print("\n📊 AUDIT FINAL (1000 cas/type)...")
        evaluate_global_accuracy(
            model, 
            n_tests_per_type=1000, 
            bounds=CONFIG["bounds_phy"], 
            x_min=CONFIG["x_min"], x_max=CONFIG["x_max"], t_max=CONFIG["Tmax"], 
            save_dir=SAVE_DIR
        )
    except Exception as e:
        print(f"❌ Erreur visus: {e}")

if __name__ == "__main__":
    run_production_v2()