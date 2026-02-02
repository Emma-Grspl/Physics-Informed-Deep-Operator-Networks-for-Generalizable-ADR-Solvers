import torch
import numpy as np
import sys
import os
from config import Config

# --- IMPORT DU SOLVEUR ---
sys.path.append(os.path.join(os.getcwd(), 'src'))
try:
    from physics.solver import get_ground_truth_CN
except ImportError:
    print("❌ Impossible d'importer get_ground_truth_CN depuis src.physics.solver")
    sys.exit(1)

def diagnose_model(model, device, threshold=0.03, t_max=None):
    """
    Diagnostique chaque famille sur la fenêtre [0, t_max] complète.
    Utilise la méthode ROBUSTE (Flatten) identique à l'audit global.
    """
    model.eval()
    
    # Par défaut, on audite tout si t_max n'est pas précisé
    if t_max is None: t_max = Config.T_max

    # Paramètres de résolution (identiques à l'audit global pour cohérence)
    Nx = Config.Nx_audit
    Nt = Config.Nt_audit

    families_map = {
        "Gaussian": [3, 4], 
        "Sin-Gauss": [1, 2], 
        "Tanh": [0]
    }
    
    failed_ids = []
    
    print(f"\n🩺 DIAGNOSTIC EN COURS (t_max={t_max}, Seuil: {threshold:.1%})...")
    
    for fam_name, type_ids in families_map.items():
        errors = []
        # On teste 20 cas aléatoires par famille
        for _ in range(20): 
            # 1. Paramètres aléatoires
            p_dict = Config.get_p_dict()
            # On force le type pour tester la famille spécifique
            current_id = np.random.choice(type_ids)
            p_dict['type'] = current_id 
            
            # 2. Vérité Terrain ROBUSTE
            # On récupère TOUTE la grille (X, T) et TOUTE la solution U
            try:
                X_grid, T_grid, U_true_np = get_ground_truth_CN(
                    p_dict, Config.x_min, Config.x_max, t_max, Nx, Nt
                )
            except Exception as e:
                # print(f"⚠️ Erreur solveur : {e}") 
                continue

            # 3. Aplatissement (Comme l'Audit Global)
            # Cela évite tout problème de transposition (Nx, Nt) vs (Nt, Nx)
            X_flat = X_grid.flatten()
            T_flat = T_grid.flatten()
            U_true = U_true_np.flatten()
            
            # 4. Préparation Input DeepONet
            xt_tensor = torch.tensor(np.stack([X_flat, T_flat], axis=1), dtype=torch.float32).to(device)
            
            p_vec = np.array([p_dict['v'], p_dict['D'], p_dict['mu'], p_dict['type'], 
                              p_dict['A'], p_dict['x0'], p_dict['sigma'], p_dict['k']])
            # On répète les paramètres pour chaque point de la grille
            p_tensor = torch.tensor(p_vec, dtype=torch.float32).unsqueeze(0).repeat(len(X_flat), 1).to(device)

            # 5. Prédiction & Erreur
            with torch.no_grad():
                u_pred = model(p_tensor, xt_tensor).cpu().numpy().flatten()
            
            # Erreur L2 Relative sur tout le volume spatio-temporel
            norm = np.linalg.norm(U_true) + 1e-7
            err = np.linalg.norm(U_true - u_pred) / norm
            errors.append(err)
            
        if not errors:
            print(f"  - {fam_name:<12} : ??? (Erreur Solveur)")
            continue

        mean_err = np.mean(errors)
        status = "✅" if mean_err < threshold else "❌"
        print(f"  - {fam_name:<12} : {mean_err:.2%} {status}")
        
        if mean_err > threshold:
            failed_ids.extend(type_ids)
            
    return failed_ids