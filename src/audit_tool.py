import torch
import numpy as np
import sys
import os

# --- IMPORT DU SOLVEUR ---
sys.path.append(os.path.join(os.getcwd(), 'src'))
try:
    from physics.solver import get_ground_truth_CN
except ImportError:
    print("❌ Impossible d'importer get_ground_truth_CN depuis src.physics.solver")
    sys.exit(1)

def diagnose_model(model, device, cfg, threshold=None, t_max=None):
    """
    Diagnostique chaque famille sur la fenêtre [0, t_max] complète.
    Prend le dictionnaire de config 'cfg' en argument.
    """
    model.eval()
    
    # 1. Gestion des valeurs par défaut depuis cfg
    # 1. Gestion des valeurs par défaut depuis cfg
    # IMPORTANT : On définit t_max D'ABORD pour savoir quel seuil appliquer
    if t_max is None: t_max = cfg['geometry']['T_max']
    
    if threshold is None:
        # Logique de sélection dynamique
        if t_max == 0.0:
            # Cas Condition Initiale : On cherche 'threshold_ic', sinon on fallback
            threshold = cfg['training'].get('threshold_ic', cfg['training'].get('threshold', 0.01))
        else:
            # Cas Propagation : On cherche 'threshold_step', sinon on fallback
            threshold = cfg['training'].get('threshold_step', cfg['training'].get('threshold', 0.05))
    
    Nx = cfg['audit']['Nx_audit']
    Nt = cfg['audit']['Nt_audit']
    x_min = cfg['geometry']['x_min']
    x_max = cfg['geometry']['x_max']
    
    # 2. Définition des familles
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
            # a. Paramètres aléatoires (dictionnaire)
            p_dict = {k: np.random.uniform(v[0], v[1]) for k, v in cfg['physics_ranges'].items()}
            
            # On force le type pour tester la famille spécifique
            current_id = np.random.choice(type_ids)
            p_dict['type'] = current_id 
            
            # b. Vérité Terrain ROBUSTE
            try:
                # Appel du solveur avec le dictionnaire cfg
                X_grid, T_grid, U_true_np = get_ground_truth_CN(
                    p_dict, cfg, t_step_max=t_max
                )
            except Exception as e:
                # print(f"⚠️ Erreur solveur : {e}") 
                continue

            # c. Aplatissement (Flatten)
            X_flat = X_grid.flatten()
            T_flat = T_grid.flatten()
            U_true = U_true_np.flatten()
            
            # d. Préparation Input DeepONet
            xt_tensor = torch.tensor(np.stack([X_flat, T_flat], axis=1), dtype=torch.float32).to(device)
            
            p_vec = np.array([p_dict['v'], p_dict['D'], p_dict['mu'], p_dict['type'], 
                              p_dict['A'], 0.0, p_dict['sigma'], p_dict['k']])
            # On répète les paramètres pour chaque point de la grille
            p_tensor = torch.tensor(p_vec, dtype=torch.float32).repeat(len(X_flat), 1).to(device)

            # e. Prédiction & Erreur
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