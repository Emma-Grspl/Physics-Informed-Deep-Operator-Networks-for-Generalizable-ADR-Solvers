import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import sys

# Import Config
from config import Config

# Imports du projet
from src.physics.adr import pde_residual_adr
from src.data.generators import generate_mixed_batch
from src.audit_tool import diagnose_model
from src.physics.solver import get_ground_truth_CN

# --- OUTILS INTERNES ---

def audit_global_fast(model, current_t_max):
    """
    Audit rapide sur tout le monde pour la phase globale.
    Retourne (bool: success, float: mean_error).
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Paramètres réduits pour aller vite
    Nx = Config.Nx_audit
    Nt = Config.Nt_audit
    n_samples = 20 # Suffisant pour une moyenne globale
    
    errors = []
    
    for _ in range(n_samples):
        p_dict = Config.get_p_dict()
        try:
            X_grid, T_grid, U_true_np = get_ground_truth_CN(p_dict, Config.x_min, Config.x_max, current_t_max, Nx, Nt)
        except: continue

        X_flat = X_grid.flatten()
        T_flat = T_grid.flatten()
        xt_tensor = torch.tensor(np.stack([X_flat, T_flat], axis=1), dtype=torch.float32).to(device)
        
        p_vec = np.array([p_dict['v'], p_dict['D'], p_dict['mu'], p_dict['type'], 
                          p_dict['A'], p_dict['x0'], p_dict['sigma'], p_dict['k']])
        p_tensor = torch.tensor(p_vec, dtype=torch.float32).unsqueeze(0).repeat(len(X_flat), 1).to(device)

        with torch.no_grad():
            u_pred = model(p_tensor, xt_tensor).cpu().numpy().flatten()
        
        U_true = U_true_np.flatten()
        err = np.linalg.norm(U_true - u_pred) / (np.linalg.norm(U_true) + 1e-6)
        errors.append(err)

    if not errors: return False, 1.0
    
    # --- CORRECTION ICI ---
    mean_err = np.mean(errors)
    return mean_err < Config.threshold, mean_err # J'avais mis 'mean_error' ici par erreur
# --- FONCTION D'ENTRAINEMENT PAR PALIER ---

def train_step_time_window(model, bounds, t_max, n_iters_main):
    device = next(model.parameters()).device
    lr_current = Config.learning_rate
    
    print(f"\n🔵 DÉBUT PALIER t=[0, {t_max}]")

    # =========================================================================
    # ÉTAPE 1 : ENTRAINEMENT GLOBAL (Adam x3 -> LBFGS)
    # =========================================================================
    global_success = False
    
    for attempt in range(4): # 0, 1, 2 (Adam), 3 (LBFGS)
        
        # --- CAS LBFGS (Dernière chance globale) ---
        if attempt == 3:
            print(f"  👉 Tentative Globale {attempt+1}/4 : LBFGS (Tout le monde)")
            lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=50, line_search_fn="strong_wolfe")
            
            def closure():
                lbfgs.zero_grad()
                params, xt, xt_ic, u_true_ic, xt_bc_l, xt_bc_r = generate_mixed_batch(
                    Config.batch_size, bounds, Config.x_min, Config.x_max, t_max
                )
                loss_pde = torch.mean(pde_residual_adr(model, params, xt)**2)
                loss_ic = torch.mean((model(params, xt_ic) - u_true_ic)**2)
                loss_bc = torch.mean((model(params, xt_bc_l) - model(params, xt_bc_r))**2)
                loss = Config.weight_res * loss_pde + Config.weight_ic * loss_ic + Config.weight_bc * loss_bc
                loss.backward()
                return loss
            
            try: lbfgs.step(closure)
            except: pass
            
        # --- CAS ADAM (Tentatives 1, 2, 3) ---
        else:
            print(f"  👉 Tentative Globale {attempt+1}/4 : Adam (LR={lr_current:.1e})")
            optimizer = optim.Adam(model.parameters(), lr=lr_current)
            model.train()
            
            # Boucle d'entrainement standard
            pbar = tqdm(range(n_iters_main), desc=f"Global Adam #{attempt+1}", leave=False)
            for i in pbar:
                optimizer.zero_grad()
                params, xt, xt_ic, u_true_ic, xt_bc_l, xt_bc_r = generate_mixed_batch(
                    Config.batch_size, bounds, Config.x_min, Config.x_max, t_max
                )
                loss_pde = torch.mean(pde_residual_adr(model, params, xt)**2)
                loss_ic = torch.mean((model(params, xt_ic) - u_true_ic)**2)
                loss_bc = torch.mean((model(params, xt_bc_l) - model(params, xt_bc_r))**2)
                loss = Config.weight_res * loss_pde + Config.weight_ic * loss_ic + Config.weight_bc * loss_bc
                
                loss.backward()
                optimizer.step()
                
                # --- MODIFICATION ICI : Print tous les 500 itérations ---
                if (i + 1) % 500 == 0:
                    msg = f"    [Global Adam] Iter {i+1}/{n_iters_main} | Loss: {loss.item():.2e}"
                    tqdm.write(msg) 

        # --- AUDIT GLOBAL ---
        success, err = audit_global_fast(model, t_max)
        print(f"     📊 Audit Global: {err:.2%} (Seuil: {Config.threshold:.0%}) -> {'✅ OK' if success else '❌ KO'}")
        
        if success:
            global_success = True
            break
        
        # Si échec, on baisse le LR pour le prochain tour Adam
        if attempt < 2:
            lr_current *= 0.5 

    if not global_success:
        print("  ⚠️ Échec de la phase globale. On passe à l'audit spécifique.")

    # =========================================================================
    # ÉTAPE 2 : AUDIT SPÉCIFIQUE & CORRECTION CIBLÉE
    # =========================================================================
    print(f"\n🩺 Audit Spécifique par Famille...")
    
    # On récupère les IDs qui posent problème (ex: [1, 2] pour Sin-Gauss)
    failed_ids = diagnose_model(model, device, threshold=Config.threshold)
    
    if not failed_ids:
        print("✅ Toutes les familles sont valides ! Passage au temps suivant.")
        return True, err

    print(f"❌ Familles en échec : IDs {failed_ids}")
    print("🚑 Lancement de la CORRECTION CIBLÉE...")

    # --- CORRECTION 1 : Adam + LBFGS sur failed_ids avec le LR actuel ---
    print(f"  🔧 Correction #1 (LR={lr_current:.1e}) sur {failed_ids}")
    
    # 1. Adam Ciblé
    optimizer_focus = optim.Adam(model.parameters(), lr=lr_current)
    model.train()
    for i in range(2000): # 2000 iters suffisent souvent pour corriger
        optimizer_focus.zero_grad()
        # On ne génère QUE les types malades
        params, xt, xt_ic, u_true_ic, xt_bc_l, xt_bc_r = generate_mixed_batch(
            Config.batch_size, bounds, Config.x_min, Config.x_max, t_max, allowed_types=failed_ids
        )
        loss = Config.weight_res * torch.mean(pde_residual_adr(model, params, xt)**2) + \
               Config.weight_ic * torch.mean((model(params, xt_ic) - u_true_ic)**2) + \
               Config.weight_bc * torch.mean((model(params, xt_bc_l) - model(params, xt_bc_r))**2)
        loss.backward()
        optimizer_focus.step()
        
        # --- MODIFICATION ICI : Print tous les 500 itérations ---
        if (i + 1) % 500 == 0: 
            print(f"    [Focus Adam #1] Iter {i+1} | Loss: {loss.item():.2e}")

    # 2. LBFGS Ciblé
    print(f"    [Focus LBFGS]...")
    lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, line_search_fn="strong_wolfe")
    def closure_focus():
        lbfgs.zero_grad()
        params, xt, xt_ic, u_true_ic, xt_bc_l, xt_bc_r = generate_mixed_batch(
            Config.batch_size, bounds, Config.x_min, Config.x_max, t_max, allowed_types=failed_ids
        )
        loss = Config.weight_res * torch.mean(pde_residual_adr(model, params, xt)**2) + \
               Config.weight_ic * torch.mean((model(params, xt_ic) - u_true_ic)**2) + \
               Config.weight_bc * torch.mean((model(params, xt_bc_l) - model(params, xt_bc_r))**2)
        loss.backward()
        return loss
    try: lbfgs.step(closure_focus)
    except: pass

    # --- RE-AUDIT SPÉCIFIQUE ---
    failed_ids_2 = diagnose_model(model, device, threshold=Config.threshold)
    if not failed_ids_2:
        print("✅ Correction #1 réussie ! Tout est rentré dans l'ordre.")
        return True, 0.0

    # --- CORRECTION 2 (Ultime) : LR / 2 + Adam + LBFGS ---
    lr_last_resort = lr_current * 0.5
    print(f"⚠️ Correction #1 échouée. Tentative Ultime avec LR={lr_last_resort:.1e} sur {failed_ids_2}...")
    
    optimizer_focus = optim.Adam(model.parameters(), lr=lr_last_resort)
    for i in range(2000):
        optimizer_focus.zero_grad()
        params, xt, xt_ic, u_true_ic, xt_bc_l, xt_bc_r = generate_mixed_batch(
            Config.batch_size, bounds, Config.x_min, Config.x_max, t_max, allowed_types=failed_ids_2
        )
        loss = Config.weight_res * torch.mean(pde_residual_adr(model, params, xt)**2) + \
               Config.weight_ic * torch.mean((model(params, xt_ic) - u_true_ic)**2) + \
               Config.weight_bc * torch.mean((model(params, xt_bc_l) - model(params, xt_bc_r))**2)
        loss.backward()
        optimizer_focus.step()
        
        # --- MODIFICATION ICI : Print tous les 500 itérations ---
        if (i + 1) % 500 == 0: 
            print(f"    [Ultime Adam] Iter {i+1} | Loss: {loss.item():.2e}")

    print(f"    [Ultime LBFGS]...")
    try: lbfgs.step(closure_focus) # On réutilise la closure ciblée
    except: pass

    # --- VERDICT FINAL ---
    failed_ids_final = diagnose_model(model, device, threshold=Config.threshold)
    if not failed_ids_final:
        print("✅ Correction Ultime réussie ! Ouf !")
        return True, 0.0
    else:
        print(f"❌ ÉCHEC DÉFINITIF sur le palier {t_max}. Les types {failed_ids_final} résistent.")
        return False, 1.0


# --- ORCHESTRATEUR PRINCIPAL ---

def train_smart_time_marching(model, bounds, n_warmup, n_iters_per_step):
    save_dir = Config.save_dir
    print(f"⚡ DÉMARRAGE TRAINING (Protocole Strict & Verbose)")
    print(f"    -> Warmup (t=0): {n_warmup} iters")
    print(f"    -> Time Step: {Config.dt}, Max T: {Config.T_max}")

    # --- WARMUP (t=0) ---
    if n_warmup > 0:
        print("\n🧊 PHASE 0 : WARMUP (Condition Initiale)...")
        optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
        model.train()
        pbar = tqdm(range(n_warmup))
        for i in pbar:
            optimizer.zero_grad()
            params, xt, xt_ic, u_true_ic, _, _ = generate_mixed_batch(
                Config.batch_size, bounds, Config.x_min, Config.x_max, 0.0
            )
            loss = torch.mean((model(params, xt_ic) - u_true_ic)**2)
            loss.backward()
            optimizer.step()
            
            # --- MODIFICATION ICI : Print tous les 500 itérations ---
            if (i + 1) % 500 == 0: 
                msg = f"    [Warmup] Iter {i+1} | Loss IC: {loss.item():.2e}"
                tqdm.write(msg)
        
        torch.save(model.state_dict(), f"{save_dir}/model_post_warmup.pth")
        print("✅ Warmup OK.")

    # --- BOUCLE TEMPORELLE ---
    T_end = Config.T_max
    time_steps = [round(t, 2) for t in np.arange(Config.dt, T_end + Config.dt/1000.0, Config.dt)]
    
    for t_step in time_steps:
        success, _ = train_step_time_window(model, bounds, t_max=t_step, n_iters_main=n_iters_per_step)
        
        if success:
            torch.save({
                't_max': t_step,
                'model_state_dict': model.state_dict()
            }, f"{save_dir}/model_checkpoint_t{t_step}.pth")
        else:
            print("🛑 Arrêt d'urgence : Le modèle ne valide pas le palier.")
            break

    print("\n🏁 Fin du programme.")
    return model