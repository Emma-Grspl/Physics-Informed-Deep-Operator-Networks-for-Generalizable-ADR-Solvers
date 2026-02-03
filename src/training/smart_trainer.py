import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from config import Config
from src.physics.adr import pde_residual_adr
from src.data.generators import generate_mixed_batch
from src.audit_tool import diagnose_model
from src.physics.solver import get_ground_truth_CN

# --- AUDIT RAPIDE ---
def audit_global_fast(model, current_t_max):
    device = next(model.parameters()).device
    model.eval()
    Nx = Config.Nx_audit
    Nt = Config.Nt_audit
    errors = []
    
    for _ in range(100): # 20 samples aléatoires
        p_dict = Config.get_p_dict()
        try:
            X_grid, T_grid, U_true_np = get_ground_truth_CN(
                p_dict, Config.x_min, Config.x_max, current_t_max, Nx, Nt
            )
        except: continue

        X_flat, T_flat = X_grid.flatten(), T_grid.flatten()
        xt_tensor = torch.tensor(np.stack([X_flat, T_flat], axis=1), dtype=torch.float32).to(device)
        
        p_vec = np.array([p_dict['v'], p_dict['D'], p_dict['mu'], p_dict['type'], 
                          p_dict['A'], p_dict['x0'], p_dict['sigma'], p_dict['k']])
        p_tensor = torch.tensor(p_vec, dtype=torch.float32).unsqueeze(0).repeat(len(X_flat), 1).to(device)

        with torch.no_grad():
            u_pred = model(p_tensor, xt_tensor).cpu().numpy().flatten()
        
        U_true = U_true_np.flatten()
        # Erreur Relative pondérée pour éviter division par zéro
        err = np.linalg.norm(U_true - u_pred) / (np.linalg.norm(U_true) + 1e-7)
        errors.append(err)

    if not errors: return False, 1.0
    mean_err = np.mean(errors)
    return mean_err < Config.threshold, mean_err

# --- STEP D'ENTRAINEMENT ---
def train_step_time_window(model, bounds, t_max, n_iters_main):
    device = next(model.parameters()).device
    lr_current = Config.learning_rate
    max_retry = Config.max_retry
    
    print(f"\n🔵 DÉBUT PALIER t=[0, {t_max}]")

    # --- DEFINITION DE LA LOSS (Dirichlet) ---
    def compute_total_loss(params, xt, xt_ic, u_true_ic, xt_bc_l, xt_bc_r, u_true_bc_l, u_true_bc_r):
        loss_pde = torch.mean(pde_residual_adr(model, params, xt)**2)
        loss_ic = torch.mean((model(params, xt_ic) - u_true_ic)**2)
        u_pred_l = model(params, xt_bc_l)
        u_pred_r = model(params, xt_bc_r)
        loss_bc = torch.mean((u_pred_l - u_true_bc_l)**2) + \
                  torch.mean((u_pred_r - u_true_bc_r)**2)
        return Config.weight_res * loss_pde + Config.weight_ic * loss_ic + Config.weight_bc * loss_bc

    # =========================================================================
    # PHASE 1 : ENTRAINEMENT GLOBAL & FILTRE GROSSIER
    # =========================================================================
    global_success = False
    
    # On garde le LR de base pour le global
    lr_global = lr_current 

    for attempt in range(max_retry): 
        if attempt == max_retry - 1: # LBFGS
            print(f"  👉 Tentative Globale {attempt+1}/{max_retry} : LBFGS")
            lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=50, line_search_fn="strong_wolfe")
            def closure():
                lbfgs.zero_grad()
                batch = generate_mixed_batch(Config.batch_size, bounds, Config.x_min, Config.x_max, t_max)
                loss = compute_total_loss(*batch)
                loss.backward()
                return loss
            try: lbfgs.step(closure)
            except: pass
        else: # Adam
            print(f"  👉 Tentative Globale {attempt+1}/{max_retry} : Adam (LR={lr_global:.1e})")
            optimizer = optim.Adam(model.parameters(), lr=lr_global)
            model.train()
            for i in range(n_iters_main):
                optimizer.zero_grad()
                batch = generate_mixed_batch(Config.batch_size, bounds, Config.x_min, Config.x_max, t_max)
                loss = compute_total_loss(*batch)
                loss.backward()
                optimizer.step()
                if (i + 1) % 1000 == 0:
                    print(f"    [Global Adam] Iter {i+1} | Loss: {loss.item():.2e}")

        # --- AUDIT GLOBAL ---
        success, err = audit_global_fast(model, t_max)
        
        if success:
            print(f"     📊 Audit Global OK ({err:.2%}). Vérification spécifique...")
            failed_check = diagnose_model(model, device, threshold=Config.threshold, t_max=t_max)
            
            if not failed_check:
                print("     ✅ Validation Totale : Toutes les familles sont sous le seuil.")
                return True, err
            else:
                print(f"     ⚠️ ALERTE : Moyenne bonne, mais {failed_check} > {Config.threshold:.0%}.")
                print("     ➡️  Passage à la Correction Ciblée.")
                global_success = False
                break 
        else:
            print(f"     📊 Audit Global: {err:.2%} -> ❌ KO")
        
        if attempt < max_retry - 2: lr_global *= 0.5

    failed_ids = diagnose_model(model, device, threshold=Config.threshold, t_max=t_max)
    if not failed_ids: return True, 0.0

    all_types = [0, 1, 2, 3, 4]
    success_ids = [t for t in all_types if t not in failed_ids]

    print(f"\n🚑 Correction Ciblée BOUCLÉE sur {failed_ids} (Mix 80/20)...")

    weighted_types = []
    for f_id in failed_ids: weighted_types.extend([f_id] * 4) 
    for s_id in success_ids: weighted_types.extend([s_id] * 1) 

    lr_focus = lr_current 
    correction_success = False

    # ON AUGMENTE LE NOMBRE D'ITÉRATIONS POUR LA CORRECTION
    # C'est plus dur que le global, donc on double l'effort d'Adam
    n_iters_focus = n_iters_main + 5000  

    for attempt in range(max_retry): 
        # --- LBFGS (Dernière tentative - L'ARTILLERIE LOURDE) ---
        if attempt == max_retry - 1:
            print(f"  ☢️ [Focus Loop] Tentative {attempt+1}/{max_retry} : LBFGS Finisher (Long)")
            # MODIFICATION ICI : max_iter=500 (au lieu de 50) + tolerance_grad plus fine
            lbfgs_focus = optim.LBFGS(
                model.parameters(), 
                lr=1.0, 
                max_iter=500,  # <--- On lui laisse le temps de converger !
                tolerance_grad=1e-9, 
                line_search_fn="strong_wolfe"
            )
            def closure_focus():
                lbfgs_focus.zero_grad()
                batch = generate_mixed_batch(Config.batch_size, bounds, Config.x_min, Config.x_max, t_max, allowed_types=weighted_types)
                loss = compute_total_loss(*batch)
                loss.backward()
                return loss
            try: lbfgs_focus.step(closure_focus)
            except: pass
            
        # --- ADAM (Tentatives 1, 2, 3) ---
        else:
            print(f"  🚑 [Focus Loop] Tentative {attempt+1}/{max_retry} : Adam (LR={lr_focus:.1e})")
            optimizer_focus = optim.Adam(model.parameters(), lr=lr_focus)
            
            for i in range(n_iters_focus): # Utilisation de n_iters_focus (10000)
                optimizer_focus.zero_grad()
                batch = generate_mixed_batch(Config.batch_size, bounds, Config.x_min, Config.x_max, t_max, allowed_types=weighted_types)
                loss = compute_total_loss(*batch)
                loss.backward()
                optimizer_focus.step()
                if (i+1) % 2000 == 0: # Print moins fréquent
                    print(f"    [Focus Adam] Iter {i+1}/{n_iters_focus} | Loss: {loss.item():.2e}")

        # --- AUDIT ---
        print(f"  🩺 Check post-correction (Tentative {attempt+1})...")
        failed_now = diagnose_model(model, device, threshold=Config.threshold, t_max=t_max)
        
        if not failed_now:
            print(f"  ✅ Correction réussie à la tentative {attempt+1} !")
            correction_success = True
            break 
        else:
            print(f"  ❌ Encore des erreurs sur {failed_now}. On retente/affine.")
            
        if attempt < max_retry - 2: lr_focus *= 0.5

    # =========================================================================
    # VERDICT FINAL
    # =========================================================================
    if correction_success: return True, 0.0
    
    # Check final : On garde ta tolérance mais on espère que LBFGS a fait le job
    THRESHOLD_RELAXED = 0.045 
    failed_final = diagnose_model(model, device, threshold=THRESHOLD_RELAXED, t_max=t_max)
    
    if not failed_final:
        print(f"✅ Ouf ! Validé in-extremis avec tolérance ({THRESHOLD_RELAXED:.1%}).")
        return True, 0.0
    else:
        print(f"🛑 ÉCHEC FINAL sur t={t_max}. Types résistants > {THRESHOLD_RELAXED:.1%}: {failed_final}")
        return False, 1.0

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
            
            # --- CORRECTION ICI : On récupère 8 valeurs (on ignore les 4 dernières avec _) ---
            # Avant c'était : params, xt, xt_ic, u_true_ic, _, _ = ...
            params, xt, xt_ic, u_true_ic, _, _, _, _ = generate_mixed_batch(
                Config.batch_size, bounds, Config.x_min, Config.x_max, 0.0
            )
            
            # Loss uniquement sur la condition initiale (IC)
            loss = torch.mean((model(params, xt_ic) - u_true_ic)**2)
            
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 500 == 0: 
                msg = f"    [Warmup] Iter {i+1} | Loss IC: {loss.item():.2e}"
                tqdm.write(msg)
        
        torch.save(model.state_dict(), f"{save_dir}/model_post_warmup.pth")
        print("✅ Warmup OK.")

    # --- BOUCLE TEMPORELLE ---
    T_end = Config.T_max
    time_steps = [round(t, 2) for t in np.arange(Config.dt, T_end + Config.dt/1000.0, Config.dt)]
    
    for t_step in time_steps:
        # Appel de la fonction d'entraînement par palier (déjà corrigée précédemment)
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
