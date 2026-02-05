import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# On importe les outils existants
from config_reprise import Config
from src.physics.adr import pde_residual_adr
from src.data.generators import generate_mixed_batch
from src.audit_tool import diagnose_model
from src.physics.solver import get_ground_truth_CN

# =============================================================================
# 1. OUTILS D'AUDIT
# =============================================================================

def audit_global_fast(model, current_t_max):
    """Évaluation rapide sur 100 cas aléatoires pour valider le palier."""
    device = next(model.parameters()).device
    model.eval()
    Nx, Nt = Config.Nx_audit, Config.Nt_audit
    errors = []
    
    for _ in range(100): 
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
        err = np.linalg.norm(U_true - u_pred) / (np.linalg.norm(U_true) + 1e-7)
        errors.append(err)

    if not errors: return False, 1.0
    mean_err = np.mean(errors)
    return mean_err < Config.threshold, mean_err


# =============================================================================
# 2. LOGIQUE D'ENTRAÎNEMENT PAR PALIER
# =============================================================================

def train_step_reprise(model, bounds, t_max, n_iters_main):
    device = next(model.parameters()).device
    lr_current = Config.learning_rate
    
    print(f"\n🔵 [REPRISE] PALIER t=[0, {t_max}]")

    # --- POIDS DYNAMIQUES MUSCLÉS ---
    # Pour la reprise, on applique directement les poids forts de Config_reprise
    w_ic_curr = Config.weight_ic
    w_res_curr = Config.weight_res
    w_bc_curr = Config.weight_bc

    print(f"   ⚖️  Poids Reprise : IC={w_ic_curr:.1f} | PDE={w_res_curr:.1f} | BC={w_bc_curr:.1f}")
    print(f"   📍 Densité de points : {Config.n_sample}")

    def compute_total_loss(params, xt, xt_ic, u_true_ic, xt_bc_l, xt_bc_r, u_true_bc_l, u_true_bc_r):
        loss_pde = torch.mean(pde_residual_adr(model, params, xt)**2)
        loss_ic = torch.mean((model(params, xt_ic) - u_true_ic)**2)
        u_pred_l, u_pred_r = model(params, xt_bc_l), model(params, xt_bc_r)
        loss_bc = torch.mean((u_pred_l - u_true_bc_l)**2) + torch.mean((u_pred_r - u_true_bc_r)**2)
        return w_res_curr * loss_pde + w_ic_curr * loss_ic + w_bc_curr * loss_bc

    # --- PHASE 1 : ADAM ---
    for attempt in range(Config.max_retry):
        if attempt == Config.max_retry - 1: # L-BFGS Finisher
            print(f"   👉 Tentative Globale {attempt+1}: L-BFGS (50 iters)")
            optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=50, line_search_fn="strong_wolfe")
            def closure():
                optimizer.zero_grad()
                # Correction : Utilisation de Config.n_sample comme premier argument
                batch = generate_mixed_batch(Config.n_sample, bounds, Config.x_min, Config.x_max, t_max)
                loss = compute_total_loss(*batch)
                loss.backward()
                return loss
            try: optimizer.step(closure)
            except: pass
        else: # Adam
            print(f"   👉 Tentative Globale {attempt+1}: Adam (LR={lr_current:.1e})")
            optimizer = optim.Adam(model.parameters(), lr=lr_current)
            for i in range(n_iters_main):
                optimizer.zero_grad()
                # Correction : Utilisation de Config.n_sample comme premier argument
                batch = generate_mixed_batch(Config.n_sample, bounds, Config.x_min, Config.x_max, t_max)
                loss = compute_total_loss(*batch)
                loss.backward()
                optimizer.step()
                if (i+1) % 2000 == 0: print(f"      Iter {i+1} | Loss: {loss.item():.2e}")
            lr_current *= 0.5

        # Audit
        success, err = audit_global_fast(model, t_max)
        if success: break
        print(f"      📊 Audit Global: {err:.2%} -> KO")

    # --- PHASE 2 : CORRECTION CIBLÉE (Focus Sin-Gauss) ---
    failed_ids = diagnose_model(model, device, threshold=Config.threshold, t_max=t_max)
    if not failed_ids: return True, 0.0

    print(f"\n🚑 Correction Ciblée sur {failed_ids} (Mix 80/20)...")
    weighted_types = []
    for f_id in failed_ids: weighted_types.extend([f_id] * 4)
    for s_id in [i for i in range(5) if i not in failed_ids]: weighted_types.extend([s_id] * 1)

    lr_focus = Config.learning_rate * 0.5
    for attempt in range(Config.max_retry):
        if attempt == Config.max_retry - 1: # Super L-BFGS
            print(f"   ☢️ [Focus] Tentative Finale : L-BFGS (500 iters)")
            optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=500, tolerance_grad=1e-9, line_search_fn="strong_wolfe")
            def closure_f():
                optimizer.zero_grad()
                # Correction : Utilisation de Config.n_sample et allowed_types
                batch = generate_mixed_batch(Config.n_sample, bounds, Config.x_min, Config.x_max, t_max, allowed_types=weighted_types)
                loss = compute_total_loss(*batch)
                loss.backward()
                return loss
            try: optimizer.step(closure_f)
            except: pass
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr_focus)
            for i in range(n_iters_main + 2000):
                optimizer.zero_grad()
                # Correction : Utilisation de Config.n_sample et allowed_types
                batch = generate_mixed_batch(Config.n_sample, bounds, Config.x_min, Config.x_max, t_max, allowed_types=weighted_types)
                loss = compute_total_loss(*batch)
                loss.backward()
                optimizer.step()
            lr_focus *= 0.5

        failed_now = diagnose_model(model, device, threshold=Config.threshold, t_max=t_max)
        if not failed_now: return True, 0.0
        print(f"      ❌ Échec correction sur {failed_now}")

    return False, 1.0

# =============================================================================
# 3. BOUCLE DE REPRISE
# =============================================================================

def train_smart_time_marching_reprise(model, bounds, start_t, n_iters_per_step):
    save_dir = Config.save_dir
    print(f"\n⚡ DÉMARRAGE REPRISE (t_start={start_t})")
    
    all_steps = [round(t, 2) for t in np.arange(Config.dt, Config.T_max + 0.001, Config.dt)]
    time_steps = [t for t in all_steps if t > start_t + 0.01]
    
    print(f"📅 Paliers restants à franchir : {time_steps}")

    for t_step in time_steps:
        success, _ = train_step_reprise(model, bounds, t_max=t_step, n_iters_main=n_iters_per_step)
        
        if success:
            path = f"{save_dir}/model_checkpoint_t{t_step}.pth"
            torch.save({'t_max': t_step, 'model_state_dict': model.state_dict()}, path)
            print(f"✅ Palier t={t_step} validé et sauvegardé.")
        else:
            print(f"🛑 Échec critique au palier t={t_step}.")
            return model

    print("\n🏁 REPRISE TERMINÉE. Fin du voyage.")
    return model