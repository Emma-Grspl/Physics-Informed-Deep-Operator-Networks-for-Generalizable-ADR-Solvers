import torch
import torch.optim as optim
import numpy as np
import copy
import yaml
import os
import glob
import re
from tqdm import tqdm

# Imports
from src.physics.adr import pde_residual_adr
from src.data.generators import generate_mixed_batch
from src.audit_tool import diagnose_model
from src.physics.solver import get_ground_truth_CN

# =============================================================================
# 1. CONFIG & UTILS
# =============================================================================

def load_config(path="src/config_ADR.yaml"):
    if not os.path.exists(path): path = "src/config_ADR.yaml"
    with open(path, 'r') as f: return yaml.safe_load(f)

cfg = load_config()

def generate_time_steps():
    steps, current_t, t_limit = [], 0.0, cfg['geometry']['T_max']
    for zone in cfg['time_stepping']['zones']:
        t_end = t_limit if zone['t_end'] == -1 else zone['t_end']
        dt = zone['dt']
        while current_t < t_end - 1e-5:
            current_t = round(current_t + dt, 3)
            if current_t > t_limit: break
            steps.append(current_t)
    return steps

def find_latest_checkpoint(save_dir):
    import os, glob, re
    
    # Force le chemin en absolu pour éviter les pièges de dossier courant
    abs_save_dir = os.path.abspath(save_dir)
    pattern = os.path.join(abs_save_dir, "model_checkpoint_t*.pth")
    
    print(f"\n🔍 [DEBUG REPRISE]")
    print(f"   📂 Dossier cible : {abs_save_dir}")
    print(f"   🔍 Pattern glob  : {pattern}")
    
    files = glob.glob(pattern)
    print(f"   📄 Fichiers trouvés : {len(files)}")
    if len(files) > 0:
        print(f"   📋 Liste : {[os.path.basename(f) for f in files[:3]]}...")

    if not files:
        return None, 0.0
    
    max_t = -1.0
    best_file = None
    
    for f in files:
        match = re.search(r"model_checkpoint_t([\d\.]+)\.pth", f)
        if match:
            t_val = float(match.group(1))
            if t_val > max_t:
                max_t = t_val
                best_file = f
                
    print(f"   🏆 Meilleur candidat : {best_file} (t={max_t})")
    return best_file, max_t

def get_loss(model, batch, wr, wi, wb):
    params, xt, xt_ic, u_true_ic, xt_bc_l, xt_bc_r, u_true_bc_l, u_true_bc_r = batch
    if not xt.requires_grad: xt.requires_grad_(True)
    l_pde = torch.mean(pde_residual_adr(model, params, xt)**2)
    l_ic = torch.mean((model(params, xt_ic) - u_true_ic)**2)
    l_bc = torch.mean((model(params, xt_bc_l) - u_true_bc_l)**2) + \
           torch.mean((model(params, xt_bc_r) - u_true_bc_r)**2)
    return wr * l_pde + wi * l_ic + wb * l_bc

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def compute_ntk_weights(model, batch, w_ic_ref):
    model.zero_grad()
    params, xt, xt_ic, u_true_ic, _, _, _, _ = batch
    if not xt.requires_grad: xt.requires_grad_(True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    loss_pde = torch.mean(pde_residual_adr(model, params, xt)**2)
    grad_pde = torch.autograd.grad(loss_pde, trainable_params, retain_graph=True, create_graph=False, allow_unused=True)
    norm_pde = torch.sqrt(sum(g.pow(2).sum() for g in grad_pde if g is not None))

    loss_ic = torch.mean((model(params, xt_ic) - u_true_ic)**2)
    grad_ic = torch.autograd.grad(loss_ic, trainable_params, retain_graph=True, create_graph=False, allow_unused=True)
    norm_ic = torch.sqrt(sum(g.pow(2).sum() for g in grad_ic if g is not None))

    new_w_pde = (norm_ic / (norm_pde + 1e-8)).item() * w_ic_ref
    return min(max(new_w_pde, 10.0), 500.0)

def monitor_gradients(model, batch):
    try:
        model.zero_grad()
        params, xt, xt_ic, u_true_ic, _, _, _, _ = batch
        if not xt.requires_grad: xt.requires_grad_(True)
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        lp = torch.mean(pde_residual_adr(model, params, xt)**2)
        gp = torch.autograd.grad(lp, trainable_params, retain_graph=True, allow_unused=True)
        fp = torch.cat([g.view(-1) for g in gp if g is not None])
        
        li = torch.mean((model(params, xt_ic) - u_true_ic)**2)
        gi = torch.autograd.grad(li, trainable_params, retain_graph=True, allow_unused=True)
        fi = torch.cat([g.view(-1) for g in gi if g is not None])

        ratio = (torch.norm(fi) / (torch.norm(fp) + 1e-8)).item()
        cos_sim = torch.nn.functional.cosine_similarity(fp, fi, dim=0).item()
        return ratio, cos_sim
    except: return 1.0, 0.0

class KingOfTheHill:
    def __init__(self, model):
        self.best_state = copy.deepcopy(model.state_dict())
        self.best_loss = float('inf')
        self.history = []

    def update(self, model, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_state = copy.deepcopy(model.state_dict())
            #print(f"      🏆 Nouveau Champion (Loss: {self.best_loss:.2e})")
            return True
        return False

# =============================================================================
# 3. AUDIT GLOBAL & STRUCTURE MIROIR
# =============================================================================

def audit_global_fast(model, t_max):
    device = next(model.parameters()).device
    model.eval()
    np.random.seed(42) 
    errors = []
    
    # --- CHOIX DU SEUIL DYNAMIQUE ---
    # Si t=0 (Warmup/IC), on prend le seuil strict. Sinon, le seuil de propagation.
    if t_max == 0.0:
        target_threshold = cfg['training'].get('threshold_ic', cfg['training'].get('threshold', 0.01))
        mode_str = "IC (Strict)"
    else:
        target_threshold = cfg['training'].get('threshold_step', cfg['training'].get('threshold', 0.05))
        mode_str = "Step (Relaxed)"
    # --------------------------------

    for _ in range(200):
        p_dict = {k: np.random.uniform(v[0], v[1]) for k, v in cfg['physics_ranges'].items()}
        p_dict['type'] = np.random.randint(0, 5)
        try:
            X_grid, T_grid, U_true = get_ground_truth_CN(p_dict, cfg, t_step_max=t_max)
            x_flat, t_flat = X_grid.flatten(), T_grid.flatten()
            p_vec = np.array([p_dict['v'], p_dict['D'], p_dict['mu'], p_dict['type'], 
                              p_dict['A'], 0.0, p_dict['sigma'], p_dict['k']])
            p_tensor = torch.tensor(p_vec, dtype=torch.float32).repeat(len(x_flat), 1).to(device)
            xt_tensor = torch.tensor(np.stack([x_flat, t_flat], axis=1), dtype=torch.float32).to(device)
            with torch.no_grad(): u_pred_flat = model(p_tensor, xt_tensor).cpu().numpy().flatten()
            err = np.linalg.norm(U_true.flatten() - u_pred_flat) / (np.linalg.norm(U_true.flatten()) + 1e-8)
            errors.append(err)
        except: continue
    np.random.seed(None)

    if not errors: return False, 1.0
    avg_err = np.mean(errors)
    
    print(f"      [Audit {mode_str}] Avg Rel L2: {avg_err:.2%} (Target: < {target_threshold:.2%})")
    
    return avg_err < target_threshold, avg_err

def targeted_correction(model, bounds, t_max, failed_ids, n_iters_base, start_lr):
    print(f"\n🚑 CORRECTION STRUCTURÉE sur {failed_ids} (Start LR={start_lr:.2e})")
    device = next(model.parameters()).device
    king_corr = KingOfTheHill(model)
    
    all_types = [0, 1, 2, 3, 4]
    weighted_types = []
    for tid in all_types:
        if tid in failed_ids: weighted_types.extend([tid] * 4) 
        else: weighted_types.extend([tid] * 1) 

    if t_max == 0.0: w_res_loc, w_bc_loc, w_ic_loc = 0.0, 0.0, 100.0
    else: w_res_loc, w_bc_loc, w_ic_loc = 500.0, cfg['loss_weights']['weight_bc'], 100.0

    current_lr = start_lr
    correction_success = False

    for macro in range(cfg['training']['nb_loop']):
        print(f"    🚑 [Correction] Macro {macro+1}/{cfg['training']['nb_loop']}")
        for retry in range(cfg['training']['max_retry']):
            model.load_state_dict(king_corr.best_state)
            opt = optim.Adam(model.parameters(), lr=current_lr)
            
            pbar = tqdm(range(n_iters_base), desc=f"    Corr-Try {retry+1}", leave=False)
            for i in pbar:
                batch = generate_mixed_batch(cfg['training']['n_sample'], bounds, 
                                             cfg['geometry']['x_min'], cfg['geometry']['x_max'], 
                                             t_max, allowed_types=weighted_types)
                params, xt, xt_ic, u_true_ic, _, _, _, _ = batch
                if not xt.requires_grad: xt.requires_grad_(True)
                opt.zero_grad()
                loss = get_loss(model, batch, w_res_loc, w_ic_loc, w_bc_loc) 
                loss.backward()
                opt.step()
                king_corr.update(model, loss.item())
                if i % 500 == 0: pbar.set_postfix({"Loss": f"{loss.item():.2e}"})

            failed_now = diagnose_model(model, device, cfg, t_max=t_max)
            if len(failed_now) == 0:
                print("      🚀 SUCCESS CORRECTION (Adam).")
                correction_success = True
                break
            current_lr *= 0.5 
            
        if correction_success: break
        
        if king_corr.best_loss < 1e-2:
            print("    ☢️ L-BFGS Finisher (Correction)...")
            model.load_state_dict(king_corr.best_state)
            lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=500, line_search_fn="strong_wolfe")
            def closure():
                lbfgs.zero_grad()
                b = generate_mixed_batch(cfg['training']['n_sample'], bounds, cfg['geometry']['x_min'], cfg['geometry']['x_max'], t_max, allowed_types=weighted_types)
                p, xt_bfgs, xic, uic, bc_l, bc_r, ubc_l, ubc_r = b
                if not xt_bfgs.requires_grad: xt_bfgs.requires_grad_(True)
                b_safe = (p, xt_bfgs, xic, uic, bc_l, bc_r, ubc_l, ubc_r)
                loss = get_loss(model, b_safe, w_res_loc, w_ic_loc, w_bc_loc)
                loss.backward()
                return loss
            try: lbfgs.step(closure)
            except: pass
            
            failed_now = diagnose_model(model, device, cfg, t_max=t_max)
            if len(failed_now) == 0:
                print("      🚀 SUCCESS CORRECTION (L-BFGS).")
                correction_success = True
                break
    
    if not correction_success:
        print("🛑 ÉCHEC CORRECTION.")
        return False
    return True

def train_step_time_window(model, bounds, t_max, n_iters_main):
    device = next(model.parameters()).device
    king = KingOfTheHill(model)
    
    w_bc = cfg['loss_weights']['weight_bc']
    t_ramp_end = 0.3
    if t_max <= t_ramp_end:
        start_w_res = 0.1
        target_w_res = cfg['loss_weights']['first_w_res']
        ratio = t_max / t_ramp_end
        w_res = start_w_res + (target_w_res - start_w_res) * (ratio * ratio) 
        w_ic = cfg['loss_weights']['weight_ic_init'] 
        mode = f"RAMPE DOUCE (w_res={w_res:.2f})"
    else:
        w_res, w_ic = cfg['loss_weights']['first_w_res'], cfg['loss_weights']['weight_ic_final']
        mode = "NTK"

    print(f"\n🔵 PALIER t={t_max} | Mode: {mode}")
    current_lr = cfg['training']['learning_rate']
    global_success = False

    for macro in range(cfg['training']['nb_loop']):
        print(f"    🌀 Macro {macro+1}/{cfg['training']['nb_loop']}")
        
        for retry in range(cfg['training']['max_retry']):
            model.load_state_dict(king.best_state)
            optimizer = optim.Adam(model.parameters(), lr=current_lr)
            for i in range(n_iters_main):
                batch = generate_mixed_batch(cfg['training']['n_sample'], bounds, 
                                             cfg['geometry']['x_min'], cfg['geometry']['x_max'], t_max)
                params, xt, xt_ic, u_true_ic, _, _, _, _ = batch
                if not xt.requires_grad: xt.requires_grad_(True)
                if mode == "NTK" and i % 100 == 0: w_res = compute_ntk_weights(model, batch, w_ic)
                optimizer.zero_grad()
                loss = get_loss(model, batch, w_res, w_ic, w_bc)
                loss.backward()
                optimizer.step()
                king.update(model, loss.item())
                if i % 1000 == 0:
                    r, c = monitor_gradients(model, batch)
                    print(f"      It {i} | Loss: {loss.item():.2e} | ForceRatio: {r:.2f} | CosSim: {c:.2f}")

            success_adam, err = audit_global_fast(model, t_max)
            if success_adam:
                print("      🚀 SUCCESS GLOBAL (Adam).")
                global_success = True
                break
            current_lr *= 0.5
        
        if global_success: break

        print("    ☢️ L-BFGS Finisher...")
        model.load_state_dict(king.best_state)
        _, err_before = audit_global_fast(model, t_max)
        lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=500, line_search_fn="strong_wolfe")
        def closure():
            lbfgs.zero_grad()
            b = generate_mixed_batch(cfg['training']['n_sample'], bounds, cfg['geometry']['x_min'], cfg['geometry']['x_max'], t_max)
            p, xt_bfgs, xic, uic, bc_l, bc_r, ubc_l, ubc_r = b
            if not xt_bfgs.requires_grad: xt_bfgs.requires_grad_(True)
            b_safe = (p, xt_bfgs, xic, uic, bc_l, bc_r, ubc_l, ubc_r)
            loss = get_loss(model, b_safe, w_res, w_ic, w_bc)
            loss.backward()
            return loss
        try: lbfgs.step(closure)
        except: pass
        
        success_lbfgs, err_after = audit_global_fast(model, t_max)
        if err_after > err_before:
            print(f"      🛡️ L-BFGS dégradé. ROLLBACK.")
            model.load_state_dict(king.best_state)
        else:
            if success_lbfgs:
                print("      🚀 SUCCESS GLOBAL (L-BFGS).")
                global_success = True
                break

    if not global_success:
        print("🛑 ÉCHEC GLOBAL : Seuil non atteint.")
        return False, 1.0

    print("\n🔍 AUDIT FINAL OBLIGATOIRE PAR TYPE...")
    model.load_state_dict(king.best_state)
    failed_ids = diagnose_model(model, device, cfg, t_max=t_max)
    
    if len(failed_ids) == 0:
        print("✅ TOUS TYPES VALIDES.")
        _, final_err = audit_global_fast(model, t_max)
        return True, final_err
    else:
        print(f"⚠️ Échec spécifique. Correction avec LR={current_lr:.2e}")
        if targeted_correction(model, bounds, t_max, failed_ids, cfg['training']['n_iters_correction'], start_lr=current_lr):
            _, final_err = audit_global_fast(model, t_max)
            return True, final_err
        else:
            return False, 1.0

# =============================================================================
# 4. MAIN LOOP (AVEC REPRISE AUTO)
# =============================================================================

def train_smart_time_marching(model, bounds):
    device = next(model.parameters()).device
    
    # 1. DOSSIER DE LECTURE (Fixe pour trouver le t1.2)
    load_dir = "/lustre/fswork/projects/rech/fdb/usv13rn/These_DeepONet_ADR/outputs/checkpoints_shared"
    
    # 2. DOSSIER D'ÉCRITURE (Le run actuel)
    # On garde ce que le YAML ou train.py a envoyé
    save_dir = cfg['audit']['save_dir'] 
    os.makedirs(save_dir, exist_ok=True)
    
    # 🕵️ REPRISE AUTOMATIQUE (On cherche dans load_dir !)
    latest_file, max_t = find_latest_checkpoint(load_dir)
    reprise_active = False

    if latest_file:
        print(f"\n🔄 REPRISE DÉTECTÉE : Chargement du checkpoint t={max_t}...")
        try:
            checkpoint = torch.load(latest_file, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Modèle chargé avec succès.")
            reprise_active = True
        except Exception as e:
            print(f"⚠️ Erreur chargement checkpoint : {e}. Démarrage à zéro.")
            max_t = -1.0
    else:
        print("\n✨ Démarrage d'un nouvel entraînement.")

    # PHASE 0 : WARMUP (Seulement si pas de reprise ou reprise à 0.0)
    if not reprise_active or max_t <= 0.0:
        n_warmup = cfg['training'].get('n_warmup', 0)
        if n_warmup > 0:
            print(f"\n🧊 PHASE 0 : WARMUP ({n_warmup} iters)")
            optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
            king = KingOfTheHill(model)
            model.train()
            for i in range(n_warmup):
                optimizer.zero_grad()
                batch = generate_mixed_batch(cfg['training']['batch_size'], bounds, cfg['geometry']['x_min'], cfg['geometry']['x_max'], 0.0)
                params, _, xt_ic, u_true_ic, _, _, _, _ = batch
                loss = torch.mean((model(params, xt_ic) - u_true_ic)**2)
                loss.backward()
                optimizer.step()
                king.update(model, loss.item())
                if (i + 1) % 1000 == 0: print(f"      [Warmup] Iter {i+1} | Loss: {loss.item():.2e}")
            
            model.load_state_dict(king.best_state)
            print("🔎 Audit Warmup Global...")
            audit_global_fast(model, 0.0)
            failed_warmup = diagnose_model(model, device, cfg, t_max=0.0)
            if failed_warmup:
                print(f"⚠️ Warmup incomplet sur {failed_warmup}. Correction immédiate...")
                if not targeted_correction(model, bounds, 0.0, failed_warmup, 5000, start_lr=cfg['training']['learning_rate']):
                     print("❌ ÉCHEC CRITIQUE WARMUP.")
                     return model
                audit_global_fast(model, 0.0)

    # PHASE 1 : TIME MARCHING
    time_steps = generate_time_steps()
    print(f"⚡ TRAINING MULTI-ZONES : {time_steps}")
    
    for t_step in time_steps:
        # ⏭️ SKIP si déjà fait (C'EST ICI LA MAGIE)
        if reprise_active and t_step <= max_t + 1e-5:
            continue
            
        success, _ = train_step_time_window(model, bounds, t_max=t_step, n_iters_main=cfg['training']['n_iters_per_step'])
        if success:
            torch.save({'t_max': t_step, 'model_state_dict': model.state_dict()}, f"{save_dir}/model_checkpoint_t{t_step}.pth")
            print(f"✅ Palier t={t_step} OK.")
        else:
            print(f"🛑 Arrêt prématuré à t={t_step}."); break
            
    return model