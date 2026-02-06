import torch
import torch.optim as optim
import numpy as np
import copy
import yaml
import os
from tqdm import tqdm

# Imports
from src.physics.adr import pde_residual_adr
from src.data.generators import generate_mixed_batch
from src.audit_tool import diagnose_model
from src.physics.solver import get_ground_truth_CN

# =============================================================================
# CONFIG & UTILS
# =============================================================================

def load_config(path="src/config_ADR.yaml"):
    if not os.path.exists(path): path = "src/config_ADR.yaml"
    with open(path, 'r') as f: return yaml.safe_load(f)

cfg = load_config()

def get_loss(model, batch, wr, wi, wb):
    params, xt, xt_ic, u_true_ic, xt_bc_l, xt_bc_r, u_true_bc_l, u_true_bc_r = batch
    # Sécurité Gradient
    if not xt.requires_grad: xt.requires_grad_(True)
    
    l_pde = torch.mean(pde_residual_adr(model, params, xt)**2)
    l_ic = torch.mean((model(params, xt_ic) - u_true_ic)**2)
    l_bc = torch.mean((model(params, xt_bc_l) - u_true_bc_l)**2) + \
           torch.mean((model(params, xt_bc_r) - u_true_bc_r)**2)
    return wr * l_pde + wi * l_ic + wb * l_bc

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
    except Exception as e:
        return 1.0, 0.0

class KingOfTheHill:
    def __init__(self, model):
        self.best_state = copy.deepcopy(model.state_dict())
        self.best_loss = float('inf')
        self.history = []
    def update(self, model, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_state = copy.deepcopy(model.state_dict())
            return True
        return False
    def check_stagnation(self, current_loss, window):
        self.history.append(current_loss)
        if len(self.history) < window * 2: return False
        avg_old = np.mean(self.history[-window*2 : -window])
        avg_new = np.mean(self.history[-window:])
        return abs(avg_old - avg_new) / (avg_old + 1e-9) < 0.001

# =============================================================================
# LOGIQUE D'ENTRAÎNEMENT & CORRECTION
# =============================================================================

def targeted_correction(model, bounds, t_max, failed_ids, n_iters):
    """ Correction ciblée avec gestion intelligente des poids (Warmup vs Normal). """
    print(f"\n🚑 CORRECTION CIBLÉE OBLIGATOIRE sur {failed_ids} ({n_iters} iters)")
    device = next(model.parameters()).device
    
    # --- 1. STRATÉGIE GUERRIER (90/10) ---
    all_types = [0, 1, 2, 3, 4]
    weighted_types = []
    
    for tid in all_types:
        if tid in failed_ids:
            weighted_types.extend([tid] * 45) # Poids massif
        else:
            weighted_types.extend([tid] * 1)  # Rappel

    print(f"   ⚔️  Mode Guerrier : Poids {len(weighted_types)} éléments (Ratio ~45:1)")

    # --- 2. ADAPTATION DES POIDS (Le Correctif) ---
    if t_max == 0.0:
        # Cas WARMUP : Pas de physique, pas de bords temporels, JUSTE L'IC
        print("   🧊 Mode Warmup : Physique désactivée (w_res=0).")
        w_res_loc = 0.0
        w_bc_loc = 0.0
        w_ic_loc = 100.0
    else:
        # Cas NORMAL : Physique forte pour caler la phase
        w_res_loc = 500.0  # On s'aligne sur ta config optimisée
        w_bc_loc = cfg['loss_weights']['weight_bc']
        w_ic_loc = 100.0

    forced_lr = 5e-5 
    opt = optim.Adam(model.parameters(), lr=forced_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iters, eta_min=1e-6)

    for i in range(n_iters):
        batch = generate_mixed_batch(cfg['training']['n_sample'], bounds, 
                                     cfg['geometry']['x_min'], cfg['geometry']['x_max'], 
                                     t_max, allowed_types=weighted_types)
        
        params, xt, xt_ic, u_true_ic, _, _, _, _ = batch
        # Sécurité Gradient toujours là
        if not xt.requires_grad: xt.requires_grad_(True)

        opt.zero_grad()
        # On utilise les poids adaptés ici
        loss = get_loss(model, batch, w_res_loc, w_ic_loc, w_bc_loc) 
        loss.backward()
        opt.step()
        scheduler.step()

        if (i+1) % 1000 == 0: 
            print(f"      [Focus] Iter {i+1} | Loss: {loss.item():.2e}")
    
    failed_now = diagnose_model(model, device, cfg, t_max=t_max)
    return len(failed_now) == 0

def train_step_time_window(model, bounds, t_max, n_iters_main):
    device = next(model.parameters()).device
    king = KingOfTheHill(model)
    
    # Config poids
    w_bc = cfg['loss_weights']['weight_bc']
    if t_max <= 0.3:
        w_res = 10.0 + (cfg['loss_weights']['first_w_res'] - 10.0) * (t_max / 0.3)
        w_ic, mode = cfg['loss_weights']['weight_ic_init'], "RAMPE"
    else:
        w_res, w_ic, mode = cfg['loss_weights']['first_w_res'], cfg['loss_weights']['weight_ic_final'], "NTK"

    print(f"\n🔵 PALIER t={t_max} | Mode: {mode}")
    current_lr = cfg['training']['learning_rate']
    
    # --- BOUCLE PRINCIPALE (ADAM + L-BFGS) ---
    for macro in range(cfg['training']['nb_loop']):
        print(f"    🌀 Macro {macro+1}/{cfg['training']['nb_loop']}")
        
        # 1. Adam
        for retry in range(cfg['training']['max_retry']):
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
                    if king.check_stagnation(loss.item(), cfg['training']['rolling_window']):
                        print("      ⏸️ Stagnation détectée."); break

            # Audit Global Rapide (Juste pour info ou sortie anticipée de la boucle Adam)
            success_global, err = audit_global_fast(model, t_max)
            if success_global:
                print("      ✅ Audit Global OK. Passage à l'audit spécifique.")
                break # On sort de la boucle retry, MAIS on ne retourne pas encore True !
            
            current_lr *= 0.5
            model.load_state_dict(king.best_state)

        # 2. L-BFGS (Seulement si l'audit global n'était pas parfait ou systématiquement)
        print("    ☢️ L-BFGS Finisher...")
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
        
        # On remet le meilleur état connu
        model.load_state_dict(king.best_state)

    # --- LE JUGE DE PAIX : DIAGNOSTIC SPÉCIFIQUE OBLIGATOIRE ---
    print("\n🔍 AUDIT FINAL OBLIGATOIRE PAR TYPE...")
    failed_ids = diagnose_model(model, device, cfg, t_max=t_max)
    
    if len(failed_ids) == 0:
        # Tout est vert partout -> SUCCÈS
        _, final_err = audit_global_fast(model, t_max)
        return True, final_err
    else:
        # Au moins un type rouge -> CORRECTION
        print(f"⚠️ Échec spécifique détecté sur {failed_ids}. Lancement Correction...")
        if targeted_correction(model, bounds, t_max, failed_ids, cfg['training']['n_iters_correction']):
            _, final_err = audit_global_fast(model, t_max)
            return True, final_err
        else:
            return False, 1.0

def audit_global_fast(model, t_max):
    device = next(model.parameters()).device
    model.eval()
    errors = []
    for _ in range(100):
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
    if not errors: return False, 1.0
    avg_err = np.mean(errors)
    print(f"      [Audit Global] Avg Rel L2: {avg_err:.2%}")
    return avg_err < cfg['training']['threshold'], avg_err

def train_smart_time_marching(model, bounds):
    device = next(model.parameters()).device
    
    # PHASE 0 : WARMUP
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
        
        # --- DIAGNOSTIC OBLIGATOIRE WARMUP ---
        print("🔎 Audit Détaillé du Warmup...")
        failed_warmup = diagnose_model(model, device, cfg, t_max=0.0)
        
        if failed_warmup:
            print(f"⚠️ Warmup incomplet sur {failed_warmup}. Correction immédiate...")
            targeted_correction(model, bounds, 0.0, failed_warmup, 5000)
            
        success_w, err_w = audit_global_fast(model, 0.0)
        print(f"      🔎 Audit Final Warmup : {err_w:.2%}")
        if not success_w and len(diagnose_model(model, device, cfg, t_max=0.0)) > 0:
             print("❌ ECHEC WARMUP : Le modèle ne peut pas apprendre t=0.")
             # On peut choisir de lever une erreur ou continuer à ses risques et périls

    # PHASE 1
    time_steps = generate_time_steps()
    print(f"⚡ TRAINING MULTI-ZONES : {time_steps}")
    for t_step in time_steps:
        success, _ = train_step_time_window(model, bounds, t_max=t_step, n_iters_main=cfg['training']['n_iters_per_step'])
        if success:
            torch.save({'t_max': t_step, 'model_state_dict': model.state_dict()}, f"{cfg['audit']['save_dir']}/model_checkpoint_t{t_step}.pth")
            print(f"✅ Palier t={t_step} OK.")
        else:
            print(f"🛑 Échec à t={t_step}."); break
    return model