import torch
import torch.optim as optim
import numpy as np
import copy
import yaml
import os
from tqdm import tqdm

# Imports des outils physiques et data
from src.physics.adr import pde_residual_adr
from src.data.generators import generate_mixed_batch
from src.audit_tool import diagnose_model
from src.physics.solver import get_ground_truth_CN

# =============================================================================
# 0. CONFIGURATION & FONCTIONS DE PERTE GLOBALES
# =============================================================================

def load_config(path="src/config_ADR.yaml"):
    """ Charge les paramètres du fichier YAML pour piloter tout le script. """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Config introuvable à {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

cfg = load_config()

def get_loss(model, batch, wr, wi, wb):
    """ Calcule la perte totale pondérée : PDE (Physique) + IC (Initiale) + BC (Bords). """
    params, xt, xt_ic, u_true_ic, xt_bc_l, xt_bc_r, u_true_bc_l, u_true_bc_r = batch
    l_pde = torch.mean(pde_residual_adr(model, params, xt)**2)
    l_ic = torch.mean((model(params, xt_ic) - u_true_ic)**2)
    l_bc = torch.mean((model(params, xt_bc_l) - u_true_bc_l)**2) + \
           torch.mean((model(params, xt_bc_r) - u_true_bc_r)**2)
    return wr * l_pde + wi * l_ic + wb * l_bc

# =============================================================================
# 1. OUTILS DE PILOTAGE (NTK, MONITORING, KING)
# =============================================================================

def compute_ntk_weights(model, batch, w_ic_ref):
    """ Ajuste dynamiquement le poids de la PDE pour qu'il équilibre la force de l'IC. """
    model.zero_grad()
    params, xt, xt_ic, u_true_ic, _, _, _, _ = batch
    
    loss_pde = torch.mean(pde_residual_adr(model, params, xt)**2)
    grad_pde = torch.autograd.grad(loss_pde, model.parameters(), retain_graph=True, create_graph=False)
    norm_pde = torch.sqrt(sum(g.pow(2).sum() for g in grad_pde))

    loss_ic = torch.mean((model(params, xt_ic) - u_true_ic)**2)
    grad_ic = torch.autograd.grad(loss_ic, model.parameters(), retain_graph=True, create_graph=False)
    norm_ic = torch.sqrt(sum(g.pow(2).sum() for g in grad_ic))

    new_w_pde = (norm_ic / (norm_pde + 1e-8)).item() * w_ic_ref
    return min(max(new_w_pde, 10.0), 500.0)

def monitor_gradients(model, batch):
    """ Calcule le Ratio de force (équilibre) et le CosSim (conflit de direction). """
    model.zero_grad()
    params, xt, xt_ic, u_true_ic, _, _, _, _ = batch
    
    lp = torch.mean(pde_residual_adr(model, params, xt)**2)
    gp = torch.autograd.grad(lp, model.parameters(), retain_graph=True)
    fp = torch.cat([g.view(-1) for g in gp])
    
    li = torch.mean((model(params, xt_ic) - u_true_ic)**2)
    gi = torch.autograd.grad(li, model.parameters(), retain_graph=True)
    fi = torch.cat([g.view(-1) for g in gi])

    ratio = (torch.norm(fi) / (torch.norm(fp) + 1e-8)).item()
    cos_sim = torch.nn.functional.cosine_similarity(fp, fi, dim=0).item()
    return ratio, cos_sim

class KingOfTheHill:
    """ Sauvegarde les meilleurs poids rencontrés et surveille la stagnation de la loss. """
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
# 2. LOGIQUE D'ENTRAÎNEMENT (PALIER & CORRECTION)
# =============================================================================

def targeted_correction(model, bounds, t_max, failed_ids, n_iters):
    """ Sprint final focalisé à 80% sur les familles d'équations en échec. """
    print(f"\n🚑 CORRECTION CIBLÉE sur {failed_ids} ({n_iters} itérations)")
    device = next(model.parameters()).device
    all_types = [0, 1, 2, 3, 4]
    weighted_types = []
    for tid in all_types:
        w = 4 if tid in failed_ids else 1
        weighted_types.extend([tid] * w)
    
    opt = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'] * 0.5)
    
    for i in range(n_iters):
        batch = generate_mixed_batch(cfg['training']['n_sample'], bounds, 
                                     cfg['geometry']['x_min'], cfg['geometry']['x_max'], 
                                     t_max, allowed_types=weighted_types)
        opt.zero_grad()
        loss = get_loss(model, batch, 100.0, 100.0, cfg['loss_weights']['weight_bc']) 
        loss.backward()
        opt.step()
        if (i+1) % 1000 == 0: print(f"      [Focus] Iter {i+1} | Loss: {loss.item():.2e}")
    
    failed_now = diagnose_model(model, device, threshold=cfg['training']['threshold'], t_max=t_max)
    return len(failed_now) == 0

def train_step_time_window(model, bounds, t_max, n_iters_main):
    """ Gère l'entraînement d'un palier : Adam Retries -> L-BFGS -> Correction Ciblée. """
    device = next(model.parameters()).device
    king = KingOfTheHill(model)
    
    # Setup des poids (Rampe < 0.3 ou NTK > 0.3)
    w_bc = cfg['loss_weights']['weight_bc']
    if t_max <= 0.3:
        w_res = 10.0 + (cfg['loss_weights']['first_w_res'] - 10.0) * (t_max / 0.3)
        w_ic, mode = cfg['loss_weights']['weight_ic_init'], "RAMPE"
    else:
        w_res, w_ic, mode = cfg['loss_weights']['first_w_res'], cfg['loss_weights']['weight_ic_final'], "NTK"

    print(f"\n🔵 PALIER t={t_max} | Mode: {mode}")

    current_lr = cfg['training']['learning_rate']
    for macro in range(cfg['training']['nb_loop']):
        print(f"    🌀 Macro {macro+1}/{cfg['training']['nb_loop']}")
        
        # 1. Boucle Adam
        for retry in range(cfg['training']['max_retry']):
            optimizer = optim.Adam(model.parameters(), lr=current_lr)
            for i in range(n_iters_main):
                batch = generate_mixed_batch(cfg['training']['n_sample'], bounds, 
                                             cfg['geometry']['x_min'], cfg['geometry']['x_max'], t_max)
                if mode == "NTK" and i % 100 == 0:
                    w_res = compute_ntk_weights(model, batch, w_ic)
                
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

            success, err = audit_global_fast(model, t_max)
            if success: return True, err
            current_lr *= 0.5
            model.load_state_dict(king.best_state)

        # 2. L-BFGS
        print("    ☢️ L-BFGS Finisher...")
        lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=500, line_search_fn="strong_wolfe")
        def closure():
            lbfgs.zero_grad()
            b = generate_mixed_batch(cfg['training']['n_sample'], bounds, cfg['geometry']['x_min'], cfg['geometry']['x_max'], t_max)
            loss = get_loss(model, b, w_res, w_ic, w_bc); loss.backward(); return loss
        try: lbfgs.step(closure)
        except: pass
        
        success, err = audit_global_fast(model, t_max)
        if success: return True, err
        model.load_state_dict(king.best_state)

    # 3. Correction de la dernière chance
    failed_ids = diagnose_model(model, device, threshold=cfg['training']['threshold'], t_max=t_max)
    if failed_ids:
        if targeted_correction(model, bounds, t_max, failed_ids, cfg['training']['n_iters_correction']):
            _, final_err = audit_global_fast(model, t_max)
            return True, final_err

    return False, 1.0

# =============================================================================
# 3. GESTION DU TEMPS & AUDIT
# =============================================================================

def generate_time_steps():
    """ Génère la liste des paliers de temps en fonction des zones définies. """
    steps, current_t, t_limit = [], 0.0, cfg['geometry']['T_max']
    for zone in cfg['time_stepping']['zones']:
        t_end = t_limit if zone['t_end'] == -1 else zone['t_end']
        dt = zone['dt']
        while current_t < t_end - 1e-5:
            current_t = round(current_t + dt, 3)
            if current_t > t_limit: break
            steps.append(current_t)
    return steps

def audit_global_fast(model, t_max):
    """ Évalue l'erreur moyenne L2 relative du modèle par rapport à la vérité terrain. """
    device = next(model.parameters()).device
    model.eval()
    errors = []
    
    # Test sur 10 cas aléatoires
    for _ in range(100):
        # 1. Génération de paramètres aléatoires
        p_dict = {k: np.random.uniform(v[0], v[1]) for k, v in cfg['physics_ranges'].items()}
        p_dict['type'] = np.random.randint(0, 5) # Choix aléatoire du type
        
        try:
            # 2. Récupération de la Vérité Terrain (Solver)
            X_grid, T_grid, U_true = get_ground_truth_CN(p_dict, cfg, t_step_max=t_max)
            
            # 3. Prédiction du Modèle (DeepONet)
            # On prépare les entrées pour le modèle : [params] et [xt]
            # Attention : Il faut formater les inputs pour qu'ils aient la shape (N_grid, ...)
            
            # Aplatissement des grilles pour le batch
            x_flat = X_grid.flatten()
            t_flat = T_grid.flatten()
            n_points = len(x_flat)
            
            # Création du vecteur de paramètres répété
            # Ordre: v, D, mu, type, A, x0, sigma, k
            p_vec = np.array([p_dict['v'], p_dict['D'], p_dict['mu'], p_dict['type'], 
                              p_dict['A'], 0.0, p_dict['sigma'], p_dict['k']]) # x0 est toujours 0
            p_tensor = torch.tensor(p_vec, dtype=torch.float32).repeat(n_points, 1).to(device)
            
            xt_tensor = torch.tensor(np.stack([x_flat, t_flat], axis=1), dtype=torch.float32).to(device)
            
            with torch.no_grad():
                u_pred_flat = model(p_tensor, xt_tensor).cpu().numpy().flatten()
            
            u_true_flat = U_true.flatten()
            
            # 4. Calcul de l'erreur Relative L2
            # On ajoute un epsilon pour éviter la division par zéro sur les zones nulles
            err = np.linalg.norm(u_true_flat - u_pred_flat) / (np.linalg.norm(u_true_flat) + 1e-8)
            errors.append(err)
            
        except Exception as e:
            # En cas d'échec du solveur (rare), on ignore ce cas
            continue

    if not errors: return False, 1.0 # Si tout a échoué
    
    avg_err = np.mean(errors)
    print(f"      [Audit] Avg Rel L2 Error: {avg_err:.2%}")
    return avg_err < cfg['training']['threshold'], avg_err

def train_smart_time_marching(model, bounds):
    """ Boucle principale : Phase 0 (Warmup) + Phases Temporelles. """
    
    # --- PHASE 0 : WARMUP (Condition Initiale seule) ---
    n_warmup = cfg['training'].get('n_warmup', 0)
    if n_warmup > 0:
        print(f"\n🧊 PHASE 0 : WARMUP ({n_warmup} itérations à t=0)")
        optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
        model.train()
        
        for i in range(n_warmup):
            optimizer.zero_grad()
            # On génère un batch forcé à t=0
            batch = generate_mixed_batch(
                cfg['training']['batch_size'], bounds, 
                cfg['geometry']['x_min'], cfg['geometry']['x_max'], 0.0
            )
            params, _, xt_ic, u_true_ic, _, _, _, _ = batch
            
            # Perte IC uniquement
            loss = torch.mean((model(params, xt_ic) - u_true_ic)**2)
            
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 1000 == 0:
                print(f"      [Warmup] Iter {i+1}/{n_warmup} | Loss IC: {loss.item():.2e}")
        
        # Sauvegarde post-warmup pour sécurité
        torch.save(model.state_dict(), f"{cfg['audit']['save_dir']}/model_post_warmup.pth")
        print("✅ Warmup terminé. Passage à la marche en temps.")

    # --- BOUCLE TEMPORELLE ---
    time_steps = generate_time_steps()
    print(f"⚡ TRAINING MULTI-ZONES : {time_steps}")
    for t_step in time_steps:
        success, _ = train_step_time_window(model, bounds, t_max=t_step, 
                                            n_iters_main=cfg['training']['n_iters_per_step'])
        if success:
            torch.save({'t_max': t_step, 'model_state_dict': model.state_dict()}, 
                       f"{cfg['audit']['save_dir']}/model_checkpoint_t{t_step}.pth")
            print(f"✅ Palier t={t_step} OK.")
        else:
            print(f"🛑 Échec à t={t_step}."); break
            
    return model