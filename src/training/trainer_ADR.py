"""
Training & Time Marching
This module orchestrates the training of the PI-DeepONet for the ADR equation.
It implements a Time Marching strategy (learning through successive time steps) with automatic auditing and targeted correction mechanisms.
INCLUDED FUNCTIONS:
    load_config: Loads YAML parameters.
    generate_time_steps: Divides time into zones for curriculum learning.
    find_latest_checkpoint: Manages automatic resumption after interruption.
    get_loss: Calculates the multi-objective cost function (PDE, IC, BC).
    compute_ntk_weights: Dynamically balances weights via the Neural Tangent Kernel.
    monitor_gradients: Analyzes training health (force and direction).
    KingOfTheHill: Saves the best-state model 
    audit_global_fast: Rapid evaluation of the overall relative L2 error.
    targeted_correction: Specific refinement on IC families failing the audit.
    train_step_time_window: Management of a specific learning step.
    train_smart_time_marching: Main training and final polishing loop.
"""

import torch
import torch.optim as optim
import numpy as np
import copy
import yaml
import os
import glob
import re
from tqdm import tqdm
import sys

file_path = os.path.abspath(__file__) 
project_root = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))

if project_root not in sys.path:
    sys.path.append(project_root)

from src.physics.residual_ADR import pde_residual_adr
from src.data.generators import generate_mixed_batch
from src.utils.metrics import diagnose_model
from src.utils.CN_ADR import get_ground_truth_CN

#Utils and configs
def load_config(path="configs/config_ADR.yaml"):
    """
    Loads the YAML configuration file. Centralizes all hyperparameters (geometry, physics, training) to ensure the reproducibility of experiments.
    Args:
        path (str): Path to the .yaml file.
    Outputs:
        dict: Loaded configuration dictionary.
"""
    if not os.path.exists(path): path = "configs/config_ADR.yaml"
    with open(path, 'r') as f: return yaml.safe_load(f)

cfg = load_config()

def generate_time_steps():
    """
    Generates the list of time steps for training. Implements the time stepping defined in the configuration for Time Marching, allowing the model to learn the solution over increasingly longer windows.
    Args:
        None (uses the global dictionary cfg).
    Outputs:
        list[float]: List of successive times t_max for training.
"""
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
    """Scans the save folder to identify the last valid checkpoint. Enables automatic resumption (resilience) of interrupted training on a computing cluster.
    Args:
        save_dir(str): Folder containing the .pth files.
    Outputs:
        tuple(str, float): Path to the most recent file and its associated t_max time.
    """
    abs_save_dir = os.path.abspath(save_dir)
    pattern = os.path.join(abs_save_dir, "model_checkpoint_t*.pth")
    
    print(f"Resume")
    print(f"Target folder : {abs_save_dir}")
    print(f"Pattern glob  : {pattern}")
    
    files = glob.glob(pattern)
    print(f"Found folders : {len(files)}")
    if len(files) > 0:
        print(f"List : {[os.path.basename(f) for f in files[:3]]}...")

    if not files: return None, 0.0
    
    max_t = -1.0
    best_file = None
    
    for f in files:
        match = re.search(r"model_checkpoint_t([\d\.]+)\.pth", f)
        if match:
            t_val = float(match.group(1))
            if t_val > max_t:
                max_t = t_val
                best_file = f
                
    print(f"Resume with : {best_file} (t={max_t})")
    return best_file, max_t

def get_loss(model, batch, wr, wi, wb):
    """
    Calculates the weighted total loss. Aggregates the PDE (physical), IC (initial condition), and BC (boundary) errors into a single scalar value for optimization.
    Args:
        model (nn.Module): The DeepONet network.
        batch (tuple): Generated data (params, xt, ic, bc).
        wr, wi, wb (float): Respective weights for PDE, IC, and BC.
    Outputs:
        torch.Tensor: Total loss value.
    """
    params, xt, xt_ic, u_true_ic, xt_bc_l, xt_bc_r, u_true_bc_l, u_true_bc_r = batch
    if not xt.requires_grad: xt.requires_grad_(True)
    l_pde = torch.mean(pde_residual_adr(model, params, xt)**2)
    l_ic = torch.mean((model(params, xt_ic) - u_true_ic)**2)
    l_bc = torch.mean((model(params, xt_bc_l) - u_true_bc_l)**2) + \
           torch.mean((model(params, xt_bc_r) - u_true_bc_r)**2)
    return wr * l_pde + wi * l_ic + wb * l_bc

# Helper functions
def compute_ntk_weights(model, batch, w_ic_ref):
    """
    Dynamically calculates the weight of the PDE loss relative to the IC. Uses the Neural Tangent Kernel heuristic to balance the gradients
of different tasks and prevent the IC or the PDE from dominating learning.
    Args:
        model(nn.Module): The model.
        batch(tuple): Current data.
        w_ic_ref(float): Reference weight for the initial condition.
    Outputs:
        float: New suggested wr weight for the PDE.
    """
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
    """
    Analyzes the strength ratio and cosine similarity of gradients. Diagnoses conflicts between the physical model (PDE) and the data (IC).
    Args:
        model(nn.Module): The model.
        batch(tuple): Current data.
    Outputs:
        tuple(float, float): Ratio of norms (IC/PDE) and cosine similarity.
    """
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
    """
    Utility class for managing the best-state model. Saves in memory the weights producing the smallest loss encountered during an optimization phase, acting as a safety net against divergences or overfitting.
    """
    def __init__(self, model):
        """
        Args:
            model (nn.Module): Model to watch.
        """
        self.best_state = copy.deepcopy(model.state_dict())
        self.best_loss = float('inf')
        self.history = []

    def update(self, model, current_loss):
        """
        Updates the champion if the current loss is better.
        Args:
            model(nn.Module): Current model.
            current_loss(float): Current round loss.
        Outputs:
            bool: True if a new record is set.
        """
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_state = copy.deepcopy(model.state_dict())
            return True
        return False

#Global audit 
def audit_global_fast(model, t_max):
    """
    Performs a quick audit of the L2 relative error across the entire domain. Statistically checks (on 200 random cases) if the model meets
the set accuracy thresholds (threshold_ic or threshold_step) before proceeding to the next time step.
    Args:
        model(nn.Module): Model to test.
        t_max(float): Current time window.
    Outputs:
        tuple(bool, float): Audit success and calculated average error.
"""
    device = next(model.parameters()).device
    model.eval()
    np.random.seed(42) 
    errors = []
    
    if t_max == 0.0:
        target_threshold = cfg['training'].get('threshold_ic', cfg['training'].get('threshold', 0.01))
        mode_str = "IC (Strict)"
    else:
        target_threshold = cfg['training'].get('threshold_step', cfg['training'].get('threshold', 0.05))
        mode_str = "Step (Relaxed)"

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
    
    print(f"Audit {mode_str} | Avg Rel L2: {avg_err:.2%} (Target: < {target_threshold:.2%})")
    
    return avg_err < target_threshold, avg_err

def get_t_failed(model, cfg, threshold=0.04):
    """
    Nouveau helper : Identifie le premier point temporel où le modèle décroche.
    """
    print(f"🔍 Recherche du point de bascule temporel (t_failed) pour batch 80/20...")
    device = next(model.parameters()).device
    model.eval()
    t_evals = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    for t_val in t_evals:
        errors = []
        for _ in range(15):
            p_dict = {k: np.random.uniform(v[0], v[1]) for k, v in cfg['physics_ranges'].items()}
            p_dict['type'] = np.random.choice([0, 1, 3])
            try:
                X_grid, T_grid, U_true = get_ground_truth_CN(p_dict, cfg, t_step_max=t_val)
                x_flat, t_flat = X_grid.flatten(), T_grid.flatten()
                p_vec = np.array([p_dict['v'], p_dict['D'], p_dict['mu'], p_dict['type'], 
                                  p_dict['A'], 0.0, p_dict['sigma'], p_dict['k']])
                p_tensor = torch.tensor(p_vec, dtype=torch.float32).repeat(len(x_flat), 1).to(device)
                xt_tensor = torch.tensor(np.stack([x_flat, t_flat], axis=1), dtype=torch.float32).to(device)
                with torch.no_grad():
                    u_pred_flat = model(p_tensor, xt_tensor).cpu().numpy().flatten()
                err = np.linalg.norm(U_true.flatten() - u_pred_flat) / (np.linalg.norm(U_true.flatten()) + 1e-8)
                errors.append(err)
            except: continue
        
        if errors and np.mean(errors) > threshold:
            print(f"⚠️ Décrochage détecté à t={t_val} (Erreur: {np.mean(errors):.2%})")
            return max(0.1, t_val) 
            
    print("✅ Le modèle semble robuste partout, fallback de t_failed à 1.5s")
    return 1.5

def targeted_correction(model, bounds, t_max, initial_failed_ids, n_iters_base, start_lr, target_threshold=None, apply_80_20=False):
    """
    Launches an intensive training phase with dynamic Active Learning.
    Evaluates failure points every 1000 iterations and adjusts batch composition and Learning Rate dynamically.

    Args:
        model(nn.Module): Model to correct.
        bounds(dict): Physical ranges.
        t_max(float): Current time.
        initial_failed_ids(list): Initial IDs of failing IC types.
        n_iters_base(int): Total number of macro iterations (e.g., 30000).
        start_lr(float): Initial learning rate.
        target_threshold(float, optional): Specific success threshold.
        apply_80_20(bool): Activer l'échantillonnage temporel biaisé (désormais 65/35).
    Outputs:
        bool: True if the correction successfully validated all types.
    """
    print(f"🚀 Démarrage du Dynamic Soft Polishing...")
    device = next(model.parameters()).device
    king_corr = KingOfTheHill(model)
    save_dir = cfg['audit']['save_dir'] 
    
    w_bc_loc = cfg['loss_weights']['weight_bc']
    if t_max == 0.0: 
        w_res_loc, w_ic_loc = 0.0, 100.0
    else: 
        w_res_loc = cfg['loss_weights']['first_w_res'] 
        w_ic_loc = cfg['loss_weights']['weight_ic_final']

    # Configuration du scheduler de LR (de start_lr à 1e-6 sur n_iters_base)
    end_lr = 1e-6
    gamma = (end_lr / start_lr) ** (1 / (n_iters_base / 1000)) if start_lr > end_lr else 1.0
    current_lr = start_lr
    
    opt = optim.Adam(model.parameters(), lr=current_lr)
    
    # État initial
    current_failed_ids = initial_failed_ids
    best_val_err = float('inf')
    
    t_failed = None
    if apply_80_20 and target_threshold is not None:
        t_failed = get_t_failed(model, cfg, threshold=target_threshold)

    pbar = tqdm(range(n_iters_base), desc="Soft Polishing", leave=True)
    
    for i in pbar:
        # --- 1. Ajustement dynamique du batch (toutes les 1000 itérations) ---
        if i % 1000 == 0:
            if i > 0:
                # Audit pour voir l'état actuel
                current_failed_ids = diagnose_model(model, device, cfg, threshold=target_threshold, t_max=t_max, silent=True) # 
                
                # Mise à jour du LR
                current_lr = max(end_lr, current_lr * gamma)
                for param_group in opt.param_groups:
                    param_group['lr'] = current_lr
                
                # Check d'arrêt anticipé : si plus aucune IC ne décroche, on a gagné !
                if len(current_failed_ids) == 0:
                    print(f"\n✅ Objectif atteint à l'itération {i}! Toutes les IC sont sous le seuil.")
                    return True

            # Construction des poids du batch en fonction de current_failed_ids
            all_types = [0, 1, 2, 3, 4]
            weighted_types = []
            for tid in all_types:
                if tid in current_failed_ids:
                    weighted_types.extend([tid] * 2) # Force douce (x2 au lieu de x4)
                else:
                    weighted_types.extend([tid] * 1)
            
            pbar.set_description(f"Polishing | LR: {current_lr:.1e} | Fails: {current_failed_ids}")

        # --- 2. Génération du batch ---
        batch = generate_mixed_batch(cfg['training']['n_sample'], bounds, 
                                     cfg['geometry']['x_min'], cfg['geometry']['x_max'], 
                                     t_max, allowed_types=weighted_types)
        params, xt, xt_ic, u_true_ic, xt_bc_l, xt_bc_r, u_true_bc_l, u_true_bc_r = batch
        
        # --- 3. Split Temporel 65/35 ---
        if apply_80_20 and t_failed is not None and t_max > t_failed:
            n_samples = xt.shape[0]
            n_fail = int(0.65 * n_samples) # 65% sur la zone difficile
            n_pass = n_samples - n_fail    # 35% sur le passé
            
            t_pass = torch.rand(n_pass, 1) * t_failed
            t_fail = torch.rand(n_fail, 1) * (t_max - t_failed) + t_failed
            
            new_t = torch.cat([t_pass, t_fail], dim=0)
            new_t = new_t[torch.randperm(n_samples)].to(device)
            
            xt_new = xt.clone().detach()
            xt_new[:, 1:2] = new_t
            xt_new.requires_grad_(True)
            xt = xt_new
            batch = (params, xt, xt_ic, u_true_ic, xt_bc_l, xt_bc_r, u_true_bc_l, u_true_bc_r)
        else:
            if not xt.requires_grad: xt.requires_grad_(True)

        # Activation NTK périodique pour éviter l'explosion
        if t_max > 0.0 and i % 100 == 0:
            w_res_loc = compute_ntk_weights(model, batch, w_ic_loc)

        # --- 4. Optimisation ---
        opt.zero_grad()
        loss = get_loss(model, batch, w_res_loc, w_ic_loc, w_bc_loc) 
        loss.backward()
        opt.step()
        
        if king_corr.update(model, loss.item()):
            # Sauvegarde silencieuse en fond du meilleur modèle pure-loss
            pass
            
        if i % 100 == 0:
            pbar.set_postfix({"Loss": f"{loss.item():.2e}"})

    # À la fin de la boucle de 30 000 itérations
    print("\nFin de la boucle de Polishing.")
    model.load_state_dict(king_corr.best_state)
    torch.save(model.state_dict(), f"{save_dir}/model_best_validation.pth")
    
    final_fails = diagnose_model(model, device, cfg, threshold=target_threshold, t_max=t_max)
    return len(final_fails) == 0

def train_step_time_window(model, bounds, t_max, n_iters_main):
    """
Executes the training over a given time window. Manages the PDE weight ramp initially and then activates the NTK. Chains Adam and L-BFGS to ensure plateau convergence.
    Args:
        model(nn.Module): The model.
        bounds(dict): Physical ranges.
        t_max(float): Target time of the window.
        n_iters_main(int): Adam iterations.
    Outputs:
        tuple(bool, float): Plateau success and final error.
"""
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

    print(f"Level t={t_max} | Mode: {mode}")
    current_lr = cfg['training']['learning_rate']
    global_success = False

    for macro in range(cfg['training']['nb_loop']):
        print(f"Lacro {macro+1}/{cfg['training']['nb_loop']}")
        
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
                    print(f"It {i} | Loss: {loss.item():.2e} | ForceRatio: {r:.2f} | CosSim: {c:.2f}")

            success_adam, err = audit_global_fast(model, t_max)
            if success_adam:
                print("Global success (Adam).")
                global_success = True
                break
            current_lr *= 0.5
        
        if global_success: break

        print("LBFGS Finisher")
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
            print(f"L-BFGS degraded. ROLLBACK.")
            model.load_state_dict(king.best_state)
        else:
            if success_lbfgs:
                print("Global success (LBFGS)")
                global_success = True
                break

    if not global_success:
        print("Global failure")
        return False, 1.0

    print("Final audit")
    model.load_state_dict(king.best_state)
    failed_ids = diagnose_model(model, device, cfg, t_max=t_max)
    
    if len(failed_ids) == 0:
        print("✅ Every types OK")
        _, final_err = audit_global_fast(model, t_max)
        return True, final_err
    else:
        print(f"Specific failure. Correction")
        if targeted_correction(model, bounds, t_max, failed_ids, cfg['training']['n_iters_correction'], start_lr=current_lr):
            _, final_err = audit_global_fast(model, t_max)
            return True, final_err
        else:
            return False, 1.0
            
#main loop
def train_smart_time_marching(model, bounds):
    """
    Main loop managing the temporal curriculum learning and final polishing. This is the conductor that manages the resumption, the warmup (t=0), the temporal progression by zones, and the final "Elite" phase to guarantee an error < 2%.
    Args:
        model(nn.Module): The model to be trained.
        bounds(dict): Ranges of physical parameters.
    Outputs:
        nn.Module: The trained and polished model.
"""
    device = next(model.parameters()).device
    
    # FORCAGE : On pointe directement sur le dossier du supercalculateur pour forcer la reprise
    load_dir = "/lustre/fswork/projects/rech/fdb/usv13rn/These_DeepONet_ADR/results/run_20260227-153846/model_latest_t3.0.pth"
    
    # Writing folder
    save_dir = cfg['audit']['save_dir'] 
    os.makedirs(save_dir, exist_ok=True)
    
    # Reprise forcée au temps 3.0
    forced_t = 3.0
    latest_file = os.path.join(load_dir, f"model_checkpoint_t{forced_t}.pth")
    max_t = forced_t
    reprise_active = False

    if os.path.exists(latest_file):
        print(f"FORCED Resumption detected. Checkpoint t={max_t} ({latest_file})...")
        try:
            checkpoint = torch.load(latest_file, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            reprise_active = True
        except Exception as e:
            print(f"Err checkpoints loading : {e}. Reset and restart at 0")
            max_t = -1.0
    else:
        print(f"⚠️ Fichier {latest_file} introuvable ! Starting a new training")
        max_t = -1.0

    # WARMUP IC
    if not reprise_active or max_t <= 0.0:
        n_warmup = cfg['training'].get('n_warmup', 0)
        if n_warmup > 0:
            print(f" Warmup ({n_warmup} iters)")
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
                if (i + 1) % 1000 == 0: print(f"[Warmup] Iter {i+1} | Loss: {loss.item():.2e}")
            
            model.load_state_dict(king.best_state)
            print("Audit Warmup Global")
            audit_global_fast(model, 0.0)
            failed_warmup = diagnose_model(model, device, cfg, t_max=0.0)
            if failed_warmup:
                print(f"Failed Warmup on {failed_warmup}. Correction.")
                if not targeted_correction(model, bounds, 0.0, failed_warmup, 5000, start_lr=cfg['training']['learning_rate']):
                     print("❌ Critical failure.")
                     return model
                audit_global_fast(model, 0.0)

    # TIME MARCHING
    time_steps = generate_time_steps()
    print(f"Multi zone training : {time_steps}")
    
    for t_step in time_steps:
        # Skip if it was already did 
        if reprise_active and t_step <= max_t + 1e-5:
            continue
            
        success, _ = train_step_time_window(model, bounds, t_max=t_step, n_iters_main=cfg['training']['n_iters_per_step'])
        if success:
            torch.save({'t_max': t_step, 'model_state_dict': model.state_dict()}, f"{save_dir}/model_checkpoint_t{t_step}.pth")
            print(f"Level t={t_step} OK.")
        else:
            print(f"Stop at à t={t_step}."); break

    #polishing
    print("Start of final polishing (Target: < 2% everywhere)")

    final_t = cfg['geometry']['T_max']
    last_checkpoint_path = f"{save_dir}/model_checkpoint_t{time_steps[-1]}.pth"
    
    if os.path.exists(last_checkpoint_path):
        print(f"📥 Chargement du checkpoint final : {last_checkpoint_path}")
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # diag
    target_strict = 0.02
    failed_ids = diagnose_model(model, device, cfg, threshold=target_strict, t_max=final_t)

    if len(failed_ids) > 0:
        print(f"IC above {target_strict:.1%} : {failed_ids}")
        print(f"Launching specific training WITH 65/35 DYNAMIC SAMPLING")

        success_polish = targeted_correction(
            model, 
            bounds, 
            t_max=final_t, 
            initial_failed_ids=failed_ids, 
            n_iters_base=30000,   
            start_lr=1e-5,        
            target_threshold=target_strict,
            apply_80_20=True 
        )

        if success_polish:
            print("Goal 2% reached.")
            torch.save({
                't_max': final_t, 
                'model_state_dict': model.state_dict(),
                'note': 'Polished < 2%'
            }, f"{save_dir}/model_final_elite_2percent.pth")
            diagnose_model(model, device, cfg, threshold=target_strict, t_max=final_t)
        else:
            print("The model didn't exceed 2% everywhere, but the best model was saved.")
    else:
        print("Everything under 2%")

    return model