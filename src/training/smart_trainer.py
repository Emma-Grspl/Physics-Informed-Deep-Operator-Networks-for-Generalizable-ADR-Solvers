import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from src.physics.adr import pde_residual_adr
from src.data.generators import generate_mixed_batch
from src.physics.solver import get_ground_truth_CN

def audit_time_window(model, current_t_max, bounds, n_samples=300, threshold=0.03):
    """ Audit rapide (inchangé) """
    device = next(model.parameters()).device
    model.eval()
    
    Nx_audit = 100
    Nt_audit = 50 
    x_min, x_max = -7.0, 7.0
    errors = []
    
    for _ in range(n_samples):
        type_id = np.random.randint(0, 5)
        p_dict = {
            'v': np.random.uniform(0.5, 2.0),
            'D': np.random.uniform(0.01, 0.2),
            'mu': np.random.uniform(0.0, 1.0),
            'type': type_id,
            'A': np.random.uniform(0.8, 1.2),
            'x0': np.random.uniform(-1, 1),
            'sigma': np.random.uniform(0.4, 0.8),
            'k': np.random.uniform(1.0, 3.0)
        }
        X_grid, T_grid, U_true_np = get_ground_truth_CN(p_dict, x_min, x_max, current_t_max, Nx_audit, Nt_audit)

        X_flat = X_grid.flatten()
        T_flat = T_grid.flatten()
        xt_in = np.stack([X_flat, T_flat], axis=1)
        xt_tensor = torch.tensor(xt_in, dtype=torch.float32).to(device)
        p_vec = np.array([p_dict['v'], p_dict['D'], p_dict['mu'], p_dict['type'], 
                          p_dict['A'], p_dict['x0'], p_dict['sigma'], p_dict['k']])
        p_tensor = torch.tensor(p_vec, dtype=torch.float32).unsqueeze(0).repeat(len(X_flat), 1).to(device)

        with torch.no_grad():
            u_pred_flat = model(p_tensor, xt_tensor).cpu().numpy().flatten()
        
        U_pred_np = u_pred_flat.reshape(Nx_audit, Nt_audit)
        err_norm = np.linalg.norm(U_true_np - U_pred_np)
        true_norm = np.linalg.norm(U_true_np)
        errors.append(err_norm / (true_norm + 1e-6))

    mean_error = np.mean(errors)
    return mean_error < threshold, mean_error

def train_step_time_window(model, bounds, t_max, n_iters_main, max_retries=3, batch_size=2048, use_lbfgs=True):
    """ Fonction Ouvrier (inchangée, mais utilise n_iters_main passé en arg) """
    target_error = 0.03
    W_IC = 100.0
    W_PDE = 20.0
    lr_base = 5e-4
    
    optimizer = optim.Adam(model.parameters(), lr=lr_base)
    model.train()
    
    # --- MAIN TRY ---
    pbar = tqdm(range(n_iters_main), desc=f"Train T={t_max:.1f}", leave=False)
    for _ in pbar:
        optimizer.zero_grad()
        params, xt, xt_ic, u_true_ic = generate_mixed_batch(batch_size, bounds, -7, 7, t_max)
        
        loss_pde = torch.mean(pde_residual_adr(model, params, xt)**2)
        loss_ic = torch.mean((model(params, xt_ic) - u_true_ic)**2)
        loss = (W_PDE * loss_pde) + (W_IC * loss_ic)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    success, err = audit_time_window(model, t_max, bounds, n_samples=300, threshold=target_error)
    if success: return True, err

    # --- RETRIES ---
    lrs_retry = [1e-4, 5e-5, 1e-5]
    for attempt in range(max_retries):
        current_lr = lrs_retry[min(attempt, len(lrs_retry)-1)]
        print(f"   ⚠️ Retry {attempt+1}/{max_retries} (LR={current_lr:.0e}, Err: {err:.2%})...")
        
        if use_lbfgs and attempt == max_retries - 1:
            lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=50, line_search_fn="strong_wolfe")
            def closure():
                lbfgs.zero_grad()
                params, xt, xt_ic, u_true_ic = generate_mixed_batch(batch_size, bounds, -7, 7, t_max)
                loss = W_PDE*torch.mean(pde_residual_adr(model, params, xt)**2) + \
                       W_IC*torch.mean((model(params, xt_ic) - u_true_ic)**2)
                loss.backward()
                return loss
            try: lbfgs.step(closure)
            except: pass
        else:
            optimizer_retry = optim.Adam(model.parameters(), lr=current_lr)
            for _ in tqdm(range(3000), desc=f"Retry #{attempt+1}", leave=False): # Retry plus court
                optimizer_retry.zero_grad()
                params, xt, xt_ic, u_true_ic = generate_mixed_batch(batch_size, bounds, -7, 7, t_max)
                loss = W_PDE*torch.mean(pde_residual_adr(model, params, xt)**2) + \
                       W_IC*torch.mean((model(params, xt_ic) - u_true_ic)**2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer_retry.step()
        
        success, err = audit_time_window(model, t_max, bounds, n_samples=300, threshold=target_error)
        if success: return True, err

    return False, err

def train_smart_time_marching(model, bounds, n_warmup, n_iters_per_step, save_dir="results"):
    """
    Args:
        n_warmup (int): Nombre d'itérations pour fixer t=0 (IC) au début.
        n_iters_per_step (int): Nombre d'itérations pour chaque palier de temps (0.1, 0.2...)
    """
    BATCH_SIZE = 2048
    
    print(f"⚡ DÉMARRAGE TRAINING (Optuna Ready)")
    print(f"   -> Warmup (t=0): {n_warmup} iters")
    print(f"   -> Time Marching: {n_iters_per_step} iters/palier")

    # --- PHASE 0 : WARMUP STRICT (t=0) ---
    # C'est important pour ta Gaussienne qui s'écrase !
    if n_warmup > 0:
        print("\n🧊 PHASE 0 : Fixation Condition Initiale (t=0)...")
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in tqdm(range(n_warmup), desc="Warmup IC"):
            optimizer.zero_grad()
            # On demande t_max = 0.0, donc generate_mixed_batch ne sortira que du t=0 (ou presque)
            params, xt, xt_ic, u_true_ic = generate_mixed_batch(BATCH_SIZE, bounds, -7, 7, 0.0)
            
            # On ne calcule QUE la perte IC ici
            u_pred_ic = model(params, xt_ic)
            loss = torch.mean((u_pred_ic - u_true_ic)**2)
            
            loss.backward()
            optimizer.step()
        print("   ✅ Warmup terminé.")

    # --- PHASE 1 : TIME MARCHING ---
    time_steps = [round(t, 1) for t in np.arange(0.1, 1.1, 0.1)]
    
    for t_max in time_steps:
        print(f"\n⏳ --- PALIER TEMPOREL : [0, {t_max}] ---")
        
        # Ici on utilise la variable n_iters_per_step qu'Optuna pourra modifier
        success, final_err = train_step_time_window(
            model, bounds, t_max, 
            n_iters_main=n_iters_per_step,  # <--- C'est ici que ça se joue
            max_retries=3, 
            batch_size=BATCH_SIZE,
            use_lbfgs=True
        )
        
        if success:
            print(f"   ✅ PALIER VALIDÉ (Err: {final_err:.2%})")
        else:
            print(f"   ❌ PALIER NON VALIDÉ (Err: {final_err:.2%}). Expansion forcée.")

    print("\n🎉 Entraînement terminé.")
    
    # Audit Final
    from src.visualization.plots import evaluate_global_accuracy
    evaluate_global_accuracy(model, 100, bounds, -7, 7, 1.0, save_dir)
    
    return model