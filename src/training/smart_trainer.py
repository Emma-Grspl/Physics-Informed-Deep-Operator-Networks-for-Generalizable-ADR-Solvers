import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm 
from src.physics.adr import pde_residual_adr
from src.data.generators import generate_mixed_batch
from src.physics.solver import get_ground_truth_CN

def run_full_audit(model, threshold=0.03, n_tests_per_type=5): 
    """
    Full audit
    """
    device = next(model.parameters()).device
    model.eval()

    type_names = {0: "Step", 1: "Tanh", 2: "Sinus", 3: "Gauss", 4: "Bell"}
    failing_types = []
    max_error_global = 0.0 

    print(f"Audit in progress ({n_tests_per_type} tests/type) ---")

    Nx_audit, Nt_audit = 100, 100 
    x_min, x_max, T_max = -7, 7, 1.0

    global_success = True

    for type_id in [0, 1, 2, 3, 4]:
        errors = []
        pbar = tqdm(range(n_tests_per_type), desc=f"   Testing {type_names[type_id]}", leave=False)

        for _ in pbar:
            # parametric draws
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

            # Ground truth
            X_grid, T_grid, U_true_np = get_ground_truth_CN(p_dict, x_min, x_max, T_max, Nx_audit, Nt_audit)

            # PI-DeepOnet
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

            #Error
            err_norm = np.linalg.norm(U_true_np - U_pred_np)
            true_norm = np.linalg.norm(U_true_np)
            rel_error = err_norm / (true_norm + 1e-6)
            errors.append(rel_error)

        avg_error = np.mean(errors)
        if avg_error > max_error_global: max_error_global = avg_error

        status = "ok" if avg_error < threshold else "no"
        print(f"   -> Type {type_id}: Erreur = {avg_error:.2%} {status}")

        if avg_error >= threshold:
            failing_types.append(type_id)
            global_success = False

    return global_success, failing_types, max_error_global

#Training loop
def run_training_loop(model, optimizer, n_iters, batch_size, bounds, allowed_types=None):
    """
    Performs a comprehensive automated audit of the model's performance across different 
    physical scenarios.

    Args:
        model (nn.Module): 
            The trained DeepONet model to evaluate.
        threshold (float, default=0.03): 
            The maximum acceptable relative error (e.g., 0.03 for 3%). 
            If the average error for a type exceeds this, the test is considered failed.
        n_tests_per_type (int, default=5): 
            The number of random scenarios to generate and test for each equation type.

    Returns:
        tuple (global_success, failing_types, max_error_global):
            - global_success (bool): True if all types passed the threshold, False otherwise.
            - failing_types (list): List of type IDs that exceeded the error threshold.
            - max_error_global (float): The highest average error observed among all tested types.
    """
    model.train()
    pbar = tqdm(range(n_iters), desc="Training", leave=True)

    for i in pbar:
        optimizer.zero_grad()
        params, xt, xt_ic, u_true_ic = generate_mixed_batch(
            batch_size, bounds, -7, 7, 1.0, allowed_types=allowed_types
        )

        res = pde_residual_adr(model, params, xt)
        loss_pde = torch.mean(res**2)
        u_pred_ic = model(params, xt_ic)
        loss_ic = torch.mean((u_pred_ic - u_true_ic)**2)

        loss = loss_pde + 10.0 * loss_ic
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            pbar.set_postfix({"L_pde": f"{loss_pde.item():.2e}", "L_ic": f"{loss_ic.item():.2e}"})

#Training function
def train_smart_adptive(model, bounds, n_warmup=5000, n_physics=10000):
    """
    Orchestrates a multi-stage, adaptive training strategy to train the physics-informed model 
    robustly.

    Args:
        model (nn.Module): 
            The neural network to train.
        bounds (dict): 
            Physical parameter bounds used for data generation.
        n_warmup (int, default=5000): 
            Number of iterations for the "Warmup" phase (training on Initial Conditions only).
        n_physics (int, default=10000): 
            Number of iterations for the main "Physics" phase (solving the PDE).

    Returns:
        model (nn.Module): 
            The fully trained and audited model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    BATCH_SIZE = 1024 

    print("Warm up = IC")
    for i in tqdm(range(n_warmup), desc="Warmup"):
        optimizer.zero_grad()
        params, xt, xt_ic, u_true_ic = generate_mixed_batch(BATCH_SIZE, bounds, -7, 7, 1.0)
        u_pred_ic = model(params, xt_ic)
        loss = torch.mean((u_pred_ic - u_true_ic)**2)
        loss.backward()
        optimizer.step()

    print("Physic training")
    run_training_loop(model, optimizer, n_physics, BATCH_SIZE, bounds)

    # small audit
    success, failing_types, _ = run_full_audit(model, n_tests_per_type=5)
    if success: return model

    strategies = [
        (2000, 1e-4, "Adam"), 
        (2000, 5e-5, "Adam"),
        (50,   1.0,  "LBFGS") 
    ]

    for n_iters, lr, algo in strategies:
        print(f"fine tuning: {algo} (lr={lr})")

        if algo == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            run_training_loop(model, optimizer, n_iters, BATCH_SIZE, bounds)
        else:
            optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=n_iters, history_size=50)
            def closure():
                optimizer.zero_grad()
                params, xt, xt_ic, u_true_ic = generate_mixed_batch(BATCH_SIZE, bounds, -7, 7, 1.0)
                loss_pde = torch.mean(pde_residual_adr(model, params, xt)**2)
                loss_ic = torch.mean((model(params, xt_ic) - u_true_ic)**2)
                loss = loss_pde + 10.0 * loss_ic
                loss.backward()
                return loss
            # We put a try/except block around the LBFGS step because it can sometimes crash.
            try:
                optimizer.step(closure)
            except Exception as e:
                print(f"LBFGS error: {e}. pass.")

        success, failing_types, _ = run_full_audit(model, n_tests_per_type=5)
        if success: return model

    # PHASE 3
    if len(failing_types) > 0 and len(failing_types) < 5:
        print(f"targeting on types: {failing_types}")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        run_training_loop(model, optimizer, 3000, BATCH_SIZE, bounds, allowed_types=failing_types)
        run_full_audit(model, n_tests_per_type=5)

    return model
