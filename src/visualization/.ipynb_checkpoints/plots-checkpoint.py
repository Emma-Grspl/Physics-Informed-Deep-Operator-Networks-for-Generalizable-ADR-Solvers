import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from tqdm import tqdm
from src.data.generators import get_ic_value, generate_mixed_batch
from src.physics.solver import crank_nicolson_adr

#device detection
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

#comparison snapshot
def compare_model_vs_cn_snapshots(model, x_min, x_max, Tmax, save_dir="results"):
    """
    Visually compare PI-DeepONet vs Crank-Nicolson and save the image.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    times_to_plot = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = ['lightpink', 'fuchsia', 'mediumvioletred', 'crimson', 'darkmagenta']

    test_cases = [
        {
            "name": "Type 1: Tanh Shock",
            "params": {"v": 1.0, "D": 0.05, "mu": 1.0},
            "ic": {"type": 0, "A": 1.0, "x0": 0.0, "sigma": 0.4, "k": 0.0},
            "bc": "tanh_pm1"
        },
        {
            "name": "Type 2: Wave Packet",
            "params": {"v": 1.5, "D": 0.01, "mu": 0.0},
            "ic": {"type": 1, "A": 1.2, "x0": 0.0, "sigma": 0.8, "k": 2.5},
            "bc": "periodic"
        },
        {
            "name": "Type 3: Diffusion",
            "params": {"v": 0.2, "D": 0.15, "mu": 0.0},
            "ic": {"type": 3, "A": 1.5, "x0": 0.0, "sigma": 0.5, "k": 0.0},
            "bc": "zero_zero"
        }
    ]

    Nx_ref = 3000
    Nt_ref = 1000
    x_ref = np.linspace(x_min, x_max, Nx_ref)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    for i, case in enumerate(test_cases):
        ax = axes[i]
        p = case["params"]
        ic = case["ic"]

        #ground truth
        u0_cn = get_ic_value(x_ref, "mixed", {"type": np.array([ic["type"]]), **ic})
        _, U_cn, t_cn = crank_nicolson_adr(
            p["v"], p["D"], p["mu"], x_min, x_max, Nx_ref, Tmax, Nt_ref,
            case["bc"], u0=u0_cn, x0=x_ref)

        #PI-deepOnet
        ones = torch.ones(Nx_ref, 1).to(DEVICE)
        params_vec = torch.cat([
            ones * p["v"], ones * p["D"], ones * p["mu"],
            ones * float(ic["type"]), ones * ic["A"], ones * ic["x0"], 
            ones * ic["sigma"], ones * ic["k"]
        ], dim=1)

        x_torch = torch.tensor(x_ref, dtype=torch.float32).view(-1, 1).to(DEVICE)

        for t_idx, t_val in enumerate(times_to_plot):
            # CN Curve
            idx_cn = (np.abs(t_cn - t_val)).argmin()
            curve_cn = U_cn[idx_cn, :]

            # PI-DeepONet Curve
            t_torch = torch.ones_like(x_torch) * t_val
            xt_input = torch.cat([x_torch, t_torch], dim=1)

            with torch.no_grad():
                u_pred = model(params_vec, xt_input).cpu().numpy().flatten()

            # Plot
            label_cn = f"CN t={t_val}" if i == 0 else None
            label_nn = f"DeepONet t={t_val}" if i == 0 else None

            ax.plot(x_ref, curve_cn, linestyle='--', linewidth=1.5, alpha=0.6, color=colors[t_idx], label=label_cn)
            ax.plot(x_ref, u_pred, linestyle='-', linewidth=2.0, color=colors[t_idx], label=label_nn)

        ax.set_title(case["name"], fontsize=11, fontweight='bold')
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("u(x,t)")
            custom_lines = [Line2D([0], [0], color=c, lw=2) for c in colors]
            ax.legend(custom_lines, [f"t={t}" for t in times_to_plot], title="Temps", loc='upper right', fontsize='small')
            ax.text(0.05, 0.95, "Plain: DeepONet\nDotted: CN", 
                    transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

    plt.suptitle("DeepONet vs Crank-Nicolson comparison", fontsize=16, y=1.02)
    plt.tight_layout()

    # SAUVEGARDE
    filename = os.path.join(save_dir, "comparison_snapshots.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close() # Libère la mémoire
    print(f"✅ Figure sauvegardée : {filename}")


# --- 2. ANIMATION (Sauvegarde GIF/MP4) ---

def animate_model_vs_cn(model, x_min, x_max, Tmax, save_dir="results"):
    """
    Generates a GIF animation and saves it.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    test_cases = [
        {"name": "Choc (Tanh)", "params": {"v": 1.0, "D": 0.05, "mu": 1.0}, "ic": {"type": 0, "A": 1.0, "x0": 0.0, "sigma": 0.4, "k": 0.0}, "bc": "tanh_pm1"},
        {"name": "Ondes (Sinus)", "params": {"v": 1.5, "D": 0.01, "mu": 0.0}, "ic": {"type": 1, "A": 1.2, "x0": 0.0, "sigma": 0.8, "k": 2.5}, "bc": "periodic"},
        {"name": "Diff (Gauss)", "params": {"v": 0.2, "D": 0.15, "mu": 0.0}, "ic": {"type": 3, "A": 1.5, "x0": 0.0, "sigma": 0.5, "k": 0.0}, "bc": "zero_zero"}
    ]

    Nx_ref = 300
    Nt_ref = 100 
    x_ref = np.linspace(x_min, x_max, Nx_ref)
    data_store = []
    for i, case in enumerate(test_cases):
        p, ic = case["params"], case["ic"]

        # CN
        u0_cn = get_ic_value(x_ref, "mixed", {"type": np.array([ic["type"]]), **ic})
        _, U_cn, t_cn = crank_nicolson_adr(
            p["v"], p["D"], p["mu"], x_min, x_max, Nx_ref, Tmax, Nt_ref,
            case["bc"], u0=u0_cn, x0=x_ref
        )

        # DeepONet
        X_mesh, T_mesh = np.meshgrid(x_ref, t_cn)
        xt_flat = torch.tensor(np.hstack((X_mesh.flatten()[:, None], T_mesh.flatten()[:, None])), dtype=torch.float32).to(DEVICE)

        N_total = len(xt_flat)
        ones = torch.ones(N_total, 1).to(DEVICE)
        params_vec = torch.cat([
            ones * p["v"], ones * p["D"], ones * p["mu"],
            ones * float(ic["type"]), ones * ic["A"], ones * ic["x0"], 
            ones * ic["sigma"], ones * ic["k"]
        ], dim=1)

        with torch.no_grad():
            U_pred = model(params_vec, xt_flat).cpu().numpy().reshape(Nt_ref, Nx_ref)

        data_store.append({"U_cn": U_cn, "U_pred": U_pred, "title": case["name"], "t": t_cn})

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    lines_cn, lines_nn, time_texts = [], [], []
    for i, ax in enumerate(axes):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-1.6, 1.6)
        ax.set_title(data_store[i]["title"])
        ax.grid(True, alpha=0.3)
        ln1, = ax.plot([], [], 'k--', linewidth=1.5, alpha=0.6, label='CN')
        ln2, = ax.plot([], [], 'r-', linewidth=2.0, label='DeepONet')
        txt = ax.text(0.05, 0.90, '', transform=ax.transAxes)
        lines_cn.append(ln1); lines_nn.append(ln2); time_texts.append(txt)
        if i == 0: ax.legend()

    def update(frame):
        for i in range(3):
            data = data_store[i]
            lines_cn[i].set_data(x_ref, data["U_cn"][frame, :])
            lines_nn[i].set_data(x_ref, data["U_pred"][frame, :])
            time_texts[i].set_text(f"t={data['t'][frame]:.2f}")
        return lines_cn + lines_nn + time_texts

    anim = animation.FuncAnimation(fig, update, frames=Nt_ref, interval=50, blit=True)

    #saving
    save_path = os.path.join(save_dir, "animation_results.gif")
    anim.save(save_path, writer='pillow', fps=20)
    plt.close()
    print(f"✅ Animation sauvegardée : {save_path}")


def plot_error_heatmaps(model, x_min, x_max, Tmax, save_dir="results"):
    """
    Generates error heatmaps.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    test_cases = [
        {"name": "Cas 1: Choc", "params": {"v": 1.0, "D": 0.05, "mu": 1.0}, "ic": {"type": 0, "A": 1.0, "x0": 0.0, "sigma": 0.4, "k": 0.0}, "bc": "tanh_pm1"},
        {"name": "Cas 2: Ondes", "params": {"v": 1.5, "D": 0.01, "mu": 0.0}, "ic": {"type": 1, "A": 1.2, "x0": 0.0, "sigma": 0.8, "k": 2.5}, "bc": "periodic"},
        {"name": "Cas 3: Diffusion", "params": {"v": 0.2, "D": 0.15, "mu": 0.0}, "ic": {"type": 3, "A": 1.5, "x0": 0.0, "sigma": 0.5, "k": 0.0}, "bc": "zero_zero"}
    ]

    Nx_vis, Nt_vis = 200, 100
    x_vis = np.linspace(x_min, x_max, Nx_vis)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)

    for i, case in enumerate(test_cases):
        p, ic = case["params"], case["ic"]

        # Ground Truth
        u0_cn = get_ic_value(x_vis, "mixed", {"type": np.array([ic["type"]]), **ic})
        _, U_cn, t_vis = crank_nicolson_adr(
            p["v"], p["D"], p["mu"], x_min, x_max, Nx_vis, Tmax, Nt_vis,
            case["bc"], u0=u0_cn, x0=x_vis
        )

        # Prediction
        X_mesh, T_mesh = np.meshgrid(x_vis, t_vis)
        xt_flat = torch.tensor(np.hstack((X_mesh.flatten()[:, None], T_mesh.flatten()[:, None])), dtype=torch.float32).to(DEVICE)

        N_total = len(xt_flat)
        ones = torch.ones(N_total, 1).to(DEVICE)
        params_vec = torch.cat([
            ones * p["v"], ones * p["D"], ones * p["mu"],
            ones * float(ic["type"]), ones * ic["A"], ones * ic["x0"], 
            ones * ic["sigma"], ones * ic["k"]
        ], dim=1)

        with torch.no_grad():
            U_pred = model(params_vec, xt_flat).cpu().numpy().reshape(Nt_vis, Nx_vis)

        Abs_Error = np.abs(U_cn - U_pred)
        l2_error = np.linalg.norm(U_cn - U_pred) / np.linalg.norm(U_cn)

        # Plot Prediction
        ax_sol = axes[0, i]
        im1 = ax_sol.pcolormesh(X_mesh, T_mesh, U_pred, cmap='RdPu', shading='auto')
        ax_sol.set_title(f"{case['name']}\nPrediction", fontweight='bold')
        fig.colorbar(im1, ax=ax_sol)

        # Plot Error
        ax_err = axes[1, i]
        im2 = ax_err.pcolormesh(X_mesh, T_mesh, Abs_Error, cmap='inferno', shading='auto')
        ax_err.set_title(f"Erreur Absolue (L2: {l2_error:.2%})", color='darkred')
        fig.colorbar(im2, ax=ax_err)

    plt.tight_layout()
    filename = os.path.join(save_dir, "error_heatmaps.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Heatmaps sauvegardées : {filename}")


def evaluate_global_accuracy(model, n_tests, bounds_phy, x_min, x_max, Tmax, save_dir="results"):
    """
    Launch the statistical audit and save the CSV report.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    Nx_eval, Nt_eval = 200, 100
    x_eval = np.linspace(x_min, x_max, Nx_eval)

    results = []

    print(f"Launching the statistical audit on {n_tests} cases")

    #GPU
    params_batch, _, _, _ = generate_mixed_batch(n_tests, bounds_phy, x_min, x_max, Tmax, difficulty=1.0)

    for i in tqdm(range(n_tests)):
        try:
            #scalars
            p_v  = params_batch[i, 0].cpu().item()
            p_D  = params_batch[i, 1].cpu().item()
            p_mu = params_batch[i, 2].cpu().item()
            type_idx = int(params_batch[i, 3].cpu().item())

            p_A     = params_batch[i, 4].cpu().item()
            p_x0    = params_batch[i, 5].cpu().item()
            p_sigma = params_batch[i, 6].cpu().item()
            p_k     = params_batch[i, 7].cpu().item()

            #CSV names
            ic_type_str = ["Step", "Tanh", "Sinus", "Gauss", "Bell"][type_idx] if type_idx < 5 else "Unknown"

            #IC values
            ic_params_single = {
                "type": np.array([type_idx]), 
                "A": p_A, "x0": p_x0, "sigma": p_sigma, "k": p_k
            }

            #Ground truth
            u0_cn = get_ic_value(x_eval, "mixed", ic_params_single)
            if isinstance(u0_cn, torch.Tensor): 
                u0_cn = u0_cn.cpu().numpy()

            # Choix BC
            bc = "periodic" if type_idx in [1, 2] else ("tanh_pm1" if type_idx in [0, 1] else "zero_zero")
            if type_idx == 1: bc = "tanh_pm1" # Tanh
            if type_idx == 2: bc = "periodic" # Sinus

            _, U_cn, t_eval = crank_nicolson_adr(
                p_v, p_D, p_mu, x_min, x_max, Nx_eval, Tmax, Nt_eval,
                bc_kind=bc, u0=u0_cn, x0=x_eval
            )

            #PI-DeepOnet pred
            X_mesh, T_mesh = np.meshgrid(x_eval, t_eval)
            xt_flat = torch.tensor(np.hstack((X_mesh.flatten()[:, None], T_mesh.flatten()[:, None])), dtype=torch.float32).to(DEVICE)

            N_points = len(xt_flat)
            param_vec = params_batch[i:i+1, :].repeat(N_points, 1)

            with torch.no_grad():
                U_pred = model(param_vec, xt_flat).cpu().numpy().reshape(Nt_eval, Nx_eval)

            #Error
            norm_diff = np.linalg.norm(U_cn - U_pred)
            norm_true = np.linalg.norm(U_cn)
            rel_l2 = norm_diff / (norm_true + 1e-6)

            results.append({
                "Type": ic_type_str, 
                "L2_Error": rel_l2, 
                "v": p_v, "D": p_D
            })

        except Exception as e:
            print(f"Error on the case {i} (Type {type_idx}): {e}")
            continue

    df = pd.DataFrame(results)

    #if the dataframe is empty
    if df.empty:
        print("warning: No cases were successful. The CSV will be empty.")
    else:
        csv_path = os.path.join(save_dir, "global_audit_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Audit saved : {csv_path}")
        print(f"Mean error L2: {df['L2_Error'].mean():.2%}")

    return df


def audit_ic_only(model, n_samples=500, threshold=0.03):
    """
    Vérifie UNIQUEMENT la condition initiale (t=0).
    Teste 100 cas pour : Tanh, Sinus, Gauss.
    Retourne : (success: bool, max_error: float)
    """
    device = next(model.parameters()).device
    model.eval() # Mode évaluation (pas de gradients)
    
    # Les types que tu veux tester spécifiquement
    # ID des types selon ton générateur : 1=Tanh, 2=Sinus, 3=Gauss
    target_types = {1: "Tanh", 2: "Sinus", 3: "Gauss"}
    
    max_error_global = 0.0
    all_pass = True
    
    print(f"   🕵️‍♀️ Audit IC en cours ({n_samples} cas x 3 types)...")
    
    # Grille spatiale fixe pour l'audit (t=0 partout)
    x = np.linspace(-7, 7, 100)
    t = np.zeros_like(x) # t=0
    
    # Conversion tenseurs fixes
    X_flat = x[:, None]
    T_flat = t[:, None]
    xt_in = np.hstack((X_flat, T_flat))
    xt_tensor = torch.tensor(xt_in, dtype=torch.float32).to(device)

    # On boucle sur les 3 types
    for type_id, type_name in target_types.items():
        type_errors = []
        
        for _ in range(n_samples):
            # 1. Tirage aléatoire des paramètres
            v = np.random.uniform(0.5, 2.0)
            D = np.random.uniform(0.01, 0.2)
            mu = np.random.uniform(0.0, 1.0)
            
            # Paramètres spécifiques à la forme
            A = np.random.uniform(0.8, 1.2)
            x0 = np.random.uniform(-1, 1)
            sigma = np.random.uniform(0.4, 0.8)
            k = np.random.uniform(1.0, 3.0) # pour sinus
            
            # 2. Vérité Terrain (Formules mathématiques exactes à t=0)
            u_true = np.zeros_like(x)
            
            if type_id == 1: # Tanh (Choc)
                # u0(x) = 0.5*(1 - tanh((x-x0)/0.5)) environ, adapte selon ta formule exacte
                # Je reprends la logique standard d'un choc tanh
                u_true = 0.5 * (1 - np.tanh((x - x0) / 0.5)) 
                
            elif type_id == 2: # Sinus
                u_true = A * np.sin(k * x + x0)
                
            elif type_id == 3: # Gauss
                u_true = A * np.exp(-((x - x0)**2) / (2 * sigma**2))
            
            # 3. Prédiction Modèle
            # Vecteur paramètres répété pour chaque point x
            p_vec = np.array([v, D, mu, type_id, A, x0, sigma, k])
            p_tensor = torch.tensor(p_vec, dtype=torch.float32).unsqueeze(0).repeat(len(x), 1).to(device)
            
            with torch.no_grad():
                u_pred = model(p_tensor, xt_tensor).cpu().numpy().flatten()
            
            # 4. Erreur L2 Relative
            diff_norm = np.linalg.norm(u_true - u_pred)
            true_norm = np.linalg.norm(u_true)
            rel_err = diff_norm / (true_norm + 1e-6)
            type_errors.append(rel_err)

        # Moyenne pour ce type
        avg_type_error = np.mean(type_errors)
        if avg_type_error > max_error_global:
            max_error_global = avg_type_error
            
        status = "✅" if avg_type_error < threshold else "❌"
        print(f"      -> {type_name}: Erreur Moyenne = {avg_type_error:.2%} {status}")
        
        if avg_type_error >= threshold:
            all_pass = False

    return all_pass, max_error_global
