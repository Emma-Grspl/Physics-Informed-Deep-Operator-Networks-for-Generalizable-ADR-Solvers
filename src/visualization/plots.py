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

from config import Config
from src.data.generators import get_ic_value, generate_mixed_batch
from src.physics.solver import crank_nicolson_adr

# Device detection
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# --- 1. SNAPSHOTS STATIC ---
def compare_model_vs_cn_snapshots(model, x_min=None, x_max=None, Tmax=None, save_dir="results"):
    """
    Visually compare PI-DeepONet vs Crank-Nicolson and save the image.
    Utilise Config pour les valeurs par défaut.
    """
    if x_min is None: x_min = Config.x_min
    if x_max is None: x_max = Config.x_max
    if Tmax is None: Tmax = Config.T_max
    
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # On choisit 5 temps répartis sur [0, Tmax]
    times_to_plot = np.linspace(0, Tmax, 5)
    colors = ['lightpink', 'fuchsia', 'mediumvioletred', 'crimson', 'darkmagenta']

    # Cas de test canoniques (pour avoir des plots jolis et stables)
    test_cases = [
        {
            "name": "Type 0: Tanh Shock",
            "params": {"v": 1.0, "D": 0.05, "mu": 1.0},
            "ic": {"type": 0, "A": 1.0, "x0": 0.0, "sigma": 0.4, "k": 0.0},
            "bc": "tanh_pm1"
        },
        {
            "name": "Type 1: Wave Packet",
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

    # Résolution fine pour le plot (codée en dur pour la beauté du plot)
    Nx_ref = 200
    Nt_ref = 1000 
    x_ref = np.linspace(x_min, x_max, Nx_ref)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    for i, case in enumerate(test_cases):
        ax = axes[i]
        p = case["params"]
        ic = case["ic"]

        # 1. Ground Truth (CN)
        # Attention : ic["type"] doit être passé proprement à get_ic_value
        u0_cn = get_ic_value(x_ref, "mixed", {"type": np.array([ic["type"]]), **ic})
        
        _, U_cn, t_cn = crank_nicolson_adr(
            p["v"], p["D"], p["mu"], x_min, x_max, Nx_ref, Tmax, Nt_ref,
            case["bc"], u0=u0_cn, x0=x_ref
        )

        # 2. Prédiction PI-DeepONet
        ones = torch.ones(Nx_ref, 1).to(DEVICE)
        # Ordre : [v, D, mu, type, A, x0, sigma, k]
        params_vec = torch.cat([
            ones * p["v"], ones * p["D"], ones * p["mu"],
            ones * float(ic["type"]), ones * ic["A"], ones * ic["x0"], 
            ones * ic["sigma"], ones * ic["k"]
        ], dim=1)

        x_torch = torch.tensor(x_ref, dtype=torch.float32).view(-1, 1).to(DEVICE)

        for t_idx, t_val in enumerate(times_to_plot):
            # CN Curve (plus proche voisin temporel)
            idx_cn = (np.abs(t_cn - t_val)).argmin()
            curve_cn = U_cn[idx_cn, :]

            # NN Curve
            t_torch = torch.ones_like(x_torch) * t_val
            xt_input = torch.cat([x_torch, t_torch], dim=1)

            with torch.no_grad():
                u_pred = model(params_vec, xt_input).cpu().numpy().flatten()

            # Plot
            label_cn = f"CN t={t_val:.1f}" if i == 0 else None
            label_nn = f"DeepONet t={t_val:.1f}" if i == 0 else None

            ax.plot(x_ref, curve_cn, linestyle='--', linewidth=1.5, alpha=0.6, color=colors[t_idx], label=label_cn)
            ax.plot(x_ref, u_pred, linestyle='-', linewidth=2.0, color=colors[t_idx], label=label_nn)

        ax.set_title(case["name"], fontsize=11, fontweight='bold')
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("u(x,t)")
            custom_lines = [Line2D([0], [0], color=c, lw=2) for c in colors]
            ax.legend(custom_lines, [f"t={t:.1f}" for t in times_to_plot], title="Temps", loc='upper right', fontsize='small')
            ax.text(0.05, 0.95, "Plain: DeepONet\nDotted: CN", 
                    transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

    plt.suptitle("DeepONet vs Crank-Nicolson comparison", fontsize=16, y=1.02)
    plt.tight_layout()

    filename = os.path.join(save_dir, "comparison_snapshots.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Figure sauvegardée : {filename}")


# --- 2. ANIMATION (GIF) ---
def animate_model_vs_cn(model, x_min=None, x_max=None, Tmax=None, save_dir="results"):
    if x_min is None: x_min = Config.x_min
    if x_max is None: x_max = Config.x_max
    if Tmax is None: Tmax = Config.T_max
    
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    test_cases = [
        {"name": "Choc (Tanh)", "params": {"v": 1.0, "D": 0.05, "mu": 1.0}, "ic": {"type": 0, "A": 1.0, "x0": 0.0, "sigma": 0.4, "k": 0.0}, "bc": "tanh_pm1"},
        {"name": "Ondes (Sinus)", "params": {"v": 1.5, "D": 0.01, "mu": 0.0}, "ic": {"type": 1, "A": 1.2, "x0": 0.0, "sigma": 0.8, "k": 2.5}, "bc": "periodic"},
        {"name": "Diff (Gauss)", "params": {"v": 0.2, "D": 0.15, "mu": 0.0}, "ic": {"type": 3, "A": 1.5, "x0": 0.0, "sigma": 0.5, "k": 0.0}, "bc": "zero_zero"}
    ]

    Nx_ref = 200
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
        # On peut sauter des frames si c'est trop lent
        # frame_idx = frame * 2 if frame * 2 < Nt_ref else Nt_ref - 1
        frame_idx = frame
        
        updated_artists = []
        for i in range(3):
            data = data_store[i]
            lines_cn[i].set_data(x_ref, data["U_cn"][frame_idx, :])
            lines_nn[i].set_data(x_ref, data["U_pred"][frame_idx, :])
            time_texts[i].set_text(f"t={data['t'][frame_idx]:.2f}")
            updated_artists.extend([lines_cn[i], lines_nn[i], time_texts[i]])
        return updated_artists

    anim = animation.FuncAnimation(fig, update, frames=Nt_ref, interval=50, blit=True)

    save_path = os.path.join(save_dir, "animation_results.gif")
    # Utilisation de Pillow pour éviter les dépendances ffmpeg parfois pénibles
    anim.save(save_path, writer='pillow', fps=20)
    plt.close()
    print(f"✅ Animation sauvegardée : {save_path}")


# --- 3. AUDIT STATISTIQUE GLOBAL ---
def evaluate_global_accuracy(model, n_tests, bounds_phy, x_min=None, x_max=None, Tmax=None, save_dir="results"):
    """
    Launch the statistical audit and save the CSV report.
    """
    if x_min is None: x_min = Config.x_min
    if x_max is None: x_max = Config.x_max
    if Tmax is None: Tmax = Config.T_max

    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    Nx_eval, Nt_eval = 200, 100
    x_eval = np.linspace(x_min, x_max, Nx_eval)

    results = []
    print(f"Launching the statistical audit on {n_tests} cases")

    # Génération d'un gros batch de paramètres pour l'audit
    # On force Tmax à être celui demandé (pour générer des temps valides)
    params_batch, _, _, _ = generate_mixed_batch(n_tests, bounds_phy, x_min, x_max, Tmax)

    for i in tqdm(range(n_tests)):
        try:
            # Extraction scalaires
            p_v  = params_batch[i, 0].cpu().item()
            p_D  = params_batch[i, 1].cpu().item()
            p_mu = params_batch[i, 2].cpu().item()
            type_idx = int(params_batch[i, 3].cpu().item())

            p_A     = params_batch[i, 4].cpu().item()
            p_x0    = params_batch[i, 5].cpu().item()
            p_sigma = params_batch[i, 6].cpu().item()
            p_k     = params_batch[i, 7].cpu().item()

            # Noms pour le CSV
            ic_type_str = ["Tanh", "GaussSin", "Sin", "Gauss", "Bell"][type_idx] if type_idx < 5 else "Unknown"

            # IC Params
            ic_params_single = {
                "type": np.array([type_idx]), 
                "A": p_A, "x0": p_x0, "sigma": p_sigma, "k": p_k
            }

            # Ground truth via Solver
            u0_cn = get_ic_value(x_eval, "mixed", ic_params_single)
            if isinstance(u0_cn, torch.Tensor): 
                u0_cn = u0_cn.cpu().numpy()

            # Choix BC (Simplifié selon le type comme souvent fait en ADR 1D)
            # Tanh (0) -> Choc -> Tanh BC
            # Sinus (1, 2) -> Périodique
            # Gauss (3, 4) -> Zero (Dirichlet)
            if type_idx == 0: bc = "tanh_pm1"
            elif type_idx in [1, 2]: bc = "periodic"
            else: bc = "zero_zero"

            _, U_cn, t_eval = crank_nicolson_adr(
                p_v, p_D, p_mu, x_min, x_max, Nx_eval, Tmax, Nt_eval,
                bc_kind=bc, u0=u0_cn, x0=x_eval
            )

            # PI-DeepOnet Prediction
            X_mesh, T_mesh = np.meshgrid(x_eval, t_eval)
            xt_flat = torch.tensor(np.hstack((X_mesh.flatten()[:, None], T_mesh.flatten()[:, None])), dtype=torch.float32).to(DEVICE)

            N_points = len(xt_flat)
            # On répète les params de ce cas i pour tous les points de la grille spatio-temporelle
            param_vec = params_batch[i:i+1, :].repeat(N_points, 1)

            with torch.no_grad():
                U_pred = model(param_vec, xt_flat).cpu().numpy().reshape(Nt_eval, Nx_eval)

            # Error metrics
            norm_diff = np.linalg.norm(U_cn - U_pred)
            norm_true = np.linalg.norm(U_cn)
            rel_l2 = norm_diff / (norm_true + 1e-6)

            results.append({
                "Type": ic_type_str, 
                "L2_Error": rel_l2, 
                "v": p_v, "D": p_D
            })

        except Exception as e:
            # En cas de divergence solveur ou erreur
            # print(f"Error on case {i}: {e}")
            continue

    df = pd.DataFrame(results)

    if df.empty:
        print("Warning: No cases were successful. CSV empty.")
    else:
        csv_path = os.path.join(save_dir, "global_audit_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Audit saved : {csv_path}")
        print(f"Mean error L2: {df['L2_Error'].mean():.2%}")

    return df


# --- 4. AUDIT IC SEULEMENT (t=0) ---
def audit_ic_only(model, n_samples=500, threshold=0.03):
    """
    Vérifie UNIQUEMENT la condition initiale (t=0).
    Utilise Config.get_p_dict() pour générer des cas cohérents.
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Récupération Config
    x_min, x_max = Config.x_min, Config.x_max
    
    # Grille spatiale fixe pour l'audit
    x = np.linspace(x_min, x_max, 200)
    t = np.zeros_like(x) # t=0
    
    # Tenseurs spatio-temporels fixes (t=0)
    X_flat = x[:, None]
    T_flat = t[:, None]
    xt_in = np.hstack((X_flat, T_flat))
    xt_tensor = torch.tensor(xt_in, dtype=torch.float32).to(device)

    errors = []
    
    print(f"   🕵️‍♀️ Audit IC en cours ({n_samples} samples)...")

    for _ in range(n_samples):
        # 1. Tirage via Config (assure qu'on teste ce qu'on a appris)
        p_dict = Config.get_p_dict()
        
        # 2. Vérité Terrain via get_ic_value (cohérence code)
        # On passe x sous forme numpy
        u_true = get_ic_value(x, "mixed", p_dict)
        
        # 3. Prédiction Modèle
        # Construction vecteur [v, D, mu, type, A, x0, sigma, k]
        p_vec_np = np.array([
            p_dict['v'], p_dict['D'], p_dict['mu'], p_dict['type'], 
            p_dict['A'], p_dict['x0'], p_dict['sigma'], p_dict['k']
        ])
        
        # Répétition pour chaque point de x
        p_tensor = torch.tensor(p_vec_np, dtype=torch.float32).unsqueeze(0).repeat(len(x), 1).to(device)
        
        with torch.no_grad():
            u_pred = model(p_tensor, xt_tensor).cpu().numpy().flatten()
        
        # 4. Erreur
        diff_norm = np.linalg.norm(u_true - u_pred)
        true_norm = np.linalg.norm(u_true)
        # Protection division
        if true_norm < 1e-6: true_norm = 1e-6
            
        rel_err = diff_norm / true_norm
        errors.append(rel_err)

    mean_err = np.mean(errors)
    status = "✅" if mean_err < threshold else "❌"
    print(f"      -> IC Mean Error: {mean_err:.2%} {status}")
    
    return (mean_err < threshold), mean_err

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