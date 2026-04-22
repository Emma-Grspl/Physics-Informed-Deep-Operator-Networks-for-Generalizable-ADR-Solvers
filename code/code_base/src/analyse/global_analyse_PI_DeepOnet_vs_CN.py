import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm 
import yaml
import os
import sys
from tqdm import tqdm
from pathlib import Path

file_path = os.path.abspath(__file__) 
project_root = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
repo_root = Path(project_root).parents[1]

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.CN_ADR import crank_nicolson_adr
from src.models.PI_DeepOnet_ADR import PI_DeepONet_ADR 
from src.data.generators import get_ic_value


def load_config(path=None):
    if path is None: 
        path = os.path.join(project_root, "configs", "config_ADR.yaml")
    with open(path, 'r') as f: 
        return yaml.safe_load(f)

def load_model(model_path, cfg, device):
    print(f"📥 Loading model : {model_path}")
    model = PI_DeepONet_ADR(cfg).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint: 
        model.load_state_dict(checkpoint['model_state_dict'])
    else: 
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def predict_all(model, p_dict, cfg, t_max, device):
    """Generates U_cn and U_don for a given set of parameters."""
    Nx = cfg['geometry'].get('Nx', 400)
    Nt = cfg['geometry'].get('Nt', 200)
    x_min = cfg['geometry'].get('x_min', -5.0)
    x_max = cfg['geometry'].get('x_max', 8.0)
    
    x = np.linspace(x_min, x_max, Nx)
    t = np.linspace(0, t_max, Nt)
    
    u0 = get_ic_value(x, "mixed", p_dict)
    bc_kind = "zero_zero" if p_dict['type'] in [1, 3] else "tanh_pm1"
    
    # 1. Crank-Nicolson (Référence numérique)
    _, U_cn, _ = crank_nicolson_adr(
        v=p_dict['v'], D=p_dict['D'], mu=p_dict['mu'],
        xL=x_min, xR=x_max, Nx=Nx, Tmax=t_max, Nt=Nt,
        bc_kind=bc_kind, x0=x, u0=u0
    )
    if U_cn.shape == (Nt, Nx): 
        U_cn = U_cn.T
        
    # 2. DeepONet
    X_grid, T_grid = np.meshgrid(x, t, indexing='ij')
    x_flat, t_flat = X_grid.flatten(), T_grid.flatten()
    p_vec = np.array([p_dict['v'], p_dict['D'], p_dict['mu'], p_dict['type'], 
                      p_dict['A'], 0.0, p_dict['sigma'], p_dict['k']])
    p_tensor = torch.tensor(p_vec, dtype=torch.float32).repeat(len(x_flat), 1).to(device)
    xt_tensor = torch.tensor(np.stack([x_flat, t_flat], axis=1), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        u_pred_flat = model(p_tensor, xt_tensor).cpu().numpy().flatten()
    U_don = u_pred_flat.reshape(Nx, Nt)
    
    return x, t, U_cn, U_don


def generate_5_plots(x, t, results_dict, title_prefix, out_dir, types_names):
    """Generates the 5 comparison graphs."""
    os.makedirs(out_dir, exist_ok=True)
    target_types = list(results_dict.keys())
    colors = cm.RdPu(np.linspace(0.4, 0.8, len(target_types)))
    
    # 1. L2 Moyenne (Barres)
    plt.figure(figsize=(10, 6))
    means = [np.mean(results_dict[tid]['global_l2']) for tid in target_types]
    stds = [np.std(results_dict[tid]['global_l2']) for tid in target_types]
    names = [types_names[tid] for tid in target_types]
    
    bars = plt.bar(names, means, yerr=stds, capsize=10, alpha=0.8, color=colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2%}', ha='center', va='bottom', fontweight='bold')
    plt.title(f"Mean L2 Relative Error (Generalization) : {title_prefix}")
    plt.ylabel("Relative Error L2")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f"{out_dir}/1_Global_Mean_L2.png", dpi=300)
    plt.close()

    # 2. L2 Temporelle (Courbes)
    plt.figure(figsize=(10, 6))
    for idx, tid in enumerate(target_types):
        mean_temp_l2 = np.mean(results_dict[tid]['temporal_l2'], axis=0)
        plt.plot(t, mean_temp_l2, label=types_names[tid], color=colors[idx], lw=2)
    plt.title(f"L2 Error Over Time : {title_prefix}")
    plt.xlabel("Time (s)")
    plt.ylabel("Relative Error L2(t)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{out_dir}/2_Temporal_L2.png", dpi=300)
    plt.close()

    # 3. Heatmaps d'Erreur Absolue
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for idx, tid in enumerate(target_types):
        ax = axes[idx]
        mean_error_grid = np.mean(results_dict[tid]['error_grids'], axis=0)
        im = ax.pcolormesh(t, x, mean_error_grid, shading='gouraud', cmap='RdPu')
        ax.set_title(f"Error: {types_names[tid]}")
        ax.set_xlabel("Time (t)")
        if idx == 0: ax.set_ylabel("Space (x)")
        plt.colorbar(im, ax=ax)
    plt.savefig(f"{out_dir}/3_Error_Heatmaps.png", dpi=300)
    plt.close()

    # 4. Snapshots
    t_indices = [0, len(t)//4, len(t)//2, -1]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 12.8), sharex=True)
    for idx, tid in enumerate(target_types):
        ax = axes[idx]
        u_ref, u_pred = results_dict[tid]['last_u_ref'], results_dict[tid]['last_u_pred']
        ax.set_title(f"Snapshots : {types_names[tid]}", fontweight='bold')
        for i, t_idx in enumerate(t_indices):
            c = cm.RdPu(np.linspace(0.4, 0.9, len(t_indices)))[i]
            ax.plot(x, u_ref[:, t_idx], color=c, linestyle='--', alpha=0.95, lw=2.2)
            ax.plot(x, u_pred[:, t_idx], color=c, linestyle='-', lw=2)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Space (x)")
    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color='black', linestyle='--', lw=2.2, label='CN'),
        Line2D([0], [0], color='deepskyblue', linestyle='-', lw=2.0, label='PyTorch'),
    ]
    time_handles = [
        Line2D([0], [0], color=cm.RdPu(np.linspace(0.4, 0.9, len(t_indices)))[i], linestyle='-', lw=3, label=f"t={t[t_idx]:.2f}s")
        for i, t_idx in enumerate(t_indices)
    ]
    fig.subplots_adjust(top=0.93, right=0.80, hspace=0.28)
    fig.legend(handles=style_handles, loc='upper right', frameon=True, bbox_to_anchor=(0.985, 0.955), title="Line style")
    fig.legend(handles=time_handles, loc='upper right', frameon=True, bbox_to_anchor=(0.985, 0.80), title="Snapshot times")
    plt.savefig(f"{out_dir}/4_Snapshots.png", dpi=300)
    plt.close()

    # 5. Animation
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    lines = []
    for idx, tid in enumerate(target_types):
        ax = axes[idx]
        l1, = ax.plot([], [], 'k--', label='CN')
        l2, = ax.plot([], [], color='deepskyblue', label='PyTorch', lw=2)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(x[0], x[-1])
        ax.set_title(types_names[tid])
        if idx == 0: ax.legend()
        lines.append((l1, l2, results_dict[tid]['last_u_ref'], results_dict[tid]['last_u_pred']))

    def update(frame):
        artists = []
        for (l1, l2, u_r, u_p) in lines:
            l1.set_data(x, u_r[:, frame])
            l2.set_data(x, u_p[:, frame])
            artists.extend([l1, l2])
        return artists

    ani = animation.FuncAnimation(fig, update, frames=range(0, len(t), 2), blit=True)
    ani.save(f"{out_dir}/5_Animation.gif", writer='pillow', fps=20)
    plt.close()

# ==========================================
# 4. EXECUTION
# ==========================================

def run_analysis(model, cfg, device):
    target_types = [0, 1, 3]
    types_names = {0: "Tanh", 1: "Sin-Gauss", 3: "Gaussian"}
    Tmax = cfg['geometry']['T_max']
    
    out_dir = "outputs/PI_DeepOnet_analyse/DeepONet_vs_CN"
    
    print("Analysis: DeepONet vs Crank-Nicolson (1000 random samples)")
    res_cn = {i: {'global_l2': [], 'temporal_l2': [], 'error_grids': [], 'last_u_ref': None, 'last_u_pred': None} for i in target_types}
    
    for tid in target_types:
        for _ in tqdm(range(1000), desc=f"Évaluation {types_names[tid]}"):
            # Paramètres aléatoires basés sur les ranges du config
            p = {k: np.random.uniform(v[0], v[1]) for k, v in cfg['physics_ranges'].items()}
            p['type'] = tid
            
            x, t, u_cn, u_don = predict_all(model, p, cfg, Tmax, device)
            
            # Calcul des métriques
            l2_val = np.linalg.norm(u_cn - u_don) / (np.linalg.norm(u_cn) + 1e-8)
            res_cn[tid]['global_l2'].append(l2_val)
            
            res_cn[tid]['temporal_l2'].append(np.linalg.norm(u_cn - u_don, axis=0) / (np.linalg.norm(u_cn, axis=0) + 1e-8))
            res_cn[tid]['error_grids'].append(np.abs(u_cn - u_don))
            
            # Sauvegarde du dernier pour les visuels
            res_cn[tid]['last_u_ref'] = u_cn
            res_cn[tid]['last_u_pred'] = u_don
            
    generate_5_plots(x, t, res_cn, "DeepONet vs CN", out_dir, types_names)

if __name__ == "__main__":
    MODEL_PATH = str(repo_root / "models" / "base_pi_deeponet_reference.pth")
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    cfg = load_config()
    model = load_model(MODEL_PATH, cfg, device)
    
    run_analysis(model, cfg, device)
    print(f"Done. The graphs are in: outputs/PI_DeepOnet_analyse/DeepONet_vs_CN/")
