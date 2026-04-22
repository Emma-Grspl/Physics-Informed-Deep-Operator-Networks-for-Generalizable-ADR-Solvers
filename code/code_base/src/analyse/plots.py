"""
Allows you to generate time-series snapshots, heatmaps, and animations for each type of initial condition. To change the outputs, simply modify the values in the analysis_results dictionaries.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import torch
import yaml
from pathlib import Path

file_path = os.path.abspath(__file__) 
project_root = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
repo_root = Path(project_root).parents[1]
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.utils.CN_ADR import crank_nicolson_adr
    from src.data.generators import get_ic_value
except ImportError as e:
    print(f"Imports error : {e}")
    sys.exit(1)

def generate_solution(mode, model, physics, geom, ic_type, device='cpu'):
    x_min, x_max, T_max = geom['x_min'], geom['x_max'], geom['T_max']
    Nx, Nt = geom['Nx'], geom['Nt']
    x = np.linspace(x_min, x_max, Nx)
    t = np.linspace(0, T_max, Nt)
    
    if ic_type == "tanh":
        type_id, bc_kind = 0, "tanh_pm1"
        ic_params = {"type": 0, "A": physics['A'], "x0": physics['x0'], "sigma": physics['sigma'], "k": physics['k']}
    elif ic_type == "sin_gauss":
        type_id, bc_kind = 1, "zero_zero"
        ic_params = {"type": 1, "A": physics['A'], "x0": physics['x0'], "sigma": physics['sigma'], "k": physics['k']}
    elif ic_type == "gauss":
        type_id, bc_kind = 3, "zero_zero"
        ic_params = {"type": 3, "A": physics['A'], "x0": physics['x0'], "sigma": physics['sigma'], "k": physics['k']}
    
    u0 = get_ic_value(x, "mixed", ic_params)

    if mode == "CN":
        _, U_matrix, _ = crank_nicolson_adr(
            v=physics['v'], D=physics['D'], mu=physics['mu'],
            xL=x_min, xR=x_max, Nx=Nx, Tmax=T_max, Nt=Nt,
            bc_kind=bc_kind, x0=x, u0=u0
        )
        if U_matrix.shape == (Nx, Nt): U_matrix = U_matrix.T
            
    elif mode == "DeepONet":
        model.eval()
        X_grid, T_grid = np.meshgrid(x, t, indexing='ij')
        x_flat, t_flat = X_grid.flatten(), T_grid.flatten()
        p_vec = np.array([physics['v'], physics['D'], physics['mu'], type_id, physics['A'], physics['x0'], physics['sigma'], physics['k']])
        p_tensor = torch.tensor(p_vec, dtype=torch.float32).repeat(len(x_flat), 1).to(device)
        xt_tensor = torch.tensor(np.stack([x_flat, t_flat], axis=1), dtype=torch.float32).to(device)
        with torch.no_grad():
            u_pred_flat = model(p_tensor, xt_tensor).cpu().numpy().flatten()
        U_matrix = u_pred_flat.reshape(Nx, Nt).T 

    return x, t, U_matrix

def analyse_results(mode="CN", model=None, device='cpu'):
    if mode == "CN":
        out_dir = "outputs/classical_solver"
        curve_color = "black"
    else:
        out_dir = "outputs/PI_DeepOnet_analyse/plot"
        curve_color = "deepskyblue"
    os.makedirs(out_dir, exist_ok=True)

    geom = {'x_min': -5.0, 'x_max': 8.0, 'T_max': 3.0, 'Nx': 400, 'Nt': 200}
    physics = {'v': 0.8, 'D': 0.1, 'mu': 0.5, 'A': 1.0, 'x0': 0.0, 'sigma': 0.5, 'k': 2.0}
    ic_types = ["tanh", "gauss", "sin_gauss"]

    phys_str = fr"$\nu$={physics['v']}, $D$={physics['D']}, $\mu$={physics['mu']}, $A$={physics['A']}, $\sigma$={physics['sigma']}, $k$={physics['k']}"

    print(f"Data generation {mode}")
    data = {}
    for ic in ic_types:
        x, t, U = generate_solution(mode, model, physics, geom, ic, device)
        data[ic] = U

    Nt = geom['Nt']
    t_indices = [0, Nt//6, Nt//4, Nt//2, Nt-1]
    colors_snapshots = cm.RdPu(np.linspace(0.3, 1.0, len(t_indices)))

    #Temporal Snapshot
    print("Snapshots")
    fig, axes = plt.subplots(3, 1, figsize=(10, 13), constrained_layout=True)
    for i, ic in enumerate(ic_types):
        ax = axes[i]
        for idx, t_idx in enumerate(t_indices):
            ax.plot(x, data[ic][t_idx, :], label=f"t={t[t_idx]:.2f}s", color=colors_snapshots[idx], lw=2)
        ax.set_title(f"Snapshots : {ic.capitalize()}\n{phys_str}", fontsize=11)
        ax.set_ylabel("u(x,t)")
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper right', fontsize='small', frameon=True)
    
    axes[-1].set_xlabel("Space (x)")
    plt.savefig(f"{out_dir}/{mode}_snapshots.png", dpi=300)
    plt.close()

    #Heatmaps x = f(t)
    print("Heatmaps generation")
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
    for i, ic in enumerate(ic_types):
        ax = axes[i]
        # On affiche t en X et x en Y
        im = ax.pcolormesh(t, x, data[ic].T, shading='gouraud', cmap='RdPu')
        ax.set_title(f"Evolution : {ic.capitalize()}\n{phys_str}", fontsize=11)
        ax.set_xlabel("Time (t)")
        if i == 0: ax.set_ylabel("Space (x)")
        plt.colorbar(im, ax=ax, shrink=0.8)
        
    fig.suptitle(f"Spatiotemporal heatmaps : {mode}", fontsize=16, fontweight='bold')
    plt.savefig(f"{out_dir}/{mode}_heatmaps.png", dpi=300)
    plt.close()

    #Animation (gif)
    print("Animation generation")
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
    lines = []
    for i, ic in enumerate(ic_types):
        ax = axes[i]
        ax.set_xlim(geom['x_min'], geom['x_max'])
        ax.set_ylim(np.min(data[ic])-0.2, np.max(data[ic])+0.2)
        ax.set_title(f"Animation : {ic.capitalize()}", fontsize=12)
        ax.grid(True, alpha=0.2)
        # Ligne fuchsia pour l'animation
        line, = ax.plot([], [], lw=2.5, color=curve_color)
        lines.append(line)

    def update(frame):
        for i, ic in enumerate(ic_types):
            lines[i].set_data(x, data[ic][frame, :])
        fig.suptitle(f"ADR ({mode}) | Time : {t[frame]:.2f}s\n{phys_str}", fontsize=13)
        return lines

    anim = animation.FuncAnimation(fig, update, frames=Nt, interval=40, blit=True)
    anim.save(f"{out_dir}/{mode}_evolution.gif", writer='pillow', fps=25)
    plt.close()

    print(f"Output folder : {out_dir}")

if __name__ == "__main__":
    import yaml
    
    #Classical Soler
    #analyse_results(mode="CN")
        
    #DeepOnet
    config_path = os.path.join(project_root, "configs", "config_ADR.yaml")
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Err : {config_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    try:
        from src.models.PI_DeepOnet_ADR import PI_DeepONet_ADR 
    except ImportError as e:
        print(f"Err : {e}")
        sys.exit(1)

    model = PI_DeepONet_ADR(cfg).to(device)
    model_path = str(repo_root / "models" / "base_pi_deeponet_reference.pth")
    
    if os.path.exists(model_path):
        print(f"Loading weights : {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        print("Everything ok")
    else:
        print(f"Err : {model_path}")
        sys.exit(1)

    analyse_results(mode="DeepONet", model=model, device=device)
