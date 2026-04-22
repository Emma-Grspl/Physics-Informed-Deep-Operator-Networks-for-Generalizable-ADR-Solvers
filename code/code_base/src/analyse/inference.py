import torch
import numpy as np
import matplotlib.pyplot as plt
import time
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
    if path is None: path = os.path.join(project_root, "configs", "config_ADR.yaml")
    with open(path, 'r') as f: return yaml.safe_load(f)

def load_model(model_path, cfg, device):
    model = PI_DeepONet_ADR(cfg).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
    else: model.load_state_dict(checkpoint)
    model.eval()
    return model

def sync_device(device):
    if device.type == 'cuda': torch.cuda.synchronize()
    elif device.type == 'mps': torch.mps.synchronize()

def run_time_jump_benchmark(model, cfg, device, batch_size=50):    
    Nx = cfg['geometry'].get('Nx', 400)
    Nt = cfg['geometry'].get('Nt', 200)
    x_min = cfg['geometry'].get('x_min', -5.0)
    x_max = cfg['geometry'].get('x_max', 8.0)
    t_target = cfg['geometry']['T_max']
    
    x = np.linspace(x_min, x_max, Nx)
    
    # Générer les scénarios
    scenarios = []
    for _ in range(batch_size):
        p = {k: np.random.uniform(v[0], v[1]) for k, v in cfg['physics_ranges'].items()}
        p['type'] = np.random.choice([0, 1, 3])
        scenarios.append(p)

    # CN timer
    print("CN compute")
    t0_cn = time.perf_counter()
    for p in tqdm(scenarios, desc="CN"):
        u0 = get_ic_value(x, "mixed", p)
        bc_kind = "zero_zero" if p['type'] in [1, 3] else "tanh_pm1"
        _ = crank_nicolson_adr(
            v=p['v'], D=p['D'], mu=p['mu'],
            xL=x_min, xR=x_max, Nx=Nx, Tmax=t_target, Nt=Nt,
            bc_kind=bc_kind, x0=x, u0=u0
        )
    t1_cn = time.perf_counter()
    total_cn = t1_cn - t0_cn

    # DeepOnet timer
    print("DeepONet compute ")
    
    t_flat = np.full(Nx, t_target)
    xt_base = torch.tensor(np.stack([x, t_flat], axis=1), dtype=torch.float32).to(device)
    xt_batch = xt_base.repeat(batch_size, 1)

    p_list = []
    for p in scenarios:
        p_vec = [p['v'], p['D'], p['mu'], p['type'], p['A'], 0.0, p['sigma'], p['k']]
        p_list.append(np.tile(p_vec, (Nx, 1))) # On répète 400 fois, pas 80 000 !
    p_batch = torch.tensor(np.concatenate(p_list, axis=0), dtype=torch.float32).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(3): model(p_batch, xt_batch)
    sync_device(device)

    t0_don = time.perf_counter()
    with torch.no_grad():
        _ = model(p_batch, xt_batch)
    sync_device(device)
    t1_don = time.perf_counter()
    total_don = t1_don - t0_don

    return total_cn, total_don, batch_size

def run_full_reconstruction_benchmark(model, cfg, device, batch_size=10):
    Nx = cfg['geometry'].get('Nx', 400)
    Nt = cfg['geometry'].get('Nt', 200)
    x_min = cfg['geometry'].get('x_min', -5.0)
    x_max = cfg['geometry'].get('x_max', 8.0)
    t_target = cfg['geometry']['T_max']

    x = np.linspace(x_min, x_max, Nx)
    t = np.linspace(0.0, t_target, Nt)

    scenarios = []
    for _ in range(batch_size):
        p = {k: np.random.uniform(v[0], v[1]) for k, v in cfg['physics_ranges'].items()}
        p['type'] = np.random.choice([0, 1, 3])
        scenarios.append(p)

    print("CN full reconstruction compute")
    t0_cn = time.perf_counter()
    for p in tqdm(scenarios, desc="CN full"):
        u0 = get_ic_value(x, "mixed", p)
        bc_kind = "zero_zero" if p['type'] in [1, 3] else "tanh_pm1"
        _ = crank_nicolson_adr(
            v=p['v'], D=p['D'], mu=p['mu'],
            xL=x_min, xR=x_max, Nx=Nx, Tmax=t_target, Nt=Nt,
            bc_kind=bc_kind, x0=x, u0=u0
        )
    total_cn = time.perf_counter() - t0_cn

    print("PyTorch full reconstruction compute")
    X_grid, T_grid = np.meshgrid(x, t, indexing='ij')
    xt_grid = np.stack([X_grid.flatten(), T_grid.flatten()], axis=1)
    xt_batch = torch.tensor(np.tile(xt_grid, (batch_size, 1)), dtype=torch.float32).to(device)

    p_list = []
    for p in scenarios:
        p_vec = [p['v'], p['D'], p['mu'], p['type'], p['A'], 0.0, p['sigma'], p['k']]
        p_list.append(np.tile(p_vec, (Nx * Nt, 1)))
    p_batch = torch.tensor(np.concatenate(p_list, axis=0), dtype=torch.float32).to(device)

    with torch.no_grad():
        for _ in range(3):
            model(p_batch, xt_batch)
    sync_device(device)

    t0_don = time.perf_counter()
    with torch.no_grad():
        _ = model(p_batch, xt_batch)
    sync_device(device)
    total_don = time.perf_counter() - t0_don

    return total_cn, total_don, batch_size

def plot_speedup(total_cn, total_don, batch_size, out_dir, title, filename, mode_label):
    os.makedirs(out_dir, exist_ok=True)
    speedup = total_cn / total_don
    
    print(f"Results ({mode_label} for {batch_size} ICs) :")
    print(f"Crank-Nicolson (Step-by-step) : {total_cn:.3f} s")
    print(f"PyTorch                 : {total_don:.3f} s")
    print(f"Speedup                    : x{speedup:.1f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['CN', 'PyTorch']
    bars = ax.bar(labels, [total_cn, total_don], color=['black', 'deepskyblue'], alpha=0.85, edgecolor='black', width=0.6)
    
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + (max([total_cn, total_don])*0.02), 
                f'{bar.get_height():.3f} s', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
    ax.set_ylabel(f'Time for {batch_size} scenarios (s)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{filename}", dpi=300)

if __name__ == "__main__":
    MODEL_PATH = str(repo_root / "models" / "base_pi_deeponet_reference.pth")
    out_dir = os.path.join(project_root, "outputs", "PI_DeepOnet_analyse", "Inference_Time")
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    cfg = load_config()
    model = load_model(MODEL_PATH, cfg, device)
    
    t_cn_full, t_don_full, bs_full = run_full_reconstruction_benchmark(model, cfg, device, batch_size=10)
    plot_speedup(
        t_cn_full,
        t_don_full,
        bs_full,
        out_dir,
        title="PyTorch Full Reconstruction vs CN",
        filename="Inference_Speedup.png",
        mode_label="full reconstruction",
    )

    t_cn_jump, t_don_jump, bs_jump = run_time_jump_benchmark(model, cfg, device, batch_size=50)
    plot_speedup(
        t_cn_jump,
        t_don_jump,
        bs_jump,
        out_dir,
        title="PyTorch Time-Jumping vs CN",
        filename="Time_Jumping_Speedup.png",
        mode_label="time jump",
    )
