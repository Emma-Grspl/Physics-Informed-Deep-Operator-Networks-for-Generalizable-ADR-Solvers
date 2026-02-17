import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm 
from matplotlib.gridspec import GridSpec
import yaml
import os
import sys
from tqdm import tqdm

# --- IMPORTS ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.physics.solver import get_ground_truth_CN
from src.models.adr import PI_DeepONet_ADR 

# =============================================================================
# 1. CHARGEMENT
# =============================================================================

def load_config(path="src/config_ADR.yaml"):
    with open(path, 'r') as f: return yaml.safe_load(f)

def load_model(model_path, cfg, device):
    print(f"📥 Chargement du modèle depuis : {model_path}")
    model = PI_DeepONet_ADR(cfg).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model

def predict_case(model, p_dict, cfg, t_max, device):
    # Appel Solveur
    X, T, U_true = get_ground_truth_CN(p_dict, cfg, t_step_max=t_max)
    
    # Préparation DeepONet
    x_flat = X.flatten()
    t_flat = T.flatten()
    
    p_vec = np.array([p_dict['v'], p_dict['D'], p_dict['mu'], p_dict['type'], 
                      p_dict['A'], 0.0, p_dict['sigma'], p_dict['k']])
    
    p_tensor = torch.tensor(p_vec, dtype=torch.float32).repeat(len(x_flat), 1).to(device)
    xt_tensor = torch.tensor(np.stack([x_flat, t_flat], axis=1), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        u_pred_flat = model(p_tensor, xt_tensor).cpu().numpy().flatten()
        
    # Reconstruction
    U_pred = u_pred_flat.reshape(X.shape)
    t_vec = T[0, :] 
    x_vec = X[:, 0]
    return x_vec, t_vec, U_true, U_pred

# =============================================================================
# 2. ANALYSES
# =============================================================================

def run_analysis(model, cfg, device):
    # SELECTION DES 3 TYPES
    target_types = [0, 1, 3]
    types_names = {
        0: "Tanh (Choc)", 
        1: "Sin-Gauss (Hautes Fréquences)", 
        3: "Gaussian (Diffusion)"
    }
    
    os.makedirs("outputs", exist_ok=True)
    
    # --- A. STATISTIQUES MASSIVES ---
    print("📊 Calcul des statistiques (1000 samples par type = 3000 total)...")
    temporal_errs = {i: [] for i in target_types}
    global_l2 = {i: [] for i in target_types}
    ref_time_grid = None
    
    # On boucle explictement sur chaque type pour en avoir 1000 pile
    for tid in target_types:
        print(f"   👉 Traitement du type : {types_names[tid]}")
        for _ in tqdm(range(1000)):
            p = {k: np.random.uniform(v[0], v[1]) for k, v in cfg['physics_ranges'].items()}
            p['type'] = tid
            
            try:
                x, t, u_true, u_pred = predict_case(model, p, cfg, 3.0, device)
                if ref_time_grid is None: ref_time_grid = t

                # Erreur L2 Globale
                num = np.linalg.norm(u_true - u_pred)
                den = np.linalg.norm(u_true) + 1e-8
                global_l2[tid].append(num / den)
                
                # Erreur L2 Temporelle
                norm_diff_t = np.linalg.norm(u_true - u_pred, axis=0)
                norm_true_t = np.linalg.norm(u_true, axis=0) + 1e-8
                temporal_errs[tid].append(norm_diff_t / norm_true_t)
                
            except Exception: continue

    # Plot 1 : Barres (3 types)
    plt.figure(figsize=(10, 6))
    means = [np.mean(global_l2[i]) if global_l2[i] else 0 for i in target_types]
    stds = [np.std(global_l2[i]) if global_l2[i] else 0 for i in target_types]
    names_list = [types_names[i] for i in target_types]
    
    colors = ['#d62728', '#ff7f0e', '#1f77b4'] # Rouge, Orange, Bleu
    bars = plt.bar(names_list, means, yerr=stds, capsize=10, alpha=0.8, color=colors)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.title("Erreur Relative L2 Moyenne (sur 3000 simulations)")
    plt.ylabel("Erreur Relative")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("outputs/1_Stats_Globales.png", dpi=300)
    plt.close()

    # Plot 2 : Temps
    plt.figure(figsize=(10, 6))
    for idx, tid in enumerate(target_types):
        if len(temporal_errs[tid]) > 0:
            arr = np.array(temporal_errs[tid])
            min_len = min(arr.shape[1], len(ref_time_grid))
            # Moyenne sur les 1000 courbes
            mean_curve = np.mean(arr[:, :min_len], axis=0)
            # Intervalle de confiance (écart-type / 2 pour la lisibilité ou std complet)
            std_curve = np.std(arr[:, :min_len], axis=0)
            
            t_axis = ref_time_grid[:min_len]
            plt.plot(t_axis, mean_curve, label=types_names[tid], linewidth=2, color=colors[idx])
            plt.fill_between(t_axis, mean_curve - 0.5*std_curve, mean_curve + 0.5*std_curve, color=colors[idx], alpha=0.1)
    
    plt.legend()
    plt.yscale('log')
    plt.xlabel("Temps (s)")
    plt.ylabel("Erreur Relative L2(t)")
    plt.title("Dynamique de l'Erreur (Moyenne sur 1000 tirs)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig("outputs/2_Evolution_Temporelle.png", dpi=300)
    plt.close()

    # --- B. GIF ANIMATION (3 Lignes) ---
    print("🎬 Génération GIF...")
    p_fixed = {'A': 0.7, 'D': 0.05, 'mu': 0.8, 'v': 1.0, 'k': 2.0, 'sigma': 0.5}
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    data_lines = [] 
    
    for idx, tid in enumerate(target_types):
        p = p_fixed.copy()
        p['type'] = tid
        x, t, u_true, u_pred = predict_case(model, p, cfg, 3.0, device)
        
        ax = axes[idx]
        l1, = ax.plot([], [], 'k--', label='Ref')
        l2, = ax.plot([], [], 'r-', label='DeepONet')
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(x[0], x[-1])
        ax.set_title(types_names[tid], fontsize=11, fontweight='bold')
        data_lines.append((l1, l2, (x, t, u_true, u_pred)))
        if idx==0: ax.legend(loc='upper right')

    def update(frame):
        artists = []
        for (l1, l2, (x, t, ut, up)) in data_lines:
            if frame < len(t):
                l1.set_data(x, ut[:, frame])
                l2.set_data(x, up[:, frame])
                artists.append(l1)
                artists.append(l2)
        return artists

    frames = range(0, len(data_lines[0][2][1]), 2)
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)
    ani.save("outputs/3_Comparaison.gif", writer='pillow', fps=20)
    plt.close()

    # --- C. SNAPSHOTS SUPERPOSÉS (3 Lignes) ---
    print("📸 Génération Snapshots Superposés...")
    target_times = [0.0, 0.7, 1.2, 1.7, 2.0, 2.5, 3.0]
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    colors_time = cm.viridis(np.linspace(0.1, 0.95, len(target_times)))

    for idx, tid in enumerate(target_types):
        p = p_fixed.copy()
        p['type'] = tid
        x, t_grid, u_true, u_pred = predict_case(model, p, cfg, 3.0, device)
        
        ax = axes[idx]
        ax.set_title(types_names[tid], fontsize=12, fontweight='bold', pad=10)
        
        for j, t_target in enumerate(target_times):
            idx_t = np.argmin(np.abs(t_grid - t_target))
            ax.plot(x, u_true[:, idx_t], color='gray', linestyle=':', alpha=0.5, linewidth=1)
            label = f"t={t_target:.1f}" if idx == 0 else ""
            ax.plot(x, u_pred[:, idx_t], color=colors_time[j], linestyle='-', linewidth=2, alpha=0.9, label=label)

        ax.set_ylim(-1.5, 1.5)
        ax.set_ylabel("u(x, t)")
        ax.grid(True, alpha=0.2)
        
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Temps (s)")

    axes[-1].set_xlabel("Position x")
    plt.tight_layout()
    plt.savefig("outputs/4_Snapshots_Superposes.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    MODEL_PATH = "outputs/first results PINN/run_20260208-092447/model_final.pth"
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = "run_20260208-092447/model_latest.pth"
        if not os.path.exists(MODEL_PATH):
            print("❌ Modèle introuvable.")
            sys.exit(1)

    device = torch.device("cpu")
    print(f"🖥️ Device : {device}")

    cfg = load_config()
    model = load_model(MODEL_PATH, cfg, device)
    
    run_analysis(model, cfg, device)
    print("\n✅ Analyse Terminée. Résultats dans 'outputs/'")