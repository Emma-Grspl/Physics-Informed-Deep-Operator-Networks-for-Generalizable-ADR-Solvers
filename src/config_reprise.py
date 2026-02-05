import numpy as np

class Config:
    # --- GÉOMÉTRIE & TEMPS ---
    x_min, x_max = -7.0, 7.0
    T_max = 2.0  
    dt = 0.2
    Nt = int(np.ceil(T_max / dt))

    # --- PHYSIQUE ---
    ranges = {
        'v': (0.5, 2.0), 'D': (0.01, 0.2), 'mu': (0.0, 1.0),
        'A': (0.8, 1.2), 'x0': (0.0, 0.0), 'sigma': (0.4, 0.8), 'k': (1.0, 3.0)
    }

    @staticmethod
    def get_p_dict():
        r = Config.ranges
        return {
            'v': np.random.uniform(*r['v']), 'D': np.random.uniform(*r['D']),
            'mu': np.random.uniform(*r['mu']), 'type': np.random.randint(0, 5),
            'A': np.random.uniform(*r['A']), 'x0': np.random.uniform(*r['x0']),
            'sigma': np.random.uniform(*r['sigma']), 'k': np.random.uniform(*r['k'])
        }

    # --- ARCHITECTURE (DOIT ÊTRE IDENTIQUE AU RUN PRÉCÉDENT) ---
    branch_depth, branch_width = 6, 256
    trunk_depth, trunk_width = 5, 256
    latent_dim = 256
    branch_dim = [branch_width] * branch_depth 
    trunk_dim = [trunk_width] * trunk_depth
    nFourier = 64
    sFourier = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 15.0, 20.0] # Ajout de hautes fréquences

    # --- HYPERPARAMÈTRES DE REPRISE ---
    learning_rate = 1e-4      # On baisse un peu le LR pour la finition
    n_iters_per_step = 8000   # Plus d'itérations pour la précision
    n_sample = 8192           # DOUBLEMENT de la densité de points (CRUCIAL)
    batch_size = 4096
    max_retry = 5
    threshold = 0.03

    # --- LOSS WEIGHTS (Focus Physique) ---
    weight_res = 400.0        # On double la sévérité de l'EDP
    weight_ic = 50.0          # L'IC est loin, on relâche
    weight_bc = 100.0

    # --- AUDIT ---
    Nx_solver, Nx_audit, Nt_audit = 100, 200, 100
    save_dir = "./results_reprise"