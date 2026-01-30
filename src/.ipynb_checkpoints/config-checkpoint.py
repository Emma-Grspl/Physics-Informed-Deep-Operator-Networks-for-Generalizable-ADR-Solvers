import numpy as np

class Config:
    # -------------------------------------------------------------------------
    # 1. GÉOMÉTRIE & TEMPS
    # -------------------------------------------------------------------------
    x_min = -7.0
    x_max = 7.0
    
    # T_max définit la borne supérieure maximale de la simulation.
    T_max = 2.0  
    
    dt = 0.5
    
    # Nombre de pas de temps total (calculé automatiquement)
    Nt = int(np.ceil(T_max / dt))

    # -------------------------------------------------------------------------
    # 2. PHYSIQUE (Intervalles & Génération)
    # -------------------------------------------------------------------------
    # Intervalles de variation pour la génération aléatoire
    ranges = {
        'v': (0.5, 2.0),
        'D': (0.01, 0.2),
        'mu': (0.0, 1.0),
        'A': (0.8, 1.2),
        'x0': (0.0, 0.0),      
        'sigma': (0.4, 0.8),
        'k': (1.0, 3.0)
    }

    @staticmethod
    def get_p_dict():
        """
        Retourne un dictionnaire de paramètres physiques aléatoires
        basé sur les intervalles définis ci-dessus.
        """
        r = Config.ranges
        type_id = np.random.randint(0, 5) # 5 types d'équations (0 à 4)
        
        return {
            'v': np.random.uniform(*r['v']),
            'D': np.random.uniform(*r['D']),
            'mu': np.random.uniform(*r['mu']),
            'type': type_id,
            'A': np.random.uniform(*r['A']),
            'x0': np.random.uniform(*r['x0']),
            'sigma': np.random.uniform(*r['sigma']),
            'k': np.random.uniform(*r['k'])
        }

    # -------------------------------------------------------------------------
    # 3. ARCHITECTURE DU MODÈLE (DeepONet)
    # -------------------------------------------------------------------------
    # Dimensions
    branch_dim = [256, 256, 256, 256]
    trunk_dim = [256, 256, 256]
    latent_dim = 256
    
    # Profondeur (Nombre de couches cachées)
    branch_layers = 8
    trunk_layers = 2
    
    # Encodage de Fourier (Trunk)
    nFourier = 64
    sFourier = [0.0, 1.0, 2.0]

    # -------------------------------------------------------------------------
    # 4. HYPERPARAMÈTRES D'ENTRAÎNEMENT
    # -------------------------------------------------------------------------
    learning_rate = 5e-4      # LR de base
    
    epochs = 10000            # Nombre global d'époques (si utilisé hors smart loop)
    n_warmup = 15000          # Itérations pour figer t=0
    n_iters_per_step = 15000  # Itérations par palier de temps
    
    n_sample = 2000           # Points de collocation
    batch_size = 4096         # Taille de lot (V100 friendly)
    
    max_retry = 3             # Nombre d'essais en cas d'échec
    threshold = 0.03          # Seuil d'erreur relative pour valider un palier

    # -------------------------------------------------------------------------
    # 5. FONCTION DE COÛT (Loss Weights)
    # -------------------------------------------------------------------------
    weight_res = 100.0   # Résidu EDP
    weight_ic = 150.0    # Condition Initiale
    weight_bc = 20.0     # Conditions aux limites

    # -------------------------------------------------------------------------
    # 6. SOLVEUR & AUDIT
    # -------------------------------------------------------------------------
    Nx_solver = 100      # Grille spatiale pour la Vérité Terrain (Solveur)
    
    Nx_audit = 200       # Résolution spatiale pour l'évaluation/plot
    Nt_audit = 100       # Résolution temporelle pour l'évaluation/plot
    
    save_dir = "./results"