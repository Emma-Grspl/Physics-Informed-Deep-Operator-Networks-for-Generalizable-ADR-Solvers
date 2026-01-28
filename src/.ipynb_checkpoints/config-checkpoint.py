import numpy as np

# Global config
CONFIG = {
    # geometrical bounds
    "x_min": -7.0, "x_max": 7.0, "Tmax": 1.0,

    # physical bounds
    # D descend à 0.01 => Chocs forts => Besoin de sFourier élevé (50.0 est parfait)
    "bounds_phy": {'v': (0.5, 2.0), 'D': (0.01, 0.5), 'mu': (0.5, 2.0)},

    # neural architecture
    "network": {
        "branch_dim": 8,            # [v, D, mu, type, A, x0, sigma, k]
        "trunk_dim": 2,             # [x, t]
        "latent_dim": 256,          # Alignement parfait avec les layers
        "branch_layers": [256, 256, 256, 256],
        "trunk_layers": [256, 256, 256],
        "nFourier": 126,            
        "sFourier": [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0] # 50.0 aide pour les chocs
    },

    # training
    "training": {
        "batch_size": 4096,       # AUGMENTÉ : Optimisé pour V100 (plus stable/rapide)
        
        # WARMUP (t=0)
        # C'est un "Max". Comme le code s'arrête dès que l'erreur < 3%, 
        # on peut mettre une valeur haute par sécurité.
        "n_warmup": 50000,        
        
        # PHYSICS (Time Marching)
        # ATTENTION : C'est le nombre d'itérations PAR PALIER (il y a 10 paliers).
        # 20 000 * 10 = 200 000 itérations totales (C'est le bon chiffre).
        "n_physics": 20000,       
        
        "learning_rate": 5e-4,
        "loss_weights": {"pde": 100.0, "ic": 150.0, "bc": 20.0}
    },

    # files
    "dirs": {
        "results": "results_train_2_final" # Nom explicite pour la V2
    }
}