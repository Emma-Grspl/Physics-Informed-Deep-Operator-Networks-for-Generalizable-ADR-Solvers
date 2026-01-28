import numpy as np

#Global config
CONFIG = {
    #geometrical bounds
    "x_min": -7.0,"x_max": 7.0,"Tmax": 1.0,

    #physical bounds
    "bounds_phy": {'v':  (0.5, 2.0),'D':  (0.01, 0.5), 'mu': (0.5, 2.0)},

    #neural architecture
    "network": {
        "branch_dim": 8,            # [v, D, mu, type, A, x0, sigma, k]
        "trunk_dim": 2,             # [x, t]
        "latent_dim": 256,          # Hidden layer width
        "branch_layers": [256, 256, 256, 256],
        "trunk_layers": [256, 256, 256],
        "nFourier": 126,            # Frequency number
        "sFourier": [0.0, 1.0]      # Scale
    },

    #training
    "training": {
        "batch_size": 1024,
        "n_warmup": 5000,           #IC
        "n_physics": 10000,         #PDE + IC
        "learning_rate": 5e-4,
        "loss_weights": {"pde": 100.0, "ic": 150.0, "bc": 20.0}
    },

    #files
    "dirs": {
        "results": "results_experiment_1"
    }
}
