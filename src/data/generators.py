import torch
import numpy as np
from config import Config  # <--- Ajout crucial

# Device detection
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps") # Mac
else:
    DEVICE = torch.device("cpu")

# --- IC Generation Utils ---
def get_ic_value(x, ic_kind, ic_params):
    """
    Génère la condition initiale (IC). Inchangé dans la logique, 
    mais utilise les params passés dynamiquement.
    """
    if ic_params is None: ic_params = {}
    
    # torch or numpy 
    is_torch = isinstance(x, torch.Tensor)
    if not is_torch and not isinstance(x, np.ndarray):
        x = np.array(x)

    # Tool configuration
    if is_torch:
        exp, sin, tanh = torch.exp, torch.sin, torch.tanh
        zeros_like = torch.zeros_like
        cast = lambda m: m.float()
        check_any = lambda m: m.sum() > 0 
    else:
        exp, sin, tanh = np.exp, np.sin, np.tanh
        zeros_like = np.zeros_like
        cast = lambda m: np.asarray(m).astype(float)
        check_any = lambda m: np.any(m)

    # Logic
    if ic_kind == "mixed":
        types = ic_params.get("type") 
        A     = ic_params.get("A", 1.0)
        x0    = ic_params.get("x0", 0.0)
        sigma = ic_params.get("sigma", 0.5)
        k     = ic_params.get("k", 2.0)

        u0 = zeros_like(x)

        # 0 = Tanh
        mask_0 = cast(types == 0)
        if check_any(mask_0): 
            u0 += mask_0 * A * tanh((x - x0) / (sigma + 1e-8))

        # 1 & 2 = Gaussian sinus
        mask_gs = cast((types == 1) | (types == 2))
        if check_any(mask_gs):
            u0 += mask_gs * A * exp( - (x - x0)**2 / (2 * sigma**2 + 1e-8) ) * sin(k * x)

        # 3 & 4 = Gaussian
        mask_gauss = cast((types == 3) | (types == 4))
        if check_any(mask_gauss):
            u0 += mask_gauss * A * exp( - (x - x0)**2 / (2 * sigma**2 + 1e-8) )

        return u0

    else:
        # Fallback pour single types (legacy)
        kind_clean = ic_kind.replace(" ", "_")
        def get_scalar(key, default):
            val = ic_params.get(key, default)
            if hasattr(val, 'item'): return val.item()
            return val

        A = get_scalar("A", 1.0)
        x0 = get_scalar("x0", 0.0)
        sigma = get_scalar("sigma", 0.5)
        k = get_scalar("k", 2.0)

        if kind_clean == "tanh": return A * tanh((x - x0)/sigma)
        elif kind_clean == "gaussian": return A * exp(-(x-x0)**2/(2*sigma**2))
        elif kind_clean == "gaussian_sinus": return A * exp(-(x-x0)**2/(2*sigma**2)) * sin(k*x)
        else: return zeros_like(x)


# --- Training Batch Generator ---
def generate_mixed_batch(n_samples, bounds_phy=None, x_min=None, x_max=None, Tmax=None):
    """
    Génère un batch d'entraînement en utilisant Config comme source de vérité par défaut.
    """
    # 1. Chargement des valeurs par défaut depuis Config si non fournies
    if bounds_phy is None: bounds_phy = Config.ranges
    if x_min is None: x_min = Config.x_min
    if x_max is None: x_max = Config.x_max
    # Tmax est obligatoire via l'argument ou Config, mais ici on le laisse souvent dynamique (current_t_max)
    if Tmax is None: Tmax = Config.T_max 

    # 2. Génération des Paramètres Physiques (v, D, mu) via Config
    v = np.random.uniform(bounds_phy['v'][0], bounds_phy['v'][1], (n_samples, 1))
    D = np.random.uniform(bounds_phy['D'][0], bounds_phy['D'][1], (n_samples, 1))
    mu = np.random.uniform(bounds_phy['mu'][0], bounds_phy['mu'][1], (n_samples, 1))

    # Types d'équation (0 à 4)
    types = np.random.choice([0, 1, 2, 3, 4], (n_samples, 1))

    # 3. Génération des Paramètres IC (A, x0, sigma, k) via Config
    # C'est ici qu'on corrige l'erreur des valeurs en dur !
    A = np.random.uniform(bounds_phy['A'][0], bounds_phy['A'][1], (n_samples, 1))
    x0 = np.random.uniform(bounds_phy['x0'][0], bounds_phy['x0'][1], (n_samples, 1))
    sigma = np.random.uniform(bounds_phy['sigma'][0], bounds_phy['sigma'][1], (n_samples, 1))
    k = np.random.uniform(bounds_phy['k'][0], bounds_phy['k'][1], (n_samples, 1))

    # Concatenation du vecteur paramètres
    params_vec = np.hstack((v, D, mu, types, A, x0, sigma, k))

    # 4. Coordonnées Spatio-Temporelles
    x = np.random.uniform(x_min, x_max, (n_samples, 1))
    t = np.random.uniform(0, Tmax, (n_samples, 1))
    xt = np.hstack((x, t))

    # 5. Points IC (t=0) pour la Loss IC
    x_ic = np.random.uniform(x_min, x_max, (n_samples, 1))
    xt_ic = np.hstack((x_ic, np.zeros_like(x_ic)))

    # Calcul de la Vérité Terrain à t=0 (u_true_ic)
    # On utilise vectorisation si possible, sinon boucle
    u_true_ic = np.zeros((n_samples, 1))
    for i in range(n_samples):
        p_dict = {
            "type": types[i,0], 
            "A": A[i,0], 
            "x0": x0[i,0], 
            "sigma": sigma[i,0], 
            "k": k[i,0]
        }
        u_true_ic[i] = get_ic_value(x_ic[i,0], "mixed", p_dict)

    # Conversion en Tensor sur Device
    return (torch.FloatTensor(params_vec).to(DEVICE),
            torch.FloatTensor(xt).to(DEVICE),
            torch.FloatTensor(xt_ic).to(DEVICE),
            torch.FloatTensor(u_true_ic).to(DEVICE))


# --- Validation Data for Solver ---
def get_validation_data_adr(N0, Nb, ic_kind="mixed", bc_kind="periodic", 
                            ic_kwargs=None, xL=None, xR=None, Tmax=None):
    """
    Prépare les grilles pour le solveur classique (Validation/Audit).
    Utilise Config pour les valeurs par défaut.
    """
    if ic_kwargs is None: ic_kwargs = {}
    
    # Valeurs par défaut Config
    if xL is None: xL = Config.x_min
    if xR is None: xR = Config.x_max
    if Tmax is None: Tmax = Config.T_max

    # Grille spatiale
    x0_np = np.linspace(xL, xR, N0)
    
    # Grille temporelle
    t_b_np = np.linspace(0, Tmax, Nb)
    
    # Condition initiale u0
    u0_np = get_ic_value(x0_np, ic_kind, ic_kwargs)

    data = {
        "x0": x0_np,      # Shape (N0,)
        "u0": u0_np,      # Shape (N0,)
        "t_b": t_b_np,    # Shape (Nb,)
        "xL": xL, "xR": xR,
        "ic_kind": ic_kind,
        "bc_kind": bc_kind
    }
    return data
