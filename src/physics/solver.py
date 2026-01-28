import numpy as np
from scipy.sparse import diags, linalg
from src.data.generators import get_validation_data_adr
from config import Config  # <--- Ajout de l'import Config

# -------------------------------------------------------------------------
# Core Solver : Crank-Nicolson (Maths pures, pas de dépendance Config directe)
# -------------------------------------------------------------------------
def crank_nicolson_adr(v, D, mu, xL, xR, Nx, Tmax, Nt, bc_kind, x0=None, u0=None):
    """
    Solves the 1D Advection-Diffusion-Reaction (ADR) equation.
    (Code mathématique inchangé, il reçoit juste des valeurs)
    """
    # Safety check sur Nt pour éviter division par zéro si Tmax=0 ou Nt=0
    if Nt == 0: Nt = 1
    
    x = np.linspace(xL, xR, Nx)
    dx = x[1] - x[0]
    dt = Tmax / Nt

    # BC bounds
    if bc_kind == "tanh_pm1":
        uL, uR = -1.0, 1.0  
    elif bc_kind in ["zero_zero", "neumann_zero", "periodic"]:
        uL, uR = 0.0, 0.0
    else:
        raise ValueError(f"Unknown bc_kind '{bc_kind}'.")

    if x0 is None or u0 is None: raise ValueError("Provide x0, u0")

    # Interpolation sur la grille du solveur
    u = np.interp(x, np.asarray(x0).flatten(), np.asarray(u0).flatten())
    U = np.zeros((Nt, Nx))
    U[0, :] = u
    t_grid = np.linspace(0, Tmax, Nt)

    # Matrix construction
    L_main = -2.0 * np.ones(Nx)
    L_off  =  1.0 * np.ones(Nx - 1)
    L = diags([L_off, L_main, L_off], offsets=[-1, 0, 1], format="csc") / (dx**2 + 1e-12)

    Dx_off = 0.5 * np.ones(Nx - 1)
    Dx = diags([-Dx_off, Dx_off], offsets=[-1, 1], format="csc") / (dx + 1e-12)
    I = diags([np.ones(Nx)], [0], format="csc")

    # BC Handling
    if bc_kind == "periodic":
        L = L.tolil(); Dx = Dx.tolil()
        L[0, -1] = 1.0/dx**2; L[-1, 0] = 1.0/dx**2
        Dx[0, -1] = -0.5/dx; Dx[-1, 0] = 0.5/dx
        L = L.tocsc(); Dx = Dx.tocsc()
    elif bc_kind == "neumann_zero":
        L = L.tolil(); Dx = Dx.tolil()
        L[0,0] = -2./dx**2; L[0,1]=2./dx**2; L[-1,-1]=-2./dx**2; L[-1,-2]=2./dx**2
        Dx[0,:]=0.0; Dx[-1,:]=0.0
        L = L.tocsc(); Dx = Dx.tocsc()

    # System Matrices
    A = (I - 0.5 * dt * (-v * Dx + D * L)).tolil()
    B = (I + 0.5 * dt * (-v * Dx + D * L)).tolil()

    if bc_kind in ["tanh_pm1", "zero_zero"]:
        for M in (A, B):
            M[0, :] = 0.0; M[-1, :] = 0.0
            M[0, 0] = 1.0; M[-1,-1] = 1.0

    A = A.tocsc(); B = B.tocsc()

    # Temporal loop
    for n in range(1, Nt):
        R = mu * (u - u**3) 
        rhs = B @ u + dt * R
        if bc_kind in ["tanh_pm1", "zero_zero"]:
            rhs[0] = uL; rhs[-1] = uR
        
        # Solve
        u = linalg.spsolve(A, rhs)  
        U[n, :] = u

    return x, U, t_grid


# -------------------------------------------------------------------------
# Audit Wrapper (Connecté à Config)
# -------------------------------------------------------------------------
def get_ground_truth_CN(params_dict, x_min=None, x_max=None, T_max=None, Nx=None, Nt=None):
    """
    Standardized interface for auditing and validation.
    Utilise Config pour les valeurs par défaut.
    """
    # 1. Valeurs par défaut depuis Config
    if x_min is None: x_min = Config.x_min
    if x_max is None: x_max = Config.x_max
    if T_max is None: T_max = Config.T_max
    if Nx is None: Nx = Config.Nx_solver  # Grille fine pour le solveur
    
    # Calcul automatique de Nt si non fourni, basé sur dt du config
    if Nt is None: 
        Nt = int(np.ceil(T_max / Config.dt))
        # Sécurité minimale : au moins 2 pas de temps
        if Nt < 2: Nt = 2

    # 2. Préparation des données initiales (IC)
    # On passe params_dict comme ic_kwargs pour que 'A', 'sigma', etc. soient utilisés
    ic_kwargs = params_dict.copy()
    
    val_data = get_validation_data_adr(
        N0=Nx, Nb=Nt, 
        ic_kind="mixed", bc_kind="periodic",
        ic_kwargs=ic_kwargs, 
        xL=x_min, xR=x_max, Tmax=T_max
    )

    # 3. Exécution du Solver
    # Attention : Nx et Nt ici sont ceux du maillage de résolution numérique
    raw_result = crank_nicolson_adr(
        v=params_dict['v'], 
        D=params_dict['D'], 
        mu=params_dict['mu'], 
        xL=x_min, xR=x_max, 
        Nx=Nx, Tmax=T_max, Nt=Nt, 
        bc_kind="periodic", 
        x0=val_data["x0"], 
        u0=val_data["u0"]
    )

    if isinstance(raw_result, tuple):
        U_true_matrix = raw_result[1] 
    else:
        U_true_matrix = raw_result

    # 4. Formatage de sortie (Shape [Nx, Nt])
    # Le solveur renvoie [Nt, Nx], on transpose pour avoir [Space, Time] comme souvent attendu
    if U_true_matrix.shape == (Nt, Nx):
        U_true_matrix = U_true_matrix.T

    # Création des grilles pour l'interpolation ou le plot
    x = np.linspace(x_min, x_max, Nx)
    t = np.linspace(0, T_max, Nt)
    X_grid, T_grid = np.meshgrid(x, t, indexing='ij')

    return X_grid, T_grid, U_true_matrix
