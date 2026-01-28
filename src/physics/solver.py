import numpy as np
from scipy.sparse import diags, linalg
from src.data.generators import get_validation_data_adr

#ref solver
def crank_nicolson_adr(v, D, mu, xL, xR, Nx, Tmax, Nt, bc_kind, x0=None, u0=None):
    """
    Solves the 1D Advection-Diffusion-Reaction (ADR) equation using a semi-implicit 
    Crank-Nicolson scheme with an explicit handling of the non-linear reaction term.

    Args:
        v (float): 
            Advection coefficient (velocity).
        D (float): 
            Diffusion coefficient (viscosity).
        mu (float): 
            Reaction coefficient controlling the magnitude of the non-linear source term.
        xL (float): 
            Left boundary of the spatial domain.
        xR (float): 
            Right boundary of the spatial domain.
        Nx (int): 
            Number of spatial grid points (resolution).
        Tmax (float): 
            Final simulation time.
        Nt (int): 
            Number of time steps.
        bc_kind (str): 
            Type of boundary conditions to apply:
            - "periodic": Periodic BCs (u(xL) = u(xR)).
            - "zero_zero": Dirichlet BCs with u=0 at both ends.
            - "tanh_pm1": Dirichlet BCs with u=-1 at left, u=1 at right.
            - "neumann_zero": Zero-flux Neumann BCs (∂u/∂x = 0 at boundaries).
        x0 (array-like): 
            Original spatial grid of the initial condition u0 (for interpolation).
        u0 (array-like): 
            Values of the initial condition.

    Returns:
        tuple (x, U, t):
            - x (np.ndarray): Spatial grid vector of size [Nx].
            - U (np.ndarray): Solution matrix of size [Nt, Nx], where U[n, :] is the solution at time t[n].
            - t (np.ndarray): Temporal grid vector of size [Nt].
    """
    x = np.linspace(xL, xR, Nx)
    dx = x[1] - x[0]
    dt = Tmax / Nt

    #BC bounds
    if bc_kind == "tanh_pm1":
        uL, uR = -1.0, 1.0  
    elif bc_kind in ["zero_zero", "neumann_zero", "periodic"]:
        uL, uR = 0.0, 0.0
    else:
        raise ValueError(f"Unknown bc_kind '{bc_kind}'.")

    if x0 is None or u0 is None: raise ValueError("Provide x0, u0")

    #Iterpolation
    u = np.interp(x, np.asarray(x0).flatten(), np.asarray(u0).flatten())
    U = np.zeros((Nt, Nx))
    U[0, :] = u
    t_grid = np.linspace(0, Tmax, Nt)

    #Matrix construction
    L_main = -2.0 * np.ones(Nx)
    L_off  =  1.0 * np.ones(Nx - 1)
    L = diags([L_off, L_main, L_off], offsets=[-1, 0, 1], format="csc") / dx**2

    Dx_off = 0.5 * np.ones(Nx - 1)
    Dx = diags([-Dx_off, Dx_off], offsets=[-1, 1], format="csc") / dx
    I = diags([np.ones(Nx)], [0], format="csc")

    #BC
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

    A = (I - 0.5 * dt * (-v * Dx + D * L)).tolil()
    B = (I + 0.5 * dt * (-v * Dx + D * L)).tolil()

    if bc_kind in ["tanh_pm1", "zero_zero"]:
        for M in (A, B):
            M[0, :] = 0.0; M[-1, :] = 0.0
            M[0, 0] = 1.0; M[-1,-1] = 1.0

    A = A.tocsc(); B = B.tocsc()

    #Temporal loop
    for n in range(1, Nt):
        R = mu * (u - u**3) 
        rhs = B @ u + dt * R
        if bc_kind in ["tanh_pm1", "zero_zero"]:
            rhs[0] = uL; rhs[-1] = uR
        u = linalg.spsolve(A, rhs)  
        U[n, :] = u

    return x, U, t_grid

# Audit wrapper
def get_ground_truth_CN(params_dict, x_min, x_max, T_max, Nx=100, Nt=100):
    """
    Standardized interface for auditing and validation.

    Args:
        params_dict (dict): 
            A dictionary containing the configuration for a specific physical scenario.
            Expected keys:
            - 'v', 'D', 'mu': Physical coefficients (Advection, Diffusion, Reaction).
            - 'ic_kind' (str): Type of initial condition (e.g., "gaussian", "mixed").
            - 'ic_params' (dict): Parameters for the IC function (Amplitude, sigma, etc.).
            - 'bc_kind' (str): Type of boundary conditions (e.g., "periodic").
        x_min (float): 
            Left boundary of the spatial domain.
        x_max (float): 
            Right boundary of the spatial domain.
        T_max (float): 
            Final time of the simulation.
        Nx (int, default=100): 
            Spatial resolution (number of grid points).
        Nt (int, default=100): 
            Temporal resolution (number of time steps).

    Returns:
        tuple (X_grid, T_grid, U_exact):
            - X_grid (np.ndarray): 2D meshgrid of spatial coordinates [Nt, Nx].
            - T_grid (np.ndarray): 2D meshgrid of temporal coordinates [Nt, Nx].
            - U_exact (np.ndarray): The numerical solution matrix [Nt, Nx] corresponding 
              to the grid points.
    """
    #preparation of initial data
    ic_kwargs = params_dict.copy()
    val_data = get_validation_data_adr(
        N0=Nx, Nb=Nt, ic_kind="mixed", bc_kind="periodic",
        ic_kwargs=ic_kwargs, xL=x_min, xR=x_max, Tmax=T_max)

    #Solver use
    raw_result = crank_nicolson_adr(
        v=params_dict['v'], D=params_dict['D'], mu=params_dict['mu'], 
        xL=x_min, xR=x_max, Nx=Nx, Tmax=T_max, Nt=Nt, 
        bc_kind="periodic", 
        x0=val_data["x0"], 
        u0=val_data["u0"])

    if isinstance(raw_result, tuple):
        U_true_matrix = raw_result[1] 
    else:
        U_true_matrix = raw_result

    # formatting
    if U_true_matrix.shape == (Nt, Nx):
        U_true_matrix = U_true_matrix.T

    x = np.linspace(x_min, x_max, Nx)
    t = np.linspace(0, T_max, Nt)
    X_grid, T_grid = np.meshgrid(x, t, indexing='ij')

    return X_grid, T_grid, U_true_matrix
