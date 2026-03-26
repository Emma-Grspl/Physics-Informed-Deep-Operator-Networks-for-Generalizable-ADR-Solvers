"""
This module provides a reference solver based on the Crank-Nicolson scheme for the Advection-Diffusion-Reaction equation. It is used to generate ground truth during the audit and validation phases.
- crank_nicolson_adr: The core algorithm (implicit scheme).
- get_ground_truth_CN: Interface wrapper for PI-DeepONet auditing.
"""
import numpy as np
from scipy.sparse import diags, linalg
import sys
import os
file_path = os.path.abspath(__file__) 
project_root = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))

if project_root not in sys.path:
    sys.path.append(project_root)


def get_ic_value_numpy(x, ic_params):
    x = np.asarray(x)
    types = ic_params.get("type")
    A = ic_params.get("A", 1.0)
    x0 = ic_params.get("x0", 0.0)
    sigma = ic_params.get("sigma", 0.5)
    k = ic_params.get("k", 2.0)

    u0 = np.zeros_like(x, dtype=float)

    if types == 0:
        u0 += np.tanh((x - x0) / (sigma + 1e-8))
    elif types in [1, 2]:
        u0 += A * np.exp(-((x - x0) ** 2) / (2 * sigma**2 + 1e-8)) * np.sin(k * x)
    elif types in [3, 4]:
        u0 += A * np.exp(-((x - x0) ** 2) / (2 * sigma**2 + 1e-8))

    return u0


def get_validation_data_adr_numpy(N0, Nb, ic_kwargs, xL, xR, Tmax):
    x0_np = np.linspace(xL, xR, N0)
    t_b_np = np.linspace(0, Tmax, Nb)
    u0_np = get_ic_value_numpy(x0_np, ic_kwargs)

    return {
        "x0": x0_np,
        "u0": u0_np,
        "t_b": t_b_np,
        "xL": xL,
        "xR": xR,
    }

#Crank Nicolson
def crank_nicolson_adr(v, D, mu, xL, xR, Nx, Tmax, Nt, bc_kind, x0=None, u0=None):
    """
Solves the ADR equation u_t + v*u_x - D*u_xx = mu*(u - u^3) using a Crank-Nicolson scheme. Provides a high-precision and stable (semi-implicit) numerical solution for validating neural network predictions. The Crank-Nicolson scheme is second-order in both time and space.
    Args:
        v (float): Advection velocity.
        D (float): Diffusion coefficient.
        mu (float): Linear reaction coefficient.
        xL, xR (float): Spatial domain boundaries.
        Nx (int): Number of spatial discretization points.
        Tmax (float): Final simulation time.
        Nt (int): Number of time steps.
        bc_kind (str): Type of boundary conditions ("tanh_pm1", "zero_zero", "neumann_zero")
        x0 (np.ndarray, optional): Spatial coordinates of the provided IC.
        u0 (np.ndarray, optional): Values ​​of the initial condition u(x, 0).
    Outputs:
        tuple: (x, U, t) where x is the spatial grid, U is the solution matrix [Nt, Nx], and t is the temporal grid.
"""
    if Nt == 0: Nt = 1
    x = np.linspace(xL, xR, Nx)
    dx = x[1] - x[0]
    dt = Tmax / Nt

    if x0 is None or u0 is None: raise ValueError("Provide x0, u0")

    # interp
    u = np.interp(x, np.asarray(x0).flatten(), np.asarray(u0).flatten())
    U = np.zeros((Nt, Nx))
    U[0, :] = u
    
    # bc 
    uL, uR = (-1.0, 1.0) if bc_kind == "tanh_pm1" else (0.0, 0.0)

    L_main = -2.0 * np.ones(Nx)
    L_off  =  1.0 * np.ones(Nx - 1)
    L = diags([L_off, L_main, L_off], offsets=[-1, 0, 1], format="lil") / (dx**2)

    Dx_off = 0.5 * np.ones(Nx - 1)
    Dx = diags([-Dx_off, Dx_off], offsets=[-1, 1], format="lil") / dx
    
    if bc_kind == "neumann_zero":
        L[0, 0:2] = [-1.0/dx**2, 1.0/dx**2]
        L[-1, -2:] = [1.0/dx**2, -1.0/dx**2]
        Dx[0, :] = 0.0
        Dx[-1, :] = 0.0

    L = L.tocsc()
    Dx = Dx.tocsc()
    I = diags([np.ones(Nx)], [0], format="csc")

    A = (I - 0.5 * dt * (-v * Dx + D * L)).tolil()
    B = (I + 0.5 * dt * (-v * Dx + D * L)).tolil()

    if bc_kind in ["tanh_pm1", "zero_zero"]:
        for M in (A, B):
            M[0, :] = 0.0; M[-1, :] = 0.0
            M[0, 0] = 1.0; M[-1, -1] = 1.0

    A, B = A.tocsc(), B.tocsc()

    for n in range(1, Nt):
        R = mu * (u - u**3)
        rhs = B @ u + dt * R
        
        if bc_kind in ["tanh_pm1", "zero_zero"]:
            rhs[0], rhs[-1] = uL, uR
        
        u = linalg.spsolve(A, rhs)
        U[n, :] = u

    return x, U, np.linspace(0, Tmax, Nt)

# Audit Wrapper
def get_ground_truth_CN(params_dict, full_cfg, t_step_max=None):
    """
    Interface for extracting ground truth during an audit. Simplifies the solver call by automatically extracting parameters from the YAML configuration file and managing grid (meshgrid) formatting for direct comparison with DeepONet outputs.
    Args:
        params_dict (dict): Dictionary of physical parameters (v, D, mu, type, etc.).
        full_cfg (dict): Complete configuration loaded from the YAML.
        t_step_max (float, optional): Maximum time for the audit (overrides the T_max in the YAML).
    Outputs:
        tuple: (X_grid, T_grid, U_true_matrix) where the grids are in [Nx, Nt] format.
    """
    # 1. Extraction des paramètres
    g_cfg = full_cfg['geometry']
    a_cfg = full_cfg['audit']
    
    x_min, x_max = g_cfg['x_min'], g_cfg['x_max']
    T_max = t_step_max if t_step_max is not None else g_cfg['T_max']
    Nx = a_cfg['Nx_solver']

    # Nt calculé pour respecter le dt d'échantillonnage
    dt_ref = full_cfg['time_stepping']['zones'][0]['dt']
    Nt = int(np.ceil(T_max / dt_ref))
    if Nt < 2: Nt = 2

    # 2. Détermination des BC
    equation_type = int(params_dict.get('type', 0))
    selected_bc = "tanh_pm1" if equation_type == 0 else "zero_zero"

    # 3. IC
    ic_kwargs = params_dict.copy()
    val_data = get_validation_data_adr_numpy(
        N0=Nx,
        Nb=Nt,
        ic_kwargs=ic_kwargs,
        xL=x_min,
        xR=x_max,
        Tmax=T_max,
    )

    # 4. Solver
    _, U_true_matrix, _ = crank_nicolson_adr(
        v=params_dict['v'], 
        D=params_dict['D'], 
        mu=params_dict['mu'], 
        xL=x_min, xR=x_max, 
        Nx=Nx, Tmax=T_max, Nt=Nt, 
        bc_kind=selected_bc, 
        x0=val_data["x0"], 
        u0=val_data["u0"]
    )

    # 5. Formatage final [Nx, Nt]
    if U_true_matrix.shape == (Nt, Nx):
        U_true_matrix = U_true_matrix.T

    x = np.linspace(x_min, x_max, Nx)
    t = np.linspace(0, T_max, Nt)
    X_grid, T_grid = np.meshgrid(x, t, indexing='ij')

    return X_grid, T_grid, U_true_matrix
