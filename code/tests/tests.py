"""
Allows verification of the validity of the calculation of pde_residual_ADR and of the validity of the classical solver which serves as a reference.
"""
import os
import sys
import torch
import numpy as np

file_path = os.path.abspath(__file__) 
project_root = os.path.dirname(os.path.dirname(file_path)) # On remonte de tests/ vers la racine
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.physics.residual_ADR import pde_residual_adr
from src.utils.CN_ADR import crank_nicolson_adr
from src.data.generators import get_ic_value

def test_pde_residual_analytical():
    """
    Verify that the residual of the PDE returns approximately 0 for an exact analytical solution. Chosen solution: u(x,t) = sin(x - v*t) (progressive wave, satisfies u_t + v*u_x = 0)
    """
    print("Test PDE")
    
    class MockModel(torch.nn.Module):
        def forward(self, params, xt):
            # params[:, 0] est la vitesse v
            v = params[:, 0:1]
            x = xt[:, 0:1]
            t = xt[:, 1:2]
            return torch.sin(x - v * t) # u(x,t) = sin(x - v*t)

    model = MockModel()
    
    # Paramètres : v=2.0, D=0.0, mu=0.0 (Pure advection)
    # [v, D, mu, type, A, x0, sigma, k]
    p_vec = torch.tensor([[2.0, 0.0, 0.0, 0, 1.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
    xt_tensor = torch.tensor([[1.0, 0.5], [0.0, 2.0], [-1.0, 1.0]], requires_grad=True)

    res = pde_residual_adr(model, p_vec, xt_tensor)
    
    # Le résidu doit être très proche de 0
    mean_res = torch.mean(torch.abs(res)).item()
    print(f"Mean res : {mean_res:.2e}")
    
    assert mean_res < 1e-4, f"Erreur : Le résidu devrait être nul, il vaut {mean_res}"
    print("Test PDE : Ok")

def test_cn_solver_pure_advection():
    """
    Compare the Crank-Nicolson solver with the analytical solution of pure advection. u_exact(x, t) = u_0(x - v*t)
    """
    print("Test CN")
    
    v_val = 1.0
    x_min, x_max = -10.0, 10.0
    Nx, Nt = 400, 200
    T_max = 2.0
    x = np.linspace(x_min, x_max, Nx)
    
    ic_configs = {
        "Tanh": {"type": 0, "bc_kind": "tanh_pm1", "params": {"type": 0, "A": 1.0, "x0": 0.0, "sigma": 0.5, "k": 2.0}},
        "Sin-Gauss": {"type": 1, "bc_kind": "zero_zero", "params": {"type": 1, "A": 1.0, "x0": 0.0, "sigma": 0.5, "k": 2.0}},
        "Gaussian": {"type": 3, "bc_kind": "zero_zero", "params": {"type": 3, "A": 1.0, "x0": 0.0, "sigma": 0.5, "k": 2.0}}
    }

    for name, config in ic_configs.items():
        u0 = get_ic_value(x, "mixed", config["params"])
        
        _, U_cn, _ = crank_nicolson_adr(
            v=v_val, D=0.0, mu=0.0,
            xL=x_min, xR=x_max, Nx=Nx, Tmax=T_max, Nt=Nt,
            bc_kind=config["bc_kind"], x0=x, u0=u0
        )
        if U_cn.shape == (Nx, Nt): U_cn = U_cn.T
        
        U_exact = np.zeros_like(U_cn)
        t_array = np.linspace(0, T_max, Nt)
        
        for i, t in enumerate(t_array):
            x_shifted = x - v_val * t
            U_exact[i, :] = get_ic_value(x_shifted, "mixed", config["params"])
            
        num = np.linalg.norm(U_exact - U_cn)
        den = np.linalg.norm(U_exact) + 1e-8
        l2_err = num / den
        
        status = "✅" if l2_err < 0.05 else "❌" 
        print(f"  - {name:<12} : {l2_err:.2%} {status}")

if __name__ == "__main__":
    test_pde_residual_analytical()
    test_cn_solver_pure_advection()