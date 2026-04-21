"""Comparison-layer module `jax_comparison.multifamily.src.pytorch.generators`.

This file supports the PyTorch versus JAX comparison workflows, including replicated implementations, scripts, or validation helpers.
"""

import torch
import numpy as np

def get_ic_value(x, ic_kind, ic_params):
    """
    Generates the initial condition (IC) values for a given spatial domain. Allows dynamic switching between different mathematical forms (Tanh, Wavelet, Gaussian) used to initialize the system at t=0. The function is data type agnostic (compatible with PyTorch Tensor and NumPy Array).
    
    Arguments:
        x(Union[torch.Tensor, np.ndarray]): Grid of spatial points.
        ic_kind(str): Identifier of the generation logic (e.g., "mixed").
        ic_params(dict): Parameters of the form (type, A, x0, sigma, k).
    Outputs:
        Union[torch.Tensor, np.ndarray]: Values u(x, 0) calculated on the grid x.

    """
    if ic_params is None: ic_params = {}
    
    is_torch = isinstance(x, torch.Tensor)
    if not is_torch and not isinstance(x, np.ndarray):
        x = np.array(x)

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

    # Mixed logic for DeepOnet
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
            u0 += mask_0 * tanh((x - x0) / (sigma + 1e-8))

        # 1 & 2 = Gaussian Sinus 
        mask_gs = cast((types == 1) | (types == 2))
        if check_any(mask_gs):
            u0 += mask_gs * A * exp( - (x - x0)**2 / (2 * sigma**2 + 1e-8) ) * sin(k * x)

        # 3 & 4 = Gaussian 
        mask_gauss = cast((types == 3) | (types == 4))
        if check_any(mask_gauss):
            u0 += mask_gauss * A * exp( - (x - x0)**2 / (2 * sigma**2 + 1e-8) )

        return u0
    else:
        return zeros_like(x)

def get_validation_data_adr(N0, Nb, ic_kind, bc_kind, ic_kwargs, xL, xR, Tmax):
    """
    Prepares the data structures necessary for a standard solver (Crank-Nicolson) or an audit. Creates uniform NumPy grids for validation. Unlike the batch generator, this function ensures an ordered distribution to facilitate standard numerical resolution.
    
    Args:
        N0 (int): Number of points in the spatial domain.
        Nb (int): Number of time steps.
        ic_kind (str): Type of initial condition.
        bc_kind (str): Type of boundary condition.
        ic_kwargs (dict): Parameters specific to the initial condition.
        xL (float): Lower bound of the spatial domain.
        xR (float): Upper bound of the spatial domain.
        Tmax (float): Maximum time horizon.
    Outputs:
        dict: Contains the spatial (x0), temporal (t_b), calculated initial condition (u0), and bounds grids.
    """
    x0_np = np.linspace(xL, xR, N0)
    t_b_np = np.linspace(0, Tmax, Nb)
    u0_np = get_ic_value(x0_np, ic_kind, ic_kwargs)

    return {
        "x0": x0_np,      
        "u0": u0_np,      
        "t_b": t_b_np,    
        "xL": xL, "xR": xR
    }

#Generator batch
def generate_mixed_batch(n_samples, bounds_phy, x_min, x_max, Tmax, allowed_types=None, device=None):
    """
    Generates a complete training batch for the PI-DeepONet. Randomly samples physical parameters, PDE collocation points, IC points, and BC points. It manages a point concentration (n_action) in areas of interest (x > 0) to improve model accuracy.

    Args:
        n_samples (int): Number of points to generate for each category (PDE, IC, BC).
        bounds_phy (dict): Intervals of physical parameters (v, D, mu, A, sigma, k, x0).
        x_min (float): Left bound of the spatial domain.
        x_max (float): Right bound of the spatial domain.
        Tmax (float): Maximum time.
        allowed_types (list, optional): List of IC types allowed for this batch.
        device (torch.device, optional): Target device for the PyTorch Tensors.
    Outputs:
        tuple: Contains the PyTorch Tensors (params_vec, xt, xt_ic, u_true_ic, xt_bc_l, xt_bc_r, u_true_bc_l, u_true_bc_r).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Physical parameters
    v = np.random.uniform(bounds_phy['v'][0], bounds_phy['v'][1], (n_samples, 1))
    D = np.random.uniform(bounds_phy['D'][0], bounds_phy['D'][1], (n_samples, 1))
    mu = np.random.uniform(bounds_phy['mu'][0], bounds_phy['mu'][1], (n_samples, 1))
    
    if allowed_types is not None and len(allowed_types) > 0:
        types = np.random.choice(allowed_types, size=(n_samples, 1))
    else:
        types = np.random.choice([0, 1, 2, 3, 4], size=(n_samples, 1))

    A = np.random.uniform(bounds_phy['A'][0], bounds_phy['A'][1], (n_samples, 1))
    x0 = np.random.uniform(bounds_phy['x0'][0], bounds_phy['x0'][1], (n_samples, 1))
    sigma = np.random.uniform(bounds_phy['sigma'][0], bounds_phy['sigma'][1], (n_samples, 1))
    k = np.random.uniform(bounds_phy['k'][0], bounds_phy['k'][1], (n_samples, 1))

    params_vec = np.hstack((v, D, mu, types, A, x0, sigma, k))

    #Collocation point
    if x_min < 0 < x_max:
        n_action = int(n_samples * 0.8)
        n_rest = n_samples - n_action
        
        #Left zone = rest because nothing expend here
        x_rest = np.random.uniform(x_min, 0.0, (n_rest, 1))
        #Right zone = action zone
        x_action = np.random.uniform(0.0, x_max, (n_action, 1))
        
        x = np.vstack((x_rest, x_action))
        np.random.shuffle(x)
    else:
        x = np.random.uniform(x_min, x_max, (n_samples, 1))

    t = np.random.uniform(0, Tmax, (n_samples, 1))
    xt = np.hstack((x, t))

    # IC points
    x_ic = np.random.uniform(x_min, x_max, (n_samples, 1))
    xt_ic = np.hstack((x_ic, np.zeros_like(x_ic)))
    
    u_true_ic = np.zeros((n_samples, 1))
    for i in range(n_samples):
        p_dict = {"type": types[i,0], "A": A[i,0], "x0": x0[i,0], "sigma": sigma[i,0], "k": k[i,0]}
        u_true_ic[i] = get_ic_value(x_ic[i,0], "mixed", p_dict)

    #BC points
    t_bc = np.random.uniform(0, Tmax, (n_samples, 1))
    xt_bc_left = np.hstack((np.full((n_samples, 1), x_min), t_bc))
    xt_bc_right = np.hstack((np.full((n_samples, 1), x_max), t_bc))

    u_true_bc_l = np.zeros((n_samples, 1))
    u_true_bc_r = np.zeros((n_samples, 1))

    for i in range(n_samples):
        if types[i, 0] == 0: # Tanh
            u_true_bc_l[i] = -1.0
            u_true_bc_r[i] =  1.0

    xt_tensor = torch.FloatTensor(xt).to(device)
    xt_tensor.requires_grad_(True)
    
    return (torch.FloatTensor(params_vec).to(device),
            xt_tensor,                                    
            torch.FloatTensor(xt_ic).to(device),     
            torch.FloatTensor(u_true_ic).to(device),
            torch.FloatTensor(xt_bc_left).to(device),   
            torch.FloatTensor(xt_bc_right).to(device),
            torch.FloatTensor(u_true_bc_l).to(device),
            torch.FloatTensor(u_true_bc_r).to(device))