import torch
import numpy as np

#Device detection
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps") #Mac
else:
    DEVICE = torch.device("cpu")

#IC value
def get_ic_value(x, ic_kind, ic_params):
    """
    Generates initial condition (IC) values on a given spatial grid, dynamically adapting 
    to the framework used (PyTorch or NumPy).

    Args:
        x (torch.Tensor, np.ndarray, or list): 
            The spatial coordinates on which to evaluate the function. 
            The return type determines the output type (Tensor -> Tensor, Array -> Array)
        
        ic_kind (str): 
            The name of the profile to generate. 
            - Simple modes: "tanh", "gaussian", "gaussian_sinus".
            - Complex mode: "mixed" (allows combining multiple types via masks).
        
        ic_params (dict, optional): 
            Dictionary containing the physical parameters of the function.
            Common keys:
            - 'A' (float): Amplitude.
            - 'x0' (float): Center of the distribution (shift).
            - 'sigma' (float): Width or steepness of the distribution.
            - 'k' (float): Frequency (for sinusoidal terms).
            - 'type' (Tensor/Array): Required only if ic_kind="mixed". An integer vector 
              of the same size as 'x' indicating which function type to apply where 

    Returns:
        u0 (torch.Tensor or np.ndarray): 
            The calculated initial condition values at points x. 
            Has the same type and shape as the input x.
    """
    if ic_params is None: ic_params = {}
    #torch or numpy 
    is_torch = isinstance(x, torch.Tensor)
    if not is_torch and not isinstance(x, np.ndarray):
        x = np.array(x)

    #tool configuration
    if is_torch:
        exp, sin, tanh = torch.exp, torch.sin, torch.tanh
        zeros_like = torch.zeros_like
        #torch
        cast = lambda m: m.float()
        check_any = lambda m: m.sum() > 0 
    else:
        exp, sin, tanh = np.exp, np.sin, np.tanh
        zeros_like = np.zeros_like
        #numpy
        cast = lambda m: np.asarray(m).astype(float)
        check_any = lambda m: np.any(m)

    #IC 
    if ic_kind == "mixed":
        types = ic_params.get("type") 
        A     = ic_params.get("A", 1.0)
        x0    = ic_params.get("x0", 0.0)
        sigma = ic_params.get("sigma", 0.5)
        k     = ic_params.get("k", 2.0)

        u0 = zeros_like(x)

        #0 = Tanh
        mask_0 = cast(types == 0)
        if check_any(mask_0): 
            u0 += mask_0 * A * tanh((x - x0) / (sigma + 1e-8))

        #1 & 2 = Gaussian sinus
        mask_gs = cast((types == 1) | (types == 2))
        if check_any(mask_gs):
            u0 += mask_gs * A * exp( - (x - x0)**2 / (2 * sigma**2 + 1e-8) ) * sin(k * x)

        #3&4 = Gaussian
        mask_gauss = cast((types == 3) | (types == 4))
        if check_any(mask_gauss):
            u0 += mask_gauss * A * exp( - (x - x0)**2 / (2 * sigma**2 + 1e-8) )

        return u0

    else:
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

#Generates training batches with all type of IC
def generate_mixed_batch(n_samples, bounds_phy, x_min, x_max, Tmax, difficulty=1.0, allowed_types=None):
    """
    Generates a training batch containing random physical parameters, spatial coordinates, 
    and temporal points.

    Args:
        n_samples (int): 
            The number of data points (collocation points) to generate for this batch.
        bounds_phy (dict): 
            A dictionary defining the sampling ranges for physical parameters. 
        x_min (float): 
            The lower bound of the spatial domain.
        x_max (float): 
            The upper bound of the spatial domain.
        Tmax (float): 
            The upper bound of the temporal domain (assuming time starts at t=0).
        difficulty (float, optional, default=1.0): 
            A scalar factor (typically between 0.0 and 1.0) used for curriculum learning. 
            It restricts the variance or range of the sampled parameters to make the 
            physics easier or harder to learn.
        allowed_types (list of int, optional): 
            A list of specific equation type IDs to sample from.
            If None, all defined types in the mixture are eligible.
    Returns:
        batch (dict): 
            A dictionary containing the generated tensors (usually PyTorch tensors):
            - 'x': Spatial coordinates with shape (n_samples, 1).
            - 't': Temporal coordinates with shape (n_samples, 1).
            - 'params': A dictionary or tensor containing the sampled physical coefficients.
            - 'type': An integer tensor indicating the equation type for each point.
    """
    #Physical parameters
    v = np.random.uniform(bounds_phy['v'][0], bounds_phy['v'][1], (n_samples, 1))
    D = np.random.uniform(bounds_phy['D'][0], bounds_phy['D'][1], (n_samples, 1))
    mu = np.random.uniform(bounds_phy['mu'][0], bounds_phy['mu'][1], (n_samples, 1))

    # Types
    if allowed_types is None:
        possible_types = [0, 1, 2, 3, 4] # Liste de tous tes types
    else:
        possible_types = allowed_types

    types = np.random.choice(possible_types, (n_samples, 1))

    #IC parameters
    A = np.random.uniform(0.5, 1.5, (n_samples, 1))
    x0 = np.random.uniform(-2, 2, (n_samples, 1))
    sigma = np.random.uniform(0.3, 1.0, (n_samples, 1))
    k_max = 1.0 + (3.0 * difficulty) # Max k=4 si diff=1
    k = np.random.uniform(1.0, k_max, (n_samples, 1))

    params_vec = np.hstack((v, D, mu, types, A, x0, sigma, k))

    #x,t
    x = np.random.uniform(x_min, x_max, (n_samples, 1))
    t = np.random.uniform(0, Tmax, (n_samples, 1))
    xt = np.hstack((x, t))

    #IC points
    x_ic = np.random.uniform(x_min, x_max, (n_samples, 1))
    xt_ic = np.hstack((x_ic, np.zeros_like(x_ic)))

    u_true_ic = np.zeros((n_samples, 1))
    for i in range(n_samples):
        p_dict = {"type": types[i,0], "A": A[i,0], "x0": x0[i,0], "sigma": sigma[i,0], "k": k[i,0]}
        u_true_ic[i] = get_ic_value(x_ic[i,0], "mixed", p_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps") 

    return (torch.FloatTensor(params_vec).to(device),
            torch.FloatTensor(xt).to(device),
            torch.FloatTensor(xt_ic).to(device),
            torch.FloatTensor(u_true_ic).to(device))

#For the classical solver
def get_validation_data_adr(N0, Nb, ic_kind="mixed", bc_kind="periodic", 
                            ic_kwargs=None, xL=-7, xR=7, Tmax=1.0):
    """
    Prepares the spatial and temporal grids, along with the initial state, required to run 
    the Crank-Nicolson solver.

    Args:
        N0 (int): 
            The number of spatial grid points (resolution of the x-axis).
        Nb (int): 
            The number of time steps (resolution of the t-axis).
        ic_kind (str, default="mixed"): 
            The name of the initial condition profile to apply (e.g., "gaussian", "tanh").
        bc_kind (str, default="periodic"): 
            The type of boundary conditions to enforce (e.g., "periodic", "dirichlet").
        ic_kwargs (dict, optional): 
            Specific parameters for the initial condition (Amplitude, sigma, etc.). 
            If None, default values are used.
        xL (float, default=-7): 
            The left boundary of the spatial domain.
        xR (float, default=7): 
            The right boundary of the spatial domain.
        Tmax (float, default=1.0): 
            The final time of the simulation.
    Returns:
        tuple (x, t, u0):
            - x (np.ndarray): The spatial grid vector of size N0.
            - t (np.ndarray): The temporal grid vector of size Nb.
            - u0 (np.ndarray): The initial condition values at t=0 on grid x.
    """
    if ic_kwargs is None: ic_kwargs = {}

    #IC generation (u0)
    x0_np = np.linspace(xL, xR, N0)
    u0_np = get_ic_value(x0_np, ic_kind, ic_kwargs)

    t_b_np = np.linspace(0, Tmax, Nb)

    data = {
        "x0": x0_np,      # Shape (N0,)
        "u0": u0_np,      # Shape (N0,)
        "t_b": t_b_np,    # Shape (Nb,)
        "xL": xL, "xR": xR,
        "ic_kind": ic_kind,
        "bc_kind": bc_kind
    }
    return data
