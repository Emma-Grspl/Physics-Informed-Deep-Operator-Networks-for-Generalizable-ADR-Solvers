import torch

def pde_residual_adr(model, params, xt):
    """
    Calculates the residual of the Advection-Diffusion-Reaction (ADR) partial differential equation (PDE). This function is the core of the model's Physics-Informed nature. It uses automatic derivation (Autograd) to evaluate whether the network prediction satisfies the equation:
u_t + v*u_x - D*u_xx - mu*(u - u^3) = 0. The returned residual must tend towards zero during training for the model to respect the underlying physical laws. A clamping safety mechanism is integrated to stabilize the nonlinear cubic reaction term.

    Args:
        model(torch.nn.Module): The DeepONet network (PI_DeepONet_ADR) to be evaluated.
        params(torch.Tensor): Input physical parameters [N, 8] (including v, D, mu).
        xt (torch.Tensor): Spatiotemporal coordinates [N, 2] where the residual is calculated.
    Outputs:
        torch.Tensor: The residual of the PDE calculated at each point [N, 1].

    """
    xt = xt.clone().detach().requires_grad_(True)
    u = model(params, xt)

    # Autograd
    grads = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xt, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]

    # Physical parameters
    v_in  = params[:, 0:1]
    D_in  = params[:, 1:2]
    mu_in = params[:, 2:3]

    # Clamping
    u_safe = torch.clamp(u, min=-1.2, max=1.2)

    # PDE Residual
    reaction = mu_in * (u_safe - u_safe**3)
    res = u_t + v_in * u_x - D_in * u_xx - reaction

    return res