import torch

# pde residual
def pde_residual_adr(model, params, xt):
    """
    Compute the residual on the pde
    Arg : 
        - model : PI_DeepOnet_ADR
        - params : (v,D,mu,sigma,k,x0,type,A)
    xt : xt

    Output : 
    res = u_t + v_in * u_x - D_in *u_xx - mu_in * (u-u_safe**3)
    """
    xt = xt.clone().detach().requires_grad_(True)

    u = model(params, xt)

    #Autograd
    grads = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xt, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]

    v_in  = params[:, 0:1]
    D_in  = params[:, 1:2]
    mu_in = params[:, 2:3]

    # An anti-explosion term is fixed on the cubed term to avoid Nan
    u_safe = torch.clamp(u, min=-1.5, max=1.5)

    # Reaction term: mu * (u - u^3)
    reaction = mu_in * (u - u_safe**3)
    res = u_t + v_in * u_x - D_in * u_xx - reaction

    return res
