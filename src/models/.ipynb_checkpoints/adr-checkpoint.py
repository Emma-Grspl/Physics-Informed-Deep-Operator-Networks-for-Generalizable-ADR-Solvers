import torch
import torch.nn as nn
import numpy as np

class MultiScaleFourierFeatureEncoding(nn.Module):
    """
This module projects the input coordinates into a higher-dimensional space composed of sinusoidal signals. This transformation is essential for helping neural networks overcome their "spectral bias" and efficiently learn phenomena containing both global patterns and rapid variations.
    Args (Init) :
        in_features : input dimension.
        num_frequencies : total number of frequency
        scales : list of scaling factors

    Output (Forward) :
        A tensor containing the concatenation of sin and cos of the projected inputs. Its output dimension is 2 x num_frequencies.
    """
    def __init__(self, in_features, num_frequencies, scales=[1.0, 10.0]):
        super().__init__()
        self.scales = scales
        n_scales = len(scales)
        freqs_per_scale = num_frequencies // n_scales

        all_B = []
        for s in scales:
            B_s = torch.randn(in_features, freqs_per_scale) * s
            all_B.append(B_s)

        remainder = num_frequencies - (freqs_per_scale * n_scales)
        if remainder > 0:
            med_scale = np.median(scales)
            all_B.append(torch.randn(in_features, remainder) * med_scale)

        self.B = nn.Parameter(torch.cat(all_B, dim=1), requires_grad=False)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PI_DeepONet_ADR(nn.Module):
    """
    A Physics-Informed DeepONet (Deep Operator Network) designed to solve the 
    Advection-Diffusion-Reaction (ADR) equation.

    Args:
        branch_dim (int): Input dimension of the Branch net (number of physical parameters).
        trunk_dim (int): Input dimension of the Trunk net (usually 2 for x, t).
        latent_dim (int): Width of the hidden layers in both sub-networks.
        branch_layers (list of int): List defining the hidden sizes of the Branch MLP.
        trunk_layers (list of int): List defining the number of layers in the Trunk (integers are unused, length matters).
        num_fourier_features (int): Number of Fourier features for positional encoding. Set to 0 to disable.
        fourier_scales (list): Scales for the Fourier encoding.
        lb_geom (list/array): Lower bounds [x_min, t_min] for coordinate normalization.
        ub_geom (list/array): Upper bounds [x_max, t_max] for coordinate normalization.
        phy_bounds (dict): Dictionary containing min/max values for physics parameters (keys: 'v', 'D', 'mu') used for normalization.

    Forward Args:
        params (torch.Tensor): Batch of physical parameters [N, branch_dim].
                               Expected order: [v, D, mu, type, A, x0, sigma, k].
        xt (torch.Tensor): Batch of spatio-temporal coordinates [N, trunk_dim].

    Returns:
        u (torch.Tensor): The predicted solution field value at (params, xt) with shape [N, 1].
    """
    def __init__(self, 
                 branch_dim, trunk_dim, latent_dim, 
                 branch_layers, trunk_layers, 
                 num_fourier_features, fourier_scales, 
                 lb_geom, ub_geom, phy_bounds):        
        super().__init__()
    
        #Bounds
        self.register_buffer('lb_geom', torch.tensor(lb_geom, dtype=torch.float32))
        self.register_buffer('ub_geom', torch.tensor(ub_geom, dtype=torch.float32))

        self.register_buffer('v_min', torch.tensor(phy_bounds['v'][0]))
        self.register_buffer('v_max', torch.tensor(phy_bounds['v'][1]))
        self.register_buffer('D_min', torch.tensor(phy_bounds['D'][0]))
        self.register_buffer('D_max', torch.tensor(phy_bounds['D'][1]))
        self.register_buffer('mu_min', torch.tensor(phy_bounds['mu'][0]))
        self.register_buffer('mu_max', torch.tensor(phy_bounds['mu'][1]))

        self.register_buffer('A_scale', torch.tensor(2.0))      
        self.register_buffer('sigma_scale', torch.tensor(2.0)) 
        self.register_buffer('k_scale', torch.tensor(5.0))      

        #Fourier free architecture
        self.num_fourier_features = num_fourier_features

        if num_fourier_features > 0:
            #Using the encoder
            self.trunk_encoder = MultiScaleFourierFeatureEncoding(trunk_dim, num_fourier_features, fourier_scales)
            trunk_in_dim = num_fourier_features * 2
        else:
            #Don't use of the encoder
            self.trunk_encoder = None
            trunk_in_dim = trunk_dim 

        self.activation = nn.SiLU() 
        self.trunk_input_map = nn.Linear(trunk_in_dim, latent_dim)

        #branch & trunk net
        self.branch_net = self._build_branch_net(branch_dim, latent_dim, branch_layers)
        self.trunk_layers_list = nn.ModuleList()
        self.branch_transform = nn.ModuleList()

        for _ in trunk_layers:
            self.trunk_layers_list.append(nn.Linear(latent_dim, latent_dim))
            self.branch_transform.append(nn.Linear(latent_dim, 2 * latent_dim))

        self.final_layer = nn.Linear(latent_dim, 1)
        self.apply(self._init_weights)

        with torch.no_grad(): self.final_layer.weight.mul_(0.01)

    #building branch net
    def _build_branch_net(self, in_dim, out_dim, hidden_list):
        layers = []
        dims = [in_dim] + hidden_list
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(self.activation)
        layers.append(nn.Linear(dims[-1], out_dim))
        return nn.Sequential(*layers)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def normalize_tensor(self, x, min_val, max_val):
        return 2.0 * (x - min_val) / (max_val - min_val) - 1.0

    #forward
    def forward(self, params, xt):
        #Normalization
        p_v  = self.normalize_tensor(params[:, 0:1], self.v_min, self.v_max)
        p_D  = self.normalize_tensor(params[:, 1:2], self.D_min, self.D_max)
        p_mu = self.normalize_tensor(params[:, 2:3], self.mu_min, self.mu_max)

        p_type  = params[:, 3:4]
        p_A     = params[:, 4:5] / self.A_scale 
        p_x0    = torch.zeros_like(params[:, 5:6]) 
        p_sigma = params[:, 6:7] / self.sigma_scale
        p_k     = params[:, 7:8] / self.k_scale
        params_norm = torch.cat([p_v, p_D, p_mu, p_type, p_A, p_x0, p_sigma, p_k], dim=1)

        #Normalization geom
        xt_norm = self.normalize_tensor(xt, self.lb_geom, self.ub_geom)

        #Trunk processing
        if self.trunk_encoder is not None:
            #Fourier
            xt_embed = self.trunk_encoder(xt_norm)
        else:
            #Without fourier
            xt_embed = xt_norm

        #scalar product 
        context_B = self.branch_net(params_norm) 
        Z = self.activation(self.trunk_input_map(xt_embed)) 

        for layer_T, layer_B in zip(self.trunk_layers_list, self.branch_transform):
            Z_trunk = layer_T(Z)
            UV = layer_B(context_B)
            U, V = torch.split(UV, Z.shape[1], dim=1)
            Z = self.activation((1 - Z_trunk) * U + Z_trunk * V)

        u = self.final_layer(Z)
        return u
