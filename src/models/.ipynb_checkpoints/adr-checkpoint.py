import torch
import torch.nn as nn
import numpy as np
from config import Config

class MultiScaleFourierFeatureEncoding(nn.Module):
    """
    Module de projection Fourier multi-échelles.
    Utilise Config.sFourier par défaut si scales n'est pas fourni.
    """
    def __init__(self, in_features, num_frequencies, scales=None):
        super().__init__()
        # Utilisation des scales du Config si non fournis
        if scales is None:
            scales = Config.sFourier
            
        self.scales = scales
        n_scales = len(scales)
        
        # Sécurité pour éviter division par zéro
        if n_scales == 0: 
            self.register_parameter("B", None)
            return

        freqs_per_scale = num_frequencies // n_scales

        all_B = []
        for s in scales:
            # Initialisation aléatoire gaussienne normale
            B_s = torch.randn(in_features, freqs_per_scale) * s
            all_B.append(B_s)

        # Gestion du reste si la division n'est pas parfaite
        remainder = num_frequencies - (freqs_per_scale * n_scales)
        if remainder > 0:
            med_scale = np.median(scales)
            all_B.append(torch.randn(in_features, remainder) * med_scale)

        self.B = nn.Parameter(torch.cat(all_B, dim=1), requires_grad=False)

    def forward(self, x):
        # Projection : x [N, in] @ B [in, num_freq] -> [N, num_freq]
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PI_DeepONet_ADR(nn.Module):
    """
    Physics-Informed DeepONet (Deep Operator Network) pour l'équation ADR.
    Entièrement piloté par Config pour l'architecture et la normalisation.
    """
    def __init__(self):          
        super().__init__()
        
        # --- 1. Récupération Paramètres Architecture depuis Config ---
        # Dimensions d'entrée (Physique fixée)
        self.input_branch_dim = 8  # [v, D, mu, type, A, x0, sigma, k]
        self.input_trunk_dim = 2   # [x, t]
        
        # Dimensions cachées (Listes d'entiers depuis Config)
        branch_layers_list = Config.branch_dim  # Ex: [256, 256, 256, 256]
        trunk_layers_list = Config.trunk_dim    # Ex: [256, 256, 256]
        latent_dim = Config.latent_dim          # Ex: 256
        
        # Fourier
        num_fourier = Config.nFourier
        scales_fourier = Config.sFourier

        # --- 2. Enregistrement des Buffers de Normalisation (Config Driven) ---
        # Géométrie
        self.register_buffer('lb_geom', torch.tensor([Config.x_min, 0.0], dtype=torch.float32))
        self.register_buffer('ub_geom', torch.tensor([Config.x_max, Config.T_max], dtype=torch.float32))

        # Physique (Standardization [-1, 1])
        r = Config.ranges
        self.register_buffer('v_min', torch.tensor(r['v'][0]))
        self.register_buffer('v_max', torch.tensor(r['v'][1]))
        self.register_buffer('D_min', torch.tensor(r['D'][0]))
        self.register_buffer('D_max', torch.tensor(r['D'][1]))
        self.register_buffer('mu_min', torch.tensor(r['mu'][0]))
        self.register_buffer('mu_max', torch.tensor(r['mu'][1]))

        # Physique (Scaling simple par division)
        # On calcule le scale max pour normaliser grossièrement autour de 1
        # On prend max(|min|, |max|) pour être sûr
        self.register_buffer('A_scale', torch.tensor(max(abs(r['A'][0]), abs(r['A'][1]))))      
        self.register_buffer('sigma_scale', torch.tensor(max(abs(r['sigma'][0]), abs(r['sigma'][1])))) 
        self.register_buffer('k_scale', torch.tensor(max(abs(r['k'][0]), abs(r['k'][1]))))      

        # --- 3. Construction Trunk (Encoder + MLP) ---
        self.num_fourier_features = num_fourier

        if num_fourier > 0:
            self.trunk_encoder = MultiScaleFourierFeatureEncoding(
                self.input_trunk_dim, num_fourier, scales_fourier
            )
            # Output de l'encoder est sin + cos -> 2 * num_fourier
            trunk_in_dim_actual = num_fourier * 2
        else:
            self.trunk_encoder = None
            trunk_in_dim_actual = self.input_trunk_dim 

        self.activation = nn.SiLU() 
        
        # Mapper vers l'espace latent
        self.trunk_input_map = nn.Linear(trunk_in_dim_actual, latent_dim)

        # --- 4. Construction Branch & Trunk Layers ---
        self.branch_net = self._build_branch_net(self.input_branch_dim, latent_dim, branch_layers_list)
        
        self.trunk_layers_list = nn.ModuleList()
        self.branch_transform = nn.ModuleList()

        # Architecture modifiée : Branch modulant Trunk à chaque couche
        # Note : On utilise la liste définie dans trunk_layers_list (config)
        for _ in trunk_layers_list:
            self.trunk_layers_list.append(nn.Linear(latent_dim, latent_dim))
            # La branche produit scales & biases pour la couche trunk correspondante -> 2 * latent
            self.branch_transform.append(nn.Linear(latent_dim, 2 * latent_dim))

        self.final_layer = nn.Linear(latent_dim, 1)
        
        # Initialisation
        self.apply(self._init_weights)
        with torch.no_grad(): 
            self.final_layer.weight.mul_(0.01)

    def _build_branch_net(self, in_dim, out_dim, hidden_list):
        layers = []
        dims = [in_dim] + hidden_list
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(self.activation)
        # Dernière couche de projection vers l'espace latent contextuel
        layers.append(nn.Linear(dims[-1], out_dim))
        return nn.Sequential(*layers)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def normalize_tensor(self, x, min_val, max_val):
        # Scaling vers [-1, 1]
        return 2.0 * (x - min_val) / (max_val - min_val) - 1.0

    def forward(self, params, xt):
        """
        params: [batch, 8] -> v, D, mu, type, A, x0, sigma, k
        xt: [batch, 2] -> x, t
        """
        # --- Normalisation Paramètres ---
        p_v  = self.normalize_tensor(params[:, 0:1], self.v_min, self.v_max)
        p_D  = self.normalize_tensor(params[:, 1:2], self.D_min, self.D_max)
        p_mu = self.normalize_tensor(params[:, 2:3], self.mu_min, self.mu_max)

        p_type  = params[:, 3:4] # On laisse tel quel ou on pourrait normaliser si besoin
        p_A     = params[:, 4:5] / self.A_scale 
        
        # Note : x0 est mis à zéro (conformément à votre code original)
        # Cela force le modèle à apprendre l'invariance par translation relative
        p_x0    = torch.zeros_like(params[:, 5:6]) 
        
        p_sigma = params[:, 6:7] / self.sigma_scale
        p_k     = params[:, 7:8] / self.k_scale
        
        params_norm = torch.cat([p_v, p_D, p_mu, p_type, p_A, p_x0, p_sigma, p_k], dim=1)

        # --- Normalisation Géométrie ---
        xt_norm = self.normalize_tensor(xt, self.lb_geom, self.ub_geom)

        # --- Trunk Processing ---
        if self.trunk_encoder is not None:
            xt_embed = self.trunk_encoder(xt_norm)
        else:
            xt_embed = xt_norm

        # --- DeepONet Forward ---
        # 1. Encodage du contexte physique par la branche
        context_B = self.branch_net(params_norm) 
        
        # 2. Entrée initiale du trunk
        Z = self.activation(self.trunk_input_map(xt_embed)) 

        # 3. Interaction dynamique couche par couche
        for layer_T, layer_B in zip(self.trunk_layers_list, self.branch_transform):
            Z_trunk = layer_T(Z)
            
            # La branche prédit les coefficients de modulation (FiLM like)
            UV = layer_B(context_B)
            U, V = torch.split(UV, Z.shape[1], dim=1)
            
            # Application de la modulation : (1-Z)*U + Z*V (Gating mechanism)
            Z = self.activation((1 - Z_trunk) * U + Z_trunk * V)

        u = self.final_layer(Z)
        return u