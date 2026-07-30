import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from core.config import RHO, KERNEL_SIZE, PADDING_R, PADDING_THETA, R_ROTOR, OMEGA

# =========================================================================
# 1. RÉSEAUX PRÉDICTIFS
# =========================================================================

class TurbineMLP(nn.Module):
    """ Stratégie GV : Le réseau global vectoriel (MLP classique) """
    def __init__(self, input_dim, output_dim, n_layers, n_neurons, dropout_rate, device='cpu'):
        super(TurbineMLP, self).__init__()
        layers = []
        in_features = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, n_neurons, device=device))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = n_neurons
        layers.append(nn.Linear(in_features, output_dim, device=device))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class PeriodicPadding2d(nn.Module):
    """ Padding physique respectant la périodicité de la grille polaire """
    def __init__(self, pad_r, pad_theta):
        super().__init__()
        self.pad_r = pad_r
        self.pad_theta = pad_theta

    def forward(self, x):
        x = F.pad(x, (0, 0, self.pad_r, self.pad_r), mode='replicate')
        x = F.pad(x, (self.pad_theta, self.pad_theta, 0, 0), mode='circular')
        return x

class TurbineCNN(nn.Module):
    """ Stratégie GM : Le réseau global matriciel (CNN) """
    def __init__(self, in_channels, out_channels, use_autoencoder, latent_dim, n_layers, base_filters, dropout_rate, device='cpu'):
        super(TurbineCNN, self).__init__()
        self.use_autoencoder = use_autoencoder
        layers = []
        current_channels = in_channels
        
        for i in range(n_layers):
            out_f = base_filters * (2**i)
            layers.append(PeriodicPadding2d(PADDING_R, PADDING_THETA))
            layers.append(nn.Conv2d(current_channels, out_f, kernel_size=KERNEL_SIZE, device=device))
            layers.append(nn.BatchNorm2d(out_f, device=device))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout2d(dropout_rate))
            current_channels = out_f
            
        self.feature_extractor = nn.Sequential(*layers)
        self.grid_r, self.grid_theta = 36, 72 
        
        if use_autoencoder:
            self.final_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(current_channels, latent_dim, device=device)
            )
        else:
            self.final_layer = nn.Sequential(
                PeriodicPadding2d(PADDING_R, PADDING_THETA),
                nn.Conv2d(current_channels, out_channels, kernel_size=KERNEL_SIZE, device=device)
            )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.final_layer(features)

# =========================================================================
# 2. AUTO-ENCODEURS (BANQUE DXY)
# =========================================================================

def _decreasing_hidden_sizes(in_features, latent_dim, n_layers):
    """ Suite de tailles de couches cachées, décroissant géométriquement de in_features vers latent_dim. """
    sizes = np.geomspace(in_features, latent_dim, num=n_layers + 2)[1:-1]
    return np.round(sizes).astype(int).tolist()

class LinearAutoencoder(nn.Module):
    """ Auto-encodeur pour la stratégie GV (1D) """
    def __init__(self, in_features=5184, latent_dim=32, n_layers=2, device='cpu'):
        super(LinearAutoencoder, self).__init__()
        hidden_sizes = _decreasing_hidden_sizes(in_features, latent_dim, n_layers)

        enc_layers = []
        prev = in_features
        for h in hidden_sizes:
            enc_layers += [nn.Linear(prev, h, device=device), nn.ReLU()]
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim, device=device))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_sizes):
            dec_layers += [nn.Linear(prev, h, device=device), nn.ReLU()]
            prev = h
        dec_layers.append(nn.Linear(prev, in_features, device=device))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))
        
    def decode(self, z):
        return self.decoder(z)
        
    def encode(self, x):
        return self.encoder(x)

class ConvolutionalAutoencoder(nn.Module):
    """ Auto-encodeur pour la stratégie GM (2D) """
    def __init__(self, in_channels=2, latent_dim=32, depth=3, base_filters=16, device='cpu'):
        super(ConvolutionalAutoencoder, self).__init__()
        self.grid_r, self.grid_theta = 36, 72 
        
        enc_layers = []
        current_channels = in_channels
        for i in range(depth):
            out_f = base_filters * (2**i)
            enc_layers.append(PeriodicPadding2d(1, 1))
            enc_layers.append(nn.Conv2d(current_channels, out_f, kernel_size=3, stride=2, device=device))
            enc_layers.append(nn.BatchNorm2d(out_f, device=device))
            enc_layers.append(nn.ReLU())
            current_channels = out_f
            
        self.encoder_conv = nn.Sequential(*enc_layers)
        
        # Calcul dynamique de la taille aplatie (évite les erreurs d'arrondi du padding)
        with torch.no_grad():
            dummy_x = torch.zeros(1, in_channels, self.grid_r, self.grid_theta, device=device)
            dummy_out = self.encoder_conv(dummy_x)
            flattened_size = dummy_out.numel()
            self.flat_r = dummy_out.size(2)
            self.flat_theta = dummy_out.size(3)
        
        self.encoder_fc = nn.Linear(flattened_size, latent_dim, device=device)
        self.decoder_fc = nn.Linear(latent_dim, flattened_size, device=device)
        self.current_channels = current_channels
        
        dec_layers = []
        for i in range(depth - 1, -1, -1):
            out_f = base_filters * (2**(i-1)) if i > 0 else in_channels
            dec_layers.append(nn.ConvTranspose2d(current_channels, out_f, kernel_size=4, stride=2, padding=1, device=device))
            if i > 0:
                dec_layers.append(nn.BatchNorm2d(out_f, device=device))
                dec_layers.append(nn.ReLU())
            current_channels = out_f
            
        self.decoder_conv = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        features = self.encoder_conv(x)
        features_flat = features.reshape(features.size(0), -1)
        return self.encoder_fc(features_flat)

    def decode(self, z):
        x_rec_flat = self.decoder_fc(z)
        x_rec_reshaped = x_rec_flat.reshape(x_rec_flat.size(0), self.current_channels, self.flat_r, self.flat_theta)
        x_rec = self.decoder_conv(x_rec_reshaped)
        if x_rec.size(2) != self.grid_r or x_rec.size(3) != self.grid_theta:
             x_rec = F.interpolate(x_rec, size=(self.grid_r, self.grid_theta), mode='bilinear', align_corners=False)
        return x_rec

# =========================================================================
# 3. GESTIONNAIRE D'ÉCHELLES (SCALERS) PYTORCH
# =========================================================================

class TorchScaler:
    """ Transforme un StandardScaler sklearn en opérations PyTorch différentiables """
    def __init__(self, sklearn_scaler, device):
        if isinstance(sklearn_scaler, TorchScaler):
            self.mean = sklearn_scaler.mean.clone().to(device)
            self.scale = sklearn_scaler.scale.clone().to(device)
        else:
            mean_val = getattr(sklearn_scaler, 'mean_', None)
            if mean_val is not None:
                self.mean = torch.tensor(mean_val, dtype=torch.float32, device=device)
            else:
                self.mean = torch.tensor(0.0, dtype=torch.float32, device=device)
            
            scale_val = getattr(sklearn_scaler, 'scale_', None)
            if scale_val is not None:
                self.scale = torch.tensor(scale_val, dtype=torch.float32, device=device)
            else:
                self.scale = torch.tensor(1.0, dtype=torch.float32, device=device)
        
    def inverse_transform(self, tensor_norm):
        orig_shape = tensor_norm.shape
        tensor_flat = tensor_norm.reshape(orig_shape[0], -1)
        tensor_phys = tensor_flat * self.scale + self.mean
        return tensor_phys.reshape(orig_shape)
        
    def transform(self, tensor_phys):
        orig_shape = tensor_phys.shape
        tensor_flat = tensor_phys.reshape(orig_shape[0], -1)
        tensor_norm = (tensor_flat - self.mean) / self.scale
        return tensor_norm.reshape(orig_shape)

# =========================================================================
# 4. MOTEUR PHYSIQUE DIFFÉRENTIABLE
# =========================================================================

def compute_cp_ct_torch(Fn_phys, Ft_phys, r_tensor, u_inf, is_cnn=False):
    """
    Intégration différentiable de Cp et Ct.
    """
    from core.config import RHO, R_ROTOR, OMEGA, PITCH_RAD
    from core.physics import get_geometry
    import torch
    import numpy as np
    
    geom = get_geometry()
    device = Fn_phys.device
    
    # 1. Reconstruction du twist_rad et phi
    r_np = r_tensor.detach().cpu().numpy()
    twist_rad = torch.tensor(geom.get_twist_rad(r_np), dtype=torch.float32, device=device)
    phi_rad = PITCH_RAD + twist_rad
    
    # 2. Projection Aérodynamique
    Ft_corrige = Ft_phys * -1.0
    cos_phi = torch.cos(phi_rad)
    sin_phi = torch.sin(phi_rad)
    
    dQ_r = Fn_phys * sin_phi + Ft_corrige * cos_phi
    dT_r = Fn_phys * cos_phi - Ft_corrige * sin_phi
    
    # 3. CALCUL DU DL NON UNIFORME
    if is_cnn:
        r_unique = r_np[0, :, 0] if r_tensor.dim() == 3 else r_np[:, 0]
    else:
        # En GV, r_tensor est 1D (2592), trié par theta puis r. 
        # Les 36 premiers éléments sont les rayons (r) uniques !
        r_unique = r_np[:36] 
        
    nodes = [0.21]
    for i in range(len(r_unique)):
        next_node = 2 * r_unique[i] - nodes[-1]
        nodes.append(next_node)
    dl_np = np.diff(nodes)
    dl_tensor = torch.tensor(dl_np, dtype=torch.float32, device=device)
    
    # 4. Intégration sur la pale
    Nb_pales = 3.0
    if is_cnn:
        dl_reshaped = dl_tensor.view(1, -1, 1) # [1, 36, 1]
        Q_theta = torch.sum(dQ_r * r_tensor * dl_reshaped, dim=1) 
        T_theta = torch.sum(dT_r * dl_reshaped, dim=1) 
        Q_mean = torch.mean(Q_theta, dim=1) * Nb_pales
        T_mean = torch.mean(T_theta, dim=1) * Nb_pales
    else:
        # dQ_r est [Batch, 2592]. On le reformate en [Batch, 72 (theta), 36 (r)]
        dQ_r_2d = dQ_r.view(-1, 72, 36)
        dT_r_2d = dT_r.view(-1, 72, 36)
        r_2d = r_tensor.view(72, 36)
        dl_reshaped = dl_tensor.view(1, 1, 36) # [1, 1, 36]
        
        # Intégration spatiale sur le rayon (dim=2)
        Q_theta = torch.sum(dQ_r_2d * r_2d * dl_reshaped, dim=2)
        T_theta = torch.sum(dT_r_2d * dl_reshaped, dim=2)
        
        # Moyenne temporelle/azimutale sur la rotation (dim=1)
        Q_mean = torch.mean(Q_theta, dim=1) * Nb_pales
        T_mean = torch.mean(T_theta, dim=1) * Nb_pales
        
    # 5. Calcul des coefficients Cp et Ct avec U_inf (Vent amont)
    A = torch.pi * (R_ROTOR**2)
    denom_p = 0.5 * RHO * (u_inf**3) * A
    denom_t = 0.5 * RHO * (u_inf**2) * A
    
    Cp = (Q_mean * OMEGA) / denom_p
    Ct = T_mean / denom_t
    
    return Cp, Ct

class PolarSurrogate(nn.Module):
    """ Interpolateur 1D vectorisé par section de pale, différentiable (Akima Spline). """
    def __init__(self, device='cpu', csv_path="data/geometry/airfoils.csv"):
        super().__init__()
        df = pd.read_csv(csv_path)
        self.alphas_np = np.sort(df['alpha_deg'].unique())
        self.radii_np = np.sort(df['r'].unique())
        
        df_cl = df.pivot(index='alpha_deg', columns='r', values='Cl')
        df_cd = df.pivot(index='alpha_deg', columns='r', values='Cd')
        
        self.register_buffer('x', torch.tensor(self.alphas_np, dtype=torch.float32, device=device))
        self.register_buffer('y_cl', torch.tensor(df_cl.values, dtype=torch.float32, device=device).T)
        self.register_buffer('y_cd', torch.tensor(df_cd.values, dtype=torch.float32, device=device).T)
        self.register_buffer('radii_grid', torch.tensor(self.radii_np, dtype=torch.float32, device=device))

        self.t_cl = self._compute_akima_derivatives(self.x, self.y_cl)
        self.t_cd = self._compute_akima_derivatives(self.x, self.y_cd)

    def _compute_akima_derivatives(self, x, y):
        dx = x[1:] - x[:-1] 
        dy = y[:, 1:] - y[:, :-1] 
        m = dy / dx 
        m_pad = torch.zeros((y.shape[0], m.shape[1] + 4), device=x.device, dtype=x.dtype)
        m_pad[:, 2:-2] = m
        m_pad[:, 1] = 2 * m[:, 0] - m[:, 1]
        m_pad[:, 0] = 2 * m_pad[:, 1] - m[:, 0]
        m_pad[:, -2] = 2 * m[:, -1] - m[:, -2]
        m_pad[:, -1] = 2 * m_pad[:, -2] - m[:, -1]

        dm = torch.abs(m_pad[:, 1:] - m_pad[:, :-1])
        w_right, w_left = dm[:, :-2], dm[:, 2:]     

        denom = w_left + w_right
        mask = denom == 0
        t = torch.zeros_like(y)
        t_num = w_right * m_pad[:, 1:-2] + w_left * m_pad[:, 2:-1]
        
        t[~mask] = t_num[~mask] / denom[~mask]
        t[mask] = 0.5 * (m_pad[:, 1:-2][mask] + m_pad[:, 2:-1][mask])
        return t

    def _evaluate_spline(self, x_new, r_flat, y_grid, t_grid):
        dists = torch.abs(r_flat.unsqueeze(1) - self.radii_grid.unsqueeze(0))
        idx_r = torch.argmin(dists, dim=1) 
        idx_x = torch.searchsorted(self.x, x_new) - 1
        idx_x = torch.clamp(idx_x, 0, len(self.x) - 2)
        
        x0, x1 = self.x[idx_x], self.x[idx_x + 1]
        dx = x1 - x0
        y0, y1 = y_grid[idx_r, idx_x], y_grid[idx_r, idx_x + 1]
        t0, t1 = t_grid[idx_r, idx_x], t_grid[idx_r, idx_x + 1]
        
        t = (x_new - x0) / dx
        t2, t3 = t * t, t * t * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        
        return h00 * y0 + h10 * dx * t0 + h01 * y1 + h11 * dx * t1

    def forward(self, alpha_deg, r_tensor):
        shape_orig = alpha_deg.shape
        alpha_flat = alpha_deg.contiguous().flatten()
        r_flat = r_tensor.contiguous().flatten()
        
        cl_interp = self._evaluate_spline(alpha_flat, r_flat, self.y_cl, self.t_cl)
        cd_interp = self._evaluate_spline(alpha_flat, r_flat, self.y_cd, self.t_cd)
        return torch.stack([cl_interp.view(shape_orig), cd_interp.view(shape_orig)], dim=-1)

def convert_v_to_f_torch(v_eff, alpha_deg, r_tensor, c_tensor, polar_surrogate, rho=RHO):
    """
    Convertit les sorties de l'interpolateur Akima (Cl, Cd) en efforts (Fn, Ft) en [N/m].
    """
    alpha_rad = alpha_deg * (torch.pi / 180.0)
    
    r_expanded = r_tensor.expand_as(alpha_deg)
    c_expanded = c_tensor.expand_as(alpha_deg)
    
    # Récupération des coefficients aérodynamiques via l'interpolation Akima
    coeffs = polar_surrogate(alpha_deg, r_expanded)
    cl, cd = coeffs[..., 0], coeffs[..., 1]
    
    # Projections géométriques (Cn, Ct)
    cn = cl * torch.cos(alpha_rad) + cd * torch.sin(alpha_rad)
    ct = cl * torch.sin(alpha_rad) - cd * torch.cos(alpha_rad)
    
    # Calcul de la pression dynamique locale et des forces (N/m)
    q = 0.5 * rho * (v_eff**2) * torch.abs(c_expanded)
    fn = q * cn
    ft = -q * ct  # Convention de signe identique à physics.py
    
    return torch.stack([fn, ft], dim=-1)

# =========================================================================
# 5. UTILITAIRES DE CONVERSION FORMAT Y (CNN <-> MLP)
# =========================================================================

def y_to_cnn_format(y: torch.Tensor) -> torch.Tensor:
    """(N, 5184) GM flat (channels-first, r-major) -> (N, 2, 36, 72). Simple reshape."""
    return y.reshape(y.size(0), 2, 36, 72)

def gv_to_gm_format(y: torch.Tensor) -> torch.Tensor:
    """(N, 5184) GV interleaved (theta-major) -> (N, 2, 36, 72) GM (r-major, channels-first)."""
    N = y.size(0)
    y_3d = y.reshape(N, 72, 36, 2)
    an = y_3d[:, :, :, 0].permute(0, 2, 1)  # (N, 36, 72)
    at = y_3d[:, :, :, 1].permute(0, 2, 1)
    return torch.stack([an, at], dim=1)      # (N, 2, 36, 72)

def y_to_mlp_format(y: torch.Tensor) -> torch.Tensor:
    """(N, 2, 36, 72) GM (r-major, channels-first) -> (N, 5184) GV interleaved (theta-major)."""
    N = y.size(0)
    an = y[:, 0].permute(0, 2, 1)           # (N, 72, 36)
    at = y[:, 1].permute(0, 2, 1)
    return torch.stack([an, at], dim=3).reshape(N, -1)

def adapt_ae_output_to_target(decoded: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Adapte la sortie décodée de l'AE au format de la cible Y du modèle prédictif."""
    if target.dim() == 4 and decoded.dim() == 2:
        return y_to_cnn_format(decoded)      # LinearAE GM flat -> GM 4D
    if target.dim() == 2 and decoded.dim() == 4:
        return y_to_mlp_format(decoded)      # ConvAE GM 4D -> GV interleaved
    if target.dim() == 2 and decoded.dim() == 2:
        return y_to_mlp_format(y_to_cnn_format(decoded))  # LinearAE GM flat -> GV interleaved
    return decoded

# =========================================================================
# 6. NOUVELLE UNIFIED LOSS (A & B)
# =========================================================================

class TurbineLoss(nn.Module):
    """ Loss globale implémentant strictement les architectures A et B. """
    def __init__(self, inter, loss_type, l1, l2, l3, ae_model, scaler_Y, 
                 r_tensor, c_tensor, polar_surrogate=None, device='cpu'):
        super().__init__()
        self.inter = inter
        self.loss_type = loss_type.upper() # 'A' ou 'B'
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        
        # NB: on n'assigne pas ae_model via self.ae_model = ... car nn.Module
        # l'enregistrerait comme sous-module : criterion.train()/.eval() le
        # basculerait alors en mode train et réactiverait la BatchNorm de l'AE.
        # object.__setattr__ contourne l'enregistrement tout en gardant l'attribut accessible.
        object.__setattr__(self, 'ae_model', ae_model)
        if self.ae_model is not None:
            self.ae_model.eval()
            for p in self.ae_model.parameters():
                p.requires_grad_(False)
        self.scaler_Y = TorchScaler(scaler_Y, device) if scaler_Y else None
        
        self.r_tensor = r_tensor.to(device)
        self.c_tensor = c_tensor.to(device)
        self.polar_surrogate = polar_surrogate.to(device) if polar_surrogate else None
        self.eps_t = 1.0

    def forward(self, preds_raw, Y_true_norm, D_phys=None, v_bem_phys=None, f_bem_phys=None, v_app=None, u_inf=None, is_cnn=False):

        is_cnn_auto = (Y_true_norm.dim() == 4)

        # 1. Décodage AE si nécessaire
        if self.ae_model is not None:
            preds_norm = adapt_ae_output_to_target(self.ae_model.decode(preds_raw), Y_true_norm)
        else:
            preds_norm = preds_raw
            
        # 2. Dénormalisation (Fn/D, Ft/D ou an, at)
        coeffs_pred = self.scaler_Y.inverse_transform(preds_norm)
        coeffs_true = self.scaler_Y.inverse_transform(Y_true_norm)
        
        loss_v = 0.0
        loss_f = 0.0
        loss_macro = 0.0
        Fn_p_abs, Ft_p_abs, Fn_t_abs, Ft_t_abs = None, None, None, None

        # ==============================================================
        # STRATÉGIE 'v' (Vitesses -> Forces -> Cp/Ct)
        # ==============================================================
        if self.inter == 'v':
            # Terme L3 : Erreur sur les vitesses
            if self.l3 > 0:
                an_p, at_p = (coeffs_pred[:,0], coeffs_pred[:,1]) if is_cnn_auto else (coeffs_pred[:,0::2], coeffs_pred[:,1::2])
                an_t, at_t = (coeffs_true[:,0], coeffs_true[:,1]) if is_cnn_auto else (coeffs_true[:,0::2], coeffs_true[:,1::2])
                
                if self.loss_type == 'A': # Relative locale pure
                    err_an = torch.mean(((an_p - an_t) / torch.abs(an_t))**2)
                    err_at = torch.mean(((at_p - at_t) / torch.abs(at_t))**2)
                    loss_v = err_an + err_at
                else: # Loss B : Normalisée par V_app
                    loss_v = nn.functional.mse_loss(coeffs_pred, coeffs_true)

            # Reconstruction des Forces Absolues pour L1 et L2
            if self.l1 > 0 or self.l2 > 0:
                if v_bem_phys is not None:
                    coeffs_pred = coeffs_pred + v_bem_phys
                    coeffs_true = coeffs_true + v_bem_phys
                    
                an_p, at_p = (coeffs_pred[:,0], coeffs_pred[:,1]) if is_cnn_auto else (coeffs_pred[:,0::2], coeffs_pred[:,1::2])
                an_t, at_t = (coeffs_true[:,0], coeffs_true[:,1]) if is_cnn_auto else (coeffs_true[:,0::2], coeffs_true[:,1::2])
                
                alpha_p_deg = torch.atan2(an_p, at_p) * (180.0 / torch.pi)
                alpha_t_deg = torch.atan2(an_t, at_t) * (180.0 / torch.pi)
                v_eff_p = v_app * torch.sqrt(an_p**2 + at_p**2)
                v_eff_t = v_app * torch.sqrt(an_t**2 + at_t**2)
                
                f_pred = convert_v_to_f_torch(v_eff_p, alpha_p_deg, self.r_tensor, self.c_tensor, self.polar_surrogate)
                f_true = convert_v_to_f_torch(v_eff_t, alpha_t_deg, self.r_tensor, self.c_tensor, self.polar_surrogate)
                
                Fn_p_abs, Ft_p_abs = f_pred[..., 0], f_pred[..., 1]
                Fn_t_abs, Ft_t_abs = f_true[..., 0], f_true[..., 1]
                
                # Terme L2 (Forces pour 'v')
                if self.l2 > 0:
                    if self.loss_type == 'A': # Relative locale
                        err_fn = torch.mean(((Fn_p_abs - Fn_t_abs) / torch.abs(Fn_t_abs))**2)
                        err_ft = torch.mean(((Ft_p_abs - Ft_t_abs) / torch.clamp(torch.abs(Ft_t_abs), min=self.eps_t))**2)
                        loss_f = err_fn + err_ft
                    else: # Loss B : Normalisation physique (Fn/D)
                        D_phys_local = 0.5 * RHO * (v_app**2) * torch.abs(self.c_tensor)
                        err_fn_D = torch.mean(((Fn_p_abs/D_phys_local) - (Fn_t_abs/D_phys_local))**2)
                        err_ft_D = torch.mean(((Ft_p_abs/D_phys_local) - (Ft_t_abs/D_phys_local))**2)
                        loss_f = err_fn_D + err_ft_D

        # ==============================================================
        # STRATÉGIE 'f' (Forces -> Cp/Ct)
        # ==============================================================
        else:
            if f_bem_phys is not None:
                coeffs_pred = coeffs_pred + f_bem_phys
                coeffs_true = coeffs_true + f_bem_phys

            Fn_p_norm, Ft_p_norm = (coeffs_pred[:,0], coeffs_pred[:,1]) if is_cnn_auto else (coeffs_pred[:,0::2], coeffs_pred[:,1::2])
            Fn_t_norm, Ft_t_norm = (coeffs_true[:,0], coeffs_true[:,1]) if is_cnn_auto else (coeffs_true[:,0::2], coeffs_true[:,1::2])
            
            D_val = D_phys if is_cnn_auto else D_phys[:, 0::2]
            
            Fn_p_abs = Fn_p_norm * D_val
            Ft_p_abs = Ft_p_norm * D_val
            Fn_t_abs = Fn_t_norm * D_val
            Ft_t_abs = Ft_t_norm * D_val

            if self.l2 > 0:
                if self.loss_type == 'A': # Relative locale
                    err_fn = torch.mean(((Fn_p_abs - Fn_t_abs) / torch.abs(Fn_t_abs))**2)
                    err_ft = torch.mean(((Ft_p_abs - Ft_t_abs) / torch.clamp(torch.abs(Ft_t_abs), min=self.eps_t))**2)
                    loss_f = err_fn + err_ft
                else: # Loss B : Normalisation physique (MSE sur Fn/D, Ft/D)
                    loss_f = nn.functional.mse_loss(coeffs_pred, coeffs_true)

        # ==============================================================
        # Terme L1 : Erreurs intégrées (Cp, Ct)
        # ==============================================================
        if self.l1 > 0 and Fn_p_abs is not None:
            Cp_p, Ct_p = compute_cp_ct_torch(Fn_p_abs, Ft_p_abs, self.r_tensor, u_inf, is_cnn_auto)
            Cp_t, Ct_t = compute_cp_ct_torch(Fn_t_abs, Ft_t_abs, self.r_tensor, u_inf, is_cnn_auto)
            
            err_cp = torch.mean(((Cp_p - Cp_t) / torch.abs(Cp_t))**2)
            err_ct = torch.mean(((Ct_p - Ct_t) / torch.abs(Ct_t))**2)
            loss_macro = err_cp + err_ct

        return self.l1 * loss_macro + self.l2 * loss_f + self.l3 * loss_v