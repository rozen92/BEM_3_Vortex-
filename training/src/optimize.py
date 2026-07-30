import gc
import optuna
import json
import torch
import os
import numpy as np
import pickle
from core.models import TurbineMLP, TurbineCNN, ConvolutionalAutoencoder, LinearAutoencoder, PolarSurrogate, TurbineLoss, TorchScaler, convert_v_to_f_torch, adapt_ae_output_to_target, gv_to_gm_format
from training.src.data_loader import format_data, get_D_tensor, get_V_app_tensor, format_bem_as_Y
from training.src.trainer import cross_validate
from core.physics import get_geometry, compute_dynamic_pressure_D
from core.config import EPOCHS_OPTUNA, CV_SPLITS, LR_BOUNDS_NOAE, LR_BOUNDS_AE, DROPOUT_BOUNDS, MLP_LAYERS_BOUNDS, MLP_NEURONS_CHOICES, CNN_LAYERS_BOUNDS, CNN_FILTERS_CHOICES, PRUNER_WARMUP, AE_NATURES, AE_DIMS, AE_JSON_PATH, AE_WEIGHTS_DIR, RHO, OMEGA, R_ROTOR, format_scaler_name, format_ae_key

def get_u_inf_tensor(df, device='cpu'):
    u_inf_list = []
    for _, group in df.groupby(['yaw', 'TSR'] if 'TSR' in df.columns else 'yaw'):
        tsr_val = group['TSR'].iloc[0] if 'TSR' in group.columns else 8.0
        u_inf = (OMEGA * R_ROTOR) / tsr_val
        u_inf_list.append(u_inf)
    return torch.tensor(u_inf_list, dtype=torch.float32, device=device)

def optimize(df_train, entree, residuelle, inter, has_ae, option, model_base_name, n_trials=40, bem_suffix=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    has_plus = '+' in str(residuelle)
    os.makedirs("training/hyperparametres", exist_ok=True)

    X_full, Y_full = format_data(df_train, entree, residuelle, inter, is_train=True, device=device, bem_suffix=bem_suffix)
    is_cnn = (entree == 'GM')

    with open(f"training/scalers/scaler_Y_{format_scaler_name(entree, residuelle, inter, bem_suffix)}.pkl", 'rb') as f:
        scaler_Y = pickle.load(f)
    scaler_Y_torch = TorchScaler(scaler_Y, device)

    # Précalcul du BEM dans l'espace Y (pour encodage dans objective_model si '2+')
    Y_bem_full = format_bem_as_Y(df_train, entree, inter, scaler_Y, bem_suffix, device) if has_plus else None
    n_scalaires_full = (2 if 'TSR' in df_train.columns else 1) if has_plus else 0

    u_inf_full = get_u_inf_tensor(df_train, device)

    # =====================================================================
    # 1. TENSEURS GÉOMÉTRIQUES 
    # =====================================================================
    geom = get_geometry()
    if is_cnn:
        r_uniques = np.sort(df_train['r'].unique())
        theta_uniques = np.sort(df_train['theta'].unique())
        R_grid, _ = np.meshgrid(r_uniques, theta_uniques, indexing='ij')
        r_tensor = torch.tensor(R_grid, dtype=torch.float32, device=device)
        c_grid = np.array([geom.get_chord(r) for r in r_uniques])
        C_grid, _ = np.meshgrid(c_grid, theta_uniques, indexing='ij')
        c_tensor = torch.tensor(C_grid, dtype=torch.float32, device=device)
    else:
        group = df_train[(df_train['yaw'] == df_train['yaw'].iloc[0])]
        if 'TSR' in group.columns: group = group[group['TSR'] == group['TSR'].iloc[0]]
        group = group.sort_values(['theta', 'r'])
        r_array = group['r'].values
        r_tensor = torch.tensor(r_array, dtype=torch.float32, device=device)
        c_tensor = torch.tensor(np.array([geom.get_chord(r) for r in r_array]), dtype=torch.float32, device=device)

    # =====================================================================
    # 2. TENSEURS SPÉCIFIQUES SELON LA STRATÉGIE
    # =====================================================================
    polar_surrogate = None
    V_BEM_phys_full = None
    F_BEM_phys_full = None 
    V_app_full = None
    D_full = None

    if inter == 'v':
        polar_surrogate = PolarSurrogate(device=device).to(device)
        V_app_full = get_V_app_tensor(df_train, entree, device)

        if str(residuelle) in ['1']:
            _, Y_full_abs_scaled = format_data(df_train, entree, '0', inter, is_train=True, device=device)
            with open(f"training/scalers/scaler_Y_{entree}_0_v.pkl", 'rb') as f_abs: scaler_v_abs = pickle.load(f_abs)
            an_at_sven = TorchScaler(scaler_v_abs, device).inverse_transform(Y_full_abs_scaled)
            an_at_delta = scaler_Y_torch.inverse_transform(Y_full)
            V_BEM_phys_full = an_at_sven - an_at_delta
    else:
        D_full = get_D_tensor(df_train, entree, device)

        # Récupération du BEM pour reconstruire les forces absolues en stratégie 'f'
        if str(residuelle) in ['1']:
            _, Y_full_abs_scaled = format_data(df_train, entree, '0', inter, is_train=True, device=device)
            with open(f"training/scalers/scaler_Y_{entree}_0_f.pkl", 'rb') as f_abs: scaler_f_abs = pickle.load(f_abs)
            fn_ft_sven = TorchScaler(scaler_f_abs, device).inverse_transform(Y_full_abs_scaled)
            fn_ft_delta = scaler_Y_torch.inverse_transform(Y_full)
            F_BEM_phys_full = fn_ft_sven - fn_ft_delta

    all_ae_params = {}
    if has_ae:
        with open(AE_JSON_PATH, "r") as f: all_ae_params = json.load(f)

    # --- MÉTRIQUES PHYSIQUES ---
    def compute_phys_score(model, X_val, Y_val, val_idx, preds_val, current_ae=None):
        preds_norm = adapt_ae_output_to_target(current_ae.decode(preds_val), Y_val) if current_ae is not None else preds_val

        coeffs_pred = scaler_Y_torch.inverse_transform(preds_norm)
        coeffs_true = scaler_Y_torch.inverse_transform(Y_val)

        if inter == 'v':
            v_bem_val = V_BEM_phys_full[val_idx] if V_BEM_phys_full is not None else None
            if v_bem_val is not None: coeffs_pred = coeffs_pred + v_bem_val; coeffs_true = coeffs_true + v_bem_val
            
            an_p, at_p = (coeffs_pred[:,0], coeffs_pred[:,1]) if is_cnn else (coeffs_pred[:,0::2], coeffs_pred[:,1::2])
            an_t, at_t = (coeffs_true[:,0], coeffs_true[:,1]) if is_cnn else (coeffs_true[:,0::2], coeffs_true[:,1::2])
            
            alpha_p_deg = torch.atan2(an_p, at_p) * (180.0 / torch.pi)
            alpha_t_deg = torch.atan2(an_t, at_t) * (180.0 / torch.pi)
            v_app_slice = V_app_full[val_idx]
            v_eff_p = v_app_slice * torch.sqrt(an_p**2 + at_p**2)
            v_eff_t = v_app_slice * torch.sqrt(an_t**2 + at_t**2)
            
            f_pred_phys = convert_v_to_f_torch(v_eff_p, alpha_p_deg, r_tensor, c_tensor, polar_surrogate)
            f_true_phys = convert_v_to_f_torch(v_eff_t, alpha_t_deg, r_tensor, c_tensor, polar_surrogate)
            Fn_p_abs, Ft_p_abs = f_pred_phys[..., 0], f_pred_phys[..., 1]
            Fn_t_abs, Ft_t_abs = f_true_phys[..., 0], f_true_phys[..., 1]
            D_phys = 0.5 * RHO * (v_app_slice**2) * torch.abs(c_tensor)
            
            Fn_p_norm, Ft_p_norm = Fn_p_abs / D_phys, Ft_p_abs / D_phys
            Fn_t_norm, Ft_t_norm = Fn_t_abs / D_phys, Ft_t_abs / D_phys
        else:
            f_bem_val = F_BEM_phys_full[val_idx] if F_BEM_phys_full is not None else None
            if f_bem_val is not None:
                coeffs_pred = coeffs_pred + f_bem_val
                coeffs_true = coeffs_true + f_bem_val
                
            if is_cnn:
                Fn_p_norm, Ft_p_norm = coeffs_pred[:, 0], coeffs_pred[:, 1]
                Fn_t_norm, Ft_t_norm = coeffs_true[:, 0], coeffs_true[:, 1]
                D_val = D_full[val_idx]
            else:
                Fn_p_norm, Ft_p_norm = coeffs_pred[:, 0::2], coeffs_pred[:, 1::2]
                Fn_t_norm, Ft_t_norm = coeffs_true[:, 0::2], coeffs_true[:, 1::2]
                D_val = D_full[val_idx][:, 0::2] 

            Fn_p_abs = Fn_p_norm * D_val; Ft_p_abs = Ft_p_norm * D_val
            Fn_t_abs = Fn_t_norm * D_val; Ft_t_abs = Ft_t_norm * D_val
            
            D_phys = D_val
            Fn_p_norm, Ft_p_norm = Fn_p_abs / D_phys, Ft_p_abs / D_phys
            Fn_t_norm, Ft_t_norm = Fn_t_abs / D_phys, Ft_t_abs / D_phys

        # RETOURNE LA MÉTRIQUE A (erreur relative) ou B (erreur normalisée)
        if option == 'A':
            err_fn = (Fn_p_abs - Fn_t_abs) / torch.abs(Fn_t_abs)
            
            # === SÉCURITÉ POUR Ft ===
            err_ft = (Ft_p_abs - Ft_t_abs) / torch.clamp(torch.abs(Ft_t_abs), min=1.0)
            
            rmse_c = torch.sqrt(torch.mean(err_fn**2)) * 100
            rmse_d = torch.sqrt(torch.mean(err_ft**2)) * 100
            return (rmse_c + rmse_d).item()
        elif option == 'B':
            rmse_e = torch.sqrt(torch.mean((Fn_p_norm - Fn_t_norm)**2)) * 100
            rmse_f = torch.sqrt(torch.mean((Ft_p_norm - Ft_t_norm)**2)) * 100
            return (rmse_e + rmse_f).item()

    def objective_model(trial):
        lr_bounds = LR_BOUNDS_AE if has_ae else LR_BOUNDS_NOAE
        lr = trial.suggest_float('lr', lr_bounds[0], lr_bounds[1], log=True)
        dropout_rate = trial.suggest_float('dropout_rate', DROPOUT_BOUNDS[0], DROPOUT_BOUNDS[1], log=True)

        if has_ae:
            ae_nature = trial.suggest_categorical('ae_nature', AE_NATURES)
            ae_dim = trial.suggest_categorical('ae_dim', AE_DIMS)
            ae_key = format_ae_key(residuelle, inter, ae_nature, ae_dim, bem_suffix)
            ae_params = all_ae_params[ae_key]
            current_ae = ConvolutionalAutoencoder(in_channels=2, latent_dim=ae_dim, depth=ae_params['ae_depth'], base_filters=ae_params['ae_base_filters'], device=device).to(device) if ae_nature == 'M' else LinearAutoencoder(in_features=5184, latent_dim=ae_dim, n_layers=ae_params['ae_depth'], device=device).to(device)
            current_ae.load_state_dict(torch.load(os.path.join(AE_WEIGHTS_DIR, f"ae_{ae_key}.pth"), map_location=device))
            current_ae.eval()
            for p in current_ae.parameters():
                p.requires_grad_(False)
            latent_dim = ae_dim
        else:
            current_ae, latent_dim, ae_nature, ae_dim = None, 0, 'None', 0

        # Encodage BEM dans l'espace latent pour '2+' (même espace entrée/sortie)
        if has_plus and current_ae is not None:
            with torch.no_grad():
                if entree == 'GV':
                    y_bem_cnn = gv_to_gm_format(Y_bem_full)
                    y_bem_input = y_bem_cnn.reshape(Y_bem_full.size(0), -1) if ae_nature == 'V' else y_bem_cnn
                    z_bem = current_ae.encode(y_bem_input)
                    X_trial = torch.cat([X_full[:, :n_scalaires_full], z_bem], dim=1)
                else:  # GM
                    y_bem_input = Y_bem_full.reshape(Y_bem_full.size(0), -1) if ae_nature == 'V' else Y_bem_full
                    z_bem = current_ae.encode(y_bem_input)
                    zb = z_bem[:, :, None, None].expand(-1, -1, 36, 72).contiguous()
                    X_trial = torch.cat([X_full[:, :-2], zb], dim=1)
        else:
            X_trial = X_full

        # --- CONTRAINTE CONVEXE POUR LES LAMBDAS ---
        if inter == 'v':
            v = sorted([trial.suggest_float('v1', 0, 1), trial.suggest_float('v2', 0, 1)])
            l1, l2, l3 = v[0], v[1] - v[0], 1 - v[1]  # uniforme sur le 2-simplexe
        else: # inter == 'f' -> l3 = 0
            u1 = trial.suggest_float('u1', 0, 1)
            l1, l2, l3 = u1, 1.0 - u1, 0.0

        if entree == 'GV':
            model_class = TurbineMLP
            model_kwargs = {'input_dim': X_trial.shape[1], 'output_dim': latent_dim if has_ae else Y_full.shape[1], 'n_layers': trial.suggest_int('n_layers', MLP_LAYERS_BOUNDS[0], MLP_LAYERS_BOUNDS[1]), 'n_neurons': trial.suggest_categorical('n_neurons', MLP_NEURONS_CHOICES), 'dropout_rate': dropout_rate, 'device': device}
        else:
            model_class = TurbineCNN
            model_kwargs = {'in_channels': X_trial.shape[1], 'out_channels': latent_dim if has_ae else Y_full.shape[1], 'use_autoencoder': has_ae, 'latent_dim': latent_dim, 'n_layers': trial.suggest_int('n_layers', CNN_LAYERS_BOUNDS[0], CNN_LAYERS_BOUNDS[1]), 'base_filters': trial.suggest_categorical('base_filters', CNN_FILTERS_CHOICES), 'dropout_rate': dropout_rate, 'device': device}

        def criterion_builder(train_idx=None, val_idx=None):
            return TurbineLoss(
                inter=inter, loss_type=option, l1=l1, l2=l2, l3=l3, ae_model=current_ae, scaler_Y=scaler_Y,
                r_tensor=r_tensor, c_tensor=c_tensor, polar_surrogate=polar_surrogate if inter == 'v' else None,
                device=device
            )

        metric_fn = lambda m, x, y, idx, p: compute_phys_score(m, x, y, idx, p, current_ae)
        mean_val_loss, mean_custom_score, _ = cross_validate(
            X_full=X_trial, Y_full=Y_full, model_class=model_class, model_kwargs=model_kwargs,
            criterion_builder=criterion_builder, epochs=EPOCHS_OPTUNA, lr=lr, n_splits=CV_SPLITS, 
            device=device, inter=inter, v_bem_phys_full=V_BEM_phys_full, D_phys_full=D_full, f_bem_phys_full=F_BEM_phys_full, v_app_full=V_app_full, u_inf_full=u_inf_full,
            compute_metrics_fn=metric_fn, trial=trial
        )

        trial.set_user_attr("cv_score_phys", mean_custom_score)
        trial.set_user_attr("ae_nature", ae_nature)
        trial.set_user_attr("ae_dim", ae_dim)
        trial.set_user_attr("l1", l1); trial.set_user_attr("l2", l2); trial.set_user_attr("l3", l3)
        del criterion_builder, metric_fn
        if has_ae:
            current_ae.cpu()
            del current_ae
        gc.collect()
        torch.cuda.empty_cache()
        return mean_custom_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_model = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=PRUNER_WARMUP))
    study_model.optimize(objective_model, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study_model.best_params
    best_params.update({
        "ae_nature": study_model.best_trial.user_attrs["ae_nature"], "ae_dim": study_model.best_trial.user_attrs["ae_dim"],
        "l1": study_model.best_trial.user_attrs["l1"], "l2": study_model.best_trial.user_attrs["l2"], "l3": study_model.best_trial.user_attrs["l3"],
        "Total_Score_CV": study_model.best_value
    })
    
    if has_ae:
        best_ae_key = format_ae_key(residuelle, inter, best_params['ae_nature'], best_params['ae_dim'], bem_suffix)
        best_params.update(all_ae_params[best_ae_key])
        
    target_json = f"training/hyperparametres/{entree.lower()}_hyperparameters.json"
    all_model_params = json.load(open(target_json, "r")) if os.path.exists(target_json) else {}
    all_model_params[model_base_name] = best_params
    with open(target_json, "w") as f: json.dump(all_model_params, f, indent=4)
    print(f"   [OK] Modèle {model_base_name} optimisé. Métrique {option} CV : {study_model.best_value:.2f} %")