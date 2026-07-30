import os
import json
import pickle
import copy
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from core.models import (TurbineMLP, TurbineCNN, ConvolutionalAutoencoder, LinearAutoencoder,
                         PolarSurrogate, TurbineLoss, TorchScaler, convert_v_to_f_torch, compute_cp_ct_torch,
                         adapt_ae_output_to_target, gv_to_gm_format)
from training.src.data_loader import format_data, get_D_tensor, get_V_app_tensor, format_bem_as_Y
from training.src.trainer import fit_model, cross_validate
from core.physics import convert_v_to_f, get_geometry, compute_dynamic_pressure_D, compute_cp, compute_V_app
from core.config import (EPOCHS_FINAL, CV_SPLITS, RATIO_THRESHOLD, RHO, OMEGA, R_ROTOR,
                         AE_NATURES, AE_DIMS, AE_JSON_PATH, AE_WEIGHTS_DIR,
                         format_scaler_name, format_model_name, format_ae_key)

XLSX_PATH = "training/performance/recap_scores.xlsx"

def save_to_xlsx(filepath, sheet_name, dict_data):
    if os.path.exists(filepath):
        with pd.ExcelFile(filepath, engine='openpyxl') as xf:
            sheets = {s: xf.parse(s) for s in xf.sheet_names}
    else:
        sheets = {}
    if sheet_name in sheets:
        df = sheets[sheet_name]
        df = df[df["Modele"] != dict_data["Modele"]]
        df = pd.concat([df, pd.DataFrame([dict_data])], ignore_index=True)
    else:
        df = pd.DataFrame([dict_data])
    sheets[sheet_name] = df
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        for s, d in sheets.items():
            d.to_excel(writer, sheet_name=s, index=False)

def get_u_inf_tensor(df, device='cpu'):
    u_inf_list = []
    for _, group in df.groupby(['yaw', 'TSR'] if 'TSR' in df.columns else 'yaw'):
        tsr_val = group['TSR'].iloc[0] if 'TSR' in df.columns else 8.0
        u_inf = (OMEGA * R_ROTOR) / tsr_val
        u_inf_list.append(u_inf)
    return torch.tensor(u_inf_list, dtype=torch.float32, device=device)

def reconstruct_predictions(df, preds_flat, entree, residuelle, inter, bem_suffix=None):
    # Les prédictions sont produites dans l'ordre (yaw, [TSR]) puis (theta, r) pour GV
    # ou (r, theta) pour GM.
    group_keys = ['yaw', 'TSR'] if 'TSR' in df.columns else ['yaw']
    sort_keys = ['theta', 'r'] if entree == 'GV' else ['r', 'theta']
    df_res = df.sort_values(group_keys + sort_keys, kind='mergesort').reset_index(drop=True)
    if inter == 'v':
        v_app = df_res['v_app'].values if 'v_app' in df_res.columns else compute_V_app(df_res)
        if str(residuelle) in ['1']:
            an_res, at_res = preds_flat[:, 0::2].flatten(), preds_flat[:, 1::2].flatten()
            an_bem_col, at_bem_col = f'an_BEM_{bem_suffix}', f'at_BEM_{bem_suffix}'
            if an_bem_col in df_res.columns:
                an_bem = df_res[an_bem_col].values
                at_bem = df_res[at_bem_col].values
            else:
                alpha_bem_rad = np.radians(df_res[f'alpha_BEM_{bem_suffix}'].values)
                v_eff_bem = df_res[f'V_eff_BEM_{bem_suffix}'].values
                an_bem = np.sin(alpha_bem_rad) * v_eff_bem / v_app
                at_bem = np.cos(alpha_bem_rad) * v_eff_bem / v_app
            an_abs = an_res + an_bem
            at_abs = at_res + at_bem
        else:
            an_abs, at_abs = preds_flat[:, 0::2].flatten(), preds_flat[:, 1::2].flatten()

        alpha_rad = np.arctan2(an_abs, at_abs)
        v_eff = v_app * np.sqrt(an_abs**2 + at_abs**2)
        fn_pred, ft_pred = convert_v_to_f(v_eff, np.degrees(alpha_rad), df_res['r'].values)
    else:
        if str(residuelle) in ['1']:
            fn_res, ft_res = preds_flat[:, 0::2].flatten(), preds_flat[:, 1::2].flatten()
            fn_pred = fn_res + df_res[f'Fn_BEM_{bem_suffix}'].values
            ft_pred = ft_res + df_res[f'Ft_BEM_{bem_suffix}'].values
        else:
            fn_pred, ft_pred = preds_flat[:, 0::2].flatten(), preds_flat[:, 1::2].flatten()

    df_res['Fn_pred'] = fn_pred
    df_res['Ft_pred'] = ft_pred
    return df_res

def denorm_and_reconstruct(df_test, preds_coeffs, entree, residuelle, inter, is_cnn, bem_suffix=None):
    if inter == 'v':
        if is_cnn:
            preds_coeffs_2d = np.zeros((preds_coeffs.shape[0], 5184))
            preds_coeffs_2d[:, 0::2] = preds_coeffs[:, 0, :, :].reshape(preds_coeffs.shape[0], -1)
            preds_coeffs_2d[:, 1::2] = preds_coeffs[:, 1, :, :].reshape(preds_coeffs.shape[0], -1)
            preds_coeffs = preds_coeffs_2d
        return reconstruct_predictions(df_test, preds_coeffs, entree, residuelle, inter, bem_suffix)

    D_test_np = get_D_tensor(df_test, entree, 'cpu').numpy()
    if is_cnn:
        D_exp = np.stack([D_test_np, D_test_np], axis=1)
        preds_denorm_4d = preds_coeffs * D_exp
        preds_flat_f = np.zeros((preds_denorm_4d.shape[0], 5184), dtype=np.float32)
        preds_flat_f[:, 0::2] = preds_denorm_4d[:, 0].reshape(preds_denorm_4d.shape[0], -1)
        preds_flat_f[:, 1::2] = preds_denorm_4d[:, 1].reshape(preds_denorm_4d.shape[0], -1)
        preds_denorm = preds_flat_f
    else:
        D_test_flat = D_test_np.reshape(D_test_np.shape[0], -1)
        preds_denorm = preds_coeffs * D_test_flat
    return reconstruct_predictions(df_test, preds_denorm, entree, residuelle, inter, bem_suffix)

def score_ABC(df_res):
    Fn_p, Ft_p = df_res['Fn_pred'].values, df_res['Ft_pred'].values
    Fn_s, Ft_s = df_res['Fn_SVEN'].values, df_res['Ft_SVEN'].values
    D_res = df_res['D_phys'].values
    m_c = np.sqrt(np.mean(((Fn_p - Fn_s) / np.abs(Fn_s))**2)) * 100
    m_d = np.sqrt(np.mean(((Ft_p - Ft_s) / np.maximum(np.abs(Ft_s), 1.0))**2)) * 100
    m_e = np.sqrt(np.mean(((Fn_p/D_res) - (Fn_s/D_res))**2)) * 100
    m_f = np.sqrt(np.mean(((Ft_p/D_res) - (Ft_s/D_res))**2)) * 100

    cp_ct_pred = compute_cp(df_res, 'Fn_pred', 'Ft_pred').set_index('yaw')
    Cp_p = df_res['yaw'].map(cp_ct_pred['Cp_pred']).values
    Ct_p = df_res['yaw'].map(cp_ct_pred['Ct_pred']).values
    cp_ct_sven = compute_cp(df_res, 'Fn_SVEN', 'Ft_SVEN').set_index('yaw')
    Cp_s = df_res['yaw'].map(cp_ct_sven['Cp_SVEN']).values
    Ct_s = df_res['yaw'].map(cp_ct_sven['Ct_SVEN']).values
    m_i = np.sqrt(np.mean(((Cp_p - Cp_s) / np.abs(Cp_s))**2)) * 100
    m_j = np.sqrt(np.mean(((Ct_p - Ct_s) / np.abs(Ct_s))**2)) * 100

    return m_c + m_d, m_e + m_f, m_i + m_j

def evaluator(df_train, df_test, entree, residuelle, inter, has_ae, option, baseline_scores, pct=100, bem_suffix=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    has_plus = '+' in str(residuelle)
    ae_label = "DXY" if has_ae else "D0"
    model_base_name = format_model_name(entree, residuelle, inter, ae_label, option, pct, bem_suffix)

    target_json = f"training/hyperparametres/{entree.lower()}_hyperparameters.json"
    if not os.path.exists(target_json):
        print(f"   [ERREUR] Aucun hyperparamètre trouvé pour {model_base_name}.")
        return

    all_hps = json.load(open(target_json, "r"))
    if model_base_name not in all_hps: return

    best_params = all_hps[model_base_name]
    ae_nature = best_params.get("ae_nature", "None")
    ae_dim = best_params.get("ae_dim", 0)
    l1, l2, l3 = best_params.get("l1", 0.0), best_params.get("l2", 1.0), best_params.get("l3", 0.0)

    final_ae_str = f"D{ae_nature}{ae_dim}" if has_ae else "D0"
    final_model_name = format_model_name(entree, residuelle, inter, final_ae_str, option, pct, bem_suffix)
    
    print(f"\n{'='*50}")
    print(f" ÉVALUATION : {final_model_name}")
    print(f"{'='*50}")

    pbar = tqdm(total=5, desc="  Progression", bar_format="  {desc} |{bar}| {n}/{total} [{elapsed}]", leave=True)

    # ── Étape 1 : Préparation des données ──────────────────────────────────
    t0 = time.perf_counter()
    X_train, Y_train = format_data(df_train, entree, residuelle, inter, is_train=True, device=device, bem_suffix=bem_suffix)
    X_test, Y_test = format_data(df_test, entree, residuelle, inter, is_train=False, device=device, bem_suffix=bem_suffix)

    with open(f"training/scalers/scaler_Y_{format_scaler_name(entree, residuelle, inter, bem_suffix)}.pkl", 'rb') as f:
        scaler_Y = pickle.load(f)
    scaler_Y_torch = TorchScaler(scaler_Y, device)

    is_cnn = (entree == 'GM')
    V_BEM_phys_train = None
    F_BEM_phys_train = None
    V_app_full_train = None
    D_train_full = None
    u_inf_full = get_u_inf_tensor(df_train, device)

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

    if inter == 'v':
        polar_surrogate = PolarSurrogate(device=device).to(device)
        V_app_full_train = get_V_app_tensor(df_train, entree, device)
        if str(residuelle) in ['1']:
            _, Y_full_abs_scaled = format_data(df_train, entree, '0', inter, is_train=True, device=device)
            with open(f"training/scalers/scaler_Y_{entree}_0_v.pkl", 'rb') as f_abs: scaler_v_abs = pickle.load(f_abs)
            an_at_sven = TorchScaler(scaler_v_abs, device).inverse_transform(Y_full_abs_scaled)
            an_at_delta = scaler_Y_torch.inverse_transform(Y_train)
            V_BEM_phys_train = an_at_sven - an_at_delta
    else:
        D_train_full = get_D_tensor(df_train, entree, device)
        if str(residuelle) in ['1']:
            _, Y_full_abs_scaled = format_data(df_train, entree, '0', inter, is_train=True, device=device)
            with open(f"training/scalers/scaler_Y_{entree}_0_f.pkl", 'rb') as f_abs: scaler_f_abs = pickle.load(f_abs)
            fn_ft_sven = TorchScaler(scaler_f_abs, device).inverse_transform(Y_full_abs_scaled)
            fn_ft_delta = scaler_Y_torch.inverse_transform(Y_train)
            F_BEM_phys_train = fn_ft_sven - fn_ft_delta

    if has_ae:
        ae_key = format_ae_key(residuelle, inter, ae_nature, ae_dim, bem_suffix)
        ae_configs = json.load(open(AE_JSON_PATH, "r"))
        ae_config = ae_configs[ae_key]
        current_ae = ConvolutionalAutoencoder(in_channels=2, latent_dim=ae_dim, depth=ae_config['ae_depth'], base_filters=ae_config['ae_base_filters'], device=device).to(device) if ae_nature == 'M' else LinearAutoencoder(in_features=5184, latent_dim=ae_dim, n_layers=ae_config['ae_depth'], device=device).to(device)
        current_ae.load_state_dict(torch.load(os.path.join(AE_WEIGHTS_DIR, f"ae_{ae_key}.pth"), map_location=device))
        current_ae.eval()
        for p in current_ae.parameters():
            p.requires_grad_(False)
    else:
        current_ae = None

    if has_plus and current_ae is not None:
        n_scalaires = 2 if 'TSR' in df_train.columns else 1
        with torch.no_grad():
            Y_bem_train = format_bem_as_Y(df_train, entree, inter, scaler_Y, device)
            Y_bem_test  = format_bem_as_Y(df_test,  entree, inter, scaler_Y, device)
            if entree == 'GV':
                tr_cnn = gv_to_gm_format(Y_bem_train)
                te_cnn = gv_to_gm_format(Y_bem_test)
                y_bem_tr = tr_cnn.reshape(Y_bem_train.size(0), -1) if ae_nature == 'V' else tr_cnn
                y_bem_te = te_cnn.reshape(Y_bem_test.size(0),  -1) if ae_nature == 'V' else te_cnn
                z_bem_train = current_ae.encode(y_bem_tr)
                z_bem_test  = current_ae.encode(y_bem_te)
                X_train = torch.cat([X_train[:, :n_scalaires], z_bem_train], dim=1)
                X_test  = torch.cat([X_test[:, :n_scalaires],  z_bem_test],  dim=1)
            else:  # GM : z_BEM broadcasté en canaux constants (N, ae_dim, 36, 72)
                y_bem_tr = Y_bem_train.reshape(Y_bem_train.size(0), -1) if ae_nature == 'V' else Y_bem_train
                y_bem_te = Y_bem_test.reshape(Y_bem_test.size(0), -1) if ae_nature == 'V' else Y_bem_test
                z_bem_train = current_ae.encode(y_bem_tr)
                z_bem_test  = current_ae.encode(y_bem_te)
                zb_tr = z_bem_train[:, :, None, None].expand(-1, -1, 36, 72).contiguous()
                zb_te = z_bem_test[:, :, None, None].expand(-1, -1, 36, 72).contiguous()
                X_train = torch.cat([X_train[:, :-2], zb_tr], dim=1)
                X_test  = torch.cat([X_test[:, :-2],  zb_te],  dim=1)

    pbar.set_postfix_str(f"Données & AE : {time.perf_counter()-t0:.1f}s")
    pbar.update(1)

    group_keys_cv = ['yaw', 'TSR'] if 'TSR' in df_train.columns else ['yaw']
    groups_list_cv = [g for _, g in df_train.groupby(group_keys_cv, sort=True)]

    def compute_phys_score(model, X_val, Y_val, val_idx, preds_val):
        is_cnn_auto = (Y_val.dim() == 4)
        preds_norm = adapt_ae_output_to_target(current_ae.decode(preds_val), Y_val) if has_ae else preds_val

        coeffs_pred = scaler_Y_torch.inverse_transform(preds_norm)
        coeffs_true = scaler_Y_torch.inverse_transform(Y_val)

        if inter == 'v':
            v_bem_val = V_BEM_phys_train[val_idx] if V_BEM_phys_train is not None else None
            coeffs_pred_abs = coeffs_pred + v_bem_val if v_bem_val is not None else coeffs_pred
            coeffs_true_abs = coeffs_true + v_bem_val if v_bem_val is not None else coeffs_true
            an_p, at_p = (coeffs_pred_abs[:,0], coeffs_pred_abs[:,1]) if is_cnn_auto else (coeffs_pred_abs[:,0::2], coeffs_pred_abs[:,1::2])
            an_t, at_t = (coeffs_true_abs[:,0], coeffs_true_abs[:,1]) if is_cnn_auto else (coeffs_true_abs[:,0::2], coeffs_true_abs[:,1::2])

            alpha_p_deg = torch.atan2(an_p, at_p) * (180.0 / torch.pi)
            alpha_t_deg = torch.atan2(an_t, at_t) * (180.0 / torch.pi)
            v_app_slice = V_app_full_train[val_idx]
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
            f_bem_val = F_BEM_phys_train[val_idx] if F_BEM_phys_train is not None else None
            coeffs_pred_abs = coeffs_pred + f_bem_val if f_bem_val is not None else coeffs_pred
            coeffs_true_abs = coeffs_true + f_bem_val if f_bem_val is not None else coeffs_true

            D_val = D_train_full[val_idx] if is_cnn_auto else D_train_full[val_idx][:, 0::2]
            Fn_p_norm, Ft_p_norm = (coeffs_pred_abs[:,0], coeffs_pred_abs[:,1]) if is_cnn_auto else (coeffs_pred_abs[:,0::2], coeffs_pred_abs[:,1::2])
            Fn_t_norm, Ft_t_norm = (coeffs_true_abs[:,0], coeffs_true_abs[:,1]) if is_cnn_auto else (coeffs_true_abs[:,0::2], coeffs_true_abs[:,1::2])

            Fn_p_abs = Fn_p_norm * D_val; Ft_p_abs = Ft_p_norm * D_val
            Fn_t_abs = Fn_t_norm * D_val; Ft_t_abs = Ft_t_norm * D_val

        err_fn = (Fn_p_abs - Fn_t_abs) / torch.abs(Fn_t_abs)
        err_ft = (Ft_p_abs - Ft_t_abs) / torch.clamp(torch.abs(Ft_t_abs), min=1.0)
        score_a = ((torch.sqrt(torch.mean(err_fn**2)) + torch.sqrt(torch.mean(err_ft**2))) * 100).item()
        score_b = ((torch.sqrt(torch.mean((Fn_p_norm - Fn_t_norm)**2)) + torch.sqrt(torch.mean((Ft_p_norm - Ft_t_norm)**2))) * 100).item()

        # Score C (Cp/Ct) : recalculé via la pipeline physique dataframe (identique à l'évaluation finale),
        # car son intégration azimutale/rayon ne se prête pas au calcul tensoriel direct ci-dessus.
        val_idx_sorted = np.sort(val_idx)
        order = torch.as_tensor(np.argsort(val_idx), dtype=torch.long, device=coeffs_pred.device)
        df_val = pd.concat([groups_list_cv[i] for i in val_idx_sorted], ignore_index=True)
        coeffs_pred_delta_np = coeffs_pred[order].detach().cpu().numpy()
        df_val['D_phys'] = compute_dynamic_pressure_D(df_val)
        df_res_val = denorm_and_reconstruct(df_val, coeffs_pred_delta_np, entree, residuelle, inter, is_cnn, bem_suffix)
        _, _, score_c = score_ABC(df_res_val)

        return {'A': score_a, 'B': score_b, 'C': score_c}

    # ── Étape 2 : Cross-validation ─────────────────────────────────────────
    t0 = time.perf_counter()
    tqdm.write(f"   [1/2] Cross-Validation ({CV_SPLITS} Folds)...")
    model_class = TurbineMLP if entree == 'GV' else TurbineCNN
    target_dim = current_ae.encoder_fc.out_features if has_ae and hasattr(current_ae, 'encoder_fc') else ae_dim if has_ae else Y_train.shape[1]
    model_kwargs = {'input_dim': X_train.shape[1], 'output_dim': target_dim, 'n_layers': best_params['n_layers'], 'n_neurons': best_params['n_neurons'], 'dropout_rate': best_params['dropout_rate'], 'device': device} if entree == 'GV' else {'in_channels': X_train.shape[1], 'out_channels': target_dim, 'use_autoencoder': has_ae, 'latent_dim': ae_dim, 'n_layers': best_params['n_layers'], 'base_filters': best_params['base_filters'], 'dropout_rate': best_params['dropout_rate'], 'device': device}

    def criterion_builder(train_idx=None, val_idx=None):
        return TurbineLoss(inter=inter, loss_type=option, l1=l1, l2=l2, l3=l3, ae_model=current_ae, scaler_Y=scaler_Y, r_tensor=r_tensor, c_tensor=c_tensor, polar_surrogate=polar_surrogate if inter == 'v' else None, device=device)

    mean_cv_loss, mean_cv_score, std_cv_score = cross_validate(X_full=X_train, Y_full=Y_train, model_class=model_class, model_kwargs=model_kwargs, criterion_builder=criterion_builder, epochs=EPOCHS_FINAL, lr=best_params['lr'], n_splits=CV_SPLITS, device=device, inter=inter, v_bem_phys_full=V_BEM_phys_train, D_phys_full=D_train_full, f_bem_phys_full=F_BEM_phys_train, v_app_full=V_app_full_train, u_inf_full=u_inf_full, compute_metrics_fn=compute_phys_score)
    pbar.set_postfix_str(f"CV : {time.perf_counter()-t0:.1f}s")
    pbar.update(1)

    # ── Étape 3 : Entraînement final ───────────────────────────────────────
    t0 = time.perf_counter()
    tqdm.write(f"   [2/2] Entraînement Final...")
    model_final = model_class(**model_kwargs).to(device)
    model_final, _ = fit_model(model=model_final, X=X_train, Y=Y_train, criterion=criterion_builder(None,None), epochs=EPOCHS_FINAL, lr=best_params['lr'], device=device, inter=inter, v_bem_phys=V_BEM_phys_train, D_phys=D_train_full, f_bem_phys=F_BEM_phys_train, v_app=V_app_full_train, u_inf=u_inf_full, show_progress=False)
    pbar.set_postfix_str(f"Train : {time.perf_counter()-t0:.1f}s")
    pbar.update(1)

    model_final.eval()
    if current_ae: current_ae.eval()

    # ── Étape 4 : Inférence sur le jeu de test ─────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        preds_raw = model_final(X_test)
        preds_norm = adapt_ae_output_to_target(current_ae.decode(preds_raw), Y_test) if has_ae else preds_raw
        preds_coeffs = scaler_Y_torch.inverse_transform(preds_norm).cpu().numpy()

    df_test['D_phys'] = compute_dynamic_pressure_D(df_test)
    df_res = denorm_and_reconstruct(df_test, preds_coeffs, entree, residuelle, inter, is_cnn, bem_suffix)

    # Borne inf du score pour les modèles à AE : decode(encode(Y_true)) vs Y_true.
    AE_Score_A = AE_Score_B = AE_Score_C = None
    if has_ae:
        with torch.no_grad():
            y_true_gm = gv_to_gm_format(Y_test) if entree == 'GV' else Y_test
            y_true_ae_in = y_true_gm.reshape(Y_test.size(0), -1) if ae_nature == 'V' else y_true_gm
            decoded_true = current_ae.decode(current_ae.encode(y_true_ae_in))
            preds_norm_ae = adapt_ae_output_to_target(decoded_true, Y_test)
            preds_coeffs_ae = scaler_Y_torch.inverse_transform(preds_norm_ae).cpu().numpy()
        df_res_ae = denorm_and_reconstruct(df_test, preds_coeffs_ae, entree, residuelle, inter, is_cnn, bem_suffix)
        AE_Score_A, AE_Score_B, AE_Score_C = score_ABC(df_res_ae)

    pbar.set_postfix_str(f"Inférence : {time.perf_counter()-t0:.1f}s")
    pbar.update(1)

    # ── Étape 5 : Métriques & sauvegarde ──────────────────────────────────
    t0 = time.perf_counter()
    cp_ct_pred = compute_cp(df_res, 'Fn_pred', 'Ft_pred').set_index('yaw')
    df_res['Cp_pred'] = df_res['yaw'].map(cp_ct_pred['Cp_pred']).values
    df_res['Ct_pred'] = df_res['yaw'].map(cp_ct_pred['Ct_pred']).values

    Fn_p, Ft_p = df_res['Fn_pred'].values, df_res['Ft_pred'].values
    Fn_s, Ft_s = df_res['Fn_SVEN'].values, df_res['Ft_SVEN'].values
    D_res = df_res['D_phys'].values
    m_a, m_b = np.sqrt(np.mean((Fn_p - Fn_s)**2)), np.sqrt(np.mean((Ft_p - Ft_s)**2))
    m_c, m_d = np.sqrt(np.mean(((Fn_p - Fn_s) / np.abs(Fn_s))**2)) * 100, np.sqrt(np.mean(((Ft_p - Ft_s) / np.maximum(np.abs(Ft_s), 1.0))**2)) * 100
    m_e, m_f = np.sqrt(np.mean(((Fn_p/D_res) - (Fn_s/D_res))**2)) * 100, np.sqrt(np.mean(((Ft_p/D_res) - (Ft_s/D_res))**2)) * 100
    Cp_p, Ct_p = df_res['Cp_pred'].values, df_res['Ct_pred'].values
    cp_ct_sven = compute_cp(df_res, 'Fn_SVEN', 'Ft_SVEN').set_index('yaw')
    Cp_s = df_res['yaw'].map(cp_ct_sven['Cp_SVEN']).values
    Ct_s = df_res['yaw'].map(cp_ct_sven['Ct_SVEN']).values
    m_g, m_h = np.sqrt(np.mean((Cp_p - Cp_s)**2)), np.sqrt(np.mean((Ct_p - Ct_s)**2))
    m_i, m_j = np.sqrt(np.mean(((Cp_p - Cp_s) / np.abs(Cp_s))**2)) * 100, np.sqrt(np.mean(((Ct_p - Ct_s) / np.abs(Ct_s))**2)) * 100

    Score_A, Score_B, Score_C = m_c + m_d, m_e + m_f, m_i + m_j
    wd_fn = wasserstein_distance(Fn_s, Fn_p)
    wd_ft = wasserstein_distance(Ft_s, Ft_p)
    wd_cp = wasserstein_distance(Cp_s, Cp_p)
    wd_ct = wasserstein_distance(Ct_s, Ct_p)
    wd_total = wasserstein_distance(np.concatenate([Fn_s, Ft_s]), np.concatenate([Fn_p, Ft_p]))

    BEM_A, BEM_B, BEM_C = baseline_scores['A'], baseline_scores['B'], baseline_scores['C']
    ratios = {'A': BEM_A / Score_A if Score_A > 0 else 0, 'B': BEM_B / Score_B if Score_B > 0 else 0, 'C': BEM_C / Score_C if Score_C > 0 else 0}
    best_metric_AB = max('A', 'B', key=lambda k: ratios[k])
    best_ratio_AB = ratios[best_metric_AB]
    ratio_C = ratios['C']

    if best_ratio_AB > RATIO_THRESHOLD:
        os.makedirs(f"training/models/{entree}", exist_ok=True)
        torch.save(model_final.state_dict(), f"training/models/{entree}/{final_model_name}.pth")
        print(f"   [SAUVEGARDE] Ratio AB {best_ratio_AB:.2f}x > {RATIO_THRESHOLD}.")

    save_to_xlsx(XLSX_PATH, "recap_dico", {"Modele": final_model_name, "Entree": entree, "Residuelle": residuelle, "Inter": inter, "Has_AE": has_ae, "AE_Nature": ae_nature, "AE_Dim": ae_dim, "Option_Loss": option, "L1": round(l1, 2), "L2": round(l2, 2), "L3": round(l3, 2)})
    super_dict = {"Modele": final_model_name, "Best_Metric": best_metric_AB, "BEM_Ratio_C": round(ratio_C, 2), "Best_BEM_Ratio_AB": round(best_ratio_AB, 2)}
    if has_ae:
        super_dict.update({"AE_Score_A": round(AE_Score_A, 2), "AE_Score_B": round(AE_Score_B, 2), "AE_Score_C": round(AE_Score_C, 2)})
    save_to_xlsx(XLSX_PATH, "recap_super", super_dict)
    save_to_xlsx(XLSX_PATH, "recap_cross_val", {
        "Modele": final_model_name,
        "CV_Score_A": round(mean_cv_score['A'], 2), "CV_Std_A": round(std_cv_score['A'], 2),
        "CV_Score_B": round(mean_cv_score['B'], 2), "CV_Std_B": round(std_cv_score['B'], 2),
        "CV_Score_C": round(mean_cv_score['C'], 2), "CV_Std_C": round(std_cv_score['C'], 2),
    })
    save_to_xlsx(XLSX_PATH, "recap_Fn", {"Modele": final_model_name, "err_abs_Nm": round(m_a, 4), "err_rel_%": round(m_c, 2), "err_normD_%": round(m_e, 2), "WD_Fn": round(wd_fn, 4)})
    save_to_xlsx(XLSX_PATH, "recap_Ft", {"Modele": final_model_name, "err_abs_Nm": round(m_b, 4), "err_rel_%": round(m_d, 2), "err_normD_%": round(m_f, 2), "WD_Ft": round(wd_ft, 4)})
    save_to_xlsx(XLSX_PATH, "recap_C_P", {"Modele": final_model_name, "err_abs": round(m_g, 6), "err_%": round(m_i, 2), "WD_Cp": round(wd_cp, 6)})
    save_to_xlsx(XLSX_PATH, "recap_C_T", {"Modele": final_model_name, "err_abs": round(m_h, 6), "err_%": round(m_j, 2), "WD_Ct": round(wd_ct, 6)})
    save_to_xlsx(XLSX_PATH, "recap_globaux", {"Modele": final_model_name, "Score_A": round(Score_A, 2), "Score_B": round(Score_B, 2), "Score_C": round(Score_C, 2), "WD_Total": round(wd_total, 4)})
    pbar.set_postfix_str(f"Métriques : {time.perf_counter()-t0:.1f}s")
    pbar.update(1)
    pbar.close()

def evaluate_baselines(df_test, bem_suffix):
    fn_bem_col, ft_bem_col = f'Fn_BEM_{bem_suffix}', f'Ft_BEM_{bem_suffix}'
    df_bem = df_test.copy()
    D_res = compute_dynamic_pressure_D(df_bem)
    df_bem['D_phys'] = D_res

    Fn_s, Ft_s, Fn_b, Ft_b = df_test['Fn_SVEN'].values, df_test['Ft_SVEN'].values, df_test[fn_bem_col].values, df_test[ft_bem_col].values
    m_a, m_b = np.sqrt(np.mean((Fn_b - Fn_s)**2)), np.sqrt(np.mean((Ft_b - Ft_s)**2))
    m_c, m_d = np.sqrt(np.mean(((Fn_b - Fn_s) / np.abs(Fn_s))**2)) * 100, np.sqrt(np.mean(((Ft_b - Ft_s) / np.maximum(np.abs(Ft_s), 1.0))**2)) * 100
    m_e, m_f = np.sqrt(np.mean(((Fn_b/D_res) - (Fn_s/D_res))**2)) * 100, np.sqrt(np.mean(((Ft_b/D_res) - (Ft_s/D_res))**2)) * 100

    c_bem = compute_cp(df_bem, fn_bem_col, ft_bem_col).set_index('yaw')  # colonnes Cp_/Ct_ nommées d'après col_fn (cf. core/physics.py::compute_cp)
    bem_model_name = fn_bem_col.replace('Fn_', '')
    Cp_b, Ct_b = df_bem['yaw'].map(c_bem[f'Cp_{bem_model_name}']).values, df_bem['yaw'].map(c_bem[f'Ct_{bem_model_name}']).values
    c_sven = compute_cp(df_bem, 'Fn_SVEN', 'Ft_SVEN').set_index('yaw')
    Cp_s, Ct_s = df_bem['yaw'].map(c_sven['Cp_SVEN']).values, df_bem['yaw'].map(c_sven['Ct_SVEN']).values

    m_g, m_h = np.sqrt(np.mean((Cp_b - Cp_s)**2)), np.sqrt(np.mean((Ct_b - Ct_s)**2))
    m_i, m_j = np.sqrt(np.mean(((Cp_b - Cp_s) / np.abs(Cp_s))**2)) * 100, np.sqrt(np.mean(((Ct_b - Ct_s) / np.abs(Ct_s))**2)) * 100

    wd_fn = wasserstein_distance(Fn_s, Fn_b)
    wd_ft = wasserstein_distance(Ft_s, Ft_b)
    wd_cp = wasserstein_distance(Cp_s, Cp_b)
    wd_ct = wasserstein_distance(Ct_s, Ct_b)
    wd_total = wasserstein_distance(np.concatenate([Fn_s, Ft_s]), np.concatenate([Fn_b, Ft_b]))

    base_dict = {"Modele": f"BASELINE_BEM_{bem_suffix}"}
    save_to_xlsx(XLSX_PATH, "recap_super", {**base_dict, "BEM_Ratio_C": 1.00, "Best_BEM_Ratio_AB": 1.00})
    save_to_xlsx(XLSX_PATH, "recap_Fn", {**base_dict, "err_abs_Nm": round(m_a, 4), "err_rel_%": round(m_c, 2), "err_normD_%": round(m_e, 2), "WD_Fn": round(wd_fn, 4)})
    save_to_xlsx(XLSX_PATH, "recap_Ft", {**base_dict, "err_abs_Nm": round(m_b, 4), "err_rel_%": round(m_d, 2), "err_normD_%": round(m_f, 2), "WD_Ft": round(wd_ft, 4)})
    save_to_xlsx(XLSX_PATH, "recap_C_P", {**base_dict, "err_abs": round(m_g, 6), "err_%": round(m_i, 2), "WD_Cp": round(wd_cp, 6)})
    save_to_xlsx(XLSX_PATH, "recap_C_T", {**base_dict, "err_abs": round(m_h, 6), "err_%": round(m_j, 2), "WD_Ct": round(wd_ct, 6)})
    save_to_xlsx(XLSX_PATH, "recap_globaux", {**base_dict, "Score_A": round(m_c+m_d, 2), "Score_B": round(m_e+m_f, 2), "Score_C": round(m_i+m_j, 2), "WD_Total": round(wd_total, 4)})

    return {'A': m_c + m_d, 'B': m_e + m_f, 'C': m_i + m_j}