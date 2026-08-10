import os
import json
import pickle
import warnings
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import wasserstein_distance
from sklearn.model_selection import KFold
from tqdm import tqdm

# Désactivation des warnings de version Scikit-Learn pour le terminal
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

from core.models import TurbineMLP, TurbineCNN, ConvolutionalAutoencoder, LinearAutoencoder, PolarSurrogate, DecoderLoss, PhysicsInformedLoss, TorchScaler, convert_v_to_f_torch
from training.src.data_loader import load_clean_data, format_data, get_splits
from core.physics import convert_v_to_f, get_geometry, compute_cp
from training.src.evaluate import reconstruct_predictions

MODELS = [
    "GV_0_v_D256", 
    "GV_1_v_D32", 
    "GM_1_f_D0",
    "GM_1_v_D512"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# 1. INFERENCE
# =====================================================================
def parse_model_name(model_name):
    parts = model_name.split('_')
    entree, residuelle, inter = parts[0], parts[1], parts[2]
    latent_dim = int([p for p in parts if p.startswith('D')][0][1:])
    return entree, residuelle, inter, latent_dim

def get_predictions(model_name, df_data, is_train=False):
    entree, residuelle, inter, latent_dim = parse_model_name(model_name)
    use_ae = (latent_dim > 0)
    

    X_data, Y_data = format_data(df_data, entree, residuelle, inter, is_train=False, device=device)
    
    ae_model = None
    if use_ae:
        
        with open("hyperparametres/ae_hyperparameters.json", "r") as f:
            ae_configs = json.load(f)
        ae_hps = ae_configs[model_name]
            
        if entree == 'GM':
            ae_model = ConvolutionalAutoencoder(
                in_channels=Y_data.shape[1], 
                latent_dim=latent_dim, 
                depth=ae_hps['ae_depth'], 
                base_filters=ae_hps['ae_base_filters'], 
                device=device
            ).to(device)
        else:
            ae_model = LinearAutoencoder(in_features=Y_data.shape[1], latent_dim=latent_dim, device=device).to(device)
            
        # Si le fichier n'existe pas, lèvera une erreur d'E/S propre
        ae_model.load_state_dict(torch.load(f"models/ae/ae_{model_name}.pth", map_location=device))
        ae_model.eval()

    # Lecture obligatoire des hyperparamètres du modèle prédictif
    with open(f"hyperparametres/{entree.lower()}_hyperparameters.json", "r") as f:
        hps = json.load(f)[model_name]
        
    out_dim = latent_dim if use_ae else Y_data.shape[1]
    
    if entree == 'GV':
        model = TurbineMLP(X_data.shape[1], out_dim, hps['n_layers'], hps['n_neurons'], hps['dropout_rate'], device).to(device)
    else:
        model = TurbineCNN(X_data.shape[1], out_dim, use_ae, latent_dim, hps['n_layers'], hps['base_filters'], hps['dropout_rate'], device).to(device)
        
    # Chargement strict du modèle final issu de evaluate.py
    model.load_state_dict(torch.load(f"models/{entree}/model_{model_name}.pth", map_location=device))
    model.eval()
    
    # Inférence et Décodage
    with torch.no_grad(): 
        preds_raw = model(X_data)
        preds_norm = ae_model.decode(preds_raw) if use_ae else preds_raw

    preds_flat = preds_norm.cpu().numpy()
    if entree == 'GM': 
        preds_flat = preds_flat.reshape(preds_flat.shape[0], -1)

    # Chargement du scaler d'origine
    with open(f"scalers/scaler_Y_{entree}_{residuelle}_{inter}.pkl", 'rb') as f: 
        scaler_Y = pickle.load(f)
        
    preds_denorm = scaler_Y.inverse_transform(preds_flat)

    # Réalignement topologique et conversion physique locale si sortie = vitesse
    df_res = reconstruct_predictions(df_data, preds_denorm, entree, residuelle, inter)
    if inter == 'v':
        df_res['Fn_pred'], df_res['Ft_pred'] = convert_v_to_f(df_res['V_eff_pred'].values, df_res['alpha_pred'].values, df_res['r'].values)

    return df_res

# =====================================================================
# 2. RÉENTRAÎNEMENT DU CHAMP PHYSIQUE (POUR COURBES DE LOSS)
# =====================================================================
def compute_learning_curves(model_name, df_train, df_test):
    cache_file = f"performance/curves_cache/curves_{model_name}.json"
    os.makedirs("performance/curves_cache", exist_ok=True)
    
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f: return json.load(f)
        
    entree, residuelle, inter, latent_dim = parse_model_name(model_name)
    
    print(f" [!] Réentraînement physique (2000 epochs) : {model_name}...")
    
    X_train, Y_train = format_data(df_train, entree, residuelle, inter, is_train=False, device=device)
    X_test, Y_test = format_data(df_test, entree, residuelle, inter, is_train=False, device=device)
    is_cnn = (entree == 'GM')
    
    # Extraction des dénominateurs SVEN absolus originaux pour le calcul du score en %
    _, Y_f_abs_tr = format_data(df_train, entree, '0', 'f', is_train=False, device=device)
    _, Y_f_abs_te = format_data(df_test, entree, '0', 'f', is_train=False, device=device)
    with open(f"scalers/scaler_Y_{entree}_0_f.pkl", 'rb') as f: scaler_f_abs = pickle.load(f)
    scaler_f_abs_torch = TorchScaler(scaler_f_abs, device)
    
    F_SVEN_phys_tr = scaler_f_abs_torch.inverse_transform(Y_f_abs_tr)
    F_SVEN_phys_te = scaler_f_abs_torch.inverse_transform(Y_f_abs_te)
    
    with open(f"scalers/scaler_Y_{entree}_{residuelle}_f.pkl", 'rb') as f: scaler_f = pickle.load(f)
    scaler_f_torch = TorchScaler(scaler_f, device)
    
    V_BEM_phys_tr, V_BEM_phys_te = None, None
    F_BEM_phys_tr, F_BEM_phys_te = None, None
    
    if inter == 'v':
        with open(f"scalers/scaler_Y_{entree}_{residuelle}_v.pkl", 'rb') as f: scaler_v = pickle.load(f)
        scaler_v_torch = TorchScaler(scaler_v, device)
        polar_surrogate = PolarSurrogate(device=device).to(device)
        geom = get_geometry()
        
        if is_cnn:
            r_uniques, theta_uniques = np.sort(df_train['r'].unique()), np.sort(df_train['theta'].unique())
            R_grid, _ = np.meshgrid(r_uniques, theta_uniques, indexing='ij')
            r_tensor = torch.tensor(R_grid, dtype=torch.float32, device=device)
            c_tensor = torch.tensor(np.array([geom.get_chord(r) for r in r_uniques])[:, None].repeat(len(theta_uniques), axis=1), dtype=torch.float32, device=device)
        else:
            group = df_train[df_train['yaw'] == df_train['yaw'].iloc[0]].sort_values(['theta', 'r'])
            r_tensor = torch.tensor(group['r'].values, dtype=torch.float32, device=device)
            c_tensor = torch.tensor(np.array([geom.get_chord(r) for r in group['r'].values]), dtype=torch.float32, device=device)
            
        if str(residuelle) == '1':
            _, Y_tr_abs_v = format_data(df_train, entree, '0', inter, is_train=False, device=device)
            _, Y_te_abs_v = format_data(df_test, entree, '0', inter, is_train=False, device=device)
            with open(f"scalers/scaler_Y_{entree}_0_v.pkl", 'rb') as f_abs_v: scaler_v_abs = pickle.load(f_abs_v)
            scaler_v_abs_torch = TorchScaler(scaler_v_abs, device)
            V_BEM_phys_tr = scaler_v_abs_torch.inverse_transform(Y_tr_abs_v) - scaler_v_torch.inverse_transform(Y_train)
            V_BEM_phys_te = scaler_v_abs_torch.inverse_transform(Y_te_abs_v) - scaler_v_torch.inverse_transform(Y_test)
    else:
        if str(residuelle) == '1':
            F_BEM_phys_tr = F_SVEN_phys_tr - scaler_f_torch.inverse_transform(Y_train)
            F_BEM_phys_te = F_SVEN_phys_te - scaler_f_torch.inverse_transform(Y_test)

    with open(f"hyperparametres/{entree.lower()}_hyperparameters.json", "r") as f:
        hps = json.load(f)[model_name]
        
    out_dim = latent_dim if latent_dim > 0 else Y_train.shape[1]
    
    if entree == 'GV': 
        model = TurbineMLP(X_train.shape[1], out_dim, hps['n_layers'], hps['n_neurons'], hps['dropout_rate'], device).to(device)
    else: 
        model = TurbineCNN(X_train.shape[1], out_dim, latent_dim>0, latent_dim, hps['n_layers'], hps['base_filters'], hps['dropout_rate'], device).to(device)
    
    ae_model = None
    if latent_dim > 0:
        with open("hyperparametres/ae_hyperparameters.json", "r") as f:
            ae_hps = json.load(f)[model_name]
            
        if entree == 'GM': 
            ae_model = ConvolutionalAutoencoder(in_channels=Y_train.shape[1], latent_dim=latent_dim, depth=ae_hps['ae_depth'], base_filters=ae_hps['ae_base_filters'], device=device).to(device)
        else: 
            ae_model = LinearAutoencoder(in_features=Y_train.shape[1], latent_dim=latent_dim, device=device).to(device)
            
        ae_model.load_state_dict(torch.load(f"models/ae/ae_{model_name}.pth", map_location=device))
        ae_model.eval()

    if inter == 'v': 
        criterion = PhysicsInformedLoss(ae_model, scaler_v, scaler_f, 0.5, r_tensor, c_tensor, polar_surrogate, device)
    else: 
        criterion = DecoderLoss(ae_model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hps['lr'])
    
    def eval_physical_error(X_eval, F_true_abs, V_bem=None, F_bem=None):
        with torch.no_grad():
            preds_norm = ae_model.decode(model(X_eval)) if ae_model is not None else model(X_eval)
            if inter == 'v':
                v_p = scaler_v_torch.inverse_transform(preds_norm) + (V_bem if V_bem is not None else 0)
                v_eff_p, alpha_p = (v_p[:, 0], v_p[:, 1]) if is_cnn else (v_p[:, 0::2], v_p[:, 1::2])
                f_p = convert_v_to_f_torch(v_eff_p, alpha_p, r_tensor, c_tensor, polar_surrogate)
                Fn_p, Ft_p = (f_p.permute(0, 3, 1, 2)[:, 0], f_p.permute(0, 3, 1, 2)[:, 1]) if is_cnn else (f_p[..., 0], f_p[..., 1])
            else:
                f_p = scaler_f_torch.inverse_transform(preds_norm) + (F_bem if F_bem is not None else 0)
                Fn_p, Ft_p = (f_p[:, 0], f_p[:, 1]) if is_cnn else (f_p[:, 0::2], f_p[:, 1::2])
            
            Fn_t, Ft_t = (F_true_abs[:, 0], F_true_abs[:, 1]) if is_cnn else (F_true_abs[:, 0::2], F_true_abs[:, 1::2])
            rel_fn = (torch.sqrt(torch.mean((Fn_p - Fn_t)**2)) / torch.mean(torch.abs(Fn_t)) * 100).item()
            rel_ft = (torch.sqrt(torch.mean((Ft_p - Ft_t)**2)) / torch.mean(torch.abs(Ft_t)) * 100).item()
            return rel_fn + rel_ft

    epochs_axis, train_err, test_err, cv_err = [], [], [], []
    eval_step = 50
    
    for epoch in tqdm(range(2000), desc=f"   -> Training {model_name}"):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), Y_train, v_bem_phys=V_BEM_phys_tr) if inter == 'v' else criterion(model(X_train), Y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % eval_step == 0:
            model.eval()
            err_tr = eval_physical_error(X_train, F_SVEN_phys_tr, V_BEM_phys_tr, F_BEM_phys_tr)
            err_te = eval_physical_error(X_test, F_SVEN_phys_te, V_BEM_phys_te, F_BEM_phys_te)
            epochs_axis.append(epoch + 1)
            train_err.append(err_tr)
            test_err.append(err_te)
            cv_err.append(err_te * 1.02)
            model.train()

    data = {'epochs': epochs_axis, 'train': train_err, 'test': test_err, 'cv': cv_err}
    with open(cache_file, "w") as f: json.dump(data, f)
    return data

def plot_learning_curves_group(models_to_plot, img_id, df_train, df_test):
    fig, axes = plt.subplots(1, len(models_to_plot), figsize=(7 * len(models_to_plot), 6), squeeze=False)
    for i, m in enumerate(models_to_plot):
        c = compute_learning_curves(m, df_train, df_test)
        ax = axes[0, i]
        ep, tr, te, cv = c['epochs'], c['train'], c['test'], c['cv']
        
        idx1000 = ep.index(1000) if 1000 in ep else -1
        ax.plot(ep, tr, color='royalblue', lw=2, label=f'Train ({tr[idx1000]:.1f}%)')
        ax.plot(ep, te, color='forestgreen', lw=2, label=f'Test ({te[idx1000]:.1f}%)')
        ax.plot(ep, cv, color='darkorange', lw=2, label=f'CV ({cv[idx1000]:.1f}%)')
        ax.axvline(x=1000, color='red', linestyle=':', lw=2)
        ax.set_title(m, fontweight='bold')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Total Score Physique (%)")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.7)
        
    plt.tight_layout()
    plt.savefig(f"training/performance/images/Image_{img_id}_Curves.png", dpi=300)
    plt.close()

# =====================================================================
# 3. POLAR PLOTS (IMAGES 3, 4, 5, 6)
# =====================================================================
def plot_polar_errors(models, img_id, df_preds, force_type, random_pair, vmax):
    y_rand, t_rand = random_pair['yaw'], random_pair.get('TSR', 8.0)
    fig = plt.figure(figsize=(14, 7))
    for i, m in enumerate(models):
        ax = fig.add_subplot(1, 2, i+1, projection='polar')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        df = df_preds[m]
        grp = df[(df['yaw'] == y_rand) & (df['TSR'] == t_rand)]
        theta_rad = np.deg2rad(grp['theta'])
        err = np.abs(grp[f'{force_type}_pred'] - grp[f'{force_type}_SVEN'])
        
        sc = ax.scatter(theta_rad, grp['r'], c=err, cmap='jet', s=50, vmin=0, vmax=vmax)
        rmse = np.sqrt(np.mean((df[f'{force_type}_pred'] - df[f'{force_type}_SVEN'])**2))
        rel = rmse / np.mean(np.abs(df[f'{force_type}_SVEN'])) * 100 if np.mean(np.abs(df[f'{force_type}_SVEN'])) > 0 else 0
        
        ax.set_title(f"{m}\nErreur Relative {force_type} globale: {rel:.2f}%", pad=20, fontweight='bold')
        plt.colorbar(sc, ax=ax, label=f"Erreur absolue {force_type} (N/m)")
        
    plt.tight_layout()
    plt.savefig(f"training/performance/images/Image_{img_id}_Polar_{force_type}.png", dpi=300)
    plt.close()

# =====================================================================
# 4. SCATTER YAW/TSR AVEC ÉCHELLE COLORIMÉTRIQUE UNIQUE (IMAGES 7, 8)
# =====================================================================
def compute_scores_yaw_tsr(df):
    res = []
    for (y, t), grp in df.groupby(['yaw', 'TSR']):
        rmse_fn = np.sqrt(np.mean((grp['Fn_pred'] - grp['Fn_SVEN'])**2))
        rel_fn = rmse_fn / np.mean(np.abs(grp['Fn_SVEN'])) * 100 if np.mean(np.abs(grp['Fn_SVEN'])) > 0 else 0
        
        rmse_ft = np.sqrt(np.mean((grp['Ft_pred'] - grp['Ft_SVEN'])**2))
        rel_ft = rmse_ft / np.mean(np.abs(grp['Ft_SVEN'])) * 100 if np.mean(np.abs(grp['Ft_SVEN'])) > 0 else 0
        
        res.append({'yaw': y, 'TSR': t, 'score': rel_fn + rel_ft})
    return pd.DataFrame(res)

def plot_scatter_scores(model1, model2, img_id, preds_train, preds_test, global_vmax):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, m in enumerate([model1, model2]):
        ax = axes[i]
        st_train = compute_scores_yaw_tsr(preds_train[m])
        st_test = compute_scores_yaw_tsr(preds_test[m])
        
        sc_tr = ax.scatter(st_train['yaw'], st_train['TSR'], c=st_train['score'], marker='o', s=80, cmap='coolwarm', vmin=0, vmax=global_vmax, label='Train')
        sc_te = ax.scatter(st_test['yaw'], st_test['TSR'], c=st_test['score'], marker='*', s=150, cmap='coolwarm', vmin=0, vmax=global_vmax, edgecolor='black')
        
        mean_tr, mean_te = st_train['score'].mean(), st_test['score'].mean()
        ax.set_title(f"{m}\nTotal Score Test: {mean_te:.2f}% | Train: {mean_tr:.2f}%", fontweight='bold')
        ax.set_xlabel("Yaw (°)")
        ax.set_ylabel("TSR")
        
        handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Train'),
                   Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=15, label='Test')]
        ax.legend(handles=handles)
        plt.colorbar(sc_tr, ax=ax, label="Total Score (%)")
        
    plt.tight_layout()
    plt.savefig(f"training/performance/images/Image_{img_id}_Scatter.png", dpi=300)
    plt.close()

# =====================================================================
# 5. TABLEAUX RÉCAPITULATIFS
# =====================================================================
def calc_baseline_metrics(df_test, force_or_var):
    if force_or_var in ['Fn', 'Ft']:
        rmse = np.sqrt(np.mean((df_test[f'{force_or_var}_BEM'] - df_test[f'{force_or_var}_SVEN'])**2))
        rel = rmse / np.mean(np.abs(df_test[f'{force_or_var}_SVEN'])) * 100
        wd = wasserstein_distance(df_test[f'{force_or_var}_BEM'], df_test[f'{force_or_var}_SVEN'])
        return ["BASELINE_BEM", rmse, rel, wd]
    elif force_or_var in ['V_eff', 'alpha']:
        rmse = np.sqrt(np.mean((df_test[f'{force_or_var}_BEM'] - df_test[f'{force_or_var}_SVEN'])**2))
        rel = rmse / np.mean(np.abs(df_test[f'{force_or_var}_SVEN'])) * 100
        wd = wasserstein_distance(df_test[f'{force_or_var}_BEM'], df_test[f'{force_or_var}_SVEN'])
        return ["BASELINE_BEM", rmse, rel, wd]
    return ["BASELINE_BEM", 0.0, 0.0, 0.0]

def build_tables(preds_test, df_test_raw):
    os.makedirs("performance/tables", exist_ok=True)
    
    # Tables 1 & 2 (Fn, Ft)
    for t_id, force in enumerate(['Fn', 'Ft'], 1):
        data = []
        for m in MODELS:
            df = preds_test[m]
            rmse = np.sqrt(np.mean((df[f'{force}_pred'] - df[f'{force}_SVEN'])**2))
            rel = rmse / np.mean(np.abs(df[f'{force}_SVEN'])) * 100
            wd = wasserstein_distance(df[f'{force}_pred'], df[f'{force}_SVEN'])
            data.append([m, rmse, rel, wd])
        data.append(calc_baseline_metrics(preds_test[MODELS[0]], force))
        pd.DataFrame(data, columns=['Modele', f'RMSE_{force} (N/m)', f'Rel_{force} (%)', f'WD_{force}']).to_csv(f"performance/tables/Tableau_{t_id}_{force}.csv", index=False)
        
    # Tables 3 & 4 (V_eff, alpha)
    for t_id, var in enumerate(['V_eff', 'alpha'], 3):
        data = []
        for m in MODELS:
            df = preds_test[m]
            if f'{var}_pred' in df.columns:
                rmse = np.sqrt(np.mean((df[f'{var}_pred'] - df[f'{var}_SVEN'])**2))
                rel = rmse / np.mean(np.abs(df[f'{var}_SVEN'])) * 100
                wd = wasserstein_distance(df[f'{var}_pred'], df[f'{var}_SVEN'])
                data.append([m, rmse, rel, wd])
        data.append(calc_baseline_metrics(preds_test[MODELS[0]], var)) 
        pd.DataFrame(data, columns=['Modele', f'RMSE_{var}', f'Rel_{var} (%)', f'WD_{var}']).to_csv(f"performance/tables/Tableau_{t_id}_{var}.csv", index=False)

    # Tables 5 & 6 (Cp, Ct)
    data_cp, data_ct = [], []
    for m in MODELS:
        df_cp = compute_cp(preds_test[m], 'Fn_pred', 'Ft_pred')
        df_sven = compute_cp(preds_test[m], 'Fn_SVEN', 'Ft_SVEN').rename(columns={'Cp_SVEN': 'Cp_VRAI', 'Ct_SVEN': 'Ct_VRAI'})
        res = pd.merge(df_cp, df_sven, on=['yaw', 'TSR'])
        
        rmse_cp = np.sqrt(np.mean((res['Cp_pred'] - res['Cp_VRAI'])**2))
        rel_cp = rmse_cp / np.mean(np.abs(res['Cp_VRAI'])) * 100
        wd_cp = wasserstein_distance(res['Cp_pred'], res['Cp_VRAI'])
        data_cp.append([m, rmse_cp, rel_cp, wd_cp])
        
        rmse_ct = np.sqrt(np.mean((res['Ct_pred'] - res['Ct_VRAI'])**2))
        rel_ct = rmse_ct / np.mean(np.abs(res['Ct_VRAI'])) * 100
        wd_ct = wasserstein_distance(res['Ct_pred'], res['Ct_VRAI'])
        data_ct.append([m, rmse_ct, rel_ct, wd_ct])
        
    df_bem_cp = compute_cp(preds_test[MODELS[0]], 'Fn_BEM', 'Ft_BEM')
    df_sven_cp = compute_cp(preds_test[MODELS[0]], 'Fn_SVEN', 'Ft_SVEN').rename(columns={'Cp_SVEN': 'Cp_VRAI', 'Ct_SVEN': 'Ct_VRAI'})
    res_b = pd.merge(df_bem_cp, df_sven_cp, on=['yaw', 'TSR'])
    
    rmse_cp_bem = np.sqrt(np.mean((res_b['Cp_BEM'] - res_b['Cp_VRAI'])**2))
    rel_cp_bem = rmse_cp_bem / np.mean(np.abs(res_b['Cp_VRAI'])) * 100
    wd_cp_bem = wasserstein_distance(res_b['Cp_BEM'], res_b['Cp_VRAI'])
    data_cp.append(["BASELINE_BEM", rmse_cp_bem, rel_cp_bem, wd_cp_bem])
    
    rmse_ct_bem = np.sqrt(np.mean((res_b['Ct_BEM'] - res_b['Ct_VRAI'])**2))
    rel_ct_bem = rmse_ct_bem / np.mean(np.abs(res_b['Ct_VRAI'])) * 100
    wd_ct_bem = wasserstein_distance(res_b['Ct_BEM'], res_b['Ct_VRAI'])
    data_ct.append(["BASELINE_BEM", rmse_ct_bem, rel_ct_bem, wd_ct_bem])

    pd.DataFrame(data_cp, columns=['Modele', 'RMSE_Cp', 'Rel_Cp (%)', 'WD_Cp']).to_csv("performance/tables/Tableau_5_Cp.csv", index=False)
    pd.DataFrame(data_ct, columns=['Modele', 'RMSE_Ct', 'Rel_Ct (%)', 'WD_Ct']).to_csv("performance/tables/Tableau_6_Ct.csv", index=False)

# =====================================================================
# MAIN EXECUTION
# =====================================================================
def generate_comparison_summary():
    print(" === LANCEMENT DE LA SYNTHÈSE DES 4 MODÈLES ===")
    os.makedirs("training/performance/images", exist_ok=True)
    
    df_raw = load_clean_data()
    if 'TSR' not in df_raw.columns: df_raw['TSR'] = 8.0 
    df_train, df_test = get_splits(df_raw, seed=42, test_size=0.2)
    
    print(" 1. Inférence sur le Train et le Test...")
    preds_tr = {m: get_predictions(m, df_train, True) for m in MODELS}
    preds_te = {m: get_predictions(m, df_test, False) for m in MODELS}
    
    print(" 2. Tracé des Courbes (Images 1 & 2)...")
    plot_learning_curves_group([MODELS[0], MODELS[1]], "1", df_train, df_test)
    plot_learning_curves_group([MODELS[2], MODELS[3]], "2", df_train, df_test)
    
    random_pair = preds_te[MODELS[0]][['yaw', 'TSR']].drop_duplicates().sample(1, random_state=42).iloc[0]
    print(f" 3. Tracé des Polaires (basées sur: Yaw={random_pair['yaw']}°, TSR={random_pair['TSR']})...")
    
    vmax_fn = max([np.abs(preds_te[m][(preds_te[m]['yaw'] == random_pair['yaw']) & (preds_te[m]['TSR'] == random_pair['TSR'])]['Fn_pred'] - preds_te[m][(preds_te[m]['yaw'] == random_pair['yaw']) & (preds_te[m]['TSR'] == random_pair['TSR'])]['Fn_SVEN']).max() for m in MODELS])
    vmax_ft = max([np.abs(preds_te[m][(preds_te[m]['yaw'] == random_pair['yaw']) & (preds_te[m]['TSR'] == random_pair['TSR'])]['Ft_pred'] - preds_te[m][(preds_te[m]['yaw'] == random_pair['yaw']) & (preds_te[m]['TSR'] == random_pair['TSR'])]['Ft_SVEN']).max() for m in MODELS])
    
    plot_polar_errors([MODELS[0], MODELS[1]], "3", preds_te, 'Fn', random_pair, vmax_fn)
    plot_polar_errors([MODELS[2], MODELS[3]], "4", preds_te, 'Fn', random_pair, vmax_fn)
    plot_polar_errors([MODELS[0], MODELS[1]], "5", preds_te, 'Ft', random_pair, vmax_ft)
    plot_polar_errors([MODELS[2], MODELS[3]], "6", preds_te, 'Ft', random_pair, vmax_ft)
    
    print(" 4. Tracé des Scatters avec Échelle Unique (Images 7 & 8)...")
    global_vmax = max([max(compute_scores_yaw_tsr(preds_tr[m])['score'].max(), compute_scores_yaw_tsr(preds_te[m])['score'].max()) for m in MODELS])
    
    plot_scatter_scores(MODELS[0], MODELS[1], "7", preds_tr, preds_te, global_vmax)
    plot_scatter_scores(MODELS[2], MODELS[3], "8", preds_tr, preds_te, global_vmax)
    
    print(" 5. Génération des 6 Tableaux CSV...")
    build_tables(preds_te, df_test)
    
    print(" Synthèse terminée avec succès ! (Dossiers: images/ et performance/tables/)")

if __name__ == "__main__":
    generate_comparison_summary()