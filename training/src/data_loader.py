import os

import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..')) 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import pickle
from sklearn.preprocessing import StandardScaler
from core.config import RHO, U_INFTY, PITCH_RAD, R_ROTOR, OMEGA, RANDOM_SEED, needs_bem_suffix, format_scaler_name

# Importations physiques globales
from core.physics import compute_V_app, get_geometry

def load_clean_data(path_forces="data/raw/fichier_forces.csv", path_vitesses="data/raw/fichier_vitesses.csv"):
    df_f = pd.read_csv(path_forces)
    df_v = pd.read_csv(path_vitesses)
    
    merge_keys = ['yaw', 'r', 'theta']
    if 'TSR' in df_f.columns and 'TSR' in df_v.columns:
        merge_keys.append('TSR')
        
    return df_f.merge(df_v, on=merge_keys)

def get_splits(df, seed=42, test_size=0.2, save_dir=None):
    yaws_uniques = df['yaw'].unique()
    train_yaw, test_yaw = train_test_split(yaws_uniques, test_size=test_size, random_state=seed)
    
    train_df = df[df['yaw'].isin(train_yaw)].copy()
    test_df = df[df['yaw'].isin(test_yaw)].copy()
    
    if save_dir is not None: 
        os.makedirs(save_dir, exist_ok=True)
        train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)
        print(f" [OK] Fichiers créés : train.csv et test.csv dans {save_dir}")
        
    return train_df, test_df

def subsample_train(df_train, pct, seed=RANDOM_SEED):
    """
    Sous-échantillonne le train à pct% des couples (yaw, TSR) uniques.
    (En LHS, un yaw correspond à au plus un unique TSR, donc échantillonner
    sur les yaw uniques revient à échantillonner sur les couples (yaw, TSR).)
    """
    if pct >= 100:
        return df_train.copy()

    yaws_uniques = df_train['yaw'].unique()
    n_keep = max(1, int(round(len(yaws_uniques) * pct / 100)))
    rng = np.random.RandomState(seed)
    kept_yaw = rng.choice(yaws_uniques, size=n_keep, replace=False)

    return df_train[df_train['yaw'].isin(kept_yaw)].copy()

class IsotropicScaler:
    """ Scaler global qui divise par l'écart-type de la Norme. """
    def __init__(self):
        self.scale_ = 1.0
    def fit(self, X):
        self.scale_ = np.std(X)
        return self
    def transform(self, X):
        return X / self.scale_
    def inverse_transform(self, X):
        return X * self.scale_

def format_data(df, entree, residuelle, inter, is_train=True, device='cpu', bem_suffix=None):
    """
    Formatte les entrées X et les cibles Y.
    - residuelle '0' : Y = SVEN, X = scalaires
    - residuelle '1' : Y = SVEN - BEM, X = scalaires + matrice BEM
    - residuelle '2' (ou '2+') : Y = SVEN, X = scalaires + matrice BEM

    `bem_suffix` (DUM/IFP/P&P) sélectionne la variante BEM à utiliser ; requis dès que la BEM est
    utilisée en entrée (residuelle '1'/'2'/'2+').
    """
    res_str = str(residuelle).replace('+', '') # On extrait le chiffre pur (0, 1 ou 2)
    has_plus = '+' in str(residuelle)
    if needs_bem_suffix(residuelle) and bem_suffix is None:
        raise ValueError(f"bem_suffix requis pour residuelle={residuelle!r}")

    X_list = []
    Y_list = []
    geom = get_geometry()

    for _, group in df.groupby(['yaw', 'TSR'] if 'TSR' in df.columns else 'yaw'):
        
        # =====================================================================
        # STRATÉGIE GLOBALE VECTORIELLE (GV)
        # =====================================================================
        if entree == 'GV':
            group = group.sort_values(['theta', 'r'])
            
            # --- Création de l'Entrée X ---
            x_val = [group['yaw'].iloc[0]]
            if 'TSR' in group.columns: 
                x_val.append(group['TSR'].iloc[0])
                
            # Si mode '1' ou '2' ou '1+' ou '2+', on intègre l'information BEM dans X
            if res_str in ['1', '2'] or has_plus:
                if inter == 'f':
                    v_app_b = compute_V_app(group)
                    chord_b = np.array([geom.get_chord(r) for r in group['r'].values])
                    D_b = 0.5 * RHO * v_app_b**2 * np.abs(chord_b)
                    x_val.extend(group[f'Fn_BEM_{bem_suffix}'].values / D_b)
                    x_val.extend(group[f'Ft_BEM_{bem_suffix}'].values / D_b)
                else: # 'v'
                    v_app_b = compute_V_app(group)
                    alpha_bem_rad = np.radians(group[f'alpha_BEM_{bem_suffix}'])
                    an_bem = np.sin(alpha_bem_rad) * group[f'V_eff_BEM_{bem_suffix}'] / v_app_b
                    at_bem = np.cos(alpha_bem_rad) * group[f'V_eff_BEM_{bem_suffix}'] / v_app_b
                    x_val.extend(an_bem)
                    x_val.extend(at_bem)
            X_list.append(x_val)

            # --- Création de la Cible Y ---
            if inter == 'f':
                c1, c2 = ('Fn_SVEN', 'Ft_SVEN') if res_str in ['0', '2'] else ('Fn_delta', 'Ft_delta')
                if res_str == '1' and 'Fn_delta' not in group.columns:
                    group['Fn_delta'] = group['Fn_SVEN'] - group[f'Fn_BEM_{bem_suffix}']
                    group['Ft_delta'] = group['Ft_SVEN'] - group[f'Ft_BEM_{bem_suffix}']
            else: # 'v'
                c1, c2 = ('an_SVEN', 'at_SVEN') if res_str in ['0', '2'] else ('an_delta', 'at_delta')
                if 'an_SVEN' not in group.columns:
                    v_app = compute_V_app(group)
                    alpha_sven_rad = np.radians(group['alpha_SVEN'])
                    group['an_SVEN'] = np.sin(alpha_sven_rad) * group['V_eff_SVEN'] / v_app
                    group['at_SVEN'] = np.cos(alpha_sven_rad) * group['V_eff_SVEN'] / v_app
                    if res_str == '1':
                        alpha_bem_rad = np.radians(group[f'alpha_BEM_{bem_suffix}'])
                        group['an_BEM'] = np.sin(alpha_bem_rad) * group[f'V_eff_BEM_{bem_suffix}'] / v_app
                        group['at_BEM'] = np.cos(alpha_bem_rad) * group[f'V_eff_BEM_{bem_suffix}'] / v_app
                        group['an_delta'] = group['an_SVEN'] - group['an_BEM']
                        group['at_delta'] = group['at_SVEN'] - group['at_BEM']
            
            y_val = []
            for _, row in group.iterrows():
                # Si inter=='f', on adimensionne TOUJOURS par D, quel que soit le mode
                if inter == 'f':
                    v_app_sq = row['v_app']**2 if 'v_app' in row else compute_V_app(pd.DataFrame([row]))[0]**2
                    chord_val = geom.get_chord(row['r'])
                    D = 0.5 * RHO * v_app_sq * np.abs(chord_val)
                    y_val.extend([row[c1] / D, row[c2] / D])
                else:
                    y_val.extend([row[c1], row[c2]])
            Y_list.append(y_val)

        # =====================================================================
        # STRATÉGIE GLOBALE MATRICIELLE (GM)
        # =====================================================================
        elif entree == 'GM':
            group = group.sort_values(['r', 'theta'])
            num_r = len(group['r'].unique())
            num_theta = len(group['theta'].unique())
            
            r_grid = group['r'].values.reshape(num_r, num_theta)
            theta_grid = group['theta'].values.reshape(num_r, num_theta)
            v_app_grid = compute_V_app(group).reshape(num_r, num_theta)
            yaw_grid = np.full_like(r_grid, group['yaw'].iloc[0])
            
            # --- Création de l'Entrée X ---
            x_channels = [r_grid, theta_grid, v_app_grid, yaw_grid]
            if 'TSR' in group.columns:
                tsr_grid = np.full_like(r_grid, group['TSR'].iloc[0])
                x_channels.append(tsr_grid)
                
            # Ajout des canaux BEM si mode 1 ou 2 ou 2+
            if res_str in ['1', '2'] or has_plus:
                if inter == 'f':
                    chord_grid_x = geom.get_chord(group['r'].values).reshape(num_r, num_theta)
                    D_grid_x = 0.5 * RHO * v_app_grid**2 * np.abs(chord_grid_x)
                    x_channels.append(group[f'Fn_BEM_{bem_suffix}'].values.reshape(num_r, num_theta) / D_grid_x)
                    x_channels.append(group[f'Ft_BEM_{bem_suffix}'].values.reshape(num_r, num_theta) / D_grid_x)
                else:
                    v_app_b = compute_V_app(group)
                    alpha_bem_rad = np.radians(group[f'alpha_BEM_{bem_suffix}'])
                    an_bem = np.sin(alpha_bem_rad) * group[f'V_eff_BEM_{bem_suffix}'] / v_app_b
                    at_bem = np.cos(alpha_bem_rad) * group[f'V_eff_BEM_{bem_suffix}'] / v_app_b
                    x_channels.append(an_bem.values.reshape(num_r, num_theta))
                    x_channels.append(at_bem.values.reshape(num_r, num_theta))

            X_list.append(np.stack(x_channels, axis=0))

            # --- Création de la Cible Y ---
            if inter == 'f':
                c1, c2 = ('Fn_SVEN', 'Ft_SVEN') if res_str in ['0', '2'] else ('Fn_delta', 'Ft_delta')
                if res_str == '1' and 'Fn_delta' not in group.columns:
                    group['Fn_delta'] = group['Fn_SVEN'] - group[f'Fn_BEM_{bem_suffix}']
                    group['Ft_delta'] = group['Ft_SVEN'] - group[f'Ft_BEM_{bem_suffix}']

                v_app_sq = v_app_grid**2
                chord_grid = geom.get_chord(group['r'].values).reshape(num_r, num_theta)
                D_grid = 0.5 * RHO * v_app_sq * np.abs(chord_grid)

                y1 = group[c1].values.reshape(num_r, num_theta) / D_grid
                y2 = group[c2].values.reshape(num_r, num_theta) / D_grid

            else: # 'v'
                c1, c2 = ('an_SVEN', 'at_SVEN') if res_str in ['0', '2'] else ('an_delta', 'at_delta')
                if 'an_SVEN' not in group.columns:
                    v_app = compute_V_app(group)
                    alpha_sven_rad = np.radians(group['alpha_SVEN'])
                    group['an_SVEN'] = np.sin(alpha_sven_rad) * group['V_eff_SVEN'] / v_app
                    group['at_SVEN'] = np.cos(alpha_sven_rad) * group['V_eff_SVEN'] / v_app
                    if res_str == '1':
                        alpha_bem_rad = np.radians(group[f'alpha_BEM_{bem_suffix}'])
                        group['an_BEM'] = np.sin(alpha_bem_rad) * group[f'V_eff_BEM_{bem_suffix}'] / v_app
                        group['at_BEM'] = np.cos(alpha_bem_rad) * group[f'V_eff_BEM_{bem_suffix}'] / v_app
                        group['an_delta'] = group['an_SVEN'] - group['an_BEM']
                        group['at_delta'] = group['at_SVEN'] - group['at_BEM']

                y1 = group[c1].values.reshape(num_r, num_theta)
                y2 = group[c2].values.reshape(num_r, num_theta)
                
            Y_list.append(np.stack([y1, y2], axis=0))

    X_np = np.array(X_list, dtype=np.float32)
    Y_np = np.array(Y_list, dtype=np.float32)

    # =========================================================================
    # APPLICATION DES SCALERS
    # =========================================================================
    model_name = format_scaler_name(entree, residuelle, inter, bem_suffix)
    os.makedirs("training/scalers", exist_ok=True)
    path_x = f"training/scalers/scaler_X_{model_name}.pkl"
    path_y = f"training/scalers/scaler_Y_{model_name}.pkl"    

    original_shape_X = X_np.shape
    original_shape_Y = Y_np.shape
    
    if entree == 'GM':
        X_np = X_np.reshape(X_np.shape[0], -1)
        Y_np = Y_np.reshape(Y_np.shape[0], -1)

    if is_train:
        # Scaler classique pour X
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_np)
        with open(path_x, 'wb') as f: pickle.dump(scaler_X, f)
        
        # Scaler classique pour la cible (Y) : standardise F_n/D, F_t/D ou a_n, a_t.
        scaler_Y = StandardScaler()
        Y_scaled = scaler_Y.fit_transform(Y_np)
        with open(path_y, 'wb') as f: pickle.dump(scaler_Y, f)

    else:
        with open(path_x, 'rb') as f: scaler_X = pickle.load(f)
        with open(path_y, 'rb') as f: scaler_Y = pickle.load(f)
        X_scaled = scaler_X.transform(X_np)
        Y_scaled = scaler_Y.transform(Y_np)

    if entree == 'GM':
        X_scaled = X_scaled.reshape(original_shape_X)
        Y_scaled = Y_scaled.reshape(original_shape_Y)

    return torch.tensor(X_scaled, device=device), torch.tensor(Y_scaled, device=device)

def format_bem_as_Y(df, entree, inter, scaler_Y, bem_suffix, device='cpu'):
    """
    Retourne les forces BEM (variante `bem_suffix` : DUM/IFP/P&P) dans le même espace de
    normalisation que Y, pour permettre leur encodage via l'AE SVEN (stratégie '2+').
    - GV : (N, 5184) — même format interleaved que Y
    - GM : (N, 2, 36, 72) — même format 2-canaux que Y
    """
    geom = get_geometry()
    Y_bem_list = []
    fn_bem_col, ft_bem_col = f'Fn_BEM_{bem_suffix}', f'Ft_BEM_{bem_suffix}'
    v_eff_bem_col, alpha_bem_col = f'V_eff_BEM_{bem_suffix}', f'alpha_BEM_{bem_suffix}'

    for _, group in df.groupby(['yaw', 'TSR'] if 'TSR' in df.columns else 'yaw'):
        if entree == 'GV':
            group = group.sort_values(['theta', 'r'])
            y_val = []
            if inter == 'f':
                for _, row in group.iterrows():
                    v_app_sq = row['v_app']**2 if 'v_app' in row else compute_V_app(pd.DataFrame([row]))[0]**2
                    D = 0.5 * RHO * v_app_sq * np.abs(geom.get_chord(row['r']))
                    y_val.extend([row[fn_bem_col] / D, row[ft_bem_col] / D])
            else:  # inter == 'v'
                v_app_b = compute_V_app(group)
                alpha_bem_rad = np.radians(group[alpha_bem_col].values)
                an_bem = np.sin(alpha_bem_rad) * group[v_eff_bem_col].values / v_app_b
                at_bem = np.cos(alpha_bem_rad) * group[v_eff_bem_col].values / v_app_b
                for an, at in zip(an_bem, at_bem):
                    y_val.extend([an, at])
            Y_bem_list.append(y_val)

        elif entree == 'GM':
            group = group.sort_values(['r', 'theta'])
            num_r = len(group['r'].unique())
            num_theta = len(group['theta'].unique())
            if inter == 'f':
                v_app_grid = compute_V_app(group).reshape(num_r, num_theta)
                chord_grid = geom.get_chord(group['r'].values).reshape(num_r, num_theta)
                D_grid = 0.5 * RHO * v_app_grid**2 * np.abs(chord_grid)
                y1 = group[fn_bem_col].values.reshape(num_r, num_theta) / D_grid
                y2 = group[ft_bem_col].values.reshape(num_r, num_theta) / D_grid
            else:  # inter == 'v'
                v_app_grid = compute_V_app(group).reshape(num_r, num_theta)
                alpha_bem_rad = np.radians(group[alpha_bem_col].values.reshape(num_r, num_theta))
                y1 = np.sin(alpha_bem_rad) * group[v_eff_bem_col].values.reshape(num_r, num_theta) / v_app_grid
                y2 = np.cos(alpha_bem_rad) * group[v_eff_bem_col].values.reshape(num_r, num_theta) / v_app_grid
            Y_bem_list.append(np.stack([y1, y2], axis=0))

    Y_bem_np = np.array(Y_bem_list, dtype=np.float32)
    original_shape = Y_bem_np.shape
    Y_bem_flat = Y_bem_np.reshape(len(Y_bem_list), -1)
    Y_bem_scaled = scaler_Y.transform(Y_bem_flat)
    if entree == 'GM':
        Y_bem_scaled = Y_bem_scaled.reshape(original_shape)

    return torch.tensor(Y_bem_scaled, dtype=torch.float32, device=device)

def get_D_tensor(df, entree, device='cpu'):
    geom = get_geometry()
    D_list = []
    for _, group in df.groupby(['yaw', 'TSR'] if 'TSR' in df.columns else 'yaw'):
        if entree == 'GV':
            group = group.sort_values(['theta', 'r'])
            D_vals = []
            for _, row in group.iterrows():
                v_app_sq = row['v_app']**2 if 'v_app' in row else compute_V_app(pd.DataFrame([row]))[0]**2
                D = 0.5 * RHO * v_app_sq * np.abs(geom.get_chord(row['r']))
                D_vals.extend([D, D]) # Dupliqué pour F_n et F_t
            D_list.append(D_vals)
        elif entree == 'GM':
            group = group.sort_values(['r', 'theta'])
            num_r = len(group['r'].unique())
            num_theta = len(group['theta'].unique())
            v_app_grid = compute_V_app(group).reshape(num_r, num_theta)
            chord_grid = geom.get_chord(group['r'].values).reshape(num_r, num_theta)
            D_grid = 0.5 * RHO * (v_app_grid**2) * np.abs(chord_grid)
            D_list.append(D_grid) # Sera redimensionné si besoin
    D_np = np.array(D_list, dtype=np.float32)
    return torch.tensor(D_np, device=device)

def get_V_app_tensor(df, entree, device='cpu'):
    V_list = []
    for _, group in df.groupby(['yaw', 'TSR'] if 'TSR' in df.columns else 'yaw'):
        if entree == 'GV':
            group = group.sort_values(['theta', 'r'])
            v_app_vals = group['v_app'].values if 'v_app' in group.columns else compute_V_app(group)
            V_list.append(v_app_vals)
        elif entree == 'GM':
            group = group.sort_values(['r', 'theta'])
            num_r = len(group['r'].unique())
            num_theta = len(group['theta'].unique())
            v_app_grid = group['v_app'].values if 'v_app' in group.columns else compute_V_app(group)
            V_list.append(v_app_grid.reshape(num_r, num_theta))
    V_np = np.array(V_list, dtype=np.float32)
    return torch.tensor(V_np, device=device)