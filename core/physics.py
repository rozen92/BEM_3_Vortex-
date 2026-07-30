import torch

import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d, RegularGridInterpolator
from core.config import RHO, U_INFTY, PITCH_RAD, R_ROTOR, OMEGA, Dist_R

## Calcul (en une fois) de dl_map et dr pour le calcul des puissances
r_unique = Dist_R
nodes = [0.21] # Rayon du moyeu
for i in range(len(r_unique)):
        next_node = 2 * r_unique[i] - nodes[-1]
        nodes.append(next_node)

dl_map = dict(zip(r_unique, np.diff(nodes)))


# ==========================================
# GESTIONNAIRE DE GÉOMÉTRIE
# ==========================================
class BladeGeometry:
    def __init__(self, geom_file="data/geometry/blade_geom.csv", airfoils_file="data/geometry/airfoils.csv"):
        self.geom_file = geom_file
        self.airfoils_file = airfoils_file
        
        self._load_blade_geometry()
        self._load_airfoils()

    def _load_blade_geometry(self):
        if not os.path.exists(self.geom_file):
            raise FileNotFoundError(f"Fichier géométrie introuvable: {self.geom_file}")

        df_geom = pd.read_csv(self.geom_file)
        self.nodesRadius = df_geom['r'].values
        self.nodesChord = df_geom['chord'].values
        self.nodesTwist_deg = df_geom['beta_deg'].values

        self.get_chord = interp1d(self.nodesRadius, self.nodesChord, kind='linear', fill_value="extrapolate")
        self.get_twist_rad = interp1d(self.nodesRadius, np.radians(self.nodesTwist_deg), kind='linear', fill_value="extrapolate")

    def _load_airfoils(self):
        if not os.path.exists(self.airfoils_file):
            raise FileNotFoundError(f"Fichier polaires introuvable: {self.airfoils_file}")

        df_airfoils = pd.read_csv(self.airfoils_file)
        
        alphas_deg = np.sort(df_airfoils['alpha_deg'].unique())
        radii = np.sort(df_airfoils['r'].unique())
        
        df_cl = df_airfoils.pivot(index='alpha_deg', columns='r', values='Cl')
        df_cd = df_airfoils.pivot(index='alpha_deg', columns='r', values='Cd')
        
        cl_matrix = df_cl.loc[alphas_deg, radii].values
        cd_matrix = df_cd.loc[alphas_deg, radii].values
        
        self._interp_cl = RegularGridInterpolator((alphas_deg, radii), cl_matrix, method='linear', bounds_error=False, fill_value=None)
        self._interp_cd = RegularGridInterpolator((alphas_deg, radii), cd_matrix, method='linear', bounds_error=False, fill_value=None)

    def get_cl_cd(self, r, alpha_deg):
        """ 
        Retourne Cl et Cd interpolés. 
        ATTENTION: alpha_deg doit être en DEGRÉS.
        """
        cl = self._interp_cl((alpha_deg, r))
        cd = self._interp_cd((alpha_deg, r))
        return float(cl), float(cd)


geom_db = None
def get_geometry():
    global geom_db
    if geom_db is None:
        geom_db = BladeGeometry()
    return geom_db


# ==========================================
# FONCTIONS DE CONVERSIONS PHYSIQUES
# ==========================================

def convert_v_to_f(V_eff, alpha_deg, r):
    """
    Convertit v (V_eff, alpha) en efforts (Fn, Ft) en [N/m].
    Prend en entrée alpha en DEGRÉS.
    """
    geom = get_geometry()
    
    alpha_rad = np.radians(alpha_deg)
    
    if isinstance(r, (list, np.ndarray, pd.Series)):
        c = np.array([geom.get_chord(ri) for ri in r])
        cl_cd = [geom.get_cl_cd(ri, ai) for ri, ai in zip(r, alpha_deg)]
        Cl = np.array([item[0] for item in cl_cd])
        Cd = np.array([item[1] for item in cl_cd])
    else:
        c = geom.get_chord(r)
        Cl, Cd = geom.get_cl_cd(r, alpha_deg)
    
    # Projections géométriques
    Cn = Cl * np.cos(alpha_rad) + Cd * np.sin(alpha_rad)
    Ct = Cl * np.sin(alpha_rad) - Cd * np.cos(alpha_rad)
    
    # Pression dynamique et Forces
    q = 0.5 * RHO * (V_eff**2) * abs(c)
    Fn = q * Cn
    Ft =- q * Ct 
    
    return Fn, Ft


def compute_cp(df, col_fn, col_ft, R_rotor=R_ROTOR, Nb_pales=3, omega=OMEGA):
    """
    Calcule Cp, Ct, ainsi que les écarts-types sur la rotation et l'envergure.
    """
    df_calc = df.copy()
    geom = get_geometry()
    
    # 1. Calcul des longueurs de sections (dl)
    r_unique = np.sort(df_calc['r'].unique())
    nodes = [0.21] # Rayon du moyeu
    for i in range(len(r_unique)):
        next_node = 2 * r_unique[i] - nodes[-1]
        nodes.append(next_node)
    
    dl_map = dict(zip(r_unique, np.diff(nodes)))
    df_calc['dl'] = df_calc['r'].map(dl_map)
    
    # 2. Angle structurel de la pale (Pitch + Twist)
    df_calc['phi_rad'] = PITCH_RAD + geom.get_twist_rad(df_calc['r'])
    
    # 3. Projections Aérodynamiques
    Ft_corrige = df_calc[col_ft] * -1
    cos_phi = np.cos(df_calc['phi_rad'])
    sin_phi = np.sin(df_calc['phi_rad'])
    
    # Couple (Force dans le plan de rotation)
    df_calc['dQ_r'] = df_calc[col_fn] * sin_phi + Ft_corrige * cos_phi
    df_calc['dQ'] = df_calc['dQ_r'] * df_calc['r'] * df_calc['dl']
    
    # Poussée axiale (Thrust - Force perpendiculaire au plan de rotation)
    df_calc['dT_r'] = df_calc[col_fn] * cos_phi - Ft_corrige * sin_phi
    df_calc['dT'] = df_calc['dT_r'] * df_calc['dl']
    
    # 4. Préparation du GroupBy
    group_cols = ['yaw']
    if 'TSR' in df_calc.columns: group_cols.append('TSR')
        
    results = []
    model_name = col_fn.replace('Fn_', '') 
    
    for name, group in df_calc.groupby(group_cols):
        if isinstance(name, tuple):
            yaw_val = name[0]
            tsr_val = name[1] if len(name) > 1 else 8.0
        else:
            yaw_val = name
            tsr_val = 8.0
            
        u_vent = (omega * R_rotor) / tsr_val
        
        # Grandeurs de référence
        P_vent = 0.5 * RHO * np.pi * (R_rotor**2) * (u_vent**3)
        F_vent = 0.5 * RHO * np.pi * (R_rotor**2) * (u_vent**2) 
        
        # Intégrations spatiales (Somme sur le rayon r pour chaque azimut theta)
        Q_theta = group.groupby('theta')['dQ'].sum() 
        T_theta = group.groupby('theta')['dT'].sum()
        
        # Séries temporelles sur un tour (Multipliées par 3 pales)
        Cp_serie = (Nb_pales * Q_theta * omega) / P_vent
        Ct_serie = (Nb_pales * T_theta) / F_vent
        
        # Moyennes et Écarts-types Temporels (sur theta)
        Cp_mean = Cp_serie.mean()
        Cp_std = Cp_serie.std()
        
        Ct_mean = Ct_serie.mean()
        Ct_std = Ct_serie.std()
        
        # Écarts-types Spatiaux (Dispersion de Fn et Ft sur r, moyennée sur theta)
        std_Fn_mean = group.groupby('theta')[col_fn].std().mean()
        std_Ft_mean = group.groupby('theta')[col_ft].std().mean()
        
        res_dict = {
            'yaw': yaw_val,
            'TSR': tsr_val,
            f'Cp_{model_name}': Cp_mean,
            f'Cp_std_{model_name}': Cp_std,
            f'Ct_{model_name}': Ct_mean,
            f'Ct_std_{model_name}': Ct_std,
            f'std_Fn_{model_name}': std_Fn_mean,
            f'std_Ft_{model_name}': std_Ft_mean
        }
        results.append(res_dict)
        
    return pd.DataFrame(results)

def compute_cp_diff(col_fn:torch.tensor, col_ft:torch.tensor, device, R_rotor=R_ROTOR, Nb_pales=3, omega=OMEGA) :
    
    if col_fn.shape[1] != 2592 or col_ft.shape[1] != 2592:
        raise Exception(f"Les tenseurs d'entrée ne sont pas de bonne dimension.")
    if col_fn.device.type != device :
        col_fn = col_fn.to(device)
    if col_ft.device.type != device :
        col_ft = col_ft.to(device)
    
    geom = get_geometry()
    
    # 1. Récupérer les distances entre chaque noeuds et les rayons
    dr_tens = torch.tensor(Dist_R*72, dtype = torch.float32, device = device) #Reshape de la distribution des rayons par azimuth
    dl_tens = torch.tensor(list(dl_map.values())*72, dtype = torch.float32, device = device)
    
    # 2. Angle structurel de la pale (Pitch + Twist) --> Pas besoin d'être auto-diff
    phi_rad_tens = PITCH_RAD + torch.tensor(geom.get_twist_rad(dr_tens.cpu().numpy()), device = device)
    
    # 3. Projections Aérodynamiques
    Ft_corrige = col_ft * -1 
    cos_phi = torch.cos(phi_rad_tens)
    sin_phi = torch.sin(phi_rad_tens)

    # 4. Calcul de la densité de puissance
    dQ_r = col_fn * sin_phi + Ft_corrige * cos_phi
    dQ = dQ_r * dr_tens * dl_tens
    
    return dQ


def compute_V_app(df):
    """
    Calcule la vitesse apparente V_app.
    """
    theta_rad = np.radians(df['theta'].values)
    yaw_rad = np.radians(df['yaw'].values)
    tsr_val = df['TSR'].values if 'TSR' in df.columns else np.full(len(df), 8.0)
    
    u_vent = (OMEGA * R_ROTOR) / tsr_val
    r_val = df['r'].values
    
    # Formule mathématique de V_app au carré (projection dans le plan local (n,t) de la section)
    v_app_sq = (
        (u_vent**2)
        + (OMEGA * r_val)**2
        - 2 * u_vent * OMEGA * r_val * np.sin(yaw_rad) * np.cos(theta_rad)
        - (u_vent * np.sin(yaw_rad) * np.sin(theta_rad))**2
    )

    return np.sqrt(v_app_sq)

def compute_dynamic_pressure_D(df):
    """
    Calcule le dénominateur de normalisation géométrique D.
    D = 0.5 * rho * V_app^2 * |c(r)|
    """
    geom = get_geometry()
    
    # 1. Calcul de V_app
    v_app = compute_V_app(df)
    
    # 2. Rayon et valeur absolue de la corde locale
    r_val = df['r'].values
    c_val = np.array([float(geom.get_chord(ri)) for ri in r_val])
    c_val = np.abs(c_val) 
    
    # 3. Pression dynamique locale
    D = 0.5 * RHO * (v_app**2) * c_val
    
    return D