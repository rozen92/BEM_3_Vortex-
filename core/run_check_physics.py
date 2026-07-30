import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path

# Récupère le chemin de la racine (un niveau au-dessus du dossier de ce script)
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from training.src.data_loader import load_clean_data
from core.models import PolarSurrogate, convert_v_to_f_torch,compute_cp_ct_torch
from core.physics import convert_v_to_f, get_geometry, compute_dynamic_pressure_D, compute_cp, compute_V_app
# Constantes pour le calcul de V_app
from core.config import OMEGA, R_ROTOR, DEFAULT_BEM_SUFFIX

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Chargement des données brutes sur {device}...")

    df = load_clean_data("data/raw/fichier_forces.csv", "data/raw/fichier_vitesses.csv")
    print(f"Nombre de points à vérifier : {len(df)}\n")

    # Interpolateur PyTorch 1D exact
    polar_surrogate = PolarSurrogate(device=device).to(device)

    geom = get_geometry()
    
    # Reconstruction des vecteurs de rayons et cordes conformes à la forme des données
    r_vals = df['r'].values
    c_vals = np.array([float(geom.get_chord(r)) for r in r_vals], dtype=np.float32)

    r_tensor = torch.tensor(r_vals, dtype=torch.float32, device=device)
    c_tensor = torch.tensor(c_vals, dtype=torch.float32, device=device)

    sources = [f'BEM_{DEFAULT_BEM_SUFFIX}', 'SVEN']
    for source_label in sources:
        print(f"\n=== VÉRIFICATION DES DONNÉES {source_label.upper()} ===")
        
        col_v, col_alpha = f'V_eff_{source_label}', f'alpha_{source_label}'
        col_fn, col_ft = f'Fn_{source_label}', f'Ft_{source_label}'
        
        if col_v not in df.columns:
            print(f"Saut : Colonne {col_v} introuvable.")
            continue
            
        val_fn, val_ft = df[col_fn].values, df[col_ft].values

        # 1. Version NUMPY (version originale non différentiable parPyTorch)
        calc_fn_np, calc_ft_np = convert_v_to_f(df[col_v].values, df[col_alpha].values, df['r'].values)
        
        # 2. Version PYTORCH (Nouvel interpolateur linéaire)
        v_eff_tensor = torch.tensor(df[col_v].values, dtype=torch.float32, device=device)
        alpha_tensor = torch.tensor(df[col_alpha].values, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            f_pred_torch = convert_v_to_f_torch(v_eff_tensor, alpha_tensor, r_tensor, c_tensor, polar_surrogate)
        
        calc_fn_pt = f_pred_torch[..., 0].cpu().numpy()
        calc_ft_pt = f_pred_torch[..., 1].cpu().numpy()

        # Calculs des erreurs relatives et absolues
        for label, pred_fn, pred_ft in [("NUMPY", calc_fn_np, calc_ft_np), ("PYTORCH", calc_fn_pt, calc_ft_pt)]:
            err_fn = np.mean(np.abs(pred_fn - val_fn))
            err_ft = np.mean(np.abs(pred_ft - val_ft))
            
            mean_fn = np.mean(np.abs(val_fn))
            mean_ft = np.mean(np.abs(val_ft))
            
            rel_fn = (err_fn / mean_fn) * 100 if mean_fn > 0 else 0
            rel_ft = (err_ft / mean_ft) * 100 if mean_ft > 0 else 0
            
            print(f" [{label}] :")
            print(f"    Erreur Fn : {err_fn:.4f} N/m ({rel_fn:.2f}%)")
            print(f"    Erreur Ft : {err_ft:.4f} N/m ({rel_ft:.2f}%)")


    # =================================================================
    # STATISTIQUES SUR a_n ET a_t (CARTÉSIEN ADIMENSIONNEL)
    # =================================================================
    print("\n=== ANALYSE DES COORDONNÉES CARTÉSIENNES (an, at) ===")
    
    # 1. Calcul de V_app
    theta_rad = np.radians(df['theta'].values)
    yaw_rad = np.radians(df['yaw'].values)
    tsr_val = df['TSR'].values if 'TSR' in df.columns else np.full(len(df), 8.0)
    u_vent = (OMEGA * R_ROTOR) / tsr_val
    
    v_app_sq = (
        (u_vent**2)
        + (OMEGA * r_vals)**2
        - 2 * u_vent * OMEGA * r_vals * np.sin(yaw_rad) * np.cos(theta_rad)
        - (u_vent * np.sin(yaw_rad) * np.sin(theta_rad))**2
    )
    V_app = np.sqrt(v_app_sq)
    
    # 2. Conversion Polaire -> Cartésien
    alpha_sven_rad = np.radians(df['alpha_SVEN'].values)
    V_eff_sven = df['V_eff_SVEN'].values
    
    an_sven = np.sin(alpha_sven_rad) * V_eff_sven / V_app
    at_sven = np.cos(alpha_sven_rad) * V_eff_sven / V_app

    print("\n[a_n SVEN]  => Composante Normale Adimensionnelle")
    print(f"  Moyenne    : {np.mean(an_sven):.4f}")
    print(f"  Écart-type : {np.std(an_sven):.4f}")
    print(f"  Min        : {np.min(an_sven):.4f}  |  Max : {np.max(an_sven):.4f}")

    print("\n[a_t SVEN]  => Composante Tangentielle Adimensionnelle")
    print(f"  Moyenne    : {np.mean(at_sven):.4f}")
    print(f"  Écart-type : {np.std(at_sven):.4f}")
    print(f"  Min        : {np.min(at_sven):.4f}  |  Max : {np.max(at_sven):.4f}")


    # =================================================================
    # VÉRIFICATION DE LA RECONSTRUCTION DIFFÉRENTIABLE
    # =================================================================
    print("\n=== VÉRIFICATION RECONSTRUCTION DIFFÉRENTIABLE (an, at -> Fn, Ft) ===")
    
    an_t = torch.tensor(an_sven, dtype=torch.float32, device=device)
    at_t = torch.tensor(at_sven, dtype=torch.float32, device=device)
    v_app_t = torch.tensor(V_app, dtype=torch.float32, device=device)

    # 1. arctan2 -> alpha_deg
    alpha_reconst_deg = torch.atan2(an_t, at_t) * (180.0 / torch.pi)
    
    # 2. V_app * norm(an, at) -> V_eff
    v_eff_reconst = v_app_t * torch.sqrt(an_t**2 + at_t**2)

    # 3. Projection PyTorch via Surrogate Polaire
    with torch.no_grad():
        f_reconst_torch = convert_v_to_f_torch(v_eff_reconst, alpha_reconst_deg, r_tensor, c_tensor, polar_surrogate)

    calc_fn_reconst = f_reconst_torch[..., 0].cpu().numpy()
    calc_ft_reconst = f_reconst_torch[..., 1].cpu().numpy()

    val_fn_sven = df['Fn_SVEN'].values
    val_ft_sven = df['Ft_SVEN'].values

    err_fn_rec = np.mean(np.abs(calc_fn_reconst - val_fn_sven))
    err_ft_rec = np.mean(np.abs(calc_ft_reconst - val_ft_sven))
    
    mean_fn_sven = np.mean(np.abs(val_fn_sven))
    mean_ft_sven = np.mean(np.abs(val_ft_sven))
    
    rel_fn_rec = (err_fn_rec / mean_fn_sven) * 100 if mean_fn_sven > 0 else 0
    rel_ft_rec = (err_ft_rec / mean_ft_sven) * 100 if mean_ft_sven > 0 else 0
    
    print(f"  Erreur Fn reconstruit : {err_fn_rec:.4f} N/m ({rel_fn_rec:.2f}%)")
    print(f"  Erreur Ft reconstruit : {err_ft_rec:.4f} N/m ({rel_ft_rec:.2f}%)")



    # =================================================================
    # ANALYSE DE LA NORMALISATION PAR PRESSION DYNAMIQUE (OPTION 4)
    # =================================================================
    print("\n=== ANALYSE DE LA NORMALISATION PHYSIQUE (OPTION 4) ===")
    
    # Calcul du dénominateur physique
    D = compute_dynamic_pressure_D(df)
    

    
    Cn_SVEN = df['Fn_SVEN'].values / D
    Ct_SVEN = df['Ft_SVEN'].values / D
    
    print(f"Nombre de points analysés : {len(df)}")
    
    print("\n[Fn_SVEN / D]  =>  (Equivalent Cn)")
    print(f"  Moyenne    : {np.mean(Cn_SVEN):.4f}")
    print(f"  Écart-type : {np.std(Cn_SVEN):.4f}")
    print(f"  Min        : {np.min(Cn_SVEN):.4f}  |  Max : {np.max(Cn_SVEN):.4f}")
    q5, q50, q95 = np.percentile(Cn_SVEN, [5, 50, 95])
    print(f"  Percentiles: 5%={q5:.4f}  |  Médiane={q50:.4f}  |  95%={q95:.4f}")

    print("\n[Ft_SVEN / D]  =>  (Equivalent Ct)")
    print(f"  Moyenne    : {np.mean(Ct_SVEN):.4f}")
    print(f"  Écart-type : {np.std(Ct_SVEN):.4f}")
    print(f"  Min        : {np.min(Ct_SVEN):.4f}  |  Max : {np.max(Ct_SVEN):.4f}")
    q5_t, q50_t, q95_t = np.percentile(Ct_SVEN, [5, 50, 95])
    print(f"  Percentiles: 5%={q5_t:.4f}  |  Médiane={q50_t:.4f}  |  95%={q95_t:.4f}")
    
    # Vérification du rapport d'échelle
    ratio = np.mean(np.abs(Cn_SVEN)) / np.mean(np.abs(Ct_SVEN))
    print(f"\nRapport d'amplitude absolu (Moy |Cn| / Moy |Ct|) : {ratio:.2f}")
   
    # =================================================================
    # VÉRIFICATION DE CP ET CT DIFFÉRENTIABLE vs NUMPY
    # =================================================================
    print("\n=== VÉRIFICATION CP & CT DIFFÉRENTIABLE vs NUMPY ===")
    
    yaw_test = df['yaw'].unique()[0]
    df_case = df[df['yaw'] == yaw_test].copy()
    
    # 1. Version NUMPY de référence
    cp_ct_np = compute_cp(df_case, 'Fn_SVEN', 'Ft_SVEN')
    cp_np = cp_ct_np['Cp_SVEN'].values[0]
    ct_np = cp_ct_np['Ct_SVEN'].values[0]
    
    # 2. Version PYTORCH (Format Grid CNN)
    df_case = df_case.sort_values(['r', 'theta'])
    r_uniques = np.sort(df_case['r'].unique())
    theta_uniques = np.sort(df_case['theta'].unique())
    
    fn_grid = df_case['Fn_SVEN'].values.reshape(len(r_uniques), len(theta_uniques))
    ft_grid = df_case['Ft_SVEN'].values.reshape(len(r_uniques), len(theta_uniques))
    
    R_grid, _ = np.meshgrid(r_uniques, theta_uniques, indexing='ij')
    
    fn_tensor = torch.tensor(fn_grid, dtype=torch.float32, device=device).unsqueeze(0)
    ft_tensor = torch.tensor(ft_grid, dtype=torch.float32, device=device).unsqueeze(0)
    r_tensor_grid = torch.tensor(R_grid, dtype=torch.float32, device=device).unsqueeze(0)
    
    tsr_test = df_case['TSR'].iloc[0] if 'TSR' in df_case.columns else 8.0
    u_inf = (OMEGA * R_ROTOR) / tsr_test
    u_inf_tensor = torch.tensor([u_inf], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        cp_pt, ct_pt = compute_cp_ct_torch(fn_tensor, ft_tensor, r_tensor_grid, u_inf_tensor, is_cnn=True)
        
    print(f"  [NUMPY]   -> Cp: {cp_np:.6f} | Ct: {ct_np:.6f}")
    print(f"  [PYTORCH] -> Cp: {cp_pt.item():.6f} | Ct: {ct_pt.item():.6f}")
    
    err_cp = abs(cp_np - cp_pt.item())
    err_ct = abs(ct_np - ct_pt.item())
    print(f"  Erreur Cp: {err_cp:.2e} | Erreur Ct: {err_ct:.2e}")
    


if __name__ == "__main__":
    main()