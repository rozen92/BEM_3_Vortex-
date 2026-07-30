"""
=============================================================================
 CONFIGURATION GLOBALE - BEM 2 VORTEX
=============================================================================
"""

# ==========================================
# 1. CONSTANTES PHYSIQUES
# ==========================================
RHO = 1.198                 # Densité de l'air [kg/m3]
U_INFTY = 12.52             # Vitesse du vent (TSR 8) [m/s]
PITCH_RAD = -0.040143       # Angle de pitch en radians (-2.3 degrés)
R_ROTOR = 2.25              # Rayon du rotor [m]
OMEGA = 44.5163679          # Vitesse de rotation [rad/s]

Dist_R = [0.2119407,  0.21968875, 0.23512587, 0.25813459, 0.28853979, 0.32611007, # Distribution des rayons de la pale
 0.3705595,  0.42154979, 0.47869288, 0.54155386, 0.60965434, 0.68247602,
 0.75946469 ,0.84003441, 0.92357201, 1.00944172, 1.09699   , 1.18555057,
 1.27444943 ,1.36301   , 1.45055828, 1.53642799, 1.61996559, 1.70053531,
 1.77752398 ,1.85034566, 1.91844614, 1.98130712, 2.03845021, 2.0894405,
 2.13388993 ,2.17146021, 2.20186541, 2.22487413, 2.24031125, 2.2480593 ]

# ==========================================
# 2. PARAMÈTRES D'ENTRAÎNEMENT (TRAINER)
# ==========================================
EPOCHS_OPTUNA = 500        
EPOCHS_FINAL = 500         
EPOCHS_AE = 500            

RANDOM_SEED = 42
CV_SPLITS = 5               
DEVICE = "cuda"             

# ==========================================
# 3. RECHERCHE OPTUNA
# ==========================================
TRIALS_GV = 150             
TRIALS_GM = 50              
TRIALS_AE = 15              

LR_BOUNDS_NOAE = (1e-5, 3e-3)
LR_BOUNDS_AE   = (5e-4, 2e-2)
LR_BOUNDS_AE_PRETRAIN = (1e-4, 5e-3)
DROPOUT_BOUNDS = (1e-4, 5e-1)

MLP_LAYERS_BOUNDS = (1, 4)
MLP_NEURONS_CHOICES = [128, 192, 256, 320, 384, 448, 512,576,640,704,768]

CNN_LAYERS_BOUNDS = (2, 6)
CNN_FILTERS_CHOICES = [16, 32, 64, 128]
KERNEL_SIZE = 3             
PADDING_R = 1               
PADDING_THETA = 1           

# =========================================================================
# GRILLE DE RECHERCHE ARCHITECTURALE
# =========================================================================
ENTREES = ['GV', 'GM']
RESIDUELLES = ['0', '1', '2', '2+']
INTERMS = ['f', 'v']
OPTIONS = ['A', 'B']

# Proportions (en %) des couples (yaw, TSR) d'entraînement utilisées pour entraîner
# les modèles prédictifs (GM/GV). Les auto-encodeurs, eux, sont toujours entraînés
# sur 100% des données (voir training/run_train_eval.py, Phase 1).
DATA_PCTS = [100]

AE_NATURES = ['V', 'M']
AE_DIMS = [16, 32, 64, 128, 256, 512, 1024]
AE_LAYERS_BOUNDS = (2, 4)

AE_RESIDUAL_ALIASES = {'2': '0'}

def get_ae_residual_key(residuelle):
    """ Clé canonique de résiduelle pour la banque d'auto-encodeurs (déduplique '0' et '2'). """
    res_base = str(residuelle).replace('+', '')
    return AE_RESIDUAL_ALIASES.get(res_base, res_base)

# =========================================================================
# VARIANTES DE CALCUL BEM (DUM / IFP / P&P)
# =========================================================================
BEM_SUFFIXES = ['DUM', 'IFP', 'P&P']
DEFAULT_BEM_SUFFIX = 'IFP'  # référence historique (= ancien Fn_BEM générique)


def needs_bem_suffix(residuelle) -> bool:
    """ True si ce type de résiduelle utilise la BEM en entrée (donc variante-dépendant). """
    return str(residuelle).replace('+', '') in ('1', '2')

def format_scaler_name(entree, residuelle, inter, bem_suffix=None):
    suffix = f"_{bem_suffix}" if needs_bem_suffix(residuelle) else ""
    return f"{entree}_{residuelle}_{inter}{suffix}"

def format_model_name(entree, residuelle, inter, ae_label, option, pct, bem_suffix=None):
    suffix = f"_{bem_suffix}" if needs_bem_suffix(residuelle) else ""
    return f"{entree}_{residuelle}_{inter}{suffix}_{ae_label}_{option}_P{pct}"

def format_ae_key(residuelle, inter, ae_nature, ae_dim, bem_suffix=None):
    res_base = get_ae_residual_key(residuelle)
    suffix = f"_{bem_suffix}" if res_base == '1' else ""
    return f"{res_base}_{inter}{suffix}_D{ae_nature}{ae_dim}"

# =========================================================================
# HYPERPARAMÈTRES D'ENTRAÎNEMENT & OPTUNA
# =========================================================================
PRUNER_WARMUP = 150      
RATIO_THRESHOLD = 2.5    

AE_JSON_PATH = "training/hyperparametres/ae_hyperparameters.json"
AE_WEIGHTS_DIR = "training/models/ae/"

