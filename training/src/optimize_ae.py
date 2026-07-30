import optuna
import json
import torch
import torch.nn as nn
import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

from core.models import ConvolutionalAutoencoder, LinearAutoencoder, TorchScaler
from training.src.data_loader import format_data, get_D_tensor
from training.src.trainer import fit_model, cross_validate
from core.config import EPOCHS_AE, TRIALS_AE, LR_BOUNDS_AE_PRETRAIN, CV_SPLITS, AE_LAYERS_BOUNDS, format_ae_key, format_scaler_name

class AEPhysicalLoss(nn.Module):
    """
    Loss pour l'AE évaluant la reconstruction dans l'espace des forces brutes (N/m),
    puis renormalisant par l'écart-type global pour garder un gradient stable.
    Opère toujours sur le format canonique GM (indépendant de l'entrée du modèle prédictif).
    """
    def __init__(self, scaler_Y, scaler_F_abs, out_dim, is_cnn_for_ae, device):
        super().__init__()
        self.scaler_Y = TorchScaler(scaler_Y, device)

        mean_abs = torch.tensor(scaler_F_abs.mean_, dtype=torch.float32, device=device)
        scale_abs = torch.tensor(scaler_F_abs.scale_, dtype=torch.float32, device=device)

        if is_cnn_for_ae:
            # Format canonique 4D (N, 2, 36, 72) : canal 0 = Fn, canal 1 = Ft
            self.mean_flat = mean_abs.view(1, 2, 1, 1)
            self.scale_flat = scale_abs.view(1, 2, 1, 1)
        else:
            # Format canonique MLP aplati (N, 5184) : [Fn_1...Fn_2592, Ft_1...Ft_2592]
            half_len = out_dim // 2
            self.mean_flat = torch.cat([mean_abs[0].repeat(half_len), mean_abs[1].repeat(half_len)])
            self.scale_flat = torch.cat([scale_abs[0].repeat(half_len), scale_abs[1].repeat(half_len)])

    def forward(self, y_pred_norm, y_true_norm, D_phys):
        pred_D = self.scaler_Y.inverse_transform(y_pred_norm)
        true_D = self.scaler_Y.inverse_transform(y_true_norm)

        pred_abs = pred_D * D_phys
        true_abs = true_D * D_phys

        err_abs = pred_abs - true_abs
        err_norm = err_abs / self.scale_flat

        return nn.functional.mse_loss(err_norm, torch.zeros_like(err_norm))


def optimize_and_train_ae(df_train, residuelle, inter, latent_dim, ae_nature, n_trials=TRIALS_AE, bem_suffix=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved_name = format_ae_key(residuelle, inter, ae_nature, latent_dim, bem_suffix)

    os.makedirs("training/hyperparametres", exist_ok=True)
    os.makedirs("training/models/ae", exist_ok=True)

    ae_weights_path = f"training/models/ae/ae_{saved_name}.pth"
    if os.path.exists(ae_weights_path):
        print(f"   [INFO] Auto-encodeur {saved_name} déjà existant. Entraînement ignoré.")
        return

    # Toujours charger Y en format canonique GM (N, 2, 36, 72),
    # indépendamment de l'entrée du modèle prédictif (GV ou GM)
    _, Y_full_raw = format_data(df_train, 'GM', residuelle, inter, is_train=True, device=device, bem_suffix=bem_suffix)

    is_cnn_for_ae = (ae_nature == 'M')

    if is_cnn_for_ae:
        out_dim = 2
        Y_target = Y_full_raw  # Déjà (N, 2, 36, 72)
    else:
        Y_target = Y_full_raw.reshape(Y_full_raw.size(0), -1)  # (N, 5184)
        out_dim = Y_target.shape[1]

    print(f"\n{'='*50}")
    print(f" CRÉATION AUTO-ENCODEUR : {saved_name}")
    print(f" Type : {'Linéaire (MLP)' if ae_nature == 'V' else 'Convolutif (CNN)'}")
    print(f"{'='*50}")

    # PRÉPARATION DE LA LOSS PHYSIQUE RENORMALISÉE (uniquement si inter == 'f')
    physical_criterion = None
    if inter == 'f':
        fn_abs = df_train['Fn_SVEN'].values.astype(np.float32).reshape(-1, 1)
        ft_abs = df_train['Ft_SVEN'].values.astype(np.float32).reshape(-1, 1)
        scaler_F_abs = StandardScaler()
        scaler_F_abs.fit(np.hstack([fn_abs, ft_abs]))

        # Scaler de l'espace cible en format canonique GM
        scaler_path = f"training/scalers/scaler_Y_{format_scaler_name('GM', residuelle, inter, bem_suffix)}.pkl"
        with open(scaler_path, 'rb') as f:
            scaler_Y = pickle.load(f)

        # D_tensor en format canonique GM : (N, 36, 72)
        D_tensor = get_D_tensor(df_train, 'GM', device)

        if is_cnn_for_ae:
            # (N, 36, 72) -> (N, 1, 36, 72) -> broadcast sur les 2 canaux
            D_tensor = D_tensor.view(-1, 1, 36, 72).expand(-1, 2, -1, -1).contiguous()
        else:
            # (N, 36, 72) -> (N, 2592) -> cat -> (N, 5184) : [D_Fn, D_Ft]
            D_flat = D_tensor.reshape(-1, 36 * 72)
            D_tensor = torch.cat([D_flat, D_flat], dim=1)

        physical_criterion = AEPhysicalLoss(scaler_Y, scaler_F_abs, out_dim, is_cnn_for_ae, device)


    cv_inter = 'f' if physical_criterion is not None else None

    def objective_ae(trial):
        lr = trial.suggest_float('ae_lr', LR_BOUNDS_AE_PRETRAIN[0], LR_BOUNDS_AE_PRETRAIN[1], log=True)
        n_layers = trial.suggest_int('ae_depth', AE_LAYERS_BOUNDS[0], AE_LAYERS_BOUNDS[1])

        if is_cnn_for_ae:
            base_filters = trial.suggest_categorical('ae_base_filters', [8, 16, 32])
            model_class = ConvolutionalAutoencoder
            model_kwargs = {'in_channels': out_dim, 'latent_dim': latent_dim, 'depth': n_layers, 'base_filters': base_filters, 'device': device}
        else:
            model_class = LinearAutoencoder
            model_kwargs = {'in_features': out_dim, 'latent_dim': latent_dim, 'n_layers': n_layers, 'device': device}

        criterion = physical_criterion if physical_criterion is not None else nn.MSELoss()

        mean_val_loss, _, _ = cross_validate(
            X_full=Y_target, Y_full=Y_target, model_class=model_class, model_kwargs=model_kwargs,
            criterion_builder=lambda train_idx, val_idx: criterion,
            epochs=EPOCHS_AE, lr=lr, n_splits=CV_SPLITS, device=device,
            inter=cv_inter, D_phys_full=D_tensor if physical_criterion is not None else None,
            trial=trial
        )
        return mean_val_loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study_ae = optuna.create_study(direction='minimize', pruner=pruner)
    print(f"   [1/2] Recherche Optuna (validation croisée, {CV_SPLITS} folds) pour l'architecture AE ({n_trials} trials)...")
    study_ae.optimize(objective_ae, n_trials=n_trials, show_progress_bar=True)

    best_ae_params = study_ae.best_params
    best_ae_params.update({'use_autoencoder': True, 'latent_dim': latent_dim})

    print(f"   -> Meilleurs paramètres trouvés : {best_ae_params}")

    print(f"   [2/2] Entraînement de l'AE final sur l'ensemble d'entraînement complet ({EPOCHS_AE} époques)...")
    criterion = physical_criterion if physical_criterion is not None else nn.MSELoss()

    if is_cnn_for_ae:
        final_ae = ConvolutionalAutoencoder(in_channels=out_dim, latent_dim=latent_dim, depth=best_ae_params['ae_depth'], base_filters=best_ae_params['ae_base_filters'], device=device).to(device)
    else:
        final_ae = LinearAutoencoder(in_features=out_dim, latent_dim=latent_dim, n_layers=best_ae_params['ae_depth'], device=device).to(device)

    final_ae, final_loss = fit_model(
        model=final_ae, X=Y_target, Y=Y_target, criterion=criterion,
        epochs=EPOCHS_AE, lr=best_ae_params['ae_lr'], device=device,
        inter=cv_inter, D_phys=D_tensor if physical_criterion is not None else None, show_progress=True
    )

    json_master_path = "training/hyperparametres/ae_hyperparameters.json"
    if os.path.exists(json_master_path):
        with open(json_master_path, "r") as f:
            all_ae_params = json.load(f)
    else:
        all_ae_params = {}

    all_ae_params[saved_name] = best_ae_params
    with open(json_master_path, "w") as f:
        json.dump(all_ae_params, f, indent=4)

    torch.save(final_ae.state_dict(), ae_weights_path)
    print(f"   [OK] Poids sauvegardés. Loss finale normalisée : {final_loss:.6f}")
