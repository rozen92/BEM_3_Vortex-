import os
import copy
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
import optuna

def train_one_epoch(model, X, Y, optimizer, criterion, device, inter=None, v_bem_phys=None, D_phys=None, f_bem_phys=None, v_app=None, u_inf=None):
    model.train()
    if hasattr(criterion, 'train'): criterion.train()

    X, Y = X.to(device), Y.to(device)
    optimizer.zero_grad()

    preds = model(X)

    kwargs = {}
    if v_app is not None: kwargs['v_app'] = v_app.to(device)
    if u_inf is not None: kwargs['u_inf'] = u_inf.to(device)
    if inter == 'v' and v_bem_phys is not None: kwargs['v_bem_phys'] = v_bem_phys.to(device)
    elif inter == 'f':
        if D_phys is not None: kwargs['D_phys'] = D_phys.to(device)
        if f_bem_phys is not None: kwargs['f_bem_phys'] = f_bem_phys.to(device)

    loss = criterion(preds, Y, **kwargs)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_model(model, X_eval, Y_eval, criterion, device, inter=None, v_bem_phys=None, D_phys=None, f_bem_phys=None, v_app=None, u_inf=None):
    model.eval()
    if hasattr(criterion, 'eval'): criterion.eval()

    X_eval, Y_eval = X_eval.to(device), Y_eval.to(device)

    with torch.no_grad():
        preds = model(X_eval)

        kwargs = {}
        if v_app is not None: kwargs['v_app'] = v_app.to(device)
        if u_inf is not None: kwargs['u_inf'] = u_inf.to(device)
        if inter == 'v' and v_bem_phys is not None: kwargs['v_bem_phys'] = v_bem_phys.to(device)
        elif inter == 'f':
            if D_phys is not None: kwargs['D_phys'] = D_phys.to(device)
            if f_bem_phys is not None: kwargs['f_bem_phys'] = f_bem_phys.to(device)

        loss = criterion(preds, Y_eval, **kwargs)
    return loss.item(), preds

def fit_model(model, X, Y, criterion, epochs, lr, device, inter=None, v_bem_phys=None, D_phys=None, f_bem_phys=None, v_app=None, u_inf=None, show_progress=True, trial=None, fold=0):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    best_weights = None

    iterator = range(epochs)
    if show_progress:
        iterator = tqdm(iterator, desc="Training Model", leave=False)

    for epoch in iterator:
        loss = train_one_epoch(model, X, Y, optimizer, criterion, device, inter, v_bem_phys, D_phys, f_bem_phys, v_app, u_inf)
        
        if loss < best_loss:
            best_loss = loss
            best_weights = copy.deepcopy(model.state_dict())
            
        if trial is not None and fold == 0:
            trial.report(loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        if show_progress and (epoch + 1) % 50 == 0:
            if hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({"Loss": f"{loss:.6f}"})
                
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model, best_loss

def cross_validate(X_full, Y_full, model_class, model_kwargs, criterion_builder, epochs, lr,
                   n_splits=3, device='cpu', inter=None, v_bem_phys_full=None, D_phys_full=None, f_bem_phys_full=None, v_app_full=None, u_inf_full=None,
                   compute_metrics_fn=None, metrics_kwargs=None, trial=None):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_losses = []
    cv_custom_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full.cpu().numpy())):
        X_tr, Y_tr = X_full[train_idx], Y_full[train_idx]
        X_val, Y_val = X_full[val_idx], Y_full[val_idx]

        v_bem_tr = v_bem_phys_full[train_idx] if v_bem_phys_full is not None else None
        v_bem_val = v_bem_phys_full[val_idx] if v_bem_phys_full is not None else None

        D_tr = D_phys_full[train_idx] if D_phys_full is not None else None
        D_val = D_phys_full[val_idx] if D_phys_full is not None else None

        f_bem_tr = f_bem_phys_full[train_idx] if f_bem_phys_full is not None else None
        f_bem_val = f_bem_phys_full[val_idx] if f_bem_phys_full is not None else None

        v_app_tr = v_app_full[train_idx] if v_app_full is not None else None
        v_app_val = v_app_full[val_idx] if v_app_full is not None else None

        u_inf_tr = u_inf_full[train_idx] if u_inf_full is not None else None
        u_inf_val = u_inf_full[val_idx] if u_inf_full is not None else None

        model = model_class(**model_kwargs).to(device)
        criterion = criterion_builder(train_idx, val_idx)

        model, _ = fit_model(
            model=model, X=X_tr, Y=Y_tr, criterion=criterion,
            epochs=epochs, lr=lr, device=device, inter=inter,
            v_bem_phys=v_bem_tr, D_phys=D_tr, f_bem_phys=f_bem_tr, v_app=v_app_tr, u_inf=u_inf_tr,
            show_progress=False, trial=trial, fold=fold
        )

        val_loss, preds_val = evaluate_model(
            model, X_val, Y_val, criterion, device, inter,
            v_bem_phys=v_bem_val, D_phys=D_val, f_bem_phys=f_bem_val, v_app=v_app_val, u_inf=u_inf_val
        )
        cv_losses.append(val_loss)
        
        if compute_metrics_fn is not None:
            kwargs = metrics_kwargs or {}
            score = compute_metrics_fn(model, X_val, Y_val, val_idx, preds_val, **kwargs)
            cv_custom_scores.append(score)
            
    mean_val_loss = np.mean(cv_losses)
    if cv_custom_scores and isinstance(cv_custom_scores[0], dict):
        keys = cv_custom_scores[0].keys()
        mean_custom_score = {k: np.mean([s[k] for s in cv_custom_scores]) for k in keys}
        std_custom_score = {k: np.std([s[k] for s in cv_custom_scores]) for k in keys}
    else:
        mean_custom_score = np.mean(cv_custom_scores) if cv_custom_scores else None
        std_custom_score = np.std(cv_custom_scores) if cv_custom_scores else None

    return mean_val_loss, mean_custom_score, std_custom_score