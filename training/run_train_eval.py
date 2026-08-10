import time
import os
import itertools
from training.src.data_loader import load_clean_data, get_splits, subsample_train
from training.src.optimize_ae import optimize_and_train_ae
from training.src.optimize import optimize
from training.src.evaluate import evaluator, evaluate_baselines
from core.config import (INTERMS, AE_NATURES, AE_DIMS, TRIALS_AE, TRIALS_GV, TRIALS_GM, DATA_PCTS,
                          DEFAULT_BEM_SUFFIX, needs_bem_suffix, get_ae_residual_key,
                          format_model_name)

# Session à lancer : une session = 12 modèles de la grille {0,1,2}x{f,v}x{ae,sans ae}
# pour (entree, option) fixés, + 1 modèle "2+" supplémentaire (13 au total).
SESSION = "GM_A"  # "GM_A", "GM_B", "GV_A" ou "GV_B"

# Le modèle "2+" (entree='GV' et has_ae=True imposés) dispatché dans chaque session.
EXTRA_2PLUS_INTER = {"GM_A": "f", "GM_B": "f", "GV_A": "v", "GV_B": "v"}


def main():
    if SESSION not in EXTRA_2PLUS_INTER:
        raise ValueError(f"SESSION invalide : {SESSION!r}. Attendu : 'GM_A', 'GM_B', 'GV_A' ou 'GV_B'.")
    entree, option = SESSION.split('_')

    os.makedirs("training/performance", exist_ok=True)
    df_full = load_clean_data()
    df_train, df_test = get_splits(df_full, seed=42)

    baseline_scores = evaluate_baselines(df_test, DEFAULT_BEM_SUFFIX)

    # 1. PRÉ-ENTRAÎNEMENT DE LA BANQUE D'AUTO-ENCODEURS

    print("\n" + "="*80)
    print(" PHASE 1 : PRÉ-ENTRAÎNEMENT DE LA BANQUE D'AUTO-ENCODEURS")
    print("="*80)

    residuelles_ae = ['0', '1']

    for r, i, nature, dim in itertools.product(residuelles_ae, INTERMS, AE_NATURES, AE_DIMS):
        # Seule l'AE '1' dépend de la variante BEM (cible Y = SVEN - BEM) ; '0' reste unique.
        bem_suffix = DEFAULT_BEM_SUFFIX if get_ae_residual_key(r) == '1' else None
        t0 = time.perf_counter()
        optimize_and_train_ae(df_train, residuelle=r, inter=i, latent_dim=dim, ae_nature=nature, n_trials=TRIALS_AE, bem_suffix=bem_suffix)
        tag = f"{r}_{i}_D{nature}{dim}" + (f"_{bem_suffix}" if bem_suffix else "")
        print(f"   [CHRONO] AE {tag} : {time.perf_counter()-t0:.1f}s")

    # 2. PLAN D'EXPÉRIENCES (13 MODÈLES DE LA SESSION)

    print("\n" + "="*80)
    print(f" PHASE 2 : PLAN D'EXPÉRIENCES - SESSION {SESSION}")
    print("="*80)

    test_models = [
        (entree, r, i, has_ae, option)
        for r, i, has_ae in itertools.product(['0', '1', '2'], INTERMS, [False, True])
    ]
    test_models.append(('GV', '2+', EXTRA_2PLUS_INTER[SESSION], True, option))

    for pct in DATA_PCTS:
        df_train_pct = subsample_train(df_train, pct)

        print("\n" + "="*80)
        print(f" PROPORTION DE DONNÉES D'ENTRAÎNEMENT : {pct}% "
              f"({df_train_pct['yaw'].nunique()}/{df_train['yaw'].nunique()} yaw)")
        print("="*80)

        for e, r, i, has_ae, opt in test_models:

            ae_label = "DXY" if has_ae else "D0"
            bem_suffix = DEFAULT_BEM_SUFFIX if needs_bem_suffix(r) else None
            model_base_name = format_model_name(e, r, i, ae_label, opt, pct, bem_suffix)

            print(f"\n\n{'#'*80}")
            print(f" PIPELINE : {model_base_name}")
            print(f"{'#'*80}")

            n_trials_current = TRIALS_GV if e == 'GV' else TRIALS_GM

            t0 = time.perf_counter()
            optimize(df_train_pct, entree=e, residuelle=r, inter=i, has_ae=has_ae, option=opt, model_base_name=model_base_name, n_trials=n_trials_current, bem_suffix=bem_suffix)
            print(f"   [CHRONO] optimize {model_base_name} : {time.perf_counter()-t0:.1f}s")

            t0 = time.perf_counter()
            evaluator(df_train_pct, df_test, entree=e, residuelle=r, inter=i, has_ae=has_ae, option=opt, baseline_scores=baseline_scores, pct=pct, bem_suffix=bem_suffix)
            print(f"   [CHRONO] evaluator {model_base_name} : {time.perf_counter()-t0:.1f}s")

if __name__ == "__main__":
    main()
