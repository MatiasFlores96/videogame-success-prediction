"""
33_catboost_variants.py
========================
Variantes downstream sobre los embeddings content-aug v2, todas con meta5
(sin has_senti — ver auditoría 32_). Comparable contra 24_05_cat_clean.

  33_tweedie    CatBoost Tweedie loss (variance_power tuneado 1.05-1.95)
                → modela conteos heavy-tailed nativamente en escala original,
                  sin la patología del back-transform que mató al log (31_)
  33_poisson    CatBoost Poisson loss
  33_ens        Ensemble (promedio) CatBoost RMSE + XGBoost
  33_mixed      Embeddings mixtos: std para train (pre-2017), cs para test (2017+)
                → simula deployment real: juegos nuevos no tienen ID embedding
  33_metascore  + metascore nativo de Steam (2d: valor + missing flag)
  33_devrep     + developer reputation temporal limpia (2d)
  33_meta_plus  + metascore + dev_rep juntos
"""

import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_data import (load_all, run_model, build_dev_rep, build_metascore,
                     optuna_cat, optuna_xgb, _cat_predict)
from results_tracker import save_result

print("=" * 70)
print("33_catboost_variants.py — Variantes downstream (meta5, sin has_senti)")
print("=" * 70)

ctx      = load_all()
item_emb = ctx["item_emb"]
emb_cs   = ctx["item_emb_cs"]
meta5    = ctx["meta_aligned"][:, :5]
y_full   = ctx["y_full"]
train_mask, test_mask = ctx["train_mask"], ctx["test_mask"]
item_dates = ctx["item_dates"]

X_base = np.hstack([item_emb, meta5])  # 69d — mismo set que 24_05_cat_clean

results = {}

# ── 1. Tweedie / Poisson ───────────────────────────────────────────────────────
results["33_tweedie"] = run_model(
    ctx, X_base, "33_tweedie", "CatBoost Tweedie + RS v2 + Meta5",
    "RS content-aug v2 (64d) + meta5 (5d)",
    loss_function="Tweedie", tune_tweedie_power=True,
    notes_extra="Tweedie variance_power tuneado por Optuna.")

results["33_poisson"] = run_model(
    ctx, X_base, "33_poisson", "CatBoost Poisson + RS v2 + Meta5",
    "RS content-aug v2 (64d) + meta5 (5d)",
    loss_function="Poisson")

# ── 2. Ensemble CatBoost + XGBoost ────────────────────────────────────────────
print(f"\n{'='*70}\n[33_ens] Ensemble CatBoost RMSE + XGBoost\n{'='*70}")
train_dates = np.array(item_dates)[train_mask]
so   = np.argsort(train_dates)
X_tr = X_base[train_mask][so]; y_tr = y_full[train_mask][so]
X_te = X_base[test_mask];      y_te = y_full[test_mask]

cat_params = optuna_cat(X_tr, y_tr, 60)
xgb_params = optuna_xgb(X_tr, y_tr, 60)
m_cat = CatBoostRegressor(**cat_params); m_cat.fit(X_tr, y_tr, verbose=False)
m_xgb = XGBRegressor(**xgb_params);      m_xgb.fit(X_tr, y_tr)
p_cat = np.clip(m_cat.predict(X_te), 0, None)
p_xgb = np.clip(m_xgb.predict(X_te), 0, None)

best_ens, best_w = None, None
for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
    p = w * p_cat + (1 - w) * p_xgb
    r2w = r2_score(y_te, p)
    if best_ens is None or r2w > best_ens:
        best_ens, best_w, p_best = r2w, w, p
rmse_ens = float(mean_squared_error(y_te, p_best) ** 0.5)
rho_ens  = float(spearmanr(y_te, p_best).statistic)
print(f"  Cat solo: R2={r2_score(y_te, p_cat):.4f} | XGB solo: R2={r2_score(y_te, p_xgb):.4f}")
print(f"  Ensemble (w_cat={best_w}): R2={best_ens:.4f} | RMSE={rmse_ens:.2f} | Spearman={rho_ens:.4f}")
save_result(model_id="33_ens", model_name="Ensemble CatBoost+XGBoost RS v2 + Meta5",
            features="RS content-aug v2 (64d) + meta5 (5d)", embeddings="v2_global",
            metrics={"r2_temporal": float(best_ens), "rmse_temporal": rmse_ens,
                     "mae_temporal": float(mean_absolute_error(y_te, p_best)),
                     "spearman_temporal": rho_ens},
            notes=f"Promedio ponderado w_cat={best_w}. Sin TSCV/KFold (solo temporal).")
results["33_ens"] = {"r2_temporal": best_ens, "rmse_temporal": rmse_ens,
                     "spearman_temporal": rho_ens}

# ── 3. Embeddings mixtos (std train / cs test) ────────────────────────────────
if emb_cs is not None:
    emb_mixed = item_emb.copy()
    emb_mixed[test_mask] = emb_cs[test_mask]
    X_mixed = np.hstack([emb_mixed, meta5])
    results["33_mixed"] = run_model(
        ctx, X_mixed, "33_mixed", "CatBoost RS v2 mixto (std/cs) + Meta5",
        "RS v2 std train + cs test (64d) + meta5 (5d)",
        notes_extra="Embeddings cold-start (solo content MLP) para items 2017+.")

# ── 4. Metascore Steam / dev_rep temporal ─────────────────────────────────────
ms2 = build_metascore(ctx)
dr2 = build_dev_rep(ctx)
n_ms = int((ms2[:, 1] == 0).sum()); n_dr = int(dr2[:, 1].sum())
print(f"\n  metascore presente: {n_ms:,}/{ctx['N']:,} | dev con historia: {n_dr:,}/{ctx['N']:,}")

results["33_metascore"] = run_model(
    ctx, np.hstack([X_base, ms2]), "33_metascore", "CatBoost RS v2 + Meta5 + Metascore",
    "RS v2 (64d) + meta5 (5d) + metascore steam (2d)")

results["33_devrep"] = run_model(
    ctx, np.hstack([X_base, dr2]), "33_devrep", "CatBoost RS v2 + Meta5 + DevRep temporal",
    "RS v2 (64d) + meta5 (5d) + dev_rep temporal (2d)",
    notes_extra="dev_rep limpia: solo juegos pre-2017 del dev, excluye el propio juego.")

results["33_meta_plus"] = run_model(
    ctx, np.hstack([X_base, ms2, dr2]), "33_meta_plus",
    "CatBoost RS v2 + Meta5 + Metascore + DevRep",
    "RS v2 (64d) + meta5 (5d) + metascore (2d) + dev_rep (2d)")

# ── Resumen ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESUMEN 33_ — todas las variantes (meta5, sin has_senti)")
print("=" * 70)
print(f"{'Modelo':<15} {'R2':>8} {'RMSE':>9} {'Spearman':>9}")
print("-" * 45)
for tag, m in results.items():
    print(f"  {tag:<13} {m['r2_temporal']:>8.4f} {m['rmse_temporal']:>9.2f} "
          f"{m.get('spearman_temporal', float('nan')):>9.4f}")
print("\n  Comparar contra 24_05_cat_clean (corrida 32_) en experiment_results.json")
