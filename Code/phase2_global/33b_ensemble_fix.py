"""
33b_ensemble_fix.py
====================
Dos correcciones/extensiones de 33_:

1. 33_ens elegía el peso del ensemble maximizando R2 EN TEST (leak de selección).
   Acá el peso se elige sobre validación temporal (último 20% del train) y solo
   después se evalúa en test. → 33b_ens

2. 33_mixed (std train / cs test) mide mismatch de distribución, no deployment.
   La variante deployment-honesta usa embeddings cold-start (solo content MLP)
   para TODOS los items, train y test. → 33b_all_cs
"""

import os, sys
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_data import load_all, run_model, optuna_cat, optuna_xgb
from results_tracker import save_result

print("=" * 70)
print("33b_ensemble_fix.py — Ensemble sin leak de peso + all-cs deployment")
print("=" * 70)

ctx = load_all()
item_emb, emb_cs = ctx["item_emb"], ctx["item_emb_cs"]
meta5  = ctx["meta_aligned"][:, :5]
y_full = ctx["y_full"]
item_dates = ctx["item_dates"]
train_mask, test_mask = ctx["train_mask"], ctx["test_mask"]

X_base = np.hstack([item_emb, meta5])

train_dates = np.array(item_dates)[train_mask]
so   = np.argsort(train_dates)
X_tr = X_base[train_mask][so]; y_tr = y_full[train_mask][so]
X_te = X_base[test_mask];      y_te = y_full[test_mask]

# ── 1. Ensemble con peso elegido en validación temporal ───────────────────────
print("\n[1] Ensemble: peso elegido en val temporal (ultimo 20% del train)...")
cut = int(len(X_tr) * 0.8)
X_fit, X_val = X_tr[:cut], X_tr[cut:]
y_fit, y_val = y_tr[:cut], y_tr[cut:]

cat_params = optuna_cat(X_tr, y_tr, 60)
xgb_params = optuna_xgb(X_tr, y_tr, 60)

# Peso óptimo en validación (modelos fiteados solo en el 80% inicial)
mc_v = CatBoostRegressor(**cat_params); mc_v.fit(X_fit, y_fit, verbose=False)
mx_v = XGBRegressor(**xgb_params);      mx_v.fit(X_fit, y_fit)
pc_v = np.clip(mc_v.predict(X_val), 0, None)
px_v = np.clip(mx_v.predict(X_val), 0, None)
ws = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
rmse_val = {w: float(mean_squared_error(y_val, w * pc_v + (1 - w) * px_v) ** 0.5) for w in ws}
w_star = min(rmse_val, key=rmse_val.get)
print(f"  RMSE val por peso: " + " ".join(f"{w:.1f}:{rmse_val[w]:.0f}" for w in ws))
print(f"  w_cat* = {w_star} (elegido en val, sin mirar test)")

# Refit en todo el train, evaluar en test con w_star
mc = CatBoostRegressor(**cat_params); mc.fit(X_tr, y_tr, verbose=False)
mx = XGBRegressor(**xgb_params);      mx.fit(X_tr, y_tr)
p_ens = w_star * np.clip(mc.predict(X_te), 0, None) + \
        (1 - w_star) * np.clip(mx.predict(X_te), 0, None)

r2   = r2_score(y_te, p_ens)
rmse = float(mean_squared_error(y_te, p_ens) ** 0.5)
rho  = float(spearmanr(y_te, p_ens).statistic)
print(f"  33b_ens: R2={r2:.4f} | RMSE={rmse:.2f} | Spearman={rho:.4f}")
save_result(model_id="33b_ens", model_name="Ensemble Cat+XGB (peso en val) + Meta5",
            features="RS content-aug v2 (64d) + meta5 (5d)", embeddings="v2_global",
            metrics={"r2_temporal": float(r2), "rmse_temporal": rmse,
                     "mae_temporal": float(mean_absolute_error(y_te, p_ens)),
                     "spearman_temporal": rho},
            notes=f"w_cat={w_star} elegido en val temporal interna. Fix del leak de seleccion de 33_ens.")

# ── 2. All-cs: deployment honesto ─────────────────────────────────────────────
if emb_cs is not None:
    print("\n[2] All-cs: embeddings cold-start (solo content MLP) para TODOS los items...")
    X_cs = np.hstack([emb_cs, meta5])
    run_model(ctx, X_cs, "33b_all_cs", "CatBoost RS v2 all-cold-start + Meta5",
              "RS v2 cs-only (64d) + meta5 (5d)",
              notes_extra="Embeddings content-MLP-only para train y test (deployment-honesto, "
                          "sin mismatch std/cs).")
