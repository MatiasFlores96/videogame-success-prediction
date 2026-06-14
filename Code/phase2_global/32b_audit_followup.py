"""
32b_audit_followup.py
======================
Follow-up de la auditoría 32_. Dos preguntas que dejó abiertas:

1. PARAMS TRANSFER — 32_ mostró has_senti con SHAP=0.000 (nunca usado) pero
   el re-run sin la columna dio R2 0.407→0.340. Si el modelo nunca splitteo
   en has_senti, fitear los MISMOS hiperparámetros sobre 69d debe dar
   predicciones idénticas. Si da igual → la caída era varianza de Optuna
   (TPE con 69 features explora otra trayectoria), no leakage causal.

2. SOFT-LEAKAGE DE num_tags — el feature #1 del modelo (SHAP=159.9, 12.8%).
   Los tags de Steam son user-generated y se ACUMULAN: un juego de 2017 que
   se hizo popular tiene más tags en el snapshot 2018. Ablaciones:
     24_05_cat_no_nt   emb + [price, ea_flag, num_genres, num_specs]  (4d)
     24_05_cat_strict  emb + [price, ea_flag, num_genres]             (3d)
"""

import os, sys
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_data import load_all, run_model, optuna_cat

print("=" * 70)
print("32b_audit_followup.py — Params transfer + soft-leakage num_tags")
print("=" * 70)

ctx = load_all()
item_emb     = ctx["item_emb"]
meta_aligned = ctx["meta_aligned"]
y_full       = ctx["y_full"]
item_dates   = ctx["item_dates"]
train_mask, test_mask = ctx["train_mask"], ctx["test_mask"]

X6 = np.hstack([item_emb, meta_aligned])         # 70d (con has_senti)
X5 = np.hstack([item_emb, meta_aligned[:, :5]])  # 69d (sin has_senti)

train_dates = np.array(item_dates)[train_mask]
so   = np.argsort(train_dates)
y_tr = y_full[train_mask][so]
y_te = y_full[test_mask]

# ── 1. Params transfer ─────────────────────────────────────────────────────────
print("\n[1] Re-derivando best params del modelo 70d (Optuna seed 42, determinista)...")
params70 = optuna_cat(X6[train_mask][so], y_tr, 60)

m70 = CatBoostRegressor(**params70); m70.fit(X6[train_mask][so], y_tr, verbose=False)
m69 = CatBoostRegressor(**params70); m69.fit(X5[train_mask][so], y_tr, verbose=False)
p70 = np.clip(m70.predict(X6[test_mask]), 0, None)
p69 = np.clip(m69.predict(X5[test_mask]), 0, None)

r2_70, r2_69 = r2_score(y_te, p70), r2_score(y_te, p69)
max_diff = float(np.max(np.abs(p70 - p69)))
print(f"  70d (mismos params): R2={r2_70:.4f}")
print(f"  69d (mismos params): R2={r2_69:.4f}")
print(f"  Max |pred70 - pred69| = {max_diff:.6f}")
if max_diff < 1e-3:
    print("  >> Predicciones IDENTICAS: has_senti jamas se uso. La caida del")
    print("     re-tuning en 32_ era varianza de Optuna, NO leakage causal.")
else:
    print("  >> Predicciones difieren: has_senti si influia en los arboles.")

# ── 2. Ablaciones de num_tags / num_specs ──────────────────────────────────────
print("\n[2] Ablaciones de soft-leakage (tags acumulativos, snapshot 2018)...")
meta_no_nt  = meta_aligned[:, [0, 1, 2, 4]]   # price, ea, num_genres, num_specs
meta_strict = meta_aligned[:, [0, 1, 2]]      # price, ea, num_genres

m_no_nt = run_model(ctx, np.hstack([item_emb, meta_no_nt]), "24_05_cat_no_nt",
                    "CatBoost RS v2 + meta sin num_tags (4d)",
                    "RS v2 (64d) + [price, ea, num_genres, num_specs]",
                    notes_extra="Ablacion 32b: num_tags excluido (soft-leakage, tags acumulativos).")

m_strict = run_model(ctx, np.hstack([item_emb, meta_strict]), "24_05_cat_strict",
                     "CatBoost RS v2 + meta estricta (3d)",
                     "RS v2 (64d) + [price, ea, num_genres]",
                     notes_extra="Ablacion 32b: sin num_tags/num_specs/has_senti (todos los canales snapshot).")

print("\n" + "=" * 70)
print("RESUMEN 32b")
print("=" * 70)
print(f"  70d mismos params:   R2={r2_70:.4f}   (referencia 24_05_cat=0.4067)")
print(f"  69d mismos params:   R2={r2_69:.4f}   (max diff pred: {max_diff:.6f})")
print(f"  sin num_tags (4d):   R2={m_no_nt['r2_temporal']:.4f} | RMSE={m_no_nt['rmse_temporal']:.2f}")
print(f"  meta estricta (3d):  R2={m_strict['r2_temporal']:.4f} | RMSE={m_strict['rmse_temporal']:.2f}")
