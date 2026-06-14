"""
32_audit_has_senti.py
======================
Auditoría de leakage del feature `has_senti` en el modelo principal (24_05_cat).

Contexto: el campo `sentiment` de steam_games.json es un snapshot (~2018) que
incluye valores como "1 user reviews", "2 user reviews"... — codifica el conteo
de reviews acumuladas. `has_senti` (sentiment no nulo) dice "este juego acumuló
reviews al momento del scrape", que correlaciona con el target. Para juegos del
test set (2017+) eso es información posterior al cutoff.

Pasos:
  1. Reproduce 24_05_cat (emb 64d + meta 6d, CatBoost, seed 42)
  2. SHAP nativo de CatBoost sobre el test set → ranking de has_senti
  3. Re-run sin has_senti (meta5) → "24_05_cat_clean"
  4. Delta R² = costo real del leakage

También audita num_tags / num_specs / has_senti por separado (tags de Steam
también se acumulan con el tiempo, aunque de forma menos directa).
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_data import load_all, run_model

print("=" * 70)
print("32_audit_has_senti.py — Auditoría de leakage en metadata")
print("=" * 70)

ctx = load_all()
item_emb     = ctx["item_emb"]
meta_aligned = ctx["meta_aligned"]
test_mask    = ctx["test_mask"]
y_full       = ctx["y_full"]

FEAT_NAMES = [f"emb_{i:02d}" for i in range(64)] + \
             ["price", "ea_flag", "num_genres", "num_tags", "num_specs", "has_senti"]

# ── 1. Reproducir 24_05_cat y extraer SHAP ─────────────────────────────────────
X6 = np.hstack([item_emb, meta_aligned])  # 70d, idéntico a 24_05_cat

metrics6, model, params, (X_tr, y_tr, X_te, y_te, pred) = run_model(
    ctx, X6, "32_repro_24_05", "Repro 24_05_cat (con has_senti)",
    "RS content-aug v2 (64d) + meta (6d)",
    save=False, return_model=True)

print("\n[SHAP] Calculando ShapValues nativos de CatBoost sobre el test set...")
from catboost import Pool
shap_vals = model.get_feature_importance(Pool(X_te, y_te), type="ShapValues")[:, :-1]
mean_abs  = np.abs(shap_vals).mean(axis=0)
order     = np.argsort(mean_abs)[::-1]

print(f"\n{'Rank':<5} {'Feature':<14} {'SHAP mean|abs|':>14}")
print("-" * 36)
for rank, fi in enumerate(order[:15], 1):
    print(f"{rank:<5} {FEAT_NAMES[fi]:<14} {mean_abs[fi]:>14.3f}")

for fname in ["has_senti", "num_tags", "num_specs"]:
    fi   = FEAT_NAMES.index(fname)
    rank = int(np.where(order == fi)[0][0]) + 1
    pct  = mean_abs[fi] / mean_abs.sum() * 100
    print(f"\n  {fname}: rank {rank}/70 | SHAP={mean_abs[fi]:.3f} ({pct:.1f}% del total)")

# Correlación directa has_senti vs target en test
hs_test = meta_aligned[test_mask, 5]
print(f"\n  has_senti en test: {hs_test.mean()*100:.1f}% de items con sentiment")
print(f"  Media reviews | has_senti=1: {y_full[test_mask][hs_test == 1].mean():.1f}")
print(f"  Media reviews | has_senti=0: {y_full[test_mask][hs_test == 0].mean():.1f}")

# ── 2. Re-run sin has_senti ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Re-run SIN has_senti (meta5)")
print("=" * 70)

meta5 = meta_aligned[:, :5]
X5 = np.hstack([item_emb, meta5])  # 69d

metrics5 = run_model(
    ctx, X5, "24_05_cat_clean", "CatBoost RS content-aug v2 + Meta5 (sin has_senti)",
    "RS content-aug v2 (64d) + meta sin has_senti (5d)",
    notes_extra="Auditoría 32_: has_senti excluido por leakage (sentiment snapshot 2018).")

# ── 3. Veredicto ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("VEREDICTO")
print("=" * 70)
d_r2   = metrics6["r2_temporal"] - metrics5["r2_temporal"]
d_rmse = metrics5["rmse_temporal"] - metrics6["rmse_temporal"]
print(f"  Con has_senti (70d):  R2={metrics6['r2_temporal']:.4f} | RMSE={metrics6['rmse_temporal']:.2f}")
print(f"  Sin has_senti (69d):  R2={metrics5['r2_temporal']:.4f} | RMSE={metrics5['rmse_temporal']:.2f}")
print(f"  Delta (costo del leakage): R2 {d_r2:+.4f} | RMSE {d_rmse:+.2f}")
if abs(d_r2) < 0.01:
    print("  → has_senti es despreciable: el resultado principal NO depende del leakage.")
else:
    print("  → has_senti contribuía de forma medible: usar 24_05_cat_clean como resultado principal.")
