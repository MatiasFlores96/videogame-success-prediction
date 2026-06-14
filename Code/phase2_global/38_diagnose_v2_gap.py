"""
38_diagnose_v2_gap.py
======================
Diagnóstico del gap entre el resultado original (24_05_cat_clean=0.340 sobre
los embeddings guardados de 23_) y la réplica multi-seed (36_v2rep=0.237±0.022).

El tower de 36_v2rep convergió casi idéntico al original (best_val 0.3260 vs
0.3261), pero su content matrix usa fit pre-2016 (FIX 1). El 23_ original
fiteaba TF-IDF y z-score sobre TODOS los items (incluidos los 4,798 de test):
fit transductivo.

Hipótesis A: el fit transductivo da ventaja sistemática (~+0.10 R²) → el 0.340
             dependía de ese soft-leakage.
Hipótesis B: el embedding original fue una realización afortunada.

Test: entrenar "v2exact" (content matrix con fit_scope='all' + uniform_all +
val legacy) con 2 seeds. Si R²≈0.32-0.34 → Hipótesis A. Si R²≈0.24 → Hipótesis B.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tower_v3 import train_tower, build_content_matrix
from v2_data import load_all, run_model

print("=" * 70)
print("38_diagnose_v2_gap.py — ¿Fit transductivo o suerte de realización?")
print("=" * 70)

ctx   = load_all()
meta5 = ctx["meta_aligned"][:, :5]

content_all = build_content_matrix(fit_scope="all")

results = {}
for seed in [42, 123]:
    tag = f"38_v2exact_s{seed}"
    print(f"\n{'#'*70}\n# v2exact (fit transductivo) SEED={seed}\n{'#'*70}")
    out = train_tower(cs_drop=0.3, seed=seed, neg_mode="uniform_all",
                      val_mode="legacy", content_matrix=content_all)
    m = run_model(ctx, np.hstack([out["emb_std"], meta5]), tag,
                  f"CatBoost RS v2exact (fit transductivo) seed={seed} + Meta5",
                  f"RS v2exact seed{seed} (64d) + meta5 (5d)",
                  notes_extra=f"38_: TF-IDF y z-score fiteados sobre TODOS los items "
                              f"(replica exacta de 23_). best_val={out['best_val']:.4f}")
    results[seed] = m

print("\n" + "=" * 70)
print("VEREDICTO 38_")
print("=" * 70)
for seed, m in results.items():
    print(f"  v2exact seed={seed}: R2={m['r2_temporal']:.4f} | "
          f"RMSE={m['rmse_temporal']:.2f} | Spearman={m['spearman_temporal']:.4f}")
print("\n  Referencias:")
print("    24_05_cat_clean (emb originales 23_):  R2=0.3404")
print("    36_v2rep (fit pre-2016, 5 seeds):      R2=0.2365 ± 0.0216")
print("  Si v2exact ~0.32+: el gap es el fit transductivo (soft-leakage de TF-IDF/z-score).")
print("  Si v2exact ~0.24:  el embedding original fue una realizacion afortunada.")
