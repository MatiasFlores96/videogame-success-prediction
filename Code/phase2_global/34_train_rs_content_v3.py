"""
34_train_rs_content_v3.py
==========================
Entrena el Two-Tower v3 (fixes metodológicos de tower_v3.py) y evalúa downstream.

Dos variantes:
  34_05_v3   — uniform negatives sobre pre-2016 (fixes 1+2+3)
  34_05_v3w  — negativos ponderados por frecuencia^0.75 (word2vec-style)

Evaluación downstream: CatBoost sobre emb (64d) + meta5 (5d, sin has_senti),
comparable contra 24_05_cat_clean (la versión sin leakage del resultado principal).

Outputs:
  Data/item_embeddings_rs_v2_content3.npy / _cs.npy    (v3 uniform)
  Data/item_embeddings_rs_v2_content3w.npy / _cs.npy   (v3 weighted)
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tower_v3 import train_tower, DATA
from v2_data import load_all, run_model

print("=" * 70)
print("34_train_rs_content_v3.py — Two-Tower v3 (fixes) + eval downstream")
print("=" * 70)

ctx   = load_all()
meta5 = ctx["meta_aligned"][:, :5]

results = {}
for tag, neg_mode, fname in [
    ("34_05_v3",  "uniform_pre2016", "item_embeddings_rs_v2_content3"),
    ("34_05_v3w", "pop075",          "item_embeddings_rs_v2_content3w"),
]:
    print(f"\n{'#'*70}\n# Entrenando tower: {tag} (neg_mode={neg_mode})\n{'#'*70}")
    out = train_tower(cs_drop=0.3, seed=42, neg_mode=neg_mode)
    np.save(os.path.join(DATA, f"{fname}.npy"),    out["emb_std"])
    np.save(os.path.join(DATA, f"{fname}_cs.npy"), out["emb_cs"])
    print(f"  [OK] {fname}.npy | best_val={out['best_val']:.4f} | epochs={out['epochs_run']}")

    X = np.hstack([out["emb_std"], meta5])
    m = run_model(ctx, X, tag,
                  f"CatBoost RS content-aug v3 ({neg_mode}) + Meta5",
                  f"RS v3 {neg_mode} (64d) + meta5 (5d)",
                  notes_extra=f"tower_v3 fixes: tfidf pre-2016, val con negativos, "
                              f"neg={neg_mode}. best_val={out['best_val']:.4f}")
    results[tag] = m

print("\n" + "=" * 70)
print("RESUMEN — v3 vs baseline v2")
print("=" * 70)
for tag, m in results.items():
    print(f"  {tag:<12} R2={m['r2_temporal']:.4f} | RMSE={m['rmse_temporal']:.2f} | "
          f"Spearman={m['spearman_temporal']:.4f}")
print("  Referencia: 24_05_cat (v2 + has_senti) R2=0.4067 | RMSE=1014.86")
print("  Referencia: 24_05_cat_clean → ver experiment_results.json (corrida 32_)")
