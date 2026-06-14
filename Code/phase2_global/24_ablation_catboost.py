"""
24_ablation_catboost.py
========================
Ablation study: ¿cuánto aporta el RS?

Modelos SIN embeddings RS (baselines de contenido puro):
  abl_01  Metadata Only                        meta (6d)
  abl_02  TF-IDF + Numeric                     tfidf (100d) + steam (2d)
  abl_03  TF-IDF + Numeric + RAWG              tfidf (100d) + steam (2d) + rawg (11d)
  abl_04  Meta + RAWG                          meta (6d) + rawg (11d)
  abl_05  Full content sin RS                  tfidf (100d) + steam (2d) + meta (6d) + rawg (11d)

Referencia (con RS embeddings v2):
  24_05_cat_clean — CatBoost RS content-aug v2 + meta5  →  ver script 32_
  34_05_v3        — CatBoost RS v3 (fixes metodológicos) →  ver script 34_

Nota de reproducibilidad:
  Todos los modelos usan CatBoost + Optuna (60 trials, tuning_seed=42).
  La evaluación incluye R² temporal, Spearman, TSCV (7 ventanas) y KFold-5.
  Los resultados se guardan automáticamente en Data/experiment_results.json.

Historia:
  Versión original (pre 2026-06) reimplementaba load_all, TF-IDF, Optuna y run_model
  localmente, usaba XGBoost para ablaciones, y no computaba Spearman.
  Esta versión unifica con v2_data.py para consistencia metodológica.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_data import load_all, run_model

print("=" * 70)
print("24_ablation_catboost.py — Ablation: contenido puro vs RS embeddings")
print("=" * 70)

ctx   = load_all()
meta  = ctx["meta_aligned"]               # 6d: price, ea_flag, n_genres, n_tags, n_specs, has_senti
tfidf = ctx["tfidf_aligned"]              # 100d: TF-IDF tags+géneros (fit pre-2017)
steam = ctx["steam_aligned"]              # 2d: price_num, ea_flag (numéricas)
rawg  = ctx["rawg_aligned"]               # 11d: RAWG quality features

print(f"\n  Train: {ctx['train_mask'].sum():,} | Test: {ctx['test_mask'].sum():,}")
print(f"  meta={meta.shape} | tfidf={tfidf.shape} | steam={steam.shape} | rawg={rawg.shape}")

# ── A) Ablation — modelos sin RS embeddings ────────────────────────────────────
print("\n" + "=" * 70)
print("A) ABLATION — modelos sin RS embeddings")
print("=" * 70)

run_model(ctx, meta,
          "abl_01", "Metadata Only (ablation)",
          "meta (6d)")

run_model(ctx, np.hstack([tfidf, steam]),
          "abl_02", "TF-IDF + Numeric (ablation)",
          "tfidf (100d) + steam (2d)")

run_model(ctx, np.hstack([tfidf, steam, rawg]),
          "abl_03", "TF-IDF + Numeric + RAWG (ablation)",
          "tfidf (100d) + steam (2d) + rawg (11d)")

run_model(ctx, np.hstack([meta, rawg]),
          "abl_04", "Meta + RAWG (ablation)",
          "meta (6d) + rawg (11d)")

run_model(ctx, np.hstack([tfidf, steam, meta, rawg]),
          "abl_05", "Full content sin RS (ablation)",
          "tfidf (100d) + steam (2d) + meta (6d) + rawg (11d)")

print("\n" + "=" * 70)
print("ABLATION COMPLETADO")
print("=" * 70)
print("Comparar contra modelos con RS en experiment_results.json:")
print("  24_05_cat_clean — CatBoost RS v2 + meta5    (ver script 32_)")
print("  34_05_v3        — CatBoost RS v3 + meta5    (ver script 34_)")
print("  39_content_summary  — content 5 seeds       (ver script 39_)")
print("  39_rs_fixed_summary — RS v3 5 seeds         (ver script 39_)")
