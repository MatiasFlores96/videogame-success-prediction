"""
39_baseline_errorbars.py
=========================
Barras de error para la comparación final RS vs contenido-puro.

El multi-seed (36_) puso barras al RS variando el seed del TOWER. Pero el
baseline de contenido (abl_04=0.314) se evaluó UNA vez. Acá se reps ambos
variando el seed de tuning (Optuna TPE + CatBoost random_seed), 5 reps c/u:

  A) Contenido-puro: meta5 (5d) + RAWG (11d)        [≈ abl_04 sin has_senti]
  B) RS fijo: embeddings v3 seed-42 + meta5          [varianza solo downstream]

Esto descompone la varianza total del RS (36_: tower+downstream = ±0.017) en
componente downstream (B) y permite comparar A vs 36_v3 con bandas en ambos lados.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_data import load_all, run_model
from results_tracker import save_result

print("=" * 70)
print("39_baseline_errorbars.py — Barras de error baseline + downstream")
print("=" * 70)

ctx    = load_all()
meta5  = ctx["meta_aligned"][:, :5]
rawg   = ctx["rawg_aligned"]

# Embeddings v3 seed 42 (guardados por 34_)
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Data")
emb_v3 = np.load(os.path.join(DATA, "item_embeddings_rs_v2_content3.npy"))

X_content = np.hstack([meta5, rawg])      # 16d, contenido-puro
X_rs      = np.hstack([emb_v3, meta5])    # 69d, RS fijo

SEEDS = [42, 123, 777, 2024, 31337]
res = {"content": [], "rs_fixed": []}

for seed in SEEDS:
    print(f"\n{'#'*70}\n# TUNING SEED = {seed}\n{'#'*70}")
    m_c = run_model(ctx, X_content, f"39_content_s{seed}",
                    f"Contenido-puro meta5+RAWG (tuning seed {seed})",
                    "meta5 (5d) + rawg (11d)", tuning_seed=seed,
                    notes_extra="Barras de error del baseline sin RS.")
    res["content"].append(m_c)

    m_r = run_model(ctx, X_rs, f"39_rsfix_s{seed}",
                    f"RS v3-s42 fijo + meta5 (tuning seed {seed})",
                    "RS v3 seed42 (64d) + meta5 (5d)", tuning_seed=seed,
                    notes_extra="Varianza downstream-only (embeddings fijos).")
    res["rs_fixed"].append(m_r)

print("\n" + "=" * 70)
print("RESUMEN 39_ — distribuciones por fuente de varianza")
print("=" * 70)
for name, ms in res.items():
    r2s  = [m["r2_temporal"] for m in ms]
    rhos = [m["spearman_temporal"] for m in ms]
    tscv = [m["r2_tscv"] for m in ms]
    print(f"\n  [{name}]")
    for seed, m in zip(SEEDS, ms):
        print(f"    seed={seed:<6} R2={m['r2_temporal']:.4f} | "
              f"Spearman={m['spearman_temporal']:.4f} | TSCV={m['r2_tscv']:.4f}")
    print(f"    R2:       {np.mean(r2s):.4f} ± {np.std(r2s, ddof=1):.4f} "
          f"[{min(r2s):.4f}, {max(r2s):.4f}]")
    print(f"    Spearman: {np.mean(rhos):.4f} ± {np.std(rhos, ddof=1):.4f}")
    print(f"    TSCV:     {np.mean(tscv):.4f} ± {np.std(tscv, ddof=1):.4f}")
    save_result(model_id=f"39_{name}_summary",
                model_name=f"Barras de error {name} (5 tuning seeds)",
                features="meta5+rawg (16d)" if name == "content" else "RS v3 s42 + meta5 (69d)",
                embeddings="" if name == "content" else "v2_global",
                metrics={"r2_temporal": float(np.mean(r2s)),
                         "r2_temporal_std": float(np.std(r2s, ddof=1)),
                         "spearman_temporal": float(np.mean(rhos)),
                         "r2_tscv": float(np.mean(tscv))},
                notes=f"5 reps tuning seeds {SEEDS}. min={min(r2s):.4f} max={max(r2s):.4f}")

print("\n  Comparación final: [content] vs 36_v3 (0.2193±0.0172 tower+downstream)")
