"""
36_multiseed_stability.py
==========================
Estabilidad multi-seed del resultado principal — EL experimento crítico tras
descubrir que la varianza estocástica (tower seed + Optuna/CatBoost RNG) es
~±0.05 R² y domina las comparaciones finas del proyecto.

Dos configuraciones × 5 seeds:

  v2rep — réplica del setup de 23_/24_ (el resultado publicado 0.407/0.340):
          neg_mode=uniform_all, val_mode=legacy (BCE solo positivos).
          Hallazgo de 34_: ese criterio "degenerado" selecciona checkpoints
          cuyos embeddings codifican popularidad global → mejor temporal.
  v3    — fixes metodológicos completos:
          neg_mode=uniform_pre2016, val_mode=with_negs.

Salida clave para la tesis: media ± std del R² temporal por config →
¿el gap RS vs contenido-puro sobrevive a las barras de error?
Bonus por config: emb_avg (promedio de embeddings de 5 seeds).
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tower_v3 import train_tower
from v2_data import load_all, run_model
from results_tracker import save_result

print("=" * 70)
print("36_multiseed_stability.py — Multi-seed, configs v2rep y v3")
print("=" * 70)

ctx   = load_all()
meta5 = ctx["meta_aligned"][:, :5]

SEEDS = [42, 123, 777, 2024, 31337]
CONFIGS = [
    ("v2rep", dict(neg_mode="uniform_all",     val_mode="legacy")),
    ("v3",    dict(neg_mode="uniform_pre2016", val_mode="with_negs")),
]

summary = {}
for cfg_name, kw in CONFIGS:
    r2s, rmses, rhos, tscvs, embs = [], [], [], [], []
    for seed in SEEDS:
        tag = f"36_{cfg_name}_s{seed}"
        print(f"\n{'#'*70}\n# CONFIG={cfg_name} SEED={seed}\n{'#'*70}")
        out = train_tower(cs_drop=0.3, seed=seed, **kw)
        embs.append(out["emb_std"])
        m = run_model(ctx, np.hstack([out["emb_std"], meta5]), tag,
                      f"CatBoost RS {cfg_name} seed={seed} + Meta5",
                      f"RS {cfg_name} seed{seed} (64d) + meta5 (5d)",
                      notes_extra=f"Multi-seed {cfg_name}. best_val={out['best_val']:.4f} "
                                  f"epochs={out['epochs_run']}")
        r2s.append(m["r2_temporal"]); rmses.append(m["rmse_temporal"])
        rhos.append(m["spearman_temporal"]); tscvs.append(m["r2_tscv"])

    print(f"\n{'#'*70}\n# CONFIG={cfg_name}: embedding average (5 seeds)\n{'#'*70}")
    emb_avg = np.mean(embs, axis=0).astype(np.float32)
    m_avg = run_model(ctx, np.hstack([emb_avg, meta5]), f"36_{cfg_name}_embavg",
                      f"CatBoost RS {cfg_name} emb-avg 5 seeds + Meta5",
                      f"RS {cfg_name} promedio 5 seeds (64d) + meta5 (5d)")

    summary[cfg_name] = (r2s, rmses, rhos, tscvs, m_avg)
    save_result(model_id=f"36_{cfg_name}_summary",
                model_name=f"Multi-seed {cfg_name} (5 seeds) resumen",
                features="RS (64d) + meta5 (5d)", embeddings="v2_global",
                metrics={"r2_temporal": float(np.mean(r2s)),
                         "r2_temporal_std": float(np.std(r2s, ddof=1)),
                         "rmse_temporal": float(np.mean(rmses)),
                         "spearman_temporal": float(np.mean(rhos)),
                         "r2_tscv": float(np.mean(tscvs))},
                notes=f"Media seeds {SEEDS}. min={min(r2s):.4f} max={max(r2s):.4f}. "
                      f"emb_avg R2={m_avg['r2_temporal']:.4f}")

print("\n" + "=" * 70)
print("RESUMEN FINAL — multi-seed por configuración")
print("=" * 70)
for cfg_name, (r2s, rmses, rhos, tscvs, m_avg) in summary.items():
    print(f"\n  [{cfg_name}]")
    for seed, r2, rho in zip(SEEDS, r2s, rhos):
        print(f"    seed={seed:<6} R2={r2:.4f} | Spearman={rho:.4f}")
    print(f"    R2:       {np.mean(r2s):.4f} ± {np.std(r2s, ddof=1):.4f} "
          f"[{min(r2s):.4f}, {max(r2s):.4f}]")
    print(f"    Spearman: {np.mean(rhos):.4f} ± {np.std(rhos, ddof=1):.4f}")
    print(f"    TSCV:     {np.mean(tscvs):.4f}")
    print(f"    emb_avg:  R2={m_avg['r2_temporal']:.4f} | "
          f"Spearman={m_avg['spearman_temporal']:.4f}")
print("\n  Referencias: 24_05_cat=0.4067 (con has_senti) | 24_05_cat_clean=0.3404 | abl_04=0.3137")
