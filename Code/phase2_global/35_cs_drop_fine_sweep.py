"""
35_cs_drop_fine_sweep.py
=========================
Sweep FINO de cold-start dropout alrededor del óptimo (0.3).

El sweep de 25_ fue grueso: {0.0, 0.3, 0.5, 1.0} → 0.3 ganó por mucho.
Acá probamos {0.20, 0.25, 0.35, 0.40} con la arquitectura v3 (fixes de
tower_v3.py). El punto 0.30 de referencia es 34_05_v3.

Eval downstream: CatBoost emb (64d) + meta5 (5d), igual que 34_.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tower_v3 import train_tower
from v2_data import load_all, run_model

print("=" * 70)
print("35_cs_drop_fine_sweep.py — Sweep fino CS_DROP (v3)")
print("=" * 70)

ctx   = load_all()
meta5 = ctx["meta_aligned"][:, :5]

results = {}
for cs in [0.20, 0.25, 0.35, 0.40]:
    tag = f"35_cs{int(cs*100):02d}"
    print(f"\n{'#'*70}\n# CS_DROP = {cs}\n{'#'*70}")
    out = train_tower(cs_drop=cs, seed=42, neg_mode="uniform_pre2016")
    X = np.hstack([out["emb_std"], meta5])
    m = run_model(ctx, X, tag, f"CatBoost RS v3 cs_drop={cs} + Meta5",
                  f"RS v3 cs{cs} (64d) + meta5 (5d)",
                  notes_extra=f"Sweep fino. best_val={out['best_val']:.4f} "
                              f"epochs={out['epochs_run']}")
    results[tag] = (cs, out["best_val"], m)

print("\n" + "=" * 70)
print("RESUMEN — sweep fino CS_DROP (+ referencia 0.30 = 34_05_v3)")
print("=" * 70)
print(f"{'CS_DROP':>8} {'val_loss':>9} {'R2':>8} {'RMSE':>9} {'Spearman':>9}")
print("-" * 48)
for tag, (cs, bv, m) in sorted(results.items(), key=lambda kv: kv[1][0]):
    print(f"{cs:>8.2f} {bv:>9.4f} {m['r2_temporal']:>8.4f} "
          f"{m['rmse_temporal']:>9.2f} {m['spearman_temporal']:>9.4f}")
print("    0.30  → ver 34_05_v3 en experiment_results.json")
