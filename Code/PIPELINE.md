# Pipeline Canónico — Capítulo 3 (Metodología)

Este documento describe el flujo de reproducción de los resultados del Capítulo 3.
Para los resultados del Capítulo 4 (análisis), ver `analysis/results_tables.py`.

---

## Flujo principal (en orden de ejecución)

```
phase2_global/
  15_preprocess_v2.py          # [1] Preprocessing: reviews → interactions_v2.parquet
  tower_v3.py                  # [2] Módulo: arquitectura Two-Tower v3 (parametrizado)
  v2_data.py                   # [3] Módulo: pipeline de evaluación compartido
  34_train_rs_content_v3.py    # [4] Entrena tower v3 + evalúa downstream
  24_ablation_catboost.py      # [5] Ablation: baselines de contenido puro
  32_audit_has_senti.py        # [6] SHAP audit: has_senti + num_tags
```

Después de [4] y [5], correr para resultados con barras de error:
```
  36_multiseed_stability.py    # [7] Estabilidad multi-seed (5 seeds × 2 configs)
  39_baseline_errorbars.py     # [8] Barras de error: contenido vs RS fijo
```

Generar tablas del capítulo:
```
analysis/results_tables.py     # [9] Tablas 1-4 para la tesis (markdown + LaTeX)
```

---

## Scripts de exploración (no reproducir — resultados en experiment_results.json)

Estos scripts documentan el proceso de diseño. Sus resultados están guardados y no
es necesario re-ejecutarlos. El pipeline canónico está en los scripts listados arriba.

| Script | Rol | Resultado / Lección |
|--------|-----|---------------------|
| `14_improvements.py` | Winsorización, Huber loss | Sin mejora sobre baseline |
| `16_train_rs_v2.py` | Primer Two-Tower sobre datos v2 | Baseline colaborativo puro |
| `17_retrain_v2_experiments.py` | Re-entrenamiento modelos v1 con emb v2 | Mejora por calidad de datos |
| `18_full_v2_pipeline.py` | Pipeline v2 completo (pre-CatBoost) | Supersedido por 23_ + 24_ |
| `20_shap_analysis.py` | SHAP sobre 19_10 | rawg_ratings_count = leakage duro |
| `21_extended_experiments.py` | CatBoost/LGBM sobre setup leakeado | R²=0.447 es artefacto (rawg_ratings_count) |
| `22_train_rs_content.py` | Content-aug tower v1 (sin CS_DROP) | CS_DROP necesario para cold-start |
| `23_train_rs_content_v2.py` | Content-aug tower v2 (baseline) | **Supersedido por `tower_v3.py`** |
| `25_two_tower_search.py` | CS_DROP sweep: 0.0, 0.5, 1.0, emb_dim=128 | CS_DROP=0.3 como punto de partida |
| `26_train_rs_improved.py` | InfoNCE + multi-task + cutoff 2017 | InfoNCE overfit, multitask = varianza |
| `27_train_rs_multitask.py` | Multi-task conservador (λ_pop × MSE) | Sin mejora sobre BCE |
| `28_train_rs_multitask_fixed.py` | Multi-task con fix de leakage | Sin mejora |
| `29_train_rs_multitask_v2.py` | Multi-task v2 | Sin mejora; BCE solo es suficiente |
| `30_train_rs_cutoff2017.py` | Cutoff 2017 en el tower (vs 2016) | Sin mejora; 2016 es el corte correcto |
| `31_log_transform.py` | Log-transform del target | RMSE mejora en log-space; R² peor en original |

---

## Archivos de datos clave

```
Data/
  interactions_v2.parquet              # Reviews como interacciones implícitas
  item2idx_v2.json / user2idx_v2.json  # Mappings AppID → idx
  steam_games.json                     # Metadata de juegos (Steam)
  rawg_enriched.csv                    # Metadata de calidad (RAWG API)
  item_embeddings_rs_v2_content2.npy   # Embeddings v2 (tower original 23_)
  item_embeddings_rs_v2_content3.npy   # Embeddings v3 (tower con fixes)
  item_embeddings_rs_v2_content3_cs.npy  # Embeddings v3 cold-start
  experiment_results.json              # Resultados de todos los experimentos
```

---

## Entorno

```bash
conda activate VG_RS
# o: C:\Users\matia\anaconda3\envs\VG_RS\python.exe <script>
```

En Windows, configurar encoding antes de correr scripts con caracteres especiales:
```powershell
$env:PYTHONIOENCODING="utf-8"
```
