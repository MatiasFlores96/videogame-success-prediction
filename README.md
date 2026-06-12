# Video Game Popularity Prediction — Steam Dataset

Predicción de popularidad de videojuegos en Steam usando embeddings de sistemas de recomendación (Two-Tower CF) como features para modelos de gradient boosting.

Proyecto de tesis de Maestría en Ciencia de Datos.

---

## Resultado Principal

> **CatBoost + Two-Tower content-augmented (64d) + metadata (6d) → R²=0.407 / RMSE=1015**

Los embeddings del RS content-augmented aportan **+0.093 R²** sobre el mejor baseline de contenido puro (meta+RAWG, R²=0.314). La arquitectura Two-Tower con cold-start dropout (p=0.3) es la clave: permite integrar señal colaborativa sin overfitting a ítems históricos.

---

## Hipótesis Central

> *¿Los embeddings de un sistema de recomendación Two-Tower mejoran la predicción de popularidad de videojuegos frente a enfoques solo-contenido?*

**Respuesta**: Sí, con matices importantes — ver [Conclusiones](#conclusiones).

---

## Tabla de Resultados — Fase 2 (Dataset Global)

| Modelo | R² Temporal | RMSE | ¿Limpio? | Notas |
|--------|:-----------:|:----:|:--------:|-------|
| **24_05_cat** — RS content-aug v2 + Meta | **0.407** | **1015** | ✅ | Resultado principal |
| 23_05 — RS content-aug v2 + Meta (XGBoost) | 0.368 | 1048 | ✅ | |
| cs10_pureContent — Two-Tower CS=1.0 | 0.318 | 1088 | ✅ | Sin señal colaborativa |
| abl_04 — Meta + RAWG raw | 0.314 | 1091 | ✅ | Mejor baseline sin RS |
| abl_05 — TF-IDF + steam + meta + RAWG raw | 0.278 | 1120 | ✅ | |
| 20_10 — Two-Tower estándar (ID-only) + RAWG | 0.147 | 1217 | ✅ | Embeddings ID-only no predicen |
| abl_01 — Metadata only | 0.111 | 1242 | ✅ | |
| 20_03 — Two-Tower estándar (ID-only) | −0.009 | 1324 | ✅ | |
| 21_10_cat — CatBoost + rawg_ratings_count | 0.447 | 979 | ❌ | Leakage temporal en RAWG |
| 19_10 — XGBoost + rawg_ratings_count | 0.438 | 987 | ❌ | Leakage temporal en RAWG |

**Null baseline** (predecir la media de training): RMSE = 1407.76. El modelo principal representa una reducción del 27.9%.

---

## Estructura del Repositorio

```
VG_Recommender/
├── Code/
│   ├── phase1_australian/   # Notebooks de Fase 1 (dataset australiano)
│   │   ├── 01-13_*.ipynb    # Exploración, modelos, ablations
│   │   └── 11_shap_error_analysis.ipynb
│   ├── phase2_global/       # Scripts de Fase 2 (dataset global Steam)
│   │   ├── 15_preprocess_v2.py          # Preprocesamiento 7.76M interacciones
│   │   ├── 16_train_rs_v2.py            # Two-Tower estándar (ID-only) v2
│   │   ├── 17_retrain_v2_experiments.py # Experimentos con emb v2 alineados
│   │   ├── 18_full_v2_pipeline.py       # Pipeline completo v2 (cutoff 2016/2017)
│   │   ├── 20_shap_analysis.py          # SHAP sobre mejor modelo pre-leakage fix
│   │   ├── 21_extended_experiments.py   # CatBoost, LightGBM, ensemble (con leakage)
│   │   ├── 22_train_rs_content.py       # Two-Tower content-aug v1 (descartado)
│   │   ├── 23_train_rs_content_v2.py    # Two-Tower content-aug v2 — arquitectura final
│   │   ├── 24_ablation_catboost.py      # Ablation study + CatBoost — RESULTADO PRINCIPAL
│   │   ├── 25_two_tower_search.py       # Sweep de arquitecturas Two-Tower
│   │   ├── 26-30_train_rs_*.py          # Intentos de mejora (multitask, InfoNCE, cutoff 2017)
│   │   ├── 31_log_transform.py          # Log-transform del target (descartado)
│   │   └── results_tracker.py           # Gestión de experiment_results.json
│   ├── analysis/            # Notebooks de análisis y presentación
│   │   ├── 00_leaderboard.ipynb
│   │   ├── 00_model_comparison.ipynb
│   │   ├── RESUMEN_MODELOS.ipynb
│   │   └── PRESENTACION_TESIS.ipynb
│   └── utils/               # Scripts auxiliares de inspección y verificación
├── Data/
│   ├── experiment_results.json          # Resultados de todos los modelos (machine-readable)
│   ├── interactions_v2.parquet          # 7.76M interacciones (user_idx, item_idx, rating)
│   ├── item2idx_v2.json                 # Mapping appid → item_idx (15,380 juegos)
│   ├── item_embeddings_rs_v2_content2.npy  # Embeddings Two-Tower v2 content-aug (15380, 64)
│   ├── rawg_enriched.csv                # Features RAWG: metacritic, rating, playtime, etc.
│   └── steam_games.json                 # Metadata de juegos Steam
├── EXPERIMENTS.md           # Registro completo de todos los experimentos
└── README.md
```

---

## Fase 1 — Dataset Australiano (3,682 juegos)

**Notebooks**: `Code/phase1_australian/01_*.ipynb` → `13_*.ipynb`

El punto de partida fue el [McAuley Australian Gaming Dataset](https://jmcauley.ucsd.edu/data/steam/): 3,682 juegos, 25,458 usuarios, 59,305 interacciones.

### Problema detectado: cold-start inevitable

El Two-Tower original se entrenaba con todas las interacciones, incluyendo post-2016 → **leakage temporal**. Post-fix (cutoff 2016), el R²≈0.79 original se redujo a R²_TSCV≈0.856–0.940 (within-distribution). Sin embargo, el test temporal (juegos 2016+) dio R²≈−0.02: los 486 juegos del test set no tenían embeddings entrenados → **cold-start**.

El análisis en `02b_cold_start_embeddings.ipynb` confirmó que el espacio colaborativo y el espacio de contenido son **ortogonales** (cosine sim=0.064 entre vecinos de contenido en el espacio RS). No existe mapping aprendible entre ambos → el cold-start es un problema estructural.

**Mejor resultado Fase 1 (limpio)**: Stage 2 content (TF-IDF + RAWG + dev_rep) → R² temporal = **+0.074**.

---

## Fase 2 — Dataset Global Steam (15,380 juegos)

**Scripts**: `Code/phase2_global/15_*.py` → `31_*.py`

### Dataset

| Métrica | Valor |
|---------|-------|
| Items (juegos) | 15,380 |
| Usuarios | 2,567,538 |
| Interacciones totales | 7,762,197 |
| Train (pre-2017) | 10,551 juegos |
| Test (2017+) | 4,798 juegos |
| TSCV | 7 ventanas (2013-07 → 2017-01) |

**Target**: `total_reviews` — número de reviews como proxy de popularidad. Distribución muy sesgada: media=504, max=183,649.

**Cutoff temporal**: 2017-01-01. Los juegos de 2017 tienen ~1 año de reviews acumuladas al momento del snapshot → distribución más uniforme que con cutoff 2016 (donde el test mezclaba cohortes de distinta antigüedad).

### Arquitectura Two-Tower Content-Augmented v2

El breakthrough vino de reemplazar el Two-Tower ID-only por un modelo que integra señal colaborativa y de contenido en el mismo embedding:

```
Item Tower:
  ID Embedding (32d)  ──[cold-start dropout p=0.3]──┐
                                                       concat(64d) → Linear(64d) → LayerNorm → item_vec(64d)
  Content MLP:                                       ──┘
    TF-IDF(100d) + numeric(4d) + RAWG quality(4d) = 108d
    → Linear(108→128) → GELU → Dropout(0.1)
    → Linear(128→64) → GELU → Linear(64→32)

User Tower:
  ID Embedding (64d) → item_vec(64d)

Loss: BCE con negative sampling (1:1)
Optimizador: AdamW, cosine annealing con warmup (2 épocas)
Anti-leakage: entrenado solo con interacciones pre-2016 (cutoff = 2016-01-01)
```

**Cold-start dropout (p=0.3)**: durante training, el ID embedding se zerea con probabilidad 30%, forzando al content MLP a aprender representaciones independientes. En inferencia, los juegos nuevos (post-2016 sin historial) se representan principalmente por el content MLP. El gap de 1 año entre el cutoff del Two-Tower (2016) y el del XGBoost (2017) crea regularización natural: CatBoost entrena con una mezcla de ítems colaborativos y cold-start → generaliza mejor al test set.

### Desarrollo (scripts principales)

| Script | Descripción | Resultado clave |
|--------|-------------|-----------------|
| 16_ | Two-Tower estándar (ID-only) | R²=−0.009 temporal — embeddings ID-only no predicen |
| 18_ | Pipeline completo v2 (cutoff 2016) | R²=+0.15 limpio (cutoff 2017 necesario) |
| 20_ | SHAP + fix leakage rawg_ratings_count | R²: 0.44 → 0.15 al quitar la feature leakeada |
| 22_ | Two-Tower content-aug v1 | R²≈0.13 — convergencia insuficiente (patience=2) |
| 23_ | Two-Tower content-aug v2 (baseline) | R²=0.368 (XGBoost), primer resultado fuerte |
| 24_ | Ablation + CatBoost | **R²=0.407** — resultado principal |
| 25_ | Sweep arquitecturas Two-Tower | CS=0.3 confirmado como óptimo |
| 26-30 | Intentos de mejora | InfoNCE, multi-task, cutoff 2017 — todos inferiores |
| 31_ | Log-transform del target | R²=0.155 (orig) — descartado |

### Ablation: contribución neta del RS

| Baseline | R² | Con RS content-aug | Delta |
|----------|----|--------------------|-------|
| Metadata only (abl_01) | 0.111 | **0.407** (24_05_cat) | **+0.296** |
| Meta + RAWG raw (abl_04) | 0.314 | **0.407** (24_05_cat) | **+0.093** |
| Full content raw, 119d (abl_05) | 0.278 | 0.368 (23_05, 70d) | **+0.090** *(con menos features)* |

### Sweep de arquitecturas (script 25)

| Config | CS_DROP | R² | Interpretación |
|--------|---------|-----|----------------|
| Baseline | 0.3 | **0.407** | Óptimo |
| cs10_pureContent | 1.0 | 0.318 | ≈ abl_04 (sin señal colaborativa) |
| cs0_collab | 0.0 | 0.211 | Overfitting total al grafo histórico |
| cs05_aggressive | 0.5 | 0.174 | Señal colaborativa insuficiente |
| emb128 | 0.3 | 0.157 | Overfitting por capacidad excesiva |
| noRawgInTower | 0.3 | 0.053 | Sin RAWG quality el content MLP no aprende señal de popularidad |

---

## Conclusiones

1. **Los embeddings RS content-augmented SÍ mejoran la predicción**: +0.093 R² sobre el mejor baseline sin RS (0.407 vs 0.314). Los embeddings ID-only no contribuyen (R²=−0.009).

2. **La clave es el cold-start dropout (p=0.3)**: sin él (CS=0.0) el modelo overfittea al grafo histórico y colapsa a R²=0.211. El dropout actúa como regularizador que integra señal colaborativa sin perder generalización.

3. **RAWG quality features son el ancla del content MLP**: sin `metacritic`, `rawg_rating`, `playtime_avg`, `esrb` en el tower, el modelo solo aprende similitud de géneros/tags — que no correlaciona con popularidad futura (R²=0.053 en noRawgInTower).

4. **La compresión del Two-Tower supera la concatenación manual**: 23_05 (RS 64d + meta 6d = 70 features, R²=0.368) > abl_05 (TF-IDF + steam + meta + RAWG = 119 features, R²=0.278). El embedding aprende representaciones más informativas que la concatenación explícita.

5. **El diseño óptimo resultó ser no obvio**: el gap de 1 año entre el cutoff del Two-Tower (2016) y el cutoff del XGBoost (2017) crea regularización natural. Los intentos de mejora (scripts 26-30) — InfoNCE, multi-task con pop_head, cutoff 2017 para ambos — todos empeoraron el resultado.

6. **Limitaciones honestas**: El target (`total_reviews`) es una muestra sesgada. La distribución es muy skewed (max=183,649). La cobertura RAWG es 2,866/15,380 ítems (18.6%) — el modelo depende de features externas disponibles en pocos ítems. El aporte neto del RS (~+0.09 sobre el mejor contenido-puro) es real pero moderado.

---

## Metodología de Validación

| Esquema | Descripción | Métrica reportada |
|---------|-------------|-------------------|
| **Temporal split** | Train = pre-2017 (10,551), Test = 2017+ (4,798) | **Métrica principal** — simula predicción de juegos futuros |
| **TSCV** | 7 ventanas rolling (2013-07 → 2017-01) | Within-distribution, estimación de varianza |
| **K-Fold** | 5 folds estándar | Referencia, no métrica principal |

**Anti-leakage**: el Two-Tower se entrena exclusivamente con interacciones de juegos pre-2016, garantizando que los embeddings no codifican popularidad de ítems del test set. El TF-IDF se fitea sobre ítems pre-2017 y se transforma sobre todos los ítems.

---

## Tecnologías

| Área | Librería |
|------|----------|
| Gradient Boosting | CatBoost, XGBoost, LightGBM |
| Deep Learning / RS | PyTorch, CUDA (RTX 4080), AMP |
| Optimización HP | Optuna (TPE sampler) |
| NLP | TF-IDF (scikit-learn) |
| Interpretabilidad | SHAP |
| Data | pandas, numpy, pyarrow |
| Externo | RAWG API |

---

## Registro Completo de Experimentos

Ver [EXPERIMENTS.md](EXPERIMENTS.md) para el historial completo: todos los experimentos por script, bugs encontrados y corregidos, análisis de leakage, y la evolución del resultado desde R²≈−0.02 (Fase 1 cold-start) hasta R²=0.407 (resultado final).
