# Video Game Popularity Prediction — Steam Dataset

Predicción de popularidad de videojuegos en Steam usando embeddings de sistemas de recomendación (Two-Tower CF) como features para modelos de gradient boosting.

Proyecto de tesis de Maestría en Ciencia de Datos.

---

## Resultado Principal

La pregunta de la tesis tiene una respuesta de **dos regímenes**, validada con multi-seed y barras de error (scripts 32-39, auditoría de robustez):

| Régimen | RS content-aug (Two-Tower) | Contenido-puro (meta+RAWG) | Veredicto |
|---------|---------------------------|----------------------------|-----------|
| **Within-distribution** (juegos con historial) — TSCV | **R²=0.699 ± 0.061** | 0.150 ± 0.097 | **RS gana ×4.7** |
| Within-distribution — KFold | **0.75** | 0.33 | RS gana ×2.3 |
| **Extrapolación temporal** (juegos 2017+, cold-start) | 0.23 ± 0.06 | 0.27 ± 0.04 | Empate (bandas solapadas) |
| Ranking temporal (Spearman) | 0.49 ± 0.02 | 0.52 ± 0.02 | Empate |

> **Los embeddings colaborativos son transformadores para juegos del catálogo con historial, pero no superan al contenido para juegos nuevos — el cold-start es estructural.** El sistema correcto es de dos etapas: RS para el catálogo, contenido para lanzamientos.

⚠️ **Nota de honestidad metodológica**: una versión anterior de este proyecto reportaba R²=0.407 temporal como resultado principal. La auditoría de robustez (multi-seed, 5 semillas × 2 configuraciones + barras de error del baseline) demostró que ese número era una **realización afortunada** (distribución real: 0.22-0.24 ± 0.02) amplificada por varianza de tuning. El proceso completo de detección está documentado en [EXPERIMENTS.md](EXPERIMENTS.md) — y es, en sí mismo, una de las contribuciones de la tesis.

---

## Hipótesis Central

> *¿Los embeddings de un sistema de recomendación Two-Tower mejoran la predicción de popularidad de videojuegos frente a enfoques solo-contenido?*

**Respuesta**: Sí within-distribution (×4.7 en TSCV, robusto); no en cold-start temporal — ver [Conclusiones](#conclusiones).

---

## Tabla de Resultados — Fase 2 (Dataset Global)

Resultados post-auditoría, con barras de error donde hay repeticiones (5 seeds):

| Modelo | R² Temporal | Spearman | TSCV | Notas |
|--------|:-----------:|:--------:|:----:|-------|
| **RS content-aug v3 + meta5** (multi-seed) | **0.219 ± 0.017** | **0.496 ± 0.009** | **0.693** | Pipeline limpio, 5 seeds |
| Contenido-puro meta5+RAWG (multi-seed) | 0.274 ± 0.037 | 0.516 ± 0.021 | 0.150 | Baseline sin RS, 5 tuning seeds |
| RS all-cold-start (deployment) | 0.191 | 0.547 | 0.345 | Solo content MLP, juegos nuevos |
| RS + Tweedie loss | 0.245 | **0.573** | 0.608 | Mejor ranking del proyecto |
| Two-Tower ID-only (20_03) | −0.009 | — | 0.654 | Sin content tower: no predice temporal |
| Metadata only (abl_01) | 0.111 | — | 0.021 | |

**Resultados históricos descartados por la auditoría:**

| Modelo | R² | Causa |
|--------|-----|-------|
| ~~24_05_cat~~ | ~~0.407~~ | Realización afortunada del tower + tuning con columna inerte (has_senti) |
| ~~21_10_cat~~ | ~~0.447~~ | Leakage duro: rawg_ratings_count |
| ~~19_10~~ | ~~0.438~~ | Leakage duro: rawg_ratings_count |

**Null baseline** (predecir la media de training): RMSE = 1407.76.

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
│   │   ├── 32-39_*.py                   # AUDITORÍA: leakage, multi-seed, fixes, barras de error
│   │   ├── v2_data.py                   # Módulo compartido: datos + features + eval (scripts 32+)
│   │   ├── tower_v3.py                  # Two-Tower parametrizado con fixes metodológicos
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
| 23_ | Two-Tower content-aug v2 | R²=0.368 (XGBoost) — luego revisado por auditoría |
| 24_ | Ablation + CatBoost | R²=0.407 — luego revisado por auditoría |
| 25_ | Sweep arquitecturas Two-Tower | CS∈{0,0.3,0.5,1.0}: extremos colapsan |
| 26-30 | Intentos de mejora | InfoNCE, multi-task, cutoff 2017 — todos inferiores |
| 31_ | Log-transform del target | R²=0.155 (orig) — descartado |
| **32-39** | **Auditoría de robustez** | Multi-seed, leakage metadata, fixes tower, barras de error — **revisa las conclusiones** |

### Auditoría de robustez (scripts 32-39)

| Script | Pregunta | Veredicto |
|--------|----------|-----------|
| 32/32b | ¿has_senti (snapshot 2018) leakea? | SHAP=0: el modelo no lo usó; la varianza de tuning (±0.05) es el hallazgo real |
| 33/33b | ¿Tweedie/Poisson/ensemble/metascore mejoran? | Tweedie gana ranking (ρ=0.573); ensemble honesto ≈ baseline; Poisson colapsa |
| 34 | ¿Fixes metodológicos del tower? | TSCV/KFold récord (0.69/0.75); el val-loss "bugueado" de 23_ seleccionaba popularidad accidentalmente |
| 35 | ¿CS_DROP=0.3 es óptimo? | Meseta plana en [0.2, 0.4] |
| 36 | ¿El 0.407 es reproducible? | **No: distribución real 0.22-0.24 ± 0.02 (5 seeds × 2 configs)** |
| 37 | ¿MiniLM > TF-IDF en el tower? | Empate — el cuello de botella no es la representación de tags |
| 38 | ¿Fit transductivo o suerte? | Réplica exacta a 4 decimales (seed 42) + seed 123 da 0.15 → **suerte** |
| 39 | ¿Barras del baseline? | Contenido 0.274±0.037; la varianza dominante es el tuning downstream (σ=0.055) |

### Ablation: contribución neta del RS (post-auditoría)

La contribución del RS depende del régimen de evaluación:

| Métrica | RS v3 + meta5 | Contenido-puro (meta5+RAWG) | Delta RS |
|---------|---------------|------------------------------|----------|
| TSCV (within-dist) | **0.699 ± 0.061** | 0.150 ± 0.097 | **+0.55** ✅ |
| KFold | **0.75** | 0.33 | **+0.42** ✅ |
| R² temporal (cold-start) | 0.22 ± 0.02 | 0.27 ± 0.04 | −0.05 (solapado) |
| Spearman temporal | 0.49 ± 0.02 | 0.52 ± 0.02 | ≈0 (solapado) |

### Sweep de arquitecturas (scripts 25 + 35)

El sweep grueso (25_) mostró que los extremos colapsan; el fino (35_) que el interior es meseta:

| CS_DROP | R² (una corrida) | Interpretación |
|---------|------------------|----------------|
| 0.0 | 0.211 | Overfitting al grafo histórico (val=0.802) |
| 0.20–0.40 | 0.21–0.23 | **Meseta plana** — diferencias dentro del ruido (σ≈0.02) |
| 0.5 | 0.174 | Señal colaborativa insuficiente |
| 1.0 | 0.318* | Contenido puro (*una corrida, sin barras) |
| EMB=128 | 0.157 | Overfitting por capacidad |
| sin RAWG en tower | 0.053 | RAWG quality es el ancla del content MLP |

---

## Conclusiones

1. **Los embeddings colaborativos son transformadores within-distribution**: TSCV 0.699±0.061 vs 0.150±0.097 del contenido-puro (×4.7, bandas sin solape). Para juegos del catálogo con historial, la señal colaborativa domina. Los embeddings ID-only sin content tower no predicen nada (R²=−0.009).

2. **El cold-start es estructural y el content-augmented tower no lo resuelve**: en extrapolación temporal (juegos 2017+), RS (0.23±0.06) y contenido-puro (0.27±0.04) empatan. La arquitectura empaqueta la señal de contenido en el embedding (útil operacionalmente: un solo vector por juego, ρ=0.547 en deployment puro), pero no crea información colaborativa donde no la hay. Confirma a escala global la conclusión de la Fase 1.

3. **El sistema correcto es de dos etapas**: RS para el catálogo con historial (TSCV 0.70), contenido para lanzamientos nuevos. Mismo diseño que concluyó la Fase 1, ahora con rigor estadístico.

4. **Una corrida no es un resultado** (lección metodológica central): la varianza estocástica total del pipeline (~±0.06 R²) superaba el efecto que se quería medir (+0.09). El "resultado principal" original (0.407) era una realización afortunada — detectado con multi-seed (5 semillas × 2 configs) y réplica exacta (38_: seed 42 reproduce a 4 decimales, seed 123 da 0.15). La fuente de varianza dominante es el **tuning downstream** (Optuna/CatBoost, σ=0.055), no el entrenamiento del tower (σ=0.017).

5. **Tres capas de leakage detectadas y cuantificadas**: `rawg_ratings_count` (duro, −0.29 R² al removerlo), metadata snapshot 2018 (`has_senti` inerte con SHAP=0; `num_tags` soft-leakage ≤0.04), y fit transductivo de TF-IDF/z-score (agrega varianza, no sesgo sistemático).

6. **RAWG quality features son el ancla del content MLP**: sin `metacritic`, `rawg_rating`, `playtime_avg`, `esrb` en el tower, el modelo colapsa (R²=0.053). El cold-start dropout funciona en meseta [0.2, 0.4] — el valor exacto no importa.

7. **R² temporal es frágil con outliers extremos (max=183,649); Spearman es estable** (±0.01-0.02 entre seeds). Para el caso de uso real (priorizar juegos por popularidad esperada), el ranking es la métrica relevante — y ahí Tweedie loss da el mejor resultado del proyecto (ρ=0.573).

8. **Limitaciones honestas**: target = muestra sesgada (reviews como proxy); cobertura RAWG 18.6%; el test set 2017 tiene solo ~1 año de acumulación de reviews; los resultados negativos (MiniLM ≈ TF-IDF, Poisson colapsa, ensemble honesto ≈ baseline, emb-averaging entre seeds destruye señal) están documentados en EXPERIMENTS.md.

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

Ver [EXPERIMENTS.md](EXPERIMENTS.md) para el historial completo: todos los experimentos por script, bugs encontrados y corregidos, las tres capas de leakage detectadas, la auditoría de robustez multi-seed (scripts 32-39), y la evolución honesta del proyecto — desde R²≈−0.02 (Fase 1 cold-start), pasando por el espejismo del 0.407, hasta la conclusión de dos regímenes con barras de error.
