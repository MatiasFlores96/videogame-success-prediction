# Registro de Experimentos — VG Recommender

Tesis de Maestría en Ciencia de Datos  
**Hipótesis central**: Los embeddings de sistemas de recomendación (Two-Tower CF) usados como features para XGBoost mejoran la predicción de popularidad de videojuegos frente a enfoques solo-contenido.

---

## Configuración General

- **Target**: `total_reviews` = número de usuarios que dejaron review en el dataset (proxy de popularidad/ventas)
- **Split temporal**: Train = juegos lanzados antes de 2016-01-01 (2621 juegos), Test = 2016+ (486 juegos)
- **TSCV**: 5 ventanas temporales dentro de pre-2016 (2013-2016), simulando predicción rolling
- **Optimización**: Optuna, 50 trials por modelo, TPE sampler, seed=42
- **Base learner**: XGBoost con tree_method='hist'

---

## Correcciones Críticas Aplicadas

### 1. Leakage Temporal en Two-Tower
El Two-Tower original se entrenaba con TODAS las interacciones incluyendo juegos post-2016.  
**Fix**: Se filtró el entrenamiento a interacciones pre-2016 (`cutoff = 2016-01-01`).  
**Impacto**: Los resultados del README original (R²≈0.79) eran inflados.

### 2. TF-IDF Fiteado en Todos los Juegos
Los vectorizadores TF-IDF en los modelos 08, 11b, 12, 13b, 13c se fiteaban sobre todos los juegos (incluyendo post-2016), incorporando información de frecuencia de tags del test set.  
**Fix**: TF-IDF ahora se fitea solo sobre juegos pre-2016 y se transforma sobre todos los juegos.  
**Impacto**: El R²_log del Modelo 13c bajó de 0.254 → 0.039. El resultado anterior estaba inflado por leakage de TF-IDF.

### 3. Optuna Split Sin Orden Temporal
El 80/20 interno de Optuna para tuneo de hiperparámetros dividía `X_train` por posición (no por fecha), mezclando juegos de distintas épocas.  
**Fix**: `X_train` y `y_train` ahora se ordenan por fecha de lanzamiento antes del split de Optuna.  
**Impacto**: TSCV mejoró en todos los modelos (ej: 03: 0.770→0.886). Modelo 11a TSCV pasó de -0.808→+0.778 (la "catástrofe" era un artefacto del bug). Modelo 12 TSCV pasó de -5.03→+0.469.

---

## Tabla de Resultados Completa (post-fix, versión final)

| ID  | Modelo                        | Features                                          | R² Temporal | RMSE Temporal | R² TSCV | RMSE TSCV | Nota |
|-----|-------------------------------|---------------------------------------------------|:-----------:|:-------------:|:-------:|:---------:|------|
| 03  | RS Embeddings Only            | RS clean (64d)                                    | −0.018      | 57.42 | **0.886** | 14.28 | Baseline RS |
| 04  | Metadata Only                 | Steam metadata (6d)                               | +0.057      | 55.24 | −0.214 | 39.44 | |
| 05  | RS + Metadata                 | RS (64d) + Metadata (6d)                          | −0.016      | 57.35 | 0.889 | 14.24 | |
| 06  | Review Text Emb               | BERT reviews (384d)                               | −0.011      | 57.20 | −0.172 | 39.51 | |
| 07  | Tag Embeddings                | BERT tags (384d)                                  | −6.93       | 16.19* | −3.776 | 24.18 | Inestable (RMSE baja por colapso) |
| 08  | Hybrid Collab-Content         | RS (64d) + TF-IDF (100d) + Numeric (2d)           | −0.024      | 57.57 | 0.856 | 16.20 | **Mejor TSCV lineal** |
| 09  | RS + Review Text              | RS (64d) + BERT reviews (384d)                    | −0.021      | 57.50 | 0.873 | 14.01 | |
| 10  | RS + Reviews + RAWG           | RS (64d) + BERT (384d) + RAWG (12d)               | −0.016      | 57.36 | 0.863 | 14.76 | |
| 11a | RS + Dev Reputation           | RS (64d) + dev_rep (1d)                           | −0.026      | 57.62 | 0.778 | 20.37 | dev_rep no aporta |
| 11b | Hybrid + Dev Reputation       | RS (64d) + TF-IDF (100d) + Num (2d) + dev_rep    | −0.016      | 57.36 | 0.858 | 16.25 | |
| 12  | Stage2 Content                | TF-IDF (100d) + Steam (2d) + RAWG (12d) + dev_rep| **+0.074**  | **54.76** | 0.469 | 26.37 | **Mejor temporal lineal** |
| 13a | RS Only (log target)          | RS clean (64d) [log(1+y)]                         | −0.026 (orig) | — | **0.937** (log) | 0.275 | TSCV log mejor que lineal |
| 13b | Hybrid (log target)           | RS+TF-IDF+Numeric [log(1+y)]                      | −0.025 (orig) | — | **0.940** (log) | 0.268 | **Mejor TSCV absoluto** |
| 13c | Stage2 Content (log target)   | TF-IDF+Steam+RAWG+dev_rep [log(1+y)]              | +0.050 (orig) | 55.43 | 0.754 (log) | 0.547 | Log no supera lineal en temporal |

> *Modelo 07: RMSE temporal aparentemente bajo porque el modelo colapsa prediciendo valores pequeños constantemente (R²=-6.93 confirma el colapso).

---

## Hallazgos Clave

### 1. Cold-Start Degrada los RS Embeddings
Los 486 juegos post-2016 no aparecen en el entrenamiento del Two-Tower. Sus embeddings son vectores no entrenados.  
→ Todos los modelos con RS obtienen R²≈−0.02 en temporal (peor que predecir la media).

### 2. RS Embeddings Superiores Within-Distribution
En el régimen pre-2016 donde todos los juegos tienen embeddings entrenados:
- **Modelo 08** (Hybrid RS+TF-IDF): TSCV R² = **0.856**, RMSE = 16.20
- **Modelo 09** (RS+Reviews): TSCV R² = **0.873**, RMSE = 14.01
- **Metadata Only**: TSCV R² = **−0.214** (muy inferior)

La hipótesis de la tesis se **confirma within-distribution** y **falla out-of-distribution** por cold-start.

### 3. El Espacio Colaborativo es Ortogonal al de Contenido
Análisis en `02b_cold_start_embeddings.ipynb`:
- Cosine similarity en features (tags, géneros): **0.987** entre vecinos más cercanos
- Cosine similarity en RS: **0.064** para los mismos pares
- k-NN (k=10) para inferir embeddings RS desde contenido: R²CV = **−0.085**

El cold-start es un problema estructural.

### 4. Developer Reputation — Resultado Revisado
Con el fix del sort de Optuna, el TSCV de dev_rep pasó de -0.808 a **+0.778** para modelo 11a.  
La "catástrofe" era un artefacto del bug de sort (Optuna veía juegos futuros durante validación).  
Conclusión real: dev_rep no mejora significativamente sobre RS solo (11a: 0.778 vs 03: 0.886).

### 5. Sistema de Dos Etapas — Resultados Finales
**Stage 1** (juegos con historial pre-2016): RS embeddings → TSCV R²=0.856-0.940  
**Stage 2** (juegos nuevos post-2016): TF-IDF + RAWG + Steam + dev_rep → temporal R²=**+0.074**

Stage 2 supera el baseline de metadata pura (0.057) pero la ventaja es modest (+0.017).  
TSCV de Stage 2 = 0.469 (positivo, no catastrófico como antes del fix del sort).

### 6. Log-Transform del Target
El log-transform mejora el TSCV within-distribution para modelos RS:
- Modelo 13b (Hybrid+log): TSCV R² = **0.940** (mejor resultado absoluto en TSCV)
- Modelo 13a (RS+log): TSCV R² = **0.937**

Sin embargo, para el Stage 2 (cold-start), el log-transform **no mejora** sobre el modelo lineal:
- Modelo 12 (lineal): temporal R² = **+0.074**, RMSE = 54.76
- Modelo 13c (log): temporal R² = **+0.050** (back-transformado), RMSE = 55.43

La mejora espectacular anteriormente reportada (R²_log=0.254) era un artefacto del leakage de TF-IDF.

---

## Notebooks por Modelo

| Notebook | Descripción |
|----------|-------------|
| `01_generate_dataset.ipynb` | Limpieza de datos Steam, australianos, RAWG |
| `02_model_keras_rs.ipynb` | Two-Tower CF con Keras/JAX. Genera embeddings 64d |
| `02b_cold_start_embeddings.ipynb` | Cold-start: k-NN + RAWG features + developer_reputation |
| `03_regressor_embeddings_only.ipynb` | Modelo 03 |
| `04_regressor_metadata_only.ipynb` | Modelo 04 |
| `05_regressor_embeddings_plus_metadata.ipynb` | Modelo 05 |
| `06_regressor_review_embeddings.ipynb` | Modelo 06 |
| `07_regressor_tag_embeddings.ipynb` | Modelo 07 |
| `08_hybrid_collaborative_content.ipynb` | Modelo 08 |
| `09_regressor_rs_plus_reviews.ipynb` | Modelo 09 |
| `10b_model_enriched.ipynb` | Modelo 10 |
| `11_developer_reputation.ipynb` | Modelos 11a, 11b |
| `12_two_stage_model.ipynb` | Modelo 12 (Stage 2 del sistema dos-etapas) |
| `13_log_transform.ipynb` | Modelos 13a/b/c: log(1+y) como target |

---

## Archivos de Datos Clave

| Archivo | Descripción |
|---------|-------------|
| `item_embeddings_rs_clean.npy` | Embeddings Two-Tower post-fix (3682, 64) |
| `user_embeddings_rs_clean.npy` | Embeddings de usuarios (25458, 64) |
| `item_embeddings_rs_coldstart.npy` | Embeddings post-2016 inferidos por k-NN (3682, 64) |
| `developer_reputation.npy` | Reputación por item_idx (3682,) float32 |
| `rawg_enriched.csv` | Features de RAWG API: metacritic, rating, playtime, etc. |
| `experiment_results.json` | Resultados de todos los modelos (formato machine-readable) |

---

## Interpretación para la Tesis

**Argumento central (sostenible post-fix)**:
> Los embeddings de CF capturan señal colaborativa que supera a las features de contenido *dentro de distribución* (TSCV R²=0.856-0.940 vs −0.214 para metadata sola). El cold-start para juegos nuevos es un problema estructural: el espacio colaborativo y de contenido son ortogonales (cosine sim=0.064), haciendo imposible una aproximación eficiente. Un sistema de dos etapas — CF para juegos conocidos, contenido enriquecido para nuevos — es el mejor compromiso práctico.

**Limitaciones honestas**:
- El target (`total_reviews`) es una muestra sesgada: usuarios australianos que escribieron reviews
- Distribución extremadamente sesgada (mayoría <20 reviews, máximo 3.759)
- Cold-start inevitable con datos colaborativos puros
- Stage 2 supera metadata pura en temporal apenas marginalmente (+0.017 de R²)
- El período de test (post-2016) coincide con crecimiento explosivo de Steam

**Sistema final recomendado**:
| Etapa | Modelo | Escenario | Métrica |
|-------|--------|-----------|---------|
| **Stage 1** | Hybrid RS+TF-IDF (08 / 13b) | Juegos con historial | TSCV R²=0.856–0.940 |
| **Stage 2** | Stage2 Content lineal (12) | Juegos nuevos | Temporal R²=+0.074 |

---

---

# FASE 2 — Dataset Global v2 (Steam completo)

*Iniciada: 2026-05-21*

## Motivación

El dataset australiano original (3,682 items, 25K usuarios) tenía problemas estructurales:
- Cold-start: 100% de los juegos del test set (post-2016) sin embeddings entrenados
- Dataset demasiado pequeño y geográficamente sesgado
- Mejor R² temporal conseguido: +0.074 (Modelo 12)

Se migró todo el pipeline al **dataset global de Steam** (~2.56M usuarios, 7.76M interacciones, 15,380 items).

## Archivos nuevos

| Archivo | Descripción |
|---------|-------------|
| `item2idx_v2.json` | Mapping appid → item_idx para 15,380 juegos |
| `interactions_v2.parquet` | 7.76M interacciones globales (user_idx, item_idx, rating) |
| `item_embeddings_rs_v2.npy` | Embeddings Two-Tower v2 (15380, 64) |
| `item_embeddings_v2_aligned.npy` | Embeddings v2 alineados al espacio v1 (3682, 64) |

## Script 16 — Reentrenamiento Two-Tower v2

**Archivo**: `16_train_rs_v2.py`  
**Datos**: 7.76M interacciones → filtradas a pre-2016 → ~5.8M interacciones de entrenamiento (anti-leakage)  
**Hardware**: RTX 4080, mixed precision (AMP)  
**Output**: `item_embeddings_rs_v2.npy` (15380, 64)

La reducción a pre-2016 para el Two-Tower es deliberada: los embeddings no deben "ver" la popularidad de juegos post-2016 que luego están en el test set.

## Script 17 — Re-experimentos con embeddings v2 alineados

**Archivo**: `17_retrain_v2_experiments.py`  
**Setup**: Embeddings v2 → alineados al espacio v1 por AppID → target australiano original  
**Resultado clave**: Primera vez que algunos modelos dan R² temporal positivo con RS embeddings

| ID | Modelo | R² Temporal | R² TSCV |
|----|--------|-------------|---------|
| 05v2 | RS + Metadata (v2) | **+0.024** | 0.335 |
| 10v2 | RS + Reviews + RAWG (v2) | **+0.021** | 0.360 |
| 11bv2 | Hybrid + Dev Reputation (v2) | +0.001 | 0.414 |
| 08v2 | Hybrid Collab-Content (v2) | -0.002 | 0.359 |

El TSCV bajó de ~0.88 a ~0.33 porque los embeddings son ahora globales pero el target sigue siendo australiano → menor coherencia entre señal y target.

## Script 18 — Pipeline full v2 (cutoff 2016)

**Archivo**: `18_full_v2_pipeline.py` (primera versión, cutoff 2016-01-01)  
**Setup**: Target = reviews globales v2, Embeddings = v2, TF-IDF y dev_rep reconstruidos para v2  
**Train**: 6,934 juegos | **Test**: 8,415 juegos (post-2016)

### Error resuelto: duplicate labels
`steam_games.json` contiene AppIDs duplicados. Fix: `.drop_duplicates(subset=["item_idx"], keep="first")` antes de cada `.set_index("item_idx")`.

### Resultados (cutoff 2016)

| ID | Modelo | R² Temporal | RMSE | R² TSCV |
|----|--------|-------------|------|---------|
| 18_11b | Hybrid + Dev Rep | -0.005 | 1547 | 0.923 |
| 18_05 | RS + Metadata | -0.012 | 1553 | 0.913 |
| 18_13a/b/log | Log-target models | ~-0.022 | ~1560 | 0.89-0.92 |
| 18_10 | Hybrid+RAWG | -0.067 | 1595 | 0.863 |

TSCV excelente (0.86-0.92) pero R² temporal sigue negativo. Diagnóstico: **distribución shift**.

### Diagnóstico de distribución shift

Script `_check_year_dist.py` reveló:
- Dataset termina en 2018
- 2017 tiene 4,759 juegos con solo ~1 año de reviews (mean=163)
- 2016 tiene 3,617 juegos con ~2 años (mean=317)
- Train pre-2016: mean=836 reviews
- Ratio train/test: 3.65x → los juegos del test no tuvieron tiempo de acumular reviews

Los modelos log-target colapsan a RMSE≈1560 (predicen una constante) porque la distribución shift los hace extrapolar fuera de distribución.

## Script 19 — Pipeline full v2 (cutoff 2017) — MEJOR RESULTADO

**Archivo**: `18_full_v2_pipeline.py` (actualizado con cutoff 2017-01-01)  
**Motivación**: Mover el cutoff a 2017 da al test set (solo juegos 2017) mayor uniformidad — todos tienen ~1 año de reviews, evitando mezcla de cohortes con distintos tiempos de acumulación.

**Train**: 10,551 juegos (pre-2017) | **Test**: 4,798 juegos (2017+)  
**TSCV**: 7 ventanas (2013-07 hasta 2017-01)

### Resultados (cutoff 2017) — **Mejor resultado temporal del proyecto hasta la fecha**

| ID | Modelo | R² Temporal | RMSE | R² TSCV | R² KFold |
|----|--------|-------------|------|---------|---------|
| **19_10** | **Hybrid+RAWG** | **+0.438** | **987** | 0.674 | 0.634 |
| 19_10_log | Hybrid+RAWG log | +0.191 | 1185 | 0.665 | 0.657 |
| 19_13b | Hybrid+TF-IDF log | +0.120 | 1236 | 0.633 | 0.611 |
| 19_11b_log | Hybrid+DevRep log | +0.118 | 1237 | 0.640 | 0.608 |
| 19_05 | RS + Metadata | +0.112 | 1242 | 0.663 | 0.645 |
| 19_08 | Hybrid Collab-Content | +0.111 | 1242 | 0.639 | 0.559 |
| 19_11b | Hybrid + Dev Rep | +0.085 | 1260 | 0.641 | 0.490 |
| 19_03 | RS Only | -0.009 | 1324 | 0.654 | 0.623 |
| 19_13a | RS Only log | -0.010 | 1324 | 0.457 | 0.398 |

**19_10 con R²=0.438 es el mejor R² temporal de toda la historia del proyecto.**

### Observaciones clave (cutoff 2017)
1. **RAWG es el feature más importante**: la diferencia entre 19_10 (R²=0.44) y 19_08 (R²=0.11) es solo la adición de RAWG features (metacritic, ratings, playtime). RAWG solo cubre 18.6% de items pero domina la predicción.
2. **RS solo no predice temporal**: 19_03 (R²=-0.009) — los embeddings capturan patrones históricos de consumo, no popularidad futura.
3. **No-log supera log en temporal**: 19_10 (0.44) >> 19_10_log (0.19). Con test set más uniforme el espacio lineal funciona mejor.
4. **TSCV bajó de 0.92 (cutoff 2016) a 0.67 (cutoff 2017)**: más ventanas y más varianza entre períodos.

## Próximos pasos (pendientes)

1. **SHAP analysis sobre 19_10**: entender qué features de RAWG impulsan el R²=0.44
2. **Expandir cobertura RAWG**: actualmente 18.6% → conseguir más datos para más items
3. **Feature engineering RAWG**: interacciones metacritic×ratings_count, percentiles por año
4. **Aumentar Optuna trials**: 60 → 150 para 19_10
5. **Ensemble 19_10 + 19_10_log**: promediar predicciones
6. **LightGBM / CatBoost**: comparar con XGBoost
7. **Análisis de RS embeddings**: entender por qué no contribuyen al test temporal

---

## Script 20 — SHAP Analysis (19_10)

**Archivo**: `20_shap_analysis.py`  
**Propósito**: Entender qué features impulsan el R²=0.438 del mejor modelo (19_10).

Nota técnica: `shap.TreeExplainer(model)` falló con error de parsing por incompatibilidad de versiones. Fix: se usó XGBoost nativo `booster.predict(dtest, pred_contribs=True)[:, :-1]`.

### Resultado principal

| Feature | SHAP mean |abs| |
|---------|-----------|
| **log_ratings_count** | **250.91** |
| rawg_rating | 32.14 |
| metacritic | 28.97 |
| dev_rep_v2 | 9.43 |
| playtime_avg_h (log) | 7.21 |
| ... | ... |

`rawg_ratings_count` (`log_ratings_count` en el modelo) domina absolutamente con SHAP=250 vs el siguiente feature en 32. Esto disparó la investigación sobre leakage temporal.

### Leakage temporal detectado: `rawg_ratings_count`

`rawg_ratings_count` es un snapshot estático de la API de RAWG (fecha de extracción desconocida). Al ser una foto fija de reviews acumuladas a lo largo de toda la historia del juego:
- No se puede particionar temporalmente (no hay información de cuándo se acumularon)
- Los juegos post-2017 que aparecen en el test ya tienen reviews acumuladas durante años
- La feature captura popularidad *acumulada*, no popularidad *futura*

**Decisión**: Excluir `rawg_ratings_count` de todos los modelos metodológicamente limpios (serie 20_xx en adelante).

---

## Script 21 — Experimentos Extendidos

**Archivo**: `21_extended_experiments.py`  
**Propósito**: Mejorar sobre 19_10 y explorar alternativas antes de abordar el leakage.

| ID | Modelo | R² Temporal | RMSE | R² TSCV | R² KFold |
|----|--------|-------------|------|---------|---------|
| **21_10_cat** | **CatBoost 60 trials** | **0.4473** | **979** | 0.671 | 0.651 |
| 19_10 | XGBoost (baseline) | 0.4384 | 987 | 0.674 | 0.634 |
| 21_10_opt | XGBoost 150 trials | 0.4258 | 998 | 0.696 | 0.600 |
| 21_10_feat | XGBoost + feat eng | 0.4137 | 1009 | 0.639 | 0.620 |
| 21_ensemble | Ensemble lin+log | 0.4035 | 1018 | N/A | N/A |
| 21_10_lgbm | LightGBM 80 trials | 0.3571 | 1056 | 0.621 | 0.581 |

Todos estos modelos incluyen `rawg_ratings_count` → **metodológicamente leakeados**.

CatBoost nuevo mejor con `rawg_ratings_count` en R²=0.4473.

---

## Script 20 (segunda pasada) — Pipeline sin rawg_ratings_count (20_xx)

**Archivo**: `18_full_v2_pipeline.py` con RAWG array reducido a 11 features (sin `log_ratings_count`)

### Resultados limpios (sin rawg_ratings_count)

| ID | Modelo | R² Temporal | RMSE | R² TSCV | R² KFold |
|----|--------|-------------|------|---------|---------|
| **20_10** | **Hybrid+RAWG** | **0.1474** | **1217** | 0.675 | 0.580 |
| 20_10_log | Hybrid+RAWG log | 0.1192 | 1237 | 0.646 | 0.613 |
| 20_13b | Hybrid+TF-IDF log | 0.1199 | 1236 | 0.633 | 0.611 |
| 20_11b_log | Hybrid+DevRep log | 0.1179 | 1237 | 0.640 | 0.608 |
| 20_05 | RS + Metadata | 0.1115 | 1242 | 0.663 | 0.645 |
| 20_08 | Hybrid Collab-Content | 0.1107 | 1242 | 0.639 | 0.559 |
| 20_11b | Hybrid + Dev Rep | 0.0850 | 1260 | 0.641 | 0.490 |
| 20_03 | RS Only | -0.009 | 1324 | 0.654 | 0.623 |
| 20_13a | RS Only log | -0.010 | 1324 | 0.457 | 0.398 |

**Impacto del leakage**: R² cae de 0.438 → 0.147 al quitar `rawg_ratings_count`. El modelo "limpio" más fuerte es 20_10 a R²=0.147.

Conclusión crítica: el RS puro (20_03) sigue en R²=-0.009. Los embeddings colaborativos estándar no contribuyen a la predicción temporal.

---

## Two-Tower Content-Augmented v1 (Script 22)

**Archivo**: `22_train_rs_content.py`  
**Motivación**: Los embeddings RS estándar (ID-only) no predicen temporal. Solución: combinar señal colaborativa + contenido en el mismo embedding mediante arquitectura Two-Tower extendida.

### Arquitectura v1
- Item tower: `Embedding(32d) + Content_MLP(54d→32d) → concat(64d) → Linear(64d) → LayerNorm`
- Content features: TF-IDF(50d) + 4 numeric = 54d
- Entrenamiento: lr=1e-3, patience=2, sin LR scheduling
- Early stopped epoch 4, val_loss=0.3095
- Output: `item_embeddings_rs_v2_content.npy`

### Resultados 22_xx (content-aug v1)

| ID | R² Temporal | RMSE | R² TSCV |
|----|-------------|------|---------|
| 22_05 | 0.1352 | 1225 | 0.528 |
| 22_10 | 0.1326 | 1227 | 0.514 |
| 22_10_log | 0.1329 | 1227 | 0.566 |
| 22_13b | 0.1200 | 1236 | 0.562 |
| 22_11b_log | 0.1307 | 1228 | 0.565 |
| 22_13a | 0.0928 | 1255 | 0.562 |
| 22_11b | 0.0622 | 1276 | 0.438 |
| 22_03 | -0.0931 | 1378 | 0.396 |
| 22_08 | -0.1041 | 1384 | 0.395 |

V1 fue un resultado mixto: los mejores modelos (22_05, 22_10) rondan R²≈0.13, similar a 20_10 limpio. 22_03 y 22_08 empeoran (embeddings de baja calidad por entrenamiento corto de 4 epochs). Causa probable: patience=2 muy agresivo + sin LR scheduling → Two-Tower no convergió suficientemente.

---

## Two-Tower Content-Augmented v2 (Script 23) — **MEJOR RESULTADO LIMPIO**

**Archivo**: `23_train_rs_content_v2.py`  
**Motivación**: Mejorar el Two-Tower content-aug con training más robusto.

### Mejoras sobre v1
1. **Cold-start dropout** (p=0.3): Durante training, zerear ID embedding con prob 30% → fuerza al content MLP a aprender representaciones independientes → embeddings útiles para nuevos items
2. **Cosine annealing con warmup**: 2 épocas de warmup + cosine decay. lr=5e-4, AdamW weight_decay=1e-4
3. **Content features expandidas**: TF-IDF(100d) + 4 numeric (price, ea, genres, tags) + 4 RAWG quality (rating, metacritic, playtime, esrb) = **108d total**
4. **Content MLP más profundo**: Linear(108→128)→GELU→Dropout(0.1)→Linear(128→64)→GELU→Linear(64→32)
5. **Patience=4** (vs 2), hasta 15 epochs

### Training v2
- Early stopped epoch 8 (vs epoch 4 en v1)
- val_loss=0.3261
- Outputs: `item_embeddings_rs_v2_content2.npy` (standard) + `item_embeddings_rs_v2_content2_cs.npy` (cold-start only)

### Resultados 23_xx — **Mejores resultados limpios del proyecto**

| ID | Modelo | R² Temporal | RMSE | R² TSCV | R² KFold |
|----|--------|-------------|------|---------|---------|
| **23_05** | **RS + Metadata** | **0.3676** | **1048** | 0.496 | 0.606 |
| 23_10 | Hybrid+RAWG | 0.3309 | 1078 | 0.479 | 0.614 |
| 23_08 | Hybrid Collab-Content | 0.2530 | 1139 | 0.319 | 0.518 |
| 23_03 | RS Only | 0.2451 | 1145 | 0.311 | 0.547 |
| 23_10_log | Hybrid+RAWG log | 0.1840 | 1190 | 0.566 | 0.529 |
| 23_11b_log | Hybrid+DevRep log | 0.1188 | 1237 | 0.578 | 0.595 |
| 23_13b | Hybrid+TF-IDF log | 0.1237 | 1233 | 0.560 | 0.530 |
| 23_11b | Hybrid+DevRep | 0.1424 | 1220 | 0.319 | 0.433 |
| 23_13a | RS Only log | 0.0802 | 1264 | 0.562 | 0.507 |

### Hallazgos clave de la serie 23

**1. Mejora dramática sobre embeddings estándar (respuesta a la pregunta central)**

| Comparación | R² Temporal | Delta |
|-------------|-------------|-------|
| RS Only estándar (20_03) | -0.009 | — |
| RS Only content-aug v2 (23_03) | +0.245 | **+0.254** |
| Hybrid+RAWG estándar limpio (20_10) | +0.147 | — |
| RS+Metadata content-aug v2 (23_05) | +0.368 | **+0.220** |

Los embeddings content-augmentados transforman el RS de no-predictivo a predictivo temporalmente.

**2. El embedding comprime mejor que la concatenación**

23_05 (RS content-aug 64d + metadata 6d = 70 features, R²=0.368) supera a 20_10 (estándar 64d + TF-IDF 100d + RAWG 11d + metadata 6d = 181 features, R²=0.147). La arquitectura Two-Tower aprendió una compresión más informativa que la concatenación manual.

**3. Añadir RAWG raw encima de content-aug HURTA**

23_10 (R²=0.331) < 23_05 (R²=0.368): agregar RAWG features externas cuando los embeddings ya las codifican internamente crea redundancia y overfitting. El Two-Tower comprimió esa información de forma más eficiente.

**4. RS Only content-aug (23_03) ya es competitivo**

R²=0.245 sin ningún feature adicional — prueba directa que el embedding per se captura señal predictiva de popularidad futura.

---

## Script 24 — Ablation Study + CatBoost (content-aug v2)

**Archivo**: `24_ablation_catboost.py`  
**Propósito**:  
- Establecer baseline de contenido puro (sin RS) para medir la contribución neta del Two-Tower  
- Aplicar CatBoost al mejor set de features limpio (24_05_cat)

### A) Ablation — modelos sin RS embeddings

| ID | Features | R² | RMSE | R² TSCV | R² KFold |
|----|----------|----|------|---------|---------|
| abl_04 | Meta (6d) + RAWG (11d) | **0.3137** | **1091** | 0.148 | 0.315 |
| abl_05 | TF-IDF (100d) + steam (2d) + meta (6d) + RAWG (11d) | 0.2777 | 1120 | 0.290 | 0.377 |
| abl_03 | TF-IDF (100d) + steam (2d) + RAWG (11d) | 0.2450 | 1145 | 0.280 | 0.359 |
| abl_01 | Metadata Only (6d) | 0.1108 | 1242 | 0.021 | 0.204 |
| abl_02 | TF-IDF (100d) + steam (2d) | 0.0820 | 1262 | 0.194 | 0.175 |

Hallazgo notable: TF-IDF raw sin RAWG (abl_02, R²=0.082) es PEOR que metadata sola (abl_01, R²=0.111). El TF-IDF no captura señal de popularidad — solo géneros y tags, que no correlacionan con reviews futuras. Sin RAWG, el contenido semántico aporta poco.

El mejor modelo sin RS es abl_04 (Meta+RAWG, 17 features, R²=0.314). Sorprendentemente, añadir TF-IDF encima de RAWG EMPEORA (abl_05, R²=0.278 < abl_04, R²=0.314) — confirmando que los tags no aportan señal de popularidad y añaden ruido.

### B) CatBoost — content-aug v2 embeddings

| ID | Features | R² | RMSE | R² TSCV | R² KFold |
|----|----------|----|------|---------|---------|
| **24_05_cat** | **RS content-aug v2 (64d) + Meta (6d)** | **0.4067** | **1015** | 0.522 | 0.638 |
| 24_10_cat | RS content-aug v2 (64d) + TF-IDF + steam + RAWG (177d) | 0.3378 | 1072 | 0.537 | 0.631 |

**24_05_cat con R²=0.407 es el nuevo mejor resultado metodológicamente limpio del proyecto.**

El patrón RAWG-externo-hurta se confirma también con CatBoost: 24_10_cat (0.338) < 24_05_cat (0.407). El Two-Tower ya codificó las 4 RAWG quality features internamente; concatenarlas de nuevo crea redundancia.

---

## Tabla Comparativa Resumen — Estado Final

| Enfoque | Mejor R² | RMSE | Modelo | ¿Limpio? |
|---------|----------|------|--------|----------|
| **CatBoost + content-aug Two-Tower v2** | **0.407** | **1015** | 24_05_cat | ✅ Sí |
| XGBoost + content-aug Two-Tower v2 | 0.368 | 1048 | 23_05 | ✅ Sí |
| Mejor contenido puro (sin RS) | 0.314 | 1091 | abl_04 | ✅ Sí |
| XGBoost + estándar Two-Tower + RAWG | 0.147 | 1217 | 20_10 | ✅ Sí |
| CatBoost (con rawg_ratings_count) | 0.447 | 979 | 21_10_cat | ❌ Leakeado |
| XGBoost (con rawg_ratings_count) | 0.438 | 987 | 19_10 | ❌ Leakeado |
| Metadata Only | 0.111 | 1242 | abl_01 | ✅ Sí |
| TF-IDF + steam (sin RS, sin RAWG) | 0.082 | 1262 | abl_02 | ✅ Sí |

---

## Ablation: Cuantificación de la Contribución del RS

| Comparación | R² | Delta RS |
|-------------|-----|---------|
| Metadata only (abl_01) | 0.111 | baseline |
| + RS content-aug (24_05_cat) | **0.407** | **+0.296** |
| — | — | — |
| Best content-only (abl_04: meta+RAWG) | 0.314 | baseline |
| + RS content-aug (24_05_cat) | **0.407** | **+0.093** |
| — | — | — |
| Full content manual (abl_05, 119d) | 0.278 | baseline |
| RS content-aug comprimido (23_05, 70d) | 0.368 | **+0.090** (¡con menos features!) |

El RS content-aug aporta **+0.093 a +0.296** de R² según la baseline de comparación. Incluso frente al mejor modelo de contenido puro (que ya tiene acceso a RAWG quality), el RS añade +0.093 R² / −76 RMSE.

---

## Respuesta a la Pregunta Central de la Tesis

> *¿Los embeddings de sistemas de recomendación mejoran la predicción de popularidad de videojuegos?*

**Respuesta final (con ablation)**:

- **Embeddings colaborativos puros** (ID-only Two-Tower): **NO** contribuyen (R²≈-0.009). Capturan patrones históricos de consumo pero no señal de popularidad futura.
- **Embeddings content-augmented** (Two-Tower con content tower + cold-start dropout): **SÍ**, de forma significativa y medible.

**Cuantificación limpia**:
- Mejor baseline sin RS (meta+RAWG): R²=0.314
- Mejor modelo con RS (CatBoost + RS content-aug + meta): R²=0.407
- **Aporte neto del RS: +0.093 R², −76 RMSE** (con el mismo acceso a RAWG quality)

**Por qué funciona**:  
El Two-Tower content-augmented aprende una representación conjunta de señal colaborativa (qué juegos consume el mismo tipo de usuario) y señal semántica (tags, géneros, calidad RAWG). Esta representación latente de 64 dimensiones comprime más información predictiva que la concatenación manual de las mismas features (abl_05, 119d, R²=0.278 < 23_05, 70d, R²=0.368). El cold-start dropout fuerza al content MLP a aprender representaciones robustas incluso para ítems nuevos.

**Limitación honesta**:  
El aporte del RS (~+0.09 sobre el mejor contenido-puro) es real pero moderado. La mayor parte del poder predictivo del modelo final proviene de las features de calidad externas (RAWG: metacritic, rating, playtime) codificadas en el embedding. El RS añade un boost adicional significativo pero no es el único factor determinante.

---

## Script 25 — Two-Tower Architecture Sweep

**Archivo**: `25_two_tower_search.py`  
**Propósito**: Identificar la configuración óptima del Two-Tower content-aug. En particular, responder: ¿el componente colaborativo (ID embedding) contribuye, o la señal proviene solo del contenido?

Cada configuración entrena el Two-Tower completo y luego evalúa con CatBoost downstream (X = emb + meta, mismo setup que 24_05_cat).

### Configuraciones testeadas

| Config | CS_DROP | EMB | Content | val_loss TT | R² | RMSE | TSCV |
|--------|---------|-----|---------|-------------|-----|------|------|
| **Baseline (23_/24_05_cat)** | **0.3** | **64** | **108d** | **0.326** | **0.407** | **1015** | **0.522** |
| cs10_pureContent | 1.0 | 64 | 108d | 0.424 | 0.318 | 1088 | 0.178 |
| cs0_collab | 0.0 | 64 | 108d | 0.802 | 0.211 | 1170 | 0.024 |
| cs05_aggressive | 0.5 | 64 | 108d | 0.582 | 0.174 | 1197 | 0.300 |
| emb128 | 0.3 | 128 | 108d | 0.736 | 0.157 | 1210 | 0.051 |
| noRawgInTower | 0.3 | 64 | 104d | 0.781 | 0.053 | 1282 | 0.056 |

**El baseline CS=0.3 sigue siendo el mejor en todas las métricas.**

### Hallazgos del sweep

**1. La señal colaborativa SÍ contribuye, pero requiere el dropout correcto**

CS=1.0 (pure content, sin ID embedding en inferencia) → R²=0.318  
CS=0.3 (collaborative + content, baseline) → R²=0.407  
→ **El componente colaborativo aporta +0.089 R² sobre el contenido puro.**

Pero sin dropout (CS=0.0) el resultado colapsa a R²=0.211 con val_loss=0.802: el ID embedding overfittea al patrón histórico colaborativo y no generaliza temporalmente. El cold-start dropout p=0.3 es el punto óptimo que evita ese overfitting.

**2. CS=1.0 ≈ meta+RAWG raw (sin Two-Tower)**

| Modelo | R² |
|--------|-----|
| abl_04: meta+RAWG raw (sin Two-Tower) | 0.314 |
| cs10_pureContent: Two-Tower CS=1.0 | 0.318 |

Son virtualmente idénticos. Esto demuestra que **la arquitectura Two-Tower per se no añade valor cuando no hay señal colaborativa** — simplemente re-aprende lo que los features de contenido ya tenían. La ventaja del CS=0.3 (R²=0.407) viene exclusivamente de la señal colaborativa integrada.

**3. RAWG quality en el tower es indispensable**

noRawgInTower (TF-IDF + 4 numeric, sin RAWG quality): R²=0.053 → colapso casi total  
Baseline con RAWG quality (rating, metacritic, playtime, esrb) en el tower: R²=0.407  
→ **Los 4 features RAWG dentro del content MLP son el ancla que permite al modelo aprender señal de calidad/popularidad.**

Sin ellos, el Two-Tower aprende a agrupar juegos por similitud de géneros/tags (señal de contenido puro), lo cual no correlaciona con popularidad futura.

**4. EMB=128 hurta (overfitting)**

128d con val_loss=0.736: más capacidad lleva a overfitting con este tamaño de dataset. El 64d es el sweet spot.

**5. CS=0.3 es el punto óptimo de un tradeoff claro**

| CS_DROP | Señal colaborativa | Riesgo overfitting | R² |
|---------|-------------------|-------------------|-----|
| 0.0 | Máxima | Alto (val=0.802) | 0.211 |
| 0.3 | Moderada | Bajo (val=0.326) | **0.407** |
| 0.5 | Baja | Moderado (val=0.582) | 0.174 |
| 1.0 | Nula | Bajo (val=0.424) | 0.318 |

El dropout actúa como regularizador: permite que el ID embedding capture señal colaborativa sin overfittear, mientras fuerza al content MLP a ser independientemente útil.

---

## Conclusión Final del Proyecto

### Tabla de todos los resultados limpios relevantes

| Modelo | R² | RMSE | Rol |
|--------|----|------|-----|
| **24_05_cat** | **0.407** | **1015** | Mejor limpio — RS content-aug (CS=0.3) + CatBoost |
| 23_05 | 0.368 | 1048 | Mejor limpio XGBoost |
| cs10_pureContent | 0.318 | 1088 | Two-Tower puro contenido (CS=1.0) |
| abl_04 | 0.314 | 1091 | Mejor sin RS (meta+RAWG raw) |
| abl_05 | 0.278 | 1120 | Full content raw sin RS (119d) |
| 20_10 | 0.147 | 1217 | Two-Tower estándar (ID-only) + RAWG |
| 20_03 | -0.009 | 1324 | Two-Tower estándar (ID-only) solo |

### Jerarquía de fuentes de señal (de mayor a menor impacto)

1. **RAWG quality features** (metacritic, rating, playtime, esrb) — sin esto, R²≈0 dentro del Two-Tower
2. **Cold-start dropout p=0.3** — permite integrar señal colaborativa sin overfitting
3. **Señal colaborativa del Two-Tower** (+0.089 R² sobre CS=1.0)
4. **Arquitectura Two-Tower vs concatenación manual** (CS=0.3 supera full-content-raw: 0.407 vs 0.278 con menos features)

### Respuesta definitiva a la hipótesis

El RS mejora la predicción de popularidad en **+0.093 R² sobre el mejor baseline de contenido puro** (0.407 vs 0.314). La fuente de esa mejora está en la señal colaborativa integrada mediante cold-start dropout — demostrado por el sweep: CS=1.0 (pure content) da R²≈0.318, igualando el baseline raw; solo con CS=0.3 se alcanza R²=0.407.

---

*Última actualización: 2026-05-22 (25_xx — sweep arquitecturas Two-Tower, conclusión definitiva)*

---

## Scripts 26-30 — Intentos de Mejora Post-Baseline

*Iniciado: 2026-05-23*

**Motivación**: Tras establecer el baseline óptimo (24_05_cat, R²=0.407), se intentaron cuatro mejoras al Two-Tower: (1) multi-task con cabeza de popularidad, (2) InfoNCE loss, (3) user content features, (4) cutoff 2017. Se documentan los resultados y los bugs encontrados — relevantes para entender los límites de la arquitectura.

### Script 26 — Todas las mejoras juntas (FALLÓ)

**Archivo**: `26_train_rs_improved.py`  
Mejoras combinadas: InfoNCE (temperature=0.1) + multi-task + user content features + cutoff 2017  
**val_nce**: 10.0 → 9.7 (casi aleatorio en todas las épocas)  
**Resultado**: std R²=-0.045, cold-start R²=0.119

**Causa del fallo**: InfoNCE con temperature=0.1 es demasiado agresivo. Las distribuciones sharpeadas con temperatura baja hacen que el modelo overfittee el grafo colaborativo de entrenamiento sin generalizar temporalmente. La user content tower (promedio de content features de ítems interactuados) creó ventaja de training que no generalizó.

---

### Scripts 27-30 — Debug iterativo del multi-task + cutoff 2017

Estos scripts aíslan y diagnostican los bugs de la implementación multi-task para entender por qué falla.

#### Script 27 — BCE + multi-task (split cronológico, doble forward pass)

**Archivo**: `27_train_rs_multitask.py`  
BCE + multi-task (λ=0.2) + cutoff 2017  
**Comportamiento**: val_loss sube desde epoch 2 (0.587 → 1.303 → ...)  
**Resultado**: std R²=0.136, cs R²=0.209 / TSCV=0.854 (sospechoso)

**Bug 1 encontrado — double forward pass**:  
El training loop llamaba `model()` dos veces por batch — una vez para positivos (para extraer `item_emb_pos`) y otra para pos+neg (para BCE). Al hacer `backward()`, todos los parámetros recibían gradientes dobles → LR efectivo 2× para user embeddings → overfitting acelerado.

**Bug 2 encontrado — split cronológico**:  
Split `val_cut = int(n * 0.9)` sin shuffle. El parquet ordenado pone en validación a usuarios que pueden no aparecer en el 90% de training → embeddings de usuario random → val_loss se dispara desde epoch 2.

---

#### Script 28 — Fix double forward pass (split cronológico persiste)

**Archivo**: `28_train_rs_multitask_fixed.py`  
Fix Bug 1 (single forward pass) + gradient clipping (norm=1.0)  
**Comportamiento**: val_loss sigue subiendo desde epoch 2  
**Resultado**: std R²=0.048, cs R²=0.014 (peor que 27_)

Confirma que Bug 2 (split cronológico) es el root cause del val_loss explosivo.

---

#### Script 29 — Fix split aleatorio + multi-task (leakage en pop_head)

**Archivo**: `29_train_rs_multitask_v2.py`  
Fix Bug 1 + Fix Bug 2 (`train_test_split` aleatorio, igual que 23_) + gradient clipping  
**Training**: Converge correctamente. val_loss: 0.553 → 0.407 → 0.365 → **0.356** → sube. Early stop epoch 8.  
**Resultado**: std R²=-0.013 / **TSCV=0.940** / cs R²=-0.002

**Bug 3 encontrado — leakage indirecto en pop_head**:  
`y_log_norm` = z-score(log1p(y_full)), donde `y_full` son interacciones TOTALES (incluyendo post-2017). El pop_head entrena los item embeddings de ítems pre-2017 a predecir su popularidad total futura. CatBoost downstream ve embeddings que encodifican implícitamente `y_full` → TSCV altísimo (todos los ítems de training tienen ese signal) pero temporal≈0 (ítems post-2017 nunca pasaron por el pop_head → sus embeddings no encodifican nada).  
El TSCV=0.940 vs temporal R²=-0.013 es la firma exacta de este tipo de leakage.

---

#### Script 30 — BCE puro + cutoff 2017 (aislando el efecto del cutoff)

**Archivo**: `30_train_rs_cutoff2017.py`  
Fix Bug 1 + Fix Bug 2 + sin multi-task (LAMBDA_POP=0)  
**Training**: Converge. val_loss: 0.553 → 0.411 → 0.364 → **0.343** → sube. Early stop epoch 8.  
**Resultado**: std R²=-0.011, cs R²=0.138

Sin el leakage del pop_head, el TSCV baja a 0.849 (desde 0.940) pero temporal sigue siendo ≈0 para std.

**Hallazgo clave — el cutoff 2017 perjudica**:

| Cutoff TT | Ítems con collab. emb. en XGB train | Ítems cold-start en XGB train | R² temporal |
|-----------|------------------------------------|-----------------------------|-------------|
| 23_ (pre-2016) | 6,953 (65%) | 3,621 (35%) — año 2016 | **0.407** |
| 30_ (pre-2017) | **10,574 (100%)** | 0 (0%) | -0.011 |

Con cutoff 2016, los ítems de 2016 en el training del XGBoost tienen embeddings cold-start (ID embedding random ≈ 0, solo content MLP). CatBoost entrena con una **mezcla de ítems con señal colaborativa y cold-start** → aprende a depender de features de contenido → generaliza bien a los ítems post-2017 (también cold-start en test).

Con cutoff 2017, **todos** los ítems de training del XGBoost tienen embeddings colaborativos entrenados. CatBoost explota esa señal → excelente TSCV/KFold (dentro del período con señal colaborativa) → colapso en temporal cuando los ítems post-2017 llegan como cold-start sin esa señal.

**El diseño de 23_ (cutoff 2016) resultó óptimo de forma no obvia**: el gap de un año entre el cutoff del Two-Tower y el cutoff del XGBoost crea regularización natural en el training del CatBoost, forzándolo a aprender representaciones que generalizan a cold-start items.

---

### Resumen de scripts 26-30

| Script | Cambio clave | Bug encontrado | val_loss | R² temporal std |
|--------|-------------|----------------|---------|-----------------|
| 26_ | InfoNCE + multi-task + cutoff 2017 | InfoNCE temp=0.1 overfittea grafo | 9.7 (nce) | -0.045 |
| 27_ | BCE + multi-task + cutoff 2017 | Double fwd + split cron. | ↑ desde ep.2 | 0.136 |
| 28_ | + fix double fwd | Split cron. | ↑ desde ep.2 | 0.048 |
| 29_ | + fix split random | Leakage pop_head (y_full) | 0.356 ✓ | -0.013 |
| 30_ | BCE puro (sin pop_head) | Cutoff 2017 rompe distribución XGB | 0.343 ✓ | -0.011 |
| **23_** | **BCE + cutoff 2016 (baseline)** | — | **0.326** | **0.407** |

**Conclusión**: Ninguna de las mejoras intentadas supera el baseline 23_/24_05_cat. El baseline resultó ser el diseño óptimo. Los experimentos 26-30 refuerzan el entendimiento del sistema:

1. InfoNCE con temperatura agresiva es contraproducente en este dominio
2. El cutoff del Two-Tower debe ser menor que el cutoff del XGBoost (gap de un año: 2016 vs 2017)
3. Un pop_head con target que incluye el futuro introduce leakage indirecto
4. El split de validación del Two-Tower debe ser aleatorio, no cronológico

*Última actualización: 2026-05-23 (scripts 26-30 — intentos de mejora y diagnóstico)*

---

## Script 31 — Log-transform del Target (CatBoost)

**Archivo**: `31_log_transform.py`  
**Propósito**: Verificar si el log-transform del target (`log1p(y)`) mejora el resultado principal (24_05_cat) y los baselines clave. La distribución del target es muy sesgada (mean=504, max=183,649), lo que podría hacer que el RMSE en escala original esté dominado por outliers.

**Setup**: Mismo pipeline que 24_ pero con `y_train = log1p(y_train)`. Las predicciones se back-transforman con `expm1()` para reportar métricas en escala original. Se evalúan tres modelos:
1. **24_05_cat_log**: RS content-aug v2 (64d) + Meta (6d) = 70 features (el modelo principal)
2. **abl_04_log**: Meta (6d) + RAWG (11d) = 17 features (mejor baseline sin RS)
3. **abl_01_log**: Meta only (6d) (baseline más simple)

### Resultados

| Modelo | R²(orig) | RMSE(orig) | R²(log) | RMSE(log) |
|--------|----------|-----------|---------|----------|
| 24_05_cat_log | 0.1547 | 1211.34 | 0.4102 | 1.2747 |
| abl_04_log | 0.2454 | 1144.51 | 0.2420 | 1.4452 |
| abl_01_log | 0.1540 | 1211.82 | 0.0121 | 1.6498 |

**Referencia sin log-transform**:

| Modelo | R²(orig) | RMSE(orig) |
|--------|----------|-----------|
| 24_05_cat | **0.4067** | **1014.86** |
| abl_04 | 0.3137 | 1091.48 |
| abl_01 | 0.1108 | 1242.41 |

### Conclusión: log-transform HURTA en escala original

El log-transform colapsa el R² en escala original del modelo principal de **0.407 → 0.155**. El RMSE sube de 1015 → 1211. El resultado es el mismo patrón observado en Fase 1 (script 13c vs 12) y en el pipeline v2 (18_log vs 18 lineal).

**Por qué ocurre**: Al entrenar en escala log, el modelo aprende a predecir `log1p(y)`. El back-transform `expm1(ŷ)` amplifica cualquier error en la zona de valores altos: un error de +1 en log-space corresponde a ×e ≈ ×2.7 en escala original. Con outliers extremos (y_max=183,649), los errores amplificados dominan el RMSE y colapsan el R².

**El R²_log=0.410 del modelo principal (log-space)** es comparable al R²=0.407 sin log — confirma que la capacidad predictiva del modelo es idéntica; solo cambia el espacio de métricas. El log no añade información, solo redistribuye los errores.

### Veredicto final

Log-transform descartado como métrica principal. Se mantiene la escala original (R²=0.407, RMSE=1015) como resultado definitivo del proyecto. El script 31 sirve como robustness check que confirma la solidez del baseline sin transformar.

*Última actualización: 2026-05-27 (script 31 — log-transform CatBoost, conclusión definitiva)*

---

# AUDITORÍA DE ROBUSTEZ (Scripts 32–39)

*Realizada: 2026-06-11/12*

Antes de la entrega se hizo una auditoría completa del resultado principal
(24_05_cat, R²=0.407): leakage en metadata, varianza estocástica, fixes
metodológicos del Two-Tower y barras de error en ambos lados de la comparación
RS vs contenido. **El resultado de la auditoría revisa las conclusiones del
proyecto** — ver "Conclusión Final Revisada" al final.

Infraestructura nueva: `v2_data.py` (módulo compartido de datos/features/eval,
agrega Spearman a las métricas) y `tower_v3.py` (entrenamiento del Two-Tower
parametrizado: cs_drop, seed, neg_mode, val_mode, fit_scope).

---

## Script 32/32b — Auditoría de leakage en metadata

**Archivos**: `32_audit_has_senti.py`, `32b_audit_followup.py`

**Disparador**: el campo `sentiment` de steam_games.json (snapshot ~2018)
incluye valores como "1 user reviews", "2 user reviews" — codifica el conteo de
reviews acumuladas. `has_senti` (sentiment no-nulo) está en los 6d de metadata
del modelo principal y para items de test 2017+ es información posterior al cutoff.

### Resultados

| Check | Resultado |
|-------|-----------|
| SHAP de has_senti en 24_05_cat | **0.000 exacto, rank 70/70** — el modelo nunca lo usó |
| Re-tune sin has_senti (24_05_cat_clean) | R²=0.3404 (−0.066 vs 0.4067) |
| Mismos hiperparámetros, 69d | R²=0.3512 (predicciones difieren: el RNG de CatBoost cambia con el nro de columnas) |
| Feature #1 del modelo | **num_tags: SHAP=159.9 (12.8% del total)** — soft-leakage (los tags se acumulan; snapshot 2018) |
| Sin num_tags (24_05_cat_no_nt) | R²=0.3060 |
| Meta estricta sin canales snapshot (24_05_cat_strict) | R²=0.3004 |

### Conclusiones del 32/32b
1. El modelo principal **no explotó** has_senti (SHAP=0) — la caída al re-tunear
   es varianza de Optuna/CatBoost, no efecto causal.
2. Esa varianza (~±0.05 R² al cambiar una columna inerte) pasó a ser el hallazgo
   central: las comparaciones finas del proyecto necesitan barras de error.
3. num_tags es soft-leakage real que el modelo SÍ usa; su costo de ablación
   (≤0.04) queda dentro del ruido.

---

## Script 33/33b — Variantes downstream (todas con meta5, sin has_senti)

**Archivos**: `33_catboost_variants.py`, `33b_ensemble_fix.py`

| Variante | R² | RMSE | Spearman | Lectura |
|----------|-----|------|----------|---------|
| 33_ens (peso elegido en test) | 0.4378 | 987.86 | 0.524 | ❌ Leak de selección |
| **33b_ens (peso en validación)** | **0.3765** | 1040.35 | 0.464 | Versión honesta |
| 33_metascore (+metascore Steam) | 0.4220 | 1001.65 | 0.513 | Mejora aparente (1 corrida) |
| 33_devrep (+dev_rep temporal limpia) | 0.3174 | 1088.58 | 0.534 | Sin ganancia clara |
| 33_tweedie | 0.2451 | 1144.75 | **0.5727** | Mejor Spearman del proyecto |
| 33_poisson | −0.0150 | colapsó | — | Var=media no banca sobredispersión |
| 33_meta_plus (metascore+devrep) | 0.2534 | 1138.43 | 0.477 | Más features = peor |
| 33_mixed (std train / cs test) | 0.0762 | 1266.32 | 0.398 | Mide mismatch, no deployment |
| 33b_all_cs (cs para todos) | 0.1910 | 1185.02 | **0.5467** | Deployment honesto: ranking casi intacto |

Notas:
- `33b_all_cs` es la evaluación deployment-honesta (juegos nuevos solo tienen
  content MLP): el R² cae pero el **ranking se mantiene** (ρ=0.547).
- Tweedie optimiza otro trade-off: pierde R² (escala), gana ranking y
  within-distribution (TSCV 0.61, KFold 0.68).
- dev_rep ahora se computa LIMPIA (`v2_data.build_dev_rep`): solo juegos
  pre-2017 del developer, excluyendo el propio juego por item_idx
  (fix del bug de 18_ donde `train_mask[idx] or True` era siempre True).

---

## Script 34 — Two-Tower v3: fixes metodológicos

**Archivos**: `tower_v3.py`, `34_train_rs_content_v3.py`

Tres fixes sobre 23_:
1. **TF-IDF del tower fiteado solo en items pre-2016** (23_ fiteaba sobre los
   15,380 incluyendo test: fit transductivo). Ídem z-score de columnas numéricas.
2. **val_loss con negativos sampleados** (23_ usaba BCE solo de positivos —
   criterio de checkpoint degenerado).
3. **Negative sampling restringido a items pre-2016** (en 23_ los items de test
   aparecían como negativos).

También se corrigió un bug de parseo de fechas (pandas 2.x column-wise infiere
un solo formato y coercea el resto a NaT: 2,669 vs 6,953 items pre-2016 —
verificado contra la tabla de 30_).

### Resultados (seed 42)

| Modelo | R² | TSCV | KFold | Spearman |
|--------|-----|------|-------|----------|
| 24_05_cat_clean (v2, emb originales) | 0.340 | 0.434 | 0.554 | 0.404 |
| **34_05_v3** (fixes) | 0.223 | **0.690** | **0.754** | 0.494 |
| 34_05_v3w (negativos pop^0.75) | 0.096 | 0.506 | 0.568 | 0.371 |

### Hallazgos
1. **Los negativos ponderados por popularidad LASTIMAN**: usar juegos populares
   como negativos enseña al tower a descartar la señal de popularidad — exactamente
   lo que el downstream necesita.
2. **El val-loss "degenerado" de 23_ era una feature accidental**: seleccionaba
   checkpoints cuyos item embeddings puntúan alto contra la masa de usuarios —
   embeddings que codifican popularidad global. El criterio limpio selecciona
   discriminación de recomendación → mejor within-distribution (TSCV/KFold récord),
   peor temporal puntual.

---

## Script 35 — Sweep fino de CS_DROP (v3)

**Archivo**: `35_cs_drop_fine_sweep.py`

| CS_DROP | R² | Spearman | TSCV |
|---------|-----|----------|------|
| 0.20 | 0.211 | 0.479 | 0.752 |
| 0.25 | 0.209 | 0.513 | 0.782 |
| 0.30 (=34_05_v3) | 0.223 | 0.494 | 0.690 |
| 0.35 | 0.219 | 0.488 | 0.761 |
| 0.40 | 0.230 | 0.534 | 0.779 |

**Conclusión**: meseta plana en [0.20, 0.40] — todas las diferencias dentro del
ruido (σ≈0.017). El "óptimo 0.3" del sweep grueso de 25_ no es un pico; el
contraste real es solo contra los extremos (0.0 y 1.0).

---

## Script 36 — Estabilidad multi-seed ⭐ (el experimento crítico)

**Archivo**: `36_multiseed_stability.py`  
5 seeds {42, 123, 777, 2024, 31337} × 2 configuraciones, tuning seed fijo (42):

| Config | R² temporal | Spearman | TSCV |
|--------|-------------|----------|------|
| v2rep (réplica setup 23_: uniform_all + val legacy, fit pre-2016) | **0.2365 ± 0.0216** [0.21, 0.25] | 0.442 ± 0.036 | 0.520 |
| v3 (fixes completos) | **0.2193 ± 0.0172** [0.20, 0.24] | **0.496 ± 0.009** | **0.693** |

**El 0.407 publicado (y el 0.340 limpio) NO están dentro de la distribución
multi-seed.** Además:
- Promediar embeddings entre seeds NO funciona (espacios rotados no alineados):
  v3 emb_avg R²=0.157, peor que cualquier seed individual.
- El Spearman de v3 es notablemente estable (±0.009) — el entrenamiento limpio
  produce calidad de ranking consistente.

---

## Script 38 — Diagnóstico del gap: ¿fit transductivo o suerte?

**Archivo**: `38_diagnose_v2_gap.py`  
Config "v2exact" = réplica EXACTA de 23_ (TF-IDF y z-score sobre todos los
items + uniform_all + val legacy):

| Corrida | R² |
|---------|-----|
| v2exact seed 42 | **0.3404** — idéntico a 4 decimales a 24_05_cat_clean (RMSE 1070.02, MAE 174.65, ρ 0.4038) |
| v2exact seed 123 | **0.1518** |

**Veredicto**:
1. La reproducción exacta del seed 42 **valida bit-a-bit la infraestructura de
   réplica** — los números del multi-seed son confiables.
2. El fit transductivo NO da ventaja sistemática (media entre seeds ≈ la de
   v2rep); lo que produce es varianza enorme (0.15 ↔ 0.34 cambiando el seed).
3. **El 0.340/0.407 original fue una realización afortunada del tower**, amplificada
   por el tuning afortunado con has_senti.

---

## Script 37 — Sentence embeddings (MiniLM) en el content tower

**Archivo**: `37_sentence_embeddings.py`  
all-MiniLM-L6-v2 (384d) sobre texto de tags+géneros, en lugar de TF-IDF (100d).

| Modelo | R² | Spearman | TSCV | KFold |
|--------|-----|----------|------|-------|
| 37_minilm | 0.195 | 0.440 | 0.756 | 0.787 |
| 34_05_v3 (TF-IDF, referencia) | 0.223 | 0.494 | 0.690 | 0.754 |

**Conclusión**: equivalente dentro del ruido. El cuello de botella no es la
representación semántica de los tags — resultado negativo documentable.

---

## Script 39 — Barras de error del baseline + descomposición de varianza

**Archivo**: `39_baseline_errorbars.py`  
5 tuning seeds (Optuna TPE + CatBoost random_seed) × 2 modelos:

| Modelo | R² temporal | Spearman | TSCV |
|--------|-------------|----------|------|
| Contenido-puro (meta5 + RAWG, 16d) | **0.2742 ± 0.0369** [0.21, 0.30] | 0.516 ± 0.021 | 0.150 ± 0.097 |
| RS v3-s42 fijo + meta5 (69d) | **0.2277 ± 0.0554** [0.15, 0.31] | 0.494 ± 0.021 | **0.699 ± 0.061** |

### Descomposición de varianza del RS
- Varianza por seed del tower (36_, tuning fijo): σ = 0.017
- Varianza por seed de tuning (39_, tower fijo): σ = **0.055**
- **El tuning downstream (Optuna/CatBoost) es la fuente dominante de varianza**,
  no el entrenamiento del tower.

---

## Conclusión Final REVISADA del Proyecto

### Tabla maestra (con barras de error)

| Modelo | R² temporal | Spearman | TSCV | KFold |
|--------|-------------|----------|------|-------|
| Contenido-puro (meta5+RAWG) | **0.274 ± 0.037** | **0.516 ± 0.021** | 0.150 ± 0.097 | 0.33 |
| RS content-aug v3 + meta5 | 0.22 ± 0.02 (tower) / ± 0.06 (tuning) | 0.494 ± 0.021 | **0.699 ± 0.061** | **0.75** |
| RS all-cold-start (deployment) | 0.191 | 0.547 | 0.345 | 0.317 |
| ~~24_05_cat (publicado)~~ | ~~0.407~~ → realización afortunada + tuning con has_senti | | | |

### Respuesta revisada a la pregunta central

> *¿Los embeddings de sistemas de recomendación mejoran la predicción de
> popularidad de videojuegos?*

**Depende del régimen — y la respuesta es ahora estadísticamente honesta:**

1. **Extrapolación temporal (juegos 2017+ nunca vistos): NO.** Las bandas de
   RS (0.23±0.06) y contenido-puro (0.27±0.04) se solapan; el contenido va
   incluso levemente arriba. La ventaja temporal reportada originalmente
   (+0.09) era una combinación de realización afortunada del tower y varianza
   de tuning. Esto es coherente con la Fase 1: el cold-start es estructural —
   la mitad colaborativa del embedding no puede ayudar a juegos sin historial.

2. **Within-distribution (juegos con historial de interacciones): SÍ,
   rotundamente.** TSCV 0.699±0.061 vs 0.150±0.097 (×4.7, bandas sin solape) y
   KFold 0.75 vs 0.33. Para juegos del catálogo, la señal colaborativa es
   transformadora.

3. **Ranking (Spearman)**: empate técnico (~0.50 ambos) en temporal; el RS
   deployment-honesto (all-cs) mantiene ρ=0.547 — para decisiones de
   priorización el sistema es útil incluso en cold-start.

### El sistema de dos etapas, ahora a escala global

La conclusión de la Fase 1 (australiana) se confirma en el dataset global con
rigor estadístico: **RS para el catálogo con historial, contenido para
lanzamientos nuevos**. La arquitectura content-augmented no "resuelve" el
cold-start — empaqueta la señal de contenido dentro del embedding (útil para
servir un solo vector), pero no agrega información colaborativa donde no la hay.

### Lecciones metodológicas (aporte de la tesis)

1. **Una corrida no es un resultado**: la varianza estocástica total (~±0.06 R²)
   superaba el efecto que se quería medir (+0.09). Multi-seed + barras de error
   son obligatorios en pipelines embedding→GBM.
2. **La fuente de varianza dominante es el tuning downstream**, no el deep
   learning — donde la intuición suele apuntar al revés.
3. **Tres capas de leakage detectadas y cuantificadas**: rawg_ratings_count
   (duro, −0.29 R²), fit transductivo de TF-IDF/z-score (varianza, no sesgo),
   metadata snapshot (has_senti inerte, num_tags ≤0.04).
4. **Criterios de model selection importan**: el val_loss sin negativos de 23_
   seleccionaba accidentalmente embeddings que codifican popularidad.
5. **R² temporal con outliers extremos es una métrica frágil**; Spearman es
   estable (±0.01-0.02) y debería acompañar siempre.

*Última actualización: 2026-06-12 (scripts 32-39 — auditoría de robustez y conclusión revisada)*
