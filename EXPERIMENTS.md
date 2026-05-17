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
| `01_data_preprocessing.ipynb` | Limpieza de datos Steam, australianos, RAWG |
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

*Última actualización: 2026-05-17 (post-fix: TF-IDF leakage + Optuna sort + R² metric space)*
