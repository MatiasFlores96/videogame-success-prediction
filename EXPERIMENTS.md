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

### Leakage Temporal en Two-Tower (encontrado y corregido)
El Two-Tower original se entrenaba con TODAS las interacciones incluyendo juegos post-2016.  
Esto inflaba los embeddings de los juegos del test set con señal de sus propias interacciones.

**Fix**: Se filtró el entrenamiento del Two-Tower a solo interacciones de juegos pre-2016 (`_cutoff = 2016-01-01`).  
Notebook: `02_model_keras_rs.ipynb` — celda con filtro anti-leakage.

**Impacto**: Los resultados del README original (R²≈0.79) eran inflados. Los resultados reales post-fix están en esta tabla.

### Bug de Archivo Stale
`item_embeddings_rs_clean.npy` era una versión antigua del 03/03/2026 (pre-fix).  
**Fix**: Sobreescrito con el embedding post-fix. Notebook 02 actualizado para guardar ambos archivos.

---

## Tabla de Resultados Completa

| ID  | Modelo                        | Features                                          | R² Temporal | RMSE  | R² TSCV | Nota |
|-----|-------------------------------|---------------------------------------------------|-------------|-------|---------|------|
| 03  | RS Embeddings Only            | RS clean (64d)                                    | -0.027      | 57.66 | 0.770   | Baseline RS |
| 04  | Metadata Only                 | Metadata Steam (6d)                               | **+0.065**  | 55.00 | -0.516  | Mejor temporal lineal |
| 05  | RS + Metadata                 | RS (64d) + Metadata (6d)                          | -0.024      | 57.57 | 0.760   | |
| 06  | Review Text Emb               | Review text BERT (384d)                           | ~0.000      | 56.90 | -0.117  | |
| 07  | Tag Embeddings                | Tag embeddings (384d)                             | -127.24     | 65.09 | -22.55  | Muy inestable |
| 08  | Hybrid Collab-Content         | RS (64d) + TF-IDF (100d) + Numeric (2d)           | -0.024      | 57.57 | **0.841** | Mejor TSCV |
| 09  | RS + Review Text              | RS (64d) + Review text (384d)                     | -0.022      | 57.53 | 0.680   | |
| 10  | RS + Reviews + RAWG           | RS (64d) + Review (384d) + RAWG (12d)             | -0.017      | 57.38 | 0.767   | |
| 11a | RS + Dev Reputation           | RS (64d) + dev_rep (1d)                           | -0.025      | 57.61 | -0.808  | dev_rep rompe TSCV (leakage interno) |
| 11b | Hybrid + Dev Reputation       | RS (64d) + TF-IDF (100d) + Num (2d) + dev_rep    | -0.024      | 57.57 | 0.722   | Mínima diferencia |
| 12  | Two-Stage: Content Model      | TF-IDF (100d) + Steam (2d) + RAWG (12d) + dev_rep| **+0.076**  | 54.69 | -5.03   | Stage 2 del sistema dos-etapas |
| 13a | RS Only (log target)          | RS clean (64d) [log(1+y)]                         | -0.386 (log) | 1.150 | 0.932 (log) | R²_orig=-0.027, cold-start no mejora |
| 13b | Hybrid (log target)           | RS+TF-IDF+Numeric [log(1+y)]                      | -0.299 (log) | 1.113 | 0.928 (log) | R²_orig=-0.025, cold-start no mejora |
| 13c | Stage2 Content (log target)   | TF-IDF+Steam+RAWG+dev_rep [log(1+y)]              | **+0.254 (log)** | 0.844 | 0.757 (log) | **Mejor temporal del proyecto** |

> Todos los R²_temporal negativos indican predicción peor que predecir la media — producto del cold-start.

---

## Hallazgos Clave

### 1. El Cold-Start Degrada los RS Embeddings
Los juegos post-2016 (486 test items) no aparecen en el entrenamiento del Two-Tower.  
Sus embeddings son vectores aleatorios inicializados pero no entrenados.  
→ El XGBoost no puede extraer señal útil de ruido aleatorio.  
→ Todos los modelos con RS embeddings obtienen R²≈-0.02 en temporal.

### 2. El Espacio Colaborativo es Ortogonal al de Contenido
Experimento en `02b_cold_start_embeddings.ipynb`:
- Similaridad coseno en espacio de features (tags, géneros): **0.987** entre vecinos más cercanos
- Similaridad coseno en espacio de embeddings RS: **0.064** para los mismos juegos
- k-NN (k=10) para inferir embeddings RS desde contenido: R²CV = -0.085 (peor que no hacer nada)

**Implicación**: No existe mapping aprendible entre contenido y embeddings colaborativos con los datos disponibles.

### 3. RS Embeddings Funcionan Bien Within-Distribution
TSCV evalúa dentro del régimen pre-2016 donde todos los juegos tienen embeddings entrenados.  
- Modelo 08: TSCV R² = **0.841**
- Modelo 03: TSCV R² = **0.770**
- Metadata Only: TSCV R² = **-0.516** (embeddings son superiores dentro de distribución)

La hipótesis de la tesis se **confirma within-distribution** y **falla out-of-distribution** por cold-start.

### 4. Developer Reputation como Feature
`developer_reputation.npy`: promedio histórico de reviews de juegos pre-2016 por desarrollador.
- 2217 / 3682 juegos tienen reputación > 0
- Mean = 20.36 reviews, top developer = Hidden Path Entertainment (1881)
- Resultado: No mejora temporal R² significativamente (+0.002 sobre baseline)
- **Problema**: La dev_rep calculada sobre todos los juegos pre-2016 introduce leakage en TSCV (usa info futura de cada ventana)

### 5. Sistema de Dos Etapas
**Stage 1** (juegos con historial): RS embeddings → TSCV R² = 0.841 (lineal), 0.932 (log-target)  
**Stage 2** (juegos nuevos): TF-IDF + RAWG + Steam + dev_rep → temporal R²_log = **0.254**  

Stage 2 supera el baseline de metadata pura (+0.011 sobre Modelo 04 en lineal; +0.189 en log-space).  
El RAWG contribuye principalmente via **metacritic**, **playtime** y **rawg_ratings_count**.

### 6. Log-Transform del Target
Aplicar log(1+y) al target mejora considerablemente la calidad de la predicción:
- **RS models (Stage 1)**: TSCV R² sube de 0.841 → 0.932 con log-target
- **Stage 2 content**: temporal R²_log = +0.254 (vs +0.076 lineal) — el mejor resultado temporal del proyecto
- **Cold-start no se resuelve**: los modelos RS siguen siendo negativos en temporal (log-space), porque el problema es de embeddings sin entrenar, no de escala del target
- **Consistencia**: 13c TSCV R²=0.757 ≈ KFold R²=0.724 → sin overfitting, el modelo generaliza bien

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

**Argumento central (sostenible)**:
> Los embeddings de CF capturan señal colaborativa que supera a las features de contenido *dentro de distribución* (TSCV R²=0.84 vs -0.52 para metadata). Sin embargo, ante el problema de cold-start para juegos nuevos, esta ventaja desaparece. Un sistema de dos etapas que use CF para juegos conocidos y contenido enriquecido para nuevos representa el mejor compromiso práctico.

**Limitaciones honestas a mencionar**:
- El target (`total_reviews`) es una muestra sesgada: usuarios australianos que escribieron reviews, no ventas globales
- La distribución del target es extremadamente sesgada (la mayoría de juegos tienen <20 reviews, máximo 3759)
- El cold-start es inevitable con datos colaborativos puros: los juegos nuevos no tienen interacciones
- El período de test (post-2016) coincide con un crecimiento explosivo de Steam, cambiando la distribución subyacente

---

*Última actualización: 2026-05-17*
