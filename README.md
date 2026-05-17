# Video Game Popularity Prediction — Steam Dataset

Predicción de popularidad de videojuegos en Steam comparando distintas estrategias de feature engineering: señales colaborativas (RS embeddings), contenido (metadata, NLP) e híbridos con datos externos.

Proyecto de tesis de Maestría en Ciencia de Datos.

---

## Hipótesis Central

> Los embeddings de un sistema de recomendación Two-Tower (CF) usados como features para XGBoost mejoran la predicción de popularidad de videojuegos frente a enfoques solo-contenido.

**Resultado:** La hipótesis se **confirma within-distribution** (TSCV R²=0.841 con RS vs −0.516 sin RS) y **falla out-of-distribution** por el problema de cold-start inevitable en sistemas colaborativos puros.

---

## Dataset

**Australian Gaming Dataset (Steam)**
- 3.682 juegos
- 25.458 usuarios
- 59.305 interacciones

**Target:** `total_reviews` — número de reviews dejadas en el dataset australiano (proxy de popularidad/ventas).  
Distribución extremadamente sesgada: mayoría de juegos con <20 reviews, máximo 3.759.

---

## Metodología

**Evaluación con validación temporal estricta:**

| Esquema | Descripción | Propósito |
|---------|-------------|-----------|
| **Temporal split** | Train = pre-2016 (2621 juegos), Test = 2016+ (486 juegos) | Simula predicción de juegos futuros — métrica principal |
| **TSCV** | 5 ventanas rolling dentro de pre-2016 | Valida within-distribution (juegos con historial conocido) |
| **K-Fold** | 5 folds estándar | Referencia de varianza |

**Base learner:** XGBoost con Optuna (50 trials, TPE sampler, seed=42)

---

## Corrección Crítica Aplicada

### Leakage Temporal en Two-Tower
El Two-Tower original se entrenaba con TODAS las interacciones, incluyendo juegos post-2016. Esto inflaba los embeddings del test set con señal de sus propias interacciones futuras.

**Fix:** El entrenamiento del Two-Tower se filtró a interacciones de juegos pre-2016 únicamente (`cutoff = 2016-01-01`). Los resultados del README previo (R²≈0.79) eran inflados. Todos los resultados en esta versión son post-fix.

Ver detalles completos en [EXPERIMENTS.md](EXPERIMENTS.md).

---

## Resultados

### Modelos Lineales (target: reviews absolutas)

| ID | Modelo | Features | R² Temporal | RMSE Temporal | R² TSCV | RMSE TSCV |
|----|--------|----------|:-----------:|:-------------:|:-------:|:---------:|
| 03 | RS Embeddings Only | RS clean (64d) | −0.018 | 57.42 | 0.886 | 14.28 |
| 04 | Metadata Only | Steam metadata (6d) | +0.057 | 55.24 | −0.214 | 39.44 |
| 05 | RS + Metadata | RS (64d) + Metadata (6d) | −0.016 | 57.35 | 0.889 | 14.24 |
| 06 | Review Text Emb | BERT reviews (384d) | −0.011 | 57.20 | −0.172 | 39.51 |
| 07 | Tag Embeddings | BERT tags (384d) | −6.93 | — | −3.776 | — |
| 08 | Hybrid Collab-Content | RS (64d) + TF-IDF (100d) + Numeric (2d) | −0.024 | 57.57 | **0.856** | **16.20** |
| 09 | RS + Review Text | RS (64d) + BERT reviews (384d) | −0.021 | 57.50 | 0.873 | 14.01 |
| 10 | RS + Reviews + RAWG | RS (64d) + BERT (384d) + RAWG (12d) | −0.016 | 57.36 | 0.863 | 14.76 |
| 11a | RS + Dev Reputation | RS (64d) + dev_rep (1d) | −0.026 | 57.62 | 0.778 | 20.37 |
| 11b | Hybrid + Dev Reputation | RS (64d) + TF-IDF (100d) + Num (2d) + dev_rep | −0.016 | 57.36 | 0.858 | 16.25 |
| 12 | Stage2 Content | TF-IDF (100d) + Steam (2d) + RAWG (12d) + dev_rep | **+0.074** | **54.76** | 0.469 | 26.37 |

> Los R² temporales negativos en modelos RS indican cold-start: los 486 juegos post-2016 no tienen embeddings entrenados.

### Modelos con Log-Transform del Target — log(1+reviews)

| ID | Modelo | R² Temporal (orig) | R² TSCV (log) | RMSE TSCV (log) | Nota |
|----|--------|:------------------:|:-------------:|:---------------:|------|
| 13a | RS Only | −0.026 | 0.937 | 0.275 | Cold-start no mejora |
| **13b** | **Hybrid RS+TF-IDF** | **−0.025** | **0.940** | **0.268** | **Mejor TSCV del proyecto** |
| 13c | Stage2 Content | +0.050 | 0.754 | 0.547 | Log no supera lineal en temporal |

> El log-transform no resuelve el cold-start. Para TSCV mejora a 0.940 (13b), pero Stage2 log (13c) es peor que el lineal (12: R²=+0.074). El resultado previo de 13c (R²_log=0.254) estaba inflado por leakage de TF-IDF — corregido.

---

## Hallazgos Clave

### 1. Cold-Start Degrada los RS Embeddings
Los 486 juegos post-2016 no aparecen en el entrenamiento del Two-Tower. Sus embeddings son vectores no entrenados (ruido aleatorio). El XGBoost no puede extraer señal útil → R²≈−0.02 en temporal para todos los modelos con RS.

### 2. RS Embeddings son Superiores Within-Distribution
En el régimen pre-2016 donde todos los juegos tienen embeddings entrenados, la diferencia es contundente:
- **Modelos RS** (03, 08, 09, 10): TSCV R² = **0.856–0.889**
- **Metadata Only** (sin RS): TSCV R² = **−0.214**, RMSE = 39.44

La hipótesis central se confirma: el espacio colaborativo captura señal que el contenido no puede replicar.

### 3. El Espacio Colaborativo es Ortogonal al de Contenido
Análisis en `02b_cold_start_embeddings.ipynb`:
- Cosine similarity en espacio de features (tags, géneros): **0.987** entre vecinos más cercanos
- Cosine similarity en espacio RS: **0.064** para los mismos pares
- k-NN (k=10) para inferir embeddings RS desde contenido: R²CV = **−0.085**

Conclusión: No existe mapping aprendible entre el espacio de contenido y el espacio colaborativo con los datos disponibles. El cold-start es un problema estructural, no de features.

### 4. Developer Reputation Introduce Leakage en TSCV
La reputación del desarrollador, calculada sobre todos los juegos pre-2016, usa información futura para las ventanas anteriores del TSCV. Resultado: TSCV R² cae de +0.770 a **−0.808** al añadirla sin control. En Stage 2 (contenido, sin TSCV relevante) sí aporta marginalmente.

### 5. Log-Transform Mejora TSCV pero No el Cold-Start
El log(1+y) como target mejora la predicción within-distribution:
- TSCV: 0.856 → **0.940** para Hybrid RS+log (13b) — mejor TSCV del proyecto
- Sin embargo, Stage2 + log (13c) obtiene R²=+0.050 temporal, **inferior** al Stage2 lineal (12: R²=+0.074)
- El resultado anterior (R²_log=0.254 para 13c) era un artefacto del leakage de TF-IDF — corregido

### 6. Sistema de Dos Etapas — Mejor Compromiso Práctico
| Etapa | Modelo | Escenario | Métrica |
|-------|--------|-----------|---------|
| **Stage 1** | RS embeddings (Modelo 08 / 13b) | Juegos con historial conocido | TSCV R²=0.856–0.940 |
| **Stage 2** | Stage2 Content lineal (Modelo 12) | Juegos nuevos (cold-start) | Temporal R²=**+0.074** |

---

## Pipeline

### Notebooks

| Notebook | Descripción |
|----------|-------------|
| `01_data_preprocessing.ipynb` | Limpieza Steam, RAWG, dataset australiano |
| `02_model_keras_rs.ipynb` | Two-Tower CF (Keras/JAX). Genera embeddings 64d. Filtro anti-leakage pre-2016 |
| `02b_cold_start_embeddings.ipynb` | Análisis cold-start: k-NN, ortogonalidad de espacios |
| `03_regressor_embeddings_only.ipynb` | Modelo 03: RS Only |
| `04_regressor_metadata_only.ipynb` | Modelo 04: Metadata Only |
| `05_regressor_embeddings_plus_metadata.ipynb` | Modelo 05: RS + Metadata |
| `06_regressor_review_embeddings.ipynb` | Modelo 06: BERT reviews |
| `07_regressor_tag_embeddings.ipynb` | Modelo 07: BERT tags |
| `08_hybrid_collaborative_content.ipynb` | Modelo 08: RS + TF-IDF + Numeric |
| `09_regressor_rs_plus_reviews.ipynb` | Modelo 09: RS + BERT reviews |
| `10a_rawg_data_fetcher.ipynb` | Recolección RAWG API |
| `10b_model_enriched.ipynb` | Modelo 10: RS + Reviews + RAWG |
| `11_developer_reputation.ipynb` | Modelos 11a/11b: Developer reputation |
| `12_two_stage_model.ipynb` | Modelo 12: Stage 2 del sistema dos etapas |
| `13_log_transform.ipynb` | Modelos 13a/b/c: log(1+y) como target |

### Archivos de Datos Clave

| Archivo | Descripción |
|---------|-------------|
| `item_embeddings_rs_clean.npy` | Embeddings Two-Tower post-fix (3682, 64) — solo juegos pre-2016 |
| `user_embeddings_rs_clean.npy` | Embeddings de usuarios (25458, 64) |
| `item_embeddings_rs_coldstart.npy` | Embeddings post-2016 inferidos por k-NN (3682, 64) |
| `developer_reputation.npy` | Reputación histórica por item_idx (3682,) float32 |
| `rawg_enriched.csv` | Features RAWG: metacritic, rating, playtime, plataformas, géneros |
| `experiment_results.json` | Resultados de todos los modelos (machine-readable) |
| `EXPERIMENTS.md` | Registro completo de experimentos, hallazgos, correcciones |

---

## Conclusiones

1. **La hipótesis se confirma within-distribution.** RS embeddings dominan cuando existe historial: TSCV R²=0.856–0.889 vs −0.214 para metadata pura. El espacio colaborativo captura patrones de consumo colectivo que ninguna descripción de contenido replica.

2. **El cold-start es una limitación estructural, no de features.** Los espacios colaborativo y de contenido son ortogonales (cosine sim=0.064). No existe mapping aprendible entre ambos con los datos disponibles.

3. **El sistema de dos etapas es el mejor compromiso práctico.** Stage 1 (RS, TSCV R²=0.856) para juegos conocidos + Stage 2 (contenido lineal, temporal R²=+0.074) para juegos nuevos = cobertura completa.

4. **El log-transform mejora TSCV pero no el cold-start.** Predecir en escala logarítmica mejora el Stage 1 hasta TSCV R²=0.940, pero Stage2 + log (R²=+0.050) es inferior al Stage2 lineal (R²=+0.074).

5. **Validación temporal estricta es indispensable.** El TSCV y el split temporal revelan dinámicas muy distintas. Evaluar solo con K-Fold hubiera dado una imagen engañosamente optimista (KFold R²=0.45 para RS vs temporal R²=−0.02).

---

## Tecnologías

| Área | Librería |
|------|----------|
| Machine Learning | XGBoost, scikit-learn, Optuna |
| Deep Learning / RS | Keras 3, keras-rs, JAX |
| NLP | Sentence-Transformers (`all-MiniLM-L6-v2`), TF-IDF |
| Data | pandas, numpy, pyarrow |
| Visualización | matplotlib, seaborn |
| Externo | RAWG API |

---

## Limitaciones

- El target (`total_reviews`) es una muestra sesgada: usuarios australianos que escribieron reviews, no ventas globales.
- La distribución del target es extremadamente sesgada (mayoría <20 reviews, máximo 3.759).
- El período de test (post-2016) coincide con crecimiento explosivo de Steam, cambiando la distribución subyacente.
- Developer reputation calculada globalmente introduce leakage en TSCV (requiere cálculo por ventana para ser rigurosa).

---

*Ver [EXPERIMENTS.md](EXPERIMENTS.md) para el registro completo de todos los experimentos, correcciones aplicadas y análisis detallados.*
