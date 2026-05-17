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
| 03 | RS Embeddings Only | RS clean (64d) | −0.027 | 57.66 | 0.770 | 19.50 |
| 04 | Metadata Only | Steam metadata (6d) | +0.065 | 55.00 | −0.516 | 42.44 |
| 05 | RS + Metadata | RS (64d) + Metadata (6d) | −0.024 | 57.57 | 0.760 | 18.66 |
| 06 | Review Text Emb | BERT reviews (384d) | ~0.000 | 56.90 | −0.117 | 39.19 |
| 07 | Tag Embeddings | BERT tags (384d) | −127.24 | 65.09 | −22.55 | 53.79 |
| 08 | Hybrid Collab-Content | RS (64d) + TF-IDF (100d) + Numeric (2d) | −0.024 | 57.57 | **0.841** | **16.76** |
| 09 | RS + Review Text | RS (64d) + BERT reviews (384d) | −0.022 | 57.53 | 0.680 | 24.34 |
| 10 | RS + Reviews + RAWG | RS (64d) + BERT (384d) + RAWG (12d) | −0.017 | 57.38 | 0.767 | 20.83 |
| 11a | RS + Dev Reputation | RS (64d) + dev_rep (1d) | −0.025 | 57.61 | −0.808 | 42.71 |
| 11b | Hybrid + Dev Reputation | RS (64d) + TF-IDF (100d) + Num (2d) + dev_rep | −0.024 | 57.57 | 0.722 | 22.73 |
| 12 | Stage2 Content | TF-IDF (100d) + Steam (2d) + RAWG (12d) + dev_rep | **+0.076** | **54.69** | −5.03 | 57.12 |

> Los R² temporales negativos en modelos RS indican cold-start: los 486 juegos post-2016 no tienen embeddings entrenados.

### Modelos con Log-Transform del Target — log(1+reviews)

| ID | Modelo | R² Temporal (log) | RMSE Temporal (log) | R² TSCV (log) | R² original back-transform |
|----|--------|:-----------------:|:-------------------:|:-------------:|:-------------------------:|
| 13a | RS Only | −0.386 | 1.150 | 0.932 | −0.027 |
| 13b | Hybrid RS+TF-IDF | −0.299 | 1.113 | 0.928 | −0.025 |
| **13c** | **Stage2 Content** | **+0.254** | **0.844** | **0.757** | +0.060 |

> El log-transform no resuelve el cold-start para modelos RS. Para Stage2 Content (13c), el R²_log=+0.254 es el mejor resultado temporal del proyecto.

---

## Hallazgos Clave

### 1. Cold-Start Degrada los RS Embeddings
Los 486 juegos post-2016 no aparecen en el entrenamiento del Two-Tower. Sus embeddings son vectores no entrenados (ruido aleatorio). El XGBoost no puede extraer señal útil → R²≈−0.02 en temporal para todos los modelos con RS.

### 2. RS Embeddings son Superiores Within-Distribution
En el régimen pre-2016 donde todos los juegos tienen embeddings entrenados, la diferencia es contundente:
- **Modelo 08** (Hybrid RS): TSCV R² = **0.841**, RMSE = 16.76
- **Metadata Only** (sin RS): TSCV R² = **−0.516**, RMSE = 42.44

La hipótesis central se confirma: el espacio colaborativo captura señal que el contenido no puede replicar.

### 3. El Espacio Colaborativo es Ortogonal al de Contenido
Análisis en `02b_cold_start_embeddings.ipynb`:
- Cosine similarity en espacio de features (tags, géneros): **0.987** entre vecinos más cercanos
- Cosine similarity en espacio RS: **0.064** para los mismos pares
- k-NN (k=10) para inferir embeddings RS desde contenido: R²CV = **−0.085**

Conclusión: No existe mapping aprendible entre el espacio de contenido y el espacio colaborativo con los datos disponibles. El cold-start es un problema estructural, no de features.

### 4. Developer Reputation Introduce Leakage en TSCV
La reputación del desarrollador, calculada sobre todos los juegos pre-2016, usa información futura para las ventanas anteriores del TSCV. Resultado: TSCV R² cae de +0.770 a **−0.808** al añadirla sin control. En Stage 2 (contenido, sin TSCV relevante) sí aporta marginalmente.

### 5. Log-Transform Mejora la Predicción Estructuralmente
Aplicar log(1+y) mejora el ajuste porque el target tiene una distribución de potencia:
- TSCV: 0.841 → 0.928 para el Hybrid RS (13b)
- **Stage2**: temporal R² +0.076 (lineal) → **+0.254 (log)** — mejor resultado del proyecto
- TSCV R²=0.757 ≈ KFold R²=0.724 para 13c → sin overfitting

### 6. Sistema de Dos Etapas — Mejor Compromiso Práctico
| Etapa | Modelo | Escenario | Métrica |
|-------|--------|-----------|---------|
| **Stage 1** | RS embeddings (Modelo 08 / 13b) | Juegos con historial conocido | TSCV R²=0.841–0.932 |
| **Stage 2** | TF-IDF + RAWG + Steam + dev_rep + log (Modelo 13c) | Juegos nuevos (cold-start) | Temporal R²_log=+0.254 |

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

1. **La hipótesis se confirma within-distribution.** RS embeddings dominan cuando existe historial: TSCV R²=0.841 vs −0.516 para metadata pura. El espacio colaborativo captura patrones de consumo colectivo que ninguna descripción de contenido replica.

2. **El cold-start es una limitación estructural, no de features.** Los espacios colaborativo y de contenido son ortogonales (cosine sim=0.064). No existe mapping aprendible entre ambos con los datos disponibles.

3. **El sistema de dos etapas es el mejor compromiso práctico.** Stage 1 (RS) para juegos conocidos + Stage 2 (contenido + log-transform) para juegos nuevos = cobertura completa con las mejores métricas en cada régimen.

4. **El log-transform del target es una mejora metodológica significativa.** Dado el target de distribución de potencia, predecir en escala logarítmica es más adecuado. Stage 2 + log alcanza R²_log=+0.254 temporal — el mejor resultado del proyecto para el escenario de juegos nuevos.

5. **Validación temporal estricta es indispensable.** El TSCV y el split temporal revelan dinámicas muy distintas. Evaluar solo con K-Fold hubiera dado una imagen engañosamente optimista (KFold R²=0.85 para RS vs temporal R²=−0.02).

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
