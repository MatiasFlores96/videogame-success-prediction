# 🎮 Video Game Popularity Prediction — Steam Dataset

Predicción de popularidad de videojuegos en Steam comparando distintas estrategias de feature engineering: señales colaborativas (RS embeddings), contenido (metadata, NLP) e híbridos con datos externos.

## Proyecto de Tesis

**Objetivo:** Determinar qué tipo de representación de un videojuego predice mejor su popularidad futura (medida como número de reviews en Steam).

**Dataset:** Australian Gaming Dataset (Steam)
- 3.682 juegos
- 25.458 usuarios
- 59.305 interacciones

**Metodología:** XGBoost con validación temporal estricta — entrenamiento en juegos lanzados antes de 2017, test en juegos lanzados en 2017 o después.

---

## Resultados Principales

| # | Notebook | Modelo | R² test | RMSE | Descripción |
|---|----------|--------|:-------:|:----:|-------------|
| 05 | `05_regressor_embeddings_plus_metadata.ipynb` | **RS + Metadata** | **0.7922** | **25.94** | 🏆 Mejor modelo |
| 09 | `09_regressor_rs_plus_reviews.ipynb` | RS + Review Text | 0.7857 | 26.34 | RS + NLP semántico |
| 10 | `10b_model_enriched.ipynb` | RS + Reviews + RAWG | 0.6750 | 32.44 | Con datos externos |
| 08 | `08_hybrid_collaborative_content.ipynb` | Hybrid Collab-Content | 0.5629 | 37.61 | RS + TF-IDF |
| 03 | `03_regressor_embeddings_only.ipynb` | RS Embeddings Only | 0.5439 | 38.42 | Baseline colaborativo |
| 04 | `04_regressor_metadata_only.ipynb` | Metadata Only | 0.0113 | 56.57 | Contenido puro |
| 06 | `06_regressor_review_embeddings.ipynb` | Review Text Emb | −0.0200 | 57.46 | NLP puro (falla) |
| 07 | `07_regressor_tag_embeddings.ipynb` | Tag Embeddings | −174.99 | 76.25 | Colapso del modelo |

### Hallazgos Clave

- **Las señales colaborativas son la base:** RS embeddings capturan el 100% del gain de importancia en el mejor modelo; la metadata agrega regularización sin quitar protagonismo.
- **NLP solo falla, combinado con RS funciona:** Review text embeddings aislados dan R² = −0.02; junto con RS alcanzan R² = 0.79.
- **Más datos no siempre ayudan:** Agregar datos externos de RAWG (precio, géneros, plataformas) reduce R² de 0.79 a 0.68, posiblemente por ruido.
- **Validación temporal es esencial:** Simula producción realista y evita data leakage cronológico.

---

## 📁 Estructura del Proyecto

```
VG_Recommender/
├── Code/
│   ├── 00_leaderboard.ipynb                        # Leaderboard interactivo de todos los modelos
│   ├── 00_model_comparison.ipynb                   # Dashboard comparativo con gráficos
│   ├── 01_generate_dataset.ipynb                   # Limpieza y generación de datasets base
│   ├── 02_model_keras_rs.ipynb                     # Two-Tower RS (Keras RS / JAX)
│   ├── 03_regressor_embeddings_only.ipynb          # Modelo 03: RS embeddings solo
│   ├── 04_regressor_metadata_only.ipynb            # Modelo 04: Metadata solo
│   ├── 05_regressor_embeddings_plus_metadata.ipynb # Modelo 05: RS + Metadata 🏆
│   ├── 06_regressor_review_embeddings.ipynb        # Modelo 06: Review text NLP
│   ├── 07_regressor_tag_embeddings.ipynb           # Modelo 07: Tag NLP
│   ├── 08_hybrid_collaborative_content.ipynb       # Modelo 08: Hybrid RS + TF-IDF
│   ├── 09_regressor_rs_plus_reviews.ipynb          # Modelo 09: RS + Review Text
│   ├── 10a_rawg_data_fetcher.ipynb                 # Recolección de datos externos (RAWG API)
│   ├── 10b_model_enriched.ipynb                    # Modelo 10: RS + Reviews + RAWG
│   ├── 11_shap_error_analysis.ipynb                # Feature importance y análisis de errores
│   ├── results_tracker.py                          # Helper para guardar/leer resultados
│   ├── RESUMEN_MODELOS.ipynb                       # Documentación detallada de cada modelo
│   └── compare_ndcg.md / literature_review_metrics.md  # Notas de investigación
├── Data/
│   ├── experiment_results.json                     # Resultados de todos los experimentos
│   ├── steam_games.json                            # Catálogo de juegos Steam
│   ├── australian_user_reviews.json                # Reviews de usuarios
│   ├── australian_interactions.csv                 # Interacciones usuario-juego
│   ├── user2idx.json / item2idx.json               # Mappings de índices
│   └── [archivos generados — excluidos de git]
└── README.md
```

---

## Pipeline Completo

### 1. Generación de Datasets
**Notebook:** `01_generate_dataset.ipynb`

Limpia el dataset crudo, filtra usuarios y juegos con interacciones mínimas, y genera:
- `interactions.parquet` — interacciones usuario-juego filtradas
- `user2idx.json` / `item2idx.json` — mappings de IDs a índices

### 2. Sistema de Recomendación (Two-Tower)
**Notebook:** `02_model_keras_rs.ipynb`

Entrena un modelo Two-Tower con Keras RS (backend JAX). Genera embeddings colaborativos de 64 dimensiones para cada juego:
- `item_embeddings_rs.npy` — embeddings de ítems (3.682 × 64)
- `user_embeddings_rs.npy` — embeddings de usuarios (25.458 × 64)

### 3. Modelos de Predicción de Popularidad
**Notebooks:** `03` al `10b`

Cada notebook entrena un XGBoost con distintas combinaciones de features y registra métricas en `Data/experiment_results.json` usando `results_tracker.py`.

Evaluación con tres esquemas:
- **Temporal split** (principal): train pre-2017, test 2017+
- **Time-Series CV** (5 folds)
- **K-Fold CV** (5 folds, referencia)

### 4. Datos Externos — RAWG API
**Notebooks:** `10a_rawg_data_fetcher.ipynb` → `10b_model_enriched.ipynb`

`10a` recolecta metadata adicional (géneros, plataformas, scores de Metacritic, precio) para cada juego vía RAWG API. `10b` incorpora esa información al mejor modelo para evaluar si enriquece la predicción.

### 5. Feature Importance y Análisis de Errores
**Notebook:** `11_shap_error_analysis.ipynb`

Análisis exhaustivo del mejor modelo (Modelo 05):
- **Feature importance (XGBoost native gain):** RS embeddings = 100%, metadata = 0%
- **Sesgo global:** −1.02 reviews (prácticamente neutro)
- **Distribución:** 47.3% subestimados / 52.7% sobestimados
- **Mejor predichos:** Gang Beasts, Darkest Dungeon, Starbound
- **Mayor subestimación:** No Man's Sky (pred: 2 vs real: 181 reviews), Stardew Valley, DOOM (2016), Dark Souls III
- **Mayor sobestimación:** Robocraft (pred: 733 vs real: 346 reviews)

---

## Tecnologías Utilizadas

| Área | Librería |
|------|----------|
| Machine Learning | XGBoost, scikit-learn, Optuna |
| Deep Learning / RS | Keras 3, keras-rs, JAX |
| NLP | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Data | pandas, numpy, pyarrow |
| Visualización | matplotlib, seaborn |
| Externo | RAWG API |

---

## Métricas de Evaluación

| Métrica | Descripción |
|---------|-------------|
| **R²** | Proporción de varianza explicada. 1.0 = perfecto, 0.0 = media, < 0 = peor que la media |
| **RMSE** | Error cuadrático medio en unidades de reviews |
| **MAE** | Error absoluto medio |
| **MAPE** | Error porcentual medio |

---

## Conclusiones

1. **Los embeddings colaborativos son el feature más poderoso.** Capturan patrones de consumo colectivo que ninguna descripción de contenido puede replicar por sí sola.

2. **El contenido solo no predice popularidad.** Metadata (precio, género, desarrollador) y texto de reviews en aislamiento tienen R² cercano a cero o negativo.

3. **La combinación RS + Metadata es el óptimo.** Agregar metadata aporta regularización ligera que mejora sobre RS solo (0.79 vs 0.54), pero sin perder la dominancia colaborativa.

4. **Más features no siempre es mejor.** Incorporar datos externos de RAWG reduce el rendimiento, sugiriendo que la señal adicional introduce ruido en este contexto.

5. **La validación temporal es crítica.** Predecir el éxito de juegos futuros con datos históricos es la configuración correcta para evaluar utilidad práctica.

---

## Licencia

Este proyecto es parte de una tesis académica.

