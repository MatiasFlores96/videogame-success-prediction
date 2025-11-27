# 🎮 Video Game Popularity Prediction - Steam Dataset

Predicción de popularidad de videojuegos en Steam utilizando diferentes enfoques: colaborativo, contenido, e híbridos.

## Proyecto de Tesis

**Objetivo:** Comparar diferentes estrategias de feature engineering para predecir la popularidad (número de reviews) de videojuegos.

**Dataset:** Australian Gaming Dataset (Steam)
- 3,682 juegos
- 25,458 usuarios
- 59,305 interacciones

**Metodología:** XGBoost con validación temporal (train: pre-2017, test: 2017+)

---

## Resultados Principales

| Modelo | R² | RMSE | Descripción |
|--------|-----|------|-------------|
| **M7: RS + Review Text** | **0.9557** | **33.13** | 🏆 **BEST MODEL** - Híbrido colaborativo + NLP |
| M1: RS Embeddings | 0.6422 | 94.17 | Baseline colaborativo |
| M3: RS + Metadata | 0.6436 | 93.99 | Mejora marginal |
| M6: Hybrid Collab-Content | 0.3823 | 123.73 | Híbrido con TF-IDF |
| M4: Review Text | -0.0248 | 159.37 | NLP solo (falla) |
| M2: Metadata | -0.0875 | 164.17 | Contenido solo (falla) |
| M5: Tag Embeddings | -18.8602 | 23.82 | Colapso del modelo |

### Hallazgo Clave:
**Las señales colaborativas y semánticas son complementarias, no redundantes.**

- Reviews **solas** fallan (R² = -0.02)
- Reviews **combinadas con RS** mejoran dramáticamente (R² = 0.96)
- Feature importance: 49% RS + 51% Reviews (perfectamente balanceado)

---

## 📁 Estructura del Proyecto

```
VG_Recommender/
├── Code/
│   ├── 00_model_comparison.ipynb           # Dashboard comparativo
│   ├── 01_generate_dataset.ipynb           # Generación de datasets
│   ├── 02_model_keras_rs.ipynb             # Sistema de recomendación
│   ├── 03_regressor_embeddings_only.ipynb  # Modelo 1: RS only
│   ├── 04_regressor_metadata_only.ipynb    # Modelo 2: Metadata
│   ├── 05_regressor_embeddings_plus_metadata.ipynb # Modelo 3: RS + Metadata
│   ├── 06_regressor_review_embeddings.ipynb # Modelo 4: Review text NLP
│   ├── 07_regressor_tag_embeddings.ipynb   # Modelo 5: Tag NLP
│   ├── 08_hybrid_collaborative_content.ipynb # Modelo 6: Hybrid TF-IDF
│   ├── 09_regressor_rs_plus_reviews.ipynb  # Modelo 7: RS + Reviews 🏆
│   ├── RESUMEN_MODELOS.ipynb               # Documentación completa
│   └── PRESENTACION_TESIS.ipynb            # Presentación para director
├── Data/                                   # (Excluido de git - archivos grandes)
│   ├── user2idx.json
│   ├── item2idx.json
│   ├── item_embeddings_rs.npy
│   ├── review_text_embeddings.npy
│   ├── steam_games.json
│   └── australian_user_reviews.json
└── README.md
```

---

## Pipeline Completo

### 1. Generación de Datasets
**Notebook:** `01_generate_dataset.ipynb`

```python
# Output:
- interactions.parquet  # Interacciones usuario-juego
- user2idx.json         # Mapeo usuario → índice
- item2idx.json         # Mapeo juego → índice
```

### 2. Entrenamiento Sistema de Recomendación
**Notebook:** `02_model_keras_rs.ipynb`

```python
# Two-Tower Model con Keras RS
# Output:
- item_embeddings_rs.npy (3682 × 64)  # Embeddings colaborativos
- user_embeddings_rs.npy (25458 × 64) # Embeddings de usuarios
```

### 3. Modelos de Predicción de Éxito
**Notebooks:** `03-09_regressor_*.ipynb`

Cada notebook evalúa un enfoque diferente:
- **Colaborativo puro** (M1, M3)
- **Contenido puro** (M2, M4, M5)
- **Híbridos** (M6, M7)

---

## Tecnologías Utilizadas

- **Python 3.11**
- **Machine Learning:** XGBoost, scikit-learn
- **Deep Learning:** Keras 3, keras-rs, JAX
- **NLP:** Sentence-Transformers (all-MiniLM-L6-v2)
- **Data:** pandas, numpy
- **Viz:** matplotlib

---

## Métricas

### R² (Coeficiente de Determinación)
- Proporción de varianza explicada
- Rango: (-∞, 1.0]
- **R² = 1.0**: Predicción perfecta
- **R² = 0.0**: Modelo predice la media
- **R² < 0.0**: Peor que predecir la media

### RMSE (Root Mean Squared Error)
- Error promedio en número de reviews
- Unidades interpretables
- Menor es mejor

---

## Conclusiones

### 1. Patrones colaborativos son la base
- RS embeddings (64-dim) explican 64% de la varianza
- Superan ampliamente a enfoques de contenido puro

### 2. Contenido SOLO falla, COMBINADO triunfa
- Reviews solas: R² = -0.02 
- RS + Reviews: R² = 0.96 
- Señales ortogonales (complementarias)

### 3. No todas las combinaciones funcionan
- RS + Review embeddings: R² = 0.96 
- RS + TF-IDF: R² = 0.38 
- Calidad > Cantidad de features

### 4. Validación temporal es esencial
- Simula producción realista
- Evita data leakage
- Más conservador pero más confiable

---

## Notebooks de Documentación

- **`RESUMEN_MODELOS.ipynb`**: Análisis exhaustivo de los 7 modelos
- **`PRESENTACION_TESIS.ipynb`**: Material para reuniones con director
- **`00_model_comparison.ipynb`**: Dashboard con resultados visuales

---

## Trabajo Futuro


---

## Licencia

Este proyecto es parte de una tesis académica.

---

