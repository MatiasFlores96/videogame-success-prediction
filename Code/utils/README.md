# Code/utils — Scripts de diagnóstico y verificación

Scripts utilitarios usados durante el desarrollo para validar datos,
configuración del entorno y supuestos del pipeline. No forman parte
del pipeline de reproducción — ver `Code/PIPELINE.md`.

| Script | Propósito |
|--------|-----------|
| `_check_dataset.py` | Estadísticas del dataset v1 (interactions.parquet, item2idx) |
| `_check_dates.py` | Distribución de release dates; verifica el corte train/test |
| `_check_gpu.py` | Verifica disponibilidad de CUDA y PyTorch |
| `_check_gpu2.py` | Diagnóstico detallado de GPU (memoria, driver) |
| `_check_ids.py` | Solapamiento de AppIDs entre v1 y v2 |
| `_check_sentiment.py` | Distribución del campo `sentiment` en steam_games.json |
| `_check_v2_scope.py` | Verifica que los embeddings v2 cubran el test set |
| `_check_year_dist.py` | Histograma de juegos por año de lanzamiento |
| `_check14.py` | Verificación de outputs del script 14_ |
| `_null_baseline.py` | Baseline trivial (predice la media del train) |
| `_peek_sg_v2.py` | Inspección rápida de steam_games.json con ítem 2idx_v2 |
| `_peek_v2.py` | Estadísticas de interactions_v2.parquet |
| `_peek_v2b.py` | Estadísticas de interacciones v2 (variante con desglose) |
| `_test_steamspy.py` | Test de conectividad con la API de SteamSpy |
| `_verify_data.py` | Verificación integral de integridad del dataset |
| `fix_notebooks.py` | Limpieza de metadatos de ejecución en notebooks |
