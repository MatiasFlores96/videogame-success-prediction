"""
results_tracker.py — Gestión de resultados de experimentos.
Guarda/carga métricas en Data/experiment_results.json.
"""
import json, os
from datetime import datetime
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(_HERE, '..', '..', 'Data', 'experiment_results.json')


def _load_raw():
    if not os.path.exists(RESULTS_FILE):
        return []
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_result(model_id, model_name, features='', embeddings='', metrics=None, notes=''):
    """Guarda (o actualiza) los resultados de un modelo en experiment_results.json."""
    data = _load_raw()
    data = [d for d in data if d['model_id'] != str(model_id)]
    data.append({
        'model_id':   str(model_id),
        'model_name': model_name,
        'features':   features,
        'embeddings': embeddings,
        'metrics':    metrics or {},
        'notes':      notes,
        'timestamp':  datetime.now().strftime('%Y-%m-%d %H:%M'),
    })
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_results():
    """Carga los resultados y devuelve un DataFrame ordenado por R² (test temporal)."""
    data = _load_raw()
    if not data:
        return pd.DataFrame()
    rows = []
    for d in data:
        m = d.get('metrics', {})
        rows.append({
            'ID':         d.get('model_id', ''),
            'Modelo':     d.get('model_name', ''),
            'Features':   d.get('features', ''),
            'Emb':        d.get('embeddings', ''),
            'R2 test':    m.get('r2_temporal'),
            'RMSE test':  m.get('rmse_temporal'),
            'MAE test':   m.get('mae_temporal'),
            'MAPE test %': m.get('mape_temporal'),
            'R2 TSCV':    m.get('r2_tscv'),
            'R2 KFold':   m.get('r2_kfold'),
            'Notas':      d.get('notes', ''),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values('R2 test', ascending=False, na_position='last').reset_index(drop=True)
    return df


def print_leaderboard():
    """Imprime el leaderboard por consola."""
    df = load_results()
    if df.empty:
        print('No hay resultados guardados.')
        return
    sep = '=' * 92
    print(sep)
    print(f"{'ID':>3}  {'Modelo':<28}  {'R2 test':>8}  {'RMSE':>7}  "
          f"{'MAE':>6}  {'MAPE%':>6}  {'R2 TSCV':>8}  {'R2 KFold':>8}")
    print(sep)
    for _, r in df.iterrows():
        def fmt(v, fmt_str):
            return format(v, fmt_str) if pd.notna(v) else '   N/A'
        r2    = fmt(r['R2 test'],    '.4f')
        rmse  = fmt(r['RMSE test'],  '.2f')
        mae   = fmt(r['MAE test'],   '.2f')
        mape  = fmt(r['MAPE test %'],'.1f')
        tscv  = fmt(r['R2 TSCV'],   '.4f')
        kfold = fmt(r['R2 KFold'],  '.4f')
        print(f"[{r['ID']:>2}]  {str(r['Modelo']):<28}  {r2:>8}  {rmse:>7}  "
              f"{mae:>6}  {mape:>6}  {tscv:>8}  {kfold:>8}")
    print(sep)
