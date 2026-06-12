"""
14_improvements.py  —  Experimentos de mejora del RMSE temporal (Stage 2)

Tres variantes sobre Modelo 12 (Stage 2 Content, 115 features):
  14a — Winsorización del target al P99
  14b — Huber loss en XGBoost
  14c — Sin features post-launch de RAWG (sin playtime_avg_h, sin rawg_ratings_count)

Baseline: Modelo 12  R²=0.0737  RMSE=54.76
"""

import pandas as pd, numpy as np, json, ast, sys, os, warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INTER     = os.path.join(BASE, 'Data', 'interactions.parquet')
ITEMMAP   = os.path.join(BASE, 'Data', 'item2idx.json')
STEAM     = os.path.join(BASE, 'Data', 'steam_games.json')
RAWG_CSV  = os.path.join(BASE, 'Data', 'rawg_enriched.csv')
DEV_REP   = os.path.join(BASE, 'Data', 'developer_reputation.npy')
CUTOFF    = pd.to_datetime('2016-01-01')
RESULTS_F = os.path.join(BASE, 'Data', 'experiment_results.json')

# ── Helpers ────────────────────────────────────────────────────────────────────
def parse_list_col(x):
    if x is None or (isinstance(x, float) and pd.isna(x)): return []
    if isinstance(x, list): return [str(t).strip() for t in x if t]
    try:
        lst = eval(x)
        return [str(t).strip() for t in lst if t] if isinstance(lst, list) else []
    except: return []

# ── Cargar datos ───────────────────────────────────────────────────────────────
print("Cargando datos...")
with open(ITEMMAP) as f:
    item2idx = {k: int(v) for k, v in json.load(f).items()}
N = len(item2idx)

df_inter  = pd.read_parquet(INTER)
target_df = df_inter.groupby('item_idx').size().reset_index(name='total_reviews')
y_full    = target_df.set_index('item_idx').reindex(range(N), fill_value=0)['total_reviews'].values
dev_rep   = np.load(DEV_REP)

games = []
with open(STEAM, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: games.append(ast.literal_eval(line))
        except: pass

df_games = pd.json_normalize(games).rename(columns={'id': 'item_id'})
df_games['item_idx']             = df_games['item_id'].map(item2idx)
df_games                         = df_games.dropna(subset=['item_idx'])
df_games['item_idx']             = df_games['item_idx'].astype(int)
df_games['release_date_parsed']  = pd.to_datetime(df_games['release_date'], errors='coerce')
df_games['tag_text']             = df_games['tags'].apply(parse_list_col).apply(' '.join)
df_games['genre_text']           = df_games['genres'].apply(parse_list_col).apply(' '.join)
df_games['content_text']         = df_games['tag_text'] + ' ' + df_games['genre_text']
def _parse_price(p):
    try: return float(p)
    except: return 0.0
df_games['price_num']            = df_games['price'].apply(_parse_price)
df_games['ea_flag']              = df_games['early_access'].apply(lambda x: 1 if x else 0)

games_w_content = df_games[df_games['content_text'].str.strip() != ''].copy()

# TF-IDF fit solo en pre-2016
tfidf = TfidfVectorizer(max_features=100, min_df=2, max_df=0.5, ngram_range=(1,1))
_pre  = games_w_content[games_w_content['release_date_parsed'] < CUTOFF]['content_text']
tfidf.fit(_pre)
tfidf_matrix    = tfidf.transform(games_w_content['content_text'])
item_to_tfidf   = {int(r['item_idx']): i for i, (_, r) in enumerate(games_w_content.iterrows())}

# RAWG
df_rawg = pd.read_csv(RAWG_CSV).drop_duplicates(subset=['item_idx'], keep='last').set_index('item_idx')
ESRB_ORDER = {'Everyone':1,'Everyone 10+':2,'Teen':3,'Mature':4,'Adults Only':5}
RAWG_OFFSET = 102  # tfidf(100) + steam(2)

def rawg_vec_full(idx):
    """12 features incluyendo playtime y ratings_count (potencial leakage)"""
    if idx not in df_rawg.index: return [0.]*12
    r = df_rawg.loc[idx]
    return [
        1.0 if pd.notna(r.get('rawg_id')) else 0.0,
        float(r['rawg_rating'])         if pd.notna(r.get('rawg_rating'))       else -1.0,
        0.0 if pd.notna(r.get('rawg_rating'))   else 1.0,
        float(r['metacritic'])           if pd.notna(r.get('metacritic'))        else -1.0,
        0.0 if pd.notna(r.get('metacritic'))    else 1.0,
        np.log1p(float(r['playtime_avg_h']))     if pd.notna(r.get('playtime_avg_h'))     else 0.0,  # POSIBLE LEAKAGE
        np.log1p(float(r['rawg_ratings_count'])) if pd.notna(r.get('rawg_ratings_count')) else 0.0,  # POSIBLE LEAKAGE
        float(len(parse_list_col(r.get('platforms', [])))),
        float(len(parse_list_col(r.get('genres',    [])))),
        float(len(parse_list_col(r.get('developers',[])))),
        float(len(parse_list_col(r.get('publishers',[])))),
        float(ESRB_ORDER.get(r.get('esrb_rating', ''), 0)),
    ]

def rawg_vec_noleak(idx):
    """10 features sin playtime ni ratings_count"""
    if idx not in df_rawg.index: return [0.]*10
    r = df_rawg.loc[idx]
    return [
        1.0 if pd.notna(r.get('rawg_id')) else 0.0,
        float(r['rawg_rating'])  if pd.notna(r.get('rawg_rating'))  else -1.0,
        0.0 if pd.notna(r.get('rawg_rating'))  else 1.0,
        float(r['metacritic'])   if pd.notna(r.get('metacritic'))   else -1.0,
        0.0 if pd.notna(r.get('metacritic'))   else 1.0,
        float(len(parse_list_col(r.get('platforms', [])))),
        float(len(parse_list_col(r.get('genres',    [])))),
        float(len(parse_list_col(r.get('developers',[])))),
        float(len(parse_list_col(r.get('publishers',[])))),
        float(ESRB_ORDER.get(r.get('esrb_rating', ''), 0)),
    ]

rawg_full   = np.array([rawg_vec_full(i)   for i in range(N)], dtype=np.float32)
rawg_noleak = np.array([rawg_vec_noleak(i) for i in range(N)], dtype=np.float32)

# ── Construir matrices ─────────────────────────────────────────────────────────
print("Construyendo matrices de features...")
rows_full, rows_noleak, valid_idxs = [], [], []
for idx in range(N):
    if idx not in item_to_tfidf: continue
    g = games_w_content[games_w_content['item_idx'] == idx]
    if g.empty: continue
    g = g.iloc[0]
    tfidf_vec = tfidf_matrix[item_to_tfidf[idx]].toarray().flatten()
    steam_vec = np.array([g['price_num'], g['ea_flag']], dtype=np.float32)
    rows_full.append(np.concatenate([tfidf_vec, steam_vec, rawg_full[idx],   [dev_rep[idx]]]))
    rows_noleak.append(np.concatenate([tfidf_vec, steam_vec, rawg_noleak[idx], [dev_rep[idx]]]))
    valid_idxs.append(idx)

X_full   = np.array(rows_full,   dtype=np.float32)   # 115d (igual que M12)
X_noleak = np.array(rows_noleak, dtype=np.float32)   # 113d

y_s2 = target_df.set_index('item_idx').reindex(valid_idxs, fill_value=0)['total_reviews'].values
data_df = pd.DataFrame({'item_idx': valid_idxs}).merge(
    df_games[['item_idx', 'release_date_parsed']], on='item_idx', how='left'
)

s2_train = (data_df['release_date_parsed'].notna() & (data_df['release_date_parsed'] < CUTOFF)).values
s2_test  = (data_df['release_date_parsed'].notna() & (data_df['release_date_parsed'] >= CUTOFF)).values

# Winsorize P99
p99 = float(np.percentile(y_s2[s2_train], 99))
print(f"P99 del target en train: {p99:.1f} reviews")
y_wins = np.clip(y_s2, 0, p99)

print(f"Train: {s2_train.sum()} | Test: {s2_test.sum()} | Features full: {X_full.shape[1]} | noleak: {X_noleak.shape[1]}")

# ── Impute RAWG missing en full ────────────────────────────────────────────────
def impute_rawg(X, train_mask, rawg_offset, missing_cols):
    X = X.copy()
    for ci in missing_cols:
        col = rawg_offset + ci
        tv  = X[train_mask, col]
        vv  = tv[tv != -1.0]
        if len(vv):
            med = float(np.median(vv))
            X[X[:, col] == -1.0, col] = med
    return X

X_full_imp   = impute_rawg(X_full,   s2_train, 102, [1, 3])   # rawg_rating, metacritic
X_noleak_imp = impute_rawg(X_noleak, s2_train, 102, [1, 3])

# ── Optuna tuner ───────────────────────────────────────────────────────────────
def tune_xgb(X_tr, y_tr, objective='reg:squarederror', n_trials=60, tag=''):
    sort_order = np.argsort(data_df.loc[s2_train, 'release_date_parsed'].values)
    X_tr_s = X_tr[sort_order]
    y_tr_s = y_tr[sort_order]
    sp = max(10, int(len(X_tr_s) * 0.8))
    X_opt, X_val = X_tr_s[:sp], X_tr_s[sp:]
    y_opt, y_val = y_tr_s[:sp], y_tr_s[sp:]

    def obj(trial):
        p = dict(
            n_estimators      = trial.suggest_int('n_estimators', 100, 800),
            learning_rate     = trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            max_depth         = trial.suggest_int('max_depth', 3, 8),
            min_child_weight  = trial.suggest_int('min_child_weight', 1, 10),
            subsample         = trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0),
            gamma             = trial.suggest_float('gamma', 0.0, 2.0),
            objective         = objective,
            random_state=42, tree_method='hist', verbosity=0,
        )
        m = XGBRegressor(**p)
        m.fit(X_opt, y_opt)
        # Siempre evaluamos RMSE en escala original para comparar
        preds = m.predict(X_val)
        return float(mean_squared_error(y_val, preds) ** 0.5)

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    best = {**study.best_params, 'objective': objective, 'random_state': 42, 'tree_method': 'hist', 'verbosity': 0}
    print(f"  [{tag}] Best val RMSE: {study.best_value:.4f}")
    return best

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTO 14a — Winsorización P99
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("14a — WINSORIZACIÓN P99  (cap=%.1f reviews)" % p99)
print("="*70)

X_tr_a, y_tr_a = X_full_imp[s2_train], y_wins[s2_train]   # train en target cappeado
X_te_a, y_te_a = X_full_imp[s2_test],  y_s2[s2_test]       # test en target original

best_a = tune_xgb(X_tr_a, y_tr_a, tag='14a')
m14a = XGBRegressor(**best_a)
m14a.fit(X_tr_a, y_tr_a)
pred_a = m14a.predict(X_te_a)

r2_a   = r2_score(y_te_a, pred_a)
rmse_a = mean_squared_error(y_te_a, pred_a) ** 0.5
mae_a  = mean_absolute_error(y_te_a, pred_a)
mape_a = mean_absolute_percentage_error(y_te_a, pred_a)
print(f"  R² temporal:   {r2_a:.4f}")
print(f"  RMSE temporal: {rmse_a:.2f}  (baseline 54.76)")
print(f"  MAE temporal:  {mae_a:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTO 14b — Huber loss
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("14b — HUBER LOSS  (reg:pseudohubererror)")
print("="*70)

X_tr_b, y_tr_b = X_full_imp[s2_train], y_s2[s2_train]   # target original
X_te_b, y_te_b = X_full_imp[s2_test],  y_s2[s2_test]

best_b = tune_xgb(X_tr_b, y_tr_b, objective='reg:pseudohubererror', tag='14b')
m14b = XGBRegressor(**best_b)
m14b.fit(X_tr_b, y_tr_b)
pred_b = m14b.predict(X_te_b)

r2_b   = r2_score(y_te_b, pred_b)
rmse_b = mean_squared_error(y_te_b, pred_b) ** 0.5
mae_b  = mean_absolute_error(y_te_b, pred_b)
mape_b = mean_absolute_percentage_error(y_te_b, pred_b)
print(f"  R² temporal:   {r2_b:.4f}")
print(f"  RMSE temporal: {rmse_b:.2f}  (baseline 54.76)")
print(f"  MAE temporal:  {mae_b:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTO 14c — Sin features post-launch de RAWG
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("14c — SIN LEAKAGE RAWG  (sin playtime_avg_h, sin rawg_ratings_count)")
print(f"      Features: {X_noleak_imp.shape[1]}d  (vs 115d en M12)")
print("="*70)

X_tr_c, y_tr_c = X_noleak_imp[s2_train], y_s2[s2_train]
X_te_c, y_te_c = X_noleak_imp[s2_test],  y_s2[s2_test]

best_c = tune_xgb(X_tr_c, y_tr_c, tag='14c')
m14c = XGBRegressor(**best_c)
m14c.fit(X_tr_c, y_tr_c)
pred_c = m14c.predict(X_te_c)

r2_c   = r2_score(y_te_c, pred_c)
rmse_c = mean_squared_error(y_te_c, pred_c) ** 0.5
mae_c  = mean_absolute_error(y_te_c, pred_c)
mape_c = mean_absolute_percentage_error(y_te_c, pred_c)
print(f"  R² temporal:   {r2_c:.4f}")
print(f"  RMSE temporal: {rmse_c:.2f}  (baseline 54.76)")
print(f"  MAE temporal:  {mae_c:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN COMPARATIVO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("RESUMEN COMPARATIVO")
print("="*70)
print(f"{'Modelo':<35} {'R² temporal':>12} {'RMSE':>8} {'MAE':>8}")
print("-"*65)
print(f"{'12 — Baseline (M12)':35s} {'0.0737':>12} {'54.76':>8} {'13.97':>8}")
print(f"{'14a — Winsorización P99':35s} {r2_a:>12.4f} {rmse_a:>8.2f} {mae_a:>8.2f}")
print(f"{'14b — Huber loss':35s} {r2_b:>12.4f} {rmse_b:>8.2f} {mae_b:>8.2f}")
print(f"{'14c — Sin leakage RAWG':35s} {r2_c:>12.4f} {rmse_c:>8.2f} {mae_c:>8.2f}")
print("="*70)

# ── Guardar en experiment_results.json ────────────────────────────────────────
with open(RESULTS_F) as f:
    all_results = json.load(f)

existing_ids = {r['model_id'] for r in all_results}

new_results = [
    {'model_id':'14a','model_name':'Stage2 Winsorized P99',
     'features':f'TF-IDF(100)+Steam(2)+RAWG(12)+dev_rep(1), target capped P99={p99:.0f}',
     'embeddings':'none',
     'metrics':{'r2_temporal':r2_a,'rmse_temporal':rmse_a,'mae_temporal':mae_a,'mape_temporal':mape_a}},
    {'model_id':'14b','model_name':'Stage2 Huber Loss',
     'features':'TF-IDF(100)+Steam(2)+RAWG(12)+dev_rep(1), objective=huber',
     'embeddings':'none',
     'metrics':{'r2_temporal':r2_b,'rmse_temporal':rmse_b,'mae_temporal':mae_b,'mape_temporal':mape_b}},
    {'model_id':'14c','model_name':'Stage2 No-Leakage RAWG',
     'features':'TF-IDF(100)+Steam(2)+RAWG(10, sin playtime/ratings_cnt)+dev_rep(1)',
     'embeddings':'none',
     'metrics':{'r2_temporal':r2_c,'rmse_temporal':rmse_c,'mae_temporal':mae_c,'mape_temporal':mape_c}},
]

for nr in new_results:
    if nr['model_id'] in existing_ids:
        all_results = [r for r in all_results if r['model_id'] != nr['model_id']]
    all_results.append(nr)

with open(RESULTS_F, 'w') as f:
    json.dump(all_results, f, indent=2)

print("\nResultados guardados en experiment_results.json")
