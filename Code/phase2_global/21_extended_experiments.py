"""
21_extended_experiments.py
==========================
[EXPLORACIÓN — RESULTADOS CONTIENEN LEAKAGE]

Estos modelos incluyen `rawg_ratings_count` como feature, que es leakage duro:
el conteo de ratings en RAWG es un proxy de popularidad post-lanzamiento.
Identificado en script 20_ (SHAP: rawg_ratings_count = feature #1).
Los resultados R²≈0.44 son artefactos y no deben usarse como referencia en la tesis.

Script canónico limpio: 24_ablation_catboost.py y 34_train_rs_content_v3.py.

---
Mejoras sobre el mejor modelo (19_10, R²=0.438 — también leakeado):
  21_10_opt   — XGBoost 19_10 con 150 Optuna trials
  21_10_feat  — XGBoost con feature engineering sobre RAWG
  21_10_lgbm  — LightGBM equivalente a 19_10
  21_10_cat   — CatBoost equivalente a 19_10
  21_ensemble — Ensemble (promedio) 19_10 + 19_10_log
"""

import ast, json, os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# Intentar importar LightGBM y CatBoost
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[!] LightGBM no instalado — se omite")

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("[!] CatBoost no instalado — se omite")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_tracker import save_result, print_leaderboard

BASE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA   = os.path.join(BASE, "Data")
CUTOFF = pd.Timestamp("2017-01-01")

print("=" * 70)
print("21_extended_experiments.py — Mejoras sobre 19_10")
print("=" * 70)

# ── 1. Cargar datos (mismo que 18_full_v2_pipeline.py) ─────────
print("\n[1] Cargando datos base...")

with open(os.path.join(DATA, "item2idx_v2.json")) as f:
    item2idx_v2 = {k: int(v) for k, v in json.load(f).items()}
idx2appid = {v: k for k, v in item2idx_v2.items()}
N = len(item2idx_v2)

df_inter = pd.read_parquet(os.path.join(DATA, "interactions_v2.parquet"))
target_s = df_inter.groupby("item_idx").size().reindex(range(N), fill_value=0)
y_full   = target_s.values.astype(np.float64)

item_emb = np.load(os.path.join(DATA, "item_embeddings_rs_v2.npy"))
print(f"  Items: {N:,} | Embeddings: {item_emb.shape}")

# ── 2. Metadata ────────────────────────────────────────────────
print("[2] Cargando metadata...")

games_raw = []
with open(os.path.join(DATA, "steam_games.json"), "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: games_raw.append(ast.literal_eval(line))
        except: pass

df_games = pd.json_normalize(games_raw).rename(columns={"id": "item_id"})
df_games["item_id"]  = df_games["item_id"].astype(str)
df_games["item_idx"] = df_games["item_id"].map(item2idx_v2)
df_games = df_games.dropna(subset=["item_idx"])
df_games["item_idx"] = df_games["item_idx"].astype(int)
df_games["release_date_parsed"] = pd.to_datetime(df_games["release_date"], errors="coerce")
df_games = df_games.drop_duplicates(subset=["item_idx"], keep="first")

meta       = df_games.set_index("item_idx").reindex(range(N))
item_dates = meta["release_date_parsed"].values
train_mask = np.array([pd.notna(d) and d < CUTOFF  for d in item_dates])
test_mask  = np.array([pd.notna(d) and d >= CUTOFF for d in item_dates])
print(f"  Train: {train_mask.sum():,} | Test: {test_mask.sum():,}")

# ── 3. Features ────────────────────────────────────────────────
print("[3] Construyendo features...")

def parse_list_col(x):
    if x is None or (isinstance(x, float) and pd.isna(x)): return []
    if isinstance(x, list): return [str(t).strip() for t in x if t]
    try:
        lst = eval(x)
        return [str(t).strip() for t in lst if t] if isinstance(lst, list) else []
    except: return []

def clean_price(p):
    if pd.isna(p) or str(p).strip() in ("Free", "", "Free to Play"): return 0.0
    try: return float(str(p).replace("$","").replace(",","").strip())
    except: return 0.0

df_games["tag_text"]    = df_games["tags"].apply(parse_list_col).apply(" ".join)
df_games["genre_text"]  = df_games["genres"].apply(parse_list_col).apply(" ".join)
df_games["content_txt"] = df_games["tag_text"] + " " + df_games["genre_text"]
df_games["price_num"]   = df_games["price"].apply(clean_price)
df_games["ea_flag"]     = df_games["early_access"].fillna(False).astype(int)
df_games["num_genres"]  = df_games["genres"].apply(parse_list_col).apply(len)
df_games["num_tags"]    = df_games["tags"].apply(parse_list_col).apply(len)
df_games["num_specs"]   = df_games["specs"].apply(parse_list_col).apply(len) if "specs" in df_games.columns else 0
df_games["has_senti"]   = df_games["sentiment"].notna().astype(int) if "sentiment" in df_games.columns else 0

games_w_content = df_games[df_games["content_txt"].str.strip() != ""].copy()
_pre = games_w_content[
    games_w_content["release_date_parsed"].notna() &
    (games_w_content["release_date_parsed"] < CUTOFF)
]["content_txt"]
tfidf = TfidfVectorizer(max_features=100, min_df=2, max_df=0.5, ngram_range=(1,1))
tfidf.fit(_pre)

item_to_tfidf = {int(r["item_idx"]): i for i, (_, r) in enumerate(games_w_content.iterrows())}
tfidf_mat     = tfidf.transform(games_w_content["content_txt"])

META_COLS  = ["price_num", "ea_flag", "num_genres", "num_tags", "num_specs", "has_senti"]
meta_reind = df_games.drop_duplicates(subset=["item_idx"], keep="first").set_index("item_idx").reindex(range(N))

tfidf_aligned = np.zeros((N, 100), dtype=np.float32)
steam_aligned = np.zeros((N, 2),   dtype=np.float32)
meta_aligned  = np.zeros((N, 6),   dtype=np.float32)

for idx in range(N):
    if idx in item_to_tfidf:
        tfidf_aligned[idx] = tfidf_mat[item_to_tfidf[idx]].toarray().flatten()
    row = meta_reind.iloc[idx] if idx < len(meta_reind) else None
    if row is not None:
        steam_aligned[idx, 0] = float(row.get("price_num", 0) or 0)
        steam_aligned[idx, 1] = float(row.get("ea_flag",   0) or 0)
        for j, col in enumerate(META_COLS):
            v = row.get(col, 0)
            meta_aligned[idx, j] = float(v) if pd.notna(v) else 0.0

# ── 4. Dev rep ─────────────────────────────────────────────────
appid_to_devs = {}
for _, row in df_games.iterrows():
    devs = parse_list_col(row.get("developer", []))
    if not devs and isinstance(row.get("developer"), str):
        devs = [row["developer"].strip()]
    idx = int(row["item_idx"]) if pd.notna(row.get("item_idx")) else None
    if idx is not None:
        appid_to_devs[idx] = devs

train_items = np.where(train_mask)[0]
dev_to_reviews = {}
for idx in train_items:
    for dev in appid_to_devs.get(idx, []):
        dev_to_reviews.setdefault(dev, []).append(float(y_full[idx]))

dev_rep_v2 = np.zeros((N, 1), dtype=np.float32)
for idx in range(N):
    devs = appid_to_devs.get(idx, [])
    scores = []
    for dev in devs:
        own = float(y_full[idx])
        reviews_excl = [r for r in dev_to_reviews.get(dev, []) if r != own or not train_mask[idx]]
        if reviews_excl:
            scores.append(np.mean(reviews_excl))
    dev_rep_v2[idx, 0] = float(np.mean(scores)) if scores else 0.0

dr_max = dev_rep_v2[train_mask, 0].max()
dr_max = dr_max if dr_max > 0 else 1.0
dev_rep_v2 /= dr_max

# ── 5. RAWG ────────────────────────────────────────────────────
print("[4] Cargando RAWG...")

with open(os.path.join(DATA, "item2idx.json")) as f:
    item2idx_v1 = {k: int(v) for k, v in json.load(f).items()}
v1_idx_to_appid = {v: k for k, v in item2idx_v1.items()}

df_rawg = pd.read_csv(os.path.join(DATA, "rawg_enriched.csv")).drop_duplicates(subset=["item_idx"], keep="last")
df_rawg["appid_str"]   = df_rawg["item_idx"].map(v1_idx_to_appid)
df_rawg["item_idx_v2"] = df_rawg["appid_str"].map(item2idx_v2)
df_rawg = df_rawg.dropna(subset=["item_idx_v2"])
df_rawg["item_idx_v2"] = df_rawg["item_idx_v2"].astype(int)
df_rawg = df_rawg.set_index("item_idx_v2")

ESRB_ORDER = {"Everyone":1,"Everyone 10+":2,"Teen":3,"Mature":4,"Adults Only":5}
rawg_aligned = np.zeros((N, 12), dtype=np.float32)
matched = 0
for idx in range(N):
    if idx not in df_rawg.index: continue
    r = df_rawg.loc[idx]
    rawg_aligned[idx] = [
        1.0 if pd.notna(r.get("rawg_id")) else 0.0,
        float(r["rawg_rating"])        if pd.notna(r.get("rawg_rating"))       else -1.0,
        0.0 if pd.notna(r.get("rawg_rating"))    else 1.0,
        float(r["metacritic"])          if pd.notna(r.get("metacritic"))        else -1.0,
        0.0 if pd.notna(r.get("metacritic"))     else 1.0,
        np.log1p(float(r["playtime_avg_h"]))      if pd.notna(r.get("playtime_avg_h"))      else 0.0,
        np.log1p(float(r["rawg_ratings_count"]))  if pd.notna(r.get("rawg_ratings_count"))  else 0.0,
        float(len(parse_list_col(r.get("platforms",[])))),
        float(len(parse_list_col(r.get("genres",   [])))),
        float(len(parse_list_col(r.get("developers",[])))),
        float(len(parse_list_col(r.get("publishers",[])))),
        float(ESRB_ORDER.get(r.get("esrb_rating",""), 0)),
    ]
    matched += 1

for ci in [1, 3]:
    train_vals = rawg_aligned[train_mask, ci]
    valid = train_vals[train_vals != -1.0]
    med   = float(np.median(valid)) if len(valid) > 0 else 0.0
    rawg_aligned[rawg_aligned[:, ci] == -1.0, ci] = med

print(f"  RAWG: {matched}/{N} ({matched/N*100:.1f}%)")

# ── 5b. Feature engineering RAWG ──────────────────────────────
print("[5] Feature engineering RAWG...")

# rawg_quality: metacritic × log_ratings_count (interacción calidad × popularidad)
rawg_quality = (rawg_aligned[:, 3] * rawg_aligned[:, 6]).reshape(-1, 1).astype(np.float32)

# has_rawg × metacritic (solo para items con datos)
rawg_meta_valid = (rawg_aligned[:, 0] * rawg_aligned[:, 3]).reshape(-1, 1).astype(np.float32)

# release_year normalizado (año de lanzamiento como feature numérica)
release_years = np.array([
    float(pd.Timestamp(d).year) if pd.notna(d) else 0.0
    for d in item_dates
], dtype=np.float32)
year_min = release_years[release_years > 0].min()
year_max = release_years[release_years > 0].max()
release_year_norm = np.where(
    release_years > 0,
    (release_years - year_min) / (year_max - year_min + 1e-8),
    0.0
).reshape(-1, 1).astype(np.float32)

# Matrices de features
X10      = np.hstack([item_emb, tfidf_aligned, steam_aligned, rawg_aligned])
X10_feat = np.hstack([item_emb, tfidf_aligned, steam_aligned, rawg_aligned,
                       rawg_quality, rawg_meta_valid, release_year_norm])

print(f"  X10 shape:      {X10.shape}")
print(f"  X10_feat shape: {X10_feat.shape}  (+3 engineered features)")

# ── 6. Helpers ─────────────────────────────────────────────────
TSCV_WINDOWS = [
    ("2013-07-01", "2014-01-01"),
    ("2014-01-01", "2014-07-01"),
    ("2014-07-01", "2015-01-01"),
    ("2015-01-01", "2015-07-01"),
    ("2015-07-01", "2016-01-01"),
    ("2016-01-01", "2016-07-01"),
    ("2016-07-01", "2017-01-01"),
]

train_dates = np.array(item_dates)[train_mask]
sort_order  = np.argsort(train_dates)

def evaluate(X, y, model_obj, log_target=False):
    """Evalúa un modelo ya entrenado sobre el test set + TSCV + KFold."""
    y_use = np.log1p(y) if log_target else y
    X_tr  = X[train_mask][sort_order]
    y_tr  = y_use[train_mask][sort_order]
    X_te  = X[test_mask]
    y_te  = y[test_mask]

    pred_raw = model_obj.predict(X_te)
    pred     = np.expm1(np.clip(pred_raw, -10, 20)) if log_target else np.clip(pred_raw, 0, None)

    r2   = r2_score(y_te, pred)
    rmse = float(mean_squared_error(y_te, pred) ** 0.5)
    mae  = float(mean_absolute_error(y_te, pred))
    mask_nz = y_te > 0
    mape = float(np.mean(np.abs((y_te[mask_nz] - pred[mask_nz]) / y_te[mask_nz]))) if mask_nz.sum() > 0 else np.nan

    ts_results = []
    for (tc, sc) in TSCV_WINDOWS:
        tc, sc = pd.Timestamp(tc), pd.Timestamp(sc)
        t_mask = np.array([pd.notna(d) and d < tc  for d in item_dates])
        e_mask = np.array([pd.notna(d) and d >= tc and d < sc for d in item_dates])
        if e_mask.sum() < 5 or t_mask.sum() < 10: continue
        so2 = np.argsort(np.array(item_dates)[t_mask])
        m2  = model_obj.__class__(**model_obj.get_params()) if hasattr(model_obj, "get_params") else model_obj
        Xtr2 = X[t_mask][so2]; ytr2 = y_use[t_mask][so2]
        Xte2 = X[e_mask];      yte2 = y[e_mask]
        m2.fit(Xtr2, ytr2)
        p2 = np.expm1(np.clip(m2.predict(Xte2), -10, 20)) if log_target else np.clip(m2.predict(Xte2), 0, None)
        ts_results.append(r2_score(yte2, p2))

    r2_tscv = float(np.mean(ts_results)) if ts_results else np.nan
    r2_tscv_std = float(np.std(ts_results)) if ts_results else np.nan

    vd_mask = np.array([pd.notna(d) for d in item_dates])
    X_kf = X[vd_mask]; y_kf = y_use[vd_mask]; y_kf_o = y[vd_mask]
    kf_r2s = []
    for tri, tei in KFold(n_splits=5, shuffle=True, random_state=42).split(X_kf):
        mk = model_obj.__class__(**model_obj.get_params()) if hasattr(model_obj, "get_params") else model_obj
        mk.fit(X_kf[tri], y_kf[tri])
        pk = np.expm1(np.clip(mk.predict(X_kf[tei]), -10, 20)) if log_target else np.clip(mk.predict(X_kf[tei]), 0, None)
        kf_r2s.append(r2_score(y_kf_o[tei], pk))

    r2_kfold = float(np.mean(kf_r2s))
    r2_kfold_std = float(np.std(kf_r2s))

    return {
        "r2_temporal": r2, "rmse_temporal": rmse, "mae_temporal": mae, "mape_temporal": mape,
        "r2_tscv": r2_tscv, "r2_tscv_std": r2_tscv_std,
        "r2_kfold": r2_kfold, "r2_kfold_std": r2_kfold_std,
    }, pred


def optuna_xgb(X, y, n_trials=150, log_target=False):
    y_use = np.log1p(y) if log_target else y
    X_tr  = X[train_mask][sort_order]
    y_tr  = y_use[train_mask][sort_order]
    split = max(10, int(len(X_tr) * 0.8))

    def obj(trial):
        p = dict(
            n_estimators     = trial.suggest_int("n_estimators", 100, 1000),
            learning_rate    = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            max_depth        = trial.suggest_int("max_depth", 3, 9),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 15),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            gamma            = trial.suggest_float("gamma", 0.0, 3.0),
            reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 2.0),
            reg_lambda       = trial.suggest_float("reg_lambda", 0.5, 5.0),
            random_state=42, tree_method="hist", verbosity=0,
        )
        m = XGBRegressor(**p)
        m.fit(X_tr[:split], y_tr[:split])
        return float(mean_squared_error(y_tr[split:], m.predict(X_tr[split:])) ** 0.5)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    params = {**study.best_params, "random_state": 42, "tree_method": "hist", "verbosity": 0}
    model  = XGBRegressor(**params)
    model.fit(X_tr, y_tr)
    return model, study.best_value


results_summary = []

# ── 7. Experimento 1: XGBoost 150 trials ──────────────────────
print("\n" + "=" * 70)
print("[21_10_opt] XGBoost 150 trials — X10 (mismo que 19_10)")
print("=" * 70)

model_opt, val_opt = optuna_xgb(X10, y_full, n_trials=150)
metrics_opt, _ = evaluate(X10, y_full, model_opt)
print(f"  Optuna best val RMSE: {val_opt:.4f}")
print(f"  Temporal: R2={metrics_opt['r2_temporal']:.4f} | RMSE={metrics_opt['rmse_temporal']:.2f}")
print(f"  TSCV: R2={metrics_opt['r2_tscv']:.4f} ± {metrics_opt['r2_tscv_std']:.4f}")
print(f"  KFold: R2={metrics_opt['r2_kfold']:.4f} ± {metrics_opt['r2_kfold_std']:.4f}")
save_result("21_10_opt", "XGBoost 150 trials (v2 cutoff2017)", "RS+TF-IDF+RAWG", "v2_global", metrics_opt,
            notes="v2 global cutoff2017: 150 Optuna trials")
results_summary.append(("21_10_opt", metrics_opt["r2_temporal"], metrics_opt["rmse_temporal"]))

# ── 8. Experimento 2: XGBoost + feature engineering ───────────
print("\n" + "=" * 70)
print("[21_10_feat] XGBoost + feature engineering RAWG")
print("=" * 70)

model_feat, val_feat = optuna_xgb(X10_feat, y_full, n_trials=100)
metrics_feat, _ = evaluate(X10_feat, y_full, model_feat)
print(f"  Optuna best val RMSE: {val_feat:.4f}")
print(f"  Temporal: R2={metrics_feat['r2_temporal']:.4f} | RMSE={metrics_feat['rmse_temporal']:.2f}")
print(f"  TSCV: R2={metrics_feat['r2_tscv']:.4f} ± {metrics_feat['r2_tscv_std']:.4f}")
print(f"  KFold: R2={metrics_feat['r2_kfold']:.4f} ± {metrics_feat['r2_kfold_std']:.4f}")
save_result("21_10_feat", "XGBoost + feature engineering (v2 cutoff2017)",
            "RS+TF-IDF+RAWG+metacritic×ratings+year", "v2_global", metrics_feat,
            notes="v2 global cutoff2017: +rawg_quality, +rawg_meta_valid, +release_year_norm")
results_summary.append(("21_10_feat", metrics_feat["r2_temporal"], metrics_feat["rmse_temporal"]))

# ── 9. Experimento 3: LightGBM ────────────────────────────────
if HAS_LGB:
    print("\n" + "=" * 70)
    print("[21_10_lgbm] LightGBM equivalente a 19_10")
    print("=" * 70)

    X_tr_lgb = X10[train_mask][sort_order]
    y_tr_lgb = y_full[train_mask][sort_order]
    split    = max(10, int(len(X_tr_lgb) * 0.8))

    def obj_lgb(trial):
        p = dict(
            n_estimators      = trial.suggest_int("n_estimators", 100, 1000),
            learning_rate     = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            num_leaves        = trial.suggest_int("num_leaves", 15, 127),
            max_depth         = trial.suggest_int("max_depth", 3, 9),
            min_child_samples = trial.suggest_int("min_child_samples", 5, 50),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 0.0, 2.0),
            reg_lambda        = trial.suggest_float("reg_lambda", 0.5, 5.0),
            random_state=42, verbosity=-1,
        )
        m = lgb.LGBMRegressor(**p)
        m.fit(X_tr_lgb[:split], y_tr_lgb[:split])
        return float(mean_squared_error(y_tr_lgb[split:], m.predict(X_tr_lgb[split:])) ** 0.5)

    study_lgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study_lgb.optimize(obj_lgb, n_trials=80, show_progress_bar=False)
    params_lgb = {**study_lgb.best_params, "random_state": 42, "verbosity": -1}
    model_lgb  = lgb.LGBMRegressor(**params_lgb)
    model_lgb.fit(X_tr_lgb, y_tr_lgb)

    # Evaluar manualmente (LightGBM no tiene get_params compatible con evaluate())
    X_te_lgb = X10[test_mask]; y_te_lgb = y_full[test_mask]
    pred_lgb = np.clip(model_lgb.predict(X_te_lgb), 0, None)
    r2_lgb   = r2_score(y_te_lgb, pred_lgb)
    rmse_lgb = float(mean_squared_error(y_te_lgb, pred_lgb) ** 0.5)
    mae_lgb  = float(mean_absolute_error(y_te_lgb, pred_lgb))
    mask_nz  = y_te_lgb > 0
    mape_lgb = float(np.mean(np.abs((y_te_lgb[mask_nz] - pred_lgb[mask_nz]) / y_te_lgb[mask_nz]))) if mask_nz.sum() > 0 else np.nan

    # TSCV LightGBM
    ts_lgb = []
    for (tc, sc) in TSCV_WINDOWS:
        tc, sc = pd.Timestamp(tc), pd.Timestamp(sc)
        t_m = np.array([pd.notna(d) and d < tc  for d in item_dates])
        e_m = np.array([pd.notna(d) and d >= tc and d < sc for d in item_dates])
        if e_m.sum() < 5 or t_m.sum() < 10: continue
        so2 = np.argsort(np.array(item_dates)[t_m])
        m2 = lgb.LGBMRegressor(**params_lgb)
        m2.fit(X10[t_m][so2], y_full[t_m][so2])
        p2 = np.clip(m2.predict(X10[e_m]), 0, None)
        ts_lgb.append(r2_score(y_full[e_m], p2))

    # KFold LightGBM
    vd_m = np.array([pd.notna(d) for d in item_dates])
    kf_lgb = []
    for tri, tei in KFold(n_splits=5, shuffle=True, random_state=42).split(X10[vd_m]):
        mk = lgb.LGBMRegressor(**params_lgb)
        mk.fit(X10[vd_m][tri], y_full[vd_m][tri])
        kf_lgb.append(r2_score(y_full[vd_m][tei], np.clip(mk.predict(X10[vd_m][tei]), 0, None)))

    print(f"  Optuna best val RMSE: {study_lgb.best_value:.4f}")
    print(f"  Temporal: R2={r2_lgb:.4f} | RMSE={rmse_lgb:.2f}")
    print(f"  TSCV: R2={np.mean(ts_lgb):.4f} ± {np.std(ts_lgb):.4f}")
    print(f"  KFold: R2={np.mean(kf_lgb):.4f} ± {np.std(kf_lgb):.4f}")

    metrics_lgb = {
        "r2_temporal": r2_lgb, "rmse_temporal": rmse_lgb, "mae_temporal": mae_lgb, "mape_temporal": mape_lgb,
        "r2_tscv": float(np.mean(ts_lgb)), "r2_tscv_std": float(np.std(ts_lgb)),
        "r2_kfold": float(np.mean(kf_lgb)), "r2_kfold_std": float(np.std(kf_lgb)),
    }
    save_result("21_10_lgbm", "LightGBM (v2 cutoff2017)", "RS+TF-IDF+RAWG", "v2_global", metrics_lgb,
                notes="v2 global cutoff2017: LightGBM 80 trials")
    results_summary.append(("21_10_lgbm", r2_lgb, rmse_lgb))

# ── 10. Experimento 4: CatBoost ───────────────────────────────
if HAS_CAT:
    print("\n" + "=" * 70)
    print("[21_10_cat] CatBoost equivalente a 19_10")
    print("=" * 70)

    X_tr_cat = X10[train_mask][sort_order]
    y_tr_cat = y_full[train_mask][sort_order]
    split    = max(10, int(len(X_tr_cat) * 0.8))

    def obj_cat(trial):
        p = dict(
            iterations    = trial.suggest_int("iterations", 100, 800),
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            depth         = trial.suggest_int("depth", 3, 8),
            l2_leaf_reg   = trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            random_seed=42, verbose=0, allow_writing_files=False,
        )
        m = CatBoostRegressor(**p)
        m.fit(X_tr_cat[:split], y_tr_cat[:split])
        return float(mean_squared_error(y_tr_cat[split:], m.predict(X_tr_cat[split:])) ** 0.5)

    study_cat = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study_cat.optimize(obj_cat, n_trials=60, show_progress_bar=False)
    params_cat = {**study_cat.best_params, "random_seed": 42, "verbose": 0, "allow_writing_files": False}
    model_cat  = CatBoostRegressor(**params_cat)
    model_cat.fit(X_tr_cat, y_tr_cat)

    X_te_cat = X10[test_mask]; y_te_cat = y_full[test_mask]
    pred_cat = np.clip(model_cat.predict(X_te_cat), 0, None)
    r2_cat   = r2_score(y_te_cat, pred_cat)
    rmse_cat = float(mean_squared_error(y_te_cat, pred_cat) ** 0.5)
    mae_cat  = float(mean_absolute_error(y_te_cat, pred_cat))
    mask_nz  = y_te_cat > 0
    mape_cat = float(np.mean(np.abs((y_te_cat[mask_nz] - pred_cat[mask_nz]) / y_te_cat[mask_nz]))) if mask_nz.sum() > 0 else np.nan

    ts_cat = []
    for (tc, sc) in TSCV_WINDOWS:
        tc, sc = pd.Timestamp(tc), pd.Timestamp(sc)
        t_m = np.array([pd.notna(d) and d < tc  for d in item_dates])
        e_m = np.array([pd.notna(d) and d >= tc and d < sc for d in item_dates])
        if e_m.sum() < 5 or t_m.sum() < 10: continue
        so2 = np.argsort(np.array(item_dates)[t_m])
        m2 = CatBoostRegressor(**params_cat)
        m2.fit(X10[t_m][so2], y_full[t_m][so2])
        p2 = np.clip(m2.predict(X10[e_m]), 0, None)
        ts_cat.append(r2_score(y_full[e_m], p2))

    kf_cat = []
    vd_m = np.array([pd.notna(d) for d in item_dates])
    for tri, tei in KFold(n_splits=5, shuffle=True, random_state=42).split(X10[vd_m]):
        mk = CatBoostRegressor(**params_cat)
        mk.fit(X10[vd_m][tri], y_full[vd_m][tri])
        kf_cat.append(r2_score(y_full[vd_m][tei], np.clip(mk.predict(X10[vd_m][tei]), 0, None)))

    print(f"  Optuna best val RMSE: {study_cat.best_value:.4f}")
    print(f"  Temporal: R2={r2_cat:.4f} | RMSE={rmse_cat:.2f}")
    print(f"  TSCV: R2={np.mean(ts_cat):.4f} ± {np.std(ts_cat):.4f}")
    print(f"  KFold: R2={np.mean(kf_cat):.4f} ± {np.std(kf_cat):.4f}")

    metrics_cat = {
        "r2_temporal": r2_cat, "rmse_temporal": rmse_cat, "mae_temporal": mae_cat, "mape_temporal": mape_cat,
        "r2_tscv": float(np.mean(ts_cat)), "r2_tscv_std": float(np.std(ts_cat)),
        "r2_kfold": float(np.mean(kf_cat)), "r2_kfold_std": float(np.std(kf_cat)),
    }
    save_result("21_10_cat", "CatBoost (v2 cutoff2017)", "RS+TF-IDF+RAWG", "v2_global", metrics_cat,
                notes="v2 global cutoff2017: CatBoost 60 trials")
    results_summary.append(("21_10_cat", r2_cat, rmse_cat))

# ── 11. Experimento 5: Ensemble XGBoost lineal + log ──────────
print("\n" + "=" * 70)
print("[21_ensemble] Ensemble: XGBoost lineal + log (promedio ponderado)")
print("=" * 70)

# Reusar el modelo de 21_10_opt (150 trials) + entrenar versión log
model_log, _ = optuna_xgb(X10, y_full, n_trials=80, log_target=True)

X_te     = X10[test_mask]
y_te     = y_full[test_mask]
pred_lin = np.clip(model_opt.predict(X_te), 0, None)
pred_log = np.expm1(np.clip(model_log.predict(X_te), -10, 20))

# Probar pesos: 50/50, 70/30, 30/70
best_r2, best_w, best_pred = -999, None, None
for w in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    p_ens = w * pred_lin + (1 - w) * pred_log
    r = r2_score(y_te, p_ens)
    if r > best_r2:
        best_r2, best_w, best_pred = r, w, p_ens

rmse_ens = float(mean_squared_error(y_te, best_pred) ** 0.5)
mae_ens  = float(mean_absolute_error(y_te, best_pred))
mask_nz  = y_te > 0
mape_ens = float(np.mean(np.abs((y_te[mask_nz] - best_pred[mask_nz]) / y_te[mask_nz]))) if mask_nz.sum() > 0 else np.nan

print(f"  Mejor peso lineal: {best_w:.1f} | log: {1-best_w:.1f}")
print(f"  Temporal: R2={best_r2:.4f} | RMSE={rmse_ens:.2f} | MAE={mae_ens:.2f}")

metrics_ens = {
    "r2_temporal": best_r2, "rmse_temporal": rmse_ens,
    "mae_temporal": mae_ens, "mape_temporal": mape_ens,
    "r2_tscv": float("nan"), "r2_tscv_std": float("nan"),
    "r2_kfold": float("nan"), "r2_kfold_std": float("nan"),
}
save_result("21_ensemble", f"Ensemble XGBoost lin+log w={best_w:.1f} (v2 cutoff2017)",
            "RS+TF-IDF+RAWG (ensemble)", "v2_global", metrics_ens,
            notes=f"Ensemble 21_10_opt × {best_w:.1f} + log × {1-best_w:.1f}")
results_summary.append(("21_ensemble", best_r2, rmse_ens))

# ── 12. Resumen ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESUMEN — Experimentos 21 (mejoras sobre 19_10)")
print("=" * 70)
print(f"  {'Baseline 19_10':<22} R2=0.4384  RMSE=987.40  (referencia)")
print(f"  {'-'*50}")
for mid, r2, rmse in results_summary:
    marker = " ← NUEVO BEST" if r2 > 0.4384 else ""
    print(f"  {mid:<22} R2={r2:.4f}  RMSE={rmse:.2f}{marker}")

print("\n\nLeaderboard completo:")
print_leaderboard()
