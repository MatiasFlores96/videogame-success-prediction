"""
24_ablation_catboost.py
========================
Dos experimentos paralelos sobre el dataset global v2 (cutoff 2017):

A) Ablation study: ¿cuánto aporta el RS?
   Corre modelos SOLO-CONTENIDO (sin ningún embedding RS) para establecer
   una baseline limpia. Luego la comparación directa es:
     content-only (abl_xx)  vs  RS+content (23_05 = R²=0.368)

B) CatBoost sobre 23_05:
   En la serie 21_xx, CatBoost superó XGBoost (+0.009 R²).
   Aplica CatBoost al mejor set de features limpio (content-aug emb + metadata).

Modelos:
  abl_01  Metadata Only                   meta (6d)
  abl_02  TF-IDF + Numeric                tfidf (100d) + steam (2d)
  abl_03  TF-IDF + Numeric + RAWG         tfidf (100d) + steam (2d) + rawg (11d)
  abl_04  Meta + RAWG                     meta (6d) + rawg (11d)
  abl_05  Full content (sin RS)           tfidf (100d) + steam (2d) + meta (6d) + rawg (11d)
  24_05_cat  CatBoost — RS content-aug + Meta   emb_v2 (64d) + meta (6d)
  24_10_cat  CatBoost — RS content-aug + RAWG   emb_v2 (64d) + meta (6d) + rawg (11d)
"""

import ast, json, os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_tracker import save_result, print_leaderboard

BASE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA   = os.path.join(BASE, "Data")
CUTOFF = pd.Timestamp("2017-01-01")

print("=" * 70)
print("24_ablation_catboost.py — Ablation RS + CatBoost content-aug v2")
print("=" * 70)

# ── 1. Datos base ──────────────────────────────────────────────────────────────
print("\n[1] Cargando datos base...")
with open(os.path.join(DATA, "item2idx_v2.json")) as f:
    item2idx_v2 = {k: int(v) for k, v in json.load(f).items()}
N = len(item2idx_v2)

df_inter = pd.read_parquet(os.path.join(DATA, "interactions_v2.parquet"))
target_s = df_inter.groupby("item_idx").size().reindex(range(N), fill_value=0)
y_full   = target_s.values.astype(np.float64)
print(f"  Items: {N:,} | Target mean={y_full.mean():.1f}")

# Embeddings content-aug v2
item_emb = np.load(os.path.join(DATA, "item_embeddings_rs_v2_content2.npy"))
print(f"  Embeddings content-aug v2: {item_emb.shape}")

# ── 2. Metadata y fechas ───────────────────────────────────────────────────────
print("\n[2] Cargando metadata y fechas...")
games_raw = []
with open(os.path.join(DATA, "steam_games.json"), "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: games_raw.append(ast.literal_eval(line))
        except: pass

df_games = pd.json_normalize(games_raw).rename(columns={"id": "item_id"})
df_games["item_id"]   = df_games["item_id"].astype(str)
df_games["item_idx"]  = df_games["item_id"].map(item2idx_v2)
df_games              = df_games.dropna(subset=["item_idx"])
df_games["item_idx"]  = df_games["item_idx"].astype(int)
df_games["release_date_parsed"] = pd.to_datetime(df_games["release_date"], errors="coerce")
df_games = df_games.drop_duplicates(subset=["item_idx"], keep="first")

meta       = df_games.set_index("item_idx").reindex(range(N))
item_dates = meta["release_date_parsed"].values

train_mask = np.array([pd.notna(d) and d < CUTOFF  for d in item_dates])
test_mask  = np.array([pd.notna(d) and d >= CUTOFF for d in item_dates])
print(f"  Train: {train_mask.sum():,}  |  Test: {test_mask.sum():,}")

# ── 3. Features de contenido ───────────────────────────────────────────────────
print("\n[3] Construyendo features de contenido...")

def parse_list_col(x):
    if x is None or (isinstance(x, float) and pd.isna(x)): return []
    if isinstance(x, list): return [str(t).strip() for t in x if t]
    try:
        lst = eval(x)
        return [str(t).strip() for t in lst if t] if isinstance(lst, list) else []
    except: return []

def clean_price(p):
    if pd.isna(p) or str(p).strip() in ("Free", "", "Free to Play"): return 0.0
    try: return float(str(p).replace("$", "").replace(",", "").strip())
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

META_COLS   = ["price_num", "ea_flag", "num_genres", "num_tags", "num_specs", "has_senti"]
meta_reind  = df_games.drop_duplicates(subset=["item_idx"], keep="first").set_index("item_idx").reindex(range(N))

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

print(f"  TF-IDF (100d) | Meta (6d) | Steam (2d)")

# ── 4. RAWG features ───────────────────────────────────────────────────────────
rawg_aligned = None
rawg_path = os.path.join(DATA, "rawg_enriched.csv")
v1map_path = os.path.join(DATA, "item2idx.json")
if os.path.exists(rawg_path) and os.path.exists(v1map_path):
    print("\n[4] Mapeando RAWG features...")
    with open(v1map_path) as f:
        item2idx_v1 = {k: int(v) for k, v in json.load(f).items()}
    v1_idx_to_appid = {v: k for k, v in item2idx_v1.items()}

    df_rawg = pd.read_csv(rawg_path).drop_duplicates(subset=["item_idx"], keep="last")
    df_rawg["appid_str"]    = df_rawg["item_idx"].map(v1_idx_to_appid)
    df_rawg["item_idx_v2"]  = df_rawg["appid_str"].map(item2idx_v2)
    df_rawg = df_rawg.dropna(subset=["item_idx_v2"])
    df_rawg["item_idx_v2"]  = df_rawg["item_idx_v2"].astype(int)
    df_rawg = df_rawg.set_index("item_idx_v2")

    ESRB_ORDER = {"Everyone":1,"Everyone 10+":2,"Teen":3,"Mature":4,"Adults Only":5}
    rawg_aligned = np.zeros((N, 11), dtype=np.float32)
    matched = 0
    for idx in range(N):
        if idx not in df_rawg.index: continue
        r = df_rawg.loc[idx]
        rawg_aligned[idx] = [
            1.0 if pd.notna(r.get("rawg_id")) else 0.0,
            float(r["rawg_rating"])      if pd.notna(r.get("rawg_rating"))      else -1.0,
            0.0 if pd.notna(r.get("rawg_rating"))   else 1.0,
            float(r["metacritic"])       if pd.notna(r.get("metacritic"))       else -1.0,
            0.0 if pd.notna(r.get("metacritic"))    else 1.0,
            np.log1p(float(r["playtime_avg_h"]))    if pd.notna(r.get("playtime_avg_h"))    else 0.0,
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
    print(f"  RAWG: {matched:,}/{N:,} items ({matched/N*100:.1f}%)")
else:
    print("\n[4] RAWG no disponible")

# ── 5. Helpers ─────────────────────────────────────────────────────────────────
TSCV_WINDOWS = [
    ("2013-07-01", "2014-01-01"), ("2014-01-01", "2014-07-01"),
    ("2014-07-01", "2015-01-01"), ("2015-01-01", "2015-07-01"),
    ("2015-07-01", "2016-01-01"), ("2016-01-01", "2016-07-01"),
    ("2016-07-01", "2017-01-01"),
]

def optuna_xgb(X_tr, y_tr, n_trials=60):
    split_idx = max(10, int(len(X_tr) * 0.8))
    Xt, Xv = X_tr[:split_idx], X_tr[split_idx:]
    yt, yv = y_tr[:split_idx], y_tr[split_idx:]
    def objective(trial):
        p = dict(
            n_estimators     = trial.suggest_int("n_estimators", 100, 800),
            learning_rate    = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            max_depth        = trial.suggest_int("max_depth", 3, 8),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            gamma            = trial.suggest_float("gamma", 0.0, 2.0),
            random_state=42, tree_method="hist", verbosity=0,
        )
        m = XGBRegressor(**p)
        m.fit(Xt, yt)
        return float(mean_squared_error(yv, m.predict(Xv)) ** 0.5)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {**study.best_params, "random_state": 42, "tree_method": "hist", "verbosity": 0}

def optuna_cat(X_tr, y_tr, n_trials=60):
    split_idx = max(10, int(len(X_tr) * 0.8))
    Xt, Xv = X_tr[:split_idx], X_tr[split_idx:]
    yt, yv = y_tr[:split_idx], y_tr[split_idx:]
    def objective(trial):
        p = dict(
            iterations       = trial.suggest_int("iterations", 200, 1000),
            learning_rate    = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            depth            = trial.suggest_int("depth", 3, 8),
            l2_leaf_reg      = trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bylevel= trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            random_seed=42, verbose=0,
        )
        m = CatBoostRegressor(**p)
        m.fit(Xt, yt, eval_set=(Xv, yv), early_stopping_rounds=50, verbose=False)
        return float(mean_squared_error(yv, m.predict(Xv)) ** 0.5)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {**study.best_params, "random_seed": 42, "verbose": 0}


def run_model(X, y, model_id, model_name, features_desc, log_target=False,
              n_trials=60, use_catboost=False):
    print(f"\n{'='*70}")
    print(f"[{model_id}] {model_name}  |  shape={X.shape}  |  log={log_target}  |  "
          f"{'CatBoost' if use_catboost else 'XGBoost'}")
    print(f"{'='*70}")

    y_use = np.log1p(y) if log_target else y

    train_dates = np.array(item_dates)[train_mask]
    sort_order  = np.argsort(train_dates)
    X_tr = X[train_mask][sort_order]
    y_tr = y_use[train_mask][sort_order]
    X_te = X[test_mask]
    y_te = y[test_mask]

    if use_catboost:
        best_params = optuna_cat(X_tr, y_tr, n_trials)
        model = CatBoostRegressor(**best_params)
        model.fit(X_tr, y_tr, verbose=False)
    else:
        best_params = optuna_xgb(X_tr, y_tr, n_trials)
        model = XGBRegressor(**best_params)
        model.fit(X_tr, y_tr)

    pred_raw = model.predict(X_te)
    pred     = np.expm1(np.clip(pred_raw, -10, 20)) if log_target else np.clip(pred_raw, 0, None)

    r2_orig   = r2_score(y_te, pred)
    rmse_orig = float(mean_squared_error(y_te, pred) ** 0.5)
    mae_orig  = float(mean_absolute_error(y_te, pred))
    mask_nz   = y_te > 0
    mape_orig = float(np.mean(np.abs((y_te[mask_nz] - pred[mask_nz]) / y_te[mask_nz]))) if mask_nz.sum() > 0 else np.nan
    r2_log    = r2_score(np.log1p(y_te), pred_raw) if log_target else None

    print(f"  Temporal: R2={r2_orig:.4f} | RMSE={rmse_orig:.2f} | MAE={mae_orig:.2f}")
    if r2_log is not None:
        print(f"  Log-space: R2_log={r2_log:.4f}")

    # TSCV
    ts_results = []
    for (tc, sc) in TSCV_WINDOWS:
        tc, sc = pd.Timestamp(tc), pd.Timestamp(sc)
        t_mask = np.array([pd.notna(d) and d < tc for d in item_dates])
        e_mask = np.array([pd.notna(d) and d >= tc and d < sc for d in item_dates])
        if e_mask.sum() < 5 or t_mask.sum() < 10: continue
        so2 = np.argsort(np.array(item_dates)[t_mask])
        Xtr2 = X[t_mask][so2];  ytr2 = y_use[t_mask][so2]
        Xte2 = X[e_mask];       yte2 = y[e_mask]
        if use_catboost:
            m2 = CatBoostRegressor(**best_params)
            m2.fit(Xtr2, ytr2, verbose=False)
        else:
            m2 = XGBRegressor(**best_params)
            m2.fit(Xtr2, ytr2)
        p2 = np.expm1(np.clip(m2.predict(Xte2), -10, 20)) if log_target else np.clip(m2.predict(Xte2), 0, None)
        ts_results.append({"R2": r2_score(yte2, p2), "RMSE": float(mean_squared_error(yte2, p2)**0.5)})

    if ts_results:
        ts_df     = pd.DataFrame(ts_results)
        r2_tscv   = ts_df["R2"].mean();   r2_tscv_std   = ts_df["R2"].std()
        rmse_tscv = ts_df["RMSE"].mean(); rmse_tscv_std = ts_df["RMSE"].std()
        print(f"  TSCV: R2={r2_tscv:.4f} ± {r2_tscv_std:.4f}")
    else:
        r2_tscv = r2_tscv_std = rmse_tscv = rmse_tscv_std = float("nan")

    # KFold
    vd_mask  = np.array([pd.notna(d) for d in item_dates])
    X_kf     = X[vd_mask];  y_kf = y_use[vd_mask]; y_kf_orig = y[vd_mask]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kf_results = []
    for tri, tei in kf.split(X_kf):
        if use_catboost:
            mk = CatBoostRegressor(**best_params)
            mk.fit(X_kf[tri], y_kf[tri], verbose=False)
        else:
            mk = XGBRegressor(**best_params)
            mk.fit(X_kf[tri], y_kf[tri])
        pk = np.expm1(np.clip(mk.predict(X_kf[tei]), -10, 20)) if log_target else np.clip(mk.predict(X_kf[tei]), 0, None)
        kf_results.append({"R2": r2_score(y_kf_orig[tei], pk),
                           "RMSE": float(mean_squared_error(y_kf_orig[tei], pk)**0.5)})
    kf_df     = pd.DataFrame(kf_results)
    r2_kfold  = kf_df["R2"].mean();  r2_kfold_std  = kf_df["R2"].std()
    rmse_kfold = kf_df["RMSE"].mean()
    print(f"  KFold: R2={r2_kfold:.4f} ± {r2_kfold_std:.4f}")

    metrics = {
        "r2_temporal": r2_orig,   "rmse_temporal": rmse_orig,
        "mae_temporal": mae_orig, "mape_temporal": mape_orig,
        "r2_tscv":      r2_tscv,  "r2_tscv_std":   r2_tscv_std,
        "rmse_tscv":    rmse_tscv,"rmse_tscv_std":  rmse_tscv_std,
        "r2_kfold":     r2_kfold, "r2_kfold_std":   r2_kfold_std,
        "rmse_kfold":   rmse_kfold,
    }
    if r2_log is not None:
        metrics["r2_temporal_log"] = r2_log

    algo = "CatBoost" if use_catboost else "XGBoost"
    save_result(model_id=model_id, model_name=model_name,
                features=features_desc, embeddings="v2_global",
                metrics=metrics,
                notes=f"v2 cutoff2017 {algo} ablation/catboost: train={train_mask.sum()} test={test_mask.sum()}")
    return r2_orig, rmse_orig


# ── 6. Feature sets ────────────────────────────────────────────────────────────
# --- Ablation (sin RS embeddings) ---
X_abl_01 = meta_aligned                                         # 6d
X_abl_02 = np.hstack([tfidf_aligned, steam_aligned])           # 102d
X_abl_03 = np.hstack([tfidf_aligned, steam_aligned,
                       rawg_aligned]) if rawg_aligned is not None else None  # 113d
X_abl_04 = np.hstack([meta_aligned,
                       rawg_aligned]) if rawg_aligned is not None else None  # 17d
X_abl_05 = np.hstack([tfidf_aligned, steam_aligned, meta_aligned,
                       rawg_aligned]) if rawg_aligned is not None else None  # 119d

# --- CatBoost con content-aug v2 ---
X05 = np.hstack([item_emb, meta_aligned])                       # 70d
X10 = np.hstack([item_emb, tfidf_aligned, steam_aligned,
                 rawg_aligned]) if rawg_aligned is not None else None  # 177d

# ── 7. Correr experimentos ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("A) ABLATION — modelos sin RS embeddings")
print("=" * 70)

results = []

r2, rmse = run_model(X_abl_01, y_full, "abl_01", "Metadata Only (ablation)",
                     "meta (6d)")
results.append(("abl_01", r2, rmse))

r2, rmse = run_model(X_abl_02, y_full, "abl_02", "TF-IDF + Numeric (ablation)",
                     "tfidf (100d) + steam (2d)")
results.append(("abl_02", r2, rmse))

if X_abl_03 is not None:
    r2, rmse = run_model(X_abl_03, y_full, "abl_03", "TF-IDF + Numeric + RAWG (ablation)",
                         "tfidf (100d) + steam (2d) + rawg (11d)")
    results.append(("abl_03", r2, rmse))

if X_abl_04 is not None:
    r2, rmse = run_model(X_abl_04, y_full, "abl_04", "Meta + RAWG (ablation)",
                         "meta (6d) + rawg (11d)")
    results.append(("abl_04", r2, rmse))

if X_abl_05 is not None:
    r2, rmse = run_model(X_abl_05, y_full, "abl_05", "Full content sin RS (ablation)",
                         "tfidf (100d) + steam (2d) + meta (6d) + rawg (11d)")
    results.append(("abl_05", r2, rmse))

print("\n" + "=" * 70)
print("B) CATBOOST — content-aug v2 embeddings")
print("=" * 70)

r2, rmse = run_model(X05, y_full, "24_05_cat", "CatBoost RS content-aug v2 + Meta",
                     "RS content-aug v2 (64d) + meta (6d)", use_catboost=True)
results.append(("24_05_cat", r2, rmse))

if X10 is not None:
    r2, rmse = run_model(X10, y_full, "24_10_cat", "CatBoost RS content-aug v2 + RAWG",
                         "RS content-aug v2 (64d) + tfidf (100d) + steam (2d) + rawg (11d)",
                         use_catboost=True)
    results.append(("24_10_cat", r2, rmse))

# ── 8. Resumen ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESUMEN — Ablation + CatBoost")
print("=" * 70)
print(f"{'Modelo':<20} {'R2':>8} {'RMSE':>10}  Descripción")
print("-" * 70)
for mid, r2, rmse in results:
    print(f"  {mid:<18} {r2:>8.4f} {rmse:>10.2f}")

# Referencia
print("\nReferencia:")
print(f"  {'23_05 (XGB)':<18} {'0.3676':>8} {'1047.78':>10}  RS content-aug v2 + Meta")
print(f"  {'20_10 (XGB)':<18} {'0.1474':>8} {'1216.58':>10}  Estándar + RAWG (sin RS ratings)")
print(f"  {'21_10_cat':<18} {'0.4473':>8} {'979.50':>10}  CatBoost leakeado (referencia)")

print("\n\nLeaderboard completo:")
print_leaderboard()
