"""
31_log_transform.py
====================
Experimento log-transform sobre el resultado principal.

Target: log1p(y_full) en lugar de y_full.
Evaluación: se reporta tanto en escala original (back-transform) como en log-space.

Modelos:
  24_05_cat_log   CatBoost — RS content-aug v2 + Meta  [log-target]
  abl_04_log      Meta + RAWG (best content-only)      [log-target]
  abl_01_log      Metadata Only                        [log-target]

Referencia guardada:
  24_05_cat   R²=0.407 (escala original, sin log)
  abl_04      R²=0.314 (escala original, sin log)
  abl_01      R²=0.111 (escala original, sin log)
"""

import ast, json, os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_tracker import save_result

BASE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA   = os.path.join(BASE, "Data")
CUTOFF = pd.Timestamp("2017-01-01")

print("=" * 70)
print("31_log_transform.py — Log-transform del target (CatBoost)")
print("=" * 70)

# ── 1. Datos base ──────────────────────────────────────────────────────────────
print("\n[1] Cargando datos...")
with open(os.path.join(DATA, "item2idx_v2.json")) as f:
    item2idx_v2 = {k: int(v) for k, v in json.load(f).items()}
N = len(item2idx_v2)

df_inter = pd.read_parquet(os.path.join(DATA, "interactions_v2.parquet"))
target_s = df_inter.groupby("item_idx").size().reindex(range(N), fill_value=0)
y_full   = target_s.values.astype(np.float64)
print(f"  Items: {N:,} | y mean={y_full.mean():.1f} | y max={y_full.max():.0f}")
print(f"  log1p(y) mean={np.log1p(y_full).mean():.3f} | log1p(y) max={np.log1p(y_full).max():.3f}")

item_emb = np.load(os.path.join(DATA, "item_embeddings_rs_v2_content2.npy"))
print(f"  Embeddings: {item_emb.shape}")

# ── 2. Fechas y máscaras ───────────────────────────────────────────────────────
print("\n[2] Fechas y split temporal...")
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

# ── 3. Features ────────────────────────────────────────────────────────────────
print("\n[3] Construyendo features...")

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

META_COLS  = ["price_num", "ea_flag", "num_genres", "num_tags", "num_specs", "has_senti"]
meta_reind = df_games.drop_duplicates(subset=["item_idx"], keep="first").set_index("item_idx").reindex(range(N))
meta_aligned = np.zeros((N, 6), dtype=np.float32)
for idx in range(N):
    row = meta_reind.iloc[idx] if idx < len(meta_reind) else None
    if row is not None:
        for j, col in enumerate(META_COLS):
            v = row.get(col, 0)
            meta_aligned[idx, j] = float(v) if pd.notna(v) else 0.0

# RAWG
rawg_aligned = None
rawg_path  = os.path.join(DATA, "rawg_enriched.csv")
v1map_path = os.path.join(DATA, "item2idx.json")
if os.path.exists(rawg_path) and os.path.exists(v1map_path):
    print("  Cargando RAWG...")
    with open(v1map_path) as f:
        item2idx_v1 = {k: int(v) for k, v in json.load(f).items()}
    v1_idx_to_appid = {v: k for k, v in item2idx_v1.items()}
    df_rawg = pd.read_csv(rawg_path).drop_duplicates(subset=["item_idx"], keep="last")
    df_rawg["appid_str"]   = df_rawg["item_idx"].map(v1_idx_to_appid)
    df_rawg["item_idx_v2"] = df_rawg["appid_str"].map(item2idx_v2)
    df_rawg = df_rawg.dropna(subset=["item_idx_v2"])
    df_rawg["item_idx_v2"] = df_rawg["item_idx_v2"].astype(int)
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
    print(f"  RAWG: {matched:,}/{N:,} items")

print(f"  Features listos: meta={meta_aligned.shape}  emb={item_emb.shape}")

# ── 4. TSCV windows ────────────────────────────────────────────────────────────
TSCV_WINDOWS = [
    ("2013-07-01", "2014-01-01"), ("2014-01-01", "2014-07-01"),
    ("2014-07-01", "2015-01-01"), ("2015-01-01", "2015-07-01"),
    ("2015-07-01", "2016-01-01"), ("2016-01-01", "2016-07-01"),
    ("2016-07-01", "2017-01-01"),
]

# ── 5. run_model ───────────────────────────────────────────────────────────────
def run_model(X, y, model_id, model_name, features_desc, log_target=True, n_trials=60):
    print(f"\n{'='*70}")
    print(f"[{model_id}] {model_name}")
    print(f"  shape={X.shape} | log_target={log_target} | CatBoost | trials={n_trials}")
    print(f"{'='*70}")

    y_use = np.log1p(y) if log_target else y

    train_dates = np.array(item_dates)[train_mask]
    sort_order  = np.argsort(train_dates)
    X_tr = X[train_mask][sort_order]
    y_tr = y_use[train_mask][sort_order]
    X_te = X[test_mask]
    y_te = y[test_mask]

    # Optuna
    split_idx = max(10, int(len(X_tr) * 0.8))
    Xt, Xv = X_tr[:split_idx], X_tr[split_idx:]
    yt, yv = y_tr[:split_idx], y_tr[split_idx:]

    def objective(trial):
        p = dict(
            iterations        = trial.suggest_int("iterations", 200, 1000),
            learning_rate     = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            depth             = trial.suggest_int("depth", 3, 8),
            l2_leaf_reg       = trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            random_seed=42, verbose=0,
        )
        m = CatBoostRegressor(**p)
        m.fit(Xt, yt, eval_set=(Xv, yv), early_stopping_rounds=50, verbose=False)
        return float(mean_squared_error(yv, m.predict(Xv)) ** 0.5)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = {**study.best_params, "random_seed": 42, "verbose": 0}
    print(f"  Optuna best val={'log-RMSE' if log_target else 'RMSE'}: {study.best_value:.4f}")
    print(f"  Params: {best_params}")

    model = CatBoostRegressor(**best_params)
    model.fit(X_tr, y_tr, verbose=False)

    pred_raw = model.predict(X_te)
    pred_orig = np.expm1(np.clip(pred_raw, -10, 20)) if log_target else np.clip(pred_raw, 0, None)

    # Métricas en escala original
    r2_orig   = r2_score(y_te, pred_orig)
    rmse_orig = float(mean_squared_error(y_te, pred_orig) ** 0.5)
    mae_orig  = float(mean_absolute_error(y_te, pred_orig))
    mask_nz   = y_te > 0
    mape_orig = float(np.mean(np.abs((y_te[mask_nz] - pred_orig[mask_nz]) / y_te[mask_nz]))) if mask_nz.sum() > 0 else np.nan

    # Métricas en log-space
    r2_log   = r2_score(np.log1p(y_te), pred_raw)
    rmse_log = float(mean_squared_error(np.log1p(y_te), pred_raw) ** 0.5)

    print(f"\n  ── ESCALA ORIGINAL ──")
    print(f"  Temporal: R²={r2_orig:.4f} | RMSE={rmse_orig:.2f} | MAE={mae_orig:.2f} | MAPE={mape_orig:.1f}%")
    print(f"\n  ── LOG-SPACE ──")
    print(f"  Temporal: R²_log={r2_log:.4f} | RMSE_log={rmse_log:.4f}")
    print(f"\n  Referencia sin log: 24_05_cat R²=0.4067 | RMSE=1014.86")

    # TSCV (evaluado en escala original)
    ts_results = []
    for (tc, sc) in TSCV_WINDOWS:
        tc, sc = pd.Timestamp(tc), pd.Timestamp(sc)
        t_mask = np.array([pd.notna(d) and d < tc for d in item_dates])
        e_mask = np.array([pd.notna(d) and d >= tc and d < sc for d in item_dates])
        if e_mask.sum() < 5 or t_mask.sum() < 10: continue
        so2    = np.argsort(np.array(item_dates)[t_mask])
        Xtr2   = X[t_mask][so2];  ytr2 = y_use[t_mask][so2]
        Xte2   = X[e_mask];       yte2 = y[e_mask]
        m2 = CatBoostRegressor(**best_params)
        m2.fit(Xtr2, ytr2, verbose=False)
        p2 = np.expm1(np.clip(m2.predict(Xte2), -10, 20)) if log_target else np.clip(m2.predict(Xte2), 0, None)
        p2_log = m2.predict(Xte2)
        ts_results.append({
            "R2":     r2_score(yte2, p2),
            "RMSE":   float(mean_squared_error(yte2, p2) ** 0.5),
            "R2_log": r2_score(np.log1p(yte2), p2_log),
        })

    if ts_results:
        ts_df       = pd.DataFrame(ts_results)
        r2_tscv     = ts_df["R2"].mean();     r2_tscv_std     = ts_df["R2"].std()
        rmse_tscv   = ts_df["RMSE"].mean();   rmse_tscv_std   = ts_df["RMSE"].std()
        r2_log_tscv = ts_df["R2_log"].mean()
        print(f"  TSCV (orig):    R²={r2_tscv:.4f} ± {r2_tscv_std:.4f}")
        print(f"  TSCV (log):     R²_log={r2_log_tscv:.4f}")
    else:
        r2_tscv = r2_tscv_std = rmse_tscv = rmse_tscv_std = r2_log_tscv = float("nan")

    # KFold
    vd_mask = np.array([pd.notna(d) for d in item_dates])
    X_kf    = X[vd_mask]; y_kf = y_use[vd_mask]; y_kf_orig = y[vd_mask]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kf_results = []
    for tri, tei in kf.split(X_kf):
        mk = CatBoostRegressor(**best_params)
        mk.fit(X_kf[tri], y_kf[tri], verbose=False)
        pk      = np.expm1(np.clip(mk.predict(X_kf[tei]), -10, 20)) if log_target else np.clip(mk.predict(X_kf[tei]), 0, None)
        pk_log  = mk.predict(X_kf[tei])
        kf_results.append({
            "R2":     r2_score(y_kf_orig[tei], pk),
            "RMSE":   float(mean_squared_error(y_kf_orig[tei], pk) ** 0.5),
            "R2_log": r2_score(np.log1p(y_kf_orig[tei]), pk_log),
        })
    kf_df       = pd.DataFrame(kf_results)
    r2_kfold    = kf_df["R2"].mean();  r2_kfold_std    = kf_df["R2"].std()
    rmse_kfold  = kf_df["RMSE"].mean()
    r2_log_kfold= kf_df["R2_log"].mean()
    print(f"  KFold (orig):   R²={r2_kfold:.4f} ± {r2_kfold_std:.4f}")
    print(f"  KFold (log):    R²_log={r2_log_kfold:.4f}")

    metrics = {
        "r2_temporal":      r2_orig,     "rmse_temporal":     rmse_orig,
        "mae_temporal":     mae_orig,    "mape_temporal":      mape_orig,
        "r2_temporal_log":  r2_log,      "rmse_temporal_log":  rmse_log,
        "r2_tscv":          r2_tscv,     "r2_tscv_std":        r2_tscv_std,
        "rmse_tscv":        rmse_tscv,   "rmse_tscv_std":      rmse_tscv_std,
        "r2_tscv_log":      r2_log_tscv,
        "r2_kfold":         r2_kfold,    "r2_kfold_std":       r2_kfold_std,
        "rmse_kfold":       rmse_kfold,  "r2_kfold_log":       r2_log_kfold,
    }
    save_result(model_id=model_id, model_name=model_name,
                features=features_desc, embeddings="v2_global",
                metrics=metrics,
                notes=f"log-transform CatBoost: train={train_mask.sum()} test={test_mask.sum()}")
    return r2_orig, rmse_orig, r2_log, rmse_log


# ── 6. Experimentos ────────────────────────────────────────────────────────────
results = []

# Principal: RS + Meta (log)
X05 = np.hstack([item_emb, meta_aligned])
r2, rmse, r2_log, rmse_log = run_model(
    X05, y_full,
    "24_05_cat_log", "CatBoost RS content-aug v2 + Meta [log]",
    "RS content-aug v2 (64d) + meta (6d) [log-target]"
)
results.append(("24_05_cat_log", r2, rmse, r2_log))

# Mejor baseline puro-contenido (log) — para comparación directa
if rawg_aligned is not None:
    X_abl04 = np.hstack([meta_aligned, rawg_aligned])
    r2, rmse, r2_log, rmse_log = run_model(
        X_abl04, y_full,
        "abl_04_log", "Meta + RAWG [log]",
        "meta (6d) + rawg (11d) [log-target]"
    )
    results.append(("abl_04_log", r2, rmse, r2_log))

# Metadata only (log) — baseline más simple
r2, rmse, r2_log, rmse_log = run_model(
    meta_aligned, y_full,
    "abl_01_log", "Metadata Only [log]",
    "meta (6d) [log-target]"
)
results.append(("abl_01_log", r2, rmse, r2_log))

# ── 7. Resumen ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESUMEN — Log-transform vs sin log")
print("=" * 70)
print(f"\n  Con log-transform (este script):")
print(f"  {'Modelo':<22} {'R²(orig)':>10} {'RMSE(orig)':>12} {'R²(log)':>10}")
print(f"  {'-'*58}")
for mid, r2, rmse, r2log in results:
    print(f"  {mid:<22} {r2:>10.4f} {rmse:>12.2f} {r2log:>10.4f}")

print(f"\n  Sin log-transform (referencia):")
print(f"  {'24_05_cat':<22} {'0.4067':>10} {'1014.86':>12}  (CatBoost, resultado principal)")
print(f"  {'abl_04':<22} {'0.3137':>10} {'1091.48':>12}  (XGBoost, Meta+RAWG)")
print(f"  {'abl_01':<22} {'0.1108':>10} {'1242.41':>12}  (XGBoost, Meta only)")
print()
