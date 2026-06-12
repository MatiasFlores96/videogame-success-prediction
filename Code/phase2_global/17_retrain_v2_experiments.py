"""
17_retrain_v2_experiments.py
=============================
Re-entrena todos los modelos que usan RS embeddings usando item_embeddings_v2_aligned.npy.
Los embeddings v2 reducen el cold-start del test set de ~100% a ~22%.

Modelos re-entrenados (nuevos IDs con sufijo 'v2'):
  03v2 — RS Only (v2)
  05v2 — RS + Metadata (v2)
  08v2 — Hybrid RS+TF-IDF+Numeric (v2)
  09v2 — RS + Review Text (v2)
  10v2 — RS + Reviews + RAWG (v2)
  11av2 — RS + Dev Reputation (v2)
  11bv2 — Hybrid + Dev Reputation (v2)
  13av2 — RS Only log-target (v2)
  13bv2 — Hybrid RS+TF-IDF log-target (v2)
"""

import ast, json, os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_tracker import save_result, print_leaderboard

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA        = os.path.join(BASE, "Data")
ITEM_EMB_V2 = os.path.join(DATA, "item_embeddings_v2_aligned.npy")   # ← nuevo
ITEM_EMB_V1 = os.path.join(DATA, "item_embeddings_rs_clean.npy")     # referencia
ITEMMAP     = os.path.join(DATA, "item2idx.json")
INTER       = os.path.join(DATA, "interactions.parquet")
STEAM       = os.path.join(DATA, "steam_games.json")
REVIEW_EMB  = os.path.join(DATA, "review_text_embeddings.npy")
RAWG_CSV    = os.path.join(DATA, "rawg_enriched.csv")
DEV_REP     = os.path.join(DATA, "developer_reputation.npy")
CUTOFF      = pd.Timestamp("2016-01-01")

print("=" * 70)
print("17_retrain_v2_experiments.py — RS v2 embeddings en todos los modelos")
print("=" * 70)

# ── 1. Cargar datos comunes ────────────────────────────────────────────────────
print("\nCargando datos...")
item_emb_v2 = np.load(ITEM_EMB_V2)    # (3682, 64)
N = item_emb_v2.shape[0]

with open(ITEMMAP) as f:
    item2idx = {k: int(v) for k, v in json.load(f).items()}

df_inter  = pd.read_parquet(INTER)
target_df = df_inter.groupby("item_idx").size().reset_index(name="total_reviews")
y_full    = target_df.set_index("item_idx").reindex(range(N), fill_value=0)["total_reviews"].values.astype(float)

games = []
with open(STEAM, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: games.append(ast.literal_eval(line))
        except: pass

df_games = pd.json_normalize(games).rename(columns={"id": "item_id"})
df_games["item_idx"]            = df_games["item_id"].map(item2idx)
df_games                        = df_games.dropna(subset=["item_idx"])
df_games["item_idx"]            = df_games["item_idx"].astype(int)
df_games["release_date_parsed"] = pd.to_datetime(df_games["release_date"], errors="coerce")

# Metadatos alineados por item_idx
meta = df_games.set_index("item_idx").reindex(range(N))
item_dates = meta["release_date_parsed"].values

train_mask = np.array([pd.notna(d) and d < CUTOFF  for d in item_dates])
test_mask  = np.array([pd.notna(d) and d >= CUTOFF for d in item_dates])
print(f"  Train: {train_mask.sum()} | Test: {test_mask.sum()} | EMB shape: {item_emb_v2.shape}")

# ── 2. Features adicionales ────────────────────────────────────────────────────
# --- Metadata (6d) ---
def clean_price(p):
    if pd.isna(p) or p == "Free" or p == "": return 0.0
    try: return float(str(p).replace("$", "").replace(",", "").strip())
    except: return 0.0

def safe_len(x): return len(x) if isinstance(x, list) else 0

df_games["price_num"]     = df_games["price"].apply(clean_price)
df_games["ea_flag"]       = df_games["early_access"].fillna(False).astype(int)
df_games["num_genres"]    = df_games["genres"].apply(safe_len)
df_games["num_tags"]      = df_games["tags"].apply(safe_len)
df_games["num_specs"]     = df_games["specs"].apply(safe_len) if "specs" in df_games.columns else 0
df_games["has_sentiment"] = df_games["sentiment"].notna().astype(int) if "sentiment" in df_games.columns else 0

META_COLS    = ["price_num", "ea_flag", "num_genres", "num_tags", "num_specs", "has_sentiment"]
meta_v2      = df_games.set_index("item_idx").reindex(range(N))
meta_features = meta_v2[META_COLS].fillna(0).values.astype(np.float32)  # (N, 6)

# --- TF-IDF (100d) ---
from sklearn.feature_extraction.text import TfidfVectorizer
def parse_list_col(x):
    if x is None or (isinstance(x, float) and pd.isna(x)): return []
    if isinstance(x, list): return [str(t).strip() for t in x if t]
    try:
        lst = eval(x); return [str(t).strip() for t in lst if t] if isinstance(lst, list) else []
    except: return []

df_games["tag_text"]    = df_games["tags"].apply(parse_list_col).apply(" ".join)
df_games["genre_text"]  = df_games["genres"].apply(parse_list_col).apply(" ".join)
df_games["content_txt"] = df_games["tag_text"] + " " + df_games["genre_text"]

games_w_content = df_games[df_games["content_txt"].str.strip() != ""].copy()
tfidf = TfidfVectorizer(max_features=100, min_df=2, max_df=0.5, ngram_range=(1,1))
_pre  = games_w_content[games_w_content["release_date_parsed"] < CUTOFF]["content_txt"]
tfidf.fit(_pre)
tfidf_mat      = tfidf.transform(games_w_content["content_txt"])
item_to_tfidf  = {int(r["item_idx"]): i for i, (_, r) in enumerate(games_w_content.iterrows())}

# Crear TF-IDF + steam_numeric alineados a (N, 102)
tfidf_aligned  = np.zeros((N, 100), dtype=np.float32)
steam_aligned  = np.zeros((N, 2),   dtype=np.float32)
for idx in range(N):
    if idx in item_to_tfidf:
        tfidf_aligned[idx] = tfidf_mat[item_to_tfidf[idx]].toarray().flatten()
        row = games_w_content[games_w_content["item_idx"] == idx]
        if not row.empty:
            steam_aligned[idx, 0] = float(row.iloc[0]["price_num"])
            steam_aligned[idx, 1] = float(row.iloc[0]["ea_flag"])

# --- Review embeddings (384d) ---
review_emb = None
if os.path.exists(REVIEW_EMB):
    review_emb = np.load(REVIEW_EMB)    # (N, 384)
    print(f"  Review embeddings: {review_emb.shape}")
else:
    print("  ADVERTENCIA: review_text_embeddings.npy no encontrado — se omiten modelos 09v2, 10v2")

# --- RAWG features (12d) ---
rawg_features = np.zeros((N, 12), dtype=np.float32)
if os.path.exists(RAWG_CSV):
    df_rawg = pd.read_csv(RAWG_CSV).drop_duplicates(subset=["item_idx"], keep="last").set_index("item_idx")
    ESRB_ORDER = {"Everyone":1,"Everyone 10+":2,"Teen":3,"Mature":4,"Adults Only":5}
    RAWG_OFFSET_FULL = 64 + 384
    IMPUTE_COLS = [1, 3]  # rawg_rating, metacritic

    for idx in range(N):
        if idx not in df_rawg.index: continue
        r = df_rawg.loc[idx]
        rawg_features[idx] = [
            1.0 if pd.notna(r.get("rawg_id")) else 0.0,
            float(r["rawg_rating"])       if pd.notna(r.get("rawg_rating"))      else -1.0,
            0.0 if pd.notna(r.get("rawg_rating"))   else 1.0,
            float(r["metacritic"])         if pd.notna(r.get("metacritic"))       else -1.0,
            0.0 if pd.notna(r.get("metacritic"))    else 1.0,
            np.log1p(float(r["playtime_avg_h"]))     if pd.notna(r.get("playtime_avg_h"))     else 0.0,
            np.log1p(float(r["rawg_ratings_count"])) if pd.notna(r.get("rawg_ratings_count")) else 0.0,
            float(len(parse_list_col(r.get("platforms",[])))),
            float(len(parse_list_col(r.get("genres",   [])))),
            float(len(parse_list_col(r.get("developers",[])))),
            float(len(parse_list_col(r.get("publishers",[])))),
            float(ESRB_ORDER.get(r.get("esrb_rating",""), 0)),
        ]
    print(f"  RAWG features cargadas: {(rawg_features[:, 0] == 1).sum()} matches")

# --- Developer reputation (1d) ---
dev_rep = np.zeros((N, 1), dtype=np.float32)
if os.path.exists(DEV_REP):
    dr = np.load(DEV_REP)
    dev_rep[:len(dr), 0] = dr[:N]
    print(f"  Developer reputation: cargado")

# ── 3. Helpers de entrenamiento ────────────────────────────────────────────────
TSCV_WINDOWS = [
    ("2013-07-01", "2014-01-01"),
    ("2014-01-01", "2014-07-01"),
    ("2014-07-01", "2015-01-01"),
    ("2015-01-01", "2015-07-01"),
    ("2015-07-01", "2016-01-01"),
]

def optuna_search(X_tr, y_tr, n_trials=60):
    """Busca hiperparámetros con Optuna (20% val temporal más reciente)."""
    sort_order = np.argsort(np.arange(len(X_tr)))  # ya viene ordenado temporalmente
    split_idx  = max(10, int(len(X_tr) * 0.8))
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

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = {**study.best_params, "random_state": 42, "tree_method": "hist", "verbosity": 0}
    return best, study.best_value


def run_model(X, y, model_id, model_name, features_desc, embeddings_tag,
              log_target=False, n_trials=60, notes=""):
    """Entrena + evalúa + guarda un modelo completo."""
    print(f"\n{'='*70}")
    print(f"[{model_id}] {model_name}")
    print(f"  Features: {features_desc}  |  Shape: {X.shape}  |  log_target={log_target}")
    print(f"{'='*70}")

    y_use = np.log1p(y) if log_target else y

    # --- Sort train by date ---
    train_dates = np.array(item_dates)[train_mask]
    sort_order  = np.argsort(train_dates)
    X_tr = X[train_mask][sort_order]
    y_tr = y_use[train_mask][sort_order]
    X_te = X[test_mask]
    y_te = y[test_mask]  # siempre evaluar en escala original

    # Optuna
    best_params, val_rmse = optuna_search(X_tr, y_tr, n_trials=n_trials)
    print(f"  Optuna val RMSE (internos)={val_rmse:.4f}")

    # Entrenar
    model = XGBRegressor(**best_params)
    model.fit(X_tr, y_tr)

    # Predicciones en escala original
    pred_log = model.predict(X_te)
    pred     = np.expm1(pred_log) if log_target else pred_log
    pred     = np.clip(pred, 0, None)

    r2_orig   = r2_score(y_te, pred)
    rmse_orig = float(mean_squared_error(y_te, pred) ** 0.5)
    mae_orig  = float(mean_absolute_error(y_te, pred))
    mask_nz   = y_te > 0
    mape_orig = float(np.mean(np.abs((y_te[mask_nz] - pred[mask_nz]) / y_te[mask_nz]))) if mask_nz.sum() > 0 else np.nan

    print(f"  Temporal: R2={r2_orig:.4f} | RMSE={rmse_orig:.2f} | MAE={mae_orig:.2f}")

    # --- R² en log-space si log_target ---
    r2_log = None
    if log_target:
        r2_log = r2_score(np.log1p(y_te), pred_log)
        print(f"  Log-space: R2_log={r2_log:.4f}")

    # --- TSCV ---
    ts_results = []
    for (tc, sc) in TSCV_WINDOWS:
        tc, sc = pd.Timestamp(tc), pd.Timestamp(sc)
        t_mask = np.array([pd.notna(d) and d < tc for d in item_dates])
        e_mask = np.array([pd.notna(d) and d >= tc and d < sc for d in item_dates])
        if e_mask.sum() < 5 or t_mask.sum() < 10: continue
        # sort train by date
        td = np.array(item_dates)[t_mask]
        so = np.argsort(td)
        Xtr2 = X[t_mask][so];  ytr2 = y_use[t_mask][so]
        Xte2 = X[e_mask];       yte2 = y[e_mask]
        m2 = XGBRegressor(**best_params)
        m2.fit(Xtr2, ytr2)
        p2 = np.expm1(m2.predict(Xte2)) if log_target else m2.predict(Xte2)
        p2 = np.clip(p2, 0, None)
        ts_results.append({"R2": r2_score(yte2, p2), "RMSE": float(mean_squared_error(yte2, p2)**0.5)})

    if ts_results:
        ts_df = pd.DataFrame(ts_results)
        r2_tscv  = ts_df["R2"].mean();   r2_tscv_std  = ts_df["R2"].std()
        rmse_tscv = ts_df["RMSE"].mean(); rmse_tscv_std = ts_df["RMSE"].std()
        print(f"  TSCV:    R2={r2_tscv:.4f} ± {r2_tscv_std:.4f}")
    else:
        r2_tscv = r2_tscv_std = rmse_tscv = rmse_tscv_std = float("nan")

    # --- KFold ---
    vd_mask = np.array([pd.notna(d) for d in item_dates])
    X_kf = X[vd_mask];  y_kf = y_use[vd_mask]; y_kf_orig = y[vd_mask]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kf_results = []
    for tri, tei in kf.split(X_kf):
        mk = XGBRegressor(**best_params)
        mk.fit(X_kf[tri], y_kf[tri])
        pk = np.expm1(mk.predict(X_kf[tei])) if log_target else mk.predict(X_kf[tei])
        pk = np.clip(pk, 0, None)
        kf_results.append({"R2": r2_score(y_kf_orig[tei], pk),
                            "RMSE": float(mean_squared_error(y_kf_orig[tei], pk)**0.5)})
    kf_df = pd.DataFrame(kf_results)
    r2_kfold  = kf_df["R2"].mean();  r2_kfold_std  = kf_df["R2"].std()
    rmse_kfold = kf_df["RMSE"].mean()
    print(f"  KFold:   R2={r2_kfold:.4f} ± {r2_kfold_std:.4f}")

    # --- Guardar ---
    metrics = {
        "r2_temporal":    r2_orig,
        "rmse_temporal":  rmse_orig,
        "mae_temporal":   mae_orig,
        "mape_temporal":  mape_orig,
        "r2_tscv":        r2_tscv,
        "r2_tscv_std":    r2_tscv_std,
        "rmse_tscv":      rmse_tscv,
        "rmse_tscv_std":  rmse_tscv_std,
        "r2_kfold":       r2_kfold,
        "r2_kfold_std":   r2_kfold_std,
        "rmse_kfold":     rmse_kfold,
    }
    if log_target and r2_log is not None:
        metrics["r2_temporal_log"] = r2_log

    save_result(model_id=model_id, model_name=model_name,
                features=features_desc, embeddings=embeddings_tag,
                metrics=metrics, notes=notes)
    return r2_orig, rmse_orig


# ── 4. Imputar RAWG para modelo 10v2 ──────────────────────────────────────────
def impute_rawg_inplace(X, tr_mask, rawg_offset, cols=(1,3)):
    """Imputa -1 con mediana del train set."""
    Xc = X.copy()
    for ci in cols:
        col = rawg_offset + ci
        train_vals = Xc[tr_mask, col]
        valid = train_vals[train_vals != -1.0]
        med   = float(np.median(valid)) if len(valid) > 0 else 0.0
        Xc[Xc[:, col] == -1.0, col] = med
    return Xc


# ── 5. Ejecutar experimentos ───────────────────────────────────────────────────
results = []

# --- 03v2: RS Only ---
X03 = item_emb_v2
r2, rmse = run_model(X03, y_full, "03v2", "RS Only (v2)", "RS v2 (64d)", "v2")
results.append(("03v2", r2, rmse))

# --- 05v2: RS + Metadata ---
X05 = np.hstack([item_emb_v2, meta_features])
r2, rmse = run_model(X05, y_full, "05v2", "RS + Metadata (v2)", "RS v2 (64d) + Metadata (6d)", "v2")
results.append(("05v2", r2, rmse))

# --- 08v2: Hybrid RS+TF-IDF+Numeric ---
X08 = np.hstack([item_emb_v2, tfidf_aligned, steam_aligned])
r2, rmse = run_model(X08, y_full, "08v2", "Hybrid Collab-Content (v2)",
                     "RS v2 (64d) + TF-IDF (100d) + Numeric (2d)", "v2")
results.append(("08v2", r2, rmse))

# --- 09v2: RS + Review Text ---
if review_emb is not None:
    X09 = np.hstack([item_emb_v2, review_emb])
    r2, rmse = run_model(X09, y_full, "09v2", "RS + Review Text (v2)",
                         "RS v2 (64d) + Review text (384d)", "v2")
    results.append(("09v2", r2, rmse))

# --- 10v2: RS + Reviews + RAWG ---
if review_emb is not None:
    X10_raw = np.hstack([item_emb_v2, review_emb, rawg_features]).astype(np.float32)
    X10     = impute_rawg_inplace(X10_raw, train_mask, rawg_offset=64+384, cols=(1,3))
    r2, rmse = run_model(X10, y_full, "10v2", "RS + Reviews + RAWG (v2)",
                         "RS v2 (64d) + Review text (384d) + RAWG (12d)", "v2")
    results.append(("10v2", r2, rmse))

# --- 11av2: RS + Dev Reputation ---
X11a = np.hstack([item_emb_v2, dev_rep])
r2, rmse = run_model(X11a, y_full, "11av2", "RS + Dev Reputation (v2)",
                     "RS v2 (64d) + dev_rep (1d)", "v2")
results.append(("11av2", r2, rmse))

# --- 11bv2: Hybrid + Dev Reputation ---
X11b = np.hstack([item_emb_v2, tfidf_aligned, steam_aligned, dev_rep])
r2, rmse = run_model(X11b, y_full, "11bv2", "Hybrid + Dev Reputation (v2)",
                     "RS v2 (64d) + TF-IDF (100d) + Numeric (2d) + dev_rep (1d)", "v2")
results.append(("11bv2", r2, rmse))

# --- 13av2: RS Only log-target ---
r2, rmse = run_model(X03, y_full, "13av2", "RS Only log-target (v2)",
                     "RS v2 (64d) [log-target]", "v2", log_target=True)
results.append(("13av2", r2, rmse))

# --- 13bv2: Hybrid RS+TF-IDF log-target ---
r2, rmse = run_model(X08, y_full, "13bv2", "Hybrid RS+TF-IDF log-target (v2)",
                     "RS v2 (64d) + TF-IDF (100d) + Numeric (2d) [log-target]", "v2", log_target=True)
results.append(("13bv2", r2, rmse))

# ── 6. Leaderboard final ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESUMEN — Mejora de R² temporal con embeddings v2")
print("=" * 70)
comparacion = {
    "03": -0.0184, "05": -0.0161, "08": -0.0240,
    "09": -0.0213, "10": -0.0163, "11a": -0.0253, "11b": -0.0238,
    "13a": -0.0263, "13b": -0.0248,
}
for mid, r2, rmse in results:
    base_id = mid.replace("v2","").replace("a","a").replace("b","b")
    old = comparacion.get(base_id, None)
    delta = f"  Δ={r2 - old:+.4f}" if old is not None else ""
    print(f"  [{mid}]  R2={r2:.4f}  RMSE={rmse:.2f}{delta}")

print("\n\nLeaderboard completo:")
print_leaderboard()
