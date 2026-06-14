"""
v2_data.py — Módulo compartido para los scripts 32+ (pipeline v2, cutoff 2017).

Replica exactamente la construcción de datos/features de 24_ablation_catboost.py
para que los resultados sean comparables, y agrega:
  - build_dev_rep():   developer reputation TEMPORAL limpia (fix del bug de 18_)
  - build_metascore(): metascore nativo de Steam (sin usar hasta ahora)
  - run_model():       parametrizado (loss_function CatBoost, feature exclusion,
                       nombres de features, retorno de modelo/preds para SHAP)

Convención meta: META_COLS = [price_num, ea_flag, num_genres, num_tags, num_specs, has_senti]
  → has_senti (col 5) es LEAKAGE potencial (sentiment codifica conteo de reviews
    del snapshot 2018). Los scripts nuevos usan meta5 = meta_aligned[:, :5].
"""

import ast, json, os, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
BASE  = os.path.dirname(os.path.dirname(_HERE))
DATA  = os.path.join(BASE, "Data")
CUTOFF = pd.Timestamp("2017-01-01")

META_COLS = ["price_num", "ea_flag", "num_genres", "num_tags", "num_specs", "has_senti"]

TSCV_WINDOWS = [
    ("2013-07-01", "2014-01-01"), ("2014-01-01", "2014-07-01"),
    ("2014-07-01", "2015-01-01"), ("2015-01-01", "2015-07-01"),
    ("2015-07-01", "2016-01-01"), ("2016-01-01", "2016-07-01"),
    ("2016-07-01", "2017-01-01"),
]


def parse_list_col(x):
    if x is None or (isinstance(x, float) and pd.isna(x)): return []
    if isinstance(x, list): return [str(t).strip() for t in x if t]
    try:
        lst = eval(x)
        return [str(t).strip() for t in lst if t] if isinstance(lst, list) else []
    except Exception:
        return []


def clean_price(p):
    if pd.isna(p) or str(p).strip() in ("Free", "", "Free to Play"): return 0.0
    try: return float(str(p).replace("$", "").replace(",", "").strip())
    except Exception: return 0.0


def load_all(verbose=True):
    """Carga todo el contexto v2: target, fechas, masks, embeddings y features.

    Devuelve un dict con:
      N, y_full, item_dates, train_mask, test_mask,
      item_emb (content-aug v2 std), item_emb_cs (cold-start only),
      tfidf_aligned (100d), steam_aligned (2d), meta_aligned (6d),
      rawg_aligned (11d), df_games, item2idx_v2
    """
    if verbose: print("[v2_data] Cargando datos base...")
    with open(os.path.join(DATA, "item2idx_v2.json")) as f:
        item2idx_v2 = {k: int(v) for k, v in json.load(f).items()}
    N = len(item2idx_v2)

    df_inter = pd.read_parquet(os.path.join(DATA, "interactions_v2.parquet"))
    target_s = df_inter.groupby("item_idx").size().reindex(range(N), fill_value=0)
    y_full   = target_s.values.astype(np.float64)

    item_emb    = np.load(os.path.join(DATA, "item_embeddings_rs_v2_content2.npy"))
    cs_path     = os.path.join(DATA, "item_embeddings_rs_v2_content2_cs.npy")
    item_emb_cs = np.load(cs_path) if os.path.exists(cs_path) else None

    games_raw = []
    with open(os.path.join(DATA, "steam_games.json"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: games_raw.append(ast.literal_eval(line))
            except Exception: pass

    df_games = pd.json_normalize(games_raw).rename(columns={"id": "item_id"})
    df_games["item_id"]  = df_games["item_id"].astype(str)
    df_games["item_idx"] = df_games["item_id"].map(item2idx_v2)
    df_games = df_games.dropna(subset=["item_idx"])
    df_games["item_idx"] = df_games["item_idx"].astype(int)
    df_games["release_date_parsed"] = pd.to_datetime(df_games["release_date"], errors="coerce")
    df_games = df_games.drop_duplicates(subset=["item_idx"], keep="first")

    meta_idx   = df_games.set_index("item_idx").reindex(range(N))
    item_dates = meta_idx["release_date_parsed"].values
    train_mask = np.array([pd.notna(d) and d < CUTOFF  for d in item_dates])
    test_mask  = np.array([pd.notna(d) and d >= CUTOFF for d in item_dates])

    # ── Features de contenido (idéntico a 24_) ────────────────────────────
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
    tfidf = TfidfVectorizer(max_features=100, min_df=2, max_df=0.5, ngram_range=(1, 1))
    tfidf.fit(_pre)

    item_to_tfidf = {int(r["item_idx"]): i for i, (_, r) in enumerate(games_w_content.iterrows())}
    tfidf_mat     = tfidf.transform(games_w_content["content_txt"])

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

    # ── RAWG (idéntico a 24_) ─────────────────────────────────────────────
    rawg_aligned = None
    rawg_path  = os.path.join(DATA, "rawg_enriched.csv")
    v1map_path = os.path.join(DATA, "item2idx.json")
    if os.path.exists(rawg_path) and os.path.exists(v1map_path):
        with open(v1map_path) as f:
            item2idx_v1 = {k: int(v) for k, v in json.load(f).items()}
        v1_idx_to_appid = {v: k for k, v in item2idx_v1.items()}
        df_rawg = pd.read_csv(rawg_path).drop_duplicates(subset=["item_idx"], keep="last")
        df_rawg["appid_str"]   = df_rawg["item_idx"].map(v1_idx_to_appid)
        df_rawg["item_idx_v2"] = df_rawg["appid_str"].map(item2idx_v2)
        df_rawg = df_rawg.dropna(subset=["item_idx_v2"])
        df_rawg["item_idx_v2"] = df_rawg["item_idx_v2"].astype(int)
        df_rawg = df_rawg.set_index("item_idx_v2")

        ESRB_ORDER = {"Everyone": 1, "Everyone 10+": 2, "Teen": 3, "Mature": 4, "Adults Only": 5}
        rawg_aligned = np.zeros((N, 11), dtype=np.float32)
        for idx in range(N):
            if idx not in df_rawg.index: continue
            r = df_rawg.loc[idx]
            rawg_aligned[idx] = [
                1.0 if pd.notna(r.get("rawg_id")) else 0.0,
                float(r["rawg_rating"]) if pd.notna(r.get("rawg_rating")) else -1.0,
                0.0 if pd.notna(r.get("rawg_rating")) else 1.0,
                float(r["metacritic"])  if pd.notna(r.get("metacritic"))  else -1.0,
                0.0 if pd.notna(r.get("metacritic")) else 1.0,
                np.log1p(float(r["playtime_avg_h"])) if pd.notna(r.get("playtime_avg_h")) else 0.0,
                float(len(parse_list_col(r.get("platforms",  [])))),
                float(len(parse_list_col(r.get("genres",     [])))),
                float(len(parse_list_col(r.get("developers", [])))),
                float(len(parse_list_col(r.get("publishers", [])))),
                float(ESRB_ORDER.get(r.get("esrb_rating", ""), 0)),
            ]
        for ci in [1, 3]:
            train_vals = rawg_aligned[train_mask, ci]
            valid = train_vals[train_vals != -1.0]
            med   = float(np.median(valid)) if len(valid) > 0 else 0.0
            rawg_aligned[rawg_aligned[:, ci] == -1.0, ci] = med

    if verbose:
        print(f"[v2_data] Items: {N:,} | Train: {train_mask.sum():,} | Test: {test_mask.sum():,}")

    return dict(N=N, y_full=y_full, item_dates=item_dates,
                train_mask=train_mask, test_mask=test_mask,
                item_emb=item_emb, item_emb_cs=item_emb_cs,
                tfidf_aligned=tfidf_aligned, steam_aligned=steam_aligned,
                meta_aligned=meta_aligned, rawg_aligned=rawg_aligned,
                df_games=df_games, item2idx_v2=item2idx_v2)


def build_dev_rep(ctx):
    """Developer reputation temporal LIMPIA (2d): [log1p(mean reviews juegos
    previos del dev en train), has_dev_history].

    Fix sobre 18_: usa solo juegos de TRAIN (pre-2017) del developer y excluye
    el propio juego por item_idx (no por valor). Para items de test solo cuenta
    historia pre-cutoff → sin información futura.
    """
    df_games, N = ctx["df_games"], ctx["N"]
    y_full, train_mask = ctx["y_full"], ctx["train_mask"]

    idx_to_devs = {}
    for _, row in df_games.iterrows():
        devs = parse_list_col(row.get("developer"))
        if not devs and isinstance(row.get("developer"), str) and row["developer"].strip():
            devs = [row["developer"].strip()]
        if devs:
            idx_to_devs[int(row["item_idx"])] = devs

    dev_to_items = {}
    for idx in range(N):
        if not train_mask[idx]: continue
        for dev in idx_to_devs.get(idx, []):
            dev_to_items.setdefault(dev, []).append(idx)

    dev_rep = np.zeros((N, 2), dtype=np.float32)
    for idx in range(N):
        pool = []
        for dev in idx_to_devs.get(idx, []):
            pool.extend(j for j in dev_to_items.get(dev, []) if j != idx)
        if pool:
            dev_rep[idx, 0] = np.log1p(float(np.mean([y_full[j] for j in pool])))
            dev_rep[idx, 1] = 1.0
    return dev_rep


def build_metascore(ctx):
    """Metascore nativo de Steam (2d): [valor (mediana-imputado del train), missing_flag]."""
    df_games, N = ctx["df_games"], ctx["N"]
    train_mask = ctx["train_mask"]

    ms = np.full(N, np.nan, dtype=np.float32)
    if "metascore" in df_games.columns:
        for _, row in df_games.iterrows():
            v = row.get("metascore")
            try:
                fv = float(v)
                if np.isfinite(fv) and fv > 0:
                    ms[int(row["item_idx"])] = fv
            except (TypeError, ValueError):
                pass

    out = np.zeros((N, 2), dtype=np.float32)
    valid_train = ms[train_mask & np.isfinite(ms)]
    med = float(np.median(valid_train)) if len(valid_train) else 0.0
    out[:, 0] = np.where(np.isfinite(ms), ms, med)
    out[:, 1] = np.where(np.isfinite(ms), 0.0, 1.0)
    return out


# ── Tuning ─────────────────────────────────────────────────────────────────────

_EXP_LOSSES = ("Poisson", "Tweedie")  # CatBoost predice en log-space → Exponent


def _cat_predict(model, X, loss_function):
    if any(loss_function.startswith(l) for l in _EXP_LOSSES):
        try:
            return model.predict(X, prediction_type="Exponent")
        except Exception:
            return np.exp(model.predict(X))
    return model.predict(X)


def optuna_cat(X_tr, y_tr, n_trials=60, loss_function="RMSE", tune_tweedie_power=False,
               seed=42):
    split_idx = max(10, int(len(X_tr) * 0.8))
    Xt, Xv = X_tr[:split_idx], X_tr[split_idx:]
    yt, yv = y_tr[:split_idx], y_tr[split_idx:]

    def objective(trial):
        lf = loss_function
        if tune_tweedie_power:
            vp = trial.suggest_float("tweedie_vp", 1.05, 1.95)
            lf = f"Tweedie:variance_power={vp:.3f}"
        p = dict(
            iterations        = trial.suggest_int("iterations", 200, 1000),
            learning_rate     = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            depth             = trial.suggest_int("depth", 3, 8),
            l2_leaf_reg       = trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            loss_function=lf, random_seed=seed, verbose=0,
        )
        m = CatBoostRegressor(**p)
        m.fit(Xt, yt, eval_set=(Xv, yv), early_stopping_rounds=50, verbose=False)
        pred = np.clip(_cat_predict(m, Xv, lf), 0, None)
        return float(mean_squared_error(yv, pred) ** 0.5)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    lf = loss_function
    if tune_tweedie_power:
        lf = f"Tweedie:variance_power={best.pop('tweedie_vp'):.3f}"
    return {**best, "loss_function": lf, "random_seed": seed, "verbose": 0}


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


# ── Evaluación ─────────────────────────────────────────────────────────────────

def run_model(ctx, X, model_id, model_name, features_desc,
              loss_function="RMSE", tune_tweedie_power=False,
              n_trials=60, save=True, return_model=False, notes_extra="",
              tuning_seed=42):
    """Entrena CatBoost (Optuna) y evalúa: temporal + Spearman + TSCV + KFold.
    Devuelve dict de métricas (+ modelo y preds si return_model)."""
    from results_tracker import save_result

    y_full     = ctx["y_full"]
    item_dates = ctx["item_dates"]
    train_mask, test_mask = ctx["train_mask"], ctx["test_mask"]

    print(f"\n{'='*70}")
    print(f"[{model_id}] {model_name}")
    print(f"  shape={X.shape} | loss={loss_function}"
          f"{' (vp tuneado)' if tune_tweedie_power else ''} | trials={n_trials}")
    print(f"{'='*70}")

    train_dates = np.array(item_dates)[train_mask]
    sort_order  = np.argsort(train_dates)
    X_tr = X[train_mask][sort_order]
    y_tr = y_full[train_mask][sort_order]
    X_te = X[test_mask]
    y_te = y_full[test_mask]

    best_params = optuna_cat(X_tr, y_tr, n_trials, loss_function, tune_tweedie_power,
                             seed=tuning_seed)
    lf_final    = best_params["loss_function"]
    model = CatBoostRegressor(**best_params)
    model.fit(X_tr, y_tr, verbose=False)

    pred = np.clip(_cat_predict(model, X_te, lf_final), 0, None)

    r2   = r2_score(y_te, pred)
    rmse = float(mean_squared_error(y_te, pred) ** 0.5)
    mae  = float(mean_absolute_error(y_te, pred))
    nz   = y_te > 0
    mape = float(np.mean(np.abs((y_te[nz] - pred[nz]) / y_te[nz]))) if nz.sum() else np.nan
    rho  = float(spearmanr(y_te, pred).statistic)
    print(f"  Temporal: R2={r2:.4f} | RMSE={rmse:.2f} | MAE={mae:.2f} | Spearman={rho:.4f}")

    # TSCV
    ts = []
    for tc, sc in TSCV_WINDOWS:
        tc, sc = pd.Timestamp(tc), pd.Timestamp(sc)
        t_mask = np.array([pd.notna(d) and d < tc for d in item_dates])
        e_mask = np.array([pd.notna(d) and tc <= d < sc for d in item_dates])
        if e_mask.sum() < 5 or t_mask.sum() < 10: continue
        so2 = np.argsort(np.array(item_dates)[t_mask])
        m2 = CatBoostRegressor(**best_params)
        m2.fit(X[t_mask][so2], y_full[t_mask][so2], verbose=False)
        p2 = np.clip(_cat_predict(m2, X[e_mask], lf_final), 0, None)
        ts.append(r2_score(y_full[e_mask], p2))
    r2_tscv, r2_tscv_std = (float(np.mean(ts)), float(np.std(ts, ddof=1))) if ts else (np.nan, np.nan)
    print(f"  TSCV: R2={r2_tscv:.4f} ± {r2_tscv_std:.4f}")

    # KFold
    vd = np.array([pd.notna(d) for d in item_dates])
    X_kf, y_kf = X[vd], y_full[vd]
    kf_scores = []
    for tri, tei in KFold(n_splits=5, shuffle=True, random_state=42).split(X_kf):
        mk = CatBoostRegressor(**best_params)
        mk.fit(X_kf[tri], y_kf[tri], verbose=False)
        pk = np.clip(_cat_predict(mk, X_kf[tei], lf_final), 0, None)
        kf_scores.append(r2_score(y_kf[tei], pk))
    r2_kfold, r2_kfold_std = float(np.mean(kf_scores)), float(np.std(kf_scores, ddof=1))
    print(f"  KFold: R2={r2_kfold:.4f} ± {r2_kfold_std:.4f}")

    metrics = {
        "r2_temporal": float(r2), "rmse_temporal": rmse,
        "mae_temporal": mae, "mape_temporal": mape,
        "spearman_temporal": rho,
        "r2_tscv": r2_tscv, "r2_tscv_std": r2_tscv_std,
        "r2_kfold": r2_kfold, "r2_kfold_std": r2_kfold_std,
    }
    if save:
        save_result(model_id=model_id, model_name=model_name,
                    features=features_desc, embeddings="v2_global",
                    metrics=metrics,
                    notes=f"v2 cutoff2017 CatBoost({lf_final}) "
                          f"train={train_mask.sum()} test={test_mask.sum()}. {notes_extra}".strip())
    if return_model:
        return metrics, model, best_params, (X_tr, y_tr, X_te, y_te, pred)
    return metrics
