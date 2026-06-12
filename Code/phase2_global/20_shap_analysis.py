"""
20_shap_analysis.py
===================
SHAP analysis del modelo 19_10 (Hybrid+RAWG, cutoff 2017).
Objetivo: entender qué features impulsan el R²=0.438.
"""

import ast, json, os, sys, warnings
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

BASE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA   = os.path.join(BASE, "Data")
PLOTS  = os.path.join(BASE, "Plots")
os.makedirs(PLOTS, exist_ok=True)

CUTOFF = pd.Timestamp("2017-01-01")

print("=" * 65)
print("20_shap_analysis.py — SHAP sobre modelo 19_10 (cutoff 2017)")
print("=" * 65)

# ── 1. Cargar datos ────────────────────────────────────────────
print("\n[1] Cargando datos...")

with open(os.path.join(DATA, "item2idx_v2.json")) as f:
    item2idx_v2 = {k: int(v) for k, v in json.load(f).items()}
idx2appid = {v: k for k, v in item2idx_v2.items()}
N = len(item2idx_v2)

df_inter = pd.read_parquet(os.path.join(DATA, "interactions_v2.parquet"))
target_s = df_inter.groupby("item_idx").size().reindex(range(N), fill_value=0)
y_full   = target_s.values.astype(np.float64)

item_emb = np.load(os.path.join(DATA, "item_embeddings_rs_v2.npy"))

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

meta      = df_games.set_index("item_idx").reindex(range(N))
item_dates = meta["release_date_parsed"].values
train_mask = np.array([pd.notna(d) and d < CUTOFF  for d in item_dates])
test_mask  = np.array([pd.notna(d) and d >= CUTOFF for d in item_dates])
print(f"  Train: {train_mask.sum():,} | Test: {test_mask.sum():,}")

# ── 3. Features de contenido ───────────────────────────────────
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

# ── 4. RAWG ────────────────────────────────────────────────────
print("[4] Cargando RAWG...")

with open(os.path.join(DATA, "item2idx.json")) as f:
    item2idx_v1 = {k: int(v) for k, v in json.load(f).items()}
v1_idx_to_appid = {v: k for k, v in item2idx_v1.items()}

df_rawg = pd.read_csv(os.path.join(DATA, "rawg_enriched.csv")).drop_duplicates(subset=["item_idx"], keep="last")
df_rawg["appid_str"]    = df_rawg["item_idx"].map(v1_idx_to_appid)
df_rawg["item_idx_v2"]  = df_rawg["appid_str"].map(item2idx_v2)
df_rawg = df_rawg.dropna(subset=["item_idx_v2"])
df_rawg["item_idx_v2"]  = df_rawg["item_idx_v2"].astype(int)
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

print(f"  RAWG: {matched}/{N} items ({matched/N*100:.1f}%)")

# ── 5. Construir X10 con nombres de features ───────────────────
X10 = np.hstack([item_emb, tfidf_aligned, steam_aligned, rawg_aligned])

rs_names    = [f"rs_{i}"     for i in range(64)]
tfidf_names = [f"tfidf_{t}"  for t in tfidf.get_feature_names_out()]
steam_names = ["price", "early_access"]
rawg_names  = ["rawg_has_data", "rawg_rating", "rawg_rating_miss",
               "metacritic", "metacritic_miss", "log_playtime",
               "log_ratings_count", "n_platforms", "n_genres",
               "n_developers", "n_publishers", "esrb"]

feature_names = rs_names + tfidf_names + steam_names + rawg_names
assert len(feature_names) == X10.shape[1], f"Feature names mismatch: {len(feature_names)} vs {X10.shape[1]}"

# ── 6. Entrenar modelo 19_10 ────────────────────────────────────
print("\n[5] Entrenando modelo 19_10 (Optuna 40 trials para SHAP)...")

train_dates = np.array(item_dates)[train_mask]
sort_order  = np.argsort(train_dates)
X_tr = X10[train_mask][sort_order]
y_tr = y_full[train_mask][sort_order]
X_te = X10[test_mask]
y_te = y_full[test_mask]

split_idx = max(10, int(len(X_tr) * 0.8))

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
    m.fit(X_tr[:split_idx], y_tr[:split_idx])
    return float(mean_squared_error(y_tr[split_idx:], m.predict(X_tr[split_idx:])) ** 0.5)

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=40, show_progress_bar=False)

best_params = {**study.best_params, "random_state": 42, "tree_method": "hist", "verbosity": 0}
model = XGBRegressor(**best_params)
model.fit(X_tr, y_tr)

pred = np.clip(model.predict(X_te), 0, None)
r2   = r2_score(y_te, pred)
rmse = mean_squared_error(y_te, pred) ** 0.5
print(f"  R²={r2:.4f}  RMSE={rmse:.2f}")

# ── 7. SHAP ────────────────────────────────────────────────────
print("\n[6] Calculando SHAP values (XGBoost native pred_contribs)...")

# Usar SHAP nativo de XGBoost — sin problemas de compatibilidad de versiones
booster = model.get_booster()
dtest   = xgb.DMatrix(X_te, feature_names=[f"f{i}" for i in range(X_te.shape[1])])
# pred_contribs devuelve (n_samples, n_features + 1); la última col es el bias
shap_values = booster.predict(dtest, pred_contribs=True)[:, :-1]

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

print("\n" + "=" * 55)
print("Top 30 features por |SHAP| medio (test set)")
print("=" * 55)
for _, row in shap_df.head(30).iterrows():
    bar = "█" * int(row["mean_abs_shap"] / shap_df["mean_abs_shap"].max() * 30)
    print(f"  {row['feature']:<30} {row['mean_abs_shap']:>8.2f}  {bar}")

# Agrupar por tipo de feature
print("\n" + "=" * 55)
print("SHAP por grupo de features")
print("=" * 55)
groups = {
    "RS embeddings (64d)":        [f for f in feature_names if f.startswith("rs_")],
    "TF-IDF (100d)":              [f for f in feature_names if f.startswith("tfidf_")],
    "Steam metadata (2d)":        ["price", "early_access"],
    "RAWG features (12d)":        rawg_names,
}
for grp, cols in groups.items():
    idx_cols = [feature_names.index(c) for c in cols if c in feature_names]
    total = np.abs(shap_values[:, idx_cols]).mean()
    print(f"  {grp:<30} mean |SHAP| = {total:.2f}")

# Top features de RAWG específicamente
print("\n" + "=" * 55)
print("SHAP desglosado — RAWG features")
print("=" * 55)
for col in rawg_names:
    idx = feature_names.index(col)
    val = np.abs(shap_values[:, idx]).mean()
    print(f"  {col:<30} {val:.2f}")

# ── 8. Plots ────────────────────────────────────────────────────
print("\n[7] Guardando plots SHAP...")

# Bar plot (mean |SHAP|) top 20
top20_idx  = shap_df.head(20).index.tolist()
top20_names = shap_df.head(20)["feature"].tolist()
top20_vals  = shap_df.head(20)["mean_abs_shap"].tolist()

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top20_names[::-1], top20_vals[::-1], color="#4C8CBF")
ax.set_xlabel("mean |SHAP value|")
ax.set_title("SHAP Feature Importance — Modelo 19_10 (top 20)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "shap_bar_top20.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Guardado: Plots/shap_bar_top20.png")

# Beeswarm / summary plot
try:
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_te, feature_names=feature_names,
                      max_display=20, show=False)
    plt.title("SHAP Beeswarm — Modelo 19_10 (top 20)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "shap_beeswarm_top20.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: Plots/shap_beeswarm_top20.png")
except Exception as e:
    print(f"  Beeswarm skipped: {e}")

print("\nDone.")
