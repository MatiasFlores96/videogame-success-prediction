"""
25_two_tower_search.py
========================
Sweep de arquitecturas del Two-Tower content-augmented.

Pregunta científica central: ¿qué componente del content-aug Two-Tower
aporta la señal predictiva — el colaborativo (ID embedding) o el contenido?

Variaciones testeadas:
  CS_DROP=0.0   → ID embedding entrenado normalmente (máx señal colaborativa)
  CS_DROP=0.3   → configuración actual (baseline, ya tenemos resultado)
  CS_DROP=0.5   → dropout agresivo, fuerza más al content MLP
  CS_DROP=1.0   → ID embedding siempre cero → Two-Tower puramente de contenido
  EMB_DIM=128   → mayor capacidad con CS_DROP=0.3
  NO_RAWG       → content = TF-IDF(100) + 4 numeric (sin RAWG quality en el tower)

Para cada config:
  1. Entrenar Two-Tower (~15 min)
  2. Extraer embeddings
  3. Evaluar con CatBoost (60 trials, misma config que 24_05_cat)
  4. Guardar resultado

Al final: tabla comparativa + salva el mejor embedding.
"""

import ast, json, os, sys, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_tracker import save_result

BASE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA   = os.path.join(BASE, "Data")
CUTOFF_TT   = pd.Timestamp("2016-01-01")  # cutoff de entrenamiento del Two-Tower
CUTOFF_EVAL = pd.Timestamp("2017-01-01")  # cutoff de evaluación XGBoost

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print("25_two_tower_search.py — Sweep arquitecturas Two-Tower content-aug")
print("=" * 70)
print(f"  Dispositivo: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ── 1. Cargar datos estáticos ──────────────────────────────────────────────────
print("\n[1] Cargando datos base...")
with open(os.path.join(DATA, "item2idx_v2.json")) as f:
    item2idx_v2 = {k: int(v) for k, v in json.load(f).items()}
num_items = len(item2idx_v2)

with open(os.path.join(DATA, "user2idx_v2.json")) as f:
    user2idx_v2 = {k: int(v) for k, v in json.load(f).items()}
num_users = len(user2idx_v2)
print(f"  Usuarios: {num_users:,}  |  Items: {num_items:,}")

df_inter = pd.read_parquet(os.path.join(DATA, "interactions_v2.parquet"))
target_s = df_inter.groupby("item_idx").size().reindex(range(num_items), fill_value=0)
y_full   = target_s.values.astype(np.float64)

# ── 2. Interacciones para Two-Tower (pre-2016) ────────────────────────────────
print("\n[2] Cargando interacciones Two-Tower (pre-2016)...")

date_map = {}
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

for _, row in df_games.iterrows():
    gid = int(row["item_idx"])
    ds  = row.get("release_date", "")
    try: date_map[gid] = pd.to_datetime(ds, dayfirst=True, errors="coerce")
    except: date_map[gid] = pd.NaT

pre2016_items = {gid for gid, d in date_map.items() if pd.notna(d) and d < CUTOFF_TT}

inter_filt = df_inter[df_inter["item_idx"].isin(pre2016_items)].copy()
u_all = inter_filt["user_idx"].values.astype(np.int64)
i_all = inter_filt["item_idx"].values.astype(np.int64)
n = len(u_all)
val_cut = int(n * 0.9)
u_tr, u_va = u_all[:val_cut], u_all[val_cut:]
i_tr, i_va = i_all[:val_cut], i_all[val_cut:]
print(f"  Train: {len(u_tr):,}  |  Valid: {len(u_va):,}")

# ── 3. Content features ────────────────────────────────────────────────────────
print("\n[3] Construyendo content features...")

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

df_games["tag_text"]   = df_games["tags"].apply(parse_list_col).apply(" ".join)
df_games["genre_text"] = df_games["genres"].apply(parse_list_col).apply(" ".join)
df_games["content_txt"]= df_games["tag_text"] + " " + df_games["genre_text"]
df_games["price_num"]  = df_games["price"].apply(clean_price)
df_games["ea_flag"]    = df_games["early_access"].fillna(False).astype(int)
df_games["num_genres"] = df_games["genres"].apply(parse_list_col).apply(len)
df_games["num_tags"]   = df_games["tags"].apply(parse_list_col).apply(len)

games_w_content = df_games[df_games["content_txt"].str.strip() != ""].copy()
_pre = games_w_content[
    games_w_content["release_date_parsed"].notna() &
    (games_w_content["release_date_parsed"] < CUTOFF_TT)
]["content_txt"]
tfidf = TfidfVectorizer(max_features=100, min_df=2, max_df=0.5)
tfidf.fit(_pre)
item_to_tfidf = {int(r["item_idx"]): i for i, (_, r) in enumerate(games_w_content.iterrows())}
tfidf_mat     = tfidf.transform(games_w_content["content_txt"])

NUMERIC_COLS  = ["price_num", "ea_flag", "num_genres", "num_tags"]
meta_reind    = df_games.set_index("item_idx").reindex(range(num_items))

# RAWG quality features
rawg_quality = np.zeros((num_items, 4), dtype=np.float32)  # rating, metacritic, playtime, esrb
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
    ESRB_ORDER = {"Everyone":1,"Everyone 10+":2,"Teen":3,"Mature":4,"Adults Only":5}
    rq_matched = 0
    for idx in range(num_items):
        if idx not in df_rawg.index: continue
        r = df_rawg.loc[idx]
        rawg_quality[idx] = [
            float(r["rawg_rating"])  if pd.notna(r.get("rawg_rating"))  else 0.0,
            float(r["metacritic"])   if pd.notna(r.get("metacritic"))   else 0.0,
            np.log1p(float(r["playtime_avg_h"])) if pd.notna(r.get("playtime_avg_h")) else 0.0,
            float(ESRB_ORDER.get(r.get("esrb_rating",""), 0)),
        ]
        rq_matched += 1
    print(f"  RAWG quality features: {rq_matched:,} items")

def build_content_matrix(use_rawg_quality=True):
    """Construye la matriz de content features (num_items, content_dim)."""
    tfidf_a = np.zeros((num_items, 100), dtype=np.float32)
    num_a   = np.zeros((num_items, 4),   dtype=np.float32)
    for idx in range(num_items):
        if idx in item_to_tfidf:
            tfidf_a[idx] = tfidf_mat[item_to_tfidf[idx]].toarray().flatten()
        row = meta_reind.iloc[idx] if idx < len(meta_reind) else None
        if row is not None:
            for j, col in enumerate(NUMERIC_COLS):
                v = row.get(col, 0)
                num_a[idx, j] = float(v) if pd.notna(v) else 0.0

    # Normalizar numérico
    scaler = StandardScaler()
    num_a_scaled = scaler.fit_transform(num_a)

    if use_rawg_quality:
        rq_scaled = StandardScaler().fit_transform(rawg_quality)
        return np.hstack([tfidf_a, num_a_scaled, rq_scaled]).astype(np.float32)
    else:
        return np.hstack([tfidf_a, num_a_scaled]).astype(np.float32)

content_full    = build_content_matrix(use_rawg_quality=True)   # 108d
content_no_rawg = build_content_matrix(use_rawg_quality=False)  # 104d
print(f"  Content matrix full: {content_full.shape}  |  no-rawg: {content_no_rawg.shape}")

# ── 4. Metadata para evaluación downstream ─────────────────────────────────────
item_dates  = meta_reind["release_date_parsed"].values
train_mask  = np.array([pd.notna(d) and d < CUTOFF_EVAL  for d in item_dates])
test_mask   = np.array([pd.notna(d) and d >= CUTOFF_EVAL for d in item_dates])

META_COLS_EVAL = ["price_num", "ea_flag", "num_genres", "num_tags"]
meta_aligned_eval = np.zeros((num_items, 6), dtype=np.float32)
df_games2 = df_games.drop_duplicates(subset=["item_idx"], keep="first").set_index("item_idx").reindex(range(num_items))
for idx in range(num_items):
    row = df_games2.iloc[idx] if idx < len(df_games2) else None
    if row is not None:
        for j, col in enumerate(META_COLS_EVAL):
            v = row.get(col, 0)
            meta_aligned_eval[idx, j] = float(v) if pd.notna(v) else 0.0

TSCV_WINDOWS = [
    ("2013-07-01","2014-01-01"),("2014-01-01","2014-07-01"),
    ("2014-07-01","2015-01-01"),("2015-01-01","2015-07-01"),
    ("2015-07-01","2016-01-01"),("2016-01-01","2016-07-01"),
    ("2016-07-01","2017-01-01"),
]

# ── 5. Arquitectura Two-Tower ──────────────────────────────────────────────────
class ContentItemTower(nn.Module):
    def __init__(self, num_items, content_dim, emb_dim=64, cs_drop=0.3):
        super().__init__()
        self.cs_drop = cs_drop
        half = emb_dim // 2
        self.id_emb = nn.Embedding(num_items, half)
        nn.init.normal_(self.id_emb.weight, std=0.01)
        self.content_net = nn.Sequential(
            nn.Linear(content_dim, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, half),
        )
        self.proj = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.LayerNorm(emb_dim))

    def forward(self, item_ids, content_feats):
        id_vec      = self.id_emb(item_ids)
        content_vec = self.content_net(content_feats)
        if self.training and self.cs_drop > 0:
            mask   = (torch.rand(id_vec.shape[0], 1, device=id_vec.device) > self.cs_drop).float()
            id_vec = id_vec * mask
        if self.cs_drop >= 1.0:
            id_vec = torch.zeros_like(id_vec)
        combined = torch.cat([id_vec, content_vec], dim=1)
        return self.proj(combined)

    def get_cold_start_emb(self, item_ids, content_feats):
        content_vec = self.content_net(content_feats)
        zeros       = torch.zeros_like(content_vec)
        return self.proj(torch.cat([zeros, content_vec], dim=1))


class ContentTwoTower(nn.Module):
    def __init__(self, num_users, num_items, content_dim, emb_dim=64, cs_drop=0.3):
        super().__init__()
        self.user_emb   = nn.Embedding(num_users, emb_dim)
        self.item_tower = ContentItemTower(num_items, content_dim, emb_dim, cs_drop)
        nn.init.normal_(self.user_emb.weight, std=0.01)

    def forward(self, user_ids, item_ids, content_feats):
        u = self.user_emb(user_ids)
        v = self.item_tower(item_ids, content_feats)
        return (u * v).sum(dim=1)

    def get_item_emb(self, item_ids, content_feats):
        return self.item_tower(item_ids, content_feats)


class ContentDataset(Dataset):
    def __init__(self, u, i): self.u, self.i = torch.tensor(u), torch.tensor(i)
    def __len__(self): return len(self.u)
    def __getitem__(self, idx): return self.u[idx], self.i[idx]


# ── 6. Training function ───────────────────────────────────────────────────────
def train_two_tower(config_name, content_matrix, emb_dim=64, cs_drop=0.3,
                    lr=5e-4, epochs=15, patience=4, batch=8192):
    print(f"\n  → Entrenando Two-Tower: {config_name}")
    content_tensor = torch.tensor(content_matrix).to(DEVICE)
    content_dim    = content_matrix.shape[1]

    model = ContentTwoTower(num_users, num_items, content_dim, emb_dim, cs_drop).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    train_loader = DataLoader(ContentDataset(u_tr, i_tr), batch_size=batch,
                              shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(ContentDataset(u_va, i_va), batch_size=batch*2,
                              shuffle=False, num_workers=0, pin_memory=True)

    WARMUP = 2 * len(train_loader)
    TOTAL  = epochs * len(train_loader)
    def lr_lambda(step):
        if step < WARMUP: return step / max(1, WARMUP)
        progress = (step - WARMUP) / max(1, TOTAL - WARMUP)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val, patience_cnt, best_state, step = float("inf"), 0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total_loss = time.time(), 0.0
        for u_batch, i_batch in train_loader:
            u_batch = u_batch.to(DEVICE); i_batch = i_batch.to(DEVICE)
            neg     = torch.randint(0, num_items, (len(u_batch),), device=DEVICE)
            u_all2  = torch.cat([u_batch, u_batch])
            i_all2  = torch.cat([i_batch, neg])
            labels  = torch.cat([torch.ones(len(u_batch), device=DEVICE),
                                  torch.zeros(len(u_batch), device=DEVICE)])
            cont_b  = content_tensor[i_all2]
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type=="cuda")):
                logits = model(u_all2, i_all2, cont_b)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update(); scheduler.step()
            total_loss += loss.item(); step += 1

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for u_b, i_b in valid_loader:
                u_b = u_b.to(DEVICE); i_b = i_b.to(DEVICE)
                with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type=="cuda")):
                    logits = model(u_b, i_b, content_tensor[i_b])
                    val_loss += nn.functional.binary_cross_entropy_with_logits(
                        logits, torch.ones_like(logits)).item()
        avg_val = val_loss / len(valid_loader)
        avg_tr  = total_loss / len(train_loader)
        print(f"    Epoch {epoch:02d}/{epochs} | train={avg_tr:.4f} | val={avg_val:.4f} | {time.time()-t0:.1f}s")

        if avg_val < best_val:
            best_val, patience_cnt = avg_val, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"    Early stopping epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    print(f"  ✓ val_loss={best_val:.4f}")

    # Extraer embeddings
    all_ids  = torch.arange(num_items, device=DEVICE)
    bs_inf   = 2048
    embs     = []
    with torch.no_grad():
        for start in range(0, num_items, bs_inf):
            ids_b  = all_ids[start:start+bs_inf]
            cont_b = content_tensor[ids_b]
            embs.append(model.get_item_emb(ids_b, cont_b).cpu().numpy())
    embeddings = np.vstack(embs)
    return embeddings, best_val


# ── 7. CatBoost evaluation ─────────────────────────────────────────────────────
def evaluate_catboost(item_emb, config_name, n_trials=60):
    print(f"  → Evaluando CatBoost: {config_name}")
    X = np.hstack([item_emb, meta_aligned_eval])  # RS (emb_dim) + meta (6d)

    train_dates = np.array(item_dates)[train_mask]
    so = np.argsort(train_dates)
    X_tr = X[train_mask][so]; y_tr = y_full[train_mask][so]
    X_te = X[test_mask];      y_te = y_full[test_mask]

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
    best_p = {**study.best_params, "random_seed": 42, "verbose": 0}

    model = CatBoostRegressor(**best_p)
    model.fit(X_tr, y_tr, verbose=False)
    pred = np.clip(model.predict(X_te), 0, None)

    r2   = r2_score(y_te, pred)
    rmse = float(mean_squared_error(y_te, pred) ** 0.5)
    mae  = float(mean_absolute_error(y_te, pred))
    mask_nz = y_te > 0
    mape = float(np.mean(np.abs((y_te[mask_nz]-pred[mask_nz])/y_te[mask_nz]))) if mask_nz.sum()>0 else np.nan

    # TSCV
    ts_r2s = []
    for (tc, sc) in TSCV_WINDOWS:
        tc, sc = pd.Timestamp(tc), pd.Timestamp(sc)
        t_m = np.array([pd.notna(d) and d < tc for d in item_dates])
        e_m = np.array([pd.notna(d) and d >= tc and d < sc for d in item_dates])
        if e_m.sum() < 5 or t_m.sum() < 10: continue
        so2 = np.argsort(np.array(item_dates)[t_m])
        Xtr2 = X[t_m][so2]; ytr2 = y_full[t_m][so2]
        Xte2 = X[e_m];      yte2 = y_full[e_m]
        m2 = CatBoostRegressor(**best_p)
        m2.fit(Xtr2, ytr2, verbose=False)
        ts_r2s.append(r2_score(yte2, np.clip(m2.predict(Xte2), 0, None)))
    r2_tscv = float(np.mean(ts_r2s)) if ts_r2s else float("nan")

    # KFold
    vd_m = np.array([pd.notna(d) for d in item_dates])
    X_kf = X[vd_m]; y_kf = y_full[vd_m]
    kf_r2s = []
    for tri, tei in KFold(n_splits=5, shuffle=True, random_state=42).split(X_kf):
        mk = CatBoostRegressor(**best_p)
        mk.fit(X_kf[tri], y_kf[tri], verbose=False)
        kf_r2s.append(r2_score(y_kf[tei], np.clip(mk.predict(X_kf[tei]), 0, None)))
    r2_kfold = float(np.mean(kf_r2s))

    print(f"  ✓ R²={r2:.4f} | RMSE={rmse:.2f} | TSCV={r2_tscv:.4f} | KFold={r2_kfold:.4f}")

    metrics = dict(r2_temporal=r2, rmse_temporal=rmse, mae_temporal=mae,
                   mape_temporal=mape, r2_tscv=r2_tscv, r2_kfold=r2_kfold)
    save_result(model_id=f"25_{config_name}_cat",
                model_name=f"TT-search {config_name} CatBoost",
                features=f"RS search ({config_name}) + meta (6d)",
                embeddings="v2_global",
                metrics=metrics,
                notes=f"Two-Tower arch search: {config_name}")
    return r2, rmse, r2_tscv


# ── 8. Configuraciones a explorar ──────────────────────────────────────────────
CONFIGS = [
    # (name,             content_matrix,  emb_dim, cs_drop, lr)
    ("cs0_collab",       content_full,    64,      0.0,     5e-4),  # sin dropout → máx señal colaborativa
    # cs_drop=0.3 ya está en 23_ (baseline), la omitimos para no re-entrenar
    ("cs05_aggressive",  content_full,    64,      0.5,     5e-4),  # dropout más agresivo
    ("cs10_pureContent", content_full,    64,      1.0,     5e-4),  # sin ID embedding → contenido puro
    ("emb128",           content_full,    128,     0.3,     5e-4),  # embedding más grande
    ("noRawgInTower",    content_no_rawg, 64,      0.3,     5e-4),  # sin RAWG quality en el tower
]

print(f"\n[5] Corriendo {len(CONFIGS)} configuraciones (~{len(CONFIGS)*20} min estimados)...")
print("="*70)

all_results = []
best_r2, best_config, best_emb = -999, None, None

for config_name, content_matrix, emb_dim, cs_drop, lr in CONFIGS:
    print(f"\n{'='*70}")
    print(f"CONFIG: {config_name}  |  emb_dim={emb_dim}  cs_drop={cs_drop}  "
          f"content={content_matrix.shape[1]}d")
    print(f"{'='*70}")
    t_start = time.time()

    # Train Two-Tower
    embeddings, val_loss = train_two_tower(
        config_name, content_matrix, emb_dim=emb_dim, cs_drop=cs_drop, lr=lr
    )

    # Guardar embeddings
    emb_path = os.path.join(DATA, f"item_embeddings_rs_v2_search_{config_name}.npy")
    np.save(emb_path, embeddings)

    # Evaluate downstream
    r2, rmse, r2_tscv = evaluate_catboost(embeddings, config_name)

    elapsed = time.time() - t_start
    all_results.append({
        "config": config_name, "emb_dim": emb_dim, "cs_drop": cs_drop,
        "val_loss_tt": val_loss, "r2": r2, "rmse": rmse, "r2_tscv": r2_tscv,
        "elapsed_min": elapsed/60,
    })

    if r2 > best_r2:
        best_r2, best_config, best_emb = r2, config_name, embeddings

# ── 9. Resumen ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESUMEN — Sweep de arquitecturas Two-Tower")
print("=" * 70)
print(f"{'Config':<22} {'EMB':>5} {'CS':>5} {'TT_val':>8} {'R²':>8} {'RMSE':>10} {'TSCV':>8}")
print("-" * 70)
for r in sorted(all_results, key=lambda x: -x["r2"]):
    print(f"  {r['config']:<20} {r['emb_dim']:>5} {r['cs_drop']:>5.1f} "
          f"{r['val_loss_tt']:>8.4f} {r['r2']:>8.4f} {r['rmse']:>10.2f} {r['r2_tscv']:>8.4f}")

print(f"\nBaseline (23_ / cs_drop=0.3): R²=0.4067  RMSE=1014.86  (CatBoost, 24_05_cat)")
print(f"\nMejor config encontrada: {best_config}  R²={best_r2:.4f}")

# Guardar el mejor embedding también con nombre especial
if best_emb is not None:
    best_path = os.path.join(DATA, "item_embeddings_rs_v2_search_best.npy")
    np.save(best_path, best_emb)
    print(f"✓ Mejor embedding guardado: item_embeddings_rs_v2_search_best.npy")

print("\n✓ Sweep completo.")
