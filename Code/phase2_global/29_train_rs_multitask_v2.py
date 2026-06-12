"""
29_train_rs_multitask_v2.py
============================
Fix completo sobre 27_/28_. Dos bugs corregidos:

BUG 1 (28_ ya lo fijó): double forward pass.
  - model() se llamaba DOS veces por batch -> LR efectivo 2x para user embeddings.
  - FIX: una sola llamada model() y extraer item_emb_pos del primer bloque.

BUG 2 (root cause del val_loss que sube desde epoch 2):
  - Split cronológico (val_cut = 90%) -> usuarios del último 10% de interacciones
    pueden no estar en el 90% de entrenamiento -> embeddings random -> val_loss dispara.
  - FIX: train_test_split aleatorio (random_state=42), igual que el 23_ baseline
    que convergió correctamente (val_loss=0.3261, epoch 8).

+ gradient clipping (norm=1.0) de 28_

Diseño:
  - Cutoff pre-2017 (fix cutoff del 23_ baseline)
  - Multi-task: BCE + lambda_pop=0.2 * MSE(log review_count)
  - User tower: ID-only (sin cambios vs baseline)
"""

import ast, json, os, sys, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_tracker import save_result

BASE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA   = os.path.join(BASE, "Data")

# ── Configuración ──────────────────────────────────────────────────────────────
CUTOFF_TT   = pd.Timestamp("2017-01-01")  # FIX: era 2016
CUTOFF_EVAL = pd.Timestamp("2017-01-01")

EMB_DIM    = 64
CS_DROP    = 0.3
LR         = 5e-4
EPOCHS     = 15
PATIENCE   = 4
BATCH      = 8192
LAMBDA_POP = 0.2   # peso multi-task loss (más conservador que 0.3)
WARMUP_EP  = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("29_train_rs_multitask_v2.py — BCE + Multi-task + cutoff 2017 (fix split + fix double-fwd)")
print("=" * 70)
print(f"  Dispositivo: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ── 1. Mappings ────────────────────────────────────────────────────────────────
print("\n[1] Cargando mappings...")
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

# Target normalizado para multi-task (z-score en log-space)
y_log_raw  = np.log1p(y_full).astype(np.float32)
y_log_mean = float(y_log_raw.mean())
y_log_std  = float(y_log_raw.std()) + 1e-8
y_log_norm = (y_log_raw - y_log_mean) / y_log_std

# ── 2. Metadata ────────────────────────────────────────────────────────────────
print("\n[2] Cargando metadata...")
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
df_games = df_games.dropna(subset=["item_idx"])
df_games["item_idx"]  = df_games["item_idx"].astype(int)
df_games["release_date_parsed"] = pd.to_datetime(df_games["release_date"], errors="coerce")
df_games = df_games.drop_duplicates(subset=["item_idx"], keep="first")

date_map = {}
for _, row in df_games.iterrows():
    gid = int(row["item_idx"])
    try: date_map[gid] = pd.to_datetime(row.get("release_date", ""), dayfirst=True, errors="coerce")
    except: date_map[gid] = pd.NaT

pre_cutoff_items = {gid for gid, d in date_map.items() if pd.notna(d) and d < CUTOFF_TT}
print(f"  Items pre-2017: {len(pre_cutoff_items):,}  (baseline usaba ~6,953 pre-2016)")

# ── 3. Content features (108d) ─────────────────────────────────────────────────
print("\n[3] Construyendo content features (108d)...")

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

NUMERIC_COLS = ["price_num", "ea_flag", "num_genres", "num_tags"]
meta_reind   = df_games.set_index("item_idx").reindex(range(num_items))

rawg_quality = np.zeros((num_items, 4), dtype=np.float32)
rawg_path  = os.path.join(DATA, "rawg_enriched.csv")
v1map_path = os.path.join(DATA, "item2idx.json")
if os.path.exists(rawg_path) and os.path.exists(v1map_path):
    with open(v1map_path) as f:
        item2idx_v1 = {k: int(v) for k, v in json.load(f).items()}
    v1_to_appid = {v: k for k, v in item2idx_v1.items()}
    df_rawg = pd.read_csv(rawg_path).drop_duplicates(subset=["item_idx"], keep="last")
    df_rawg["appid_str"]   = df_rawg["item_idx"].map(v1_to_appid)
    df_rawg["item_idx_v2"] = df_rawg["appid_str"].map(item2idx_v2)
    df_rawg = df_rawg.dropna(subset=["item_idx_v2"])
    df_rawg["item_idx_v2"] = df_rawg["item_idx_v2"].astype(int)
    df_rawg = df_rawg.set_index("item_idx_v2")
    ESRB = {"Everyone":1,"Everyone 10+":2,"Teen":3,"Mature":4,"Adults Only":5}
    for idx in range(num_items):
        if idx not in df_rawg.index: continue
        r = df_rawg.loc[idx]
        rawg_quality[idx] = [
            float(r["rawg_rating"])  if pd.notna(r.get("rawg_rating"))  else 0.0,
            float(r["metacritic"])   if pd.notna(r.get("metacritic"))   else 0.0,
            np.log1p(float(r["playtime_avg_h"])) if pd.notna(r.get("playtime_avg_h")) else 0.0,
            float(ESRB.get(r.get("esrb_rating",""), 0)),
        ]

tfidf_aligned = np.zeros((num_items, 100), dtype=np.float32)
num_aligned   = np.zeros((num_items, 4),   dtype=np.float32)
for idx in range(num_items):
    if idx in item_to_tfidf:
        tfidf_aligned[idx] = tfidf_mat[item_to_tfidf[idx]].toarray().flatten()
    row = meta_reind.iloc[idx] if idx < len(meta_reind) else None
    if row is not None:
        for j, col in enumerate(NUMERIC_COLS):
            v = row.get(col, 0)
            num_aligned[idx, j] = float(v) if pd.notna(v) else 0.0

train_items_list = list(pre_cutoff_items)
scaler_num  = StandardScaler().fit(num_aligned[train_items_list])
scaler_rawg = StandardScaler().fit(rawg_quality[train_items_list])
num_scaled  = scaler_num.transform(num_aligned).astype(np.float32)
rawg_scaled = scaler_rawg.transform(rawg_quality).astype(np.float32)

content_matrix = np.hstack([tfidf_aligned, num_scaled, rawg_scaled])  # (N, 108)
CONTENT_DIM    = content_matrix.shape[1]
print(f"  Content matrix: {content_matrix.shape}")

# ── 4. Interacciones ───────────────────────────────────────────────────────────
print("\n[4] Cargando interacciones (pre-2017)...")
inter_filt = df_inter[df_inter["item_idx"].isin(pre_cutoff_items)].copy()
u_all = inter_filt["user_idx"].values.astype(np.int64)
i_all = inter_filt["item_idx"].values.astype(np.int64)
n     = len(u_all)
# FIX BUG 2: split aleatorio (no cronológico) — igual que 23_ baseline.
# Split cronológico ponía usuarios del último 10% que nunca aparecían en el 90%
# de train → embeddings random → val_loss subía desde epoch 2.
idx_all = np.arange(n)
idx_tr, idx_va = train_test_split(idx_all, test_size=0.10, random_state=42)
u_tr, u_va = u_all[idx_tr], u_all[idx_va]
i_tr, i_va = i_all[idx_tr], i_all[idx_va]
print(f"  {n:,} interacciones  |  Train: {len(u_tr):,}  |  Valid: {len(u_va):,}")
print(f"  Steps/epoch: {(len(u_tr) + BATCH - 1) // BATCH:,}")

# ── 5. Arquitectura (BCE + multi-task, sin cambios en user tower) ──────────────
print("\n[5] Construyendo modelo...")

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
        # Multi-task head: popularidad
        self.pop_head = nn.Sequential(
            nn.Linear(emb_dim, 32), nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, item_ids, content_feats):
        id_vec      = self.id_emb(item_ids)
        content_vec = self.content_net(content_feats)
        if self.training and self.cs_drop > 0:
            mask   = (torch.rand(id_vec.shape[0], 1, device=id_vec.device) > self.cs_drop).float()
            id_vec = id_vec * mask
        return self.proj(torch.cat([id_vec, content_vec], dim=1))

    def get_cold_start_emb(self, item_ids, content_feats):
        content_vec = self.content_net(content_feats)
        zeros       = torch.zeros_like(content_vec)
        return self.proj(torch.cat([zeros, content_vec], dim=1))

    def predict_pop(self, emb):
        return self.pop_head(emb).squeeze(-1)


class MultiTaskTwoTower(nn.Module):
    def __init__(self, num_users, num_items, content_dim, emb_dim=64, cs_drop=0.3):
        super().__init__()
        # User tower: ID-only (igual que baseline — no cambiamos lo que funciona)
        self.user_emb  = nn.Embedding(num_users, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        self.item_tower = ContentItemTower(num_items, content_dim, emb_dim, cs_drop)

    def forward(self, user_ids, item_ids, content_feats):
        u = self.user_emb(user_ids)
        v = self.item_tower(item_ids, content_feats)
        return (u * v).sum(dim=1), v   # dot product + item embedding

    def get_item_emb(self, item_ids, content_feats):
        return self.item_tower(item_ids, content_feats)

    def get_cold_start_emb(self, item_ids, content_feats):
        return self.item_tower.get_cold_start_emb(item_ids, content_feats)


model = MultiTaskTwoTower(num_users, num_items, CONTENT_DIM, EMB_DIM, CS_DROP).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parámetros: {n_params:,}  ({n_params*4/1e6:.0f} MB FP32)")

# ── 6. DataLoader ──────────────────────────────────────────────────────────────
class ContentDataset(Dataset):
    def __init__(self, u, i):
        self.u = torch.tensor(u, dtype=torch.long)
        self.i = torch.tensor(i, dtype=torch.long)
    def __len__(self): return len(self.u)
    def __getitem__(self, idx): return self.u[idx], self.i[idx]

train_loader = DataLoader(ContentDataset(u_tr, i_tr), batch_size=BATCH,
                          shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(ContentDataset(u_va, i_va), batch_size=BATCH*2,
                          shuffle=False, num_workers=0, pin_memory=True)

content_tensor = torch.tensor(content_matrix, dtype=torch.float32).to(DEVICE)
y_log_tensor   = torch.tensor(y_log_norm, dtype=torch.float32).to(DEVICE)

# ── 7. Optimizador ─────────────────────────────────────────────────────────────
optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
criterion  = nn.BCEWithLogitsLoss()
scaler_amp = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

WARMUP_STEPS = WARMUP_EP * len(train_loader)
TOTAL_STEPS  = EPOCHS * len(train_loader)

def lr_lambda(step):
    if step < WARMUP_STEPS: return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ── 8. Training ────────────────────────────────────────────────────────────────
print(f"\n[6] Entrenando ({EPOCHS} ep, patience={PATIENCE}, BCE + λ_pop={LAMBDA_POP})...")

best_val, patience_cnt, best_state, step = float("inf"), 0, None, 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    total_bce, total_pop, n_batches = 0.0, 0.0, 0

    for u_batch, i_batch in train_loader:
        u_batch = u_batch.to(DEVICE)
        i_batch = i_batch.to(DEVICE)

        neg_items  = torch.randint(0, num_items, (len(u_batch),), device=DEVICE)
        users_all  = torch.cat([u_batch, u_batch])
        items_all  = torch.cat([i_batch, neg_items])
        labels     = torch.cat([torch.ones(len(u_batch), device=DEVICE),
                                 torch.zeros(len(u_batch), device=DEVICE)])
        cont_all   = content_tensor[items_all]
        pop_target = y_log_tensor[i_batch]   # solo para positivos

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type=="cuda")):
            # FIX: una sola llamada al modelo para pos+neg
            logits_all, item_embs_all = model(users_all, items_all, cont_all)
            loss_bce = criterion(logits_all, labels)

            # Multi-task: embeddings de positivos son los primeros len(u_batch)
            item_emb_pos = item_embs_all[:len(u_batch)]
            pop_pred = model.item_tower.predict_pop(item_emb_pos)
            loss_pop = F.mse_loss(pop_pred, pop_target)

            loss = loss_bce + LAMBDA_POP * loss_pop

        scaler_amp.scale(loss).backward()
        # Gradient clipping para estabilidad
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        scheduler.step()

        total_bce += loss_bce.item()
        total_pop += loss_pop.item()
        n_batches  += 1
        step += 1

    # Validación (solo BCE)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for u_b, i_b in valid_loader:
            u_b = u_b.to(DEVICE); i_b = i_b.to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type=="cuda")):
                logits = model.user_emb(u_b) * model.item_tower(i_b, content_tensor[i_b])
                logits = logits.sum(dim=1)
                val_loss += F.binary_cross_entropy_with_logits(
                    logits, torch.ones_like(logits)).item()
    avg_val = val_loss / len(valid_loader)

    print(f"  Epoch {epoch:02d}/{EPOCHS} | bce={total_bce/n_batches:.4f} | "
          f"pop={total_pop/n_batches:.4f} | val={avg_val:.4f} | {time.time()-t0:.1f}s")

    if avg_val < best_val:
        best_val, patience_cnt = avg_val, 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"  Early stopping en epoch {epoch}")
            break

model.load_state_dict(best_state)
model.eval()
print(f"\n✓ Mejor val_loss: {best_val:.4f}")

# ── 9. Extraer embeddings ──────────────────────────────────────────────────────
print("\n[7] Extrayendo embeddings...")
all_ids = torch.arange(num_items, device=DEVICE)
BS_INF  = 2048
embs_std, embs_cs = [], []
with torch.no_grad():
    for start in range(0, num_items, BS_INF):
        ids_b  = all_ids[start:start+BS_INF]
        cont_b = content_tensor[ids_b]
        embs_std.append(model.get_item_emb(ids_b, cont_b).cpu().numpy())
        embs_cs.append(model.get_cold_start_emb(ids_b, cont_b).cpu().numpy())

item_emb_std = np.vstack(embs_std)
item_emb_cs  = np.vstack(embs_cs)
np.save(os.path.join(DATA, "item_embeddings_rs_v2_multitask_v2.npy"),    item_emb_std)
np.save(os.path.join(DATA, "item_embeddings_rs_v2_multitask_v2_cs.npy"), item_emb_cs)
print(f"  ✓ {item_emb_std.shape}  guardados")

# ── 10. Evaluación CatBoost ────────────────────────────────────────────────────
print("\n[8] Evaluación CatBoost...")

item_dates = meta_reind["release_date_parsed"].values
train_mask = np.array([pd.notna(d) and d < CUTOFF_EVAL for d in item_dates])
test_mask  = np.array([pd.notna(d) and d >= CUTOFF_EVAL for d in item_dates])

META_COLS_EVAL = ["price_num", "ea_flag", "num_genres", "num_tags"]
meta_eval = np.zeros((num_items, 6), dtype=np.float32)
df_g2 = df_games.set_index("item_idx").reindex(range(num_items))
for idx in range(num_items):
    row = df_g2.iloc[idx] if idx < len(df_g2) else None
    if row is not None:
        for j, col in enumerate(META_COLS_EVAL):
            v = row.get(col, 0)
            meta_eval[idx, j] = float(v) if pd.notna(v) else 0.0

TSCV_WINDOWS = [
    ("2013-07-01","2014-01-01"),("2014-01-01","2014-07-01"),
    ("2014-07-01","2015-01-01"),("2015-01-01","2015-07-01"),
    ("2015-07-01","2016-01-01"),("2016-01-01","2016-07-01"),
    ("2016-07-01","2017-01-01"),
]


def eval_catboost(item_emb, label, n_trials=60):
    X = np.hstack([item_emb, meta_eval])
    print(f"\n  [{label}] shape={X.shape}")

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

    model_cat = CatBoostRegressor(**best_p)
    model_cat.fit(X_tr, y_tr, verbose=False)
    pred = np.clip(model_cat.predict(X_te), 0, None)

    r2   = r2_score(y_te, pred)
    rmse = float(mean_squared_error(y_te, pred) ** 0.5)
    mae  = float(mean_absolute_error(y_te, pred))
    mask_nz = y_te > 0
    mape = float(np.mean(np.abs((y_te[mask_nz]-pred[mask_nz])/y_te[mask_nz]))) if mask_nz.sum()>0 else np.nan
    print(f"  Temporal: R²={r2:.4f} | RMSE={rmse:.2f} | MAE={mae:.2f}")

    ts_r2s = []
    for (tc, sc) in TSCV_WINDOWS:
        tc, sc = pd.Timestamp(tc), pd.Timestamp(sc)
        t_m = np.array([pd.notna(d) and d < tc for d in item_dates])
        e_m = np.array([pd.notna(d) and d >= tc and d < sc for d in item_dates])
        if e_m.sum() < 5 or t_m.sum() < 10: continue
        so2 = np.argsort(np.array(item_dates)[t_m])
        m2 = CatBoostRegressor(**best_p)
        m2.fit(X[t_m][so2], y_full[t_m][so2], verbose=False)
        ts_r2s.append(r2_score(y_full[e_m], np.clip(m2.predict(X[e_m]), 0, None)))
    r2_tscv = float(np.mean(ts_r2s)) if ts_r2s else float("nan")
    print(f"  TSCV: R²={r2_tscv:.4f}")

    kf_r2s = []
    vd_m = np.array([pd.notna(d) for d in item_dates])
    X_kf = X[vd_m]; y_kf = y_full[vd_m]
    for tri, tei in KFold(n_splits=5, shuffle=True, random_state=42).split(X_kf):
        mk = CatBoostRegressor(**best_p)
        mk.fit(X_kf[tri], y_kf[tri], verbose=False)
        kf_r2s.append(r2_score(y_kf[tei], np.clip(mk.predict(X_kf[tei]), 0, None)))
    r2_kfold = float(np.mean(kf_r2s))
    print(f"  KFold: R²={r2_kfold:.4f}")

    metrics = dict(r2_temporal=r2, rmse_temporal=rmse, mae_temporal=mae,
                   mape_temporal=mape, r2_tscv=r2_tscv, r2_kfold=r2_kfold)
    save_result(model_id=f"29_{label}_cat",
                model_name=f"MultiTask TT v2 ({label}) + CatBoost",
                features=f"RS multitask v2 ({label}) (64d) + meta (6d)",
                embeddings="v2_global",
                metrics=metrics,
                notes=f"v2 cutoff2017TT BCE+multitask v2 fix-split λ={LAMBDA_POP}: "
                      f"train={train_mask.sum()} test={test_mask.sum()}")
    return r2, rmse, r2_tscv, r2_kfold


r2_s, rmse_s, tscv_s, kf_s = eval_catboost(item_emb_std, "std")
r2_c, rmse_c, tscv_c, kf_c = eval_catboost(item_emb_cs,  "cs")

# ── 11. Resumen ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESUMEN — 29_xx MultiTask TT v2 (BCE + cutoff2017 + multi-task + fix-split + grad_clip)")
print("=" * 70)
print(f"{'Modelo':<35} {'R²':>8} {'RMSE':>10} {'TSCV':>8} {'KFold':>8}")
print("-" * 70)
print(f"  {'27_std (ID+content)':<33} {r2_s:>8.4f} {rmse_s:>10.2f} {tscv_s:>8.4f} {kf_s:>8.4f}")
print(f"  {'27_cs  (content only)':<33} {r2_c:>8.4f} {rmse_c:>10.2f} {tscv_c:>8.4f} {kf_c:>8.4f}")
print(f"\nBaseline (24_05_cat):  R²=0.4067 | RMSE=1014.86")
print(f"Delta std vs baseline: {r2_s - 0.4067:+.4f} R²  |  {1014.86 - rmse_s:+.2f} RMSE")
print("\n✓ Listo.")
