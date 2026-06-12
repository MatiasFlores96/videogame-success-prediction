"""
26_train_rs_improved.py
========================
Two-Tower mejorado con 4 cambios sobre el baseline (23_train_rs_content_v2.py):

1. Cutoff pre-2017 (era pre-2016):
   → 2016 games con señal colaborativa → mejores embeddings para el train set del regresor

2. InfoNCE loss (era BCE + random negatives):
   → In-batch negatives contrastivos: B positivos vs B×(B-1) negativos implícitos
   → Temperatura aprendible → embeddings más discriminativos

3. Multi-task: head auxiliar predice log(review_count) del item
   → El Two-Tower optimiza directamente para popularidad, no solo para recomendación

4. User content features (user tower era ID-only):
   → User tower: ID_Emb(32) + ContentMLP(perfil_avg_items→32) → concat(64) → proj
   → Permite inferir preferencias de usuarios sin historial (semi-cold-start)

Output:
  item_embeddings_rs_v2_improved.npy   (15380, 64) — standard
  item_embeddings_rs_v2_improved_cs.npy (15380, 64) — cold-start (ID=0)
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
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_tracker import save_result

BASE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA   = os.path.join(BASE, "Data")

# ── Configuración ──────────────────────────────────────────────────────────────
CUTOFF_TT   = pd.Timestamp("2017-01-01")  # FIX: era 2016, ahora usamos todo el train
CUTOFF_EVAL = pd.Timestamp("2017-01-01")  # cutoff del regresor downstream

EMB_DIM     = 64
CS_DROP     = 0.3
LR          = 5e-4
EPOCHS      = 15
PATIENCE    = 4
BATCH       = 4096          # más pequeño que 8192 porque InfoNCE usa B×B
TEMPERATURE = 0.1           # temperatura InfoNCE (fija; podría ser aprendible)
LAMBDA_POP  = 0.3           # peso del multi-task loss de popularidad
WARMUP_EP   = 2             # épocas de warmup para LR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("26_train_rs_improved.py — Two-Tower con 4 mejoras")
print("=" * 70)
print(f"  Dispositivo: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── 1. Mappings ────────────────────────────────────────────────────────────────
print("\n[1] Cargando mappings...")
with open(os.path.join(DATA, "item2idx_v2.json")) as f:
    item2idx_v2 = {k: int(v) for k, v in json.load(f).items()}
num_items = len(item2idx_v2)

with open(os.path.join(DATA, "user2idx_v2.json")) as f:
    user2idx_v2 = {k: int(v) for k, v in json.load(f).items()}
num_users = len(user2idx_v2)
print(f"  Usuarios: {num_users:,}  |  Items: {num_items:,}")

# Target (review counts) para multi-task
df_inter = pd.read_parquet(os.path.join(DATA, "interactions_v2.parquet"))
target_s = df_inter.groupby("item_idx").size().reindex(range(num_items), fill_value=0)
y_full   = target_s.values.astype(np.float64)

# Target normalizado en log-space para el multi-task head
y_log_raw = np.log1p(y_full).astype(np.float32)
y_log_mean = float(y_log_raw.mean())
y_log_std  = float(y_log_raw.std()) + 1e-8
y_log_norm = (y_log_raw - y_log_mean) / y_log_std  # z-score → multi-task target

# ── 2. Metadata y fechas ───────────────────────────────────────────────────────
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

# FIX: pre-2017 en vez de pre-2016
pre_cutoff_items = {gid for gid, d in date_map.items() if pd.notna(d) and d < CUTOFF_TT}
print(f"  Items pre-2017 (Two-Tower train): {len(pre_cutoff_items):,}  (era ~6,953 pre-2016)")

# ── 3. Content features ────────────────────────────────────────────────────────
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

# RAWG quality
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
    rq_matched = 0
    for idx in range(num_items):
        if idx not in df_rawg.index: continue
        r = df_rawg.loc[idx]
        rawg_quality[idx] = [
            float(r["rawg_rating"])  if pd.notna(r.get("rawg_rating"))  else 0.0,
            float(r["metacritic"])   if pd.notna(r.get("metacritic"))   else 0.0,
            np.log1p(float(r["playtime_avg_h"])) if pd.notna(r.get("playtime_avg_h")) else 0.0,
            float(ESRB.get(r.get("esrb_rating",""), 0)),
        ]
        rq_matched += 1
    print(f"  RAWG quality: {rq_matched:,} items")

# Construir content matrix (108d)
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

scaler_num  = StandardScaler().fit(num_aligned[list(pre_cutoff_items)])
scaler_rawg = StandardScaler().fit(rawg_quality[list(pre_cutoff_items)])
num_scaled  = scaler_num.transform(num_aligned).astype(np.float32)
rawg_scaled = scaler_rawg.transform(rawg_quality).astype(np.float32)

content_matrix = np.hstack([tfidf_aligned, num_scaled, rawg_scaled])  # (N, 108)
CONTENT_DIM    = content_matrix.shape[1]
print(f"  Content matrix: {content_matrix.shape}")

# ── 4. Interacciones Two-Tower ─────────────────────────────────────────────────
print("\n[4] Cargando interacciones (pre-2017)...")
inter_filt = df_inter[df_inter["item_idx"].isin(pre_cutoff_items)].copy()
u_all = inter_filt["user_idx"].values.astype(np.int64)
i_all = inter_filt["item_idx"].values.astype(np.int64)
n     = len(u_all)
val_cut = int(n * 0.9)
u_tr, u_va = u_all[:val_cut], u_all[val_cut:]
i_tr, i_va = i_all[:val_cut], i_all[val_cut:]
print(f"  {n:,} interacciones  ({len(pre_cutoff_items):,} items pre-2017)")
print(f"  Train: {len(u_tr):,}  |  Valid: {len(u_va):,}")

# ── 5. User content profiles (para user content tower) ─────────────────────────
print("\n[5] Computando user content profiles...")
# Para cada usuario: promedio de content features de los items con los que interactuó
USER_CONTENT_DIM = CONTENT_DIM
user_content_sum = np.zeros((num_users, USER_CONTENT_DIM), dtype=np.float64)
user_content_cnt = np.zeros(num_users, dtype=np.int32)
# Solo usar train (no val para evitar leakage)
for u, i in zip(u_tr, i_tr):
    user_content_sum[u] += content_matrix[i]
    user_content_cnt[u] += 1
mask = user_content_cnt > 0
user_content_avg = np.zeros((num_users, USER_CONTENT_DIM), dtype=np.float32)
user_content_avg[mask] = (user_content_sum[mask] / user_content_cnt[mask, None]).astype(np.float32)
n_users_with_profile = int(mask.sum())
print(f"  Usuarios con perfil de contenido: {n_users_with_profile:,}/{num_users:,} "
      f"({n_users_with_profile/num_users*100:.1f}%)")

# Tensores en CPU (demasiado grande para GPU, se carga por batch)
content_tensor      = torch.tensor(content_matrix,  dtype=torch.float32)
user_content_tensor = torch.tensor(user_content_avg, dtype=torch.float32)
y_log_tensor        = torch.tensor(y_log_norm,       dtype=torch.float32)

# ── 6. Arquitectura ────────────────────────────────────────────────────────────
print("\n[6] Construyendo modelo mejorado...")

class ItemTower(nn.Module):
    """ID embedding + content MLP + cold-start dropout + popularity head."""
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
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )
        # Multi-task: predice popularidad desde el embedding del item
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
        emb = self.proj(torch.cat([id_vec, content_vec], dim=1))
        return emb

    def get_cold_start_emb(self, item_ids, content_feats):
        content_vec = self.content_net(content_feats)
        zeros       = torch.zeros_like(content_vec)
        return self.proj(torch.cat([zeros, content_vec], dim=1))

    def predict_popularity(self, emb):
        return self.pop_head(emb).squeeze(-1)


class UserTower(nn.Module):
    """ID embedding + user content profile MLP."""
    def __init__(self, num_users, user_content_dim, emb_dim=64):
        super().__init__()
        half = emb_dim // 2

        self.id_emb = nn.Embedding(num_users, half)
        nn.init.normal_(self.id_emb.weight, std=0.01)

        self.content_net = nn.Sequential(
            nn.Linear(user_content_dim, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, half),
        )
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, user_ids, user_content_feats):
        id_vec      = self.id_emb(user_ids)
        content_vec = self.content_net(user_content_feats)
        return self.proj(torch.cat([id_vec, content_vec], dim=1))


class ImprovedTwoTower(nn.Module):
    def __init__(self, num_users, num_items, content_dim, emb_dim=64, cs_drop=0.3):
        super().__init__()
        self.user_tower = UserTower(num_users, content_dim, emb_dim)
        self.item_tower = ItemTower(num_items, content_dim, emb_dim, cs_drop)

    def forward(self, user_ids, item_ids, content_feats, user_content_feats):
        u = self.user_tower(user_ids, user_content_feats)
        v = self.item_tower(item_ids, content_feats)
        return u, v

    def get_item_emb(self, item_ids, content_feats):
        return self.item_tower(item_ids, content_feats)

    def get_cold_start_emb(self, item_ids, content_feats):
        return self.item_tower.get_cold_start_emb(item_ids, content_feats)


# InfoNCE loss
def info_nce_loss(u_emb, v_emb, temperature=0.1):
    """
    Contrastive loss: para cada usuario, identifica su item positivo
    entre todos los items del batch (in-batch negatives).
    """
    u_norm = F.normalize(u_emb, dim=1)
    v_norm = F.normalize(v_emb, dim=1)
    # sim_matrix[i,j] = similitud entre usuario i e item j
    sim = torch.matmul(u_norm, v_norm.T) / temperature   # (B, B)
    labels = torch.arange(len(u_emb), device=u_emb.device)
    # Cross-entropy simétrica: user→item + item→user
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
    return loss


model = ImprovedTwoTower(num_users, num_items, CONTENT_DIM, EMB_DIM, CS_DROP).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parámetros: {n_params:,}  ({n_params*4/1e6:.0f} MB FP32)")

# ── 7. DataLoader ──────────────────────────────────────────────────────────────
class InteractionDataset(Dataset):
    def __init__(self, u, i):
        self.u = torch.tensor(u, dtype=torch.long)
        self.i = torch.tensor(i, dtype=torch.long)
    def __len__(self): return len(self.u)
    def __getitem__(self, idx): return self.u[idx], self.i[idx]

train_loader = DataLoader(InteractionDataset(u_tr, i_tr), batch_size=BATCH,
                          shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(InteractionDataset(u_va, i_va), batch_size=BATCH*2,
                          shuffle=False, num_workers=0, pin_memory=True)
print(f"  Batch: {BATCH}  |  Steps/epoch: {len(train_loader):,}")

# Mover tensores grandes a device (content y user_content)
content_tensor      = content_tensor.to(DEVICE)
user_content_tensor = user_content_tensor.to(DEVICE)
y_log_tensor        = y_log_tensor.to(DEVICE)

# ── 8. Optimizador y scheduler ─────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scaler_amp = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

WARMUP_STEPS = WARMUP_EP * len(train_loader)
TOTAL_STEPS  = EPOCHS * len(train_loader)

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ── 9. Training ────────────────────────────────────────────────────────────────
print(f"\n[7] Entrenando ({EPOCHS} epochs, patience={PATIENCE}, "
      f"lr={LR}, cs_drop={CS_DROP}, temp={TEMPERATURE}, λ_pop={LAMBDA_POP})...")

best_val, patience_cnt, best_state, step = float("inf"), 0, None, 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    total_nce, total_pop, total_n = 0.0, 0.0, 0

    for u_batch, i_batch in train_loader:
        u_batch = u_batch.to(DEVICE)
        i_batch = i_batch.to(DEVICE)

        cont_b      = content_tensor[i_batch]
        user_cont_b = user_content_tensor[u_batch]
        pop_target  = y_log_tensor[i_batch]

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type=="cuda")):
            u_emb, v_emb = model(u_batch, i_batch, cont_b, user_cont_b)

            # InfoNCE contrastiva
            loss_nce = info_nce_loss(u_emb, v_emb, TEMPERATURE)

            # Multi-task: popularidad del item
            pop_pred  = model.item_tower.predict_popularity(v_emb)
            loss_pop  = F.mse_loss(pop_pred, pop_target)

            loss = loss_nce + LAMBDA_POP * loss_pop

        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()
        scheduler.step()

        total_nce += loss_nce.item()
        total_pop += loss_pop.item()
        total_n   += 1
        step += 1

    avg_nce = total_nce / total_n
    avg_pop = total_pop / total_n

    # Validación: solo InfoNCE (sin multi-task para medir calidad colaborativa)
    model.eval()
    val_nce = 0.0
    with torch.no_grad():
        for u_b, i_b in valid_loader:
            u_b = u_b.to(DEVICE); i_b = i_b.to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type=="cuda")):
                u_e, v_e = model(u_b, i_b, content_tensor[i_b], user_content_tensor[u_b])
                val_nce += info_nce_loss(u_e, v_e, TEMPERATURE).item()
    avg_val = val_nce / len(valid_loader)

    print(f"  Epoch {epoch:02d}/{EPOCHS} | nce={avg_nce:.4f} | pop={avg_pop:.4f} | "
          f"val_nce={avg_val:.4f} | {time.time()-t0:.1f}s")

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
print(f"\n✓ Mejor val_nce: {best_val:.4f}")

# ── 10. Extraer embeddings ─────────────────────────────────────────────────────
print("\n[8] Extrayendo embeddings...")
all_ids  = torch.arange(num_items, device=DEVICE)
BS_INF   = 2048
embs_std, embs_cs = [], []

with torch.no_grad():
    for start in range(0, num_items, BS_INF):
        ids_b  = all_ids[start:start+BS_INF]
        cont_b = content_tensor[ids_b]
        embs_std.append(model.get_item_emb(ids_b, cont_b).cpu().numpy())
        embs_cs.append(model.get_cold_start_emb(ids_b, cont_b).cpu().numpy())

item_emb_std = np.vstack(embs_std)
item_emb_cs  = np.vstack(embs_cs)
print(f"  Standard:   {item_emb_std.shape}")
print(f"  Cold-start: {item_emb_cs.shape}")

np.save(os.path.join(DATA, "item_embeddings_rs_v2_improved.npy"),    item_emb_std)
np.save(os.path.join(DATA, "item_embeddings_rs_v2_improved_cs.npy"), item_emb_cs)
print("  ✓ Guardados")

# ── 11. Evaluación downstream con CatBoost ────────────────────────────────────
print("\n[9] Evaluación downstream (CatBoost, mismo setup que 24_05_cat)...")

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
    print(f"  Features: {X.shape}")

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
    save_result(model_id=f"26_{label}_cat",
                model_name=f"Improved TT ({label}) + CatBoost",
                features=f"RS improved ({label}) (64d) + meta (6d)",
                embeddings="v2_global",
                metrics=metrics,
                notes=f"v2 cutoff2017 InfoNCE+multitask+userContent+cutoff2017TT: "
                      f"train={train_mask.sum()} test={test_mask.sum()}")
    return r2, rmse, r2_tscv, r2_kfold


print("\n--- Embedding standard (ID + content) ---")
r2_s, rmse_s, tscv_s, kf_s = eval_catboost(item_emb_std, "std")

print("\n--- Embedding cold-start (solo content MLP) ---")
r2_c, rmse_c, tscv_c, kf_c = eval_catboost(item_emb_cs, "cs")

# ── 12. Resumen ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESUMEN FINAL — Two-Tower mejorado (26_xx)")
print("=" * 70)
print(f"{'Modelo':<35} {'R²':>8} {'RMSE':>10} {'TSCV':>8} {'KFold':>8}")
print("-" * 70)
print(f"  {'26 Standard (ID+content)':<33} {r2_s:>8.4f} {rmse_s:>10.2f} {tscv_s:>8.4f} {kf_s:>8.4f}")
print(f"  {'26 Cold-start (content only)':<33} {r2_c:>8.4f} {rmse_c:>10.2f} {tscv_c:>8.4f} {kf_c:>8.4f}")
print(f"\nBaseline a superar (24_05_cat):  R²=0.4067 | RMSE=1014.86 | TSCV=0.5221")
print(f"Mejora en R²: {r2_s - 0.4067:+.4f}  |  Mejora en RMSE: {1014.86 - rmse_s:+.2f}")
print("\n✓ Listo.")
