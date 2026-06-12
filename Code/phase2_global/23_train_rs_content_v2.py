"""
23_train_rs_content_v2.py
==========================
Two-Tower content-augmented v2. Mejoras sobre 22_:

  1. Cold-start dropout (p=0.3 en ID embedding durante training)
     → Fuerza al content MLP a aprender señal independiente
     → En inferencia para juegos nuevos: ID emb es random, content MLP domina

  2. Cosine annealing LR + warmup
     → Mejor convergencia que LR fijo

  3. Más features de contenido: TF-IDF(100d) + numeric(4d) + RAWG quality(4d) = 108d
     → metacritic, rawg_rating, log_playtime, n_platforms

  4. Patience = 4 (antes: 2)

  5. Arquitectura item tower más profunda
     content_net: Linear(108→128) → GELU → Linear(128→64) → GELU → Linear(64→32)

Outputs:
  Data/item_embeddings_rs_v2_content2.npy        — (15380, 64)
  Data/item_embeddings_v2_content2_aligned.npy   — (3682, 64)
"""

import ast, json, time, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

os.environ["PYTHONHASHSEED"] = "42"
torch.manual_seed(42)
np.random.seed(42)

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

BASE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA   = f"{BASE}/Data"
GAMES  = f"{DATA}/steam_games.json"
CUTOFF = pd.Timestamp("2016-01-01")
EMB_DIM = 64
CS_DROP  = 0.3   # probabilidad de zerear ID emb (cold-start dropout)

print("\n" + "=" * 65)
print("23_train_rs_content_v2.py — Two-Tower content-augmented v2")
print("=" * 65)

# ── 1. Mappings ────────────────────────────────────────────────
print("\n[1] Cargando mappings...")
with open(f"{DATA}/user2idx_v2.json") as f:
    user2idx_v2 = {k: int(v) for k, v in json.load(f).items()}
with open(f"{DATA}/item2idx_v2.json") as f:
    item2idx_v2 = {k: int(v) for k, v in json.load(f).items()}

num_users = len(user2idx_v2)
num_items = len(item2idx_v2)
idx2appid = {v: k for k, v in item2idx_v2.items()}
print(f"  Usuarios: {num_users:,}  |  Items: {num_items:,}")

# ── 2. Content features ────────────────────────────────────────
print("\n[2] Construyendo content features (108d)...")

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

games_raw = []
with open(GAMES, "r", encoding="utf-8") as f:
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
df_games = df_games.drop_duplicates(subset=["item_idx"], keep="first")

df_games["tag_text"]   = df_games["tags"].apply(parse_list_col).apply(" ".join)
df_games["genre_text"] = df_games["genres"].apply(parse_list_col).apply(" ".join)
df_games["content"]    = df_games["tag_text"] + " " + df_games["genre_text"]
df_games["price"]      = df_games["price"].apply(clean_price)
df_games["ea_flag"]    = df_games["early_access"].fillna(False).astype(float)
df_games["num_tags"]   = df_games["tags"].apply(parse_list_col).apply(len).astype(float)
df_games["num_genres"] = df_games["genres"].apply(parse_list_col).apply(len).astype(float)

# TF-IDF 100d
valid_content = df_games[df_games["content"].str.strip() != ""]
tfidf = TfidfVectorizer(max_features=100, min_df=2, max_df=0.6, ngram_range=(1,1))
tfidf.fit(valid_content["content"])

# Cargar RAWG para las 4 features de calidad (sin ratings_count)
rawg_quality = np.zeros((num_items, 4), dtype=np.float32)
rawg_path  = f"{DATA}/rawg_enriched.csv"
v1map_path = f"{DATA}/item2idx.json"
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
    for idx in range(num_items):
        if idx not in df_rawg.index: continue
        r = df_rawg.loc[idx]
        rawg_quality[idx] = [
            float(r["rawg_rating"])  if pd.notna(r.get("rawg_rating"))  else 0.0,
            float(r["metacritic"])   if pd.notna(r.get("metacritic"))   else 0.0,
            np.log1p(float(r["playtime_avg_h"])) if pd.notna(r.get("playtime_avg_h")) else 0.0,
            float(len(parse_list_col(r.get("platforms", [])))),
        ]
    print(f"  RAWG quality features cargados para {(rawg_quality.sum(axis=1) > 0).sum()} items")

# Construir matriz content (N, 108)
CONTENT_DIM = 100 + 4 + 4  # tfidf + numeric + rawg_quality
content_matrix = np.zeros((num_items, CONTENT_DIM), dtype=np.float32)

item_to_row = {int(r["item_idx"]): i for i, (_, r) in enumerate(df_games.iterrows())}
tfidf_mat   = tfidf.transform(df_games["content"].fillna("")).toarray().astype(np.float32)

for idx in range(num_items):
    if idx not in item_to_row: continue
    row_i = item_to_row[idx]
    row   = df_games.iloc[row_i]
    content_matrix[idx, :100] = tfidf_mat[row_i]
    content_matrix[idx, 100]  = float(row["price"]     or 0)
    content_matrix[idx, 101]  = float(row["ea_flag"]   or 0)
    content_matrix[idx, 102]  = float(row["num_tags"]  or 0)
    content_matrix[idx, 103]  = float(row["num_genres"]or 0)
    content_matrix[idx, 104:] = rawg_quality[idx]

# Normalizar columnas numéricas (100-107)
for ci in range(100, CONTENT_DIM):
    vals = content_matrix[:, ci]
    mu, sigma = vals.mean(), vals.std()
    if sigma > 1e-8:
        content_matrix[:, ci] = (vals - mu) / sigma

content_tensor = torch.from_numpy(content_matrix)
print(f"  Content matrix: {content_matrix.shape}  ({content_matrix.nbytes/1e6:.1f} MB)")

# ── 3. Interacciones + filtro anti-leakage ─────────────────────
print("\n[3] Cargando interacciones (filtro pre-2016)...")
df = pd.read_parquet(f"{DATA}/interactions_v2.parquet")

date_map = {}
with open(GAMES, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            g = ast.literal_eval(line)
            gid, ds = str(g.get("id","")), g.get("release_date","")
            if gid and ds:
                try: date_map[gid] = pd.to_datetime(ds, dayfirst=True, errors="coerce")
                except: pass
        except: pass

pre2016_idxs = {item2idx_v2[aid] for aid, dt in date_map.items()
                if pd.notna(dt) and dt < CUTOFF and aid in item2idx_v2}
df = df[df["item_idx"].isin(pre2016_idxs)].reset_index(drop=True)
print(f"  {len(df):,} interacciones  ({df['item_idx'].nunique():,} items pre-2016)")

# ── 4. Split ───────────────────────────────────────────────────
users = df["user_idx"].values.astype(np.int64)
items_arr = df["item_idx"].values.astype(np.int64)
idx_tr, idx_va = train_test_split(np.arange(len(users)), test_size=0.10, random_state=42)
u_tr, i_tr = users[idx_tr], items_arr[idx_tr]
u_va, i_va = users[idx_va], items_arr[idx_va]
print(f"  Train: {len(u_tr):,}  |  Valid: {len(u_va):,}")

class ContentDataset(Dataset):
    def __init__(self, u, i):
        self.u = torch.from_numpy(u)
        self.i = torch.from_numpy(i)
    def __len__(self): return len(self.u)
    def __getitem__(self, idx): return self.u[idx], self.i[idx]

BATCH = 8192
train_loader = DataLoader(ContentDataset(u_tr, i_tr), batch_size=BATCH,
                          shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(ContentDataset(u_va, i_va), batch_size=BATCH*2,
                          shuffle=False, num_workers=0, pin_memory=True)
print(f"  Batch size: {BATCH}  |  Steps/epoch: {len(train_loader):,}")

# ── 5. Modelo con cold-start dropout ──────────────────────────
print("\n[4] Construyendo modelo v2 (cold-start dropout)...")

class ContentItemTowerV2(nn.Module):
    """
    Item tower v2 con cold-start dropout.
    Durante training: zerear ID emb con prob CS_DROP → fuerza content MLP a aprender.
    Durante inferencia: usar ambos (o solo content para items no vistos).
    """
    def __init__(self, num_items, content_dim, emb_dim=64, cs_drop=0.3):
        super().__init__()
        self.cs_drop = cs_drop
        half = emb_dim // 2

        self.id_emb = nn.Embedding(num_items, half)
        nn.init.normal_(self.id_emb.weight, std=0.01)

        self.content_net = nn.Sequential(
            nn.Linear(content_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, half),
        )

        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, item_ids, content_feats):
        id_vec      = self.id_emb(item_ids)           # (B, 32)
        content_vec = self.content_net(content_feats) # (B, 32)

        # Cold-start dropout: zerear ID emb durante training
        if self.training and self.cs_drop > 0:
            mask = (torch.rand(id_vec.shape[0], 1, device=id_vec.device) > self.cs_drop).float()
            id_vec = id_vec * mask

        combined = torch.cat([id_vec, content_vec], dim=1)  # (B, 64)
        return self.proj(combined)

    def get_cold_start_emb(self, item_ids, content_feats):
        """Inferencia para items nuevos: solo usa content (ID emb = 0)."""
        content_vec = self.content_net(content_feats)
        zeros       = torch.zeros_like(content_vec)
        combined    = torch.cat([zeros, content_vec], dim=1)
        return self.proj(combined)


class ContentTwoTowerV2(nn.Module):
    def __init__(self, num_users, num_items, content_dim, emb_dim=64, cs_drop=0.3):
        super().__init__()
        self.user_emb   = nn.Embedding(num_users, emb_dim)
        self.item_tower = ContentItemTowerV2(num_items, content_dim, emb_dim, cs_drop)
        nn.init.normal_(self.user_emb.weight, std=0.01)

    def forward(self, user_ids, item_ids, content_feats):
        u = self.user_emb(user_ids)
        v = self.item_tower(item_ids, content_feats)
        return (u * v).sum(dim=1)

    def get_item_emb(self, item_ids, content_feats, cold_start=False):
        if cold_start:
            return self.item_tower.get_cold_start_emb(item_ids, content_feats)
        return self.item_tower(item_ids, content_feats)


model = ContentTwoTowerV2(num_users, num_items, CONTENT_DIM, EMB_DIM, CS_DROP).to(DEVICE)
content_tensor = content_tensor.to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parámetros: {n_params:,}  ({n_params*4/1e6:.0f} MB en FP32)")

EPOCHS   = 15
PATIENCE = 4
LR       = 5e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()
scaler    = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

# Cosine annealing con warmup (2 epochs warmup)
WARMUP_STEPS = 2 * len(train_loader)
TOTAL_STEPS  = EPOCHS * len(train_loader)

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ── 6. Entrenamiento ───────────────────────────────────────────
print(f"\n[5] Entrenando ({EPOCHS} epochs, patience={PATIENCE}, lr={LR}, cs_drop={CS_DROP})...")
best_val, patience_cnt, best_state, step = float("inf"), 0, None, 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    t0, total_loss = time.time(), 0.0

    for u_batch, i_batch in train_loader:
        u_batch = u_batch.to(DEVICE)
        i_batch = i_batch.to(DEVICE)

        neg_items   = torch.randint(0, num_items, (len(u_batch),), device=DEVICE)
        users_all   = torch.cat([u_batch, u_batch])
        items_all   = torch.cat([i_batch, neg_items])
        labels      = torch.cat([torch.ones(len(u_batch), device=DEVICE),
                                  torch.zeros(len(u_batch), device=DEVICE)])
        content_all = content_tensor[items_all]

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type=="cuda")):
            logits = model(users_all, items_all, content_all)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
        step += 1

    avg_train = total_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for u_b, i_b in valid_loader:
            u_b = u_b.to(DEVICE); i_b = i_b.to(DEVICE)
            cont_b = content_tensor[i_b]
            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type=="cuda")):
                logits = model(u_b, i_b, cont_b)
                val_loss += nn.functional.binary_cross_entropy_with_logits(
                    logits, torch.ones_like(logits)).item()
    avg_val = val_loss / len(valid_loader)

    lr_now = scheduler.get_last_lr()[0] * LR
    print(f"  Epoch {epoch:02d}/{EPOCHS} | train={avg_train:.4f} | val={avg_val:.4f} | "
          f"lr={lr_now:.2e} | {time.time()-t0:.1f}s")

    if avg_val < best_val:
        best_val, patience_cnt = avg_val, 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"  Early stopping en epoch {epoch}")
            break

model.load_state_dict(best_state)
print(f"\n✓ Mejor val_loss: {best_val:.4f}")

# ── 7. Extraer embeddings ──────────────────────────────────────
print("\n[6] Extrayendo embeddings...")
model.eval(); model.cpu(); content_tensor = content_tensor.cpu()
all_ids = torch.arange(num_items, dtype=torch.long)
INFER_BATCH = 2048

# Embeddings estándar (ID + content) para items de training
emb_list = []
with torch.no_grad():
    for s in range(0, num_items, INFER_BATCH):
        e   = min(s + INFER_BATCH, num_items)
        emb = model.get_item_emb(all_ids[s:e], content_tensor[s:e], cold_start=False)
        emb_list.append(emb.numpy())
item_emb = np.vstack(emb_list).astype(np.float32)

np.save(f"{DATA}/item_embeddings_rs_v2_content2.npy", item_emb)
print(f"  ✓ item_embeddings_rs_v2_content2.npy  {item_emb.shape}")

# Embeddings cold-start (solo content) para TODOS los items (incluyendo post-2016)
emb_cs_list = []
with torch.no_grad():
    for s in range(0, num_items, INFER_BATCH):
        e   = min(s + INFER_BATCH, num_items)
        emb = model.get_item_emb(all_ids[s:e], content_tensor[s:e], cold_start=True)
        emb_cs_list.append(emb.numpy())
item_emb_cs = np.vstack(emb_cs_list).astype(np.float32)

np.save(f"{DATA}/item_embeddings_rs_v2_content2_cs.npy", item_emb_cs)
print(f"  ✓ item_embeddings_rs_v2_content2_cs.npy  {item_emb_cs.shape}  (cold-start: solo content)")

# ── 8. Alinear a v1 ────────────────────────────────────────────
print("\n[7] Alineando a v1...")
with open(f"{DATA}/item2idx.json") as f:
    item2idx_v1 = {k: int(v) for k, v in json.load(f).items()}

N_v1 = len(item2idx_v1)
for suffix, emb_src in [("content2", item_emb), ("content2_cs", item_emb_cs)]:
    aligned = np.zeros((N_v1, EMB_DIM), dtype=np.float32)
    matched = 0
    for appid, v1_idx in item2idx_v1.items():
        if appid in item2idx_v2:
            aligned[v1_idx] = emb_src[item2idx_v2[appid]]
            matched += 1
        else:
            aligned[v1_idx] = np.random.normal(0, 0.01, EMB_DIM).astype(np.float32)
    np.save(f"{DATA}/item_embeddings_v2_{suffix}_aligned.npy", aligned)
    print(f"  ✓ item_embeddings_v2_{suffix}_aligned.npy  ({matched}/{N_v1} mapeados)")

print("\n" + "=" * 65)
print("✓ Listo.")
print("  Usar content2    para items con historial (ID + content)")
print("  Usar content2_cs para cold-start puro (solo content MLP)")
print("=" * 65)
