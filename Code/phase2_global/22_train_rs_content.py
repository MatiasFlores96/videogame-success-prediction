"""
22_train_rs_content.py
=======================
Two-Tower RS con content-augmented item tower.

El item tower clásico solo recibe item_idx → embedding (ID-based).
Para juegos sin historial (cold-start), el embedding es ruido.

Mejora: el item tower combina ID embedding + content features:
  item_id  →  Embedding(32d) ──┐
                                 concat(64d) → Linear(64,64) → item_vec
  content  →  MLP(content_dim→32d) ──┘

Esto permite producir embeddings informativos para juegos nuevos (2017+)
basados en sus tags, géneros y precio — aunque nunca aparezcan en training.

Outputs:
  Data/item_embeddings_rs_v2_content.npy  — (15380, 64) todos los items
  Data/item_embeddings_v2_content_aligned.npy — (3682, 64) alineados a v1
"""

import ast, json, time, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

os.environ["PYTHONHASHSEED"] = "42"
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

BASE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA   = f"{BASE}/Data"
GAMES  = f"{DATA}/steam_games.json"
CUTOFF = pd.Timestamp("2016-01-01")
EMB_DIM = 64

print("\n" + "=" * 65)
print("22_train_rs_content.py — Two-Tower content-augmented")
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

# ── 2. Construir content features para todos los items ─────────
print("\n[2] Construyendo content features...")

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

# TF-IDF: fitear sobre todos los items (tags/géneros son estáticos, no revelan popularidad)
valid_content = df_games[df_games["content"].str.strip() != ""]
tfidf = TfidfVectorizer(max_features=50, min_df=2, max_df=0.6, ngram_range=(1,1))
tfidf.fit(valid_content["content"])
print(f"  TF-IDF vocab: {len(tfidf.vocabulary_)} features")

# Construir matriz content (N, 54): TF-IDF(50) + price, ea, num_tags, num_genres
CONTENT_DIM = 50 + 4
content_matrix = np.zeros((num_items, CONTENT_DIM), dtype=np.float32)

item_to_row = {int(r["item_idx"]): i for i, (_, r) in enumerate(df_games.iterrows())}
tfidf_transformed = tfidf.transform(df_games["content"].fillna("")).toarray().astype(np.float32)

for idx in range(num_items):
    if idx not in item_to_row: continue
    row_i = item_to_row[idx]
    row   = df_games.iloc[row_i]
    content_matrix[idx, :50] = tfidf_transformed[row_i]
    content_matrix[idx, 50]  = float(row["price"]     or 0)
    content_matrix[idx, 51]  = float(row["ea_flag"]   or 0)
    content_matrix[idx, 52]  = float(row["num_tags"]  or 0)
    content_matrix[idx, 53]  = float(row["num_genres"]or 0)

# Normalizar features numéricas (cols 50-53) con estadísticas de todos los items
for ci in range(50, 54):
    vals = content_matrix[:, ci]
    mu, sigma = vals.mean(), vals.std()
    if sigma > 0:
        content_matrix[:, ci] = (vals - mu) / sigma

content_tensor = torch.from_numpy(content_matrix)  # (N, 54) en CPU
print(f"  Content matrix: {content_matrix.shape}  ({content_matrix.nbytes/1e6:.1f} MB)")

# ── 3. Cargar interacciones y filtro anti-leakage ─────────────
print("\n[3] Cargando interacciones (filtro pre-2016)...")
df = pd.read_parquet(f"{DATA}/interactions_v2.parquet")

date_map = {}
with open(GAMES, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            g = ast.literal_eval(line)
            gid = str(g.get("id", ""))
            ds  = g.get("release_date", "")
            if gid and ds:
                try: date_map[gid] = pd.to_datetime(ds, dayfirst=True, errors="coerce")
                except: pass
        except: pass

pre2016_appids    = {gid for gid, dt in date_map.items() if pd.notna(dt) and dt < CUTOFF}
pre2016_item_idxs = {item2idx_v2[aid] for aid in pre2016_appids if aid in item2idx_v2}

n_before = len(df)
df = df[df["item_idx"].isin(pre2016_item_idxs)].reset_index(drop=True)
print(f"  {n_before:,} → {len(df):,} interacciones  ({df['item_idx'].nunique():,} items únicos pre-2016)")

# ── 4. Split ───────────────────────────────────────────────────
print("\n[4] Split 90/10...")
users = df["user_idx"].values.astype(np.int64)
items = df["item_idx"].values.astype(np.int64)
idx_tr, idx_va = train_test_split(np.arange(len(users)), test_size=0.10, random_state=42)
u_tr, i_tr = users[idx_tr], items[idx_tr]
u_va, i_va = users[idx_va], items[idx_va]
print(f"  Train: {len(u_tr):,}  |  Valid: {len(u_va):,}")

# ── 5. Dataset ─────────────────────────────────────────────────
class ContentDataset(Dataset):
    """Retorna (user_idx, item_idx) — content se lookea en forward pass."""
    def __init__(self, users, items):
        self.users = torch.from_numpy(users)
        self.items = torch.from_numpy(items)
    def __len__(self): return len(self.users)
    def __getitem__(self, idx): return self.users[idx], self.items[idx]

BATCH = 8192
train_loader = DataLoader(ContentDataset(u_tr, i_tr), batch_size=BATCH,
                          shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(ContentDataset(u_va, i_va), batch_size=BATCH * 2,
                          shuffle=False, num_workers=0, pin_memory=True)
print(f"  Batch size: {BATCH}  |  Steps/epoch: {len(train_loader):,}")

# ── 6. Modelo Content-Augmented Two-Tower ─────────────────────
print("\n[5] Construyendo modelo content-augmented...")

class ContentItemTower(nn.Module):
    """Item tower: ID emb(32d) + content MLP(32d) → concat → Linear(64d)."""
    def __init__(self, num_items, content_dim, emb_dim=64):
        super().__init__()
        half = emb_dim // 2
        self.id_emb = nn.Embedding(num_items, half)
        self.content_net = nn.Sequential(
            nn.Linear(content_dim, half * 2),
            nn.GELU(),
            nn.Linear(half * 2, half),
        )
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        nn.init.normal_(self.id_emb.weight, std=0.01)

    def forward(self, item_ids, content_feats):
        id_vec      = self.id_emb(item_ids)                    # (B, 32)
        content_vec = self.content_net(content_feats)          # (B, 32)
        combined    = torch.cat([id_vec, content_vec], dim=1)  # (B, 64)
        return self.norm(self.proj(combined))                   # (B, 64)


class ContentTwoTower(nn.Module):
    def __init__(self, num_users, num_items, content_dim, emb_dim=64):
        super().__init__()
        self.user_emb  = nn.Embedding(num_users, emb_dim)
        self.item_tower = ContentItemTower(num_items, content_dim, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)

    def get_item_emb(self, item_ids, content_feats):
        return self.item_tower(item_ids, content_feats)

    def forward(self, user_ids, item_ids, content_feats):
        u = self.user_emb(user_ids)
        v = self.item_tower(item_ids, content_feats)
        return (u * v).sum(dim=1)   # dot product logit


model = ContentTwoTower(num_users, num_items, CONTENT_DIM, emb_dim=EMB_DIM).to(DEVICE)
content_tensor = content_tensor.to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"  Parámetros: {n_params:,}  ({n_params*4/1e6:.0f} MB en FP32)")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
scaler    = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

# ── 7. Entrenamiento ───────────────────────────────────────────
print("\n[6] Entrenando...")
EPOCHS   = 12
PATIENCE = 2
best_val  = float("inf")
patience_cnt = 0
best_state   = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    total_loss = 0.0

    for u_batch, i_batch in train_loader:
        u_batch = u_batch.to(DEVICE)
        i_batch = i_batch.to(DEVICE)

        # In-batch negatives
        neg_items = torch.randint(0, num_items, (len(u_batch),), device=DEVICE)

        users_all  = torch.cat([u_batch, u_batch])
        items_all  = torch.cat([i_batch, neg_items])
        labels     = torch.cat([
            torch.ones(len(u_batch),  device=DEVICE),
            torch.zeros(len(u_batch), device=DEVICE)
        ])
        content_all = content_tensor[items_all]

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type=="cuda")):
            logits = model(users_all, items_all, content_all)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    avg_train = total_loss / len(train_loader)

    # Validación
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for u_batch, i_batch in valid_loader:
            u_batch = u_batch.to(DEVICE)
            i_batch = i_batch.to(DEVICE)
            content_b = content_tensor[i_batch]
            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type=="cuda")):
                logits = model(u_batch, i_batch, content_b)
                loss   = nn.functional.binary_cross_entropy_with_logits(
                    logits, torch.ones_like(logits))
            val_loss += loss.item()
    avg_val = val_loss / len(valid_loader)

    elapsed = time.time() - t0
    print(f"  Epoch {epoch:02d}/{EPOCHS} | train={avg_train:.4f} | val={avg_val:.4f} | {elapsed:.1f}s")

    if avg_val < best_val:
        best_val = avg_val
        patience_cnt = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"  Early stopping en epoch {epoch}")
            break

model.load_state_dict(best_state)
print(f"\n✓ Mejor val_loss: {best_val:.4f}")

# ── 8. Extraer embeddings para TODOS los items ─────────────────
print("\n[7] Extrayendo embeddings para todos los items...")
model.eval()
model.cpu()
content_tensor = content_tensor.cpu()

all_ids = torch.arange(num_items, dtype=torch.long)
INFER_BATCH = 2048
emb_list = []

with torch.no_grad():
    for start in range(0, num_items, INFER_BATCH):
        end      = min(start + INFER_BATCH, num_items)
        ids_b    = all_ids[start:end]
        cont_b   = content_tensor[start:end]
        emb_b    = model.get_item_emb(ids_b, cont_b)
        emb_list.append(emb_b.numpy())

item_emb_content = np.vstack(emb_list).astype(np.float32)  # (num_items, 64)
print(f"  item_emb_content shape: {item_emb_content.shape}")

# Guardar
np.save(f"{DATA}/item_embeddings_rs_v2_content.npy", item_emb_content)
print(f"  ✓ Guardado: item_embeddings_rs_v2_content.npy")

# ── 9. Alinear a v1 ────────────────────────────────────────────
print("\n[8] Alineando a v1 item2idx...")
with open(f"{DATA}/item2idx.json") as f:
    item2idx_v1 = {k: int(v) for k, v in json.load(f).items()}

N_v1    = len(item2idx_v1)
aligned = np.zeros((N_v1, EMB_DIM), dtype=np.float32)
matched = 0

for appid_str, v1_idx in item2idx_v1.items():
    if appid_str in item2idx_v2:
        aligned[v1_idx] = item_emb_content[item2idx_v2[appid_str]]
        matched += 1
    else:
        aligned[v1_idx] = np.random.normal(0, 0.01, EMB_DIM).astype(np.float32)

np.save(f"{DATA}/item_embeddings_v2_content_aligned.npy", aligned)
print(f"  Mapeados: {matched}/{N_v1} ({matched/N_v1*100:.1f}%)")
print(f"  ✓ Guardado: item_embeddings_v2_content_aligned.npy")

print("\n" + "=" * 65)
print("✓ Listo. Usar item_embeddings_rs_v2_content.npy en el pipeline.")
print("=" * 65)
