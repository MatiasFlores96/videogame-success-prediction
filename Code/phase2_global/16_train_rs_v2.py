"""
16_train_rs_v2.py
==================
Reentrena el Two-Tower RS usando steam_reviews v2 (7.76M interacciones).

Cambios vs notebook 02:
  - PyTorch + CUDA (RTX 4080)
  - Negative sampling explícito (1:1 pos:neg) — el original no tenía negatives
  - Batch size 8192 aprovechando GPU
  - AMP (mixed precision) para velocidad

FILTRO ANTI-LEAKAGE: igual que notebook 02, solo interacciones de juegos
lanzados ANTES de 2016. Los embeddings de test no codifican su propia popularidad.

Outputs:
  Data/item_embeddings_rs_v2.npy      — (N_items_v2, 64)
  Data/item_embeddings_v2_aligned.npy — (3682, 64)  alineados a v1 item2idx
  Data/user_embeddings_rs_v2.npy      — (N_users_v2, 64)
"""

import ast, json, time, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

os.environ["PYTHONHASHSEED"] = "42"
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = f"{BASE}/Data"
GAMES = f"{DATA}/steam_games.json"
CUTOFF = pd.Timestamp("2016-01-01")

print("\n" + "=" * 65)
print("16_train_rs_v2.py — Two-Tower con datos v2 + GPU")
print("=" * 65)

# ── 1. Cargar interacciones v2 ─────────────────────────────────────────────
print("\nPaso 1: Cargando datos ...")
df = pd.read_parquet(f"{DATA}/interactions_v2.parquet")

with open(f"{DATA}/user2idx_v2.json") as f:
    user2idx_v2 = {k: int(v) for k, v in json.load(f).items()}
with open(f"{DATA}/item2idx_v2.json") as f:
    item2idx_v2 = {k: int(v) for k, v in json.load(f).items()}

num_users_v2 = len(user2idx_v2)
num_items_v2 = len(item2idx_v2)
print(f"  Usuarios: {num_users_v2:,}  |  Items: {num_items_v2:,}")
print(f"  Interacciones totales: {len(df):,}")

# ── 2. Filtro anti-leakage: solo juegos pre-2016 ───────────────────────────
print("\nPaso 2: Filtro anti-leakage (juegos pre-2016) ...")
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
                try:
                    date_map[gid] = pd.to_datetime(ds, dayfirst=True, errors="coerce")
                except Exception: pass
        except Exception: pass

pre2016_appids    = {gid for gid, dt in date_map.items() if pd.notna(dt) and dt < CUTOFF}
pre2016_item_idxs = {item2idx_v2[aid] for aid in pre2016_appids if aid in item2idx_v2}

n_before = len(df)
df = df[df["item_idx"].isin(pre2016_item_idxs)].reset_index(drop=True)
print(f"  {n_before:,} → {len(df):,} interacciones  ({df['item_idx'].nunique():,} items únicos pre-2016)")

# ── 3. Arrays y split ──────────────────────────────────────────────────────
print("\nPaso 3: Split 90/10 ...")
users  = df["user_idx"].values.astype(np.int64)
items  = df["item_idx"].values.astype(np.int64)

idx_tr, idx_va = train_test_split(np.arange(len(users)), test_size=0.10, random_state=42)
u_tr, i_tr = users[idx_tr], items[idx_tr]
u_va, i_va = users[idx_va], items[idx_va]
print(f"  Train: {len(u_tr):,}  |  Valid: {len(u_va):,}")

# ── 4. Dataset con negative sampling ──────────────────────────────────────
class ImplicitDataset(Dataset):
    """Para cada positivo (u,i) genera un negativo (u, random_j)."""
    def __init__(self, users, items, num_items, neg_ratio=1):
        self.users     = torch.from_numpy(users)
        self.items     = torch.from_numpy(items)
        self.num_items = num_items
        self.neg_ratio = neg_ratio

    def __len__(self):
        return len(self.users) * (1 + self.neg_ratio)

    def __getitem__(self, idx):
        n_pos = len(self.users)
        if idx < n_pos:
            return self.users[idx], self.items[idx], torch.tensor(1.0)
        else:
            # Negativo aleatorio
            pos_idx = (idx - n_pos) % n_pos
            u = self.users[pos_idx]
            j = torch.randint(0, self.num_items, (1,)).item()
            return u, torch.tensor(j, dtype=torch.long), torch.tensor(0.0)

# Usamos solo el train set para negatives (valid set evalúa solo positivos)
# Para eficiencia, los negatives son random en batch (más rápido que por índice)
class FastImplicitDataset(Dataset):
    """Retorna solo positivos; los negatives se generan en el training loop."""
    def __init__(self, users, items):
        self.users = torch.from_numpy(users)
        self.items = torch.from_numpy(items)

    def __len__(self): return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx]

train_ds = FastImplicitDataset(u_tr, i_tr)
valid_ds = FastImplicitDataset(u_va, i_va)

BATCH = 8192
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_ds, batch_size=BATCH * 2, shuffle=False,
                          num_workers=0, pin_memory=True)

print(f"  Batch size: {BATCH}  |  Steps/epoch: {len(train_loader):,}")

# ── 5. Modelo Two-Tower ────────────────────────────────────────────────────
print("\nPaso 4: Construyendo modelo ...")

class TwoTower(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)   # (B, 64)
        v = self.item_emb(item_ids)   # (B, 64)
        return (u * v).sum(dim=1)     # dot product → scalar logit

model = TwoTower(num_users_v2, num_items_v2, emb_dim=64).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parámetros: {n_params:,}  ({n_params*4/1e6:.0f} MB en FP32)")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
scaler    = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

# ── 6. Entrenamiento ───────────────────────────────────────────────────────
print("\nPaso 5: Entrenando ...")
EPOCHS   = 10
PATIENCE = 2
best_val = float("inf")
patience_cnt = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    model.train()
    t0 = time.time()
    total_loss = 0.0

    for u_batch, i_batch in train_loader:
        u_batch = u_batch.to(DEVICE)
        i_batch = i_batch.to(DEVICE)

        # Generar negatives en-batch (mismo usuario, item aleatorio)
        neg_items = torch.randint(0, num_items_v2, (len(u_batch),), device=DEVICE)

        users_combined = torch.cat([u_batch, u_batch])
        items_combined = torch.cat([i_batch, neg_items])
        labels = torch.cat([
            torch.ones(len(u_batch),  device=DEVICE),
            torch.zeros(len(u_batch), device=DEVICE)
        ])

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type=="cuda")):
            logits = model(users_combined, items_combined)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # --- Validation (solo positivos) ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for u_batch, i_batch in valid_loader:
            u_batch = u_batch.to(DEVICE)
            i_batch = i_batch.to(DEVICE)
            # Solo positivos en validación
            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type=="cuda")):
                logits = model(u_batch, i_batch)
                loss   = nn.functional.binary_cross_entropy_with_logits(
                    logits, torch.ones_like(logits))
            val_loss += loss.item()
    avg_val_loss = val_loss / len(valid_loader)

    elapsed = time.time() - t0
    print(f"  Epoch {epoch:02d}/{EPOCHS} | train_loss={avg_train_loss:.4f} | "
          f"val_loss={avg_val_loss:.4f} | {elapsed:.1f}s")

    # Early stopping
    if avg_val_loss < best_val:
        best_val = avg_val_loss
        patience_cnt = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"  Early stopping en epoch {epoch}")
            break

model.load_state_dict(best_state)
print(f"\n✓ Mejor val_loss: {best_val:.4f}")

# ── 7. Extraer y guardar embeddings ───────────────────────────────────────
print("\nPaso 6: Guardando embeddings ...")
model.eval()
model.cpu()

item_emb_matrix = model.item_emb.weight.detach().numpy()   # (num_items_v2, 64)
user_emb_matrix = model.user_emb.weight.detach().numpy()   # (num_users_v2, 64)

np.save(f"{DATA}/item_embeddings_rs_v2.npy", item_emb_matrix)
np.save(f"{DATA}/user_embeddings_rs_v2.npy", user_emb_matrix)
print(f"  item_embeddings_rs_v2.npy: {item_emb_matrix.shape}")
print(f"  user_embeddings_rs_v2.npy: {user_emb_matrix.shape}")

# ── 8. Alinear a v1 item2idx (para usar en experimentos XGBoost) ───────────
print("\nPaso 7: Alineando a v1 item2idx ...")

with open(f"{DATA}/item2idx.json") as f:
    item2idx_v1 = {k: int(v) for k, v in json.load(f).items()}

N_v1    = len(item2idx_v1)
aligned = np.zeros((N_v1, 64), dtype=np.float32)
matched = 0; cold = 0

for appid_str, v1_idx in item2idx_v1.items():
    if appid_str in item2idx_v2:
        aligned[v1_idx] = item_emb_matrix[item2idx_v2[appid_str]]
        matched += 1
    else:
        aligned[v1_idx] = np.random.normal(0, 0.01, 64).astype(np.float32)
        cold += 1

np.save(f"{DATA}/item_embeddings_v2_aligned.npy", aligned)

print(f"  Total juegos v1:   {N_v1:,}")
print(f"  Con embedding v2:  {matched:,}  ({matched/N_v1*100:.1f}%)")
print(f"  Cold-start:        {cold:,}    ({cold/N_v1*100:.1f}%)")
print(f"  → item_embeddings_v2_aligned.npy: {aligned.shape}")

print("\n" + "=" * 65)
print("✓ Listo. Usar item_embeddings_v2_aligned.npy en los experimentos.")
print("=" * 65)
