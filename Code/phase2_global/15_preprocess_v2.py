"""
15_preprocess_v2.py
====================
Preprocesa steam_reviews.json.gz (v2, 7.79M reviews) para reentrenar el Two-Tower.

Señal de interacción: presencia de review + hours > 0  →  positivo implícito

Outputs:
  Data/interactions_v2.parquet   — (user_idx, item_idx, hours) solo positivos
  Data/user2idx_v2.json          — username → idx
  Data/item2idx_v2.json          — product_id (AppID) → idx
  (imprime estadísticas de solapamiento con v1)
"""

import gzip, json, time
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

BASE    = Path(__file__).resolve().parent.parent.parent
SRC_V2  = Path("C:/Users/matia/Downloads/steam_reviews.json.gz")
OUT_DIR = BASE / "Data"

MIN_HOURS = 0.0   # umbral: hours > MIN_HOURS → positivo
# Usamos 0 porque el acto de reseñar ya es señal de engagement

print("=" * 65)
print("15_preprocess_v2.py — Preprocesando steam_reviews v2")
print("=" * 65)
print(f"Umbral de horas: > {MIN_HOURS}")
print()

# ── 1. Parsear v2 ─────────────────────────────────────────────────────────────
print("Paso 1: Parseando steam_reviews.json.gz ...")
t0 = time.time()

users_set  = set()
items_set  = set()
raw_rows   = []          # (username, product_id, hours)
skipped    = 0

with gzip.open(SRC_V2, "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            try:
                rec = eval(line)
            except Exception:
                skipped += 1
                continue

        username   = rec.get("username", "")
        product_id = str(rec.get("product_id", "")).strip()
        hours_raw  = rec.get("hours", "0")

        if not username or not product_id:
            skipped += 1
            continue

        try:
            hours = float(str(hours_raw).replace(",", "."))
        except Exception:
            hours = 0.0

        if hours > MIN_HOURS:   # filtro
            users_set.add(username)
            items_set.add(product_id)
            raw_rows.append((username, product_id, hours))

        if (i + 1) % 1_000_000 == 0:
            print(f"  {i+1:,} líneas leídas | positivos hasta ahora: {len(raw_rows):,}  ({time.time()-t0:.0f}s)")

print(f"\n  Total líneas leídas: {i+1:,}")
print(f"  Líneas con parse error: {skipped:,}")
print(f"  Interacciones positivas (hours > {MIN_HOURS}): {len(raw_rows):,}")
print(f"  Usuarios únicos: {len(users_set):,}")
print(f"  Juegos únicos:   {len(items_set):,}")
print(f"  Tiempo lectura: {time.time()-t0:.1f}s")

# ── 2. Crear índices ───────────────────────────────────────────────────────────
print("\nPaso 2: Creando mappings user2idx / item2idx ...")

user2idx = {u: i for i, u in enumerate(sorted(users_set))}
item2idx = {g: i for i, g in enumerate(sorted(items_set))}

with open(OUT_DIR / "user2idx_v2.json", "w") as f:
    json.dump(user2idx, f)
with open(OUT_DIR / "item2idx_v2.json", "w") as f:
    json.dump(item2idx, f)

print(f"  user2idx_v2: {len(user2idx):,} usuarios")
print(f"  item2idx_v2: {len(item2idx):,} juegos")

# ── 3. Construir DataFrame y guardar parquet ───────────────────────────────────
print("\nPaso 3: Construyendo parquet de interacciones ...")

df = pd.DataFrame(raw_rows, columns=["username", "product_id", "hours"])
df["user_idx"] = df["username"].map(user2idx).astype(np.int32)
df["item_idx"] = df["product_id"].map(item2idx).astype(np.int32)
df["hours"]    = df["hours"].astype(np.float32)
df = df[["user_idx", "item_idx", "hours"]].copy()

out_path = OUT_DIR / "interactions_v2.parquet"
df.to_parquet(out_path, index=False)

print(f"  Guardado: {out_path}")
print(f"  Shape:    {df.shape}")
print(f"  Memoria:  {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# ── 4. Solapamiento con v1 (target variable) ───────────────────────────────────
print("\nPaso 4: Verificando solapamiento con v1 ...")

# Cargar item2idx de v1
with open(OUT_DIR / "item2idx.json") as f:
    item2idx_v1 = {k: int(v) for k, v in json.load(f).items()}

v1_appids = set(item2idx_v1.keys())      # AppIDs en dataset v1 (target)
v2_appids = set(item2idx.keys())         # AppIDs en v2 reviews

overlap     = v1_appids & v2_appids
only_in_v1  = v1_appids - v2_appids     # cold-start en v2 también
only_in_v2  = v2_appids - v1_appids     # juegos en v2 sin target

print(f"  Juegos en v1 (con target): {len(v1_appids):,}")
print(f"  Juegos en v2 (con reviews): {len(v2_appids):,}")
print(f"  SOLAPAMIENTO (tienen target Y reviews v2): {len(overlap):,}  ({len(overlap)/len(v1_appids)*100:.1f}% de v1)")
print(f"  Solo en v1 (cold-start incluso en v2):    {len(only_in_v1):,}")
print(f"  Solo en v2 (sin target):                  {len(only_in_v2):,}")

# ── 5. Análisis de cold-start antes vs después ────────────────────────────────
print("\nPaso 5: Análisis cold-start antes vs después ...")

import ast
# Cargar release dates
games = []
with open(BASE / "Data" / "steam_games.json", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: games.append(ast.literal_eval(line))
        except: pass

df_games = pd.json_normalize(games).rename(columns={"id": "item_id"})
df_games["release_date_parsed"] = pd.to_datetime(df_games["release_date"], errors="coerce")
df_games["item_id"] = df_games["item_id"].astype(str)
id_to_date = df_games.set_index("item_id")["release_date_parsed"].to_dict()

CUTOFF = pd.Timestamp("2016-01-01")

cold_v1  = {aid for aid in v1_appids if id_to_date.get(str(aid), pd.NaT) is not pd.NaT
            and pd.notna(id_to_date.get(str(aid), pd.NaT))
            and id_to_date[str(aid)] >= CUTOFF}

# En v2, un juego ya NO es cold-start si tiene reviews
cold_v2  = cold_v1 - v2_appids   # post-2016 SIN reviews en v2 → sigue siendo cold-start

print(f"  Juegos post-2016 en v1 (test set): {len(cold_v1):,}")
print(f"  De esos, CON reviews en v2 (ya no cold-start): {len(cold_v1 & v2_appids):,}  ({len(cold_v1 & v2_appids)/max(len(cold_v1),1)*100:.1f}%)")
print(f"  De esos, AÚN cold-start (sin reviews en v2):   {len(cold_v2):,}  ({len(cold_v2)/max(len(cold_v1),1)*100:.1f}%)")

print("\n✓ Preprocesamiento completo.")
print(f"  → interactions_v2.parquet listo para reentrenar Two-Tower")
