import pandas as pd, numpy as np, json, ast

DATA = "Data"

# V2 interactions
df = pd.read_parquet(f"{DATA}/interactions_v2.parquet")
print(f"interactions_v2: {len(df):,} rows")
target_v2 = df.groupby("item_idx").size().reset_index(name="total_reviews")
print(f"Items con reviews: {len(target_v2):,}")
tv = target_v2.total_reviews
print(f"Target stats: min={tv.min()} max={tv.max()} mean={tv.mean():.1f} median={tv.median():.0f} p99={tv.quantile(.99):.0f}")

# Embeddings v2
emb = np.load(f"{DATA}/item_embeddings_rs_v2.npy")
print(f"\nitem_embeddings_rs_v2: {emb.shape}")

with open(f"{DATA}/item2idx_v2.json") as f:
    item2idx_v2 = {k: int(v) for k, v in json.load(f).items()}
print(f"item2idx_v2: {len(item2idx_v2):,} items")

# Cuantos tienen metadata en steam_games.json
date_map, meta_map = {}, {}
with open(f"{DATA}/steam_games.json", encoding="utf-8") as f:
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
            meta_map[gid] = g
        except: pass

appids_v2 = set(item2idx_v2.keys())
appids_with_date = {k for k, v in date_map.items() if pd.notna(v)}
overlap = appids_v2 & appids_with_date

CUTOFF = pd.Timestamp("2016-01-01")
pre2016  = {k for k in overlap if date_map[k] < CUTOFF}
post2016 = {k for k in overlap if date_map[k] >= CUTOFF}

print(f"\nItems v2 con fecha en steam_games: {len(overlap):,} ({len(overlap)/len(appids_v2)*100:.1f}%)")
print(f"  Pre-2016  (train): {len(pre2016):,}")
print(f"  Post-2016 (test):  {len(post2016):,}")
print(f"  Sin fecha:         {len(appids_v2)-len(overlap):,}")

# Target stats para pre/post split
idx2appid = {v: k for k, v in item2idx_v2.items()}
target_v2["appid"] = target_v2["item_idx"].map(idx2appid)
pre_ids  = {item2idx_v2[a] for a in pre2016}
post_ids = {item2idx_v2[a] for a in post2016}
tr = target_v2[target_v2["item_idx"].isin(pre_ids)]["total_reviews"]
te = target_v2[target_v2["item_idx"].isin(post_ids)]["total_reviews"]
print(f"\nTarget en train (pre-2016): n={len(tr):,}  mean={tr.mean():.1f}  median={tr.median():.0f}  max={tr.max()}")
print(f"Target en test  (post-2016): n={len(te):,}  mean={te.mean():.1f}  median={te.median():.0f}  max={te.max()}")

# Comparar con v1
with open(f"{DATA}/item2idx.json") as f:
    item2idx_v1 = {k: int(v) for k, v in json.load(f).items()}
print(f"\nComparacion universo:")
print(f"  v1 (australiano): {len(item2idx_v1):,} items")
print(f"  v2 (global):      {len(item2idx_v2):,} items total  /  {len(overlap):,} con fecha")
print(f"  Train: {len(pre2016):,} vs 2,621 antes  ({len(pre2016)/2621:.1f}x)")
print(f"  Test:  {len(post2016):,} vs 486 antes    ({len(post2016)/486:.1f}x)")
