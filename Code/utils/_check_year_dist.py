import pandas as pd, numpy as np, json, ast

DATA = "Data"

with open(f"{DATA}/item2idx_v2.json") as f:
    item2idx_v2 = {k: int(v) for k, v in json.load(f).items()}
idx2appid = {v: k for k, v in item2idx_v2.items()}
N = len(item2idx_v2)

# Target v2
df_inter = pd.read_parquet(f"{DATA}/interactions_v2.parquet")
target_s = df_inter.groupby("item_idx").size().reindex(range(N), fill_value=0)
y_full   = target_s.values.astype(np.float64)

# Fechas
games_raw = []
with open(f"{DATA}/steam_games.json", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: games_raw.append(ast.literal_eval(line))
        except: pass

df_games = pd.json_normalize(games_raw).rename(columns={"id": "item_id"})
df_games["item_id"]  = df_games["item_id"].astype(str)
df_games["item_idx"] = df_games["item_id"].map(item2idx_v2)
df_games = df_games.dropna(subset=["item_idx"]).drop_duplicates(subset=["item_idx"])
df_games["item_idx"] = df_games["item_idx"].astype(int)
df_games["release_date_parsed"] = pd.to_datetime(df_games["release_date"], errors="coerce")
df_games["year"] = df_games["release_date_parsed"].dt.year
df_games["reviews"] = df_games["item_idx"].map(lambda i: y_full[i])

print("=" * 65)
print("Distribución de juegos y reviews por año")
print("=" * 65)
print(f"{'Año':>6} {'N juegos':>10} {'Mean rev':>10} {'Median':>8} {'Max':>8} {'Total rev':>12}")
print("-" * 65)

for year in sorted(df_games["year"].dropna().unique().astype(int)):
    sub = df_games[df_games["year"] == year]
    n    = len(sub)
    mean = sub["reviews"].mean()
    med  = sub["reviews"].median()
    mx   = sub["reviews"].max()
    tot  = sub["reviews"].sum()
    print(f"{year:>6} {n:>10,} {mean:>10.1f} {med:>8.0f} {mx:>8.0f} {tot:>12,.0f}")

print()
print("=" * 65)
print("Simulación de splits (train < cutoff, test >= cutoff)")
print("=" * 65)
print(f"{'Cutoff':>12} {'N train':>9} {'N test':>9} {'Mean train':>11} {'Mean test':>11} {'Ratio':>7}")
print("-" * 65)

cutoffs = ["2014-01-01","2014-07-01","2015-01-01","2015-07-01",
           "2016-01-01","2016-07-01","2017-01-01","2017-07-01","2018-01-01"]

for c in cutoffs:
    ct = pd.Timestamp(c)
    tr = df_games[df_games["release_date_parsed"].notna() & (df_games["release_date_parsed"] < ct)]
    te = df_games[df_games["release_date_parsed"].notna() & (df_games["release_date_parsed"] >= ct)]
    if len(te) == 0: continue
    ratio = tr["reviews"].mean() / te["reviews"].mean() if te["reviews"].mean() > 0 else 999
    print(f"{c:>12} {len(tr):>9,} {len(te):>9,} {tr['reviews'].mean():>11.1f} {te['reviews'].mean():>11.1f} {ratio:>7.2f}x")
