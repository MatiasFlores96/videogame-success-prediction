import json, ast
import numpy as np
import pandas as pd

DATA = "Data"

with open(f"{DATA}/item2idx_v2.json") as f:
    item2idx_v2 = {k: int(v) for k, v in json.load(f).items()}
num_items = len(item2idx_v2)

df_inter = pd.read_parquet(f"{DATA}/interactions_v2.parquet")
target_s = df_inter.groupby("item_idx").size().reindex(range(num_items), fill_value=0)
y_full = target_s.values.astype(np.float64)

games_raw = []
with open(f"{DATA}/steam_games.json", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: games_raw.append(ast.literal_eval(line))
        except: pass

df_games = pd.json_normalize(games_raw).rename(columns={"id": "item_id"})
df_games["item_id"] = df_games["item_id"].astype(str)
df_games["item_idx"] = df_games["item_id"].map(item2idx_v2)
df_games = df_games.dropna(subset=["item_idx"])
df_games["item_idx"] = df_games["item_idx"].astype(int)
df_games["release_date_parsed"] = pd.to_datetime(df_games["release_date"], errors="coerce")
df_games = df_games.drop_duplicates(subset=["item_idx"], keep="first")

CUTOFF = pd.Timestamp("2017-01-01")
meta_reind = df_games.set_index("item_idx").reindex(range(num_items))
item_dates = meta_reind["release_date_parsed"].values

train_mask = np.array([pd.notna(d) and d < CUTOFF for d in item_dates])
test_mask  = np.array([pd.notna(d) and d >= CUTOFF for d in item_dates])

y_train = y_full[train_mask]
y_test  = y_full[test_mask]

mean_train = y_train.mean()
rmse_null  = np.sqrt(np.mean((y_test - mean_train)**2))
rmse_model = 1014.86

print(f"Items train (pre-2017): {train_mask.sum():,}")
print(f"Items test  (post-2017): {test_mask.sum():,}")
print()
print(f"y_test  media:  {y_test.mean():.1f}")
print(f"y_test  std:    {y_test.std():.1f}")
print(f"y_test  mediana:{np.median(y_test):.1f}")
print(f"y_test  max:    {y_test.max():.0f}")
print()
print(f"--- y_full (todos los items) ---")
print(f"  media:   {y_full.mean():.1f}")
print(f"  std:     {y_full.std():.1f}")
print(f"  mediana: {np.median(y_full):.1f}")
print(f"  max:     {y_full.max():.0f}  ({df_inter.groupby('item_idx').size().idxmax()})")
print(f"  top 5:   {sorted(y_full, reverse=True)[:5]}")
print()
print(f"--- y_train (pre-2017, {train_mask.sum():,} items) ---")
print(f"  media:   {y_train.mean():.1f}")
print(f"  max:     {y_train.max():.0f}")
print(f"  top 5:   {sorted(y_train, reverse=True)[:5]}")
print()
print(f"--- y_test (post-2017, {test_mask.sum():,} items) ---")
print(f"  media:   {y_test.mean():.1f}")
print(f"  mediana: {np.median(y_test):.1f}")
print(f"  max:     {y_test.max():.0f}")
print()
print(f"Prediccion nula (media del train): {mean_train:.1f}")
print(f"RMSE nulo  (predecir media): {rmse_null:.2f}")
print(f"RMSE modelo (24_05_cat):     {rmse_model:.2f}")
print(f"Reduccion:  {rmse_null - rmse_model:.2f}  ({(rmse_null - rmse_model)/rmse_null*100:.1f}%)")
