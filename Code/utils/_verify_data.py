import pandas as pd
import numpy as np
import json, ast

DATA = "Data"

# ── 1. Dataset crudo ───────────────────────────────────────────────────────────
df_inter = pd.read_parquet(f"{DATA}/interactions_v2.parquet")
print("=== interactions_v2.parquet (lo que tenemos guardado) ===")
print(f"  Filas totales:         {len(df_inter):,}")
print(f"  Usuarios unicos:       {df_inter.user_idx.nunique():,}")
print(f"  Items unicos:          {df_inter.item_idx.nunique():,}")
print(f"  Hours > 0:             {(df_inter.hours > 0).sum():,}  (todas, MIN_HOURS=0)")
print(f"  Dataset McAuley total: 7,793,069  (segun web)")
print(f"  Diferencia (filtrados con hours=0): {7_793_069 - len(df_inter):,}")
print()

# ── 2. Lo que usa y_full (el target del modelo) ────────────────────────────────
with open(f"{DATA}/item2idx_v2.json") as f:
    item2idx_v2 = {k: int(v) for k, v in json.load(f).items()}
num_items = len(item2idx_v2)

target_s = df_inter.groupby("item_idx").size().reindex(range(num_items), fill_value=0)
y_full = target_s.values.astype(np.float64)

print("=== y_full — target del modelo (usa TODAS las interacciones) ===")
print(f"  Items totales mapeados: {num_items:,}")
print(f"  Interacciones sumadas:  {int(y_full.sum()):,}  (debe == filas del parquet)")
print(f"  Items con 0 reviews:    {(y_full == 0).sum():,}")
print()

# ── 3. Filtro para Two-Tower training (pre-2017) ───────────────────────────────
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

CUTOFF_TT   = pd.Timestamp("2016-01-01")  # cutoff Two-Tower (23_ baseline)
CUTOFF_EVAL = pd.Timestamp("2017-01-01")  # cutoff XGBoost eval

meta_reind = df_games.set_index("item_idx").reindex(range(num_items))
item_dates = meta_reind["release_date_parsed"].values

pre2016 = {int(r.item_idx) for _, r in df_games.iterrows()
           if pd.notna(r.release_date_parsed) and r.release_date_parsed < CUTOFF_TT}
pre2017 = {int(r.item_idx) for _, r in df_games.iterrows()
           if pd.notna(r.release_date_parsed) and r.release_date_parsed < CUTOFF_EVAL}

inter_tt_train = df_inter[df_inter["item_idx"].isin(pre2016)]
inter_eval_all = df_inter  # y_full usa todas

train_mask = np.array([pd.notna(d) and d < CUTOFF_EVAL for d in item_dates])
test_mask  = np.array([pd.notna(d) and d >= CUTOFF_EVAL for d in item_dates])

print("=== Desglose de uso de interacciones ===")
print(f"  Total en parquet:               {len(df_inter):,}")
print(f"  Usadas en Two-Tower training    ")
print(f"    (items pre-2016, cutoff TT):  {len(inter_tt_train):,}  ({len(inter_tt_train)/len(df_inter)*100:.1f}%)")
print(f"  Items pre-2016 (TT training):   {len(pre2016):,}")
print(f"  Items pre-2017 (XGB train):     {sum(train_mask):,}")
print(f"  Items post-2017 (XGB test):     {sum(test_mask):,}")
print()
print("=== y_full por split (target del XGBoost) ===")
print(f"  Interacciones contadas en y_full[train_mask]: {int(y_full[train_mask].sum()):,}")
print(f"  Interacciones contadas en y_full[test_mask]:  {int(y_full[test_mask].sum()):,}")
print(f"  Total:                                        {int(y_full.sum()):,}")
print()
print("CONCLUSION: y_full usa el parquet COMPLETO (las {len(df_inter):,} filas)")
print("El Two-Tower usa solo el subconjunto pre-2016 para ENTRENAMIENTO del RS.")
print("Pero el TARGET y_full siempre cuenta reviews de TODOS los períodos.")
