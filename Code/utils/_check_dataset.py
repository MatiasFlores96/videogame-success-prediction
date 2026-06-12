import pandas as pd
import numpy as np
import ast

df = pd.read_parquet("Data/interactions_v2.parquet")
print("=== interactions_v2.parquet ===")
print(f"Filas:            {len(df):,}")
print(f"Usuarios unicos:  {df.user_idx.nunique():,}")
print(f"Items unicos:     {df.item_idx.nunique():,}")
print(f"hours min:        {df.hours.min():.2f}")
print(f"hours max:        {df.hours.max():.1f}")
print(f"hours mean:       {df.hours.mean():.2f}")
print(f"hours == 0:       {(df.hours == 0).sum():,}")
print(f"hours < 0.1:      {(df.hours < 0.1).sum():,}")
print()

# Dataset source - ver si hay README o scripts de descarga
import os
for fname in os.listdir("."):
    if fname.lower() in ["readme.md", "readme.txt", "data_readme.md"]:
        print(f"Encontrado: {fname}")

for fname in os.listdir("Code"):
    if "download" in fname.lower() or "preprocess" in fname.lower() or "01_" in fname:
        print(f"Code/{fname}")

print()

# Ver campos del steam_games.json
games_raw = []
with open("Data/steam_games.json", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 2: break
        line = line.strip()
        if not line:
            continue
        try:
            games_raw.append(ast.literal_eval(line))
        except:
            pass

if games_raw:
    print("=== Campos en steam_games.json (primer item) ===")
    g = games_raw[0]
    print(list(g.keys()))
    print()
    print("Ejemplo:")
    for k, v in list(g.items())[:8]:
        print(f"  {k}: {str(v)[:80]}")
