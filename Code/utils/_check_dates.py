import pandas as pd, json, ast
import warnings; warnings.filterwarnings('ignore')

df_inter = pd.read_parquet("C:/Users/matia/Projects/VG_Recommender/Data/interactions.parquet")
print("=== INTERACCIONES ===")
print("Shape:", df_inter.shape)
print("Cols:", df_inter.columns.tolist())

if "posted" in df_inter.columns:
    df_inter["posted"] = pd.to_datetime(df_inter["posted"], errors="coerce")
    print("Rango reviews:", df_inter["posted"].min(), "->", df_inter["posted"].max())
    vc = df_inter["posted"].dt.year.value_counts().sort_index()
    print("Reviews por año:")
    for y, c in vc.items():
        print(f"  {y}: {c}")

games = []
with open("C:/Users/matia/Projects/VG_Recommender/Data/steam_games.json", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: games.append(ast.literal_eval(line))
        except: pass

df_g = pd.json_normalize(games)
df_g["release_date_parsed"] = pd.to_datetime(df_g["release_date"], errors="coerce")
print()
print("=== JUEGOS — rango release dates ===")
print("Total en steam_games.json:", len(df_g))
print("Con fecha:", df_g["release_date_parsed"].notna().sum())
print("Rango:", df_g["release_date_parsed"].min(), "->", df_g["release_date_parsed"].max())
print("Por año (últimos 10):")
vc2 = df_g["release_date_parsed"].dt.year.value_counts().sort_index()
for y, c in vc2.tail(10).items():
    print(f"  {y}: {c}")
