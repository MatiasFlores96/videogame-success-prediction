import ast, json, pandas as pd
from collections import Counter

games = []
with open("C:/Users/matia/Projects/VG_Recommender/Data/steam_games.json", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: games.append(ast.literal_eval(line))
        except: pass

df = pd.json_normalize(games)
print("Columnas:", df.columns.tolist())
print()

# Analizar 'sentiment' si existe
if "sentiment" in df.columns:
    vc = df["sentiment"].value_counts().head(20)
    print("=== sentiment (top 20) ===")
    print(vc.to_string())
    print()

# Analizar 'reviews_url'
if "reviews_url" in df.columns:
    print("=== reviews_url sample (5) ===")
    print(df["reviews_url"].dropna().head(5).to_string())
    print()

# ¿Hay columnas de reviews count?
review_cols = [c for c in df.columns if "review" in c.lower() or "rating" in c.lower() or "score" in c.lower()]
print("Columnas relacionadas a reviews/rating:", review_cols)

# Cuántos tienen sentiment vs no
if "sentiment" in df.columns:
    print()
    print("Con sentiment:", df["sentiment"].notna().sum(), "/ Sin:", df["sentiment"].isna().sum())

# Distribución de release_date por año
df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
print()
print("=== Juegos por año (2013–2021) ===")
yr = df["year"].value_counts().sort_index()
for y in range(2013, 2022):
    print(f"  {y}: {yr.get(y, 0)}")
