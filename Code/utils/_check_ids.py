import json, ast

games = []
with open("C:/Users/matia/Projects/VG_Recommender/Data/steam_games.json", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: games.append(ast.literal_eval(line))
        except: pass

print("Total juegos en steam_games.json:", len(games))
ids = [g.get("id") for g in games if g.get("id")]
print("Con ID:", len(ids))
print("Sample IDs:", ids[:5])
print("Sample keys:", list(games[0].keys()))

# Ver cuántos no tienen precio (free-to-play) vs de pago
free = sum(1 for g in games if str(g.get("price", "")).lower() in ["free", "free to play", "0", "0.0", ""])
print("Free-to-play / sin precio:", free)

# Ver rango de IDs (Steam AppIDs)
numeric_ids = [int(i) for i in ids if str(i).isdigit()]
if numeric_ids:
    print("AppID rango:", min(numeric_ids), "->", max(numeric_ids))
