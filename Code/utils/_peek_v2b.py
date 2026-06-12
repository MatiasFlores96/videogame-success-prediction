"""
Exploración completa de:
  1. australian_users_items.json.gz  → playtime data
  2. steam_reviews.json.gz (v2)      → total de líneas, juegos únicos, rango temporal
  3. steam_games.json.gz (v2)        → qué juegos cubre
"""
import gzip, json
from collections import Counter
import time

# ─────────────────────────────────────────────────────────────────────────────
# 1. australian_users_items.json.gz — conteo completo
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("1. australian_users_items.json.gz  (COMPLETO)")
print("=" * 65)

path_items = "C:/Users/matia/Downloads/australian_users_items.json.gz"
n_users = 0; n_interactions = 0; games_seen = set()

t0 = time.time()
with gzip.open(path_items, "rt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: rec = json.loads(line)
        except:
            try: rec = eval(line)
            except: continue
        n_users += 1
        for item in rec.get("items", []):
            n_interactions += 1
            games_seen.add(item.get("item_id"))

print(f"Usuarios:        {n_users:,}")
print(f"Interacciones:   {n_interactions:,}  (playtime_forever >= 0)")
print(f"Juegos únicos:   {len(games_seen):,}")
print(f"Media items/usr: {n_interactions/n_users:.1f}")
print(f"Tiempo: {time.time()-t0:.1f}s")

# Interacciones con playtime > 0
print("\n(Para saber cuántos tienen playtime > 0, muestra de 5K users...)")
path_items2 = "C:/Users/matia/Downloads/australian_users_items.json.gz"
n5k = 0; played_pos = 0
with gzip.open(path_items2, "rt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: rec = json.loads(line)
        except:
            try: rec = eval(line)
            except: continue
        n5k += 1
        for item in rec.get("items", []):
            if int(item.get("playtime_forever", 0)) > 0:
                played_pos += 1
        if n5k >= 5000: break

print(f"En 5K usuarios: {played_pos:,} interacciones con playtime > 0 ({played_pos/(n5k*97.5)*100:.1f}% del total)")

# ─────────────────────────────────────────────────────────────────────────────
# 2. steam_reviews.json.gz (v2) — conteo completo (puede tardar ~2 min)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("2. steam_reviews.json.gz  (v2, conteo completo)")
print("=" * 65)
print("Contando... (puede tardar 1-2 min)")

path_v2 = "C:/Users/matia/Downloads/steam_reviews.json.gz"
total_v2 = 0; games_v2 = set(); users_v2 = set(); years_v2 = Counter()
t0 = time.time()

with gzip.open(path_v2, "rt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: rec = json.loads(line)
        except:
            try: rec = eval(line)
            except: continue
        total_v2 += 1
        games_v2.add(rec.get("product_id"))
        users_v2.add(rec.get("username"))
        date_raw = rec.get("date","")
        if date_raw:
            try: years_v2[int(str(date_raw)[:4])] += 1
            except: pass
        if total_v2 % 500_000 == 0:
            print(f"  {total_v2:,} líneas procesadas ({time.time()-t0:.0f}s)...")

print(f"\nTotal reviews:   {total_v2:,}")
print(f"Juegos únicos:   {len(games_v2):,}")
print(f"Usuarios únicos: {len(users_v2):,}")
print(f"Tiempo: {time.time()-t0:.1f}s")
print("\nDistribución por año:")
for y in sorted(years_v2):
    print(f"  {y}: {years_v2[y]:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. steam_games.json.gz (v2) — ¿qué juegos cubre?
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("3. steam_games.json.gz  (v2 metadata)")
print("=" * 65)

path_sg = "C:/Users/matia/Downloads/steam_games.json.gz"
sg_total = 0; sg_years = Counter(); sg_sample = []

with gzip.open(path_sg, "rt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: rec = json.loads(line)
        except:
            try: rec = eval(line)
            except: continue
        sg_total += 1
        date_raw = rec.get("release_date","")
        if date_raw:
            import pandas as pd
            try:
                y = pd.to_datetime(date_raw, errors="coerce")
                if y is not None and not pd.isna(y):
                    sg_years[y.year] += 1
            except: pass
        if sg_total <= 1: sg_sample.append(rec)

print(f"Total juegos: {sg_total:,}")
print(f"Keys: {list(sg_sample[0].keys()) if sg_sample else 'N/A'}")
print("Por año (últimos):")
import pandas as pd
for y in sorted(sg_years.keys())[-10:]:
    print(f"  {y}: {sg_years[y]:,}")
