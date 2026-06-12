"""Peek rápido de steam_games.json.gz (v2)"""
import gzip, json
import pandas as pd
from collections import Counter

path = "C:/Users/matia/Downloads/steam_games.json.gz"
total = 0; years = Counter(); samples = []

with gzip.open(path, "rt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: rec = json.loads(line)
        except:
            try: rec = eval(line)
            except: continue
        total += 1
        if total <= 2: samples.append(rec)
        date = rec.get("release_date", "")
        try:
            y = pd.to_datetime(date, errors="coerce")
            if pd.notna(y): years[y.year] += 1
        except: pass

print(f"Total juegos en steam_games.json.gz (v2): {total:,}")
print(f"Keys: {list(samples[0].keys()) if samples else 'N/A'}")
print("\nPor año:")
for y in sorted(years.keys()):
    print(f"  {y}: {years[y]:,}")
print("\nMuestra 1:")
if samples:
    print({k: str(v)[:60] for k, v in samples[0].items()})
