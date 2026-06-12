"""
Peek at steam_reviews.json.gz (v2, 1.3GB) without loading everything.
Also peek at australian_users_items.json.gz.
"""
import gzip, json
from collections import Counter
from datetime import datetime

# ── steam_reviews.json.gz (v2) ─────────────────────────────────────────────
print("=" * 65)
print("steam_reviews.json.gz  (McAuley v2)")
print("=" * 65)

path_v2 = "C:/Users/matia/Downloads/steam_reviews.json.gz"
samples = []
years = Counter()
has_recommend = 0
total = 0

with gzip.open(path_v2, "rt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            try:
                rec = eval(line)
            except Exception:
                continue

        total += 1

        if total <= 3:
            samples.append(rec)

        # fecha
        date_raw = rec.get("date") or rec.get("posted") or ""
        if date_raw:
            try:
                y = int(str(date_raw)[:4])
                if 2000 < y < 2030:
                    years[y] += 1
            except Exception:
                pass

        if "recommend" in rec:
            has_recommend += 1

        if total == 50_000:   # primer vistazo rápido
            break

print(f"Líneas leídas (primer vistazo): {total:,}")
print(f"Keys de muestra: {list(samples[0].keys()) if samples else 'N/A'}")
print(f"¿Tiene campo 'recommend'?: {'Sí' if has_recommend > 0 else 'No'}")
print()
print("Primeros 2 registros:")
for s in samples[:2]:
    print(" ", {k: str(v)[:80] for k, v in s.items()})
print()
print("Distribución por año (primeros 50K registros):")
for y in sorted(years):
    print(f"  {y}: {years[y]:,}")

# ── australian_users_items.json.gz ─────────────────────────────────────────
print()
print("=" * 65)
print("australian_users_items.json.gz  (playtime + purchases, v1)")
print("=" * 65)

path_items = "C:/Users/matia/Downloads/australian_users_items.json.gz"
items_samples = []
total_items = 0
total_interactions = 0

with gzip.open(path_items, "rt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            try:
                rec = eval(line)
            except Exception:
                continue

        total_items += 1
        items = rec.get("items", [])
        total_interactions += len(items)

        if total_items <= 2:
            items_samples.append(rec)

        if total_items == 5_000:
            break

print(f"Usuarios (primer vistazo 5K): {total_items:,}")
print(f"Interacciones totales (5K users): {total_interactions:,}")
print(f"Promedio items/usuario: {total_interactions/max(total_items,1):.1f}")
print()
print("Keys del registro:", list(items_samples[0].keys()) if items_samples else "N/A")
if items_samples and items_samples[0].get("items"):
    print("Keys de un item:", list(items_samples[0]["items"][0].keys()))
    print("Muestra item:", items_samples[0]["items"][0])
