"""
37_sentence_embeddings.py
==========================
Reemplaza el TF-IDF (100d) del content tower por sentence embeddings de
all-MiniLM-L6-v2 (384d) sobre el texto de tags + géneros.

Hipótesis: el TF-IDF trata los tags como bolsa de palabras; un encoder
semántico captura relaciones entre tags ("roguelike" ≈ "dungeon crawler")
que podrían dar mejor representación para cold-start.

Content matrix v37: MiniLM(384d) + numeric(4d) + RAWG quality(4d) = 392d
Tower v3 (fixes), cs_drop=0.3, seed=42.
Eval downstream: CatBoost emb (64d) + meta5 (5d).

Output: Data/item_embeddings_rs_v2_content_minilm.npy / _cs.npy
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tower_v3
from tower_v3 import train_tower, get_base_data, build_content_matrix, DATA, _parse_list_col
from v2_data import load_all, run_model

print("=" * 70)
print("37_sentence_embeddings.py — MiniLM content tower")
print("=" * 70)

# ── 1. Encodear tags+géneros con MiniLM ────────────────────────────────────────
base = get_base_data()
df_games  = base["df_games"]
num_items = base["num_items"]
pre2016   = base["pre2016_idxs"]

print("\n[1] Encodeando texto de tags+géneros con all-MiniLM-L6-v2...")
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

texts = [""] * num_items
for _, row in df_games.iterrows():
    tags   = _parse_list_col(row.get("tags"))
    genres = _parse_list_col(row.get("genres"))
    texts[int(row["item_idx"])] = ", ".join(genres + tags)

sent_emb = encoder.encode(texts, batch_size=512, show_progress_bar=True,
                          convert_to_numpy=True).astype(np.float32)
print(f"  Sentence embeddings: {sent_emb.shape}")

# Liberar VRAM del encoder antes de entrenar el tower
del encoder
import torch; torch.cuda.empty_cache()

# ── 2. Content matrix v37 = MiniLM(384) + numeric(4) + RAWG(4) ────────────────
# Reusa las columnas numéricas/RAWG ya construidas (y z-scoreadas pre-2016)
# por build_content_matrix: son las últimas 8 columnas de la matriz estándar.
std_content = build_content_matrix()
content_37  = np.hstack([sent_emb, std_content[:, 100:]]).astype(np.float32)
print(f"  Content matrix v37: {content_37.shape}")

# ── 3. Entrenar tower + eval ───────────────────────────────────────────────────
out = train_tower(cs_drop=0.3, seed=42, neg_mode="uniform_pre2016",
                  content_matrix=content_37)
np.save(os.path.join(DATA, "item_embeddings_rs_v2_content_minilm.npy"),    out["emb_std"])
np.save(os.path.join(DATA, "item_embeddings_rs_v2_content_minilm_cs.npy"), out["emb_cs"])
print(f"  [OK] embeddings MiniLM guardados | best_val={out['best_val']:.4f}")

ctx   = load_all()
meta5 = ctx["meta_aligned"][:, :5]
m = run_model(ctx, np.hstack([out["emb_std"], meta5]), "37_minilm",
              "CatBoost RS v3-MiniLM + Meta5",
              "RS v3 MiniLM-384 content (64d) + meta5 (5d)",
              notes_extra=f"Content tower con all-MiniLM-L6-v2 en vez de TF-IDF. "
                          f"best_val={out['best_val']:.4f}")

print("\n" + "=" * 70)
print(f"37_minilm: R2={m['r2_temporal']:.4f} | RMSE={m['rmse_temporal']:.2f} | "
      f"Spearman={m['spearman_temporal']:.4f}")
print("Comparar contra 34_05_v3 (TF-IDF) en experiment_results.json")
print("=" * 70)
