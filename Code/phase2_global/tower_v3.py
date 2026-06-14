"""
tower_v3.py — Two-Tower content-augmented v3: módulo de entrenamiento parametrizado.

Misma arquitectura e hiperparámetros que 23_train_rs_content_v2.py, con tres
fixes metodológicos:

  FIX 1 — TF-IDF del tower fiteado SOLO en items pre-2016 (en 23_ se fiteaba
          sobre los 15,380 items, inconsistente con el resto del pipeline).
  FIX 2 — val_loss CON negativos sampleados (en 23_ era BCE solo de positivos:
          criterio de checkpoint degenerado — premiaba inflar todos los logits).
  FIX 3 — Negative sampling restringido a items pre-2016 (en 23_ los items de
          test aparecían como negativos; sin leakage, pero más prolijo).
          Opcional: sampling ponderado por frecuencia^0.75 (word2vec-style).

Además, la normalización z-score de las columnas numéricas del content matrix
usa estadísticos de items pre-2016 (en 23_ usaba todos los items).

Usado por: 34_ (v3 baseline + weighted), 35_ (sweep fino CS_DROP),
           36_ (estabilidad multi-seed), 37_ (sentence embeddings).
"""

import ast, json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

_HERE  = os.path.dirname(os.path.abspath(__file__))
BASE   = os.path.dirname(os.path.dirname(_HERE))
DATA   = os.path.join(BASE, "Data")
GAMES  = os.path.join(DATA, "steam_games.json")
CUTOFF = pd.Timestamp("2016-01-01")   # cutoff del TOWER (gap de 1 año vs XGB)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_cache = {}


def _parse_list_col(x):
    if x is None or (isinstance(x, float) and pd.isna(x)): return []
    if isinstance(x, list): return [str(t).strip() for t in x if t]
    try:
        lst = eval(x)
        return [str(t).strip() for t in lst if t] if isinstance(lst, list) else []
    except Exception:
        return []


def _clean_price(p):
    if pd.isna(p) or str(p).strip() in ("Free", "", "Free to Play"): return 0.0
    try: return float(str(p).replace("$", "").replace(",", "").strip())
    except Exception: return 0.0


def get_base_data(verbose=True):
    """Carga (con cache de módulo) mappings, df_games, fechas, interacciones
    pre-2016 y el set de item_idx pre-2016."""
    if "base" in _cache:
        return _cache["base"]

    if verbose: print("[tower_v3] Cargando datos base...")
    with open(os.path.join(DATA, "user2idx_v2.json")) as f:
        user2idx = {k: int(v) for k, v in json.load(f).items()}
    with open(os.path.join(DATA, "item2idx_v2.json")) as f:
        item2idx = {k: int(v) for k, v in json.load(f).items()}
    num_users, num_items = len(user2idx), len(item2idx)

    games_raw = []
    with open(GAMES, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: games_raw.append(ast.literal_eval(line))
            except Exception: pass

    df_games = pd.json_normalize(games_raw).rename(columns={"id": "item_id"})
    df_games["item_id"]  = df_games["item_id"].astype(str)
    df_games["item_idx"] = df_games["item_id"].map(item2idx)
    df_games = df_games.dropna(subset=["item_idx"])
    df_games["item_idx"] = df_games["item_idx"].astype(int)
    df_games = df_games.drop_duplicates(subset=["item_idx"], keep="first")
    # Parseo POR ELEMENTO (igual que 23_): el parseo column-wise de pandas 2.x
    # infiere un solo formato y coercea el resto a NaT (2,669 vs 6,953 items pre-2016)
    df_games["release_dt"] = df_games["release_date"].apply(
        lambda s: pd.to_datetime(s, dayfirst=True, errors="coerce") if s else pd.NaT)

    pre2016_idxs = set(df_games[df_games["release_dt"].notna() &
                                (df_games["release_dt"] < CUTOFF)]["item_idx"])

    df_int = pd.read_parquet(os.path.join(DATA, "interactions_v2.parquet"))
    df_int = df_int[df_int["item_idx"].isin(pre2016_idxs)].reset_index(drop=True)
    if verbose:
        print(f"[tower_v3] {len(df_int):,} interacciones pre-2016 "
              f"({df_int['item_idx'].nunique():,} items)")

    _cache["base"] = dict(user2idx=user2idx, item2idx=item2idx,
                          num_users=num_users, num_items=num_items,
                          df_games=df_games, df_int=df_int,
                          pre2016_idxs=pre2016_idxs)
    return _cache["base"]


def build_content_matrix(verbose=True, fit_scope="pre2016"):
    """Content matrix (N, 108) = TF-IDF(100) + numeric(4) + RAWG quality(4).

    fit_scope: 'pre2016' (FIX 1: TF-IDF y z-score con items pre-2016) o
               'all' (replica EXACTA de 23_: fit transductivo sobre los 15,380
               items, incluidos los de test — usado por 38_ para diagnosticar
               el gap v2-original vs v2rep).
    """
    cache_key = f"content_{fit_scope}"
    if cache_key in _cache:
        return _cache[cache_key]

    base = get_base_data(verbose)
    df_games  = base["df_games"]
    item2idx  = base["item2idx"]
    num_items = base["num_items"]
    pre2016   = base["pre2016_idxs"]

    df = df_games.copy()
    df["tag_text"]   = df["tags"].apply(_parse_list_col).apply(" ".join)
    df["genre_text"] = df["genres"].apply(_parse_list_col).apply(" ".join)
    df["content"]    = df["tag_text"] + " " + df["genre_text"]
    df["price_n"]    = df["price"].apply(_clean_price)
    df["ea_flag"]    = df["early_access"].fillna(False).astype(float)
    df["num_tags"]   = df["tags"].apply(_parse_list_col).apply(len).astype(float)
    df["num_genres"] = df["genres"].apply(_parse_list_col).apply(len).astype(float)

    # FIX 1: TF-IDF fiteado solo en items pre-2016 (o todos si fit_scope='all')
    valid = df[df["content"].str.strip() != ""]
    fit_corpus = valid[valid["item_idx"].isin(pre2016)]["content"] \
        if fit_scope == "pre2016" else valid["content"]
    tfidf = TfidfVectorizer(max_features=100, min_df=2, max_df=0.6, ngram_range=(1, 1))
    tfidf.fit(fit_corpus)
    if verbose:
        print(f"[tower_v3] TF-IDF fit en {len(fit_corpus):,} items (scope={fit_scope})")

    # RAWG quality (idéntico a 23_)
    rawg_quality = np.zeros((num_items, 4), dtype=np.float32)
    rawg_path, v1map_path = os.path.join(DATA, "rawg_enriched.csv"), os.path.join(DATA, "item2idx.json")
    if os.path.exists(rawg_path) and os.path.exists(v1map_path):
        with open(v1map_path) as f:
            item2idx_v1 = {k: int(v) for k, v in json.load(f).items()}
        v1_to_appid = {v: k for k, v in item2idx_v1.items()}
        df_rawg = pd.read_csv(rawg_path).drop_duplicates(subset=["item_idx"], keep="last")
        df_rawg["appid_str"]   = df_rawg["item_idx"].map(v1_to_appid)
        df_rawg["item_idx_v2"] = df_rawg["appid_str"].map(item2idx)
        df_rawg = df_rawg.dropna(subset=["item_idx_v2"])
        df_rawg["item_idx_v2"] = df_rawg["item_idx_v2"].astype(int)
        df_rawg = df_rawg.set_index("item_idx_v2")
        for idx in range(num_items):
            if idx not in df_rawg.index: continue
            r = df_rawg.loc[idx]
            rawg_quality[idx] = [
                float(r["rawg_rating"]) if pd.notna(r.get("rawg_rating")) else 0.0,
                float(r["metacritic"])  if pd.notna(r.get("metacritic"))  else 0.0,
                np.log1p(float(r["playtime_avg_h"])) if pd.notna(r.get("playtime_avg_h")) else 0.0,
                float(len(_parse_list_col(r.get("platforms", [])))),
            ]

    CONTENT_DIM = 108
    content = np.zeros((num_items, CONTENT_DIM), dtype=np.float32)
    item_to_row = {int(r["item_idx"]): i for i, (_, r) in enumerate(df.iterrows())}
    tfidf_mat = tfidf.transform(df["content"].fillna("")).toarray().astype(np.float32)
    for idx in range(num_items):
        if idx not in item_to_row: continue
        ri  = item_to_row[idx]
        row = df.iloc[ri]
        content[idx, :100] = tfidf_mat[ri]
        content[idx, 100]  = float(row["price_n"]    or 0)
        content[idx, 101]  = float(row["ea_flag"]    or 0)
        content[idx, 102]  = float(row["num_tags"]   or 0)
        content[idx, 103]  = float(row["num_genres"] or 0)
        content[idx, 104:] = rawg_quality[idx]

    # Z-score con estadísticos según fit_scope
    stat_rows = np.array(sorted(pre2016)) if fit_scope == "pre2016" \
        else np.arange(num_items)
    for ci in range(100, CONTENT_DIM):
        mu, sigma = content[stat_rows, ci].mean(), content[stat_rows, ci].std()
        if sigma > 1e-8:
            content[:, ci] = (content[:, ci] - mu) / sigma

    _cache[cache_key] = content
    return content


# ── Arquitectura (idéntica a 23_) ─────────────────────────────────────────────

class ContentItemTower(nn.Module):
    def __init__(self, num_items, content_dim, emb_dim=64, cs_drop=0.3):
        super().__init__()
        self.cs_drop = cs_drop
        half = emb_dim // 2
        self.id_emb = nn.Embedding(num_items, half)
        nn.init.normal_(self.id_emb.weight, std=0.01)
        self.content_net = nn.Sequential(
            nn.Linear(content_dim, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, half),
        )
        self.proj = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.LayerNorm(emb_dim))

    def forward(self, item_ids, content_feats):
        id_vec      = self.id_emb(item_ids)
        content_vec = self.content_net(content_feats)
        if self.training and self.cs_drop > 0:
            mask = (torch.rand(id_vec.shape[0], 1, device=id_vec.device) > self.cs_drop).float()
            id_vec = id_vec * mask
        return self.proj(torch.cat([id_vec, content_vec], dim=1))

    def cold_start(self, content_feats):
        content_vec = self.content_net(content_feats)
        return self.proj(torch.cat([torch.zeros_like(content_vec), content_vec], dim=1))


class ContentTwoTower(nn.Module):
    def __init__(self, num_users, num_items, content_dim, emb_dim=64, cs_drop=0.3):
        super().__init__()
        self.user_emb   = nn.Embedding(num_users, emb_dim)
        self.item_tower = ContentItemTower(num_items, content_dim, emb_dim, cs_drop)
        nn.init.normal_(self.user_emb.weight, std=0.01)

    def forward(self, user_ids, item_ids, content_feats):
        u = self.user_emb(user_ids)
        v = self.item_tower(item_ids, content_feats)
        return (u * v).sum(dim=1)


class _PairDataset(Dataset):
    def __init__(self, u, i):
        self.u = torch.from_numpy(u); self.i = torch.from_numpy(i)
    def __len__(self): return len(self.u)
    def __getitem__(self, idx): return self.u[idx], self.i[idx]


def train_tower(cs_drop=0.3, seed=42, neg_mode="uniform_pre2016",
                val_mode="with_negs", content_matrix=None, emb_dim=64,
                epochs=15, patience=4, lr=5e-4, batch=8192, verbose=True):
    """Entrena el Two-Tower v3 y devuelve dict(emb_std, emb_cs, best_val, epochs_run).

    neg_mode: 'uniform_pre2016' (FIX 3), 'pop075' (frecuencia^0.75), 'uniform_all' (legacy 23_)
    val_mode: 'with_negs' (FIX 2) o 'legacy' (BCE solo positivos, replica el
              criterio de checkpoint de 23_ — selecciona embeddings que codifican
              popularidad global; ver hallazgo de 34_)
    content_matrix: override (N, D) para experimentos con otros encoders (37_).
    """
    base = get_base_data(verbose)
    num_users, num_items = base["num_users"], base["num_items"]
    df_int  = base["df_int"]
    pre_arr = np.array(sorted(base["pre2016_idxs"]), dtype=np.int64)

    if content_matrix is None:
        content_matrix = build_content_matrix(verbose)
    content_dim = content_matrix.shape[1]

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed); np.random.seed(seed)

    users = df_int["user_idx"].values.astype(np.int64)
    items = df_int["item_idx"].values.astype(np.int64)
    idx_tr, idx_va = train_test_split(np.arange(len(users)), test_size=0.10, random_state=42)
    u_tr, i_tr = users[idx_tr], items[idx_tr]
    u_va, i_va = users[idx_va], items[idx_va]

    train_loader = DataLoader(_PairDataset(u_tr, i_tr), batch_size=batch,
                              shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(_PairDataset(u_va, i_va), batch_size=batch * 2,
                              shuffle=False, num_workers=0, pin_memory=True)

    # FIX 3: pool de negativos = items pre-2016
    pre_t = torch.from_numpy(pre_arr).to(DEVICE)
    if neg_mode == "pop075":
        freq = df_int["item_idx"].value_counts().reindex(pre_arr, fill_value=0).values.astype(np.float64)
        probs = torch.from_numpy((freq ** 0.75) / (freq ** 0.75).sum()).float().to(DEVICE)
    def sample_neg(n, gen=None):
        if neg_mode == "uniform_all":
            return torch.randint(0, num_items, (n,), device=DEVICE, generator=gen)
        if neg_mode == "pop075":
            return pre_t[torch.multinomial(probs, n, replacement=True, generator=gen)]
        return pre_t[torch.randint(0, len(pre_t), (n,), device=DEVICE, generator=gen)]

    # FIX 2: negativos FIJOS para validación (mismos en todas las épocas)
    gen_val = torch.Generator(device=DEVICE); gen_val.manual_seed(seed + 1)
    val_negs = sample_neg(len(u_va), gen_val).cpu()

    model = ContentTwoTower(num_users, num_items, content_dim, emb_dim, cs_drop).to(DEVICE)
    content_t = torch.from_numpy(content_matrix).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    warmup = 2 * len(train_loader)
    total  = epochs * len(train_loader)
    def lr_lambda(step):
        if step < warmup: return step / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + np.cos(np.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if verbose:
        print(f"[tower_v3] cs_drop={cs_drop} seed={seed} neg={neg_mode} "
              f"content_dim={content_dim} | train={len(u_tr):,} val={len(u_va):,}")

    best_val, patience_cnt, best_state, epochs_run = float("inf"), 0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for u_b, i_b in train_loader:
            u_b, i_b = u_b.to(DEVICE), i_b.to(DEVICE)
            neg = sample_neg(len(u_b))
            users_all = torch.cat([u_b, u_b])
            items_all = torch.cat([i_b, neg])
            labels    = torch.cat([torch.ones(len(u_b), device=DEVICE),
                                   torch.zeros(len(u_b), device=DEVICE)])
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                loss = criterion(model(users_all, items_all, content_t[items_all]), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update(); scheduler.step()
            tot += loss.item()
        avg_train = tot / len(train_loader)

        # FIX 2: validación con positivos + negativos fijos
        # (val_mode='legacy' replica 23_: BCE solo positivos → selecciona
        #  checkpoints con embeddings que codifican popularidad global)
        model.eval()
        val_loss, off = 0.0, 0
        with torch.no_grad():
            for u_b, i_b in valid_loader:
                nb = len(u_b)
                u_b, i_b = u_b.to(DEVICE), i_b.to(DEVICE)
                if val_mode == "legacy":
                    users_all, items_all = u_b, i_b
                    labels = torch.ones(nb, device=DEVICE)
                else:
                    n_b = val_negs[off:off + nb].to(DEVICE); off += nb
                    users_all = torch.cat([u_b, u_b])
                    items_all = torch.cat([i_b, n_b])
                    labels    = torch.cat([torch.ones(nb, device=DEVICE),
                                           torch.zeros(nb, device=DEVICE)])
                with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                    val_loss += criterion(model(users_all, items_all, content_t[items_all]),
                                          labels).item()
        avg_val = val_loss / len(valid_loader)
        epochs_run = epoch
        if verbose:
            print(f"  Epoch {epoch:02d}/{epochs} | train={avg_train:.4f} | "
                  f"val={avg_val:.4f} | {time.time()-t0:.1f}s")

        if avg_val < best_val:
            best_val, patience_cnt = avg_val, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                if verbose: print(f"  Early stopping en epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval().cpu()
    content_cpu = torch.from_numpy(content_matrix)

    emb_std, emb_cs = [], []
    all_ids = torch.arange(num_items, dtype=torch.long)
    with torch.no_grad():
        for s in range(0, num_items, 2048):
            e = min(s + 2048, num_items)
            emb_std.append(model.item_tower(all_ids[s:e], content_cpu[s:e]).numpy())
            emb_cs.append(model.item_tower.cold_start(content_cpu[s:e]).numpy())
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return dict(emb_std=np.vstack(emb_std).astype(np.float32),
                emb_cs=np.vstack(emb_cs).astype(np.float32),
                best_val=best_val, epochs_run=epochs_run)
