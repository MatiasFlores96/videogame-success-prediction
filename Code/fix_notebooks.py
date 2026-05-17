"""
fix_notebooks.py - Applies two correctness fixes to all regressor notebooks:

FIX 1 (TF-IDF leakage): Fit TF-IDF only on pre-2016 games, then transform all.
  Affected: 08, 12, 13

FIX 2 (Optuna sort): Sort X_train / y_train by release date before 80/20 split
  so Optuna uses the 80% oldest games to tune and validates on the 20% newest.
  Affected: 03, 04, 05, 06, 07, 08, 09, 10b, 11, 12, 13

FIX 3 (R² metric space in 13): Store back-transformed R² as r2_temporal so the
  leaderboard is comparable across all models.  r2_temporal_log added separately.
"""

import json, re, copy, sys

NB_DIR = "."  # run from Code/


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save(nb, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  Saved {path}")


def find_cell(nb, cell_id):
    for cell in nb["cells"]:
        if cell.get("id") == cell_id:
            return cell
    return None


def src(cell):
    return "".join(cell["source"])


def set_src(cell, new_source):
    cell["source"] = [line + "\n" for line in new_source.split("\n")]
    # Fix last line (remove trailing \n if original had none)
    if cell["source"] and cell["source"][-1] == "\n":
        cell["source"][-1] = ""


# ─── FIX 1: TF-IDF ──────────────────────────────────────────────────────────

TFIDF_OLD_08 = (
    "games_with_content = df_games[df_games['content_text'] != ''].copy()\n"
    "tfidf_matrix = tfidf.fit_transform(games_with_content['content_text'])"
)
TFIDF_NEW_08 = (
    "games_with_content = df_games[df_games['content_text'] != ''].copy()\n"
    "# Fit TF-IDF only on pre-2016 games to avoid leakage from test set\n"
    "_pre2016_content = games_with_content[\n"
    "    games_with_content['release_date_parsed'] < pd.to_datetime('2016-01-01')\n"
    "]['content_text']\n"
    "tfidf.fit(_pre2016_content)\n"
    "tfidf_matrix = tfidf.transform(games_with_content['content_text'])"
)

TFIDF_OLD_12 = (
    "tfidf_matrix = tfidf.fit_transform(games_w_content['content_text'])"
)
TFIDF_NEW_12 = (
    "# Fit TF-IDF only on pre-2016 games to avoid leakage from test set\n"
    "_pre2016_content = games_w_content[\n"
    "    games_w_content['release_date_parsed'] < CUTOFF\n"
    "]['content_text']\n"
    "tfidf.fit(_pre2016_content)\n"
    "tfidf_matrix = tfidf.transform(games_w_content['content_text'])"
)

TFIDF_OLD_13 = (
    "tfidf_matrix = tfidf.fit_transform(games_w_content['content_text'])"
)
TFIDF_NEW_13 = (
    "# Fit TF-IDF only on pre-2016 games to avoid leakage from test set\n"
    "_pre2016_content = games_w_content[\n"
    "    games_w_content['release_date_parsed'] < CUTOFF\n"
    "]['content_text']\n"
    "tfidf.fit(_pre2016_content)\n"
    "tfidf_matrix = tfidf.transform(games_w_content['content_text'])"
)

# ─── FIX 2: Optuna sort ─────────────────────────────────────────────────────
# Pattern used in 03, 04, 05, 07 (data_df / train_mask)
SORT_OLD_STD = (
    "_n_tr      = len(X_train_temporal)\n"
    "_split_idx = max(10, int(_n_tr * 0.8))\n"
    "_X_tr,  _X_val = X_train_temporal[:_split_idx], X_train_temporal[_split_idx:]\n"
    "_y_tr,  _y_val = y_train_temporal[:_split_idx], y_train_temporal[_split_idx:]"
)
SORT_NEW_DATA_DF = (
    "# Sort by release date so Optuna val set is always the most recent 20%\n"
    "_sort_order = np.argsort(data_df.loc[train_mask, 'release_date_parsed'].values)\n"
    "X_train_temporal = X_train_temporal[_sort_order]\n"
    "y_train_temporal = y_train_temporal[_sort_order]\n"
    "\n"
    "_n_tr      = len(X_train_temporal)\n"
    "_split_idx = max(10, int(_n_tr * 0.8))\n"
    "_X_tr,  _X_val = X_train_temporal[:_split_idx], X_train_temporal[_split_idx:]\n"
    "_y_tr,  _y_val = y_train_temporal[:_split_idx], y_train_temporal[_split_idx:]"
)

# 06: df_train instead of data_df
SORT_NEW_DF_TRAIN = (
    "# Sort by release date so Optuna val set is always the most recent 20%\n"
    "_sort_order = np.argsort(df_train.loc[train_mask, 'release_date_parsed'].values)\n"
    "X_train_temporal = X_train_temporal[_sort_order]\n"
    "y_train_temporal = y_train_temporal[_sort_order]\n"
    "\n"
    "_n_tr      = len(X_train_temporal)\n"
    "_split_idx = max(10, int(_n_tr * 0.8))\n"
    "_X_tr,  _X_val = X_train_temporal[:_split_idx], X_train_temporal[_split_idx:]\n"
    "_y_tr,  _y_val = y_train_temporal[:_split_idx], y_train_temporal[_split_idx:]"
)

# 08: valid_items_df instead of data_df
SORT_NEW_VALID_ITEMS_DF = (
    "# Sort by release date so Optuna val set is always the most recent 20%\n"
    "_sort_order = np.argsort(valid_items_df.loc[train_mask, 'release_date_parsed'].values)\n"
    "X_train_temporal = X_train_temporal[_sort_order]\n"
    "y_train_temporal = y_train_temporal[_sort_order]\n"
    "\n"
    "_n_tr      = len(X_train_temporal)\n"
    "_split_idx = max(10, int(_n_tr * 0.8))\n"
    "_X_tr,  _X_val = X_train_temporal[:_split_idx], X_train_temporal[_split_idx:]\n"
    "_y_tr,  _y_val = y_train_temporal[:_split_idx], y_train_temporal[_split_idx:]"
)

# 09: uses `dates` list with train_mask numpy bool
SORT_OLD_09 = (
    "_n_tr      = len(X_train_temporal)\n"
    "_split_idx = max(10, int(_n_tr * 0.8))\n"
    "_X_tr,  _X_val = X_train_temporal[:_split_idx], X_train_temporal[_split_idx:]\n"
    "_y_tr,  _y_val = y_train_temporal[:_split_idx], y_train_temporal[_split_idx:]"
)
SORT_NEW_09 = (
    "# Sort by release date so Optuna val set is always the most recent 20%\n"
    "_train_dates = np.array([d for d, m in zip(dates, train_mask) if m])\n"
    "_sort_order = np.argsort(_train_dates)\n"
    "X_train_temporal = X_train_temporal[_sort_order]\n"
    "y_train_temporal = y_train_temporal[_sort_order]\n"
    "\n"
    "_n_tr      = len(X_train_temporal)\n"
    "_split_idx = max(10, int(_n_tr * 0.8))\n"
    "_X_tr,  _X_val = X_train_temporal[:_split_idx], X_train_temporal[_split_idx:]\n"
    "_y_tr,  _y_val = y_train_temporal[:_split_idx], y_train_temporal[_split_idx:]"
)

# 10b: uses _X_tr_raw / _y_tr_raw / item_dates
SORT_OLD_10B = (
    "_X_tr, _X_val = _X_tr_raw[:_vsp], _X_tr_raw[_vsp:]\n"
    "_y_tr, _y_val = _y_tr_raw[:_vsp], _y_tr_raw[_vsp:]"
)
SORT_NEW_10B = (
    "# Sort by release date so Optuna val set is always the most recent 20%\n"
    "_train_dates = np.array(item_dates)[train_mask]\n"
    "_sort_order = np.argsort(_train_dates)\n"
    "_X_tr_raw = _X_tr_raw[_sort_order]\n"
    "_y_tr_raw = _y_tr_raw[_sort_order]\n"
    "\n"
    "_X_tr, _X_val = _X_tr_raw[:_vsp], _X_tr_raw[_vsp:]\n"
    "_y_tr, _y_val = _y_tr_raw[:_vsp], _y_tr_raw[_vsp:]"
)

# 11: runner function — fix inside run_experiment
SORT_OLD_11 = (
    "    n_tr = len(X_tr)\n"
    "    split_idx = max(10, int(n_tr * 0.8))\n"
    "    X_opt, X_val = X_tr[:split_idx], X_tr[split_idx:]\n"
    "    y_opt, y_val = y_tr[:split_idx], y_tr[split_idx:]"
)
SORT_NEW_11 = (
    "    # Sort by release date so Optuna val set is always the most recent 20%\n"
    "    _sort_order = np.argsort(data_df.loc[train_mask, 'release_date_parsed'].values)\n"
    "    X_tr = X_tr[_sort_order]\n"
    "    y_tr = y_tr[_sort_order]\n"
    "\n"
    "    n_tr = len(X_tr)\n"
    "    split_idx = max(10, int(n_tr * 0.8))\n"
    "    X_opt, X_val = X_tr[:split_idx], X_tr[split_idx:]\n"
    "    y_opt, y_val = y_tr[:split_idx], y_tr[split_idx:]"
)

# 12: X_tr / y_tr + s2_train
SORT_OLD_12 = (
    "# Optuna tuning\n"
    "sp = max(10, int(len(X_tr)*0.8))\n"
    "X_opt, X_val = X_tr[:sp], X_tr[sp:]\n"
    "y_opt, y_val = y_tr[:sp], y_tr[sp:]"
)
SORT_NEW_12 = (
    "# Sort by release date so Optuna val set is always the most recent 20%\n"
    "_sort_order = np.argsort(data_df.loc[s2_train, 'release_date_parsed'].values)\n"
    "X_tr = X_tr[_sort_order]\n"
    "y_tr = y_tr[_sort_order]\n"
    "\n"
    "# Optuna tuning\n"
    "sp = max(10, int(len(X_tr)*0.8))\n"
    "X_opt, X_val = X_tr[:sp], X_tr[sp:]\n"
    "y_opt, y_val = y_tr[:sp], y_tr[sp:]"
)

# 13: runner function — sort X_tr using item_dates_arr passed in
SORT_OLD_13 = (
    "    sp = max(10, int(len(X_tr)*0.8))\n"
    "    X_opt, X_val = X_tr[:sp], X_tr[sp:]\n"
    "    y_opt, y_val = y_tr[:sp], y_tr[sp:]"
)
SORT_NEW_13 = (
    "    # Sort by release date so Optuna val set is always the most recent 20%\n"
    "    _train_dates = item_dates_arr[tr_mask]\n"
    "    _sort_order = np.argsort(_train_dates)\n"
    "    X_tr = X_tr[_sort_order]\n"
    "    y_tr = y_tr[_sort_order]\n"
    "\n"
    "    sp = max(10, int(len(X_tr)*0.8))\n"
    "    X_opt, X_val = X_tr[:sp], X_tr[sp:]\n"
    "    y_opt, y_val = y_tr[:sp], y_tr[sp:]"
)

# ─── FIX 3: R² metric space in notebook 13 ──────────────────────────────────
# In runner-13, change save_result so r2_temporal = r2_raw (back-transformed)
# and add r2_temporal_log separately
R2_OLD_13 = (
    "        'r2_temporal':   r2_log,"
)
R2_NEW_13 = (
    "        'r2_temporal':   r2_raw,   # back-transformed (comparable with linear models)\n"
    "        'r2_temporal_log': r2_log, # in log-space (for reference)"
)


def apply_fix(nb, cell_id, old_str, new_str, label):
    cell = find_cell(nb, cell_id)
    if cell is None:
        print(f"  WARNING: cell {cell_id} not found")
        return False
    s = src(cell)
    if old_str not in s:
        print(f"  WARNING: pattern not found in {cell_id} for {label}")
        return False
    new_s = s.replace(old_str, new_str, 1)
    set_src(cell, new_s)
    print(f"  Fixed: {label}")
    return True


# ============================================================
print("=== Fixing notebook 03 ===")
nb = load("03_regressor_embeddings_only.ipynb")
apply_fix(nb, "5749ab01", SORT_OLD_STD, SORT_NEW_DATA_DF, "Optuna sort (data_df)")
save(nb, "03_regressor_embeddings_only.ipynb")

print("\n=== Fixing notebook 04 ===")
nb = load("04_regressor_metadata_only.ipynb")
apply_fix(nb, "05b295ca", SORT_OLD_STD, SORT_NEW_DATA_DF, "Optuna sort (data_df)")
save(nb, "04_regressor_metadata_only.ipynb")

print("\n=== Fixing notebook 05 ===")
nb = load("05_regressor_embeddings_plus_metadata.ipynb")
apply_fix(nb, "1ed8d4f3", SORT_OLD_STD, SORT_NEW_DATA_DF, "Optuna sort (data_df)")
save(nb, "05_regressor_embeddings_plus_metadata.ipynb")

print("\n=== Fixing notebook 06 ===")
nb = load("06_regressor_review_embeddings.ipynb")
apply_fix(nb, "409a18cb", SORT_OLD_STD, SORT_NEW_DF_TRAIN, "Optuna sort (df_train)")
save(nb, "06_regressor_review_embeddings.ipynb")

print("\n=== Fixing notebook 07 ===")
nb = load("07_regressor_tag_embeddings.ipynb")
apply_fix(nb, "9cdb2316", SORT_OLD_STD, SORT_NEW_DATA_DF, "Optuna sort (data_df)")
save(nb, "07_regressor_tag_embeddings.ipynb")

print("\n=== Fixing notebook 08 ===")
nb = load("08_hybrid_collaborative_content.ipynb")
apply_fix(nb, "cfad60d2", TFIDF_OLD_08, TFIDF_NEW_08, "TF-IDF pre-2016 only")
apply_fix(nb, "87527df9", SORT_OLD_STD, SORT_NEW_VALID_ITEMS_DF, "Optuna sort (valid_items_df)")
save(nb, "08_hybrid_collaborative_content.ipynb")

print("\n=== Fixing notebook 09 ===")
nb = load("09_regressor_rs_plus_reviews.ipynb")
apply_fix(nb, "a8e17cf1", SORT_OLD_09, SORT_NEW_09, "Optuna sort (dates list)")
save(nb, "09_regressor_rs_plus_reviews.ipynb")

print("\n=== Fixing notebook 10b ===")
nb = load("10b_model_enriched.ipynb")
apply_fix(nb, "cb10-06", SORT_OLD_10B, SORT_NEW_10B, "Optuna sort (item_dates)")
save(nb, "10b_model_enriched.ipynb")

print("\n=== Fixing notebook 11 ===")
nb = load("11_developer_reputation.ipynb")
apply_fix(nb, "runner-fn", SORT_OLD_11, SORT_NEW_11, "Optuna sort in run_experiment (data_df)")
save(nb, "11_developer_reputation.ipynb")

print("\n=== Fixing notebook 12 ===")
nb = load("12_two_stage_model.ipynb")
apply_fix(nb, "features-12", TFIDF_OLD_12, TFIDF_NEW_12, "TF-IDF pre-2016 only")
apply_fix(nb, "train-12", SORT_OLD_12, SORT_NEW_12, "Optuna sort (s2_train)")
save(nb, "12_two_stage_model.ipynb")

print("\n=== Fixing notebook 13 ===")
nb = load("13_log_transform.ipynb")
apply_fix(nb, "load-features-13", TFIDF_OLD_13, TFIDF_NEW_13, "TF-IDF pre-2016 only")
apply_fix(nb, "runner-13", SORT_OLD_13, SORT_NEW_13, "Optuna sort (item_dates_arr)")
apply_fix(nb, "runner-13", R2_OLD_13, R2_NEW_13, "R² metric: store r2_raw as r2_temporal")
save(nb, "13_log_transform.ipynb")

print("\nAll fixes applied.")
