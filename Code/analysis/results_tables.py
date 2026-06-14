"""
results_tables.py
==================
Genera las tablas del Capítulo 3 (Metodología) y Capítulo 4 (Resultados) de la tesis.

Carga experiment_results.json y produce:
  TABLE 1 — Ablación (content-only baselines)
  TABLE 2 — Comparación principal RS vs contenido (con barras de error)
  TABLE 3 — Auditoría de robustez (preguntas metodológicas)
  TABLE 4 — Sweep de hiperparámetros (CS_DROP, neg_mode)

Salidas:
  - Tablas en consola (markdown)
  - Tablas en LaTeX (para copiar a la tesis)

Ejecutar:
  python Code/analysis/results_tables.py
"""

import json, os
import numpy as np
import pandas as pd

BASE    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(BASE, "Data", "experiment_results.json")

# ── Carga ─────────────────────────────────────────────────────────────────────
with open(RESULTS, encoding="utf-8") as f:
    raw = json.load(f)
data = {e["model_id"]: e for e in raw}


def get(model_id, metric, default=None):
    if model_id not in data:
        return default
    return data[model_id]["metrics"].get(metric, default)


def fmt(v, decimals=4, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if pct:
        return f"{v*100:.1f}%"
    return f"{v:.{decimals}f}"


def md_table(headers, rows, alignments=None):
    """Genera una tabla en formato markdown."""
    col_widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
                  for i, h in enumerate(headers)]
    def row_str(row):
        return "| " + " | ".join(str(r).ljust(w) for r, w in zip(row, col_widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    lines = [row_str(headers), sep] + [row_str(r) for r in rows]
    return "\n".join(lines)


def latex_table(headers, rows, caption="", label=""):
    """Genera una tabla en formato LaTeX (booktabs)."""
    ncols = len(headers)
    col_fmt = "l" + "r" * (ncols - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + col_fmt + r"}",
        r"\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\",
        r"\midrule",
    ]
    for i, row in enumerate(rows):
        # Separador antes de filas de "totales" o secciones
        if any(str(row[0]).startswith(marker) for marker in ["RS", "\\textit"]):
            lines.append(r"\midrule")
        lines.append(" & ".join(str(c) for c in row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1: Ablación — content-only baselines
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TABLE 1: Ablación — baselines de contenido puro")
print("=" * 70)

abl_ids = [
    ("abl_01", "Metadata Only",                    "meta (6d)"),
    ("abl_02", "TF-IDF + Numeric",                 "tfidf (100d) + steam (2d)"),
    ("abl_03", "TF-IDF + Numeric + RAWG",          "tfidf (100d) + steam (2d) + rawg (11d)"),
    ("abl_04", "Meta + RAWG",                      "meta (6d) + rawg (11d)"),
    ("abl_05", "Full content (sin RS)",            "tfidf + steam + meta + rawg (119d)"),
]

headers_abl = ["Modelo", "Features", "R² test", "Spearman", "R² TSCV", "R² KFold"]
rows_abl = []
for mid, name, feats in abl_ids:
    r2   = get(mid, "r2_temporal")
    rho  = get(mid, "spearman_temporal")
    tscv = get(mid, "r2_tscv")
    kf   = get(mid, "r2_kfold")
    rows_abl.append([name, feats, fmt(r2), fmt(rho), fmt(tscv), fmt(kf)])

print(md_table(headers_abl, rows_abl))

print("\nLaTeX:")
print(latex_table(
    ["Modelo", "Features", "$R^2$ test", "Spearman", "$R^2$ TSCV", "$R^2$ KFold"],
    rows_abl,
    caption="Ablación de features de contenido puro (sin embeddings RS). "
            "Evaluación en split temporal (train pre-2017, test 2017+).",
    label="tab:ablation",
))


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 2: Comparación principal RS vs contenido
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TABLE 2: Comparación principal — RS vs contenido (con barras de error)")
print("=" * 70)

# Referencia puntual (un seed)
# Barras de error (5 seeds de tuning)
comp_rows_data = [
    # (id,   nombre,                                 notas de std)
    ("39_content_summary", "Contenido puro (5 seeds)",  "r2_temporal_std"),
    ("34_05_v3",           "RS v3 seed-42 (referencia)", None),
    ("36_v3_summary",      "RS v3 (5 seeds tower)",      "r2_temporal_std"),
    ("39_rs_fixed_summary","RS v3 (5 seeds tuning)",     "r2_temporal_std"),
    ("24_05_cat_clean",    "RS v2 clean (referencia)",  None),
    ("38_v2exact_s42",     "RS v2 exact seed-42",        None),
    ("38_v2exact_s123",    "RS v2 exact seed-123",       None),
]

headers_comp = ["Modelo", "R² test (±σ)", "Spearman", "R² TSCV"]
rows_comp = []
for mid, name, std_key in comp_rows_data:
    r2   = get(mid, "r2_temporal")
    rho  = get(mid, "spearman_temporal")
    tscv = get(mid, "r2_tscv")
    std  = get(mid, std_key) if std_key else None

    r2_str = f"{fmt(r2)}" + (f" ±{fmt(std)}" if std else "")
    rows_comp.append([name, r2_str, fmt(rho), fmt(tscv)])

print(md_table(headers_comp, rows_comp))

# Ratio clave
r2_content_tscv = get("39_content_summary", "r2_tscv")
r2_rs_tscv      = get("39_rs_fixed_summary", "r2_tscv")
if r2_content_tscv and r2_rs_tscv:
    print(f"\n  Ratio TSCV (RS / contenido): {r2_rs_tscv/r2_content_tscv:.1f}x")

print("\nLaTeX:")
print(latex_table(
    ["Modelo", "$R^2$ test ($\\pm\\sigma$)", "Spearman", "$R^2$ TSCV"],
    rows_comp,
    caption="Comparación principal entre modelos de contenido puro y RS embeddings. "
            r"Los modelos con 5 seeds reportan media $\pm$ desviación estándar.",
    label="tab:main_comparison",
))


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 3: Auditoría de robustez
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TABLE 3: Auditoría de robustez")
print("=" * 70)

audit_rows = [
    # (pregunta, modelo_id, resultado_clave, veredicto)
    ("¿has_senti es leakage?",       "24_05_cat_clean",  "SHAP=0.000, rank 70/70",   "Inerte (safe)"),
    ("¿num_tags es leakage?",        "24_05_cat_no_nt",  "ΔR²=−0.035",               "Soft-leakage ≤0.04"),
    ("Tweedie vs RMSE",              "33_tweedie",       "Spearman=0.573",            "Mejor ranking"),
    ("Metascore (+2d Steam)",        "33_metascore",     "R²=0.422",                  "Mejora real"),
    ("Ensemble Cat+XGB (honesto)",   "33b_ens",          "R²=0.377",                  "Modesto (+0.037)"),
    ("CS_DROP sweep 0.20→0.40",      "35_cs35",          "R²=0.219±0.01",             "Plateau, no pico"),
    ("MiniLM-384 vs TF-IDF",        "37_minilm",        "R²=0.195",                  "Equivalente"),
    ("v2 original reproducible?",   "38_v2exact_s42",   "R²=0.3404 (exact match)",   "Sí, era suerte"),
    ("σ tower vs σ tuning",         "36_v3_summary",    "σ_tower=0.017, σ_tun=0.055","Tuning domina"),
]

headers_aud = ["Pregunta", "Experimento", "Resultado clave", "Veredicto"]
rows_aud = [(q, mid, res, verd) for q, mid, res, verd in audit_rows]
print(md_table(headers_aud, rows_aud))


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 4: Sweep de configuración del tower
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TABLE 4: Sweep de hiperparámetros del Two-Tower")
print("=" * 70)

sweep_ids = [
    ("25_cs05_aggressive_cat", "CS=0.5 (agresivo)", "v2"),
    ("25_cs0_collab_cat",      "CS=0.0 (puro colab.)", "v2"),
    ("25_cs10_pureContent_cat","CS=1.0 (puro cont.)", "v2"),
    ("25_emb128_cat",          "emb_dim=128", "v2"),
    ("35_cs20",                "CS=0.20", "v3"),
    ("35_cs25",                "CS=0.25", "v3"),
    ("34_05_v3",               "CS=0.30 (elegido)", "v3"),
    ("35_cs35",                "CS=0.35", "v3"),
    ("35_cs40",                "CS=0.40", "v3"),
    ("34_05_v3w",              "neg pop^0.75", "v3"),
]

headers_sweep = ["Config", "Tower", "R² test", "R² TSCV", "Spearman"]
rows_sweep = []
for mid, name, ver in sweep_ids:
    r2   = get(mid, "r2_temporal")
    tscv = get(mid, "r2_tscv")
    rho  = get(mid, "spearman_temporal")
    rows_sweep.append([name, ver, fmt(r2), fmt(tscv), fmt(rho)])

print(md_table(headers_sweep, rows_sweep))
print("\nLaTeX:")
print(latex_table(
    ["Configuración", "Tower", "$R^2$ test", "$R^2$ TSCV", "Spearman"],
    rows_sweep,
    caption="Sweep de configuración del Two-Tower. CS = cold-start dropout. "
            "v2 = setup original; v3 = setup con fixes metodológicos.",
    label="tab:tower_sweep",
))


# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN EJECUTIVO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESUMEN EJECUTIVO — Números clave para la tesis")
print("=" * 70)

content_tscv = get("39_content_summary", "r2_tscv")
rs_tscv      = get("39_rs_fixed_summary", "r2_tscv")
content_r2   = get("39_content_summary", "r2_temporal")
rs_r2        = get("39_rs_fixed_summary", "r2_temporal")
rs_r2_std    = get("39_rs_fixed_summary", "r2_temporal_std")
content_rho  = get("39_content_summary", "spearman_temporal")
rs_rho       = get("39_rs_fixed_summary", "spearman_temporal")

print(f"""
  Within-distribution (TSCV):
    Contenido-puro:  R²_TSCV = {fmt(content_tscv)}
    RS v3 (5 seeds): R²_TSCV = {fmt(rs_tscv)}
    Ratio:           {rs_tscv/content_tscv:.1f}x  ← argumento principal

  Temporal (train pre-2017, test 2017+):
    Contenido-puro:  R² = {fmt(content_r2)} (Spearman={fmt(content_rho)})
    RS v3 (5 seeds): R² = {fmt(rs_r2)} ± {fmt(rs_r2_std)} (Spearman={fmt(rs_rho)})

  Estabilidad:
    σ_tower (5 seeds)  = {fmt(get("36_v3_summary", "r2_temporal_std"))}   ← componente DL
    σ_tuning (5 seeds) = {fmt(get("39_rs_fixed_summary", "r2_temporal_std"))}   ← componente GB

  Resultado original (R²=0.407):
    seed=42 exacto: {fmt(get("38_v2exact_s42", "r2_temporal"))} (match 4 decimales)
    seed=123 mismo setup: {fmt(get("38_v2exact_s123", "r2_temporal"))}
    → Fue una realización afortunada del setup transductivo
""")
