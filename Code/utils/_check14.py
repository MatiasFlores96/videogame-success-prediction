import json
with open("C:/Users/matia/Projects/VG_Recommender/Data/experiment_results.json") as f:
    results = json.load(f)
found = [r for r in results if r["model_id"].startswith("14")]
if not found:
    print("No se encontraron resultados 14x — el script no terminó")
for r in found:
    m = r.get("metrics", {})
    print(r["model_id"], "|", r["model_name"])
    print("  R2=", round(m.get("r2_temporal", float("nan")), 4),
          " RMSE=", round(m.get("rmse_temporal", float("nan")), 2),
          " MAE=", round(m.get("mae_temporal", float("nan")), 2))
