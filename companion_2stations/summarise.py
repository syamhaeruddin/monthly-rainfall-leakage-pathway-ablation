"""Summarise the two-station rerun: NSE per station (mean of four horizons) and panel DM tests.

Usage:  python summarise.py <rerun folder> <output json>
"""
import json
import sys

import numpy as np

R = json.load(open(f"{sys.argv[1]}/hasil/hasil_evaluasi.json"))   # path inside the companion project copy that was run
M = ["Klim-30th", "Rata-rata", "NNLS", "Model terbaik", "L-G3", "L-G4", "GG3", "GG4", "GG4-transfer"]   # keys of the companion version that was run
EN = {"Klim-30th": "Clim-30yr", "Rata-rata": "Mean", "Model terbaik": "Best model"}
en = lambda s: " vs ".join(EN.get(p, p) for p in s.split(" vs "))
out = {"nse_mean_of_horizons": {}, "panel_dm": {}}
for per in ("2015-2025", "2015-2020"):
    out["nse_mean_of_horizons"][per] = {st: {en(m): round(float(np.mean([d[h][m] for h in d])), 4) for m in M if m in d["1"]}
                                        for st, d in R["nse"][per].items()}
for k in ("GG3 vs L-G3", "GG4 vs L-G4", "GG3 vs Klim-30th", "GG3 vs NNLS", "GG3 vs Rata-rata", "GG4-transfer vs GG4"):
    out["panel_dm"][en(k)] = {kk: {"DM": round(v["DM"], 3), "p_holm": round(v["p_holm_horizon"], 4)}
                          for kk, v in R["panel"].get(k, {}).items() if not kk.startswith("2021")}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out["nse_mean_of_horizons"]["2015-2025"], indent=1))
