"""Stage 3: null test on surrogate series (Validation 4 of the MethodsX article).

Surrogates: within each calendar month, rainfall values are shuffled across years. The seasonal cycle and the
marginal distribution of each month are kept, but the interannual signal and the link with climate predictors are lost.
An honest design should not beat climatology on these series.
  --fast  : P0 (original protocol) and P1 (L3 open) - one full-series decomposition per surrogate is enough.
  --step  : P2 and P3 (all pathways closed) - about 960 stepwise decompositions per surrogate.
Each surrogate is stored separately so that an interrupted run can be resumed.
Usage:  python lpa_surrogate.py --fast 100 --step 5 --n_jobs 14
"""
import argparse
import json
import os
import time

import numpy as np

from lpa_common import (MT, N, OUT_DIR, STATIONS, T0, TR, bands, ceemdan, clim_fixed, clim_rolling30,
                        decompose_series, dm_test, grouping, load_station, log, metrics)

SDIR = os.path.join(OUT_DIR, "surrogate")
os.makedirs(SDIR, exist_ok=True)


def surrogate(y, month, k):
    rng = np.random.default_rng(1000 + k)
    ys = y.copy()
    for m in range(1, 13):
        idx = np.where(month == m)[0]
        ys[idx] = y[rng.permutation(idx)]
    return ys


def summary_metrics(a, f, clim, clim30):
    return {"NSE": metrics(a, f)["NSE"], "DM_vs_clim": dm_test(a - f, a - clim), "DM_vs_clim30": dm_test(a - f, a - clim30)}


def fast_one(st, k, seeds):
    fn = os.path.join(SDIR, f"{st}_fast_{k:03d}.json")
    if os.path.exists(fn):
        return fn
    import lpa_common; lpa_common.SEEDS = seeds
    import lpa_main
    lpa_main.SEEDS = seeds
    d0 = load_station(st); ys = surrogate(d0["y"], d0["month"], k)
    d = load_station(st, y_override=ys); a = ys[MT:]
    clim, clim30 = clim_fixed(ys, d["month"]), clim_rolling30(ys)
    full = ceemdan(ys); bF = bands(full, grouping(full))
    p0, FP0 = lpa_main.run_p0(d, bF)
    rows = lpa_main.det_rows(d, bF, lambda t: bF, False)
    F1 = []
    for sd in seeds:
        phi = lpa_main.lstm_rows(bF, lambda t: bF, lambda t: bF, False, sd)
        F1.append(lpa_main.stack(ys, rows, phi, sd, False)[0])
    R = {"k": k, "clim_fixed": metrics(a, clim)["NSE"], "clim_rolling30": metrics(a, clim30)["NSE"],
         "P0_72": summary_metrics(a, FP0["P0_72"], clim, clim30),
         "P0_72_L1open": summary_metrics(a, FP0["P0_72_L1open"], clim, clim30),
         "P1": summary_metrics(a, np.mean(F1, 0), clim, clim30)}
    with open(fn, "w") as f:
        json.dump(R, f, indent=1)
    return fn


def step_one(st, k, n_jobs):
    fn = os.path.join(SDIR, f"{st}_step_{k:03d}.json")
    if os.path.exists(fn):
        return
    import lpa_main
    d0 = load_station(st); ys = surrogate(d0["y"], d0["month"], k)
    d = load_station(st, y_override=ys); a = ys[MT:]
    clim, clim30 = clim_fixed(ys, d["month"]), clim_rolling30(ys)
    D = decompose_series(ys, os.path.join(SDIR, f"dec_{st}_{k:03d}.npz"), n_jobs, label=f"surrogate {st} {k}")
    mapT = grouping(D["train"]); bT = bands(D["train"], mapT)
    dec = lambda t: D["full"] if t == N else D[f"o{t}"]
    cacheT = {t: bands(dec(t), mapT) for t in range(T0, N + 1)}
    _, FP2 = lpa_main.run_cell(d, bT, lambda t: cacheT[t], lambda t: cacheT[t + 1], d["Xlag"])
    _, FP3 = lpa_main.run_p3(d, cacheT)
    R = {"k": k, "clim_fixed": metrics(a, clim)["NSE"], "clim_rolling30": metrics(a, clim30)["NSE"],
         "P2": summary_metrics(a, FP2, clim, clim30), "P3": summary_metrics(a, FP3, clim, clim30)}
    with open(fn, "w") as f:
        json.dump(R, f, indent=1)
    log(f"surrogate {st} {k}: P2 {R['P2']['NSE']:.3f}, P3 {R['P3']['NSE']:.3f}, clim {R['clim_fixed']:.3f}", "surrogate.log")


def summarize():
    import glob
    out = {}
    for st in STATIONS:
        out[st] = {}
        for kind, cfgs in [("fast", ["P0_72", "P0_72_L1open", "P1"]), ("step", ["P2", "P3"])]:
            R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(SDIR, f"{st}_{kind}_*.json")))]
            if not R:
                continue
            for c in ["clim_fixed", "clim_rolling30"]:
                v = np.array([r[c] for r in R]); out[st][f"{kind}:{c}"] = {"n": len(v), "NSE_mean": float(v.mean()), "NSE_sd": float(v.std(ddof=1)) if len(v) > 1 else None}
            for c in cfgs:
                v = np.array([r[c]["NSE"] for r in R])
                sig = np.mean([(r[c]["DM_vs_clim"]["p"] < 0.05) and (r[c]["DM_vs_clim"]["DM"] < 0) for r in R])
                beat = np.mean([r[c]["NSE"] > r["clim_fixed"] for r in R])
                out[st][c] = {"n": len(v), "NSE_mean": float(v.mean()), "NSE_sd": float(v.std(ddof=1)) if len(v) > 1 else None,
                              "share_NSE_above_clim": float(beat), "share_signif_better_than_clim": float(sig)}
    with open(os.path.join(OUT_DIR, "lpa_surrogate_summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    log(f"surrogate summary: {json.dumps(out)}", "surrogate.log")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", type=int, default=100)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--n_jobs", type=int, default=14)
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    a = ap.parse_args()
    from joblib import Parallel, delayed
    for st in STATIONS:
        t0 = time.time()
        Parallel(n_jobs=a.n_jobs, batch_size=1)(delayed(fast_one)(st, k, a.seeds) for k in range(a.fast))
        log(f"fast surrogates {st}: {a.fast} done in {(time.time() - t0) / 60:.1f} min", "surrogate.log")
    summarize()
    import lpa_common, lpa_main
    lpa_common.SEEDS = a.seeds; lpa_main.SEEDS = a.seeds
    for k in range(a.step):
        for st in STATIONS:
            step_one(st, k, a.n_jobs)
        summarize()
    log("DONE lpa_surrogate", "surrogate.log")
