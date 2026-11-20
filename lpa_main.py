"""Stage 2: experiments of the MethodsX article (LPA protocol).

Contents:
  A. Reproduction of P0 (original notebook protocol), including the L1-open variant on the same test window.
  B. 2^3 factorial design of pathways L1 (meta-learner overlap), L2 (contemporaneous input) and
     L3 (full-series decomposition), five seeds, with Shapley values and interactions.
     Cell (L1,L2,L3) = (closed, closed, open) equals P1 of the notebook; the all-closed cell equals P2.
  C. P3: all pathways closed AND training features built stepwise (decomposition at every training origin).
  D. Fixed and rolling 30-year climatology; Diebold-Mariano tests.
Usage:  python lpa_main.py
"""
import itertools
import platform
import time
from math import factorial

import numpy as np
import pandas as pd

from lpa_common import (MT, N, OUT_DIR, SAR, SEEDS, STATIONS, T0, TR, bands, clim_fixed, clim_rolling30,
                        dm_test, grouping, load_station, log, lstm_model, metrics, mlp, save_json, seed,
                        tf_mod, windows, xgb)
import os

PATHS = ["L1", "L2", "L3"]


def prep(st):
    D = dict(np.load(os.path.join(OUT_DIR, f"dec_{st}.npz")))
    d = load_station(st)
    full = D["full"]; mapF = grouping(full); bF = bands(full, mapF)
    mapT = grouping(D["train"]); bT = bands(D["train"], mapT)
    dec = lambda t: D["full"] if t == N else D[f"o{t}"]
    cacheT = {t: bands(dec(t), mapT) for t in range(T0, N + 1)}       # bands from the decomposition of y[0..t-1]
    n_imf = np.array([dec(t).shape[0] for t in range(T0, N + 1)])
    extra = {"n_imf_train": int(D["train"].shape[0]), "n_imf_min": int(n_imf.min()), "n_imf_max": int(n_imf.max()),
             "share_origins_with_extra_imf": float(np.mean(n_imf > len(mapT))),
             "share_test_origins_with_extra_imf": float(np.mean(n_imf[TR - T0:N - T0] > len(mapT)))}
    return d, D, bF, bT, mapT, cacheT, extra


# ------------------------------------------------------------------ A. P0, original protocol (verbatim from the notebook)
def run_p0(d, bF):
    from sklearn.preprocessing import MinMaxScaler
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y, X = d["y"], d["X"]; act = y[TR:]
    p_lo = SARIMAX(bF["L"][:TR], exog=X[:TR], **SAR).fit(disp=False).forecast(N - TR, exog=X[TR:])
    mid = bF["M"]; Xm, ym = windows(mid[:TR], 7)
    p_mi = xgb().fit(Xm, ym).predict(windows(np.concatenate([mid[TR - 7:TR], mid[TR:]]), 7)[0])
    out = {k: [] for k in ["P0_240", "P0_72", "P0_72_L1open", "P0_sum72"]}; F = {"P0_72": [], "P0_72_L1open": []}
    for sd in SEEDS:
        seed(sd)
        hi = bF["H"]; sc = MinMaxScaler(); xtr = sc.fit_transform(hi[:TR, None]); xte = sc.transform(hi[TR:, None])
        m = lstm_model(1); m.fit(xtr[:, None, :], xtr, epochs=50, batch_size=16, verbose=0)
        p_hi = sc.inverse_transform(m.predict(xte[:, None, :], verbose=0)).ravel()
        Z = np.column_stack([p_lo, p_mi, p_hi, p_lo + p_mi + p_hi])
        fin = mlp(sd).fit(Z[:168], act[:168]).predict(Z)
        fin_open = mlp(sd).fit(Z, act).predict(Z)                 # meta-learner sees 2015-2020
        out["P0_240"].append(metrics(act, fin)); out["P0_72"].append(metrics(act[168:], fin[168:]))
        out["P0_72_L1open"].append(metrics(act[168:], fin_open[168:]))
        out["P0_sum72"].append(metrics(act[168:], Z[168:, 3]))
        F["P0_72"].append(fin[168:]); F["P0_72_L1open"].append(fin_open[168:])
    return out, {k: np.mean(v, 0) for k, v in F.items()}


# ------------------------------------------------------------------ B. factorial
def det_rows(d, btrain, hist, L2open, ex=None):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    if ex is None:
        ex = d["X"] if L2open else d["Xlag"]
    sm = SARIMAX(btrain["L"][1:TR], exog=ex[1:TR], **SAR).fit(disp=False)
    Xm, ym = windows(btrain["M"][:TR], 7); xg = xgb().fit(Xm, ym)
    rows = []
    for t in range(TR, N):
        b = hist(t)
        plo = float(sm.apply(b["L"][1:t], exog=ex[1:t]).forecast(1, exog=ex[t:t + 1])[0])
        pmi = float(xg.predict(b["M"][t - 7:t][None, :])[0])
        rows.append([plo, pmi])
    return np.array(rows)


def lstm_rows(btrain, hist, cont, L2open, sd):
    from sklearn.preprocessing import MinMaxScaler
    seed(sd)
    sc = MinMaxScaler().fit(btrain["H"][:TR, None]); hs = sc.transform(btrain["H"][:TR, None]).ravel()
    if L2open:   # 12-month window ending AT the target month (contains the value of month t)
        Xh = np.array([hs[i - 11:i + 1] for i in range(11, TR)]); yh = hs[11:TR]
        hw = np.array([sc.transform(cont(t)["H"][t - 11:t + 1, None]).ravel() for t in range(TR, N)])
    else:
        Xh, yh = windows(hs, 12)
        hw = np.array([sc.transform(hist(t)["H"][t - 12:t, None]).ravel() for t in range(TR, N)])
    lm = lstm_model(12); lm.fit(Xh[:, :, None], yh, epochs=50, batch_size=16, verbose=0)
    return sc.inverse_transform(lm.predict(hw[:, :, None], verbose=0)).ravel()


def stack(y, rows, phi, sd, L1open):
    Z = np.column_stack([rows, phi]); Z = np.column_stack([Z, Z.sum(1)])
    a = y[TR:]; n = N - TR if L1open else MT - TR
    fin = mlp(sd).fit(Z[:n], a[:n]).predict(Z[MT - TR:])
    return fin, Z[MT - TR:, 3]


def run_factorial(d, bF, bT, cacheT):
    y = d["y"]; a = y[MT:]
    res, fc = {}, {}
    for L2, L3 in itertools.product([0, 1], [0, 1]):
        t0 = time.time()
        if L3:
            btrain, hist, cont = bF, (lambda t: bF), (lambda t: bF)
        else:
            btrain, hist, cont = bT, (lambda t: cacheT[t]), (lambda t: cacheT[t + 1])
        rows = det_rows(d, btrain, hist, bool(L2))
        per = {0: [], 1: []}; per_sum = []; fpred = {0: [], 1: []}
        for sd in SEEDS:
            phi = lstm_rows(btrain, hist, cont, bool(L2), sd)
            for L1 in (0, 1):
                fin, s = stack(y, rows, phi, sd, bool(L1))
                per[L1].append(metrics(a, fin)); fpred[L1].append(fin)
            per_sum.append(metrics(a, s))
        for L1 in (0, 1):
            key = f"L1={L1},L2={L2},L3={L3}"
            res[key] = {"seeds": per[L1], "sum_no_mlp": per_sum}
            fc[key] = np.mean(fpred[L1], 0)
        log(f"  factorial L2={L2} L3={L3} done ({(time.time() - t0) / 60:.1f} min) | "
            f"NSE L1=0: {np.mean([m['NSE'] for m in per[0]]):.3f}, L1=1: {np.mean([m['NSE'] for m in per[1]]):.3f}")
    return res, fc


def shapley(v):
    """v: dict tuple(L1,L2,L3) -> value. Contribution of each pathway to v(all open) - v(all closed)."""
    k = 3; phi = {}
    for i in range(k):
        others = [j for j in range(k) if j != i]; tot = 0.0
        for r in range(k):
            for S in itertools.combinations(others, r):
                w = factorial(len(S)) * factorial(k - len(S) - 1) / factorial(k)
                on = tuple(1 if (j in S or j == i) else 0 for j in range(k))
                off = tuple(1 if j in S else 0 for j in range(k))
                tot += w * (v[on] - v[off])
        phi[PATHS[i]] = tot
    return phi


def effects(v):
    """Main effects and interactions of the 2^3 factorial design (coding -1 closed / +1 open)."""
    out = {}
    terms = {"L1": (0,), "L2": (1,), "L3": (2,), "L1xL2": (0, 1), "L1xL3": (0, 2), "L2xL3": (1, 2), "L1xL2xL3": (0, 1, 2)}
    for name, idx in terms.items():
        out[name] = sum(np.prod([2 * c[j] - 1 for j in idx]) * val for c, val in v.items()) / 4
    return out


# ------------------------------------------------------------------ C. P3
def release_lagged(d):
    """L4 closed: climate indices used only after release (Nino, DMI, PDO at t-2; SOI and SILO weather at t-1)."""
    from lpa_common import EXOG
    X = d["X"]; Xr = d["Xlag"].copy()
    X2 = np.vstack([X[:2], X[:-2]])
    for j, c in enumerate(EXOG):
        if c.startswith("NINO") or c in ("DMI", "PDO"):
            Xr[:, j] = X2[:, j]
    return Xr


def run_cell(d, btrain, hist, cont, ex):
    """One closed cell (L1, L2 closed) with a given exogenous matrix; five seeds."""
    y = d["y"]; a = y[MT:]
    rows = det_rows(d, btrain, hist, False, ex=ex); per, F = [], []
    for sd in SEEDS:
        phi = lstm_rows(btrain, hist, cont, False, sd)
        fin, _ = stack(y, rows, phi, sd, False); per.append(metrics(a, fin)); F.append(fin)
    return {"seeds": per}, np.mean(F, 0)


def run_p3(d, cacheT, ex=None):
    from sklearn.preprocessing import MinMaxScaler
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y = d["y"]; a = y[MT:]
    Xlag = d["Xlag"] if ex is None else ex
    # edge series: band value at month s from the decomposition of y[0..s] (past and month s only)
    e = {k: np.array([cacheT[s + 1][k][s] for s in range(T0, N)]) for k in "LMH"}
    E = lambda k, lo, hi: e[k][lo - T0:hi - T0]
    sm = SARIMAX(E("L", T0, TR), exog=Xlag[T0:TR], **SAR).fit(disp=False)
    tr = np.arange(T0, TR)
    xg = xgb().fit(np.array([cacheT[s]["M"][s - 7:s] for s in tr]), E("M", T0, TR))
    rows = np.array([[float(sm.apply(E("L", T0, t), exog=Xlag[T0:t]).forecast(1, exog=Xlag[t:t + 1])[0]),
                      float(xg.predict(cacheT[t]["M"][t - 7:t][None, :])[0])] for t in range(TR, N)])
    per, per_sum, F = [], [], []
    for sd in SEEDS:
        seed(sd)
        sc = MinMaxScaler().fit(E("H", T0, TR)[:, None])
        Xh = np.array([sc.transform(cacheT[s]["H"][s - 12:s, None]).ravel() for s in tr])
        yh = sc.transform(E("H", T0, TR)[:, None]).ravel()
        lm = lstm_model(12); lm.fit(Xh[:, :, None], yh, epochs=50, batch_size=16, verbose=0)
        hw = np.array([sc.transform(cacheT[t]["H"][t - 12:t, None]).ravel() for t in range(TR, N)])
        phi = sc.inverse_transform(lm.predict(hw[:, :, None], verbose=0)).ravel()
        fin, s = stack(y, rows, phi, sd, False)
        per.append(metrics(a, fin)); per_sum.append(metrics(a, s)); F.append(fin)
    return {"seeds": per, "sum_no_mlp": per_sum}, np.mean(F, 0)


def mean_sd(lst, m="NSE"):
    v = np.array([x[m] for x in lst]); return float(v.mean()), float(v.std(ddof=1))


if __name__ == "__main__":
    import sklearn, statsmodels, xgboost, scipy, PyEMD
    tf = tf_mod()
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__, "tensorflow": tf.__version__, "xgboost": xgboost.__version__,
                "statsmodels": statsmodels.__version__, "sklearn": sklearn.__version__,
                "PyEMD": getattr(PyEMD, "__version__", "?"), "platform": platform.platform(),
                "processor": platform.processor()}
    log(f"versions: {versions}")
    ALL = {"versions": versions}
    for st in STATIONS:
        T = time.time(); log(f"=== {st} ===")
        d, D, bF, bT, mapT, cacheT, extra = prep(st)
        y = d["y"]; a = y[MT:]
        clim = clim_fixed(y, d["month"]); clim30 = clim_rolling30(y)
        R = {"imf": extra, "clim_fixed": metrics(a, clim), "clim_rolling30": metrics(a, clim30)}
        log(f"  fixed climatology NSE {R['clim_fixed']['NSE']:.3f}, rolling 30-yr NSE {R['clim_rolling30']['NSE']:.3f}")
        R["P0"], FP0 = run_p0(d, bF)
        log("  P0: " + ", ".join(f"{k} {mean_sd(v)[0]:.3f}" for k, v in R["P0"].items()))
        R["factorial"], FF = run_factorial(d, bF, bT, cacheT)
        v = {tuple(int(x.split('=')[1]) for x in k.split(',')): mean_sd(r["seeds"])[0] for k, r in R["factorial"].items()}
        R["shapley"] = shapley(v); R["effects"] = effects(v)
        R["shapley_by_seed"] = []
        for i in range(len(SEEDS)):
            vi = {tuple(int(x.split('=')[1]) for x in k.split(',')): r["seeds"][i]["NSE"] for k, r in R["factorial"].items()}
            R["shapley_by_seed"].append(shapley(vi))
        log(f"  Shapley: {R['shapley']} | total {v[(1, 1, 1)] - v[(0, 0, 0)]:.3f}")
        R["P3"], FP3 = run_p3(d, cacheT)
        log(f"  P3 NSE {mean_sd(R['P3']['seeds'])[0]:.3f} (without MLP {mean_sd(R['P3']['sum_no_mlp'])[0]:.3f})")
        Xr = release_lagged(d)
        R["L4_closed"] = {}
        R["L4_closed"]["P2"], FP2r = run_cell(d, bT, lambda t: cacheT[t], lambda t: cacheT[t + 1], Xr)
        R["L4_closed"]["P3"], FP3r = run_p3(d, cacheT, ex=Xr)
        log(f"  L4 closed: P2 NSE {mean_sd(R['L4_closed']['P2']['seeds'])[0]:.3f}, P3 NSE {mean_sd(R['L4_closed']['P3']['seeds'])[0]:.3f}")
        FP2 = FF["L1=0,L2=0,L3=0"]
        R["DM"] = {}
        for name, f in {"P0_72": FP0["P0_72"], "P0_72_L1open": FP0["P0_72_L1open"], "P2": FP2, "P3": FP3}.items():
            R["DM"][name] = {"vs_clim_fixed": dm_test(a - f, a - clim), "vs_clim_rolling30": dm_test(a - f, a - clim30)}
        R["DM"]["P3_vs_P2"] = dm_test(a - FP3, a - FP2)
        np.savez(os.path.join(OUT_DIR, f"forecasts_lpa_{st}.npz"), actual=a, clim=clim, clim30=clim30,
                 P3=FP3, **{k.replace(",", "_").replace("=", ""): v_ for k, v_ in FF.items()}, **FP0)
        ALL[st] = R
        save_json(R, f"lpa_main_{st}.json")
        log(f"=== {st} done in {(time.time() - T) / 60:.1f} min ===")
    save_json(ALL, "lpa_main_all.json")

    # summary table
    rows = []
    for st in STATIONS:
        R = ALL[st]
        for k, lab in [("P0_240", "P0, scored 2001-2020"), ("P0_72_L1open", "P0, MLP trained 2001-2020, scored 2015-2020"),
                       ("P0_72", "P0-72"), ("P0_sum72", "P0-72 without MLP")]:
            m, s = mean_sd(R["P0"][k]); rows.append({"Station": st, "Configuration": lab, "NSE": m, "sd": s})
        for k, r in R["factorial"].items():
            m, s = mean_sd(r["seeds"]); rows.append({"Station": st, "Configuration": f"Factorial {k} (1=open)", "NSE": m, "sd": s})
        m, s = mean_sd(R["P3"]["seeds"]); rows.append({"Station": st, "Configuration": "P3", "NSE": m, "sd": s})
        m, s = mean_sd(R["P3"]["sum_no_mlp"]); rows.append({"Station": st, "Configuration": "P3 without MLP", "NSE": m, "sd": s})
        for k in ("P2", "P3"):
            m, s = mean_sd(R["L4_closed"][k]["seeds"]); rows.append({"Station": st, "Configuration": f"{k}, L4 closed", "NSE": m, "sd": s})
        rows.append({"Station": st, "Configuration": "Fixed climatology 1921-2014", "NSE": R["clim_fixed"]["NSE"]})
        rows.append({"Station": st, "Configuration": "Rolling 30-yr climatology", "NSE": R["clim_rolling30"]["NSE"]})
    tab = pd.DataFrame(rows)
    with pd.ExcelWriter(os.path.join(OUT_DIR, "lpa_summary.xlsx")) as w:
        tab.to_excel(w, sheet_name="NSE", index=False)
        pd.DataFrame({st: ALL[st]["shapley"] for st in STATIONS}).to_excel(w, sheet_name="Shapley")
        pd.DataFrame({st: ALL[st]["effects"] for st in STATIONS}).to_excel(w, sheet_name="Factorial effects")
        pd.DataFrame([{"Station": st, "Comparison": k, "Benchmark": kk, **vv} for st in STATIONS
                      for k, v in ALL[st]["DM"].items() for kk, vv in (v.items() if "DM" not in v else [("", v)])]).to_excel(w, sheet_name="DM", index=False)
    print(tab.round(3).to_string())
    log("DONE lpa_main")
