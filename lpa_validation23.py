"""Validation 2 (edge effect of CEEMDAN, VMD and causal a-trous wavelet) and Validation 3 (multi-horizon raw and
hybrid base models, combiners and the adaptive LSTM gate) of the MethodsX article.

The model code is copied without change from Part B of the original notebook (Naskah_IJIES_Final.ipynb, cells
33-38), so that the numbers in the article can be regenerated from this repository. Two additions:

* --l4 {open,closed}. "open" reproduces the notebook: at origin t every exogenous predictor enters with its value
  of month t-1. "closed" uses release-lagged predictors as in lpa_main.py (Nino, DMI and PDO at t-2; SOI and SILO
  weather at t-1), so that all pathways L1-L4 are closed.
* The VMD decomposition at every origin is computed once in parallel and cached (identical to the notebook loop).

Requires the CEEMDAN cache results/dec_<station>.npz from lpa_decompose.py.
Usage:
    python lpa_validation23.py vmd --n_jobs 14                 # VMD cache, both stations
    python lpa_validation23.py run --station Darwin --l4 open  # base models + combiners for one station
    python lpa_validation23.py summary                         # numbers quoted in the article
Outputs: results/validation23/
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import lpa_common as C
from lpa_common import EXOG, MT, N, OUT_DIR, SEEDS, STATIONS, T0, TR, dm_test, grouping, load_station, metrics, seed

HORIZONS = [1, 3, 6, 12]
VDIR = os.path.join(OUT_DIR, "validation23"); os.makedirs(VDIR, exist_ok=True)


# ------------------------------------------------------------------ decompositions (verbatim from the notebook)
def atrous(x, J=5):
    """Causal Haar a-trous wavelet: every component uses current and past values only; components sum to x."""
    c = np.asarray(x, float).copy(); comps = []
    for j in range(1, J + 1):
        s_ = 2 ** (j - 1)
        prev = np.concatenate([np.full(s_, c[0]), c[:-s_]])
        cn = 0.5 * (c + prev); comps.append(c - cn); c = cn
    comps.append(c)
    return np.array(comps)


def vmd(x):
    from vmdpy import VMD
    x = np.asarray(x, float)[len(x) % 2:]      # vmdpy needs an even length; drop the earliest sample
    u, _, _ = VMD(x, 2000, 0, 6, 0, 1, 1e-7)
    return u


def bands3(imfs, mp):
    out = np.zeros((3, imfs.shape[1]))
    for k in range(imfs.shape[0]):
        out[{"L": 0, "M": 1, "H": 2}[mp.get(k, "L")]] += imfs[k]
    return out


def msc(idx):
    m = (np.asarray(idx) % 12) + 1
    return np.column_stack([np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)])


def _vmd_origin(y, t):
    u = vmd(y[:t]); b = bands3(u, grouping(u))
    return b[:, -12:], b[:, -1]


def vmd_cache(st, n_jobs):
    fn = os.path.join(VDIR, f"vmd_{st}.npz")
    if os.path.exists(fn):
        return fn
    from joblib import Parallel, delayed
    y = load_station(st)["y"]; t0 = time.time()
    res = Parallel(n_jobs=n_jobs, batch_size=1)(delayed(_vmd_origin)(y, t) for t in range(T0, N))
    arr = np.zeros((N, 3, 12)); last = np.zeros((N, 3))
    for t, (w, l) in zip(range(T0, N), res):
        arr[t], last[t] = w, l
    np.savez(fn, arr=arr, last=last)
    C.log(f"validation23 VMD cache {st}: {(time.time() - t0) / 60:.1f} min", "validation23.log")
    return fn


def exog_available(d, l4):
    """Row t = exogenous values available at origin t (notebook: month t-1 for every predictor)."""
    X = d["X"]; A = np.vstack([X[:1], X[:-1]])
    if l4 == "closed":
        X2 = np.vstack([X[:2], X[:-2]])
        for j, c in enumerate(EXOG):
            if c.startswith("NINO") or c in ("DMI", "PDO"):
                A[:, j] = X2[:, j]
    return A


# ------------------------------------------------------------------ base models (notebook cell 33)
def build_base(st, l4):
    from scipy.optimize import nnls  # noqa: F401
    from sklearn.svm import SVR
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from tensorflow.keras import Model, callbacks, layers
    from xgboost import XGBRegressor
    C.tf_mod()
    d = load_station(st); D = dict(np.load(os.path.join(OUT_DIR, f"dec_{st}.npz"))); y = d["y"]; t0 = time.time()
    XA = exog_available(d, l4)
    xm, xs = d["X"][:TR].mean(0), d["X"][:TR].std(0); Xz = (XA - xm) / xs        # Xz[t] is available at origin t
    ym, ys = y[:TR].mean(), y[:TR].std()
    mapC = grouping(D["train"])
    bC_full = bands3(D["full"], mapC)
    full_w = atrous(y); bW_full = bands3(full_w, grouping(full_w))
    full_v = vmd(y); bV_full = bands3(full_v, grouping(full_v))
    V = np.load(os.path.join(VDIR, f"vmd_{st}.npz"))
    LAST, EDGE = {}, {}
    for meth in ["CEEMDAN", "VMD", "WT"]:
        arr = np.zeros((N, 3, 12)); edge = []
        for t in range(T0, N):
            if meth == "CEEMDAN":
                b = bands3(D[f"o{t}"], mapC); bf = bC_full; arr[t] = b[:, -12:]; lastv = b[:, -1]
            elif meth == "VMD":
                bf = bV_full; arr[t] = V["arr"][t]; lastv = V["last"][t]
            else:
                b = bW_full[:, :t]; bf = bW_full; arr[t] = b[:, -12:]; lastv = b[:, -1]
            if t >= TR:
                edge.append(np.abs(lastv - bf[:, t - 1]))
        LAST[meth] = arr
        EDGE[meth] = {"MAD": np.mean(edge, 0).tolist(), "std": bf[:, TR:].std(1).tolist()}
    H = HORIZONS; HM = max(H); ORIG = np.arange(TR, N)
    tgt = lambda t, h: y[t + h - 1] if t + h - 1 < N else np.nan
    feat_raw = lambda t, h: np.concatenate([(y[t - 12:t] - ym) / ys, Xz[t], msc([t + h - 1])[0]])
    feat_hyb = lambda m, t, h: np.concatenate([LAST[m][t].ravel() / ys, Xz[t], msc([t + h - 1])[0]])
    seq_raw = lambda t: np.column_stack([(y[t - 12:t] - ym) / ys, Xz[t - 11:t + 1], msc(np.arange(t - 12, t))])
    seq_hyb = lambda m, t: np.column_stack([LAST[m][t].T / ys, msc(np.arange(t - 12, t))])
    FC = {}
    sm = SARIMAX(y[:TR], order=(2, 1, 2), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    FC["SARIMA"] = np.array([[sm.apply(y[:t]).forecast(HM)[h - 1] for h in H] for t in ORIG])

    def tab_model(kind, featf):
        out = np.zeros((len(ORIG), len(H)))
        for j, h in enumerate(H):
            tr = np.arange(T0, TR - h + 1)
            Xt = np.array([featf(t, h) for t in tr]); yt = (np.array([tgt(t, h) for t in tr]) - ym) / ys
            m = SVR(C=1.0, epsilon=0.1).fit(Xt, yt) if kind == "SVR" else \
                XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42).fit(Xt, yt)
            out[:, j] = m.predict(np.array([featf(t, h) for t in ORIG])) * ys + ym
        return out
    FC["SVR"] = tab_model("SVR", feat_raw); FC["XGBoost"] = tab_model("XGB", feat_raw)
    for meth in ["CEEMDAN", "VMD", "WT"]:
        FC[f"XGBoost-{meth}"] = tab_model("XGB", lambda t, h, m=meth: feat_hyb(m, t, h))

    def rnn_model(seqf, cell):
        tr = np.arange(T0, TR - HM + 1)
        Xs = np.array([seqf(t) for t in tr]); Ys = (np.array([[tgt(t, h) for h in H] for t in tr]) - ym) / ys
        seed(42)
        inp = layers.Input(Xs.shape[1:])
        z = (layers.LSTM if cell == "LSTM" else layers.GRU)(64)(inp)
        z = layers.Dropout(0.2)(z); z = layers.Dense(32, activation="relu")(z)
        m = Model(inp, layers.Dense(len(H))(z)); m.compile(optimizer="adam", loss="mse")
        m.fit(Xs, Ys, epochs=200, batch_size=32, validation_split=0.15, verbose=0,
              callbacks=[callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
        return m.predict(np.array([seqf(t) for t in ORIG]), verbose=0) * ys + ym
    FC["LSTM"] = rnn_model(seq_raw, "LSTM"); FC["GRU"] = rnn_model(seq_raw, "GRU")
    for meth in ["CEEMDAN", "VMD", "WT"]:
        FC[f"LSTM-{meth}"] = rnn_model(lambda t, m=meth: seq_hyb(m, t), "LSTM")
    month = d["month"]
    Y = np.array([[tgt(t, h) for h in H] for t in ORIG])
    clim = np.array([[y[:MT][month[:MT] == ((t + h - 1) % 12) + 1].mean() for h in H] for t in ORIG])
    C.log(f"validation23 {st} L4 {l4}: base models in {(time.time() - t0) / 60:.1f} min", "validation23.log")
    return {"names": list(FC), "F": np.stack([FC[k] for k in FC]), "Y": Y, "clim": clim, "orig": ORIG, "edge": EDGE}


# ------------------------------------------------------------------ combiners and adaptive gate (notebook cell 34)
RAW = ["SARIMA", "SVR", "XGBoost", "LSTM", "GRU"]
SC = 100.0


def run_meta(B):
    import tensorflow as tf
    from scipy.optimize import nnls
    from sklearn.neural_network import MLPRegressor
    from tensorflow.keras import Model, callbacks, layers, regularizers
    names, F, Y, CL, orig = B["names"], B["F"], B["Y"], B["clim"], B["orig"]
    HYB = [n for n in names if "-" in n]
    R = {"base": {}, "comb": {}, "dm": {}}
    for j, h in enumerate(HORIZONS):
        T = orig + h - 1
        rows = {int(t): k for k, t in enumerate(T) if t < N}
        tr_T = list(range(970 + 2 * h, MT)); te_T = list(range(MT, N))
        P = lambda pool, Ts: np.array([[F[names.index(n), rows[t], j] for n in pool] for t in Ts])
        ytr = np.array([Y[rows[t], j] for t in tr_T]); yte = np.array([Y[rows[t], j] for t in te_T])
        cte = np.array([CL[rows[t], j] for t in te_T])
        R["base"][h] = {n: metrics(yte, P([n], te_T)[:, 0]) for n in names}
        R["base"][h]["Climatology"] = metrics(yte, cte)
        R["comb"][h], R["dm"][h] = {}, {}
        for pool_name, pool in [("raw", RAW), ("raw+hybrid", RAW + HYB)]:
            Ptr, Pte = P(pool, tr_T), P(pool, te_T)
            res = {"Mean": Pte.mean(1)}
            w, _ = nnls(Ptr, ytr); res["NNLS stacking"] = Pte @ w
            best = int(np.argmin(((Ptr - ytr[:, None]) ** 2).mean(0))); res["Best model"] = Pte[:, best]
            mlp_f = [MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=s_).fit(Ptr / SC, ytr / SC).predict(Pte / SC) * SC for s_ in SEEDS]

            def err_seq(Ts):
                out = []
                for t in Ts:
                    o = t - h + 1
                    out.append(np.array([[Y[rows[g], j] - F[names.index(n), rows[g], j] for n in pool] for g in range(o - 12, o)]) / SC)
                return np.array(out)
            Etr, Ete, Mtr, Mte = err_seq(tr_T), err_seq(te_T), msc(tr_T), msc(te_T)
            ada_f = []
            for s_ in SEEDS:
                seed(s_)
                ei = layers.Input(Etr.shape[1:]); fi = layers.Input((len(pool),)); mi = layers.Input((2,))
                z = layers.LSTM(16, kernel_regularizer=regularizers.l2(1e-3))(ei)
                z = layers.Concatenate()([z, mi, fi]); z = layers.Dropout(0.2)(z)
                z = layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(z)
                wts = layers.Dense(len(pool), activation="softmax", name="w")(z)
                m = Model([ei, fi, mi], layers.Dot(axes=1)([wts, fi])); m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
                nv = max(12, int(0.2 * len(tr_T)))
                m.fit([Etr[:-nv], Ptr[:-nv] / SC, Mtr[:-nv]], ytr[:-nv] / SC, epochs=400, batch_size=16, verbose=0,
                      validation_data=([Etr[-nv:], Ptr[-nv:] / SC, Mtr[-nv:]], ytr[-nv:] / SC),
                      callbacks=[callbacks.EarlyStopping(patience=40, restore_best_weights=True)])
                ada_f.append(m.predict([Ete, Pte / SC, Mte], verbose=0).ravel() * SC)
            out = {k: metrics(yte, v) for k, v in res.items()}
            for k, v in {"MLP": mlp_f, "Adaptive LSTM": ada_f}.items():
                ms_ = [metrics(yte, f) for f in v]
                out[k] = {m_: float(np.mean([x[m_] for x in ms_])) for m_ in ["MAE", "RMSE", "NSE"]}
                out[k].update({m_ + "_sd": float(np.std([x[m_] for x in ms_], ddof=1)) for m_ in ["MAE", "RMSE", "NSE"]})
            out["best_model"] = pool[best]
            R["comb"][h][pool_name] = out
            am = np.mean(ada_f, 0)
            R["dm"][h][pool_name] = {k: dm_test(yte - am, yte - v, h) for k, v in
                                     {"Climatology": cte, "Mean": res["Mean"], "NNLS stacking": res["NNLS stacking"],
                                      "Best model": res["Best model"], "MLP": np.mean(mlp_f, 0)}.items()}
    return R


def dm_hybrid(B):
    """Notebook cell 38: hybrid vs raw model of the same architecture (DM < 0: hybrid better)."""
    names, F, Y, orig = B["names"], B["F"], B["Y"], B["orig"]; rows_ = []
    for j, h in enumerate(HORIZONS):
        T = orig + h - 1; rows = {int(t): k for k, t in enumerate(T) if t < N}
        te = [rows[t] for t in range(MT, N)]; yte = Y[te, j]
        for base_m in ["XGBoost", "LSTM"]:
            for meth in ["CEEMDAN", "VMD", "WT"]:
                r = dm_test(yte - F[names.index(f"{base_m}-{meth}"), te, j], yte - F[names.index(base_m), te, j], h)
                rows_.append({"h": h, "Hybrid": f"{base_m}-{meth}", "Raw": base_m,
                              "NSE_hybrid": metrics(yte, F[names.index(f"{base_m}-{meth}"), te, j])["NSE"],
                              "NSE_raw": metrics(yte, F[names.index(base_m), te, j])["NSE"], **r})
    return rows_


def run(st, l4):
    tag = f"{st}_L4{l4}"; t0 = time.time()
    B = build_base(st, l4)
    np.savez(os.path.join(VDIR, f"base_{tag}.npz"), names=np.array(B["names"]), F=B["F"], Y=B["Y"], clim=B["clim"], orig=B["orig"])
    json.dump(B["edge"], open(os.path.join(VDIR, f"edge_effect_{st}.json"), "w"), indent=1)
    R = run_meta(B); R["dm_hybrid"] = dm_hybrid(B)
    json.dump(R, open(os.path.join(VDIR, f"meta_{tag}.json"), "w"), indent=1)
    C.log(f"validation23 {tag}: done in {(time.time() - t0) / 60:.1f} min", "validation23.log")


def summary():
    """Numbers quoted in Validation 2 and 3, for each L4 setting that has been run."""
    S = {"edge": {}}
    for st in STATIONS:
        E = json.load(open(os.path.join(VDIR, f"edge_effect_{st}.json")))
        S["edge"][st] = {m: {b: {"mm": round(E[m]["MAD"][i], 1), "pct": round(100 * E[m]["MAD"][i] / E[m]["std"][i])}
                             for i, b in enumerate(["low", "middle", "high"])} for m in E}
    for l4 in ("open", "closed"):
        fns = {st: os.path.join(VDIR, f"meta_{st}_L4{l4}.json") for st in STATIONS}
        if not all(os.path.exists(f) for f in fns.values()):
            continue
        R = {st: json.load(open(f)) for st, f in fns.items()}
        out = {}
        for base_m in ("LSTM", "XGBoost"):
            rows = [r for st in STATIONS for r in R[st]["dm_hybrid"] if r["Raw"] == base_m]
            gains = [r["NSE_hybrid"] - r["NSE_raw"] for r in rows]
            out[base_m] = {"n_cases": len(rows), "n_improved": int(sum(g > 0 for g in gains)), "max_gain": round(max(gains), 3),
                           "n_signif_better": int(sum(r["p"] < 0.05 and r["DM"] < 0 for r in rows)),
                           "h1_Darwin_p": {r["Hybrid"]: round(r["p"], 3) for r in R["Darwin"]["dm_hybrid"] if r["h"] == 1 and r["Raw"] == base_m}}
        for st in STATIONS:
            c = R[st]["comb"]; dm = R[st]["dm"]
            out[st] = {"adaptive_NSE_range": [round(min(c[str(h)]["raw+hybrid"]["Adaptive LSTM"]["NSE"] for h in HORIZONS), 3),
                                             round(max(c[str(h)]["raw+hybrid"]["Adaptive LSTM"]["NSE"] for h in HORIZONS), 3)],
                       "climatology_NSE_h1": round(R[st]["base"]["1"]["Climatology"]["NSE"], 3),
                       "p_vs_climatology": {h: round(dm[str(h)]["raw+hybrid"]["Climatology"]["p"], 3) for h in HORIZONS},
                       "DM_vs_climatology": {h: round(dm[str(h)]["raw+hybrid"]["Climatology"]["DM"], 2) for h in HORIZONS},
                       "p_vs_NNLS": {h: round(dm[str(h)]["raw+hybrid"]["NNLS stacking"]["p"], 3) for h in HORIZONS}}
        S[f"L4_{l4}"] = out
    json.dump(S, open(os.path.join(VDIR, "validation23_summary.json"), "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["vmd", "run", "summary"])
    ap.add_argument("--station", default=None)
    ap.add_argument("--l4", choices=["open", "closed"], default="open")
    ap.add_argument("--n_jobs", type=int, default=14)
    a = ap.parse_args()
    if a.stage == "vmd":
        for st in ([a.station] if a.station else STATIONS):
            vmd_cache(st, a.n_jobs)
    elif a.stage == "run":
        for st in ([a.station] if a.station else STATIONS):
            run(st, a.l4)
    else:
        summary()
