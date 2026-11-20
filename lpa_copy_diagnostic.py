"""Copy diagnostic (step 7, rule 6): correlation of the high-band LSTM output with the band value of the target month.

P0 (L2 open, as in lpa_main.run_p0): the LSTM receives the high-band value of month t and is asked for the same
value, so it can copy its input. P2 and P3 (L2 closed, as in lpa_main.lstm_rows and run_p3): the input ends at
t - 1. A correlation above 0.999 flags copying. Seeds 1-5, test window 2015-2020.
Usage:  python lpa_copy_diagnostic.py      Output: results/copy_diagnostic.json
"""
import numpy as np

import lpa_main as M
from lpa_common import MT, N, SEEDS, STATIONS, T0, TR, log, lstm_model, save_json, seed


def p0_high(bF, sd):
    from sklearn.preprocessing import MinMaxScaler
    seed(sd)
    hi = bF["H"]; sc = MinMaxScaler(); xtr = sc.fit_transform(hi[:TR, None]); xte = sc.transform(hi[TR:, None])
    m = lstm_model(1); m.fit(xtr[:, None, :], xtr, epochs=50, batch_size=16, verbose=0)
    return sc.inverse_transform(m.predict(xte[:, None, :], verbose=0)).ravel()


def p3_high(cacheT, sd):
    from sklearn.preprocessing import MinMaxScaler
    seed(sd)
    e = np.array([cacheT[s + 1]["H"][s] for s in range(T0, N)]); tr = np.arange(T0, TR)
    sc = MinMaxScaler().fit(e[:TR - T0, None])
    Xh = np.array([sc.transform(cacheT[s]["H"][s - 12:s, None]).ravel() for s in tr]); yh = sc.transform(e[:TR - T0, None]).ravel()
    m = lstm_model(12); m.fit(Xh[:, :, None], yh, epochs=50, batch_size=16, verbose=0)
    hw = np.array([sc.transform(cacheT[t]["H"][t - 12:t, None]).ravel() for t in range(TR, N)])
    return sc.inverse_transform(m.predict(hw[:, :, None], verbose=0)).ravel(), e[TR - T0:]


if __name__ == "__main__":
    R = {}
    for st in STATIONS:
        d, D, bF, bT, mapT, cacheT, extra = M.prep(st)
        te = slice(MT - TR, N - TR)                     # 2015-2020 within the 2001-2020 forecast rows
        r = {"P0": [], "P2": [], "P3": []}
        for sd in SEEDS:
            r["P0"].append(float(np.corrcoef(p0_high(bF, sd)[te], bF["H"][MT:])[0, 1]))
            phi2 = M.lstm_rows(bT, lambda t: cacheT[t], lambda t: cacheT[t + 1], False, sd)
            edge = np.array([cacheT[t + 1]["H"][t] for t in range(TR, N)])
            r["P2"].append(float(np.corrcoef(phi2[te], edge[te])[0, 1]))
            phi3, e3 = p3_high(cacheT, sd)
            r["P3"].append(float(np.corrcoef(phi3[te], e3[te])[0, 1]))
        R[st] = {k: {"min": min(v), "max": max(v), "seeds": v, "flag_copy": bool(min(v) > 0.999)} for k, v in r.items()}
        log(f"copy diagnostic {st}: " + ", ".join(f"{k} {min(v):.5f}-{max(v):.5f}" for k, v in r.items()))
    save_json(R, "copy_diagnostic.json")
