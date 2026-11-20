"""Stage 5: original design (P0) and causal design (P2, P3; L4 open/closed) with the LSTM meta-learner.

Identical to lpa_main.py except for the meta-learner: the MLP is replaced by an LSTM (64 units, dropout 0.2, dense 32)
that reads the last 12 months of base-model forecasts (including month t), trained on 2001-2014 with
early stopping on the last 20% of the training data. Usage:  python lpa_metalstm.py
"""
import os
import time

import numpy as np

import lpa_common as C
import lpa_main as M
from lpa_common import MT, N, STATIONS, TR, clim_fixed, clim_rolling30, dm_test, log, metrics, save_json, seed, tf_mod


def lstm_meta(Z, y, ntr, sd, win=12):
    tf = tf_mod()
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    seed(sd)
    Z = np.asarray(Z, float); y = np.asarray(y, float)
    sx = MinMaxScaler().fit(Z[:ntr]); sy = MinMaxScaler().fit(y[:ntr, None])
    Zs = sx.transform(Z)
    pad = np.vstack([np.repeat(Zs[:1], win - 1, 0), Zs])
    X = np.stack([pad[i:i + win] for i in range(len(Zs))])
    yt = sy.transform(y[:ntr, None]).ravel()
    nv = max(12, int(0.2 * ntr))
    m = Sequential([Input((win, Z.shape[1])), LSTM(64, activation="tanh"), Dropout(0.2), Dense(32, activation="relu"), Dense(1)])
    m.compile(optimizer="adam", loss="mse")
    m.fit(X[:ntr - nv], yt[:ntr - nv], validation_data=(X[ntr - nv:ntr], yt[ntr - nv:ntr]), epochs=300, batch_size=16,
          verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True)])
    return sy.inverse_transform(m.predict(X, verbose=0)).ravel()


class LSTMMeta:
    """Replaces MLPRegressor in run_p0: fit(Z[:n], y[:n]) then predict(Z) over all rows 2001-2020."""
    def __init__(self, rs=C.MAIN_SEED):
        self.rs = rs

    def fit(self, X, y):
        self.X, self.y = np.asarray(X), np.asarray(y); return self

    def predict(self, Z):
        n = len(self.X)
        assert np.allclose(Z[:n], self.X), "predict must receive all rows from 2001"
        return lstm_meta(Z, np.concatenate([self.y, np.zeros(len(Z) - n)]), n, self.rs)


def stack_lstm(y, rows, phi, sd, L1open):
    Z = np.column_stack([rows, phi]); Z = np.column_stack([Z, Z.sum(1)])
    a = y[TR:]; n = N - TR if L1open else MT - TR
    return lstm_meta(Z, a, n, sd)[MT - TR:], Z[MT - TR:, 3]


M.mlp = lambda rs=C.MAIN_SEED: LSTMMeta(rs)
M.stack = stack_lstm

if __name__ == "__main__":
    ALL = {}
    for st in STATIONS:
        T0_ = time.time(); log(f"=== LSTM meta-learner: {st} ===", "metalstm.log")
        d, D, bF, bT, mapT, cacheT, extra = M.prep(st)
        y = d["y"]; a = y[MT:]; clim, clim30 = clim_fixed(y, d["month"]), clim_rolling30(y)
        R = {"clim_fixed": metrics(a, clim), "clim_rolling30": metrics(a, clim30)}
        R["P0"], FP0 = M.run_p0(d, bF)
        Xr = M.release_lagged(d)
        R["P2"], FP2 = M.run_cell(d, bT, lambda t: cacheT[t], lambda t: cacheT[t + 1], d["Xlag"])
        R["P2_L4closed"], _ = M.run_cell(d, bT, lambda t: cacheT[t], lambda t: cacheT[t + 1], Xr)
        R["P3"], FP3 = M.run_p3(d, cacheT)
        R["P3_L4closed"], FP3r = M.run_p3(d, cacheT, ex=Xr)
        R["DM"] = {k: {"vs_clim_fixed": dm_test(a - f, a - clim), "vs_clim_rolling30": dm_test(a - f, a - clim30)}
                   for k, f in {"P2": FP2, "P3": FP3, "P3_L4closed": FP3r}.items()}
        ms = lambda r: (float(np.mean([x["NSE"] for x in r])), float(np.std([x["NSE"] for x in r], ddof=1)))
        log(f"  {st}: P0_240 {ms(R['P0']['P0_240'])[0]:.3f}, P0_72 {ms(R['P0']['P0_72'])[0]:.3f}, "
            f"P2 {ms(R['P2']['seeds'])[0]:.3f}, P3 {ms(R['P3']['seeds'])[0]:.3f}, P3+L4 {ms(R['P3_L4closed']['seeds'])[0]:.3f} "
            f"({(time.time() - T0_) / 60:.1f} min)", "metalstm.log")
        ALL[st] = R
        save_json(R, f"lpa_metalstm_{st}.json")
    save_json(ALL, "lpa_metalstm_all.json")
    log("DONE lpa_metalstm", "metalstm.log")
