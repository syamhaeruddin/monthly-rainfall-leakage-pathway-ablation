"""Shared functions for the MethodsX experiments (LPA protocol).

Model functions, band grouping and the DM test are copied from Naskah_IJIES_Final.ipynb
without change, so that results can be compared directly with the article.
"""
import os

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"]:
    os.environ.setdefault(_v, "1")
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "results")
os.makedirs(OUT_DIR, exist_ok=True)

STATIONS = ["Darwin", "Perth"]
T0, TR, MT, N = 240, 960, 1128, 1200
CEEMDAN_TRIALS, CEEMDAN_EPS, CEEMDAN_SEED = 100, 0.005, 42
MAIN_SEED = 42
SEEDS = [1, 2, 3, 4, 5]
EXOG = ["max_temp_mean", "min_temp_mean", "radiation_mean", "rh_tmax_mean", "rh_tmin_mean",
        "SOI", "NINO 12", "NINO 3", "NINO 3.4", "NINO 4", "DMI", "PDO"]
SAR = dict(order=(2, 1, 2), seasonal_order=(1, 1, 1, 12),
           enforce_stationarity=False, enforce_invertibility=False)


def log(msg, fn="progress.log"):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(BASE, "logs", fn), "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- data
def load_station(st, y_override=None):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{st}_100y_features.csv"))
    y = df["mon_rain_sum"].astype(float).interpolate().values if y_override is None else np.asarray(y_override, float)
    X = df[EXOG].astype(float).values
    return {"st": st, "df": df, "y": y, "X": X,
            "Xlag": np.vstack([X[:1], X[:-1]]),
            "month": df["month"].values}


# ---------------------------------------------------------------- decomposition
def ceemdan(x):
    from PyEMD import CEEMDAN
    # parallel=False: PyEMD >= 1.10 uses an internal Pool by default; parallelism is handled outside (joblib)
    c = CEEMDAN(trials=CEEMDAN_TRIALS, epsilon=CEEMDAN_EPS, parallel=False)
    c.noise_seed(CEEMDAN_SEED)
    return c.ceemdan(np.asarray(x, float), np.arange(len(x), dtype=float), max_imf=-1)


def decompose_series(y, fn, n_jobs, origins=range(T0, N), label=""):
    """CEEMDAN of the whole series, of 1921-2000, and of every origin t (data 0..t-1). Stored as a cache."""
    from joblib import Parallel, delayed
    jobs = [("full", N), ("train", TR)] + [(f"o{t}", t) for t in origins]
    D = dict(np.load(fn)) if os.path.exists(fn) else {}
    todo = [(k, L) for k, L in jobs if k not in D]
    if todo:
        t0 = time.time()
        # longest series first, so the parallel load is balanced
        todo.sort(key=lambda kl: -kl[1])
        res = Parallel(n_jobs=n_jobs, verbose=0, batch_size=1)(delayed(ceemdan)(y[:L]) for _, L in todo)
        D.update({k: r for (k, _), r in zip(todo, res)})
        np.savez_compressed(fn, **D)
        log(f"{label}: {len(todo)} new decompositions in {(time.time() - t0) / 60:.1f} min")
    return D


# ---------------------------------------------------------------- helpers (copied from the notebook)
def metrics(a, p):
    a, p = np.asarray(a), np.asarray(p); e = a - p
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "NSE": float(1 - np.sum(e ** 2) / np.sum((a - a.mean()) ** 2))}


def dom_freq(s):
    s = s - s.mean(); p = np.abs(np.fft.rfft(s)) ** 2; f = np.fft.rfftfreq(len(s)); p[0] = 0
    return f[np.argmax(p)]


def grouping(imfs):
    order = np.argsort([dom_freq(i) for i in imfs], kind="stable")
    lo, mi, hi = np.array_split(order, 3)
    mp = {int(k): "L" for k in lo}
    mp.update({int(k): "M" for k in mi}); mp.update({int(k): "H" for k in hi})
    return mp


def bands(imfs, mapping):
    out = {k: np.zeros(imfs.shape[1]) for k in "LMH"}
    for k in range(imfs.shape[0]):
        out[mapping.get(k, "L")] += imfs[k]
    return out


def windows(s, w):
    return np.array([s[i - w:i] for i in range(w, len(s))]), s[w:]


def dm_test(e1, e2, h=1):
    d = e1 ** 2 - e2 ** 2; n = len(d); dc = d - d.mean()
    gam = [np.sum(dc[k:] * dc[:n - k]) / n for k in range(h)]
    v = gam[0] + 2 * sum(gam[1:])
    v = v if v > 0 else gam[0]
    stat = d.mean() / np.sqrt(v / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return {"DM": float(stat), "p": float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1)))}


def clim_fixed(y, month):
    return np.array([y[:MT][month[:MT] == mo].mean() for mo in month[MT:]])


def clim_rolling30(y):
    """Rolling 30-year climatology: mean of the same calendar month over the 360 months before t."""
    return np.array([np.mean(y[t - 360:t][::-1][11::12]) for t in range(MT, N)])


# ---------------------------------------------------------------- TF-dependent (lazy import)
_TF = {}


def tf_mod():
    if not _TF:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        try:
            tf.config.experimental.enable_op_determinism()
        except AttributeError:
            pass
        _TF["tf"] = tf
    return _TF["tf"]


def seed(s=MAIN_SEED):
    tf = tf_mod()
    np.random.seed(s); tf.random.set_seed(s)
    try:
        tf.keras.utils.set_random_seed(s)
    except AttributeError:
        import random; random.seed(s)


def lstm_model(win):
    tf = tf_mod()
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    m = Sequential([Input((win, 1)), LSTM(64, activation="tanh"), Dropout(0.2), Dense(32, activation="relu"), Dense(1)])
    m.compile(optimizer="adam", loss="mse")
    return m


def mlp(rs=MAIN_SEED):
    from sklearn.neural_network import MLPRegressor
    return MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", max_iter=1000, random_state=rs)


def xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)


def save_json(obj, name):
    with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
