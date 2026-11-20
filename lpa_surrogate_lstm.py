"""Stage 3 with the LSTM meta-learner: null test on surrogate series (as lpa_surrogate.py), output to
results/surrogate_lstm/ and results/lstm/lpa_surrogate_summary.json. Stepwise surrogate decompositions are reused from
results/surrogate/ when present (the decomposition does not depend on the meta-learner).
Usage:  python lpa_surrogate_lstm.py --fast 100 --step 5 --n_jobs 14
"""
import argparse
import glob
import os
import shutil
import time

from lpa_common import OUT_DIR, STATIONS, log

SDIR = os.path.join(OUT_DIR, "surrogate_lstm")
LDIR = os.path.join(OUT_DIR, "lstm")


def _setup():
    import lpa_metalstm  # noqa: F401
    import lpa_surrogate as S
    S.SDIR = SDIR; S.OUT_DIR = LDIR
    return S


def fast_one(st, k, seeds):
    return _setup().fast_one(st, k, seeds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", type=int, default=100)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--n_jobs", type=int, default=14)
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    a = ap.parse_args()
    os.makedirs(SDIR, exist_ok=True); os.makedirs(LDIR, exist_ok=True)
    for f in glob.glob(os.path.join(OUT_DIR, "surrogate", "dec_*.npz")):
        dst = os.path.join(SDIR, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy2(f, dst)
    from joblib import Parallel, delayed
    S = _setup()
    for st in STATIONS:
        t0 = time.time()
        Parallel(n_jobs=a.n_jobs, batch_size=1)(delayed(fast_one)(st, k, a.seeds) for k in range(a.fast))
        log(f"[LSTM] fast surrogates {st}: {a.fast} done in {(time.time() - t0) / 60:.1f} min", "surrogate_lstm.log")
    S.summarize()
    import lpa_common, lpa_main
    lpa_common.SEEDS = a.seeds; lpa_main.SEEDS = a.seeds
    for k in range(a.step):
        for st in STATIONS:
            S.step_one(st, k, a.n_jobs)
        S.summarize()
    log("DONE lpa_surrogate_lstm", "surrogate_lstm.log")
