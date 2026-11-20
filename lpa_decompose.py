"""Stage 1: CEEMDAN cache (whole series, 1921-2000, and origins 240..1199) for every station.

Usage:
    python lpa_decompose.py --bench            # time one decomposition
    python lpa_decompose.py --n_jobs 14        # compute the whole cache (can be resumed if interrupted)
"""
import argparse
import os
import time

import numpy as np

from lpa_common import (OUT_DIR, STATIONS, T0, N, ceemdan, decompose_series, load_station, log)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_jobs", type=int, default=14)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--stations", nargs="*", default=STATIONS)
    a = ap.parse_args()
    if a.bench:
        y = load_station("Darwin")["y"]
        for L in (600, 1200):
            t0 = time.time(); ceemdan(y[:L]); log(f"bench: CEEMDAN length {L} = {time.time() - t0:.1f} s")
    else:
        for st in a.stations:
            y = load_station(st)["y"]
            # save after every block so that an interrupted run can be resumed
            fn = os.path.join(OUT_DIR, f"dec_{st}.npz")
            for lo in range(T0, N, 120):
                decompose_series(y, fn, a.n_jobs, origins=range(lo, min(lo + 120, N)), label=f"{st} o{lo}-o{min(lo + 120, N) - 1}")
            D = dict(np.load(fn))
            log(f"{st}: cache complete, {len(D)} decompositions")
