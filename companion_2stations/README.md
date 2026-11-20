# Two-station rerun of the companion study [31]

Validation 3 of the MethodsX article reports what happens when the cross-station gate of the companion study [31]
is trained on Darwin and Perth only. This folder reproduces that number.

* `run_companion_2stations.ps1 -Src <companion project> -Dst <new folder>` copies the companion project, applies
  `patch_copy.py` to the copy (the station list is read from `STATIONS=Darwin,Perth`; transfer runs only for the two
  folds that contain these stations), builds a fresh environment from the companion `requirements.txt`, runs the full
  pipeline and writes `companion_2stations_summary.json`. The original project is not modified.
* `companion_2stations_summary.json` is the result of our run (Windows, Python 3.13, companion requirements).

Result (test 2015–2025, NSE averaged over h = 1, 3, 6, 12):

| | Rolling 30-yr climatology | Local gate L-G3 | GG3, Darwin + Perth only (4,528 meta-training samples) | GG3, 15 stations (31,160) [31] |
|---|---|---|---|---|
| Darwin | 0.699 | 0.698 | 0.699 | 0.708 |
| Perth | 0.670 | 0.667 | 0.667 | 0.671 |

The base-model forecasts of the rerun are identical to those of the companion study (maximum difference 0), so the
difference between the last two columns comes from the extra meta-training samples. With two stations, GG3 is not
different from L-G3 (panel DM over 2015–2025, Holm-adjusted p >= 0.38 at all four horizons).
