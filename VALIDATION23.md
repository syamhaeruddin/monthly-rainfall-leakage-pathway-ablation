# Validation 2, Validation 3 and the copy diagnostic

These scripts regenerate the parts of the MethodsX article that were first produced in the original notebook
(`Naskah_IJIES_Final.ipynb`, Part B). The model code is copied from the notebook without change.

| Script | Article | Output |
| --- | --- | --- |
| `lpa_validation23.py` | Step 6 and Validation 2 (edge effect of CEEMDAN, VMD and causal à trous wavelet); Validation 3 (raw and hybrid base models at h = 1, 3, 6, 12; mean, NNLS, best model, MLP and the adaptive LSTM gate with softmax weights) | `results/validation23/` |
| `lpa_copy_diagnostic.py` | Step 7, rule 6, and the copy diagnostic quoted in Validation 1 | `results/copy_diagnostic.json` |

Both need the CEEMDAN cache `results/dec_<station>.npz` from `lpa_decompose.py`.

```powershell
powershell -ExecutionPolicy Bypass -File run_validation23.ps1 -NJobs 14   # ~11 min on 16 threads
python lpa_copy_diagnostic.py                                            # ~5 min
```

`lpa_validation23.py run --l4 open` reproduces the notebook, where every exogenous predictor enters at month t − 1.
`--l4 closed` uses release-lagged predictors (Niño, DMI, PDO at t − 2), so that L1–L4 are all closed; the article
reports this setting and quotes the L4-open values for comparison. `results/validation23/validation23_summary.json`
collects every number quoted in the article for both settings.

Results of the run on 20 November 2026:

| | L4 open (notebook) | L4 closed (article) |
| --- | --- | --- |
| Edge effect, all 18 values | identical to the article | – |
| LSTM: hybrid better than raw | 23/24 cases, up to +0.148 | 21/24, up to +0.168 |
| LSTM, h = 1, Darwin, DM p | 0.006–0.015 | 0.012–0.026 |
| XGBoost: hybrid better than raw | 11/24, none significant | 14/24, one significant |
| Adaptive gate NSE, Darwin / Perth | 0.599–0.639 / 0.535–0.568 | 0.603–0.649 / 0.523–0.578 |
| Gate vs fixed climatology, Perth, p | ≤ 0.021 at all horizons | ≤ 0.013 at h = 3, 6, 12; 0.066 at h = 1 |

The gate NSE with L4 open differs from the notebook output (0.603–0.638 / 0.544–0.568) by up to 0.009; the rest of
the L4-open numbers match the notebook exactly. Copy diagnostic: P0 0.99994–0.99998; P2 and P3 0.35–0.69.
