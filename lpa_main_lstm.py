"""Stage 2 with the LSTM meta-learner: the whole of lpa_main.py (P0, 2^3 factorial + Shapley, P3, L4, DM) is rerun
with the MLP replaced by the LSTM meta-learner of lpa_metalstm.py. Output goes to results/lstm/ so the MLP results are kept.
Usage:  python lpa_main_lstm.py
"""
import os
import textwrap

import lpa_metalstm  # noqa: F401  (replaces lpa_main.mlp and lpa_main.stack with the LSTM versions)
import lpa_main as M
from lpa_common import OUT_DIR

os.makedirs(os.path.join(OUT_DIR, "lstm"), exist_ok=True)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lpa_main.py"), encoding="utf-8").read()
body = textwrap.dedent(src.split('if __name__ == "__main__":', 1)[1])
for a, b in [('f"forecasts_lpa_{st}.npz"', 'f"lstm/forecasts_lpa_{st}.npz"'), ('f"lpa_main_{st}.json"', 'f"lstm/lpa_main_{st}.json"'),
             ('"lpa_main_all.json"', '"lstm/lpa_main_all.json"'), ('"lpa_summary.xlsx"', '"lstm/lpa_summary.xlsx"'),
             ('log("DONE lpa_main")', 'log("DONE lpa_main_lstm")')]:
    assert body.count(a) == 1, a
    body = body.replace(a, b)
M.log("##### LSTM meta-learner: lpa_main rerun #####")
exec(compile(body, "lpa_main.py(__main__, LSTM)", "exec"), M.__dict__)
