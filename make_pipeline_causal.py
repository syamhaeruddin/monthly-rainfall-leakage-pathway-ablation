"""Pipeline figure of the CAUSAL design (P3), companion of fig0_pipeline, with pathways L1-L4 closed.

Usage:  python make_pipeline_causal.py en results/figures   (or "id")
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RED, BLUE = "#008300", "#2a78d6"   # green = pathway closed
plt.rcParams.update({"font.family": "DejaVu Sans"})

T = {
"en": dict(
 title="Causal design (P3): the same pipeline with every leakage pathway closed",
 data=("Darwin and Perth", ["SILO + BoM/NOAA, monthly", "Jan 1921 – Dec 2020", "1,200 months per station"]),
 prep=("1  Preprocessing", ["same as P0; predictors lagged", "to t − 1; Niño, DMI, PDO", "to t − 2 (release dates)"]),
 dec=("2  Stepwise CEEMDAN", ["re-run at EVERY origin t", "on months 1 … t − 1 only", "(training and test origins)"]),
 band=("3  Band grouping", ["mapping fixed on the 1921–", "2000 decomposition; extra", "IMFs go to the low band"]),
 lo=("4a  Low band → SARIMAX", ["trained on stepwise band values;", "predictors of t − 1; one step", "ahead, state updated every month"]),
 mi=("4b  Middle band → XGBoost", ["7 band values t − 7 … t − 1", "from the partial decomposition", "(training features built the same way)"]),
 hi=("4c  High band → LSTM", ["window of 12 band values", "t − 12 … t − 1, never the", "target month itself"]),
 mlp=("5  MLP meta-learner", ["same inputs and architecture;", "trained on 2001–2014 only,", "never on test months"]),
 ev=("6  Evaluation", ["NSE on 2015–2020 only; fixed", "and rolling 30-year climatology;", "Diebold–Mariano tests"]),
 res=("Honest NSE", ["Darwin 0.400 (best climatology 0.643)", "Perth 0.055 (best climatology 0.590)"]),
 split=["Base models: 1921–2000", "MLP: 2001–2014", "Test: 2015–2020"],
 leg=[("L3", "Closed: each origin sees a decomposition of months 1 … t − 1 only, and training features are built the same way (P3)."),
      ("L2", "Closed: every input ends at month t − 1; no model sees a value from the month it forecasts."),
      ("L4", "Closed: each climate index is used only after its release date."),
      ("L1", "Closed: the meta-learner is trained on 2001–2014 and scored only on the unseen 2015–2020 window.")],
 fixlab="Result: skill falls from 0.877 / 0.936 (P0) to 0.400 / 0.055 (Darwin / Perth), below climatology at both stations; the gap is the inflation caused by leakage.",
 ),
"id": dict(
 title="Rancangan kausal (P3): alur yang sama dengan semua jalur kebocoran ditutup",
 data=("Darwin dan Perth", ["SILO + BoM/NOAA, bulanan", "Jan 1921 – Des 2020", "1.200 bulan per stasiun"]),
 prep=("1  Prapemrosesan", ["sama dengan P0; prediktor", "digeser ke t − 1; Niño, DMI,", "PDO ke t − 2 (tanggal rilis)"]),
 dec=("2  CEEMDAN bertahap", ["diulang di SETIAP titik awal t", "hanya pada bulan 1 … t − 1", "(titik awal latih dan uji)"]),
 band=("3  Pembagian pita", ["pemetaan ditetapkan dari", "dekomposisi 1921–2000; IMF", "tambahan masuk pita rendah"]),
 lo=("4a  Pita rendah → SARIMAX", ["dilatih pada nilai pita bertahap;", "prediktor t − 1; satu langkah", "ke depan, status diperbarui tiap bulan"]),
 mi=("4b  Pita sedang → XGBoost", ["7 nilai pita t − 7 … t − 1", "dari dekomposisi parsial", "(fitur latih dibentuk sama)"]),
 hi=("4c  Pita tinggi → LSTM", ["jendela 12 nilai pita", "t − 12 … t − 1, tidak pernah", "bulan target itu sendiri"]),
 mlp=("5  Meta-learner MLP", ["masukan dan arsitektur sama;", "dilatih hanya pada 2001–2014,", "tidak pernah pada bulan uji"]),
 ev=("6  Evaluasi", ["NSE hanya pada 2015–2020;", "klimatologi tetap dan bergulir", "30 tahun; uji Diebold–Mariano"]),
 res=("NSE jujur", ["Darwin 0,400 (klimatologi terbaik 0,643)", "Perth 0,055 (klimatologi terbaik 0,590)"]),
 split=["Model dasar: 1921–2000", "MLP: 2001–2014", "Uji: 2015–2020"],
 leg=[("L3", "Ditutup: setiap titik awal hanya melihat dekomposisi bulan 1 … t − 1, dan fitur latih dibentuk dengan cara yang sama (P3)."),
      ("L2", "Ditutup: semua masukan berakhir di bulan t − 1; tidak ada model yang melihat nilai bulan yang diprakirakannya."),
      ("L4", "Ditutup: setiap indeks iklim hanya dipakai setelah tanggal rilisnya."),
      ("L1", "Ditutup: meta-learner dilatih pada 2001–2014 dan dinilai hanya pada jendela 2015–2020 yang belum pernah dilihat.")],
 fixlab="Hasil: kinerja turun dari 0,877 / 0,936 (P0) menjadi 0,400 / 0,055 (Darwin / Perth), di bawah klimatologi pada kedua stasiun; selisihnya adalah penggelembungan akibat kebocoran.",
 )}[LANG]


def box(ax, x, y, w, h, title, lines, leak=False, accent=False):
    ec = RED if leak else (BLUE if accent else INK2)
    fc = "#e5f4e4" if leak else ("#e7f1fb" if accent else "white")
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=1.5 if (leak or accent) else 0.9))
    ax.text(x + 1.4, y + h - 1.6, title, fontsize=7.2, fontweight="bold", va="top", color=INK)
    for i, l in enumerate(lines):
        ax.text(x + 1.4, y + h - 5.4 - i * 2.9, l, fontsize=6.1, va="top", color=INK2)


def tag(ax, x, y, lab):
    ax.add_patch(plt.Circle((x, y), 2.1, color=RED, zorder=5))
    ax.text(x, y - 0.1, lab, color="white", fontsize=6.6, fontweight="bold", ha="center", va="center", zorder=6)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, mutation_scale=8))


import textwrap
Y0, Y1 = 20, 94          # y range of the diagram body
LEGH = 31
H = (Y1 - Y0 + LEGH) * 0.05
figc = plt.figure(figsize=(7.2, H))
ax = figc.add_axes([0, LEGH / (Y1 - Y0 + LEGH), 1, (Y1 - Y0) / (Y1 - Y0 + LEGH)]); ax.set_xlim(0, 144); ax.set_ylim(Y0, Y1); ax.axis("off")
la = figc.add_axes([0, 0, 1, LEGH / (Y1 - Y0 + LEGH)]); la.set_xlim(0, 144); la.set_ylim(0, LEGH); la.axis("off")
ax.text(2, 92.5, T["title"], fontsize=8.6, fontweight="bold", va="top")

# row 1
y1, h1, w = 69, 17, 32
xs = [2, 38, 74, 110]
box(ax, xs[0], y1, w, h1, *T["data"], accent=True)
box(ax, xs[1], y1, w, h1, *T["prep"])
box(ax, xs[2], y1, w, h1, *T["dec"], leak=True)
box(ax, xs[3], y1, w, h1, *T["band"], leak=True)
for i in range(3):
    arrow(ax, xs[i] + w, y1 + h1 / 2, xs[i + 1], y1 + h1 / 2)
tag(ax, xs[2] + w - 2.5, y1 + h1 - 2.5, "L3"); tag(ax, xs[3] + w - 2.5, y1 + h1 - 2.5, "L3")

# timeline
ty = 62.5
spans = [(2, 82, "#f0efec"), (84, 113, "#e7f1fb"), (117, 142, "#f9eeda")]
for (a_, b_, c_), lab in zip(spans, T["split"]):
    ax.add_patch(Rectangle((a_, ty), b_ - a_, 4, fc=c_, ec="none"))
    ax.text((a_ + b_) / 2, ty + 2, lab, fontsize=6.2, ha="center", va="center", color=INK)

# row 2
y2, h2, w2 = 41, 15, 44
xb = [2, 50, 98]
box(ax, xb[0], y2, w2, h2, *T["lo"], leak=True)
box(ax, xb[1], y2, w2, h2, *T["mi"])
box(ax, xb[2], y2, w2, h2, *T["hi"], leak=True)
tag(ax, xb[0] + w2 - 2.5, y2 + h2 - 2.5, "L2"); tag(ax, xb[0] + w2 - 7.5, y2 + h2 - 2.5, "L4")
tag(ax, xb[2] + w2 - 2.5, y2 + h2 - 2.5, "L2")
bx, yb = 115, 59.5
ax.plot([bx, bx], [y1, yb], color=INK2, lw=0.9)
ax.plot([xb[0] + w2 / 2, xb[2] + w2 / 2], [yb, yb], color=INK2, lw=0.9)
for xx in xb:
    arrow(ax, xx + w2 / 2, yb, xx + w2 / 2, y2 + h2)

# row 3
y3, h3 = 21, 15
box(ax, 2, y3, 44, h3, *T["mlp"])
box(ax, 50, y3, 44, h3, *T["ev"], leak=True)
box(ax, 98, y3, 44, h3, *T["res"], accent=True)
tag(ax, 50 + 44 - 2.5, y3 + h3 - 2.5, "L1")
yj = 38.5
ax.plot([xb[0] + w2 / 2, xb[2] + w2 / 2], [yj, yj], color=INK2, lw=0.9)
for xx in xb:
    ax.plot([xx + w2 / 2, xx + w2 / 2], [y2, yj], color=INK2, lw=0.9)
arrow(ax, 24, yj, 24, y3 + h3)
arrow(ax, 46, y3 + h3 / 2, 50, y3 + h3 / 2)
arrow(ax, 94, y3 + h3 / 2, 98, y3 + h3 / 2)

for i, (k, txt) in enumerate(T["leg"]):
    yy = 29 - i * 5.4
    la.add_patch(plt.Circle((4, yy - 1.3), 1.8, color=RED))
    la.text(4, yy - 1.4, k, color="white", fontsize=6, fontweight="bold", ha="center", va="center")
    la.text(8, yy, txt, fontsize=6.1, va="top", color=INK)
la.text(2, 6.5, "\n".join(textwrap.wrap(T["fixlab"], 150)), fontsize=6.1, va="top", color=BLUE)
for ext in ("png", "pdf"):
    figc.savefig(os.path.join(OUT, f"fig0b_pipeline_causal_{LANG}.{ext}"), dpi=600, facecolor="white", bbox_inches="tight")
print("ok")
