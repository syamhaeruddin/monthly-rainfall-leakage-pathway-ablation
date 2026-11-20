"""Pipeline figure of the original design (P0) in Naskah_IJIES_Final.ipynb with the entry points of leakage L1-L4.

Usage:  python make_pipeline.py en results/figures   (or "id")
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
RED, BLUE = "#d03b3b", "#2a78d6"
plt.rcParams.update({"font.family": "DejaVu Sans"})

T = {
"en": dict(
 title="Original design (P0) of the CEEMDAN band ensemble and where information leaks",
 data=("Darwin and Perth", ["SILO + BoM/NOAA, monthly", "Jan 1921 – Dec 2020", "1,200 months per station"]),
 prep=("1  Preprocessing", ["rainfall = monthly sum,", "gaps interpolated", "12 predictors: 5 weather,", "7 climate indices"]),
 dec=("2  CEEMDAN", ["run ONCE on all 1,200", "months (I = 100,", "ε = 0.005, seed 42)"]),
 band=("3  Band grouping", ["IMFs sorted by FFT", "dominant frequency into", "low / middle / high bands"]),
 lo=("4a  Low band → SARIMAX", ["(2,1,2)(1,1,1)12 with the 12", "predictors of month t; fits 1921–", "2000, forecasts 2001–2020 at once"]),
 mi=("4b  Middle band → XGBoost", ["300 trees, depth 5;", "input = 7 previous band values", "(t − 7 … t − 1)"]),
 hi=("4c  High band → LSTM", ["64 units; input = band value", "of the target month t itself", "→ output r > 0.9999 with target"]),
 mlp=("5  MLP meta-learner", ["64–32 neurons; inputs =", "3 band forecasts + their sum;", "trained on 2001–2014"]),
 ev=("6  Evaluation", ["NSE on 2001–2020, including", "the 168 months the MLP was", "trained on; fixed climatology"]),
 res=("Reported NSE", ["Darwin 0.880", "Perth 0.936"]),
 split=["Base models: 1921–2000", "MLP: 2001–2014", "Test: 2015–2020"],
 leg=[("L3", "Full-series decomposition: every band value at month t is shaped by months after t; the band mapping is also fixed on the full series."),
      ("L2", "Contemporaneous input: SARIMAX uses predictors of the target month; the LSTM receives the target month's own high-band value."),
      ("L4", "Release latency: Niño, DMI and PDO for month t are not published until about month t + 2."),
      ("L1", "Meta-learner overlap: the score includes the months used to train the MLP.")],
 fixlab="Causal design (P2/P3): re-decompose at every origin with months 1 … t − 1, inputs up to t − 1, MLP scored only on 2015–2020 → NSE 0.17–0.37 (Darwin), 0.10–0.11 (Perth).",
 ),
"id": dict(
 title="Rancangan awal (P0) ensemble pita CEEMDAN dan titik-titik kebocoran informasi",
 data=("Darwin dan Perth", ["SILO + BoM/NOAA, bulanan", "Jan 1921 – Des 2020", "1.200 bulan per stasiun"]),
 prep=("1  Prapemrosesan", ["curah hujan = jumlah bulanan,", "nilai kosong diinterpolasi", "12 prediktor: 5 cuaca,", "7 indeks iklim"]),
 dec=("2  CEEMDAN", ["dijalankan SEKALI pada", "1.200 bulan (I = 100,", "ε = 0,005, seed 42)"]),
 band=("3  Pembagian pita", ["IMF diurutkan menurut", "frekuensi dominan FFT ke", "pita rendah / sedang / tinggi"]),
 lo=("4a  Pita rendah → SARIMAX", ["(2,1,2)(1,1,1)12 dengan 12", "prediktor bulan t; dilatih 1921–", "2000, prakiraan 2001–2020 sekaligus"]),
 mi=("4b  Pita sedang → XGBoost", ["300 pohon, kedalaman 5;", "masukan = 7 nilai pita", "sebelumnya (t − 7 … t − 1)"]),
 hi=("4c  Pita tinggi → LSTM", ["64 unit; masukan = nilai pita", "bulan target t itu sendiri", "→ keluaran r > 0,9999 dgn target"]),
 mlp=("5  Meta-learner MLP", ["64–32 neuron; masukan =", "3 prakiraan pita + jumlahnya;", "dilatih pada 2001–2014"]),
 ev=("6  Evaluasi", ["NSE pada 2001–2020, termasuk", "168 bulan yang dipakai melatih", "MLP; klimatologi tetap"]),
 res=("NSE dilaporkan", ["Darwin 0,880", "Perth 0,936"]),
 split=["Model dasar: 1921–2000", "MLP: 2001–2014", "Uji: 2015–2020"],
 leg=[("L3", "Dekomposisi seluruh deret: nilai pita bulan t ikut dibentuk oleh bulan-bulan sesudah t; pemetaan pita juga ditetapkan dari seluruh deret."),
      ("L2", "Masukan kontemporer: SARIMAX memakai prediktor bulan target; LSTM menerima nilai pita tinggi bulan target itu sendiri."),
      ("L4", "Latensi rilis: Niño, DMI, dan PDO untuk bulan t baru terbit sekitar bulan t + 2."),
      ("L1", "Tumpang tindih meta-learner: skor mencakup bulan yang dipakai melatih MLP.")],
 fixlab="Rancangan kausal (P2/P3): dekomposisi ulang tiap titik awal dengan bulan 1 … t − 1, masukan hingga t − 1, MLP dinilai hanya pada 2015–2020 → NSE 0,17–0,37 (Darwin), 0,10–0,11 (Perth).",
 )}[LANG]


def box(ax, x, y, w, h, title, lines, leak=False, accent=False):
    ec = RED if leak else (BLUE if accent else INK2)
    fc = "#fbebeb" if leak else ("#e7f1fb" if accent else "white")
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
    figc.savefig(os.path.join(OUT, f"fig0_pipeline_{LANG}.{ext}"), dpi=600, facecolor="white", bbox_inches="tight")
print("ok")
