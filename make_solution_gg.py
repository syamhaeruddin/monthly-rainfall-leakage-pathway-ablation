"""Improvement figure from Eksperimen_Gate_Global_Lintas_Stasiun.ipynb, described for Darwin and Perth.

Sources of the numbers (notebook outputs):
  - Tables 1 and 6 (NSE per station, mean of four horizons, test 2015-2025)
  - laporan_eksperimen.py, section "diag" (ensemble feasibility diagnosis, pre-test 1962-2014)
  - Figure 4 (mean GG3 weight per model group by target month, h = 1); the values were
    read back from the pixels of the notebook figure (gg3_weights_digitised.json, precision ~0.005)
Usage:  python make_solution_gg.py en results/figures   (or "id")
"""
import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter

LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GREEN, BLUE, ORANGE = "#008300", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans"})

RES = {"Darwin": {"Klim30": 0.699, "Rata2": 0.686, "NNLS": 0.681, "L-G3": 0.698, "GG3": 0.708, "GG4-T": 0.699},
       "Perth": {"Klim30": 0.670, "Rata2": 0.647, "NNLS": 0.665, "L-G3": 0.667, "GG3": 0.671, "GG4-T": 0.663}}
DIAG = {"Darwin": {"kor": 0.925, "klim": 0.675, "gain": 0.0059}, "Perth": {"kor": 0.854, "klim": 0.663, "gain": 0.0055}}
W = json.load(open(os.path.join(HERE, "gg3_weights_digitised.json")))
WET = {"Darwin": [11, 12, 1, 2, 3, 4], "Perth": [5, 6, 7, 8, 9, 10]}
GRP = ["SARIMA", "ML", "DL", "Hybrid", "Clim"]
GCOL = {"SARIMA": "#cc79a7", "ML": "#e69f00", "DL": "#56b4e9", "Hybrid": "#009e73", "Clim": "#8c8c8c"}

T = {
"en": dict(
 title="Improvement with the global gate meta-learner (GG3), explained for Darwin and Perth",
 data=("Darwin and Perth", ["Darwin · tropics · 140.3 mm/mo", "Perth · winter rain · 65.7 mm/mo", "SILO 1926–2025; 132 test months"]),
 prep=("1  Preprocessing", ["targets: months with ≥ 90%", "observed days; Niño3.4, DMI", "at o − 2, SOI at o − 1"]),
 base=("2  Causal base models", ["SARIMA, SVR, XGBoost, LSTM,", "GRU, + VMD / causal wavelet;", "retrained every 5 years"]),
 pool=("3  Pool of 11 members", ["9 base models +", "expanding climatology +", "rolling 30-year climatology"]),
 inA=("4a  Error history", ["last 12 months of each of the", "11 members, scaled by the", "station's σ; read by an LSTM"]),
 inB=("4b  Context", ["current forecasts, target month,", "ENSO/IOD regime (6 features),", "horizon one-hot"]),
 inC=("4c  Station identity", ["Darwin and Perth get their own", "embedding + static features", "(location, annual rain, season)"]),
 gate=("5  Global gate (GG3)", ["Darwin or Perth alone: ≤ 2,264", "meta-training samples; GG3 pools", "them with 13 more stations: 31,160"]),
 out=("Weighted forecast", ["softmax → convex weights per", "station, target month, horizon;", "panel (b) shows them"]),
 ev=("6  Evaluation", ["2015–2025, h = 1, 3, 6, 12;", "vs Klim30, NNLS, average,", "local gate (Darwin/Perth only)"]),
 split=["Meta-training: 1962–2009", "Early stop: 2010–2014", "Test: 2015–2025"],
 pa="(a)  NSE, mean of four horizons, test 2015–2025",
 pb="(b)  GG3 weights by target month (h = 1); blue bar = six wettest months",
 pc="(c)  Why the room for improvement is small (pre-test 1962–2014)",
 lab={"Klim30": "Rolling 30-yr climatology", "Rata2": "Simple average", "NNLS": "NNLS stacking",
      "L-G3": "Local gate (own station)", "GG3": "Global gate (GG3)", "GG4-T": "Transfer gate (station unseen)"},
 glab={"SARIMA": "SARIMA", "ML": "ML", "DL": "DL", "Hybrid": "Hybrid", "Clim": "Climatology"},
 months="JFMAMJJASOND", wy="Weight",
 dhead=["", "Darwin", "Perth"],
 drows=[("Error correlation between members", "kor", "{:.3f}"),
        ("Variance explained by climatology", "klim", "{:.3f}"),
        ("NSE gain, global vs local gate (GG4 vs L-G4)", "gain", "+{:.4f}")],
 note=("At both stations the members' errors are highly correlated and climatology already explains about two-thirds of the variance, "
       "so any combination has little to add. GG3 is the best method at both (+0.010 and +0.004 NSE over the local gate, +0.009 and +0.001 over Klim30): "
       "the gain is small, but the method now sits above its climatology, whereas the leakage-free single-station design stayed below it. GG3 also learns the season: "
       "it leans on climatology in Darwin's dry season and gives ML more weight in Perth's wet winter. "
       "NSE here is not comparable with the Darwin–Perth audit (different period, pool and test window)."),
 ),
"id": dict(
 title="Peningkatan dengan gate meta-learner global (GG3), dijelaskan untuk Darwin dan Perth",
 data=("Darwin dan Perth", ["Darwin · tropis · 140,3 mm/bln", "Perth · musim dingin · 65,7 mm/bln", "SILO 1926–2025; 132 bulan uji"]),
 prep=("1  Prapemrosesan", ["target: bulan dengan ≥ 90%", "hari teramati; Niño3.4, DMI", "pada o − 2, SOI pada o − 1"]),
 base=("2  Model dasar kausal", ["SARIMA, SVR, XGBoost, LSTM,", "GRU, + VMD / wavelet kausal;", "dilatih ulang tiap 5 tahun"]),
 pool=("3  Pool 11 anggota", ["9 model dasar +", "klimatologi ekspansif +", "klimatologi bergulir 30 tahun"]),
 inA=("4a  Riwayat galat", ["12 bulan terakhir tiap anggota", "(11), diskalakan σ stasiun;", "dibaca LSTM (16 unit)"]),
 inB=("4b  Konteks", ["prakiraan saat ini, bulan target,", "rezim ENSO/IOD (6 fitur),", "horizon one-hot"]),
 inC=("4c  Identitas stasiun", ["Darwin dan Perth punya", "embedding + fitur statis sendiri", "(lokasi, hujan tahunan, musim)"]),
 gate=("5  Gate global (GG3)", ["Darwin atau Perth saja: ≤ 2.264", "sampel meta-latih; GG3 menggabung", "dengan 13 stasiun lain: 31.160"]),
 out=("Prakiraan terbobot", ["softmax → bobot konveks per", "stasiun, bulan target, horizon;", "ditampilkan di panel (b)"]),
 ev=("6  Evaluasi", ["2015–2025, h = 1, 3, 6, 12;", "vs Klim30, NNLS, rata-rata,", "gate lokal (Darwin/Perth saja)"]),
 split=["Latih meta: 1962–2009", "Early stop: 2010–2014", "Uji: 2015–2025"],
 pa="(a)  NSE, rata-rata empat horizon, uji 2015–2025",
 pb="(b)  Bobot GG3 menurut bulan target (h = 1); garis biru = enam bulan terbasah",
 pc="(c)  Mengapa ruang peningkatannya kecil (pra-uji 1962–2014)",
 lab={"Klim30": "Klimatologi bergulir 30 th", "Rata2": "Rata-rata sederhana", "NNLS": "Stacking NNLS",
      "L-G3": "Gate lokal (stasiun sendiri)", "GG3": "Gate global (GG3)", "GG4-T": "Gate transfer (stasiun baru)"},
 glab={"SARIMA": "SARIMA", "ML": "ML", "DL": "DL", "Hybrid": "Hibrida", "Clim": "Klimatologi"},
 months="JFMAMJJASOND", wy="Bobot",
 dhead=["", "Darwin", "Perth"],
 drows=[("Korelasi galat antaranggota", "kor", "{:.3f}"),
        ("Varians yang dijelaskan klimatologi", "klim", "{:.3f}"),
        ("Tambahan NSE gate global vs lokal (GG4 vs L-G4)", "gain", "+{:.4f}")],
 note=("Di kedua stasiun, galat antaranggota sangat berkorelasi dan klimatologi sudah menjelaskan sekitar dua pertiga varians, "
       "sehingga kombinasi apa pun hanya punya sedikit ruang. GG3 menjadi metode terbaik di keduanya (+0,010 dan +0,004 NSE atas gate lokal, "
       "+0,009 dan +0,001 atas Klim30): peningkatannya kecil, tetapi metode kini berada di atas klimatologinya, sedangkan rancangan satu stasiun tanpa kebocoran tetap di bawahnya. "
       "GG3 juga mempelajari musim: bersandar pada klimatologi di musim kemarau Darwin dan memberi ML bobot lebih besar di musim dingin basah Perth. "
       "NSE di sini tidak dapat dibandingkan langsung dengan audit Darwin–Perth (periode, pool, dan jendela uji berbeda)."),
 )}[LANG]
dec = (lambda s: s) if LANG == "en" else (lambda s: s.replace(".", ","))
fmt = lambda v: dec(f"{v:.3f}")


def box(ax, x, y, w, h, title, lines, kind=None):
    ec = {"g": GREEN, "b": BLUE}.get(kind, INK2)
    fc = {"g": "#e5f4e4", "b": "#e7f1fb"}.get(kind, "white")
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=1.5 if kind else 0.9))
    ax.text(x + 1.4, y + h - 1.6, title, fontsize=7.2, fontweight="bold", va="top", color=INK)
    for i, l in enumerate(lines):
        ax.text(x + 1.4, y + h - 5.4 - i * 2.9, l, fontsize=6.1, va="top", color=INK2)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, mutation_scale=8))


def clean(a):
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_color(INK2); a.spines[sp].set_linewidth(0.6)


# heights (units): diagram, panel a, panel b, panel c + notes
Y0, Y1 = 14, 89.5
HA, HB, HC = 38, 46, 40
TOT = (Y1 - Y0) + HA + HB + HC
fig = plt.figure(figsize=(7.2, TOT * 0.05))
fy = lambda u: u / TOT

ax = fig.add_axes([0, fy(HA + HB + HC), 1, fy(Y1 - Y0)]); ax.set_xlim(0, 144); ax.set_ylim(Y0, Y1); ax.axis("off")
ax.text(2, 88.5, T["title"], fontsize=8.4, fontweight="bold", va="top")
y1, h1, w = 66, 17, 32
xs = [2, 38, 74, 110]
box(ax, xs[0], y1, w, h1, *T["data"], kind="b")
box(ax, xs[1], y1, w, h1, *T["prep"])
box(ax, xs[2], y1, w, h1, *T["base"])
box(ax, xs[3], y1, w, h1, *T["pool"])
for i in range(3):
    arrow(ax, xs[i] + w, y1 + h1 / 2, xs[i + 1], y1 + h1 / 2)
ty = 59.5
for (a_, b_, c_), lab in zip([(2, 82, "#f0efec"), (84, 113, "#e7f1fb"), (117, 142, "#f9eeda")], T["split"]):
    ax.add_patch(Rectangle((a_, ty), b_ - a_, 4, fc=c_, ec="none"))
    ax.text((a_ + b_) / 2, ty + 2, lab, fontsize=6.2, ha="center", va="center", color=INK)
y2, h2, w2 = 38, 15, 44
xb = [2, 50, 98]
box(ax, xb[0], y2, w2, h2, *T["inA"])
box(ax, xb[1], y2, w2, h2, *T["inB"])
box(ax, xb[2], y2, w2, h2, *T["inC"], kind="g")
bx, yb = 115, 56.5
ax.plot([bx, bx], [y1, yb], color=INK2, lw=0.9)
ax.plot([xb[0] + w2 / 2, xb[2] + w2 / 2], [yb, yb], color=INK2, lw=0.9)
for xx in xb:
    arrow(ax, xx + w2 / 2, yb, xx + w2 / 2, y2 + h2)
y3, h3 = 15, 15
box(ax, 2, y3, 44, h3, *T["gate"], kind="g")
box(ax, 50, y3, 44, h3, *T["out"])
box(ax, 98, y3, 44, h3, *T["ev"])
yj = 35
ax.plot([xb[0] + w2 / 2, xb[2] + w2 / 2], [yj, yj], color=INK2, lw=0.9)
for xx in xb:
    ax.plot([xx + w2 / 2, xx + w2 / 2], [y2, yj], color=INK2, lw=0.9)
arrow(ax, 24, yj, 24, y3 + h3)
arrow(ax, 46, y3 + h3 / 2, 50, y3 + h3 / 2)
arrow(ax, 94, y3 + h3 / 2, 98, y3 + h3 / 2)

# (a) NSE
base_a = HB + HC
fig.text(2 / 144, fy(base_a + HA - 2), T["pa"], fontsize=7.2, fontweight="bold", va="top", color=INK)
order = ["Klim30", "Rata2", "NNLS", "L-G3", "GG4-T", "GG3"]
for k, st in enumerate(["Darwin", "Perth"]):
    sub = fig.add_axes([0.235 + k * 0.40, fy(base_a + 5), 0.33, fy(HA - 12)])
    vals = [RES[st][m] for m in order]
    lo, hi = min(vals) - 0.012, max(vals) + 0.012
    for i, m in enumerate(order):
        c = GREEN if m == "GG3" else (ORANGE if m == "Klim30" else BLUE)
        sub.plot([lo, RES[st][m]], [i, i], color=GRID, lw=0.8, zorder=1)
        sub.scatter(RES[st][m], i, s=22, color=c, zorder=3, edgecolors="white", linewidths=0.6)
        sub.text(RES[st][m] + 0.0012, i, fmt(RES[st][m]), fontsize=5.8, va="center", color=INK,
                 fontweight="bold" if m == "GG3" else "normal")
    sub.set_xlim(lo, hi + 0.006); sub.set_ylim(-0.6, len(order) - 0.4)
    sub.set_yticks(range(len(order)))
    sub.set_yticklabels([T["lab"][m] for m in order] if k == 0 else [""] * len(order), fontsize=5.9)
    sub.set_title(st, fontsize=7.2, loc="left", fontweight="bold", pad=2)
    sub.tick_params(axis="x", labelsize=5.6, length=2); sub.tick_params(axis="y", length=0)
    clean(sub); sub.spines["left"].set_visible(False)
    sub.xaxis.set_major_formatter(FuncFormatter(lambda v, p: dec(f"{v:.2f}")))
    sub.locator_params(axis="x", nbins=4)

# (b) GG3 weights
base_b = HC
fig.text(2 / 144, fy(base_b + HB - 1), T["pb"], fontsize=7.2, fontweight="bold", va="top", color=INK)
x = np.arange(12)
for k, st in enumerate(["Darwin", "Perth"]):
    sub = fig.add_axes([0.075 + k * 0.47, fy(base_b + 7), 0.40, fy(HB - 19)])
    M = np.array([[W[st][m][g] for g in GRP] for m in range(12)])
    M = M / M.sum(1, keepdims=True)
    sub.stackplot(x, M.T, colors=[GCOL[g] for g in GRP], linewidth=0, labels=[T["glab"][g] for g in GRP])
    for m in WET[st]:
        a0, a1 = max(m - 1.5, 0), min(m - 0.5, 11)
        sub.add_patch(Rectangle((a0, 1.005), a1 - a0, 0.03, color=BLUE, clip_on=False, lw=0))
    sub.set_xlim(0, 11); sub.set_ylim(0, 1)
    sub.set_xticks(x); sub.set_xticklabels(list(T["months"]), fontsize=5.6)
    sub.set_yticks([0, 0.5, 1]); sub.tick_params(labelsize=5.6, length=2)
    sub.yaxis.set_major_formatter(FuncFormatter(lambda v, p: dec(f"{v:.1f}")))
    if k == 0:
        sub.set_ylabel(T["wy"], fontsize=6)
    sub.set_title(st, fontsize=7.2, loc="left", fontweight="bold", pad=5)
    clean(sub)
    if k == 1:
        h_, l_ = sub.get_legend_handles_labels()
fig.legend(h_[::-1], l_[::-1], loc="lower center", bbox_to_anchor=(0.5, fy(base_b + 0.3)), ncol=5, fontsize=5.8, frameon=False,
           handlelength=1.2, columnspacing=1.4)

# (c) diagnosis + notes
pc = fig.add_axes([0, 0, 1, fy(HC)]); pc.set_xlim(0, 144); pc.set_ylim(0, HC); pc.axis("off")
pc.text(2, HC - 1.5, T["pc"], fontsize=7.2, fontweight="bold", va="top", color=INK)
cx = [4, 82, 102]
yy = HC - 7.5
for j, hd in enumerate(T["dhead"]):
    pc.text(cx[j], yy, hd, fontsize=6.2, fontweight="bold", va="top", color=INK)
pc.plot([2, 120], [yy - 3.2, yy - 3.2], color=INK2, lw=0.5)
for i, (lab, key, f) in enumerate(T["drows"]):
    yr = yy - 4.6 - i * 3.6
    pc.text(cx[0], yr, lab, fontsize=6.1, va="top", color=INK)
    for j, st in enumerate(["Darwin", "Perth"]):
        pc.text(cx[j + 1], yr, dec(f.format(DIAG[st][key])), fontsize=6.1, va="top", color=INK)
pc.text(2, yy - 17.5, "\n".join(textwrap.wrap(T["note"], 152)), fontsize=6.0, va="top", color=INK2, linespacing=1.45)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig0d_solution_gg_darwin_perth_{LANG}.{ext}"), dpi=600, facecolor="white", bbox_inches="tight")
print("ok")
