"""Improvement figure (manuscript v5: cross-station meta-learner gate), results for Darwin and Perth.

Numbers from Naskah_Meta-Learning_Lintas_Stasiun_IJIES_v5 (Table 5: NSE per station, mean of four horizons, test 2015-2025).
Usage:  python make_solution_v5.py en results/figures   (or "id")
"""
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GREEN, BLUE, ORANGE = "#008300", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans"})

# Table 5 of manuscript v5 (mean NSE over h = 1, 3, 6, 12; test 2015-2025)
RES = {"Darwin": {"Klim30": 0.699, "Rata2": 0.686, "NNLS": 0.681, "L-G3": 0.698, "GG3": 0.708, "GG4-T": 0.699},
       "Perth": {"Klim30": 0.670, "Rata2": 0.647, "NNLS": 0.665, "L-G3": 0.667, "GG3": 0.671, "GG4-T": 0.663}}

T = {
"en": dict(
 title="Improvement: one gate meta-learner trained across stations — Darwin and Perth",
 data=("Darwin and Perth", ["+ 13 other SILO stations used", "only to train the gate;", "monthly, 1926–2025"]),
 prep=("1  Preprocessing", ["targets: months with ≥ 90%", "observed days; Niño3.4, DMI", "at o − 2, SOI at o − 1"]),
 base=("2  Causal base models", ["SARIMA, SVR, XGBoost, LSTM,", "GRU, + VMD / causal wavelet;", "retrained every 5 years"]),
 pool=("3  Pool of 11 members", ["9 base models +", "expanding climatology +", "rolling 30-year climatology"]),
 inA=("4a  Error history", ["12 months × 11 members,", "scaled by station σ;", "read by an LSTM (16 units)"]),
 inB=("4b  Context", ["current forecasts, month,", "ENSO/IOD regime (6),", "horizon one-hot"]),
 inC=("4c  Station identity", ["embedding (4 dims) +", "6 static features (location,", "annual rain, seasonality)"]),
 gate=("5  Global gate (GG3)", ["ONE gate trained on pooled", "samples of 15 stations:", "31,160 vs ≤ 2,264 per station"]),
 ev=("6  Evaluation", ["2015–2025, h = 1, 3, 6, 12;", "vs Klim30, NNLS, average,", "local gate; panel DM + Holm"]),
 out=("Weighted forecast", ["softmax → convex weights", "per station, month, horizon;", "stays within member range"]),
 split=["Meta-training: 1962–2009", "Early stop: 2010–2014", "Test: 2015–2025"],
 why="Why it helps: a gate trained on one station sees ≤ 2,264 meta-samples for about 2,600 parameters; pooling 15 stations gives 31,160, so the weights can be estimated without overfitting.",
 ptitle="NSE at Darwin and Perth (mean of four horizons, test 2015–2025)",
 lab={"Klim30": "Rolling 30-yr climatology", "Rata2": "Simple average", "NNLS": "NNLS stacking",
      "L-G3": "Local gate (one station)", "GG3": "Global gate (15 stations)", "GG4-T": "Transfer gate (station unseen)"},
 note="Gains at these two stations are small (+0.010 and +0.004 NSE over the local gate); across 13 test stations the global gate beat the local gate significantly at h = 1, 3, 6 (panel DM, Holm). NSE here is not comparable with the Darwin–Perth audit (different data period, pool and test window).",
 ),
"id": dict(
 title="Peningkatan: satu gate meta-learner dilatih lintas stasiun — Darwin dan Perth",
 data=("Darwin dan Perth", ["+ 13 stasiun SILO lain yang", "hanya dipakai melatih gate;", "bulanan, 1926–2025"]),
 prep=("1  Prapemrosesan", ["target: bulan dengan ≥ 90%", "hari teramati; Niño3.4, DMI", "pada o − 2, SOI pada o − 1"]),
 base=("2  Model dasar kausal", ["SARIMA, SVR, XGBoost, LSTM,", "GRU, + VMD / wavelet kausal;", "dilatih ulang tiap 5 tahun"]),
 pool=("3  Pool 11 anggota", ["9 model dasar +", "klimatologi ekspansif +", "klimatologi bergulir 30 tahun"]),
 inA=("4a  Riwayat galat", ["12 bulan × 11 anggota,", "diskalakan σ stasiun;", "dibaca LSTM (16 unit)"]),
 inB=("4b  Konteks", ["prakiraan saat ini, bulan,", "rezim ENSO/IOD (6),", "horizon one-hot"]),
 inC=("4c  Identitas stasiun", ["embedding (4 dimensi) +", "6 fitur statis (lokasi,", "hujan tahunan, musim)"]),
 gate=("5  Gate global (GG3)", ["SATU gate dilatih pada sampel", "gabungan 15 stasiun:", "31.160 vs ≤ 2.264 per stasiun"]),
 ev=("6  Evaluasi", ["2015–2025, h = 1, 3, 6, 12;", "vs Klim30, NNLS, rata-rata,", "gate lokal; DM panel + Holm"]),
 out=("Prakiraan terbobot", ["softmax → bobot konveks", "per stasiun, bulan, horizon;", "selalu dalam rentang anggota"]),
 split=["Latih meta: 1962–2009", "Early stop: 2010–2014", "Uji: 2015–2025"],
 why="Mengapa membantu: gate yang dilatih pada satu stasiun hanya melihat ≤ 2.264 sampel meta untuk sekitar 2.600 parameter; menggabungkan 15 stasiun memberi 31.160 sampel, sehingga bobot dapat diestimasi tanpa overfitting.",
 ptitle="NSE di Darwin dan Perth (rata-rata empat horizon, uji 2015–2025)",
 lab={"Klim30": "Klimatologi bergulir 30 th", "Rata2": "Rata-rata sederhana", "NNLS": "Stacking NNLS",
      "L-G3": "Gate lokal (satu stasiun)", "GG3": "Gate global (15 stasiun)", "GG4-T": "Gate transfer (stasiun baru)"},
 note="Peningkatan di dua stasiun ini kecil (+0,010 dan +0,004 NSE atas gate lokal); pada 13 stasiun uji, gate global signifikan lebih baik daripada gate lokal pada h = 1, 3, 6 (DM panel, Holm). NSE di sini tidak dapat dibandingkan langsung dengan audit Darwin–Perth (periode data, pool, dan jendela uji berbeda).",
 )}[LANG]
fmt = (lambda v: f"{v:.3f}") if LANG == "en" else (lambda v: f"{v:.3f}".replace(".", ","))


def box(ax, x, y, w, h, title, lines, kind=None):
    ec = {"g": GREEN, "b": BLUE}.get(kind, INK2)
    fc = {"g": "#e5f4e4", "b": "#e7f1fb"}.get(kind, "white")
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=1.5 if kind else 0.9))
    ax.text(x + 1.4, y + h - 1.6, title, fontsize=7.2, fontweight="bold", va="top", color=INK)
    for i, l in enumerate(lines):
        ax.text(x + 1.4, y + h - 5.4 - i * 2.9, l, fontsize=6.1, va="top", color=INK2)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, mutation_scale=8))


Y0, Y1, PH = 14, 89.5, 50          # diagram body (units), height of the result panel (units)
TOT = (Y1 - Y0) + PH
fig = plt.figure(figsize=(7.2, TOT * 0.05))
ax = fig.add_axes([0, PH / TOT, 1, (Y1 - Y0) / TOT]); ax.set_xlim(0, 144); ax.set_ylim(Y0, Y1); ax.axis("off")
ax.text(2, 88.5, "\n".join(textwrap.wrap(T["title"], 105)), fontsize=8.4, fontweight="bold", va="top")

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

# text panel + results
pa = fig.add_axes([0, 0, 1, PH / TOT]); pa.set_xlim(0, 144); pa.set_ylim(0, PH); pa.axis("off")
pa.text(2, PH - 1, "\n".join(textwrap.wrap(T["why"], 150)), fontsize=6.2, va="top", color=GREEN)
pa.text(2, PH - 8.5, T["ptitle"], fontsize=7.2, fontweight="bold", va="top", color=INK)
order = ["Klim30", "Rata2", "NNLS", "L-G3", "GG4-T", "GG3"]
for k, st in enumerate(["Darwin", "Perth"]):
    sub = fig.add_axes([0.235 + k * 0.40, 13.5 / TOT, 0.33, (PH - 30) / TOT])
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
    for sp in ("top", "right", "left"):
        sub.spines[sp].set_visible(False)
    sub.spines["bottom"].set_color(INK2); sub.spines["bottom"].set_linewidth(0.6)
    if LANG == "id":
        from matplotlib.ticker import FuncFormatter
        sub.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.2f}".replace(".", ",")))
    sub.locator_params(axis="x", nbins=4)
pa.text(2, 6.0, "\n".join(textwrap.wrap(T["note"], 160)), fontsize=5.8, va="top", color=INK2)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig0c_solution_v5_{LANG}.{ext}"), dpi=600, facecolor="white", bbox_inches="tight")
print("ok")
