"""Improvement figure: the global meta-learner gate (code of the companion study [31]) trained on Darwin and Perth only.

Sources of the numbers:
  - gg2_summary.json       : NSE per station and GG3 weights (h = 1) from the two-station rerun (companion_2stations),
                             plus GG3 on 15 stations from the evaluation results of the companion study;
  - lpa_main_all.json      : P3 with the LSTM meta-learner and the rolling 30-year climatology from the LPA audit (results/lstm/).
Panel (a) uses NSE differences to each setup's own rolling 30-year climatology, because the audit data
(He et al., 1921-2020, interpolated) and the companion data (SILO, months with >= 90% observed days) differ.
Usage:  python make_improvement_2st.py en <output folder> <gg2_summary.json> <lpa_main_all.json>
"""
import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter

LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"
G = json.load(open(sys.argv[3] if len(sys.argv) > 3 else os.path.join("companion_2stations", "gg2_summary.json")))
A = json.load(open(sys.argv[4] if len(sys.argv) > 4 else os.path.join("results", "lstm", "lpa_main_all.json")))
os.makedirs(OUT, exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GREEN, BLUE, ORANGE, GREY = "#008300", "#2a78d6", "#eb6834", "#8c8c8c"
plt.rcParams.update({"font.family": "DejaVu Sans"})
STS = ["Darwin", "Perth"]

p3 = {st: float(np.mean([x["NSE"] for x in A[st]["L4_closed"]["P3"]["seeds"]])) - A[st]["clim_rolling30"]["NSE"] for st in STS}
gap = {}
for st in STS:
    for per in ("2015-2020", "2015-2025"):
        v = G["nse"][f"{per}|{st}"]
        gap[(st, per)] = {"L-G3": v["L-G3"] - v["Clim-30yr"], "GG3": v["GG3"] - v["Clim-30yr"], "GG3_15": v["GG3_15"] - v["Clim-30yr"]}
W = G["weights_h1_test2015_2025"]
GRP = ["SARIMA", "ML", "DL", "Hybrid", "Clim"]
GCOL = {"SARIMA": "#cc79a7", "ML": "#e69f00", "DL": "#56b4e9", "Hybrid": "#009e73", "Clim": "#8c8c8c"}
WET = {"Darwin": [11, 12, 1, 2, 3, 4], "Perth": [5, 6, 7, 8, 9, 10]}

T = {
"en": dict(
 title="Improvement: one gate meta-learner trained on Darwin and Perth together",
 data=("Darwin and Perth", ["SILO daily rainfall, 1926–2025;", "targets: months with ≥ 90%", "directly observed days"]),
 base=("1  Causal base models", ["SARIMA, SVR, XGBoost, LSTM, GRU,", "+ VMD / causal wavelet; retrained", "every 5 yr; indices at o − 2 / o − 1"]),
 pool=("2  Pool of 11 members", ["9 base models +", "expanding and rolling 30-yr", "climatology"]),
 inA=("3a  Error history", ["last 12 months of each", "member, scaled by the", "station's σ; read by an LSTM"]),
 inB=("3b  Context", ["current forecasts, target", "month, ENSO/IOD regime,", "horizon"]),
 inC=("3c  Station identity", ["embedding + static features", "for Darwin and Perth", "(location, rain, season)"]),
 gate=("4  Global gate (GG3)", ["ONE gate for both stations,", "trained on their pooled", "4,528 meta-samples (1962–2009)"]),
 out=("Weighted forecast", ["softmax → convex weights per", "station, month and horizon;", "early stop on 2010–2014"]),
 ev=("5  Evaluation", ["2015–2020 (audit window)", "and 2015–2025; panel DM", "with Holm correction"]),
 pa="(a)  NSE gain over each setup's rolling 30-year climatology (0 = climatology)",
 pb="(b)  Weights of the two-station gate by target month (h = 1, test 2015–2025); blue bar = six wettest months",
 rows={"P3": "Single-station causal ensemble\n(P3, LSTM meta-learner)", "L-G3": "Local gate, one station [31]",
       "GG3": "Global gate, Darwin + Perth", "GG3_15": "Global gate, 15 stations [31]"},
 leg=["test 2015–2020", "test 2015–2025"],
 glab={"SARIMA": "SARIMA", "ML": "ML", "DL": "DL", "Hybrid": "Hybrid", "Clim": "Climatology"},
 wy="Weight", months="JFMAMJJASOND",
 note=("Once leakage is closed, the single-station ensemble falls below climatology at Perth ({p3p}) and only just above it at Darwin ({p3d}). "
       "A gate trained on Darwin and Perth together brings both stations to climatology level ({g2d} and {g2p} in 2015–2020; {g2d5} and {g2p5} in 2015–2025), "
       "but it is not significantly different from the local gate or from climatology (panel DM, Holm p ≥ 0.17). With 13 more stations the same gate gains more at Darwin "
       "({g15d}) [31]. Gains are measured against each setup's own climatology because the audit and the companion study use different data versions."),
 ),
"id": dict(
 title="Peningkatan: satu gate meta-learner dilatih bersama pada Darwin dan Perth",
 data=("Darwin dan Perth", ["curah hujan harian SILO, 1926–2025;", "target: bulan dengan ≥ 90%", "hari teramati langsung"]),
 base=("1  Model dasar kausal", ["SARIMA, SVR, XGBoost, LSTM, GRU,", "+ VMD / wavelet kausal; dilatih", "ulang tiap 5 th; indeks o − 2 / o − 1"]),
 pool=("2  Pool 11 anggota", ["9 model dasar +", "klimatologi ekspansif dan", "bergulir 30 tahun"]),
 inA=("3a  Riwayat galat", ["12 bulan terakhir tiap", "anggota, diskalakan σ", "stasiun; dibaca LSTM"]),
 inB=("3b  Konteks", ["prakiraan saat ini, bulan", "target, rezim ENSO/IOD,", "horizon"]),
 inC=("3c  Identitas stasiun", ["embedding + fitur statis", "untuk Darwin dan Perth", "(lokasi, hujan, musim)"]),
 gate=("4  Gate global (GG3)", ["SATU gate untuk kedua stasiun,", "dilatih pada gabungan", "4.528 sampel meta (1962–2009)"]),
 out=("Prakiraan terbobot", ["softmax → bobot konveks per", "stasiun, bulan, dan horizon;", "early stop pada 2010–2014"]),
 ev=("5  Evaluasi", ["2015–2020 (jendela audit)", "dan 2015–2025; DM panel", "dengan koreksi Holm"]),
 pa="(a)  Tambahan NSE di atas klimatologi bergulir 30 tahun masing-masing (0 = klimatologi)",
 pb="(b)  Bobot gate dua stasiun menurut bulan target (h = 1, uji 2015–2025); garis biru = enam bulan terbasah",
 rows={"P3": "Ensemble kausal satu stasiun\n(P3, meta-learner LSTM)", "L-G3": "Gate lokal, satu stasiun [31]",
       "GG3": "Gate global, Darwin + Perth", "GG3_15": "Gate global, 15 stasiun [31]"},
 leg=["uji 2015–2020", "uji 2015–2025"],
 glab={"SARIMA": "SARIMA", "ML": "ML", "DL": "DL", "Hybrid": "Hibrida", "Clim": "Klimatologi"},
 wy="Bobot", months="JFMAMJJASOND",
 note=("Setelah kebocoran ditutup, ensemble satu stasiun berada di bawah klimatologi di Perth ({p3p}) dan hanya sedikit di atasnya di Darwin ({p3d}). "
       "Gate yang dilatih bersama pada Darwin dan Perth membawa kedua stasiun ke tingkat klimatologi ({g2d} dan {g2p} pada 2015–2020; {g2d5} dan {g2p5} pada 2015–2025), "
       "tetapi tidak berbeda signifikan dari gate lokal maupun klimatologi (DM panel, Holm p ≥ 0,17). Dengan tambahan 13 stasiun, gate yang sama memberi tambahan lebih besar di Darwin "
       "({g15d}) [31]. Tambahan diukur terhadap klimatologi masing-masing karena audit dan studi pendamping memakai versi data yang berbeda."),
 )}[LANG]
dec = (lambda s: s) if LANG == "en" else (lambda s: s.replace(".", ","))
sg = lambda v: dec(f"{v:+.3f}").replace("-", "−")


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


DY = 62          # diagram
HA, HB, HN = 36, 44, 17
TOT = DY + HA + HB + HN
fig = plt.figure(figsize=(7.2, TOT * 0.05))
fy = lambda u: u / TOT
ax = fig.add_axes([0, fy(HA + HB + HN), 1, fy(DY)]); ax.set_xlim(0, 144); ax.set_ylim(0, DY); ax.axis("off")
ax.text(2, DY - 1, T["title"], fontsize=8.4, fontweight="bold", va="top")
y1, h1 = 40, 15
w1 = 44; xb = [2, 50, 98]
box(ax, xb[0], y1, w1, h1, *T["data"], kind="b"); box(ax, xb[1], y1, w1, h1, *T["base"]); box(ax, xb[2], y1, w1, h1, *T["pool"])
arrow(ax, 46, y1 + h1 / 2, 50, y1 + h1 / 2); arrow(ax, 94, y1 + h1 / 2, 98, y1 + h1 / 2)
y2 = 21
box(ax, xb[0], y2, w1, h1, *T["inA"]); box(ax, xb[1], y2, w1, h1, *T["inB"]); box(ax, xb[2], y2, w1, h1, *T["inC"], kind="g")
ax.plot([120, 120], [y1, 38], color=INK2, lw=0.9); ax.plot([24, 120], [38, 38], color=INK2, lw=0.9)
for xx in xb:
    arrow(ax, xx + 22, 38, xx + 22, y2 + h1)
y3 = 2
box(ax, xb[0], y3, w1, h1, *T["gate"], kind="g"); box(ax, xb[1], y3, w1, h1, *T["out"]); box(ax, xb[2], y3, w1, h1, *T["ev"])
ax.plot([24, 120], [19.2, 19.2], color=INK2, lw=0.9)
for xx in xb:
    ax.plot([xx + 22, xx + 22], [y2, 19.2], color=INK2, lw=0.9)
arrow(ax, 24, 19.2, 24, y3 + h1)
arrow(ax, 46, y3 + h1 / 2, 50, y3 + h1 / 2); arrow(ax, 94, y3 + h1 / 2, 98, y3 + h1 / 2)

# (a) gap to climatology
base_a = HB + HN
fig.text(2 / 144, fy(base_a + HA - 1), T["pa"], fontsize=7.2, fontweight="bold", va="top", color=INK)
order = ["P3", "L-G3", "GG3", "GG3_15"]
for k, st in enumerate(STS):
    sub = fig.add_axes([0.25 + k * 0.385, fy(base_a + 5), 0.32, fy(HA - 15)])
    sub.axvline(0, color=ORANGE, lw=0.9, ls="--", zorder=1)
    for i, m in enumerate(order[::-1]):
        c = GREEN if m == "GG3" else (GREY if m == "P3" else BLUE)
        if m == "P3":
            v = p3[st]; sub.scatter(v, i, s=26, color=c, zorder=3, edgecolors="white", linewidths=0.6)
            sub.text(v + 0.003, i, sg(v), fontsize=5.6, va="center", color=INK)
            continue
        v0, v1 = gap[(st, "2015-2020")][m], gap[(st, "2015-2025")][m]
        sub.plot([v0, v1], [i, i], color=GRID, lw=1.6, zorder=1)
        sub.scatter(v0, i, s=26, color=c, zorder=3, edgecolors="white", linewidths=0.6)
        sub.scatter(v1, i, s=26, facecolors="white", edgecolors=c, linewidths=1.1, zorder=3)
        sub.text(max(v0, v1) + 0.003, i, sg(v0), fontsize=5.6, va="center", color=INK, fontweight="bold" if m == "GG3" else "normal")
    sub.set_ylim(-0.6, len(order) - 0.3); sub.set_xlim(-0.055, 0.05)
    sub.set_yticks(range(len(order)))
    sub.set_yticklabels([T["rows"][m] for m in order[::-1]] if k == 0 else [""] * len(order), fontsize=5.8)
    sub.set_title(st, fontsize=7.2, loc="left", fontweight="bold", pad=2)
    sub.tick_params(axis="x", labelsize=5.6, length=2); sub.tick_params(axis="y", length=0)
    clean(sub); sub.spines["left"].set_visible(False)
    sub.xaxis.set_major_formatter(FuncFormatter(lambda v, p: dec(f"{v:+.02f}").replace("-", "−") if abs(v) > 1e-9 else "0"))
    sub.set_xticks([-0.04, -0.02, 0, 0.02, 0.04])
hd = [Line2D([], [], marker="o", ls="", color=GREY, markersize=4.5, label=T["leg"][0]),
      Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=GREY, markersize=4.5, label=T["leg"][1])]
fig.legend(handles=hd, loc="upper right", bbox_to_anchor=(0.98, fy(base_a + HA - 0.5)), ncol=2, fontsize=5.8, frameon=False)

# (b) weights
base_b = HN
fig.text(2 / 144, fy(base_b + HB - 1), T["pb"], fontsize=7.2, fontweight="bold", va="top", color=INK)
x = np.arange(12)
for k, st in enumerate(STS):
    sub = fig.add_axes([0.075 + k * 0.47, fy(base_b + 7), 0.40, fy(HB - 19)])
    Mw = np.array([[W[st][m][g] for g in GRP] for m in range(12)]); Mw = Mw / Mw.sum(1, keepdims=True)
    sub.stackplot(x, Mw.T, colors=[GCOL[g] for g in GRP], linewidth=0, labels=[T["glab"][g] for g in GRP])
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
    h_, l_ = sub.get_legend_handles_labels()
fig.legend(h_[::-1], l_[::-1], loc="lower center", bbox_to_anchor=(0.5, fy(base_b + 0.3)), ncol=5, fontsize=5.8, frameon=False,
           handlelength=1.2, columnspacing=1.4)

pn = fig.add_axes([0, 0, 1, fy(HN)]); pn.set_xlim(0, 144); pn.set_ylim(0, HN); pn.axis("off")
note = T["note"].format(p3d=sg(p3["Darwin"]), p3p=sg(p3["Perth"]), g2d=sg(gap[("Darwin", "2015-2020")]["GG3"]),
                        g2p=sg(gap[("Perth", "2015-2020")]["GG3"]), g2d5=sg(gap[("Darwin", "2015-2025")]["GG3"]),
                        g2p5=sg(gap[("Perth", "2015-2025")]["GG3"]), g15d=sg(gap[("Darwin", "2015-2020")]["GG3_15"]))
pn.text(2, HN - 1.5, "\n".join(textwrap.wrap(note, 152)), fontsize=6.0, va="top", color=INK2, linespacing=1.45)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_improvement_2st_{LANG}.{ext}"), dpi=600, facecolor="white", bbox_inches="tight")
print("ok")
