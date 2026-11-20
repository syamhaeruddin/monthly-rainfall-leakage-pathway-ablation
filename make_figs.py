"""Stage 4: figures of the MethodsX article (LPA protocol). Output: PNG 600 dpi + vector PDF in results/figures.

Usage:  python make_figs.py results/lpa_main_all.json en     (or "id" for Indonesian labels)
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "figures"); os.makedirs(OUT, exist_ok=True)
A = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "lpa_main_all.json")))
SURR = os.path.join(HERE, "results", "surrogate")
LANG = sys.argv[2] if len(sys.argv) > 2 else "en"

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
STN = ["Darwin", "Perth"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 600})
T = {"en": dict(nse="NSE (test period 2015–2020)", clim="Climatology 1921–2014", clim30="Rolling 30-year climatology",
                cfg=["P0\n2001–2020", "P0, MLP\nsees test", "P0-72", "P1", "P2", "P3"],
                open=["L1 L2 L3", "L1 L2 L3", "L2 L3", "L3", "–", "–"], openlab="open:",
                path={"L1": "L1 Meta-learner\noverlap", "L2": "L2 Contemporaneous\ninput", "L3": "L3 Full-series\ndecomposition"},
                phi="Shapley contribution to inflated NSE", surr="NSE on surrogate series (no interannual signal)",
                scfg=["P0, MLP\nsees test", "P0-72", "P1", "P2", "P3"], climband="climatology, 5th–95th percentile"),
     "id": dict(nse="NSE (periode uji 2015–2020)", clim="Klimatologi 1921–2014", clim30="Klimatologi bergulir 30 tahun",
                cfg=["P0\n2001–2020", "P0, MLP\nmelihat uji", "P0-72", "P1", "P2", "P3"],
                open=["L1 L2 L3", "L1 L2 L3", "L2 L3", "L3", "–", "–"], openlab="terbuka:",
                path={"L1": "L1 Tumpang tindih\nmeta-learner", "L2": "L2 Masukan\nkontemporer", "L3": "L3 Dekomposisi\nseluruh deret"},
                phi="Sumbangan Shapley terhadap NSE yang menggelembung", surr="NSE pada deret surogat (tanpa sinyal antartahun)",
                scfg=["P0, MLP\nmelihat uji", "P0-72", "P1", "P2", "P3"], climband="klimatologi, persentil 5–95")}[LANG]
fmt = (lambda v: f"{v:.2f}") if LANG == "en" else (lambda v: f"{v:.2f}".replace(".", ","))


def ms(lst, m="NSE"):
    v = np.array([x[m] for x in lst]); return v.mean(), v.std(ddof=1)


def style(ax, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(length=2.5)


def localize(fig):
    from matplotlib.ticker import FuncFormatter, ScalarFormatter
    if LANG != "id":
        return
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if isinstance(axis.get_major_formatter(), ScalarFormatter):
                axis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}".replace(".", ",").replace("-", "−")))


def save(fig, name):
    localize(fig)
    fig.savefig(os.path.join(OUT, f"{name}_{LANG}.png"), bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(OUT, f"{name}_{LANG}.pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ladder_values(st):
    R = A[st]
    rows = [ms(R["P0"]["P0_240"]), ms(R["P0"]["P0_72_L1open"]), ms(R["P0"]["P0_72"]),
            ms(R["factorial"]["L1=0,L2=0,L3=1"]["seeds"]), ms(R["factorial"]["L1=0,L2=0,L3=0"]["seeds"]), ms(R["P3"]["seeds"])]
    return np.array(rows), R["clim_fixed"]["NSE"], R["clim_rolling30"]["NSE"]


# ---------------------------------------------------------------- Fig. 2 ladder
def fig_ladder():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    x = np.arange(6)
    for ax, st in zip(axes, STN):
        v, c, c30 = ladder_values(st)
        ax.axhline(c, color=INK2, lw=0.9, ls=(0, (4, 2)), zorder=1)
        ax.axhline(c30, color=ORANGE, lw=0.9, ls=(0, (1, 1.5)), zorder=1)
        ax.errorbar(x, v[:, 0], yerr=v[:, 1], fmt="o", ms=4.5, color=BLUE, ecolor=BLUE, elinewidth=1, capsize=0, zorder=3)
        ax.plot(x[2:], v[2:, 0], color=BLUE, lw=0.8, alpha=0.5, zorder=2)
        for xi, (m, s) in zip(x, v):
            ax.text(xi + 0.14, m, fmt(m), va="center", ha="left", fontsize=6.8, color=INK)
        ax.set_xticks(x); ax.set_xticklabels([f"{c}\n[{o}]" for c, o in zip(T["cfg"], T["open"])], fontsize=6.2)
        ax.set_title(st, fontsize=8.5, loc="left", fontweight="bold")
        ax.set_ylim(-0.32, 1.02); ax.set_xlim(-0.5, 5.75)
        style(ax)
        ax.axvline(0.5, color=GRID, lw=0.8, zorder=0)
    axes[0].set_ylabel(T["nse"])
    axes[1].text(5.72, A["Perth"]["clim_rolling30"]["NSE"] + 0.03, T["clim30"], ha="right", va="bottom", fontsize=6.2, color=ORANGE)
    axes[1].text(5.72, A["Perth"]["clim_fixed"]["NSE"] - 0.03, T["clim"], ha="right", va="top", fontsize=6.2, color=INK2)
    fig.text(0.5, -0.02, ("[ ] = pathways left open" if LANG == "en" else "[ ] = jalur yang masih terbuka"), fontsize=6.2, color=INK2, ha="center", va="top")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig2_ladder")


# ---------------------------------------------------------------- Fig. 3 Shapley
def fig_shapley():
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    paths = ["L2", "L3", "L1"]; y = np.arange(3)[::-1]; h = 0.34
    for k, (st, col) in enumerate(zip(STN, [BLUE, ORANGE])):
        R = A[st]; sh = np.array([[d[p] for p in paths] for d in R["shapley_by_seed"]])
        mean = np.array([R["shapley"][p] for p in paths]); sd = sh.std(0, ddof=1)
        yy = y + (h / 2 + 0.02) * (1 if k == 0 else -1)
        ax.barh(yy, mean, height=h, color=col, edgecolor="white", linewidth=0.8, label=st, zorder=2)
        ax.errorbar(mean, yy, xerr=sd, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=0, zorder=3)
        for yi, m, s in zip(yy, mean, sd):
            ax.text(m + s + 0.012, yi, fmt(m), va="center", fontsize=6.6, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([T["path"][p] for p in paths], fontsize=7)
    ax.set_xlabel(T["phi"], fontsize=7); ax.set_xlim(0, 0.55)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=2.5)
    ax.legend(frameon=False, fontsize=7, loc="lower right", handlelength=1.2)
    fig.tight_layout()
    save(fig, "fig3_shapley")


# ---------------------------------------------------------------- Fig. 4 surrogates
def load_surr(st, kind):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(SURR, f"{st}_{kind}_*.json")))]


def fig_surrogate():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    rng = np.random.default_rng(0)
    for ax, st in zip(axes, STN):
        F, S = load_surr(st, "fast"), load_surr(st, "step")
        cl = np.array([r["clim_fixed"] for r in F])
        lo, hi = np.percentile(cl, [5, 95])
        ax.axhspan(lo, hi, color=GRID, alpha=0.9, lw=0, zorder=0)
        ax.axhline(np.median(cl), color=INK2, lw=0.9, ls=(0, (4, 2)), zorder=1)
        series = [[r["P0_72_L1open"]["NSE"] for r in F], [r["P0_72"]["NSE"] for r in F], [r["P1"]["NSE"] for r in F],
                  [r["P2"]["NSE"] for r in S], [r["P3"]["NSE"] for r in S]]
        for i, v in enumerate(series):
            v = np.array(v); vv = np.clip(v, -0.5, 1.0)
            jit = rng.uniform(-0.17, 0.17, len(v)) if len(v) > 10 else np.linspace(-0.12, 0.12, len(v))
            ax.scatter(i + jit, vv, s=7 if len(v) > 10 else 16, color=BLUE, alpha=0.45 if len(v) > 10 else 0.9,
                       edgecolors="white", linewidths=0.4, zorder=2)
            ax.plot([i - 0.26, i + 0.26], [np.median(v)] * 2, color=INK, lw=1.4, zorder=3)
            n_clip = int(np.sum(v < -0.5))
            if n_clip:
                ax.text(i, -0.47, f"+{n_clip} < −0.5", ha="center", va="bottom", fontsize=5.6, color=INK2)
            ax.text(i, 1.035, f"n={len(v)}", ha="center", va="bottom", fontsize=5.8, color=INK2)
        ax.set_xticks(range(5)); ax.set_xticklabels(T["scfg"], fontsize=6.6)
        ax.set_ylim(-0.5, 1.1); ax.set_xlim(-0.6, 4.6)
        ax.set_title(st, fontsize=8.5, loc="left", fontweight="bold", pad=10)
        style(ax)
        ax.axvline(2.5, color=GRID, lw=0.8, zorder=0)
    axes[0].set_ylabel(T["surr"], fontsize=7)
    fig.text(0.5, -0.02, ("Grey band and dashed line: " if LANG == "en" else "Pita abu-abu dan garis putus-putus: ") + T["climband"]
             + ("; black bar: median." if LANG == "en" else "; garis hitam: median."), fontsize=6.4, color=INK2, ha="center", va="top")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig4_surrogate")


# ---------------------------------------------------------------- Fig. 1 workflow
def box(ax, x, y, w, h, title, lines, accent=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                fc=(BLUE + "1f") if accent else "white", ec=BLUE if accent else INK2, lw=1.3 if accent else 0.8))
    ax.text(x + 0.012, y + h - 0.012, title, fontsize=7.2, fontweight="bold", va="top", color=INK)
    for i, l in enumerate(lines):
        ax.text(x + 0.012, y + h - 0.042 - i * 0.022, l, fontsize=6.2, va="top", color=INK2)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, mutation_scale=7))


def fig_workflow():
    W = {"en": [("1  Fix models & window", ["same models, splits, seeds", "test window 2015–2020 for all"]),
                ("2  Define pathways", ["L1 meta-learner overlap", "L2 contemporaneous input", "L3 full-series decomposition",
                                         "L4 latency  ·  L5 post-hoc design"]),
                ("3  Factorial runs", ["all 2^k open/closed combinations", "× 5 seeds"]),
                ("4  Attribute", ["Shapley φ per pathway", "two-way interactions"]),
                ("P3  Matched features", ["stepwise decomposition at", "every training origin", "Δleak = P0 − P3",
                                           "Δmismatch = P3 − P2"]),
                ("5–6  Causal features", ["re-decompose at each origin", "edge-effect metric E_G"]),
                ("7  Evaluate", ["fixed + rolling climatology", "DM test (HLN, Holm, panel)", "null test on surrogates"])],
         "id": [("1  Tetapkan model", ["model, pembagian, seed sama", "jendela uji 2015–2020"]),
                ("2  Definisikan jalur", ["L1 tumpang tindih meta-learner", "L2 masukan kontemporer", "L3 dekomposisi seluruh deret",
                                           "L4 latensi  ·  L5 pascauji"]),
                ("3  Run faktorial", ["semua 2^k kombinasi buka/tutup", "× 5 seed"]),
                ("4  Atribusi", ["Shapley φ per jalur", "interaksi dua arah"]),
                ("P3  Fitur setara", ["dekomposisi bertahap pada", "setiap titik awal latih", "Δbocor = P0 − P3",
                                       "Δtaksesuai = P3 − P2"]),
                ("5–6  Fitur kausal", ["dekomposisi ulang tiap titik awal", "metrik efek tepi E_G"]),
                ("7  Evaluasi", ["klimatologi tetap + bergulir", "uji DM (HLN, Holm, panel)", "uji nol pada surogat"])]}[LANG]
    fig = plt.figure(figsize=(7.0, 2.24)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 0.32); ax.axis("off")
    bw, bh = 0.228, 0.135
    xs = [0.008, 0.256, 0.504, 0.752]; yt, yb = 0.175, 0.01
    for i in range(4):
        box(ax, xs[i], yt, bw, bh, *W[i])
    box(ax, xs[1], yb, bw, bh, *W[5]); box(ax, xs[2], yb, bw, bh, *W[4], accent=True); box(ax, xs[3], yb, bw, bh, *W[6])
    for i in range(3):
        arrow(ax, xs[i] + bw, yt + bh / 2, xs[i + 1], yt + bh / 2)
    arrow(ax, xs[1] + bw / 2, yt, xs[1] + bw / 2, yb + bh)
    arrow(ax, xs[1] + bw, yb + bh / 2, xs[2], yb + bh / 2)
    arrow(ax, xs[2] + bw, yb + bh / 2, xs[3], yb + bh / 2)
    arrow(ax, xs[3] + bw / 2, yt, xs[3] + bw / 2, yb + bh)
    save(fig, "fig1_workflow")


# ---------------------------------------------------------------- Graphical abstract (1328 x 531 px minimum)
def graphical_abstract():
    lab = {"en": dict(t="Closing leakage pathways removes the apparent skill of a CEEMDAN ensemble",
                      a=["Original\ndesign", "close\nL1", "+ close\nL2", "+ close\nL3", "+ matched\nfeatures"],
                      sh="Shapley share of\ninflated NSE", clim="best climatology",
                      p={"L2": "L2 contemporaneous\ninput", "L3": "L3 full-series\ndecomposition", "L1": "L1 meta-learner\noverlap"}),
           "id": dict(t="Menutup jalur kebocoran menghapus kinerja semu ensemble CEEMDAN",
                      a=["Rancangan\nawal", "tutup\nL1", "+ tutup\nL2", "+ tutup\nL3", "+ fitur\nsetara"],
                      sh="Porsi Shapley NSE\nyang menggelembung", clim="klimatologi terbaik",
                      p={"L2": "L2 masukan\nkontemporer", "L3": "L3 dekomposisi\nseluruh deret", "L1": "L1 tumpang tindih\nmeta-learner"})}[LANG]
    fig = plt.figure(figsize=(13.28 / 2.5, 5.31 / 2.5))
    ax = fig.add_axes([0.09, 0.2, 0.50, 0.6])
    x = np.arange(5)
    for st, col, mk in zip(STN, [BLUE, ORANGE], ["o", "s"]):
        v, c, c30 = ladder_values(st)
        vals = v[[0, 2, 3, 4, 5], 0]
        ax.plot(x, vals, color=col, lw=1.4, marker=mk, ms=4.5, label=st, zorder=3)
        ax.text(4.12, vals[-1], st, color=col, fontsize=7, va="center", fontweight="bold")
        ax.axhline(max(c, c30), color=col, lw=0.7, ls=(0, (3, 2)), alpha=0.8, zorder=1)
    ax.text(4.75, max(A["Darwin"]["clim_fixed"]["NSE"], A["Darwin"]["clim_rolling30"]["NSE"]) + 0.03, ha="right",
            s=
lab["clim"], fontsize=6, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(lab["a"], fontsize=6.2); ax.set_ylabel("NSE", fontsize=7)
    ax.set_ylim(0, 1); ax.set_xlim(-0.4, 4.8); style(ax)
    ax2 = fig.add_axes([0.80, 0.2, 0.18, 0.6])
    paths = ["L2", "L3", "L1"]; y = np.arange(3)[::-1]
    for k, (st, col) in enumerate(zip(STN, [BLUE, ORANGE])):
        m = [A[st]["shapley"][p] for p in paths]
        ax2.barh(y + (0.19 if k == 0 else -0.19), m, height=0.36, color=col, edgecolor="white", lw=0.6)
    ax2.set_yticks(y); ax2.set_yticklabels([lab["p"][p] for p in paths], fontsize=6); ax2.set_xlim(0, 0.45)
    ax2.set_xlabel(lab["sh"], fontsize=6.2); ax2.tick_params(labelsize=6, length=2)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    fig.text(0.09, 0.95, lab["t"], fontsize=7.6, fontweight="bold", va="top")
    localize(fig)
    fig.savefig(os.path.join(OUT, f"graphical_abstract_{LANG}.png"), dpi=500, facecolor="white")
    fig.savefig(os.path.join(OUT, f"graphical_abstract_{LANG}.pdf"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_workflow(); fig_ladder(); fig_shapley(); fig_surrogate(); graphical_abstract()
    print(sorted(os.listdir(OUT)))
