"""Data panels for Fig. 1 (figures4papers house style), exported as vector SVG for embedding in draw.io.

Sizes are given in points so 1 pt == 1 draw.io px on the 2x canvas; numbers come from the paper text / original Fig. 1.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent.parent / "panels"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": False,
    "svg.fonttype": "path",  # glyphs as paths: renders identically inside draw.io
})
PT = 1 / 72
RED_STRONG, RED_2 = "#B64342", "#E9A6A1"  # figures4papers comparator family
SSP = {"Static": "#8C8C8C", "SSP1": "#00A087", "SSP2": "#3775BA", "SSP3": "#E64B35",
       "SSP4": "#F39B33", "SSP5": "#7E4FA0"}


def decomposition():
    # SSP5 2100 mean 2-m air warming over the built footprint (K) and intensification share (original Fig. 1)
    cities = ["Montreal", "Toronto", "Vancouver"]
    total = [0.58, 0.38, 0.14]
    share = [0.15, 0.15, 0.07]
    inten = [t * s for t, s in zip(total, share)]
    expan = [t - i for t, i in zip(total, inten)]
    assert all(abs(i + e - t) < 1e-9 for i, e, t in zip(inten, expan, total))

    fig, ax = plt.subplots(figsize=(206 * PT, 212 * PT))
    x = range(len(cities))
    kw = dict(width=0.62, edgecolor="black", linewidth=0.6)
    ax.bar(x, inten, color=RED_STRONG, label="Intensification (existing core)", **kw)
    ax.bar(x, expan, bottom=inten, color=RED_2, label="Expansion (new construction)", **kw)
    for xi, t, i, s in zip(x, total, inten, share):
        ax.text(xi, t + 0.012, f"+{t:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.text(xi, i + 0.008, f"{s:.0%}", ha="center", va="bottom", fontsize=11, color="#7A1F1E")
    ax.set_xticks(list(x), cities, fontsize=11.5)
    ax.tick_params(axis="y", labelsize=11.5)
    ax.set_ylim(0, 0.72)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    # same quantity as the original panel: SSP5 mean warming over the built footprint, 2100
    ax.set_ylabel("Mean ΔT$_{2m}$ (K)", fontsize=11.5)
    fig.text(0.0, 0.985, "Static-form bias, SSP5 2100", ha="left", va="top", fontsize=13, fontweight="bold")
    fig.text(0.0, 0.905, "mean over the built footprint", ha="left", va="top", fontsize=11.5, color="#595959")
    fig.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=11.5, handlelength=1.0,
               handleheight=0.9, borderaxespad=0, labelspacing=0.3)
    fig.subplots_adjust(left=0.20, right=0.99, top=0.80, bottom=0.30)
    fig.savefig(OUT / "decomp.svg", transparent=True)
    plt.close(fig)


def scenario_matrix():
    rows = ["Static", "SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]
    cols = ["4.5", "6.0", "8.5"]
    pairs = {"Static": cols, "SSP1": ["4.5"], "SSP2": ["4.5"], "SSP3": ["6.0", "8.5"],
             "SSP4": ["4.5", "6.0"], "SSP5": ["4.5", "8.5"]}
    n = sum(len(v) for v in pairs.values())
    assert n == 11 and n - 3 == 8  # 8 SSP–RCP combinations + static under 3 RCPs (Methods)

    fig, ax = plt.subplots(figsize=(214 * PT, 140 * PT))
    cell = 1.0
    for r, name in enumerate(rows):
        for c, rcp in enumerate(cols):
            on = rcp in pairs[name]
            ax.add_patch(Rectangle((c + 0.08, r + 0.08), cell - 0.16, cell - 0.16,
                                   facecolor=SSP[name] if on else "white",
                                   edgecolor=SSP[name] if on else "#BFBFBF", linewidth=0.6))
    ax.set_xlim(0, 3)
    ax.set_ylim(6, 0)
    ax.set_aspect("auto")  # wider-than-tall cells keep the RCP labels apart
    ax.set_xticks([0.5, 1.5, 2.5], cols, fontsize=11.5)
    ax.set_yticks([r + 0.5 for r in range(6)], rows, fontsize=12)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("CESM forcing (RCP)", fontsize=12)
    ax.xaxis.set_label_position("top")
    fig.text(0.62, 0.40, f"Filled: run\n({n} pairs)\nOpen: not run", ha="left", va="center",
             fontsize=11.5, color="#595959", linespacing=1.25)
    fig.subplots_adjust(left=0.19, right=0.58, top=0.76, bottom=0.0)
    fig.savefig(OUT / "matrix.svg", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    decomposition()
    scenario_matrix()
    print("ok", sorted(p.name for p in OUT.iterdir()))
