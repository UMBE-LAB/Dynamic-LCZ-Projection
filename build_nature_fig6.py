import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import ncc_figures as F

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('FIGURE_OUT', HERE / 'figures'))
OUT.mkdir(parents=True, exist_ok=True)

W_IN = F.NATURE_WIDTH_IN
H_IN = 4.35
FS = F.NATURE_FONT_SIZE
FS_SMALL = 6
LW = 0.5

INK = '#1a1a1a'
ARROW = '#7a8691'
DATA_FC, DATA_EC = '#eef2f6', '#a4b3c1'
PROC_FC, PROC_EC = '#e3f0ec', '#1e9583'
OUT_FC, OUT_EC = '#fdf0e2', '#e78731'
PLUS_FC, PLUS_EC = '#fafbfc', '#ccd4db'
KEY_FC, KEY_EC = '#efe9f4', '#8036a7'

ASPECT = W_IN / H_IN


def box(ax, cx, cy, w, h, text, fc, ec, fs=FS_SMALL):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle='round,pad=0,rounding_size=0.010',
        facecolor=fc, edgecolor=ec, linewidth=LW, zorder=2))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            color=INK, linespacing=1.5, zorder=3)
    return dict(cx=cx, cy=cy, w=w, h=h)


def proc(ax, cx, cy, w, h, n, text):
    b = box(ax, cx, cy, w, h, text, PROC_FC, PROC_EC)
    ax.text(cx - w / 2 + 0.010, cy + h / 2 - 0.013, str(n), ha='left', va='center',
            fontsize=FS_SMALL, color=PROC_EC, fontweight='bold', zorder=3)
    return b


L = lambda b: (b['cx'] - b['w'] / 2, b['cy'])
R = lambda b: (b['cx'] + b['w'] / 2, b['cy'])
T = lambda b: (b['cx'], b['cy'] + b['h'] / 2)
B = lambda b: (b['cx'], b['cy'] - b['h'] / 2)


def head(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=5.0,
                                 linewidth=0.6, color=ARROW, shrinkA=0, shrinkB=0,
                                 zorder=4, joinstyle='miter'))


def seg(ax, pts):
    xs, ys = zip(*pts[:-1])
    if len(pts) > 2:
        ax.plot(xs, ys, color=ARROW, linewidth=0.6, zorder=4,
                solid_capstyle='round', solid_joinstyle='round')
    head(ax, pts[-2], pts[-1])


def straight(ax, p0, p1, gap=0.004):
    (x0, y0), (x1, y1) = p0, p1
    d = gap if x1 > x0 else -gap
    seg(ax, [(x0 + d, y0), (x1 - d, y1)])


def hvh(ax, p0, p1, mx, gap=0.004):
    (x0, y0), (x1, y1) = p0, p1
    d = gap if x1 > x0 else -gap
    seg(ax, [(x0 + d, y0), (mx, y0), (mx, y1), (x1 - d, y1)])


def vhv(ax, p0, p1, my, gap=0.004):
    (x0, y0), (x1, y1) = p0, p1
    d = gap if y1 > y0 else -gap
    seg(ax, [(x0, y0 + d), (x0, my), (x1, my), (x1, y1 - d)])


def label(ax, x, y, letter, title):
    ax.text(x, y, letter, ha='left', va='center',
            fontsize=F.NATURE_PANEL_LABEL, fontweight='bold', color=INK, zorder=6)
    ax.text(x + 0.022, y, title, ha='left', va='center', fontsize=FS,
            fontweight='bold', color=INK, zorder=6)


def build():
    fig = plt.figure(figsize=(W_IN, H_IN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis('off')

    DW, PW, OW = 0.195, 0.175, 0.215
    BH, BH2 = 0.090, 0.080

    # ---- a ---------------------------------------------------------------
    label(ax, 0.006, 0.972, 'a', 'Scenario demand and historical LCZ composition')

    urban = box(ax, 0.125, 0.884, DW, BH, 'Global urban land and\nurban fraction, 2020–2100', DATA_FC, DATA_EC)
    natural = box(ax, 0.125, 0.780, DW, BH, 'Global natural land\ncover, 2020–2100', DATA_FC, DATA_EC)
    p1 = proc(ax, 0.375, 0.832, PW, BH, 1, 'Integration\nby SSP')
    frac = box(ax, 0.620, 0.832, OW, BH, 'Land-category\nfractions by SSP', OUT_FC, OUT_EC)

    hvh(ax, R(urban), L(p1), 0.253)
    hvh(ax, R(natural), L(p1), 0.253)
    straight(ax, R(p1), L(frac))

    hist = box(ax, 0.125, 0.664, DW, BH2, 'Historical LCZ maps\n1986–2016', DATA_FC, DATA_EC)
    p2 = proc(ax, 0.375, 0.664, PW, BH2, 2, 'Markov-chain\nanalysis')
    comp = box(ax, 0.620, 0.664, OW, BH2, 'LCZ composition within\neach land category', OUT_FC, OUT_EC)
    straight(ax, R(hist), L(p2))
    straight(ax, R(p2), L(comp))

    # ---- b ---------------------------------------------------------------
    label(ax, 0.006, 0.576, 'b', 'Spatial allocation: Patch-generating Land Use Simulation (PLUS)')
    ax.add_patch(FancyBboxPatch(
        (0.028, 0.212), 0.950, 0.332,
        boxstyle='round,pad=0,rounding_size=0.012',
        facecolor=PLUS_FC, edgecolor=PLUS_EC, linewidth=LW, zorder=1))

    y_hi, y_lo, y_mid = 0.470, 0.362, 0.416
    p3 = proc(ax, 0.118, y_hi, 0.160, BH2, 3, 'Land-expansion\nanalysis')
    emap = box(ax, 0.320, y_hi, 0.185, BH2, 'LCZ expansion maps', DATA_FC, DATA_EC)
    drv = box(ax, 0.320, y_lo, 0.185, BH2, '25 driving factors', DATA_FC, DATA_EC)
    p4 = proc(ax, 0.505, y_mid, 0.150, BH2, 4, 'Random-forest\nmodel')
    grow = box(ax, 0.700, y_hi, 0.175, BH2, 'Growth probabilities', OUT_FC, OUT_EC)
    dem = box(ax, 0.700, y_lo, 0.175, BH2, 'LCZ demand by SSP', OUT_FC, OUT_EC)
    p5 = proc(ax, 0.888, y_mid, 0.165, BH2, 5, 'Cellular automaton,\nrandom-patch seeding')

    straight(ax, R(p3), L(emap))
    hvh(ax, R(emap), L(p4), 0.421)
    hvh(ax, R(drv), L(p4), 0.421)
    hvh(ax, R(p4), L(grow), 0.600)
    hvh(ax, R(grow), L(p5), 0.797)
    hvh(ax, R(dem), L(p5), 0.797)

    infl = box(ax, 0.370, 0.266, 0.470, 0.064,
               'Quantified influence of each driving factor on each LCZ expansion',
               KEY_FC, KEY_EC)
    vhv(ax, B(p4), T(infl), 0.318)

    # both panel-a outputs set the demand the automaton allocates
    MERGE = 0.845
    for src in (frac, comp):
        seg(ax, [(src['cx'] + src['w'] / 2 + 0.004, src['cy']), (MERGE, src['cy']),
                 (MERGE, 0.298), (dem['cx'], 0.298),
                 (dem['cx'], dem['cy'] - dem['h'] / 2 - 0.004)])

    # ---- c ---------------------------------------------------------------
    label(ax, 0.006, 0.168, 'c', 'Decadal LCZ surfaces, 2030–2100, and 1-km downscaling')

    chain = ['Projected LCZ surfaces\nstatic 2020 and dynamic SSP',
             'Surface parameters\nWUDAPT-to-WRF',
             'CASPER\nU-Net emulator',
             '1-km climate fields\nT$_{2}$, UHI, wind, humidex']
    fcs = [DATA_FC, DATA_FC, KEY_FC, OUT_FC]
    ecs = [DATA_EC, DATA_EC, KEY_EC, OUT_EC]
    cw, cy, ch = 0.206, 0.062, 0.084
    cxs = [0.131, 0.371, 0.611, 0.851]
    prev = None
    for cx, t, fc, ec in zip(cxs, chain, fcs, ecs):
        b = box(ax, cx, cy, cw, ch, t, fc, ec)
        if prev is not None:
            straight(ax, R(prev), L(b))
        prev = b

    for ext in ('png', 'pdf'):
        p = OUT / f'fig6_methodology.{ext}'
        fig.savefig(p, dpi=600, bbox_inches='tight', pad_inches=0.02, facecolor='white')
        print('wrote', p)
    plt.close(fig)


if __name__ == '__main__':
    build()
