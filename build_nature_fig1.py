import glob
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import xarray as xr
from PIL import Image

import ncc_figures as F
from shoreline import shoreline_polylines

HERE = Path(__file__).resolve().parent
ASSETS = HERE / 'figure_assets'
OUT = Path(os.environ.get('FIGURE_OUT', HERE / 'figures'))
OUT.mkdir(parents=True, exist_ok=True)


def required(name):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f'set {name}; see the figures section of README.md')
    return Path(value)


LCZ_DIR = required('LCZ_PROJECT_DIR') / 'Montreal' / 'Future_LCZ_geo_em'
DATA = required('CASPER_DATA')
INF = required('CASPER_CACHE')

W_IN, H_IN = F.NATURE_WIDTH_IN, 4.55
FS, FS_S, FS_T = F.NATURE_FONT_SIZE, 6, 5.4
LW = 0.5
INK, MUTED, ARROW = '#1a1a1a', '#5c6670', '#8a949e'
STATIC_C, DYN_C = '#4a4a4a', '#8036a7'
LCZ_BUILT = ['#8c0000', '#d10000', '#ff0000', '#bf4d00', '#ff6600',
             '#ff9955', '#faee05', '#bcbcbc', '#ffccaa', '#555555']
LAND_C, WATER_C = (0.906, 0.937, 0.902), (0.788, 0.867, 0.925)
SSP_RCP = {'SSP1': 'RCP4.5', 'SSP2': 'RCP4.5', 'SSP3': 'RCP6.0',
           'SSP4': 'RCP4.5', 'SSP5': 'RCP4.5'}
_NORM = _LAYOUT = None


def _ctx():
    global _NORM, _LAYOUT
    if _NORM is None:
        _NORM = F.load_norm_params(DATA)
        _LAYOUT = F.build_layout(*F.get_output_layout(DATA))
    return _NORM, _LAYOUT


def _built_mask(split):
    norm, _ = _ctx()
    lup = norm['input']['static_LU_INDEX']
    f = sorted(glob.glob(str(DATA / f'{split}_preprocessed' / 'sample_*.npz')))[0]
    lu = np.rint(np.load(f)['input'][51] * lup['std'] + lup['mean']).astype(int)
    return lu >= 31


def _union_mask(city):
    m = None
    for ssp, rcp in SSP_RCP.items():
        b = _built_mask(f'{city}_{ssp}_{rcp}_2100')
        m = b if m is None else (m | b)
    return m


def _sm(split):
    norm, layout = _ctx()
    return F.ensure_spatial_mean(split, INF, DATA, norm, layout, None)


def _pathway_bias(city):
    D = _union_mask(city)
    return {s: float(_sm(f'{city}_{s}_{r}_2100')['T2'][D].mean()
                     - _sm(f'{city}_static_{r}_2100')['T2'][D].mean())
            for s, r in SSP_RCP.items()}


def lcz_image(path):
    with xr.open_dataset(path) as ds:
        lu, lm = ds['LU_INDEX'].values[0], ds['LANDMASK'].values[0]
    img = np.empty(lu.shape + (3,))
    img[:] = LAND_C
    img[lm == 0] = WATER_C
    for i, c in enumerate(LCZ_BUILT):
        img[lu == 31 + i] = mcolors.to_rgb(c)
    return img


def frame(ax, color, lw=LW):
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True), s.set_linewidth(lw), s.set_color(color)


def tidy(ax):
    ax.tick_params(labelsize=FS_S, width=LW, length=2, pad=1.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_linewidth(LW)


def plab(fig, x, y, letter, title):
    fig.text(x, y, letter, ha='left', va='baseline',
             fontsize=F.NATURE_PANEL_LABEL, fontweight='bold', color=INK)
    fig.text(x + 0.017, y, title, ha='left', va='baseline',
             fontsize=FS, fontweight='bold', color=INK)


def arrow(ov, p0, p1, color=ARROW, lw=0.6, ms=5.0):
    ov.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms,
                                 linewidth=lw, color=color, shrinkA=0, shrinkB=0))


def build():
    fig = plt.figure(figsize=(W_IN, H_IN))
    ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.axis('off')

    # ===================== a : the urban surface ==========================
    plab(fig, 0.006, 0.962, 'a', 'The urban surface: held static, or evolved under each pathway')

    mw, mh, my = 0.2235, 0.247, 0.640
    mx = [0.028, 0.2655, 0.5030, 0.7405]
    specs = [('SSP0_2016', 'Present day, held to 2100', STATIC_C),
             ('SSP1_2100', 'SSP1  compact, 2100', DYN_C),
             ('SSP3_2100', 'SSP3  contraction, 2100', DYN_C),
             ('SSP5_2100', 'SSP5  sprawl, 2100', DYN_C)]
    for x, (scen, t, c) in zip(mx, specs):
        ax = fig.add_axes([x, my, mw, mh])
        img = lcz_image(LCZ_DIR / scen / 'geo_em.d03_LCZ_params.nc')
        ax.imshow(img, origin='lower', aspect='auto', interpolation='nearest')
        for sx, sy in shoreline_polylines('Montreal'):
            ax.plot(sx, sy, color='#3b3b3b', linewidth=0.22)
        ax.set_xlim(-0.5, img.shape[1] - 0.5); ax.set_ylim(-0.5, img.shape[0] - 0.5)
        frame(ax, c)
        ax.set_title(t, fontsize=FS_S, color=INK, pad=2.0)

    ov.plot([mx[0], mx[0] + mw], [0.925, 0.925], color=STATIC_C, lw=0.8)
    ov.text(mx[0] + mw / 2, 0.931, 'STATIC   current practice', ha='center', va='bottom',
            fontsize=FS_T, color=STATIC_C, fontweight='bold')
    ov.plot([mx[1], mx[3] + mw], [0.925, 0.925], color=DYN_C, lw=0.8)
    ov.text((mx[1] + mx[3] + mw) / 2, 0.931, 'DYNAMIC   this study', ha='center', va='bottom',
            fontsize=FS_T, color=DYN_C, fontweight='bold')

    lx, ly, sw = 0.028, 0.606, 0.0135
    ov.text(lx, ly + 0.0075, 'Built LCZ', fontsize=FS_T, color=MUTED, va='center')
    for i, c in enumerate(LCZ_BUILT):
        ov.add_patch(Rectangle((lx + 0.058 + i * sw, ly), sw, 0.015, facecolor=c, lw=0))
        ov.text(lx + 0.058 + i * sw + sw / 2, ly - 0.004, str(i + 1), ha='center',
                va='top', fontsize=4.6, color=MUTED)
    for j, (c, t) in enumerate(((LAND_C, 'natural'), (WATER_C, 'water'))):
        x0 = lx + 0.058 + (len(LCZ_BUILT) + 0.7 + j * 2.9) * sw
        ov.add_patch(Rectangle((x0, ly), sw, 0.015, facecolor=c, edgecolor='#b6bcc2', lw=0.3))
        ov.text(x0 + sw + 0.004, ly + 0.0075, t, ha='left', va='center',
                fontsize=FS_T, color=MUTED)

    # ===================== b : the paired experiment ======================
    plab(fig, 0.006, 0.556, 'b', 'Paired downscaling: only the surface differs')

    axc = fig.add_axes([0.030, 0.372, 0.112, 0.145])
    axc.imshow(Image.open(ASSETS / 'nwp_grid.png')); axc.axis('off')
    ov.text(0.086, 0.360, 'CESM atmosphere\nidentical in both arms', ha='center', va='top',
            fontsize=FS_T, color=INK, linespacing=1.35)

    bx, by, bw, bh = 0.176, 0.410, 0.108, 0.075
    ov.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle='round,pad=0,rounding_size=0.008',
                                facecolor='#efe9f4', edgecolor=DYN_C, linewidth=LW))
    ov.text(bx + bw / 2, by + bh / 2 + 0.013, 'CASPER', ha='center', va='center',
            fontsize=FS_S, fontweight='bold', color=INK)
    ov.text(bx + bw / 2, by + bh / 2 - 0.014, 'U-Net emulator\nof WRF', ha='center',
            va='center', fontsize=FS_T, color=INK, linespacing=1.35)
    arrow(ov, (0.146, 0.4475), (bx - 0.003, 0.4475))
    arrow(ov, (bx + bw / 2, 0.526), (bx + bw / 2, by + bh + 0.003))
    ov.text(bx + bw / 2 - 0.007, 0.508, 'surface from a', ha='right', va='center',
            fontsize=FS_T, color=MUTED, style='italic')

    rx, rw, rh = 0.308, 0.112, 0.034
    for k, (t, c, yy) in enumerate((('1-km fields  static', STATIC_C, 0.455),
                                    ('1-km fields  dynamic', DYN_C, 0.404))):
        ov.add_patch(FancyBboxPatch((rx, yy), rw, rh, boxstyle='round,pad=0,rounding_size=0.006',
                                    facecolor='white', edgecolor=c, linewidth=LW))
        ov.text(rx + rw / 2, yy + rh / 2, t, ha='center', va='center',
                fontsize=FS_T, color=INK)
    arrow(ov, (bx + bw + 0.003, 0.4475), (rx - 0.003, 0.472))
    arrow(ov, (bx + bw + 0.003, 0.4475), (rx - 0.003, 0.421))
    ov.text(rx + rw + 0.014, 0.4475, '−', ha='center', va='center', fontsize=11, color=INK)
    arrow(ov, (rx + rw + 0.026, 0.4475), (0.499, 0.4475))

    # ===================== c : the resulting bias =========================
    plab(fig, 0.486, 0.556, 'c', 'Morphological warming, Montreal 2100')
    axm = fig.add_axes([0.512, 0.350, 0.268, 0.170])
    D = _union_mask('Montreal')
    diff = _sm('Montreal_SSP5_RCP4.5_2100')['T2'] - _sm('Montreal_static_RCP4.5_2100')['T2']
    im = axm.imshow(diff, origin='lower', aspect='auto', cmap='RdBu_r',
                    vmin=-2.5, vmax=2.5, interpolation='nearest')
    for sx, sy in shoreline_polylines('Montreal'):
        axm.plot(sx, sy, color='#3b3b3b', linewidth=0.22)
    axm.set_xlim(-0.5, diff.shape[1] - 0.5); axm.set_ylim(-0.5, diff.shape[0] - 0.5)
    frame(axm, '#b6bcc2')
    cax = fig.add_axes([0.788, 0.350, 0.0085, 0.170])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('SSP5 − static, 2-m air (K)', fontsize=FS_T, labelpad=2)
    cb.ax.tick_params(labelsize=FS_T, width=LW, length=1.8, pad=1.2)
    cb.outline.set_linewidth(LW)
    ov.text(0.646, 0.340, f'footprint mean {diff[D].mean():+.2f} K', ha='center', va='top',
            fontsize=FS_T, color=MUTED)

    # ===================== d : decomposition ==============================
    plab(fig, 0.006, 0.284, 'd', 'Expansion dominates the bias')
    R = json.load(open(ASSETS / 'bias_decomposition.json'))
    axd = fig.add_axes([0.070, 0.070, 0.300, 0.165])
    cities = [r['city'] for r in R]
    x = np.arange(len(cities))
    inten = np.array([r['T2']['intensif_contrib'] for r in R])
    expan = np.array([r['T2']['expansion_contrib'] for r in R])
    axd.bar(x, inten, 0.55, color='#b2182b', lw=0, label='Intensification')
    axd.bar(x, expan, 0.55, bottom=inten, color='#f4a582', lw=0, label='Expansion')
    for i, r in enumerate(R):
        axd.text(i, r['T2']['total_bias'] + 0.014, f"{r['T2']['total_bias']:+.2f}",
                 ha='center', va='bottom', fontsize=FS_T, fontweight='bold', color=INK)
    axd.set_xticks(x); axd.set_xticklabels(cities, fontsize=FS_S)
    axd.set_ylabel('SSP5 warming over the\nbuilt footprint, 2100 (K)', fontsize=FS_S, labelpad=2)
    axd.set_ylim(0, 0.72)
    tidy(axd)
    axd.legend(fontsize=FS_T, frameon=False, loc='upper right', handlelength=0.9,
               borderpad=0.0, labelspacing=0.25, handletextpad=0.4)

    # ===================== e : pathway spread =============================
    plab(fig, 0.436, 0.284, 'e', 'Sign and size follow the pathway')
    axe = fig.add_axes([0.500, 0.070, 0.430, 0.165])
    axe.axhspan(-0.1, 0.1, color='#f0f1f2', zorder=0, lw=0)
    axe.axhline(0, color='#b8bdc2', lw=0.4, zorder=1)
    for ci, city in enumerate(cities):
        pb = _pathway_bias(city)
        if ci:
            axe.axvline(ci - 0.5, color='#e3e5e7', lw=0.4, zorder=0)
        for si, ssp in enumerate(F.ALL_SSPS):
            xx = ci + (si - 2) * 0.16
            axe.plot([xx, xx], [0, pb[ssp]], color=F.SSP_COLORS[ssp], lw=0.8, zorder=2)
            axe.plot([xx], [pb[ssp]], 'o', ms=2.8, color=F.SSP_COLORS[ssp],
                     mew=0, zorder=3, label=ssp if ci == 0 else None)
        print(f'  {city}: ' + ' '.join(f'{s} {pb[s]:+.3f}' for s in F.ALL_SSPS))
    axe.set_xticks(np.arange(len(cities))); axe.set_xticklabels(cities, fontsize=FS_S)
    axe.set_xlim(-0.5, len(cities) - 0.5)
    axe.set_ylim(-0.23, 0.82)
    axe.set_ylabel('Footprint-mean morphological\nwarming at 2100 (K)', fontsize=FS_S, labelpad=2)
    tidy(axe)
    axe.text(len(cities) - 0.44, 0.0, 'detection\nthreshold', fontsize=FS_T, color=MUTED,
             ha='left', va='center', linespacing=1.3, clip_on=False)
    axe.legend(fontsize=FS_T, frameon=False, loc='upper left', ncol=5, handlelength=0.6,
               borderpad=0.0, labelspacing=0.2, handletextpad=0.3, columnspacing=0.7)

    for ext in ('png', 'pdf'):
        p = OUT / f'fig1_overview.{ext}'
        fig.savefig(p, dpi=600, bbox_inches='tight', pad_inches=0.02, facecolor='white')
        print('wrote', p)
    plt.close(fig)


if __name__ == '__main__':
    build()
