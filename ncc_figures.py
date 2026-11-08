"""
ncc_figures.py — Generate journal-quality figures for Nature Climate Change submission.

Figures:
  Fig. 2  Spatial DeltaT2 (SSP - Static), rows=cities, cols=SSPs, RCP8.5, 2100 decade
  Fig. 3  Smoothed annual T2 timeseries, rows=cities, cols=RCPs
  Fig. 4  UHI intensity (TSK urban-rural) timeseries, rows=cities, cols=RCPs
  Fig. 5  Compound heat risk index timeseries, rows=cities, cols=RCPs
  Table 1 Cross-city cross-scenario summary statistics

All analysis uses level 0 only.  Caches intermediate results aggressively.
Currently implemented for Montreal; Vancouver and Toronto to be added.

Usage:
  python ncc_figures.py --data_path processed_data_CESM --inference_dir CESM \
      --output_dir paper/figures --unet_model checkpoints/best.pt
"""

import argparse
import json
import logging
import re
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import numpy as np
import torch
from tqdm import tqdm
import yaml

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

LEVEL_IDX = 0  # all analysis at level 0

# ---------------------------------------------------------------------------
# Nature Climate Change figure style
# ---------------------------------------------------------------------------
NATURE_WIDTH_MM    = 183
NATURE_WIDTH_IN    = NATURE_WIDTH_MM / 25.4
NATURE_DPI         = 300
NATURE_FONT_FAMILY = 'Arial'
NATURE_FONT_SIZE   = 7
NATURE_PANEL_LABEL = 8

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   [NATURE_FONT_FAMILY, 'Helvetica', 'DejaVu Sans'],
    'font.size':         NATURE_FONT_SIZE,
    'axes.labelsize':    NATURE_FONT_SIZE,
    'axes.titlesize':    NATURE_FONT_SIZE,
    'xtick.labelsize':   NATURE_FONT_SIZE,
    'ytick.labelsize':   NATURE_FONT_SIZE,
    'legend.fontsize':   6,
    'axes.linewidth':    0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'lines.linewidth':   0.8,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
    'axes.grid':         False,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# IPCC AR6 official colours
SSP_COLORS = {
    'SSP1': '#1e9583', 'SSP2': '#4576be', 'SSP3': '#f11111',
    'SSP4': '#e78731', 'SSP5': '#8036a7',
}
STATIC_COLOR = '#555555'
STATIC_STYLE = '--'
GLOBAL_COLOR = '#888888'
GLOBAL_STYLE = ':'

CITY_COLORS = {
    'Montreal': '#1b9e77', 'Toronto': '#d95f02', 'Vancouver': '#7570b3',
}

ALL_SSPS  = ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']
DECADES   = list(range(2030, 2110, 10))

# WRF LU_INDEX categories >= 31 are urban LCZ types
URBAN_LCZ_MIN = 31

VAR_LABELS = {
    'T':    'Temperature (K)',
    'U':    'U-wind (m s$^{-1}$)',
    'V':    'V-wind (m s$^{-1}$)',
    'RH':   'Relative humidity (%)',
    'T2':   '2-m temperature (K)',
    'TSK':  'Skin temperature (K)',
}


# ===================================================================
# Caching helpers
# ===================================================================

def _cache_path(inference_dir, split_name, tag):
    return Path(inference_dir) / split_name / f'{tag}_{split_name}.npz'


def parse_sample_datetime(npz_path):
    m = re.search(r'sample_(\d{8})_(\d{6})', Path(npz_path).name)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
        except ValueError:
            pass
    return None


# ===================================================================
# Normalization
# ===================================================================

def load_norm_params(data_path):
    candidates = [
        data_path / 'inference_info' / 'normalization_params.json',
        data_path / 'normalization_params.json',
        data_path / 'training_preprocessed' / 'metadata.json',
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                raw = json.load(f)
            return raw.get('normalization_params') or raw.get('normalization_parameters') or raw
    return None


def denormalize(data, var_name, norm_params, source='output'):
    if norm_params is None:
        return data
    params = (norm_params.get(source, {}).get(var_name) or norm_params.get(var_name))
    if params is None:
        return data
    method = params.get('method', 'standard')
    if method in ('standard', 'standardize'):
        return data * params.get('std', 1) + params.get('mean', 0)
    elif method in ('minmax', 'normalize'):
        lo, hi = params.get('min', 0), params.get('max', 1)
        return data * (hi - lo) + lo
    return data


# ===================================================================
# Output layout
# ===================================================================

def get_output_layout(data_path):
    """Return (var_list, output_levels) from var_info.json."""
    for split_dir in sorted(data_path.glob('*_preprocessed')):
        vi = split_dir / 'var_info.json'
        if vi.exists():
            with open(vi) as f:
                info = json.load(f)
            return info.get('output', []), info.get('output_levels', 10)
    raise RuntimeError("No var_info.json found")


def build_layout(output_vars, levels):
    layout, ch = [], 0
    for v in output_vars:
        name, is_3d = v['name'], v.get('is_3d', False)
        n = levels if is_3d else 1
        layout.append((name, is_3d, ch, n))
        ch += n
    return layout


def var_channel(layout, var_name, level_idx=0):
    """Return the channel index for var_name at level_idx."""
    for name, is_3d, start, n_ch in layout:
        if name == var_name:
            return start + (min(level_idx, n_ch - 1) if is_3d else 0)
    raise KeyError(f"Variable {var_name} not in layout")


# ===================================================================
# Model loading and inference (reuse from inference.py)
# ===================================================================

def _setup_inference(args, data_path):
    """Load model, detect shapes, return (model, config, device, training_shape)."""
    from inference import load_model, pick_device
    from adaptive_spatial import detect_grid_shape
    device = pick_device()
    batch_size = getattr(args, 'batch_size', 1)
    logger.info(f"Device: {device}  |  inference batch size: {batch_size}")
    model, config = load_model(args.unet_model, device)
    training_shape = detect_grid_shape(data_path, 'training')
    return model, config, device, training_shape, batch_size


def run_split_inference_spatial(model, data_path, split_name, device,
                                training_shape, config, layout, batch_size=1):
    """
    Run inference on a split, cache and return decade-mean spatial fields
    per variable at level 0.
    Cache: spatial_mean_{split}.npz  with keys = var names, each (H, W).
    """
    from inference import load_split_samples, run_inference, get_spatial_transform

    data_cfg = config.get('data', {})
    use_static = data_cfg.get('use_static_features', False)
    static_channels = 2 if use_static else 0

    sample_files = load_split_samples(data_path, split_name, use_static, static_channels)
    spatial_transform = get_spatial_transform(training_shape, split_name, data_path)

    # Accumulate running mean spatially to avoid memory blow-up
    accum = {}
    count = 0
    for _, pred in run_inference(model, sample_files, spatial_transform,
                                  device, use_static, static_channels,
                                  batch_size=batch_size):
        for var_name, is_3d, start, n_ch in layout:
            ch = start + (min(LEVEL_IDX, n_ch - 1) if is_3d else 0)
            field = pred[ch].astype(np.float64)
            if var_name not in accum:
                accum[var_name] = np.zeros_like(field)
            accum[var_name] += field
        count += 1

    if count == 0:
        return None
    return {k: (v / count).astype(np.float32) for k, v in accum.items()}


def ensure_spatial_mean(split_name, inference_dir, data_path, norm_params,
                        layout, model_bundle=None):
    """
    Load or compute decade-mean spatial fields for a split.
    Returns dict {var_name: np.ndarray (H,W)} denormalized, or None.
    """
    cache = _cache_path(inference_dir, split_name, 'spatial_mean')
    if cache.exists():
        logger.info(f"Loading cached spatial mean: {split_name}")
        with np.load(cache) as d:
            result = {k: d[k] for k in d.files}
        # Denormalize
        return {k: denormalize(v, k, norm_params) for k, v in result.items()}

    # Check if preprocessed data exists
    split_dir = Path(data_path) / f'{split_name}_preprocessed'
    if not split_dir.exists() or not list(split_dir.glob('sample_*.npz')):
        logger.warning(f"No preprocessed data for {split_name}")
        return None

    if model_bundle is None:
        logger.warning(f"No model provided, cannot compute spatial mean for {split_name}")
        return None

    model, config, device, training_shape, batch_size = model_bundle
    logger.info(f"Computing spatial mean for {split_name} ...")
    raw = run_split_inference_spatial(model, data_path, split_name, device,
                                      training_shape, config, layout, batch_size)
    if raw is None:
        return None

    # Save raw (normalized) cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **raw)
    logger.info(f"Saved spatial mean cache: {cache}")

    return {k: denormalize(v, k, norm_params) for k, v in raw.items()}


# ===================================================================
# Area-mean timeseries (reuse cache from inference_ncc.py)
# ===================================================================

def ensure_area_means(split_name, inference_dir, data_path, norm_params,
                      layout, model_bundle=None):
    """
    Load or compute area-mean timeseries for a split at level 0.
    Returns (area_means {var: ndarray(N,)}, datetimes list) or (None, None).
    """
    cache = _cache_path(inference_dir, split_name, 'area_means')
    if cache.exists():
        logger.info(f"Loading cached area means: {split_name}")
        with np.load(cache, allow_pickle=False) as d:
            area_means = {k: d[k] for k in d.files if k != 'datetimes'}
            dts_raw = d.get('datetimes', None)
        if dts_raw is not None:
            dts = [datetime.fromisoformat(s) if s else None for s in dts_raw.tolist()]
        else:
            dts = [None] * len(next(iter(area_means.values())))
        return area_means, dts

    split_dir = Path(data_path) / f'{split_name}_preprocessed'
    if not split_dir.exists():
        return None, None

    sample_files = sorted(split_dir.glob('sample_*.npz'))
    if not sample_files:
        return None, None

    if model_bundle is None:
        logger.warning(f"No model, cannot compute area means for {split_name}")
        return None, None

    model, config, device, training_shape, batch_size = model_bundle
    from inference import load_split_samples, run_inference, get_spatial_transform

    data_cfg = config.get('data', {})
    use_static = data_cfg.get('use_static_features', False)
    static_channels = 2 if use_static else 0

    sample_files = load_split_samples(data_path, split_name, use_static, static_channels)
    spatial_transform = get_spatial_transform(training_shape, split_name, data_path)

    accum = {vn: [] for vn, *_ in layout}
    dts = []

    logger.info(f"Streaming area means for {split_name} ({len(sample_files)} samples) ...")
    for (_, pred), fpath in zip(
            run_inference(model, sample_files, spatial_transform,
                          device, use_static, static_channels,
                          batch_size=batch_size),
            sample_files):
        for var_name, is_3d, start, n_ch in layout:
            ch = start + (min(LEVEL_IDX, n_ch - 1) if is_3d else 0)
            field = pred[ch]
            accum[var_name].append(
                float(denormalize(field, var_name, norm_params).mean()))
        dts.append(parse_sample_datetime(fpath))

    area_means = {k: np.array(v) for k, v in accum.items()}
    dts_str = np.array([dt.isoformat() if dt else '' for dt in dts])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, datetimes=dts_str, **area_means)
    logger.info(f"Saved area means: {cache}")
    return area_means, dts


def concat_decades_area_means(city, ssp, rcp, inference_dir, data_path,
                               norm_params, layout, model_bundle=None):
    """Concatenate area means across all decades for one scenario."""
    merged, all_dt = {}, []
    for decade in DECADES:
        split = f"{city}_{ssp}_{rcp}_{decade}"
        am, dts = ensure_area_means(split, inference_dir, data_path,
                                     norm_params, layout, model_bundle)
        if am is None:
            continue
        for var, arr in am.items():
            merged.setdefault(var, []).append(arr)
        all_dt.extend(dts)
    if not merged:
        return None, None
    return {v: np.concatenate(a) for v, a in merged.items()}, all_dt


# ===================================================================
# UHI: urban/rural mask from geo_em
# ===================================================================

def load_urban_mask(geo_em_path):
    """Return boolean mask (H, W) where True = urban (LU_INDEX >= 31)."""
    import xarray as xr
    # Handle Windows SMB U+F03A path issue
    path_str = str(geo_em_path)
    try:
        ds = xr.open_dataset(path_str)
    except Exception:
        from preprocess import get_openable_path
        ds = xr.open_dataset(get_openable_path(path_str))
    lu = ds['LU_INDEX'].values[0]  # (H, W)
    ds.close()
    return lu >= URBAN_LCZ_MIN


def get_geo_em_path(city, ssp, rcp, decade, config_path='config.yaml'):
    """Look up geo_em path from config.yaml, with filesystem fallback."""
    split_name = f"{city}_{ssp}_{rcp}_{decade}"

    # 1. Try config.yaml (paths live under data.paths)
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        for key in ('paths', 'splits'):
            section = cfg.get('data', {}).get(key, {})
            if split_name in section:
                geo = section[split_name].get('geo_em')
                if geo and Path(geo).exists():
                    return geo
    except Exception:
        pass

    # 2. Filesystem fallback for SSP scenarios
    if ssp != 'static':
        candidates = [
            Path(f'E:/LCZ Projection/{city}/Future_LCZ_geo_em/{ssp}_{decade}/geo_em.d03_LCZ_params.nc'),
            Path(f'E:/LCZ Projection/{city}/Future_LCZ_geo_em/{ssp}_{decade}/geo_em.d03.nc'),
        ]
        for p in candidates:
            if p.exists():
                return str(p)

    # 3. Filesystem fallback for static
    if ssp == 'static':
        candidates = [
            Path(f'G:/montreal_2100_july/geo_em/geo_em.d03.nc'),
            Path(f'G:/{city.lower()}_2100_july/geo_em/geo_em.d03.nc'),
        ]
        for p in candidates:
            if p.exists():
                return str(p)

    return None


def ensure_urban_rural_means(split_name, inference_dir, data_path, norm_params,
                              layout, geo_em_path, model_bundle=None):
    """
    Compute urban and rural area means for TSK, T2, and RH at level 0.
    Cache: urban_rural_means_{split}.npz with keys
      '{var}_urban', '{var}_rural' for each var, plus 'datetimes'.
    Returns dict {var: (urban_arr, rural_arr)}, dts  or  None, None.
    """
    EXTRACT_VARS = ['TSK', 'T2', 'RH']
    cache = _cache_path(inference_dir, split_name, 'urban_rural_means')

    # Also accept legacy cache that only has TSK
    legacy_cache = _cache_path(inference_dir, split_name, 'uhi_means')

    if cache.exists():
        logger.info(f"Loading cached urban/rural means: {split_name}")
        with np.load(cache, allow_pickle=False) as d:
            dts_raw = d.get('datetimes', None)
            result = {}
            for vn in EXTRACT_VARS:
                uk, rk = f'{vn}_urban', f'{vn}_rural'
                if uk in d.files and rk in d.files:
                    result[vn] = (d[uk], d[rk])
        if dts_raw is not None:
            dts = [datetime.fromisoformat(s) if s else None for s in dts_raw.tolist()]
        else:
            n = len(next(iter(result.values()))[0]) if result else 0
            dts = [None] * n
        if result:
            return result, dts

    split_dir = Path(data_path) / f'{split_name}_preprocessed'
    if not split_dir.exists():
        return None, None

    sample_files = sorted(split_dir.glob('sample_*.npz'))
    if not sample_files:
        return None, None

    if model_bundle is None:
        logger.warning(f"No model, cannot compute urban/rural means for {split_name}")
        return None, None

    if geo_em_path is None or not Path(geo_em_path).exists():
        logger.warning(f"No geo_em for {split_name}, cannot compute urban/rural means")
        return None, None

    urban_mask = load_urban_mask(geo_em_path)
    rural_mask = ~urban_mask

    # Resolve channel indices for each variable
    var_channels = {}
    for vn in EXTRACT_VARS:
        try:
            var_channels[vn] = var_channel(layout, vn, LEVEL_IDX)
        except KeyError:
            pass

    model, config, device, training_shape, batch_size = model_bundle
    from inference import load_split_samples, run_inference, get_spatial_transform

    data_cfg = config.get('data', {})
    use_static = data_cfg.get('use_static_features', False)
    static_channels = 2 if use_static else 0

    inf_files = load_split_samples(data_path, split_name, use_static, static_channels)
    spatial_transform = get_spatial_transform(training_shape, split_name, data_path)

    accum_urban = {vn: [] for vn in var_channels}
    accum_rural = {vn: [] for vn in var_channels}
    dts = []

    logger.info(f"Streaming urban/rural means for {split_name} ({len(inf_files)} samples) ...")
    for (_, pred), fpath in zip(
            run_inference(model, inf_files, spatial_transform,
                          device, use_static, static_channels,
                          batch_size=batch_size),
            inf_files):
        for vn, ch in var_channels.items():
            field = denormalize(pred[ch], vn, norm_params)
            accum_urban[vn].append(float(field[urban_mask].mean()))
            accum_rural[vn].append(float(field[rural_mask].mean()))
        dts.append(parse_sample_datetime(fpath))

    # Build result and cache
    save_dict = {}
    result = {}
    for vn in var_channels:
        u = np.array(accum_urban[vn])
        r = np.array(accum_rural[vn])
        result[vn] = (u, r)
        save_dict[f'{vn}_urban'] = u
        save_dict[f'{vn}_rural'] = r

    dts_str = np.array([dt.isoformat() if dt else '' for dt in dts])
    save_dict['datetimes'] = dts_str
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **save_dict)
    logger.info(f"Saved urban/rural means: {cache}")

    # Clean up legacy cache if it exists
    if legacy_cache.exists():
        legacy_cache.unlink(missing_ok=True)

    return result, dts


# ===================================================================
# Heat risk index
# ===================================================================

def humidex(t2_k, rh_pct):
    """Compute Humidex from 2-m temperature (K) and relative humidity (%).
    Returns values in Celsius-equivalent Humidex units."""
    t_c = t2_k - 273.15
    # Dewpoint approximation from RH and T
    a, b = 17.27, 237.7
    gamma = (a * t_c) / (b + t_c) + np.log(np.clip(rh_pct, 1, 100) / 100.0)
    td = (b * gamma) / (a - gamma)
    e = 6.11 * np.exp(5417.7530 * (1/273.16 - 1/(td + 273.15)))
    h = t_c + 0.5555 * (e - 10.0)
    return h


# ===================================================================
# Aggregation helpers
# ===================================================================

def aggregate_annual(series, datetimes, months=None):
    """Annual mean. If months is set (e.g. [6,7,8]), only those months."""
    buckets = defaultdict(list)
    for val, dt in zip(series, datetimes):
        if dt is None:
            continue
        if months is not None and dt.month not in months:
            continue
        buckets[datetime(dt.year, 1, 1)].append(val)
    if not buckets:
        return [], np.array([])
    keys = sorted(buckets)
    return keys, np.array([np.mean(buckets[k]) for k in keys])


def aggregate_annual_exceedance_days(series, datetimes, threshold, months=None):
    """Count unique days per year where ANY timestep has series >= threshold.
    If months is set, only count within those months."""
    # First pass: find which (year, day_of_year) pairs have an exceedance
    exceed_days = defaultdict(set)    # year -> set of day_of_year
    years_seen  = set()
    for val, dt in zip(series, datetimes):
        if dt is None:
            continue
        if months is not None and dt.month not in months:
            continue
        years_seen.add(dt.year)
        if val >= threshold:
            exceed_days[dt.year].add(dt.timetuple().tm_yday)
    if not years_seen:
        return [], np.array([])
    keys = sorted(years_seen)
    counts = np.array([len(exceed_days.get(y, set())) for y in keys], dtype=float)
    date_keys = [datetime(y, 1, 1) for y in keys]
    return date_keys, counts


def aggregate_decade(series, datetimes):
    buckets = defaultdict(list)
    for val, dt in zip(series, datetimes):
        if dt is None:
            continue
        buckets[(dt.year // 10) * 10].append(val)
    if not buckets:
        return [], np.array([]), np.array([])
    keys = sorted(buckets)
    means = np.array([np.mean(buckets[k]) for k in keys])
    stds = np.array([np.std(buckets[k]) for k in keys])
    return keys, means, stds


def gaussian_smooth(y, sigma=2):
    if len(y) < 3:
        return y.copy()
    radius = int(4 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(y, radius, mode='reflect')
    return np.convolve(padded, kernel, mode='same')[radius:-radius]


# ===================================================================
# Percentage annotation helper
# ===================================================================

def compute_pct_range(ys_smooth, g_smooth):
    """Compute min and max % difference of ys_smooth relative to g_smooth.
    Uses the overlapping portion of both arrays (aligned by index).
    Returns (pct_min, pct_max) or None if not computable."""
    n = min(len(ys_smooth), len(g_smooth))
    if n == 0:
        return None
    ys = ys_smooth[:n]
    gs = g_smooth[:n]
    # Avoid division by near-zero
    valid = np.abs(gs) > 0.01
    if not valid.any():
        return None
    pcts = 100.0 * (ys[valid] - gs[valid]) / np.abs(gs[valid])
    return float(pcts.min()), float(pcts.max())


def annotate_pct_ranges(ax, annotations, fontsize=5):
    """Place non-overlapping % range annotations to the right of the plot.
    annotations: list of (label, color, pct_min, pct_max).
    Placed as a compact text block in the upper-right corner."""
    if not annotations:
        return
    lines = []
    for label, color, pmin, pmax in annotations:
        if abs(pmin - pmax) < 0.5:
            # Range is negligible, show single value
            sign = '+' if pmin >= 0 else ''
            lines.append((f'{label}: {sign}{pmin:.1f}%', color))
        else:
            smin = '+' if pmin >= 0 else ''
            smax = '+' if pmax >= 0 else ''
            lines.append((f'{label}: {smin}{pmin:.1f}% to {smax}{pmax:.1f}%', color))

    # Draw as stacked text in upper-right
    for i, (txt, color) in enumerate(lines):
        ax.text(0.98, 0.97 - i * 0.08, txt,
                transform=ax.transAxes, fontsize=fontsize,
                color=color, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.7))


# ===================================================================
# Saving
# ===================================================================

def save_fig(fig, out_path):
    fig.savefig(out_path, dpi=NATURE_DPI, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {out_path} + .pdf")


# ===================================================================
# Fig. 2 — Spatial DeltaT2 (SSP - Static), RCP/SSP paired columns
# ===================================================================

# Default RCP/SSP pairs for Fig. 2 — fills all 5 columns
FIG2_PAIRS = [
    ('RCP4.5', 'SSP1'),
    ('RCP4.5', 'SSP2'),
    ('RCP6.0', 'SSP3'),
    ('RCP6.0', 'SSP4'),
    ('RCP8.5', 'SSP5'),
]

from shoreline import shoreline_polylines


def fig2_spatial_delta(cities, decade, inference_dir, data_path,
                       norm_params, layout, model_bundle, out_dir,
                       pairs=None, config_path='config.yaml', vmax=None):
    """
    Rows = cities, cols = RCP/SSP pairs.
    Each panel: decade-mean T2 (SSP) - T2 (Static for same RCP).
    """
    if pairs is None:
        pairs = FIG2_PAIRS
    n_rows = len(cities)
    n_cols = len(pairs)

    # --- Compute all diff fields first (shared color limits + per-city aspect) ---
    diffs = {}
    shapes = {}
    for r, city in enumerate(cities):
        # Cache static fields per RCP (same RCP may appear in multiple columns)
        static_cache = {}
        for c, (rcp, ssp) in enumerate(pairs):
            if rcp not in static_cache:
                static_split = f"{city}_static_{rcp}_{decade}"
                static_cache[rcp] = ensure_spatial_mean(
                    static_split, inference_dir, data_path,
                    norm_params, layout, model_bundle)
            static_field = static_cache[rcp]
            ssp_split = f"{city}_{ssp}_{rcp}_{decade}"
            ssp_field = ensure_spatial_mean(ssp_split, inference_dir, data_path,
                                            norm_params, layout, model_bundle)
            if ssp_field is not None and static_field is not None:
                if 'T2' in ssp_field and 'T2' in static_field:
                    d = ssp_field['T2'] - static_field['T2']
                    diffs[(r, c)] = d
                    shapes[r] = d.shape

    # Symmetric color limits from robust (98th pct) magnitude, not single-pixel outliers
    if vmax is None:
        if diffs:
            vmax = max(np.nanpercentile(np.abs(d), 98) for d in diffs.values())
            vmax = max(vmax, 0.1)
        else:
            vmax = 1.0

    shorelines = {r: shoreline_polylines(city) for r, city in enumerate(cities)}

    cmap = plt.cm.RdBu_r
    cnorm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

    # --- Manual geometry: row height tracks each city's map aspect ratio so the
    #     gap between cities is uniform, and the colorbar sits one gap below. ---
    fig_w = NATURE_WIDTH_IN
    left_in, right_in = 0.55, 0.12
    top_in = 0.48                      # room for two-line column titles
    hgap_in = 0.05                     # gap between columns
    vgap_in = 0.30                     # uniform gap: between cities AND row->colorbar
    cbar_h_in = 0.12
    cbar_label_in = 0.44               # room below colorbar for ticks + label

    panel_w = (fig_w - left_in - right_in - (n_cols - 1) * hgap_in) / n_cols
    aspects = [(shapes[r][0] / shapes[r][1]) if r in shapes else 1.0
               for r in range(n_rows)]
    row_h = [panel_w * a for a in aspects]

    fig_h = (top_in + sum(row_h) + (n_rows - 1) * vgap_in
             + vgap_in + cbar_h_in + cbar_label_in)

    fig = plt.figure(figsize=(fig_w, fig_h))

    panel_idx = 0
    y_top_in = fig_h - top_in
    for r, city in enumerate(cities):
        ph = row_h[r]
        ax_bottom_in = y_top_in - ph
        for c, (rcp, ssp) in enumerate(pairs):
            x_in = left_in + c * (panel_w + hgap_in)
            ax = fig.add_axes([x_in / fig_w, ax_bottom_in / fig_h,
                               panel_w / fig_w, ph / fig_h])
            if (r, c) in diffs:
                ax.imshow(diffs[(r, c)], origin='lower', cmap=cmap,
                          norm=cnorm, aspect='auto')
                for sx, sy in shorelines.get(r, []):
                    ax.plot(sx, sy, color='#333333', linewidth=0.4)
                ax.set_xlim(-0.5, diffs[(r, c)].shape[1] - 0.5)
                ax.set_ylim(-0.5, diffs[(r, c)].shape[0] - 0.5)
            else:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        ha='center', va='center', fontsize=NATURE_FONT_SIZE,
                        color='#999999')
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f'{ssp}\n({rcp})', fontweight='bold', pad=4)
            if c == 0:
                ax.set_ylabel(city, fontweight='bold')
            label = chr(ord('a') + panel_idx)
            ax.text(0.5, -0.06, label, transform=ax.transAxes,
                    fontsize=NATURE_PANEL_LABEL, fontweight='bold',
                    ha='center', va='top')
            panel_idx += 1
        y_top_in = ax_bottom_in - vgap_in

    # Colorbar: one uniform gap below the last city row
    cbar_bottom_in = y_top_in - cbar_h_in
    cbar_w = 0.6 * fig_w
    cbar_left_in = (fig_w - cbar_w) / 2
    cbar_ax = fig.add_axes([cbar_left_in / fig_w, cbar_bottom_in / fig_h,
                            cbar_w / fig_w, cbar_h_in / fig_h])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=cnorm, cmap=cmap),
                        cax=cbar_ax, orientation='horizontal')
    cbar.set_label('\u0394 2-m temperature (K)')

    save_fig(fig, out_dir / 'fig3_spatial.png')


# ===================================================================
# Fig. 3 — End-of-century morphology bias in urban-heat-island intensity
# ===================================================================

FIG3_BIAS_VARS = [
    ('T2',  'Air UHI bias (K)'),
    ('TSK', 'Surface UHI bias (K)'),
]


def fig3_morphology_bias(cities, rcps, rcp_ssps, inference_dir, data_path,
                         norm_params, layout, model_bundle, out_dir):
    # Rows = variables, cols = RCPs, x-axis = SSP growth gradient.
    # Bars coloured by city: morphology bias = dynamic UHI minus static UHI at 2100.
    variables = [(vn, vl) for vn, vl in FIG3_BIAS_VARS
                 if any(n == vn for n, *_ in layout)]
    n_rows = len(variables)
    n_cols = len(rcps)
    n_cities = len(cities)

    def uhi_2100(city, scenario, rcp, var_name):
        split = f'{city}_{scenario}_{rcp}_2100'
        result, _ = ensure_urban_rural_means(
            split, inference_dir, data_path, norm_params, layout, None, model_bundle)
        if result is None or var_name not in result:
            return np.nan
        u, r = result[var_name]
        return float(np.nanmean(u - r))

    def bias(city, ssp, rcp, var_name):
        return uhi_2100(city, ssp, rcp, var_name) - uhi_2100(city, 'static', rcp, var_name)

    # grid[r][c] = (ssps, {city: [bias per ssp]})
    grid = [[None]*n_cols for _ in range(n_rows)]
    row_vals = [[] for _ in range(n_rows)]

    for r, (var_name, _) in enumerate(variables):
        for c, rcp in enumerate(rcps):
            ssps = list(rcp_ssps.get(rcp, []))
            city_bias = {}
            for city in cities:
                bs = [bias(city, ssp, rcp, var_name) for ssp in ssps]
                city_bias[city] = bs
                row_vals[r].extend([v for v in bs if not np.isnan(v)])
            grid[r][c] = (ssps, city_bias)

    # Per-row shared y-limits across all cities/RCPs
    row_ylim = []
    for vals in row_vals:
        if vals:
            lo, hi = min(0, min(vals)), max(0, max(vals))
            margin = (hi - lo) * 0.15 if hi > lo else 0.2
            row_ylim.append((lo - margin, hi + margin))
        else:
            row_ylim.append((-1, 1))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(NATURE_WIDTH_IN, 1.9 * n_rows),
                             squeeze=False)

    bar_w = 0.8 / max(n_cities, 1)
    panel_idx = 0
    for r, (var_name, var_label) in enumerate(variables):
        for c, rcp in enumerate(rcps):
            ax = axes[r, c]
            ssps, city_bias = grid[r][c]
            x = np.arange(len(ssps))
            for i, city in enumerate(cities):
                offset = (i - (n_cities - 1) / 2) * bar_w
                ax.bar(x + offset, city_bias[city], width=bar_w,
                       color=CITY_COLORS.get(city, '#333333'), alpha=0.9,
                       label=city if (r == 0 and c == 0) else None)
            ax.set_xticks(x)
            ax.set_xticklabels(ssps, rotation=45, ha='right')
            ax.axhline(0, color='#999999', linewidth=0.4)
            ax.set_ylim(row_ylim[r])
            ax.tick_params(direction='out')
            if r == 0:
                ax.set_title(rcp, fontweight='bold', pad=4)
            ax.set_ylabel(var_label if c == 0 else '')

            label = chr(ord('a') + panel_idx)
            ax.text(0.5, -0.32, label, transform=ax.transAxes,
                    fontsize=NATURE_PANEL_LABEL, fontweight='bold',
                    ha='center', va='top')
            panel_idx += 1

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=n_cities,
                   frameon=False, bbox_to_anchor=(0.5, 0.995))

    fig.subplots_adjust(wspace=0.18, hspace=0.5, top=0.92)
    save_fig(fig, out_dir / 'fig3_morphology_bias.png')


# ===================================================================
# Fig. 4 — UHI intensity (TSK) timeseries
# ===================================================================

def fig4_uhi_timeseries(cities, rcps, rcp_ssps, inference_dir, data_path,
                        norm_params, layout, model_bundle, out_dir,
                        config_path='config.yaml', var='TSK', out_name='figS11_uhi_timeseries.png',
                        metric_label='UHI'):
    """Rows = cities, cols = RCPs.  UHI = TSK(urban) - TSK(rural).
    Shared y-axis across all panels, legend on every panel."""
    n_rows = len(cities)
    n_cols = len(rcps)
    sigma = 2

    # --- First pass: collect all smoothed series to find shared y-range ---
    all_series = {}   # (r,c) -> list of (xs, ys_smooth, label, color, style)
    for r, city in enumerate(cities):
        for c, rcp in enumerate(rcps):
            panel_series = []
            ssps = rcp_ssps.get(rcp, [])

            for ssp in ssps:
                all_uhi, all_dt = [], []
                for decade in DECADES:
                    split = f"{city}_{ssp}_{rcp}_{decade}"
                    geo = get_geo_em_path(city, ssp, rcp, decade, config_path)
                    result, dts = ensure_urban_rural_means(
                        split, inference_dir, data_path, norm_params,
                        layout, geo, model_bundle)
                    if result is None or var not in result:
                        continue
                    urb, rur = result[var]
                    all_uhi.extend((urb - rur).tolist())
                    all_dt.extend(dts)
                if not all_uhi:
                    continue
                xs, ys = aggregate_annual(all_uhi, all_dt)
                if xs:
                    panel_series.append((xs, ys, gaussian_smooth(ys, sigma),
                                         ssp, SSP_COLORS.get(ssp, '#333333'), '-'))

            # Static UHI
            all_uhi, all_dt = [], []
            for decade in DECADES:
                split = f"{city}_static_{rcp}_{decade}"
                geo = get_geo_em_path(city, 'static', rcp, decade, config_path)
                result, dts = ensure_urban_rural_means(
                    split, inference_dir, data_path, norm_params,
                    layout, geo, model_bundle)
                if result is None or var not in result:
                    continue
                urb, rur = result[var]
                all_uhi.extend((urb - rur).tolist())
                all_dt.extend(dts)
            if all_uhi:
                xs, ys = aggregate_annual(all_uhi, all_dt)
                if xs:
                    panel_series.append((xs, ys, gaussian_smooth(ys, sigma),
                                         'Static', STATIC_COLOR, STATIC_STYLE))

            all_series[(r, c)] = panel_series

    # Per-city (per-row) y-range: shared across a city's RCP columns, NOT across
    # cities, so each city's variation is legible. Extra top headroom leaves room
    # for the per-panel % annotation block.
    row_ylim = []
    for r in range(n_rows):
        row_smooth = [s for c in range(n_cols)
                      for _, _, s, *_ in all_series.get((r, c), [])]
        if row_smooth:
            rmin = min(s.min() for s in row_smooth)
            rmax = max(s.max() for s in row_smooth)
            span = (rmax - rmin) if rmax > rmin else 1.0
            row_ylim.append((rmin - 0.10 * span, rmax + 0.38 * span))
        else:
            row_ylim.append((0.0, 1.0))

    # --- Second pass: plot ---
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(NATURE_WIDTH_IN, 2.4 * n_rows),
                             squeeze=False)

    panel_idx = 0
    for r, city in enumerate(cities):
        for c, rcp in enumerate(rcps):
            ax = axes[r, c]
            panel = all_series.get((r, c), [])

            # Find Static smoothed as reference for %
            static_smooth = None
            for _, _, ys_s, lbl, _, _ in panel:
                if lbl == 'Static':
                    static_smooth = ys_s
                    break

            pct_annotations = []
            for xs, ys_raw, ys_smooth, lbl, color, style in panel:
                ax.plot(xs, ys_raw, color=color, linestyle=style,
                        linewidth=0.4, alpha=0.25)
                ax.plot(xs, ys_smooth, label=lbl, color=color,
                        linestyle=style)
                # % range relative to Static
                if static_smooth is not None and lbl != 'Static':
                    pct = compute_pct_range(ys_smooth, static_smooth)
                    if pct is not None:
                        pct_annotations.append((lbl, color, pct[0], pct[1]))

            annotate_pct_ranges(ax, pct_annotations)

            ax.set_ylim(row_ylim[r])
            if r == 0:
                ax.set_title(rcp, fontweight='bold', pad=4)
            if c == 0:
                ax.set_ylabel(f'{city}\n{metric_label} intensity (K)')
            else:
                ax.set_ylabel('')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(20))
            # y-limits are shared across a row, so only the left panel is labelled
            ax.tick_params(direction='out', labelleft=(c == 0))
            ax.legend(loc='upper left', frameon=False, handlelength=1.5)
            label = chr(ord('a') + panel_idx)
            ax.text(0.5, -0.18, label, transform=ax.transAxes,
                    fontsize=NATURE_PANEL_LABEL, fontweight='bold',
                    ha='center', va='top')
            panel_idx += 1

    fig.subplots_adjust(wspace=0.08, hspace=0.45)
    save_fig(fig, Path(os.environ.get('FIGURE_OUT', 'figures')) / out_name)


# ===================================================================
# Global (input) area means helper for Fig. 5
# ===================================================================

SUMMER = [6, 7, 8]   # JJA
HUMIDEX_THRESHOLD = 30   # °C — "some discomfort" level


def _get_global_exceedance_days(city, rcp, first_ssp, inference_dir, data_path,
                                 norm_params, layout):
    """Annual summer Humidex exceedance days from input (pre-downscaling) T and RH."""
    from inference_ncc import concat_decades_input
    output_vars = [n for n, *_ in layout]
    out_levels = max((nch for _, is3d, _, nch in layout if is3d), default=10)

    splits = [f"{city}_{first_ssp}_{rcp}_{d}" for d in DECADES]
    input_means, input_dts = concat_decades_input(
        splits, data_path, inference_dir,
        output_vars, out_levels, LEVEL_IDX, norm_params)
    if input_means is None:
        return None, None
    t_key = 'T2' if 'T2' in input_means else 'T'
    rh_key = 'RH'
    if t_key not in input_means or rh_key not in input_means:
        return None, None
    h = humidex(input_means[t_key], input_means[rh_key])
    return aggregate_annual_exceedance_days(h, input_dts, HUMIDEX_THRESHOLD,
                                            months=SUMMER)


# ===================================================================
# Fig. 5 — Summer Humidex exceedance days (JJA, >= 30 C)
# ===================================================================

def fig5_heat_risk(cities, rcps, rcp_ssps, inference_dir, data_path,
                   norm_params, layout, model_bundle, out_dir,
                   config_path='config.yaml'):
    """Rows = cities, cols = RCPs.  Annual count of summer days (JJA) where
    urban-pixel Humidex >= 30 C.  Shared y-axis, legend + % range annotations
    on every panel, Global (input) line included."""
    n_rows = len(cities)
    n_cols = len(rcps)
    sigma = 2

    def _humidex_exceedance_from_urban(city, ssp, rcp, config_path):
        """Compute Humidex exceedance days from urban-only T2 + RH."""
        all_h, all_dt = [], []
        for decade in DECADES:
            split = f"{city}_{ssp}_{rcp}_{decade}"
            geo = get_geo_em_path(city, ssp, rcp, decade, config_path)
            result, dts = ensure_urban_rural_means(
                split, inference_dir, data_path, norm_params,
                layout, geo, model_bundle)
            if result is None or 'T2' not in result or 'RH' not in result:
                continue
            t2_urban = result['T2'][0]   # urban array
            rh_urban = result['RH'][0]
            h = humidex(t2_urban, rh_urban)
            all_h.extend(h.tolist())
            all_dt.extend(dts)
        if not all_h:
            return None, None
        return aggregate_annual_exceedance_days(
            all_h, all_dt, HUMIDEX_THRESHOLD, months=SUMMER)

    # --- First pass: collect all series + global ---
    panel_data = {}
    global_data = {}

    for r, city in enumerate(cities):
        for c, rcp in enumerate(rcps):
            series = []
            ssps = rcp_ssps.get(rcp, [])

            for ssp in ssps:
                xs, ys = _humidex_exceedance_from_urban(
                    city, ssp, rcp, config_path)
                if xs:
                    series.append((xs, ys, gaussian_smooth(ys, sigma),
                                   ssp, SSP_COLORS.get(ssp, '#333333'), '-'))

            # Static
            xs, ys = _humidex_exceedance_from_urban(
                city, 'static', rcp, config_path)
            if xs:
                series.append((xs, ys, gaussian_smooth(ys, sigma),
                               'Static', STATIC_COLOR, STATIC_STYLE))

            # Global (input) — area-mean (no urban mask available for input)
            first_ssp = ssps[0] if ssps else None
            if first_ssp:
                gxs, gys = _get_global_exceedance_days(
                    city, rcp, first_ssp, inference_dir, data_path,
                    norm_params, layout)
                if gxs:
                    gys_s = gaussian_smooth(gys, sigma)
                    series.append((gxs, gys, gys_s,
                                   'Global (input)', GLOBAL_COLOR, GLOBAL_STYLE))
                    global_data[(r, c)] = (gxs, gys_s)

            panel_data[(r, c)] = series

    # Shared y-range across ALL panels
    all_smooth = [s for ps in panel_data.values() for _, _, s, *_ in ps]
    if all_smooth:
        ymin = min(0, min(s.min() for s in all_smooth))
        ymax = max(s.max() for s in all_smooth)
        margin = max((ymax - ymin) * 0.1, 1)
    else:
        ymin, ymax, margin = 0, 10, 1

    # --- Second pass: plot ---
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(NATURE_WIDTH_IN, 2.4 * n_rows),
                             squeeze=False)

    panel_idx = 0
    for r, city in enumerate(cities):
        for c, rcp in enumerate(rcps):
            ax = axes[r, c]
            _, g_smooth = global_data.get((r, c), (None, None))
            panel = panel_data.get((r, c), [])

            pct_annotations = []
            for xs, ys_raw, ys_smooth, lbl, color, style in panel:
                ax.plot(xs, ys_raw, color=color, linestyle=style,
                        linewidth=0.4, alpha=0.25)
                ax.plot(xs, ys_smooth, label=lbl, color=color, linestyle=style)

                # % range relative to Global across all years
                if g_smooth is not None and lbl != 'Global (input)':
                    pct = compute_pct_range(ys_smooth, g_smooth)
                    if pct is not None:
                        pct_annotations.append((lbl, color, pct[0], pct[1]))

            annotate_pct_ranges(ax, pct_annotations)

            ax.set_ylim(ymin - margin, ymax + margin)
            if r == 0:
                ax.set_title(rcp, fontweight='bold', pad=4)
            if c == 0:
                ax.set_ylabel(f'{city}\nDays Humidex \u2265 {HUMIDEX_THRESHOLD}\u00b0C')
            else:
                ax.set_ylabel('')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(20))
            ax.tick_params(direction='out')
            ax.legend(loc='upper left', frameon=False, handlelength=1.5)
            label = chr(ord('a') + panel_idx)
            ax.text(0.5, -0.18, label, transform=ax.transAxes,
                    fontsize=NATURE_PANEL_LABEL, fontweight='bold',
                    ha='center', va='top')
            panel_idx += 1

    fig.subplots_adjust(wspace=0.12, hspace=0.45)
    save_fig(fig, out_dir / 'fig5_humidex_exceedance.png')


# ===================================================================
# Fig. 5 — Nocturnal surface urban heat island
# ===================================================================

NIGHT_UTC = 6   # overnight timestep (≈ 22–02 h local across the study cities)


def fig5_nocturnal_uhi(cities, rcps, rcp_ssps, inference_dir, data_path,
                       norm_params, layout, model_bundle, out_dir):
    # Rows = cities, cols = RCPs. Decade-mean nocturnal surface UHI (skin
    # urban-minus-rural at the overnight timestep) for each SSP and the static baseline.
    def night_series(city, scen, rcp):
        buck = defaultdict(list)
        for decade in DECADES:
            result, dts = ensure_urban_rural_means(
                f'{city}_{scen}_{rcp}_{decade}', inference_dir, data_path,
                norm_params, layout, None, model_bundle)
            if result is None or var not in result:
                continue
            u, r = result['TSK']
            for du, dr, dt in zip(u, r, dts):
                if dt is None or dt.hour != NIGHT_UTC:
                    continue
                buck[(dt.year // 10) * 10].append(du - dr)
        ks = sorted(buck)
        return ks, np.array([np.mean(buck[k]) for k in ks])

    n_rows, n_cols = len(cities), len(rcps)
    store = {}
    row_vals = [[] for _ in cities]
    for r, city in enumerate(cities):
        for c, rcp in enumerate(rcps):
            ser = []
            for scen in list(rcp_ssps.get(rcp, [])) + ['static']:
                ks, ys = night_series(city, scen, rcp)
                if len(ks):
                    ser.append((scen, ks, ys))
                    row_vals[r].extend(ys.tolist())
            store[(r, c)] = ser

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(NATURE_WIDTH_IN, 2.1 * n_rows), squeeze=False)
    panel_idx = 0
    for r, city in enumerate(cities):
        lo, hi = (min(row_vals[r]), max(row_vals[r])) if row_vals[r] else (0, 1)
        margin = (hi - lo) * 0.10 or 0.2
        for c, rcp in enumerate(rcps):
            ax = axes[r][c]
            for scen, ks, ys in store[(r, c)]:
                if scen == 'static':
                    ax.plot(ks, ys, color=STATIC_COLOR, linestyle=STATIC_STYLE,
                            linewidth=1.6, label='Static')
                else:
                    ax.plot(ks, ys, '-', color=SSP_COLORS.get(scen, '#333333'),
                            linewidth=1.4, label=scen)
            ax.set_ylim(lo - margin, hi + margin)
            ax.tick_params(direction='out')
            if r == 0:
                ax.set_title(rcp, fontweight='bold', pad=4)
            if c == 0:
                ax.set_ylabel(f'{city}\nNocturnal surface UHI (K)')
            ax.legend(loc='upper left', frameon=False, fontsize=5.5,
                      handlelength=1.2, ncol=2)
            label = chr(ord('a') + panel_idx)
            ax.text(0.5, -0.16, label, transform=ax.transAxes,
                    fontsize=NATURE_PANEL_LABEL, fontweight='bold', ha='center', va='top')
            panel_idx += 1
    fig.subplots_adjust(left=0.11, right=0.98, top=0.94, bottom=0.06, wspace=0.2, hspace=0.3)
    save_fig(fig, out_dir / 'fig5_nocturnal_uhi.png')


# ===================================================================
# Table 1 — Summary statistics
# ===================================================================

def table1_summary(cities, rcps, rcp_ssps, inference_dir, data_path,
                   norm_params, layout, model_bundle, out_dir):
    """Generate CSV summary table."""
    rows = []
    header = ['City', 'RCP', 'Scenario',
              'T2_2030s (K)', 'T2_2090s (K)', 'DeltaT2 (K)',
              'Humidex_2030s (C)', 'Humidex_2090s (C)', 'DeltaHumidex (C)']

    for city in cities:
        for rcp in rcps:
            scenarios = list(rcp_ssps.get(rcp, [])) + ['static']
            for ssp in scenarios:
                am, dts = concat_decades_area_means(
                    city, ssp, rcp, inference_dir, data_path,
                    norm_params, layout, model_bundle)
                if am is None or 'T2' not in am:
                    continue

                decs, means, _ = aggregate_decade(am['T2'], dts)
                dec_map = dict(zip(decs, means))
                t2_30 = dec_map.get(2030, np.nan)
                t2_90 = dec_map.get(2090, np.nan)
                dt2 = t2_90 - t2_30

                if 'RH' in am:
                    h = humidex(am['T2'], am['RH'])
                    decs_h, means_h, _ = aggregate_decade(h, dts)
                    dec_map_h = dict(zip(decs_h, means_h))
                    h30 = dec_map_h.get(2030, np.nan)
                    h90 = dec_map_h.get(2090, np.nan)
                    dh = h90 - h30
                else:
                    h30 = h90 = dh = np.nan

                rows.append([city, rcp, ssp,
                             f'{t2_30:.2f}', f'{t2_90:.2f}', f'{dt2:.3f}',
                             f'{h30:.2f}', f'{h90:.2f}', f'{dh:.3f}'])

    out_path = out_dir / 'table1_summary.csv'
    with open(out_path, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(row) + '\n')
    logger.info(f"Saved: {out_path}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate Nature Climate Change journal figures')
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--inference_dir', required=True)
    parser.add_argument('--output_dir', default=os.environ.get('FIGURE_OUT', 'figures'))
    parser.add_argument('--unet_model', default=None,
                        help='Model checkpoint (needed for splits without cached results)')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--cities', nargs='+', default=['Montreal'])
    parser.add_argument('--rcps', nargs='+', default=['RCP4.5', 'RCP6.0', 'RCP8.5'])
    parser.add_argument('--rcp_ssps', nargs='+',
                        default=[
                            'RCP4.5:SSP1,SSP2,SSP4,SSP5',
                            'RCP6.0:SSP3,SSP4',
                            'RCP8.5:SSP3,SSP5',
                        ])
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Inference batch size (samples per forward pass). '
                             '1 = original behaviour. Raise it (e.g. 16-32) on a '
                             'GPU with spare memory for a large speedup; results '
                             'are numerically identical.')
    parser.add_argument('--fig2_rcp', default='RCP8.5',
                        help='RCP for spatial difference maps (Fig. 2)')
    parser.add_argument('--fig2_vmax', type=float, default=None,
                        help='Fixed symmetric colour limit (K) for the spatial delta figure')
    parser.add_argument('--fig2_decade', type=int, default=2100,
                        help='Decade for spatial difference maps (Fig. 2)')
    parser.add_argument('--figures', nargs='+',
                        default=['fig2', 'fig3', 'fig4', 'fig5', 'table1'],
                        help='Which outputs to generate')
    args = parser.parse_args()

    data_path = Path(args.data_path)
    inference_dir = Path(args.inference_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    norm_params = load_norm_params(data_path)
    output_vars, output_levels = get_output_layout(data_path)
    layout = build_layout(output_vars, output_levels)
    logger.info(f"Output layout: {[(n, '3D' if d else '2D') for n, d, *_ in layout]}")

    # Parse rcp_ssps
    rcp_ssps = {}
    for token in args.rcp_ssps:
        rcp, ssp_str = token.split(':')
        rcp_ssps[rcp.strip()] = [s.strip() for s in ssp_str.split(',')]

    # Model (lazy — only loaded if needed)
    model_bundle = None
    if args.unet_model:
        model_bundle = _setup_inference(args, data_path)

    if 'fig2' in args.figures:
        logger.info("=== Fig. 2: Spatial Delta T2 ===")
        fig2_spatial_delta(args.cities, args.fig2_decade,
                           inference_dir, data_path, norm_params, layout,
                           model_bundle, out_dir, config_path=args.config,
                           vmax=args.fig2_vmax)

    if 'fig3' in args.figures:
        logger.info("=== Fig. 3: Morphology Bias in UHI Intensity ===")
        fig3_morphology_bias(args.cities, args.rcps, rcp_ssps,
                             inference_dir, data_path, norm_params, layout,
                             model_bundle, out_dir)

    if 'fig4' in args.figures:
        logger.info("=== Fig. 4: UHI Timeseries ===")
        fig4_uhi_timeseries(args.cities, args.rcps, rcp_ssps,
                             inference_dir, data_path, norm_params, layout,
                             model_bundle, out_dir, args.config)

    if 'fig5' in args.figures:
        logger.info("=== Fig. 5: Nocturnal Surface UHI ===")
        fig5_nocturnal_uhi(args.cities, args.rcps, rcp_ssps,
                           inference_dir, data_path, norm_params, layout,
                           model_bundle, out_dir)

    if 'fig5_exposure' in args.figures:
        logger.info("=== SI: Humidex Exceedance Days (urban) ===")
        fig5_heat_risk(args.cities, args.rcps, rcp_ssps,
                        inference_dir, data_path, norm_params, layout,
                        model_bundle, out_dir, args.config)

    if 'table1' in args.figures:
        logger.info("=== Table 1: Summary ===")
        table1_summary(args.cities, args.rcps, rcp_ssps,
                        inference_dir, data_path, norm_params, layout,
                        model_bundle, out_dir)

    logger.info("Done.")


if __name__ == '__main__':
    main()
