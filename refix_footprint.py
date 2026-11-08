"""refix_footprint.py — recompute urban/rural means over a FIXED 2100 urban
footprint (union of all dynamic SSP 2100 extents) for EVERY split, so Fig 4 (UHI)
and Fig 5 (humidex) compare static vs dynamic over identical cells.

- Fixed mask per city = union of {SSP}_2100 LCZ urban extents (LU_INDEX>=31).
- Toronto/Vancouver STATIC splits also get channel-51 injected (their predictions
  were built from MODIS land use); Montreal static + all dynamic predictions are
  already correct, so no injection there.
- area_means (mask-independent) and spatial_mean (full field) are kept where valid;
  only Toronto/Vancouver static need those recomputed (ch51 fix).
- Resumable: a split with a `.fixed2100_done` marker is skipped.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

import inference
import precompute_caches as pc
from ncc_figures import load_norm_params, get_output_layout, build_layout
from inference import load_model, pick_device
from adaptive_spatial import detect_grid_shape

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger('refix_fp')

LU_MEAN, LU_STD = 8.425, 6.773
CH51_INDEX = 51
LCZ = os.environ.get('LCZ_PROJECT_DIR', '')
# Cities whose STATIC predictions were built from MODIS -> need ch51 injection.
STATIC_GEOEM_DIR = Path(os.environ.get('STATIC_GEOEM_DIR', ''))
STATIC_GEOEM = {c: str(STATIC_GEOEM_DIR / c / 'geo_em.d03_LCZ_params.nc')
                for c in ('Toronto', 'Vancouver')}

# --- channel-51 injection (static-with-wrong-prediction only) ---
_CH51 = {'arr': None}
_orig_read = inference.read_input


def _read(npz, us, sc):
    inp = _orig_read(npz, us, sc)
    if _CH51['arr'] is not None:
        inp[CH51_INDEX] = _CH51['arr']
    return inp


inference.read_input = _read

# --- fixed union-2100 mask override (applies to every split) ---
_MASK = {'arr': None}
_orig_mask = pc.load_urban_mask


def _load_mask(p):
    return _MASK['arr'] if _MASK['arr'] is not None else _orig_mask(p)


pc.load_urban_mask = _load_mask

# --- make the geo path always resolve to an existing file (mask comes from _MASK) ---
_orig_geo = pc.get_geo_em_path


def _geo(city, ssp, rcp, decade, cfg='config.yaml'):
    p = _orig_geo(city, ssp, rcp, decade, cfg)
    if p and Path(p).exists():
        return p
    fb = f"{LCZ}/{city}/Future_LCZ_geo_em/SSP5_2100/geo_em.d03_LCZ_params.nc"
    return fb if Path(fb).exists() else p


pc.get_geo_em_path = _geo


def union_2100_mask(city, rcp_ssps):
    ssps = sorted({s for lst in rcp_ssps.values() for s in lst})
    m = None
    used = []
    for ssp in ssps:
        p = f"{LCZ}/{city}/Future_LCZ_geo_em/{ssp}_2100/geo_em.d03_LCZ_params.nc"
        if not Path(p).exists():
            continue
        with xr.open_dataset(p) as ds:
            um = ds['LU_INDEX'].values[0] >= 31
        m = um if m is None else (m | um)
        used.append(ssp)
    if m is None:
        raise RuntimeError(f"No 2100 geo_em found for {city}")
    logger.info(f"{city}: union-2100 mask from {used} -> {int(m.sum())}/{m.size} urban cells")
    return m


def compute_ch51(city):
    with xr.open_dataset(STATIC_GEOEM[city]) as ds:
        lu = ds['LU_INDEX'].values[0].astype(np.float32)
    return ((lu - LU_MEAN) / LU_STD).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--inference_dir', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--unet_model', required=True)
    ap.add_argument('--cities', nargs='+', required=True)
    ap.add_argument('--decades', nargs='+', type=int, default=None,
                    help='subset of decades (default: all)')
    ap.add_argument('--batch_size', type=int, default=16)
    args = ap.parse_args()
    decades = args.decades if args.decades else pc.DECADES

    data_path = Path(args.data_path)
    inference_dir = Path(args.inference_dir)
    norm_params = load_norm_params(data_path)
    ov, ol = get_output_layout(data_path)
    layout = build_layout(ov, ol)
    device = pick_device()
    logger.info(f"Device: {device} | batch size: {args.batch_size}")
    model, config = load_model(args.unet_model, device)
    training_shape = detect_grid_shape(data_path, 'training')
    bundle = (model, config, device, training_shape, args.batch_size)
    rcp_ssps = pc.RCP_SSPS

    done = skipped = 0
    for city in args.cities:
        fixed_mask = union_2100_mask(city, rcp_ssps)
        ch51_static = compute_ch51(city) if city in STATIC_GEOEM else None
        for rcp, ssps in rcp_ssps.items():
            for ssp in list(ssps) + ['static']:
                for decade in decades:
                    split = f"{split_name(city, ssp, rcp, decade)}"
                    split_out = inference_dir / split
                    if not (data_path / f'{split}_preprocessed').exists():
                        continue
                    marker = split_out / '.fixed2100_done'
                    if marker.exists():
                        skipped += 1
                        continue
                    is_static = (ssp == 'static')
                    # Always recompute urban_rural with the fixed mask.
                    ur = split_out / f'urban_rural_means_{split}.npz'
                    if ur.exists():
                        ur.unlink()
                    # Static Toronto/Vancouver: predictions were wrong -> also redo area/spatial.
                    if is_static and ch51_static is not None:
                        for tag in ('area_means', 'spatial_mean'):
                            p = split_out / f'{tag}_{split}.npz'
                            if p.exists():
                                p.unlink()
                    _MASK['arr'] = fixed_mask
                    _CH51['arr'] = ch51_static if is_static else None
                    logger.info(f"=== {split} (fixed-2100 mask{' + ch51' if _CH51['arr'] is not None else ''}) ===")
                    pc.precompute_split(split, decade, city, ssp, rcp, bundle,
                                        data_path, inference_dir, norm_params, layout,
                                        args.config, want_spatial=(decade == 2100))
                    split_out.mkdir(parents=True, exist_ok=True)
                    marker.touch()
                    done += 1
    logger.info(f"=== FOOTPRINT REFIX COMPLETE (done={done} skipped={skipped}) ===")


def split_name(city, ssp, rcp, decade):
    return f"{city}_{ssp}_{rcp}_{decade}"


if __name__ == '__main__':
    main()
