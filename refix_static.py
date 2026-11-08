"""refix_static.py — regenerate Toronto/Vancouver static caches with the correct
present-day LCZ land use, without re-preprocessing.

Only input channel 51 (static_LU_INDEX) depends on the geo_em; every other channel
(atmospheric forcing + terrain) is identical between static and dynamic. So we:
  1. inject a corrected channel 51 = (LCZ_LU_INDEX - 8.425)/6.773 at inference time,
  2. use the new w2w LCZ geo_em for the urban/rural mask,
and reuse precompute_caches.precompute_split for the single-pass cache write.
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
from ncc_figures import (load_norm_params, get_output_layout, build_layout)
from inference import load_model, pick_device
from adaptive_spatial import detect_grid_shape

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger('refix')

LU_MEAN, LU_STD = 8.425, 6.773          # static_LU_INDEX normalization (verified)
CH51_INDEX = 51
STATIC_GEOEM_DIR = Path(os.environ.get('STATIC_GEOEM_DIR', ''))
STATIC_GEOEM = {c: str(STATIC_GEOEM_DIR / c / 'geo_em.d03_LCZ_params.nc')
                for c in ('Toronto', 'Vancouver')}

# --- channel-51 injection (time-invariant per split) ---
_CH51 = {'arr': None}
_orig_read_input = inference.read_input


def _patched_read_input(npz_path, use_static, static_channels):
    inp = _orig_read_input(npz_path, use_static, static_channels)
    if _CH51['arr'] is not None:
        if inp.shape[0] <= CH51_INDEX:
            raise RuntimeError(f"input has {inp.shape[0]} channels, cannot inject ch{CH51_INDEX}")
        inp[CH51_INDEX] = _CH51['arr']
    return inp


inference.read_input = _patched_read_input

# --- geo_em override: static splits use the NEW LCZ geo_em for the mask ---
_orig_get_geo = pc.get_geo_em_path


def _patched_get_geo(city, ssp, rcp, decade, config_path='config.yaml'):
    if ssp == 'static' and city in STATIC_GEOEM:
        return STATIC_GEOEM[city]
    return _orig_get_geo(city, ssp, rcp, decade, config_path)


pc.get_geo_em_path = _patched_get_geo


def compute_ch51(city):
    with xr.open_dataset(STATIC_GEOEM[city]) as ds:
        lu = ds['LU_INDEX'].values[0].astype(np.float32)
    return ((lu - LU_MEAN) / LU_STD).astype(np.float32)


def _already_fixed(split_out, split):
    """A split is fixed once its urban_rural cache exists with non-NaN urban means."""
    ur = split_out / f'urban_rural_means_{split}.npz'
    if not ur.exists():
        return False
    try:
        with np.load(ur) as d:
            return 'TSK_urban' in d.files and not np.isnan(d['TSK_urban']).all()
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--inference_dir', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--unet_model', required=True)
    ap.add_argument('--cities', nargs='+', required=True)
    ap.add_argument('--rcps', nargs='+', default=['RCP4.5', 'RCP6.0', 'RCP8.5'])
    ap.add_argument('--decades', nargs='+', type=int, default=pc.DECADES)
    ap.add_argument('--batch_size', type=int, default=16)
    args = ap.parse_args()

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

    for city in args.cities:
        ch51 = compute_ch51(city)
        logger.info(f"{city}: new ch51 shape={ch51.shape} range=[{ch51.min():.3f},{ch51.max():.3f}] "
                    f"(urban cells LU>=31 -> ch51>={(31-LU_MEAN)/LU_STD:.3f})")
        for rcp in args.rcps:
            for decade in args.decades:
                split = f"{city}_static_{rcp}_{decade}"
                if not (data_path / f'{split}_preprocessed').exists():
                    logger.warning(f"skip {split}: no preprocessed dir")
                    continue
                split_out = inference_dir / split
                if _already_fixed(split_out, split):
                    logger.info(f"skip {split}: already fixed (non-NaN urban)")
                    continue
                # stale MODIS-based caches -> delete so they are recomputed
                for tag in ('area_means', 'urban_rural_means', 'spatial_mean'):
                    p = split_out / f'{tag}_{split}.npz'
                    if p.exists():
                        p.unlink()
                _CH51['arr'] = ch51
                logger.info(f"=== recompute {split} (inject ch51 + new LCZ mask) ===")
                pc.precompute_split(split, decade, city, 'static', rcp, bundle,
                                    data_path, inference_dir, norm_params, layout,
                                    args.config, want_spatial=(decade == 2100))
    logger.info("=== REFIX COMPLETE ===")


if __name__ == '__main__':
    main()
