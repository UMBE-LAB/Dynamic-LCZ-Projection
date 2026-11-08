"""precompute_caches.py — one inference pass per split, all figure caches at once.

For each split, runs the model once and writes the caches the NCC figures read:
  area_means_{split}.npz        (fig3, table1)  — every decade
  urban_rural_means_{split}.npz (fig4, fig5)    — every decade, needs geo_em
  spatial_mean_{split}.npz      (fig2)          — decade 2100 only

Numerics come from the same helpers ncc_figures uses, so the caches are identical
to running its ensure_* functions separately — just without the redundant second
inference pass. Resumable: any cache that already exists is left untouched.
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from ncc_figures import (
    LEVEL_IDX, denormalize, parse_sample_datetime, var_channel,
    load_urban_mask, load_norm_params, get_output_layout, build_layout,
    get_geo_em_path, _cache_path,
)
from inference import (
    load_model, pick_device, load_split_samples, run_inference,
    get_spatial_transform,
)
from adaptive_spatial import detect_grid_shape

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

URBAN_VARS = ['TSK', 'T2', 'RH']
DECADES = list(range(2030, 2110, 10))
RCP_SSPS = {
    'RCP4.5': ['SSP1', 'SSP2', 'SSP4', 'SSP5'],
    'RCP6.0': ['SSP3', 'SSP4'],
    'RCP8.5': ['SSP3', 'SSP5'],
}


def targets_for(split_dir, split_name, decade, want_urban):
    need = {'area_means': not (split_dir / f'area_means_{split_name}.npz').exists()}
    if want_urban:
        need['urban_rural_means'] = not (split_dir / f'urban_rural_means_{split_name}.npz').exists()
    if decade == 2100:
        need['spatial_mean'] = not (split_dir / f'spatial_mean_{split_name}.npz').exists()
    return {k for k, missing in need.items() if missing}


def precompute_split(split_name, decade, city, ssp, rcp, model_bundle,
                     data_path, inference_dir, norm_params, layout,
                     config_path, want_spatial):
    model, config, device, training_shape, batch_size = model_bundle
    data_cfg = config.get('data', {})
    use_static = data_cfg.get('use_static_features', False)
    static_channels = 2 if use_static else 0

    geo = get_geo_em_path(city, ssp, rcp, decade, config_path)
    urban_mask = None
    if geo is not None and Path(geo).exists():
        urban_mask = load_urban_mask(geo)
        rural_mask = ~urban_mask
        urban_channels = {}
        for vn in URBAN_VARS:
            try:
                urban_channels[vn] = var_channel(layout, vn, LEVEL_IDX)
            except KeyError:
                pass

    sample_files = load_split_samples(data_path, split_name, use_static, static_channels)
    spatial_transform = get_spatial_transform(training_shape, split_name, data_path)

    area_accum = {vn: [] for vn, *_ in layout}
    urban_accum = {vn: [] for vn in (urban_channels if urban_mask is not None else {})}
    rural_accum = {vn: [] for vn in (urban_channels if urban_mask is not None else {})}
    spatial_sum = {}
    dts = []
    count = 0

    for (_, pred), fpath in zip(
            run_inference(model, sample_files, spatial_transform, device,
                          use_static, static_channels, batch_size=batch_size),
            sample_files):
        for var_name, is_3d, start, n_ch in layout:
            ch = start + (min(LEVEL_IDX, n_ch - 1) if is_3d else 0)
            raw = pred[ch]
            denorm = denormalize(raw, var_name, norm_params)
            area_accum[var_name].append(float(denorm.mean()))
            if want_spatial:
                field64 = raw.astype(np.float64)
                if var_name not in spatial_sum:
                    spatial_sum[var_name] = np.zeros_like(field64)
                spatial_sum[var_name] += field64
            if urban_mask is not None and var_name in urban_accum:
                urban_accum[var_name].append(float(denorm[urban_mask].mean()))
                rural_accum[var_name].append(float(denorm[rural_mask].mean()))
        dts.append(parse_sample_datetime(fpath))
        count += 1

    if count == 0:
        return
    dts_str = np.array([dt.isoformat() if dt else '' for dt in dts])
    split_out = Path(inference_dir) / split_name
    split_out.mkdir(parents=True, exist_ok=True)

    area_cache = _cache_path(inference_dir, split_name, 'area_means')
    if not area_cache.exists():
        np.savez_compressed(area_cache, datetimes=dts_str,
                            **{k: np.array(v) for k, v in area_accum.items()})
        logger.info(f"Saved area_means: {split_name}")

    if urban_mask is not None:
        ur_cache = _cache_path(inference_dir, split_name, 'urban_rural_means')
        if not ur_cache.exists():
            save = {'datetimes': dts_str}
            for vn in urban_accum:
                save[f'{vn}_urban'] = np.array(urban_accum[vn])
                save[f'{vn}_rural'] = np.array(rural_accum[vn])
            np.savez_compressed(ur_cache, **save)
            logger.info(f"Saved urban_rural_means: {split_name}")

    if want_spatial and spatial_sum:
        sp_cache = _cache_path(inference_dir, split_name, 'spatial_mean')
        if not sp_cache.exists():
            np.savez_compressed(sp_cache,
                                **{k: (v / count).astype(np.float32)
                                   for k, v in spatial_sum.items()})
            logger.info(f"Saved spatial_mean: {split_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--inference_dir', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--unet_model', required=True)
    ap.add_argument('--cities', nargs='+', required=True)
    ap.add_argument('--batch_size', type=int, default=16)
    args = ap.parse_args()

    data_path = Path(args.data_path)
    inference_dir = Path(args.inference_dir)
    norm_params = load_norm_params(data_path)
    output_vars, output_levels = get_output_layout(data_path)
    layout = build_layout(output_vars, output_levels)
    logger.info(f"Layout: {[(n, '3D' if d else '2D') for n, d, *_ in layout]}")

    device = pick_device()
    logger.info(f"Device: {device} | batch size: {args.batch_size}")
    model, config = load_model(args.unet_model, device)
    training_shape = detect_grid_shape(data_path, 'training')
    model_bundle = (model, config, device, training_shape, args.batch_size)

    jobs = []
    for city in args.cities:
        for rcp, ssps in RCP_SSPS.items():
            for ssp in list(ssps) + ['static']:
                for decade in DECADES:
                    jobs.append((f"{city}_{ssp}_{rcp}_{decade}", decade, city, ssp, rcp))

    done = skipped = 0
    for split_name, decade, city, ssp, rcp in jobs:
        split_pre = data_path / f'{split_name}_preprocessed'
        if not split_pre.exists() or not any(split_pre.glob('sample_*.npz')):
            logger.warning(f"No preprocessed data: {split_name} — skipping")
            skipped += 1
            continue
        split_out = inference_dir / split_name
        want = targets_for(split_out, split_name, decade, want_urban=True)
        if not want:
            skipped += 1
            continue
        logger.info(f"[{done+skipped+1}/{len(jobs)}] {split_name}  -> compute {sorted(want)}")
        precompute_split(split_name, decade, city, ssp, rcp, model_bundle,
                         data_path, inference_dir, norm_params, layout,
                         args.config, want_spatial=(decade == 2100))
        done += 1

    logger.info(f"Done. computed {done} splits, skipped {skipped}.")


if __name__ == '__main__':
    main()
