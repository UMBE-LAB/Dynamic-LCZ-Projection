import shutil
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from pyproj import Transformer
from rasterio.transform import from_origin

HERE = Path(__file__).resolve().parent
INPUT = HERE / 'input'

SEED = 7
NAME = 'Demo'
HIST_YEARS = [1996, 2006, 2016]
PROJ_YEARS = list(range(2030, 2101, 10))
SSPS = [1, 2, 3, 4, 5]

LCZ_CRS = 'EPSG:32618'
LCZ_ORIGIN = (595000.0, 5064000.0)
LCZ_RES = 1000.0
LCZ_SHAPE = (36, 36)

BUILT, FOREST, SHRUB, CROP, BARE, WATER = 3, 11, 13, 14, 15, 17
URBAN_PFT, BARREN_PFT = 31, 32


def lcz_map(rng, urban_radius, n_patches):
    h, w = LCZ_SHAPE
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full(LCZ_SHAPE, CROP, dtype=np.uint8)
    img[(xx + yy) % 17 < 3] = FOREST
    img[(xx * 2 + yy) % 23 < 2] = SHRUB
    img[np.abs(yy - 0.45 * xx - 6) < 1.2] = WATER

    core = np.hypot(yy - h / 2, xx - w / 2) < urban_radius
    img[core] = BUILT
    for _ in range(n_patches):
        cy, cx = rng.integers(3, h - 3), rng.integers(3, w - 3)
        img[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2] = BUILT
    img[img == BUILT] = rng.choice([2, 3, 5, 6, 8], size=int((img == BUILT).sum()),
                                   p=[0.1, 0.3, 0.25, 0.25, 0.1])
    img[(xx < 2) | (yy < 2)] = BARE
    return img


def write_lcz():
    out = INPUT / 'lcz'
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    transform = from_origin(LCZ_ORIGIN[0], LCZ_ORIGIN[1], LCZ_RES, LCZ_RES)
    for i, year in enumerate(HIST_YEARS):
        img = lcz_map(rng, urban_radius=4.0 + 1.6 * i, n_patches=6 + 5 * i)
        with rasterio.open(out / f'{NAME}_LCZ{year}.tif', 'w', driver='GTiff',
                           height=LCZ_SHAPE[0], width=LCZ_SHAPE[1], count=1,
                           dtype='uint8', crs=LCZ_CRS, transform=transform) as dst:
            dst.write(img, 1)
    return transform


def wgs84_bounds(transform):
    h, w = LCZ_SHAPE
    xs = [transform.c, transform.c + w * LCZ_RES]
    ys = [transform.f - h * LCZ_RES, transform.f]
    t = Transformer.from_crs(LCZ_CRS, 'EPSG:4326', always_xy=True)
    lons, lats = zip(*[t.transform(x, y) for x in xs for y in ys])
    return min(lats), max(lats), min(lons), max(lons)


def write_urban(bounds):
    min_lat, max_lat, min_lon, max_lon = bounds
    t = Transformer.from_crs('EPSG:4326', 'EPSG:6933', always_xy=True)
    (x0, y0), (x1, y1) = t.transform(min_lon, min_lat), t.transform(max_lon, max_lat)
    pad, res = 20000.0, 1000.0
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    w, h = int((x1 - x0) / res), int((y1 - y0) / res)
    transform = from_origin(x0, y1, res, res)
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(yy - h / 2, xx - w / 2)

    # SSP1 densifies, SSP3 contracts, SSP5 sprawls; SSP2 and SSP4 sit between
    growth = {1: 0.18, 2: 0.30, 3: -0.12, 4: 0.24, 5: 0.55}
    for ssp in SSPS:
        d = INPUT / 'global_urban' / f'SSP{ssp}'
        d.mkdir(parents=True, exist_ok=True)
        for year in PROJ_YEARS:
            frac = (year - 2020) / 80.0
            radius = max(2.0, 10.0 * (1.0 + growth[ssp] * frac))
            img = np.ones((h, w), dtype=np.uint8)
            img[dist < radius] = 2
            with rasterio.open(d / f'global_SSP{ssp}_{year}.tif', 'w', driver='GTiff',
                               height=h, width=w, count=1, dtype='uint8',
                               crs='EPSG:6933', transform=transform) as dst:
                dst.write(img, 1)


def write_landuse(bounds):
    min_lat, max_lat, min_lon, max_lon = bounds
    pad, res = 0.4, 0.05
    lats = np.arange(min_lat - pad, max_lat + pad, res)
    lons = np.arange(min_lon - pad, max_lon + pad, res)
    yy, xx = np.meshgrid(np.arange(len(lats)), np.arange(len(lons)), indexing='ij')
    dist = np.hypot(yy - len(lats) / 2, xx - len(lons) / 2)

    out = INPUT / 'global_landuse'
    out.mkdir(parents=True, exist_ok=True)
    urban_growth = {1: 0.18, 2: 0.30, 3: -0.12, 4: 0.24, 5: 0.55}
    for ssp in SSPS:
        for year in PROJ_YEARS:
            frac = (year - 2020) / 80.0
            urban = np.clip(0.35 * np.exp(-dist / 4.0) * (1 + urban_growth[ssp] * frac), 0, 0.9)
            forest = np.clip(0.30 * (1 - urban), 0, 1)
            crops = np.clip(0.40 * (1 - urban), 0, 1)
            grass = np.clip(1 - urban - forest - crops - 0.02, 0, 1)
            data = {}
            for pft in range(33):
                if pft == URBAN_PFT:
                    arr = urban
                elif pft == BARREN_PFT:
                    arr = np.full_like(urban, 0.01)
                elif pft == 0:
                    arr = np.full_like(urban, 0.01)
                elif 1 <= pft <= 8:
                    arr = forest / 8.0
                elif 9 <= pft <= 14:
                    arr = grass / 6.0
                else:
                    arr = crops / 16.0
                data[f'PFT{pft:02d}'] = (('lat', 'lon'), arr.astype('float32'))
            ds = xr.Dataset(data, coords={'lat': lats, 'lon': lons})
            ds.to_netcdf(out / f'GCAM_Demeter_LU_ssp{ssp}_demo_{year}.nc')
            ds.close()


def main():
    if INPUT.exists():
        shutil.rmtree(INPUT)
    transform = write_lcz()
    bounds = wgs84_bounds(transform)
    write_urban(bounds)
    write_landuse(bounds)
    n = sum(1 for _ in INPUT.rglob('*') if _.is_file())
    print(f'wrote {n} input files to {INPUT}')


if __name__ == '__main__':
    main()
