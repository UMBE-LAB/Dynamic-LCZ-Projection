import json
import os
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import LinearNDInterpolator

NE_DIR = Path(os.environ.get('NE_DATA', ''))
LCZ_DRIVE = Path(os.environ.get('LCZ_PROJECT_DIR', ''))
SOURCES = ('ne_10m_coastline', 'ne_10m_lakes', 'ne_10m_rivers_lake_centerlines')

_NE = {}
_CACHE = {}


def _load_ne(name):
    if name not in _NE:
        src = NE_DIR / f'{name}.geojson'
        if not src.exists():
            raise SystemExit(f'{src} missing; set NE_DATA to a Natural Earth 10m geojson directory')
        with open(src) as f:
            _NE[name] = json.load(f)
    return _NE[name]


def _iter_rings(geom):
    t = geom['type']; c = geom['coordinates']
    if t == 'LineString':
        yield c
    elif t == 'MultiLineString':
        yield from c
    elif t == 'Polygon':
        yield from c
    elif t == 'MultiPolygon':
        for poly in c:
            yield from poly


def _rings_in_bbox(fc, bbox):
    lo0, lo1, la0, la1 = bbox
    out = []
    for feat in fc['features']:
        g = feat.get('geometry')
        if not g:
            continue
        for ring in _iter_rings(g):
            arr = np.asarray(ring, float)
            if arr.ndim != 2 or arr.shape[1] < 2:
                continue
            lon, lat = arr[:, 0], arr[:, 1]
            if np.any((lon >= lo0) & (lon <= lo1) & (lat >= la0) & (lat <= la1)):
                out.append(arr[:, :2])
    return out


def geoem_for_city(city):
    return LCZ_DRIVE / city / 'Future_LCZ_geo_em' / 'SSP5_2100' / 'geo_em.d03_LCZ_params.nc'


def shoreline_polylines(city, margin=0.05):
    if city in _CACHE:
        return _CACHE[city]
    p = geoem_for_city(city)
    if not p.exists():
        _CACHE[city] = []
        return []
    with xr.open_dataset(p) as ds:
        lat = ds['XLAT_M'].values[0]
        lon = ds['XLONG_M'].values[0]
    H, W = lat.shape
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    interp = LinearNDInterpolator(
        np.column_stack([lon.ravel(), lat.ravel()]),
        np.column_stack([cols.ravel().astype(float), rows.ravel().astype(float)]))
    bbox = (lon.min() - margin, lon.max() + margin, lat.min() - margin, lat.max() + margin)
    polylines = []
    for name in SOURCES:
        for ring in _rings_in_bbox(_load_ne(name), bbox):
            xy = interp(ring[:, 0], ring[:, 1])
            x, y = xy[:, 0], xy[:, 1]
            if np.all(np.isnan(x)):
                continue
            polylines.append((x, y))
    _CACHE[city] = polylines
    return polylines
