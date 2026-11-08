import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
QS = ROOT / 'examples' / 'quickstart'
EXPECTED = QS / 'expected_output'
OUTPUT = QS / 'output'

TABLES = [
    'Demo_category_pixel_counts.csv',
    'Demo_lcz_composition_fractions.csv',
    'Demo_lcz_pixel_projections.csv',
]


@pytest.fixture(scope='module')
def pipeline_output():
    subprocess.run([sys.executable, str(QS / 'make_inputs.py')], check=True, cwd=ROOT)
    subprocess.run([
        sys.executable, str(ROOT / 'run_pipeline.py'),
        '--lcz-dir', str(QS / 'input' / 'lcz'),
        '--global-urban-dir', str(QS / 'input' / 'global_urban'),
        '--global-landuse-dir', str(QS / 'input' / 'global_landuse'),
        '--output-dir', str(OUTPUT),
        '--no-validation',
    ], check=True, cwd=ROOT)
    return OUTPUT


@pytest.mark.parametrize('name', TABLES)
def test_matches_expected(pipeline_output, name):
    got = pd.read_csv(pipeline_output / name)
    want = pd.read_csv(EXPECTED / name)
    assert list(got.columns) == list(want.columns)
    assert len(got) == len(want)
    for col in want.columns:
        if pd.api.types.is_numeric_dtype(want[col]):
            pd.testing.assert_series_equal(got[col], want[col], atol=1e-6, rtol=0)
        else:
            assert got[col].tolist() == want[col].tolist()


GRID_PIXELS = 36 * 36
ROUNDING_TOLERANCE = 2  # category counts are split by fraction then rounded per LCZ class


def test_pixel_counts_conserve_domain(pipeline_output):
    df = pd.read_csv(pipeline_output / 'Demo_lcz_pixel_projections.csv')
    totals = df.groupby(['SSP', 'Year'])['Pixel_Count'].sum()
    assert (totals - GRID_PIXELS).abs().max() <= ROUNDING_TOLERANCE


def test_urban_grows_most_under_ssp5(pipeline_output):
    df = pd.read_csv(pipeline_output / 'Demo_lcz_pixel_projections.csv')
    urban = df[df['Category'] == 'Urban Land']
    at = lambda ssp, yr: urban[(urban.SSP == ssp) & (urban.Year == yr)]['Pixel_Count'].sum()
    growth = {s: at(s, 2100) - at(s, 2030) for s in ('SSP1', 'SSP3', 'SSP5')}
    assert growth['SSP5'] == max(growth.values())
    assert growth['SSP3'] == min(growth.values())
