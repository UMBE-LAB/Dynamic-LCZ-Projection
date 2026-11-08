# Dynamic LCZ Projection

Projects how the built surface of a city changes to 2100 under each Shared Socioeconomic
Pathway. It then downscales climate onto every projected surface at 1 km.

Urban climate projections almost always hold the built surface at its present-day state,
across horizons over which that surface changes substantially. This repository removes
that assumption. It evolves the surface under each pathway, downscales the same
atmosphere onto both the evolving and the static surface, and takes the difference. That
difference isolates the climate signal of urban form alone.

## Quick start

This runs the whole of stage 1 on synthetic input, with no download and no external
dataset. It takes about 5 seconds.

```bash
pip install -r requirements.txt
bash examples/quickstart/run_example.sh
pytest tests
```

See `examples/quickstart/README.md` for what it writes and what the numbers mean.

## Stage 1: projecting the surface

The surface is described as Local Climate Zones, a standard typology of built and natural
surface types. The projection runs in four steps and one optional step.

| Step | Module | What it does |
|---|---|---|
| 0 | `data_integration.detect_study_area` | Reads the study area name, coordinate system, grid, bounds and historical years from the input LCZ files. That grid becomes the reference everything else is reprojected onto. |
| 1 | `data_integration.CategoryPixelCounter` | Crops global land projections to the study area, reprojects them onto the reference grid, and counts pixels per land category for each pathway and year. |
| 2 | `lcz_composition.LCZCompositionProjector` | Measures the LCZ composition of each land category in the historical maps, then projects each fraction forward with a three-member ensemble. |
| 3 | `lcz_projector.LCZPixelProjector` | Multiplies category pixel counts by composition fractions to give LCZ pixel counts per pathway and year. |
| 4 | `transition_matrix.TransitionAnalyzer` | Optional. Computes pixel-by-pixel Markov transition probabilities between LCZ classes from the historical maps. |

The step 2 ensemble combines a damped trend-momentum projection, a drift-and-volatility
random walk with mean reversion, and a diminishing-trend pattern continuation. The
weights adapt to each series. A trending series leans on momentum, and a volatile series
leans on the random walk. The walk takes its seed from `PipelineConfig.ensemble_seed`, so
runs reproduce.

The study area, its grid, its coordinate system and its historical years all come from
the input LCZ files. To run a new city, point the pipeline at a new directory.

```bash
# The study area comes from the LCZ directory
python run_pipeline.py --lcz-dir Vancouver_LCZs

# Name the global datasets and the output directory
python run_pipeline.py \
  --lcz-dir Vancouver_LCZs \
  --global-urban-dir /data/Global_Urban \
  --global-landuse-dir /data/Global_LU \
  --output-dir Vancouver_lcz_projection_results

# Skip the optional transition matrix and the validation imagery
python run_pipeline.py --lcz-dir Vancouver_LCZs --skip-transition --no-validation
```

`config.py` holds the defaults for the dataset paths, the pathway list, the projection
years, the category groupings and the ensemble seed. Override them on the command line,
or build a `PipelineConfig` and call `run_pipeline.run` directly.

Results go to `<StudyArea>_lcz_projection_results/`.

| File | Contents |
|---|---|
| `<name>_category_pixel_counts.csv` | pixels per land category, per pathway and year |
| `<name>_lcz_composition_fractions.csv` | LCZ fractions within each category, historical and projected |
| `<name>_lcz_pixel_projections.csv` | the headline result: LCZ pixel counts per pathway and year |
| `<name>_lcz_pixel_projections.xlsx` | the same, pivoted one sheet per pathway, with summary and metadata |
| `<name>_lcz_transition_matrix.xlsx` | transition probabilities, persistence and the largest off-diagonal moves |
| `*_plots.png`, `*_heatmaps.png` | composition, projection and transition figures |
| `validation_maps/` | category rasters, urban overlays and three-panel checking images per pathway and year |

Counts are in grid cells. Each cell is 1 km across in the published runs, so a count is
also an area in square kilometres. Fractions are unitless and sum to one within a land
category. Nothing in that directory is tracked here. Rerun the pipeline to regenerate it.

## Stage 2: downscaling onto the projected surfaces

Stage 1 produces the surfaces. These scripts downscale climate onto them with CASPER, a
U-Net emulator of the WRF model.

Run `run_precompute.sh` to process the full matrix of cities, pathways, forcing levels
and decades. It makes one inference pass per scenario and writes the cached fields that
the analysis reads. The run is resumable. It skips any scenario whose caches already
exist, and it restarts itself after a crash.

Two repair passes follow the main run. `refix_static.py` rebuilds the static-baseline
caches after a correction to the present-day land use. `refix_footprint.py` re-evaluates
every cache on a fixed 2100 urban footprint, so that a growing urban extent does not
dilute the urban mean. Both also read `STATIC_GEOEM_DIR`, which holds the corrected
present-day geo_em file per city, and `refix_footprint.py` reads `LCZ_PROJECT_DIR`.

These three scripts import `inference` and `adaptive_spatial` from the CASPER repository,
which is linked at the end of this file. Put that repository on `PYTHONPATH` before you
run them, and install its requirements. The `requirements.txt` here covers stage 1 only.

The three drivers read their paths from the environment. Set `CASPER_DATA` to the
preprocessed forcing directory and `CASPER_MODEL` to the trained checkpoint. Both are
required and the driver stops if either is missing. `CASPER_CACHE`, `CASPER_CONFIG`,
`CASPER_CITIES` and `CASPER_BATCH` have defaults.

```bash
export CASPER_DATA=/path/to/processed_data_CESM
export CASPER_MODEL=/path/to/CASPER-CESM.pt
bash run_precompute.sh
```

The trained checkpoint is 7.35 GB. It is too large for version control and is archived
separately.

## Reproducing the paper figures

`make_figures.sh` regenerates Figures 2 to 5 and the supplementary figures and tables from
the cached fields, through `ncc_figures.py`.

```bash
pip install -r requirements.txt -r requirements-figures.txt
export CASPER_DATA=/path/to/processed_data_CESM
export CASPER_CACHE=/path/to/downscaling/caches
export CASPER_MODEL=/path/to/CASPER-CESM.pt
export LCZ_PROJECT_DIR=/path/to/lcz/projection
export NE_DATA=/path/to/natural_earth_geojson
bash make_figures.sh
```

Every path comes from the environment and the driver stops with a named variable if one
is missing. Figures go to `figures/`, or to `FIGURE_OUT` if you set it. The coastlines
need the 10 m `coastline`, `lakes` and `rivers_lake_centerlines` geojson files from
[Natural Earth](https://www.naturalearthdata.com/downloads/10m-physical-vectors/), which
are too large to keep here.

Figures 1 and 6 are schematics rather than plots, and they are authored in draw.io rather
than in Python. `figures_1_and_6/` holds the editable sources. Open `figures_1_and_6.drawio`
in diagrams.net; it has one page per figure. `make_panels.py` regenerates the two vector
analysis panels that the schematics embed, and `NOTES.txt` records the design decisions
and two caveats worth reading: the decomposition bar widths are computed from the rounded
values the paper reports rather than from raw output, and the humidex curves are preserved
from the earlier figure with newly drawn axes calibrated to it.

## Data from this project

The projected LCZ maps for the three cities, the WRF surface parameters derived from
them, the downscaled 1-km fields and the trained CASPER weights are archived at Zenodo
under doi:10.5281/zenodo.24060753.

The archive holds four items. `projected_lcz_maps.tar` has 41 rasters per city, which is
the present-day baseline plus five pathways across eight decades. `lcz_surface_parameters.tar`
has the four geo_em files for each of those scenarios. `downscaling_results.tar` has the
cached 1-km fields per scenario. `CASPER-CESM.pt` is the trained emulator.

## Layout

```
run_pipeline.py        stage 1 orchestrator and command line
config.py              PipelineConfig: paths, pathways, category and LCZ mappings
data_integration.py    steps 0 and 1, and the validation maps
lcz_composition.py     step 2
lcz_projector.py       step 3
transition_matrix.py   step 4
__main__.py            lets the directory itself be run with `python .`

ncc_figures.py         figure and cache library shared by stage 2 and the figure scripts
shoreline.py           coastlines for the map panels
make_figures.sh        regenerates Figures 2 to 5 and the supplementary items
figures_1_and_6/       draw.io sources for the two schematic figures

precompute_caches.py   one CASPER inference pass per scenario, writes the field caches
refix_static.py        rebuilds the static-baseline caches from the corrected present-day LCZ
refix_footprint.py     re-evaluates every cache on the fixed 2100 urban footprint
run_precompute.sh      driver for the full city, pathway, forcing and decade matrix
run_refix.sh           driver for refix_static.py
run_footprint_full.sh  driver for refix_footprint.py

examples/quickstart/   synthetic input, runner and expected output
tests/                 regression tests over the quick-start example
```

## Requirements

Python 3.11 or newer. The pipeline is pure Python and runs on Linux, macOS and Windows.
Continuous integration tests it on Linux. Development used macOS and Windows.

```bash
pip install -r requirements.txt
```

`rasterio` and `pyproj` need GDAL and PROJ. A conda environment is the easier route on
Windows. If `import rasterio` fails with a missing GDAL library, install `rasterio` from
conda-forge rather than from pip. If `pyproj` cannot find its projection database, set
`PROJ_DATA` to the `share/proj` directory of your environment.

Run the tests with `pytest tests`. They run the quick-start example and compare the
output against stored tables.

Stage 2 needs a GPU for a practical runtime. The published runs used one consumer GPU and
processed about twelve six-hourly 1-km fields per second.

## Input data

Three inputs drive stage 1, and one drives stage 2. None of them live in this repository.

Historical LCZ maps define the study area. Supply one GeoTIFF per census year. The code
finds them by globbing `*.tif`, reads the year from the filename, and skips `_uc`
variants. Maps for Canadian cities are on
[figshare](https://figshare.com/articles/dataset/Multitemporal_LCZ_maps_for_Canadian_functional_urban_areas/13387082).

Global urban land and urban fraction set the scenario demand. Supply one GeoTIFF per
pathway and year as `<global_urban_dir>/SSP<n>/global_SSP<n>_<year>.tif`, in EPSG:6933 at
1 km, with 0 outside, 1 non-urban and 2 urban. The projections used here are on
[PANGAEA](https://doi.pangaea.de/10.1594/PANGAEA.905890).

Global natural land cover supplies the non-urban categories. Supply NetCDF in WGS84 at
0.05 degrees, matched as `GCAM_Demeter_LU_ssp<n>_*_<year>.nc`. Latitude and longitude are
transposed in some files, and the reader detects and corrects this. The projections used
here are on [Zenodo](https://doi.org/10.5281/zenodo.4584775).

Stage 2 needs bias-corrected global climate output as its atmospheric forcing. The
[NSF NCAR Geoscience Data Exchange](https://gdex.ucar.edu/datasets/d316001/dataaccess/)
holds the forcing used here.

## Built on

Stage 1 allocates demand in space with the [PLUS model][plus]. Stage 2 calls
[CASPER][casper] for the downscaling. [WUDAPT-to-WRF][w2w] converts a projected LCZ map
into the surface parameters that the emulator reads.

[plus]: https://github.com/HPSCIL/Patch-generating_Land_Use_Simulation_Model
[casper]: https://github.com/UMBE-LAB/CASPER-Context-Aware-Structural-Prior-Enhanced-Resolution
[w2w]: https://github.com/matthiasdemuzere/w2w

## Citing this work

`CITATION.cff` holds the citation entries, and GitHub renders them from the sidebar.
`AUTHORS.md` lists the contributors and `CONTRIBUTING.md` explains how to propose a
change. `AGENTS.md` records the files that must not change without review.
