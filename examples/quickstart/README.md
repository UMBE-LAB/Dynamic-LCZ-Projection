# Quick-start example

This example runs the whole of stage 1 on synthetic input. It needs no external
dataset and no download.

```bash
bash examples/quickstart/run_example.sh
```

`make_inputs.py` writes a small artificial study area called Demo into `input/`:
three historical LCZ rasters of 36 by 36 cells at 1 km, global urban rasters for
five pathways and eight decades, and matching land-cover NetCDF files. The urban
core grows under SSP5, holds under SSP2, and contracts under SSP3.

`run_pipeline.py` then runs all four steps and writes `output/`.

## Expected result

The run takes about 5 seconds and needs under 500 MB of memory. It writes ten
files. The three tables below are stored in `expected_output/` and the tests
compare against them.

| File | Rows |
|---|---|
| `Demo_category_pixel_counts.csv` | 40 |
| `Demo_lcz_composition_fractions.csv` | 110 |
| `Demo_lcz_pixel_projections.csv` | 680 |

The first rows of the headline table look like this.

```
SSP,Year,LCZ_Type,LCZ_Description,Category,Pixel_Count,Pct_of_Total
SSP1,2030,1,Compact high-rise,Urban Land,0,0.0
SSP1,2030,2,Compact midrise,Urban Land,108,8.33
```

Pixel counts are split by fraction and then rounded per LCZ class. A domain total
can therefore differ from the 1296 cells of the grid by up to two pixels.

## Checking the result

```bash
pytest tests
```

Five tests run. Three compare the tables against `expected_output/`. One checks
the rounding tolerance. One checks that urban land grows most under SSP5 and
least under SSP3.
