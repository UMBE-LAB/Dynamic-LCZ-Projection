# AGENTS.md

## Project purpose

This code projects how the built surface of a city changes to 2100 under each
Shared Socioeconomic Pathway. It then downscales climate onto every projected
surface at 1 km. The difference between an evolving and a static surface is the
quantity the paper reports.

## Important commands

Install: `pip install -r requirements.txt`

Run tests: `pytest tests`

Run the quick-start example: `bash examples/quickstart/run_example.sh`

## Coding rules

Modules are flat. One module covers one pipeline step. Configuration lives in
`config.py`, not in the module that reads it. Match the naming and the structure
of the surrounding code.

## Scientific rules

The reference grid, the coordinate system and the historical years come from the
input LCZ files. Everything else is reprojected onto that grid.

Pixel counts are split by fraction and then rounded per LCZ class, so a domain
total can differ from the grid size by up to two pixels. `tests/test_quickstart.py`
states that tolerance.

`PipelineConfig.ensemble_seed` fixes the stage 1 random walk. Do not change it
without re-recording the expected output.

## Do not change without human approval

- `config.py`, in particular `category_lcz_mapping` and `ensemble_seed`.
- The ensemble in `lcz_composition.py`.
- The fixed 2100 footprint convention in `refix_footprint.py`.
- `examples/quickstart/expected_output/`, which the regression tests compare against.

Each one silently changes published numbers.

## Definition of done

A change is complete when `pytest tests` passes, the README still describes what
the code does, and any change to the expected output is deliberate and explained.
