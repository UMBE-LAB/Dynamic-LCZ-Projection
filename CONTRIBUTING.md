# Contributing

This is a research code. It supports a published paper, so correctness of the
numbers matters more than features.

## Reporting a problem

Open a GitHub issue. State the command you ran, the input you used, and what you
expected. Attach the log output. Use these labels: `bug`, `documentation`,
`validation`, `testing`, `good-first-issue`, `performance`, `refactor`,
`question`, `publication-result`.

## Making a change

1. Branch from `main`. Name the branch `fix/short-description` or
   `feat/short-description`.
2. Run the tests before you open a pull request:

   ```bash
   pytest tests
   ```

3. Open a pull request against `main`. Describe what changed and why.
4. A maintainer reviews every pull request. Do not merge your own.

## Coding style

Match the surrounding code. Modules are flat and one module covers one pipeline
step. Names say what the thing is. Keep configuration in `config.py` rather than
in the module that uses it.

## Changes that need extra care

Two files decide every published number. `config.py` holds the category-to-LCZ
mapping and the ensemble seed. `lcz_composition.py` holds the projection
ensemble. A change to either one moves the results. If you change them, say so
in the pull request and expect the stored expected output to change.

## Starter tasks

- Pack a reduced single-city dataset so stage 1 runs on real input.
- State the units of every output field in the README.
- Add troubleshooting notes for common GDAL and PROJ install failures.
- Replace the absolute paths in the three stage 2 shell drivers with a config file.
