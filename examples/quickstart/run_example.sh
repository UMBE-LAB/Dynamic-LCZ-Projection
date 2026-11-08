#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-python}"

cd "$ROOT"
"$PY" examples/quickstart/make_inputs.py
"$PY" run_pipeline.py \
    --lcz-dir           examples/quickstart/input/lcz \
    --global-urban-dir  examples/quickstart/input/global_urban \
    --global-landuse-dir examples/quickstart/input/global_landuse \
    --output-dir        examples/quickstart/output \
    --no-validation

echo
echo "Results are in examples/quickstart/output"
echo "Compare against examples/quickstart/expected_output with: pytest tests"
