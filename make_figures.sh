#!/usr/bin/env bash
set -u

: "${CASPER_DATA:?set CASPER_DATA to the preprocessed forcing directory}"
: "${CASPER_CACHE:?set CASPER_CACHE to the directory holding the downscaling caches}"
: "${CASPER_MODEL:?set CASPER_MODEL to the trained CASPER checkpoint}"
: "${LCZ_PROJECT_DIR:?set LCZ_PROJECT_DIR to the directory holding <city>/Future_LCZ_geo_em}"
: "${NE_DATA:?set NE_DATA to a directory of Natural Earth 10m geojson files}"
: "${CASPER_CONFIG:=$PWD/config_mac.yaml}"
: "${CASPER_CITIES:=Montreal Toronto Vancouver}"
: "${FIGURE_OUT:=$PWD/figures}"
export FIGURE_OUT

mkdir -p "$FIGURE_OUT"
export MPLBACKEND=Agg

python ncc_figures.py \
    --data_path     "$CASPER_DATA" \
    --inference_dir "$CASPER_CACHE" \
    --output_dir    "$FIGURE_OUT" \
    --config        "$CASPER_CONFIG" \
    --unet_model    "$CASPER_MODEL" \
    --cities $CASPER_CITIES \
    --fig2_decade 2100 --batch_size 16

python build_nature_fig1.py
python build_nature_fig6.py

echo "=== FIGURES COMPLETE ($(date)) -> $FIGURE_OUT ==="
