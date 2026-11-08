#!/usr/bin/env bash
set -u

: "${CASPER_DATA:?set CASPER_DATA to the preprocessed forcing directory}"
: "${CASPER_MODEL:?set CASPER_MODEL to the trained CASPER checkpoint}"
: "${CASPER_CACHE:=$PWD/cache}"
: "${CASPER_CONFIG:=$PWD/config_mac.yaml}"
: "${STATIC_GEOEM_DIR:?set STATIC_GEOEM_DIR to the directory of corrected present-day geo_em files}"
: "${LCZ_PROJECT_DIR:?set LCZ_PROJECT_DIR to the directory holding <city>/Future_LCZ_geo_em}"
: "${CASPER_CITIES:=Montreal Toronto Vancouver}"
: "${CASPER_BATCH:=16}"

export PYTORCH_ENABLE_MPS_FALLBACK=1
export MPLBACKEND=Agg

until python refix_footprint.py \
    --data_path     "$CASPER_DATA" \
    --inference_dir "$CASPER_CACHE" \
    --config        "$CASPER_CONFIG" \
    --unet_model    "$CASPER_MODEL" \
    --cities $CASPER_CITIES \
    --batch_size "$CASPER_BATCH" ; do
  echo "=== FOOTPRINT crashed, retrying in 30s ($(date)) ==="
  sleep 30
done
echo "=== FOOTPRINT COMPLETE ($(date)) ==="
