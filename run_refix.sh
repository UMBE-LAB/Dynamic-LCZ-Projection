#!/usr/bin/env bash
set -u

: "${CASPER_DATA:?set CASPER_DATA to the preprocessed forcing directory}"
: "${CASPER_MODEL:?set CASPER_MODEL to the trained CASPER checkpoint}"
: "${CASPER_CACHE:=$PWD/cache}"
: "${CASPER_CONFIG:=$PWD/config_mac.yaml}"
: "${CASPER_CITIES:=Toronto Vancouver}"
: "${CASPER_BATCH:=16}"

export PYTORCH_ENABLE_MPS_FALLBACK=1
export MPLBACKEND=Agg

until python refix_static.py \
    --data_path     "$CASPER_DATA" \
    --inference_dir "$CASPER_CACHE" \
    --config        "$CASPER_CONFIG" \
    --unet_model    "$CASPER_MODEL" \
    --cities $CASPER_CITIES \
    --batch_size "$CASPER_BATCH" ; do
  echo "=== REFIX crashed, retrying in 30s ($(date)) ==="
  sleep 30
done
echo "=== REFIX COMPLETE ($(date)) ==="
