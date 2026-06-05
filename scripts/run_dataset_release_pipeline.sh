#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-logs}"
ASSET_DIR="${ASSET_DIR:-release_assets}"
mkdir -p "$LOG_DIR"

log_file="$LOG_DIR/dataset_release_pipeline_$(date +%Y%m%d_%H%M%S).log"

{
  date
  ./scripts/package_radiomapseer_release.sh \
    /data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/MultiScene20_RF300M8Runs_RadioMapSeerPack \
    "$ASSET_DIR"
  ./scripts/upload_dataset_release.sh "$ASSET_DIR"
  date
} 2>&1 | tee "$log_file"

echo "Pipeline log: $log_file"
