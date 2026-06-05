#!/usr/bin/env bash
set -euo pipefail

ASSET_DIR="${1:-release_assets}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
log_file="$LOG_DIR/wait_then_upload_dataset_release_$(date +%Y%m%d_%H%M%S).log"

{
  echo "Started at: $(date)"
  echo "Waiting for dataset packaging processes to finish..."

  while true; do
    mapfile -t pids < <(pgrep -f 'bash ./scripts/package_radiomapseer_release.sh|tar -C .*MultiScene20_RF300M8Runs_RadioMapSeerPack|split -b 1900m')
    if [[ ${#pids[@]} -eq 0 ]]; then
      break
    fi

    echo "packaging pids: ${pids[*]}"
    du -sh "$ASSET_DIR" 2>/dev/null || true
    find "$ASSET_DIR" -maxdepth 1 -type f -name '*.part-*' 2>/dev/null | wc -l \
      | awk '{print "part files: " $1}'
    sleep 300
  done

  echo "Packaging appears complete at: $(date)"
  echo "Uploading release assets..."
  ./scripts/upload_dataset_release.sh "$ASSET_DIR"
  echo "Upload complete at: $(date)"
} 2>&1 | tee "$log_file"

echo "Log written to: $log_file"
