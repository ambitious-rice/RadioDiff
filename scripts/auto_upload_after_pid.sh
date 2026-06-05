#!/usr/bin/env bash
set -u

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <packaging-pid> [asset-dir]" >&2
  exit 1
fi

PACK_PID="$1"
ASSET_DIR="${2:-release_assets}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/auto_upload_after_packaging_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "$@" | tee -a "$LOG_FILE"
}

log "Started at: $(date)"
log "Watching packaging pid: $PACK_PID"

while kill -0 "$PACK_PID" 2>/dev/null; do
  log "$(date)"
  du -sh "$ASSET_DIR" 2>/dev/null | tee -a "$LOG_FILE" || true
  count="$(find "$ASSET_DIR" -maxdepth 1 -type f -name '*.part-*' 2>/dev/null | wc -l)"
  log "part files: $count"
  sleep 300
done

log "Packaging pid finished at: $(date)"
log "Uploading release assets..."

if ./scripts/upload_dataset_release.sh "$ASSET_DIR" 2>&1 | tee -a "$LOG_FILE"; then
  log "Upload completed at: $(date)"
else
  status=$?
  log "Upload failed with status $status at: $(date)"
  exit "$status"
fi
