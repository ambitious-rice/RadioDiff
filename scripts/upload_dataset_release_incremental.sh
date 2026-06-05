#!/usr/bin/env bash
set -u

REPO="${REPO:-ambitious-rice/RadioDiff}"
TAG="${TAG:-dynamic-radiomap-pack-v1}"
TITLE="${TITLE:-DynamicRadioMap MultiScene20 RF300M8Runs RadioMapSeerPack}"
ASSET_DIR="${1:-release_assets}"
PACK_PID="${2:-}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/incremental_dataset_release_upload_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "$@" | tee -a "$LOG_FILE"
}

ensure_release() {
  if ! gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
    gh release create "$TAG" --repo "$REPO" --title "$TITLE" \
      --notes "DynamicRadioMap dataset pack only. VAE and diffusion weights are intentionally excluded." \
      2>&1 | tee -a "$LOG_FILE"
  fi
}

uploaded_assets() {
  gh release view "$TAG" --repo "$REPO" --json assets \
    | jq -r '.assets[].name' 2>/dev/null
}

upload_one() {
  local asset="$1"
  local name
  name="$(basename "$asset")"

  if grep -qxF "$name" "$uploaded_list"; then
    log "skip uploaded: $name"
    return 0
  fi

  log "uploading: $name"
  if gh release upload "$TAG" "$asset" --repo "$REPO" --clobber 2>&1 | tee -a "$LOG_FILE"; then
    echo "$name" >> "$uploaded_list"
    log "uploaded: $name"
  else
    log "upload failed, will retry later: $name"
  fi
}

packaging_running() {
  [[ -n "$PACK_PID" ]] && kill -0 "$PACK_PID" 2>/dev/null
}

if ! command -v gh >/dev/null 2>&1; then
  echo "gh is required." >&2
  exit 1
fi

ensure_release
uploaded_list="$(mktemp)"
uploaded_assets > "$uploaded_list"

log "Started at: $(date)"
log "Repo: $REPO"
log "Tag: $TAG"
log "Asset dir: $ASSET_DIR"
log "Packaging pid: ${PACK_PID:-none}"

while true; do
  mapfile -t parts < <(find "$ASSET_DIR" -maxdepth 1 -type f -name '*.part-*' | sort)
  if [[ ${#parts[@]} -gt 0 ]]; then
    limit=${#parts[@]}
    if packaging_running; then
      limit=$((limit - 1))
    fi

    for ((i = 0; i < limit; i++)); do
      upload_one "${parts[$i]}"
    done
  fi

  if ! packaging_running; then
    mapfile -t final_assets < <(find "$ASSET_DIR" -maxdepth 1 -type f \( -name '*.part-*' -o -name '*.sha256' \) | sort)
    for asset in "${final_assets[@]}"; do
      upload_one "$asset"
    done
    break
  fi

  du -sh "$ASSET_DIR" 2>/dev/null | tee -a "$LOG_FILE" || true
  log "waiting before next upload scan..."
  sleep 300
done

rm -f "$uploaded_list"
log "Finished at: $(date)"
