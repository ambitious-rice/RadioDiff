#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-/data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/MultiScene20_RF300M8Runs_RadioMapSeerPack}"
OUT_DIR="${2:-release_assets}"
ARCHIVE_NAME="${3:-MultiScene20_RF300M8Runs_RadioMapSeerPack.tar.gz}"
CHUNK_SIZE="${CHUNK_SIZE:-1900m}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Data directory not found: $DATA_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/$ARCHIVE_NAME" "$OUT_DIR/$ARCHIVE_NAME".part-*

tar -C "$(dirname "$DATA_DIR")" -czf - "$(basename "$DATA_DIR")" \
  | tee >(sha256sum | awk -v name="$ARCHIVE_NAME" '{print $1 "  " name}' > "$OUT_DIR/$ARCHIVE_NAME.sha256") \
  | split -b "$CHUNK_SIZE" -d -a 3 - "$OUT_DIR/$ARCHIVE_NAME.part-"

echo "Created release assets in: $OUT_DIR"
ls -lh "$OUT_DIR"
