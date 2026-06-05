#!/usr/bin/env bash
set -euo pipefail

ASSET_DIR="${1:-release_assets}"
DEST_DIR="${2:-data}"
ARCHIVE_NAME="${3:-MultiScene20_RF300M8Runs_RadioMapSeerPack.tar.gz}"

mkdir -p "$DEST_DIR"

if [[ -f "$ASSET_DIR/$ARCHIVE_NAME.sha256" ]]; then
  expected="$(awk '{print $1}' "$ASSET_DIR/$ARCHIVE_NAME.sha256")"
  actual="$(cat "$ASSET_DIR/$ARCHIVE_NAME".part-* | sha256sum | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "Checksum mismatch for $ARCHIVE_NAME" >&2
    echo "expected: $expected" >&2
    echo "actual:   $actual" >&2
    exit 1
  fi
fi

cat "$ASSET_DIR/$ARCHIVE_NAME".part-* | tar -C "$DEST_DIR" -xzf -
echo "Unpacked dataset to: $DEST_DIR/MultiScene20_RF300M8Runs_RadioMapSeerPack"
