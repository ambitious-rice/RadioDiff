#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-ambitious-rice/RadioDiff}"
TAG="${TAG:-dynamic-radiomap-pack-v1}"
ASSET_DIR="${1:-release_assets}"
DEST_DIR="${2:-/data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap}"
ARCHIVE_NAME="${ARCHIVE_NAME:-MultiScene20_RF300M8Runs_RadioMapSeerPack.tar.gz}"

mkdir -p "$ASSET_DIR" "$DEST_DIR"

if command -v gh >/dev/null 2>&1; then
  gh release download "$TAG" --repo "$REPO" --dir "$ASSET_DIR" --pattern "$ARCHIVE_NAME*"
else
  echo "GitHub CLI is required for download convenience." >&2
  echo "Install it, then run: gh auth login" >&2
  exit 1
fi

"$(dirname "$0")/unpack_radiomapseer_release.sh" "$ASSET_DIR" "$DEST_DIR" "$ARCHIVE_NAME"
