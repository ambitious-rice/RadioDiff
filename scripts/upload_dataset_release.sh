#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-ambitious-rice/RadioDiff}"
TAG="${TAG:-dynamic-radiomap-pack-v1}"
TITLE="${TITLE:-DynamicRadioMap MultiScene20 RF300M8Runs RadioMapSeerPack}"
ASSET_DIR="${1:-release_assets}"
NOTES="${NOTES:-DynamicRadioMap dataset pack only. VAE and diffusion weights are intentionally excluded.}"

if [[ ! -d "$ASSET_DIR" ]]; then
  echo "Asset directory not found: $ASSET_DIR" >&2
  exit 1
fi

shopt -s nullglob
assets=("$ASSET_DIR"/*.sha256 "$ASSET_DIR"/*.part-*)
if [[ ${#assets[@]} -eq 0 ]]; then
  echo "No release assets found in: $ASSET_DIR" >&2
  exit 1
fi

if command -v gh >/dev/null 2>&1; then
  if ! gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
    gh release create "$TAG" --repo "$REPO" --title "$TITLE" --notes "$NOTES"
  fi
  gh release upload "$TAG" "${assets[@]}" --repo "$REPO" --clobber
  echo "Uploaded ${#assets[@]} assets to https://github.com/$REPO/releases/tag/$TAG"
  exit 0
fi

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  cat >&2 <<EOF
GitHub CLI is not installed and GITHUB_TOKEN is not set.

Install gh and login:
  sudo apt-get update && sudo apt-get install -y gh
  gh auth login

Or export a token with repo contents write permission:
  export GITHUB_TOKEN=...

Then rerun:
  ./scripts/upload_dataset_release.sh $ASSET_DIR
EOF
  exit 1
fi

owner="${REPO%%/*}"
repo="${REPO#*/}"
api="https://api.github.com/repos/$owner/$repo"

release_json="$(curl -fsS \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  "$api/releases/tags/$TAG" 2>/dev/null || true)"

if [[ -z "$release_json" ]]; then
  release_json="$(jq -n \
    --arg tag "$TAG" \
    --arg title "$TITLE" \
    --arg notes "$NOTES" \
    '{tag_name:$tag, name:$title, body:$notes, draft:false, prerelease:false}' \
    | curl -fsS \
      -X POST \
      -H "Authorization: Bearer '"$GITHUB_TOKEN"'" \
      -H "Accept: application/vnd.github+json" \
      -d @- \
      "$api/releases")"
fi

upload_url="$(jq -r '.upload_url | sub("\\{.*$"; "")' <<<"$release_json")"
if [[ -z "$upload_url" || "$upload_url" == "null" ]]; then
  echo "Could not resolve release upload URL." >&2
  exit 1
fi

for asset in "${assets[@]}"; do
  name="$(basename "$asset")"
  echo "Uploading $name"
  existing_id="$(curl -fsS \
    -H "Authorization: Bearer $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    "$api/releases/tags/$TAG" \
    | jq -r --arg name "$name" '.assets[]? | select(.name == $name) | .id' \
    | head -1)"
  if [[ -n "$existing_id" ]]; then
    curl -fsS \
      -X DELETE \
      -H "Authorization: Bearer $GITHUB_TOKEN" \
      -H "Accept: application/vnd.github+json" \
      "$api/releases/assets/$existing_id" >/dev/null
  fi
  curl -fsS \
    -X POST \
    -H "Authorization: Bearer $GITHUB_TOKEN" \
    -H "Content-Type: application/octet-stream" \
    --data-binary @"$asset" \
    "$upload_url?name=$name" >/dev/null
done

echo "Uploaded ${#assets[@]} assets to https://github.com/$REPO/releases/tag/$TAG"
