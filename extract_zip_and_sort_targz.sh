#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./extract_zip_and_sort_targz.sh [directory]
# If no directory is supplied, uses current directory.

TARGET_DIR="${1:-.}"

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Error: directory not found: $TARGET_DIR" >&2
  exit 1
fi

cd "$TARGET_DIR"

echo "Target directory: $(pwd)"
echo "1) Unzip all .zip files into main directory"
echo "2) For yyyymmdd*.tar.gz: move to yyyy/mm/ then extract INTO that yyyy/mm/ directory"
echo

shopt -s nullglob

# --- Step 1: Unzip all .zip into main directory ---
for z in *.zip; do
  echo "Unzipping: $z -> $(pwd)"
  unzip -o -q "$z" -d .
done

# --- Step 2: Move tarballs into yyyy/mm/ then extract there ---
for t in *.tar.gz; do
  [[ -e "$t" ]] || continue

  prefix="${t:0:8}"
  if [[ ! "$prefix" =~ ^[0-9]{8}$ ]]; then
    echo "Skipping (does not start with yyyymmdd): $t"
    continue
  fi

  yyyy="${prefix:0:4}"
  mm="${prefix:4:2}"
  dd="${prefix:6:2}"

  if (( 10#$mm < 1 || 10#$mm > 12 || 10#$dd < 1 || 10#$dd > 31 )); then
    echo "Skipping (invalid date prefix): $t"
    continue
  fi

  dest_dir="${yyyy}/${mm}"
  mkdir -p "$dest_dir"

  dest_path="${dest_dir}/${t}"
  if [[ -e "$dest_path" ]]; then
    ts="$(date +%Y%m%d_%H%M%S)"
    dest_path="${dest_dir}/${ts}_${t}"
    echo "Name collision; moving with timestamp: $t -> $dest_path"
  else
    echo "Moving: $t -> $dest_path"
  fi

  mv -- "$t" "$dest_path"

  echo "Extracting: $dest_path -> $dest_dir/"
  tar -xzf "$dest_path" -C "$dest_dir"
done

echo
echo "Done."
