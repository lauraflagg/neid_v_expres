#!/usr/bin/env bash
set -euo pipefail

# Move FITS files (*.fits, *.fit, *.fts) from any directory named "fitspec"
# up one level to that directory's parent.
#
# Usage:
#   ./move_fits_up_from_fitspec.sh /path/to/parent
#   DRY_RUN=1 ./move_fits_up_from_fitspec.sh /path/to/parent
#
# Notes:
# - By default, it will NOT overwrite existing files in the parent directory.
# - If a collision occurs, it will auto-rename by appending _1, _2, ... before the extension.

PARENT="${1:-}"
if [[ -z "${PARENT}" ]]; then
  echo "Usage: $0 /path/to/parent" >&2
  exit 1
fi

if [[ ! -d "${PARENT}" ]]; then
  echo "Error: '${PARENT}' is not a directory" >&2
  exit 1
fi

DRY_RUN="${DRY_RUN:-0}"

log() { echo "[INFO] $*"; }
run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] $*"
  else
    eval "$@"
  fi
}

# Given a destination path, return a non-colliding destination by appending _N.
# Example: /dest/file.fits -> /dest/file_1.fits (if file exists), etc.
unique_dest() {
  local dest="$1"
  if [[ ! -e "$dest" ]]; then
    printf '%s\n' "$dest"
    return 0
  fi

  local dir base name ext n
  dir="$(dirname -- "$dest")"
  base="$(basename -- "$dest")"

  # Split extension (keep .fits etc)
  if [[ "$base" == *.* ]]; then
    name="${base%.*}"
    ext=".${base##*.}"
  else
    name="$base"
    ext=""
  fi

  n=1
  while [[ -e "${dir}/${name}_${n}${ext}" ]]; do
    n=$((n+1))
  done
  printf '%s\n' "${dir}/${name}_${n}${ext}"
}

log "Searching under: ${PARENT}"
log "Dry run: ${DRY_RUN}"

# Find directories named exactly "fitspec"
# Use -print0 to safely handle spaces/newlines
while IFS= read -r -d '' fitspec_dir; do
  parent_dir="$(dirname -- "$fitspec_dir")"

  # Find FITS-like files directly inside fitspec_dir (not recursive inside it)
  # If you want recursive within fitspec as well, remove -maxdepth 1.
  while IFS= read -r -d '' f; do
    base="$(basename -- "$f")"
    dest="${parent_dir}/${base}"
    dest_unique="$(unique_dest "$dest")"

    if [[ "$dest_unique" != "$dest" ]]; then
      log "Collision: '${dest}' exists. Renaming to: '$(basename -- "$dest_unique")'"
    fi

    log "Move: '$f' -> '$dest_unique'"
    run "mv -- \"\$1\" \"\$2\"" _ "$f" "$dest_unique"
  done < <(
    find "$fitspec_dir" -maxdepth 1 -type f \( \
      -iname "*.fits" -o -iname "*.fit" -o -iname "*.fts" \
    \) -print0
  )

done < <(find "$PARENT" -type d -name "fitspec" -print0)

log "Done."