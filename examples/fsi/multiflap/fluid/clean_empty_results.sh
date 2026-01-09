#!/usr/bin/env bash

# Delete OpenFOAM time directories that do NOT contain a velocity field file "U".
# - Default is DRY-RUN: it only prints what would be deleted.
# - Use --force to actually delete.
# - Use --case <path> to point to a case directory (default: current directory).
# - Handles serial times (e.g., 0, 0.1, 100) and parallel times inside processor*/.

set -uo pipefail

# Default to dry-run for safety; use --force to actually delete
DRY_RUN=1
CASE_DIR="$(pwd)"

usage() {
  echo "Usage: $(basename "$0") [--case <path>] [--force]"
  echo "  --case   Case directory to clean (default: current directory)"
  echo "  --force  Perform deletion (default: dry-run)"
}

is_time_dir_name() {
  local name="$1"
  [[ "$name" =~ ^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]]
}

has_U_file() {
  # Check for velocity field files named "U" or compressed "U.gz"
  # Search up to two levels below the time directory to support multi-region layouts
  # e.g., <time>/fluid/U or processor*/<time>/region/U
  local dir="$1"
  if find "$dir" -maxdepth 2 -type f \( -name U -o -name 'U.gz' \) -print -quit | grep -q .; then
    return 0
  else
    return 1
  fi
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --case)
      [[ $# -lt 2 ]] && { echo "Error: --case requires a path"; usage; exit 2; }
      CASE_DIR="$2"; shift 2 ;;
    --force)
      DRY_RUN=0; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done

[[ -d "$CASE_DIR" ]] || { echo "Error: case directory not found: $CASE_DIR"; exit 2; }

echo "Case: $CASE_DIR"
echo "Mode: $([[ "$DRY_RUN" -eq 1 ]] && echo DRY-RUN || echo FORCE)"

to_delete=()

# Collect serial time directories at case root (excluding processor*)
while IFS= read -r -d '' dir; do
  name="$(basename "$dir")"
  if is_time_dir_name "$name"; then
    if ! has_U_file "$dir"; then
      to_delete+=("$dir")
    fi
  fi
done < <(find "$CASE_DIR" -mindepth 1 -maxdepth 1 -type d ! -name 'processor*' -print0)

# Collect parallel time directories inside processor*/
while IFS= read -r -d '' pdir; do
  while IFS= read -r -d '' tdir; do
    tname="$(basename "$tdir")"
    if is_time_dir_name "$tname"; then
      if ! has_U_file "$tdir"; then
        to_delete+=("$tdir")
      fi
    fi
  done < <(find "$pdir" -mindepth 1 -maxdepth 1 -type d -print0)
done < <(find "$CASE_DIR" -mindepth 1 -maxdepth 1 -type d -name 'processor*' -print0)

if [[ ${#to_delete[@]} -eq 0 ]]; then
  echo "No directories eligible for deletion (no empty-U results found)."
  exit 0
fi

echo "Directories without U (will be deleted):"
for d in "${to_delete[@]}"; do
  echo "  $d"
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry-run complete. Use --force to delete."
  exit 0
fi

echo "Deleting..."
for d in "${to_delete[@]}"; do
  rm -rf "$d"
done
echo "Done."
