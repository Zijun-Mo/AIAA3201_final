#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  bash scripts/sync_best_aliases.sh --a-exp-id <phase1_exp_id> --b-exp-id <phase2_exp_id>

Options:
  --a-exp-id <id>   Source Phase 1 experiment id (for alias A-best)
  --b-exp-id <id>   Source Phase 2 experiment id (for alias B-best)
  --help            Show this help message

Behavior:
  - Copy (not move) source experiment directories into:
      outputs/videos/A-best, outputs/metrics/A-best, outputs/figures/A-best
      outputs/videos/B-best, outputs/metrics/B-best, outputs/figures/B-best
  - For `outputs/videos/*-best`, skip `_candidates/` by default to avoid large duplicate intermediates.
  - Preserve original exp_id directories.
  - Update outputs/metrics/best_alias_map.json
EOF
}

A_EXP_ID=""
B_EXP_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --a-exp-id)
      A_EXP_ID="${2:-}"
      shift 2
      ;;
    --b-exp-id)
      B_EXP_ID="${2:-}"
      shift 2
      ;;
    --help|-h)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      show_help
      exit 1
      ;;
  esac
done

if [[ -z "$A_EXP_ID" || -z "$B_EXP_ID" ]]; then
  echo "Both --a-exp-id and --b-exp-id are required."
  show_help
  exit 1
fi

copy_alias_dir() {
  local root="$1"
  local source_id="$2"
  local alias_name="$3"
  local skip_candidates="${4:-false}"
  local src="$root/$source_id"
  local dst="$root/$alias_name"

  if [[ ! -d "$src" ]]; then
    echo "Missing source directory: $src"
    exit 1
  fi

  rm -rf "$dst"
  mkdir -p "$dst"

  # Copy directory entries one by one so we can skip huge intermediate folders.
  local entry base
  shopt -s nullglob dotglob
  for entry in "$src"/*; do
    base="$(basename "$entry")"
    if [[ "$base" == "." || "$base" == ".." ]]; then
      continue
    fi
    if [[ "$skip_candidates" == "true" && "$base" == "_candidates" ]]; then
      continue
    fi
    cp -a "$entry" "$dst/"
  done
  shopt -u nullglob dotglob
}

copy_alias_dir "outputs/videos" "$A_EXP_ID" "A-best" "true"
copy_alias_dir "outputs/metrics" "$A_EXP_ID" "A-best"
copy_alias_dir "outputs/figures" "$A_EXP_ID" "A-best"

copy_alias_dir "outputs/videos" "$B_EXP_ID" "B-best" "true"
copy_alias_dir "outputs/metrics" "$B_EXP_ID" "B-best"
copy_alias_dir "outputs/figures" "$B_EXP_ID" "B-best"

python3 - "$A_EXP_ID" "$B_EXP_ID" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

a_exp_id = sys.argv[1]
b_exp_id = sys.argv[2]
map_path = Path("outputs/metrics/best_alias_map.json")
payload = {}
if map_path.exists():
    try:
        payload = json.loads(map_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}

aliases = payload.get("aliases", {})
aliases["A-best"] = {
    "source_exp_id": a_exp_id,
    "videos": f"outputs/videos/{a_exp_id}",
    "metrics": f"outputs/metrics/{a_exp_id}",
    "figures": f"outputs/figures/{a_exp_id}",
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
}
aliases["B-best"] = {
    "source_exp_id": b_exp_id,
    "videos": f"outputs/videos/{b_exp_id}",
    "metrics": f"outputs/metrics/{b_exp_id}",
    "figures": f"outputs/figures/{b_exp_id}",
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
}

payload["aliases"] = aliases
payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
map_path.parent.mkdir(parents=True, exist_ok=True)
map_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

echo "Alias sync complete:"
echo "  A-best <- $A_EXP_ID"
echo "  B-best <- $B_EXP_ID"
echo "  map: outputs/metrics/best_alias_map.json"
