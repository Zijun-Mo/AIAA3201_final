#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  bash scripts/run_phase6.sh [options]

Description:
  Run Phase 6 as a narrow E/F core stacking gate (H0 references + H1 F-best then SAM3).

Options:
  --config <path>                    Config path. Default: configs/base.yaml
  --datasets <spec>                  mandatory | all | csv-list. Default: mandatory
  --exp-id <id>                      Output experiment id. Default: phase6_core_maskscore_fastvqa_<timestamp>
  --phase2-exp-id <id>               B-best reference. Default: phase2_maskscore_fastvqa_20260510_023457_pl220
  --phase3-exp-id <id>               E-best reference. Default: phase3_maskscore_fastvqa_20260510_023457_pl220
  --phase4-exp-id <id>               F-best reference. Default: phase4_maskscore_fastvqa_altbackend_20260510_pl220
  --sam3-env-name <name>             Conda env used by SAM3 subprocess. Default: sam3
  --conda-env <name|current>         Conda env used by Phase 6 driver. Default: aiaa3201
  --strict-sam3-permission <bool>    Require SAM3 permission probe to pass. Default: true
  --seed <int>                       Random seed. Default: 42
  --max-frames <int>                 Optional smoke-test frame cap.
  --auto-install-missing <bool>      Allow dependency auto-install hooks. Default: true
  --skip-check <bool>                Skip Phase 6 gate check. Default: false
  --help                             Show this help
EOF
}

normalize_bool() {
  local v="${1:-}"
  case "${v,,}" in
    1|true|yes|y) echo "true" ;;
    0|false|no|n) echo "false" ;;
    *)
      echo "Invalid boolean value: ${v}" >&2
      exit 1
      ;;
  esac
}

CONFIG="configs/base.yaml"
DATASETS="mandatory"
EXP_ID="phase6_core_maskscore_fastvqa_$(date +%Y%m%d_%H%M%S)"
PHASE2_EXP_ID="phase2_maskscore_fastvqa_20260510_023457_pl220"
PHASE3_EXP_ID="phase3_maskscore_fastvqa_20260510_023457_pl220"
PHASE4_EXP_ID="phase4_maskscore_fastvqa_altbackend_20260510_pl220"
SAM3_ENV_NAME="sam3"
CONDA_ENV="${PHASE6_CONDA_ENV:-aiaa3201}"
STRICT_SAM3_PERMISSION="true"
SEED="42"
MAX_FRAMES=""
AUTO_INSTALL_MISSING="true"
SKIP_CHECK="false"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)
      CONFIG="${2:-}"
      shift 2
      ;;
    --datasets)
      DATASETS="${2:-}"
      shift 2
      ;;
    --exp-id)
      EXP_ID="${2:-}"
      shift 2
      ;;
    --phase2-exp-id)
      PHASE2_EXP_ID="${2:-}"
      shift 2
      ;;
    --phase3-exp-id)
      PHASE3_EXP_ID="${2:-}"
      shift 2
      ;;
    --phase4-exp-id)
      PHASE4_EXP_ID="${2:-}"
      shift 2
      ;;
    --sam3-env-name)
      SAM3_ENV_NAME="${2:-}"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="${2:-}"
      shift 2
      ;;
    --strict-sam3-permission)
      STRICT_SAM3_PERMISSION="$(normalize_bool "${2:-}")"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --max-frames)
      MAX_FRAMES="${2:-}"
      shift 2
      ;;
    --auto-install-missing)
      AUTO_INSTALL_MISSING="$(normalize_bool "${2:-}")"
      shift 2
      ;;
    --skip-check)
      SKIP_CHECK="$(normalize_bool "${2:-}")"
      shift 2
      ;;
    --help|-h)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      show_help
      exit 1
      ;;
  esac
done

if [ -z "${EXP_ID}" ]; then
  echo "--exp-id cannot be empty" >&2
  exit 1
fi

python_cmd=(python3)
if [ -n "${CONDA_ENV}" ] && [ "${CONDA_ENV}" != "current" ] && [ "${CONDA_ENV}" != "none" ]; then
  python_cmd=(conda run -n "${CONDA_ENV}" python3)
fi

cmd=(
  "${python_cmd[@]}" src/part3/run_explore.py
  --config "${CONFIG}"
  --datasets "${DATASETS}"
  --exp-id "${EXP_ID}"
  --phase2-exp-id "${PHASE2_EXP_ID}"
  --phase3-exp-id "${PHASE3_EXP_ID}"
  --phase4-exp-id "${PHASE4_EXP_ID}"
  --stages H0,H1
  --phase phase6
  --sam3-env-name "${SAM3_ENV_NAME}"
  --strict-sam3-permission "${STRICT_SAM3_PERMISSION}"
  --seed "${SEED}"
  --auto-install-missing "${AUTO_INSTALL_MISSING}"
)

if [ -n "${MAX_FRAMES}" ]; then
  cmd+=(--max-frames "${MAX_FRAMES}")
fi

printf '+ '
printf '%q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

if [ "${SKIP_CHECK}" != "true" ]; then
  "${python_cmd[@]}" src/common/check_phase6.py \
    --config "${CONFIG}" \
    --exp-id "${EXP_ID}" \
    --strict-sam3-permission "${STRICT_SAM3_PERMISSION}"
fi
