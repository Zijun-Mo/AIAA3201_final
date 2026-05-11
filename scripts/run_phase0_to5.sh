#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  bash scripts/run_phase0_to5.sh [options]

Description:
  One-command runner for Phase 0 -> Phase 5 with gate checks after each phase.
  This script generates unified exp_id values and runs:
    Phase0: preprocess + evaluate + check_phase0
    Phase1: run_baseline + check_phase1
    Phase2: run_sota + check_phase2
    Phase3: run_explore(E) + check_phase3
    Phase4: run_explore(F) + check_phase4
    Phase5: run_diffusion(G) + check_phase5

Options:
  --config <path>                    Config path. Default: configs/base.yaml
  --datasets <spec>                  mandatory | all | csv-list. Default: mandatory
  --conda-env <name>                 Conda env to activate. Default: aiaa3201
  --activate-conda <bool>            Activate conda env before running phases. Default: true
  --phase3-conda-env <name>          Conda env used only for Phase3. Default: sam3
  --switch-phase3-conda <bool>       Switch to phase3 env for Phase3, then switch back. Default: true
  --seed <int>                       Global seed for Phase1-5. Default: 42
  --device <name>                    Phase5 device. Default: cuda
  --strict-dual-run <bool>           Phase2 strict dual-run check. Default: false
  --strict-sam3-permission <bool>    Phase3 SAM3 permission strict check. Default: true
  --phase2-stages <csv>              Phase2 stages. Default: B1,B2,B3,B4,B5
  --phase3-stages <csv>              Phase3 stages. Default: E1,E2,E3,E4
  --phase4-stages <csv>              Phase4 stages. Default: F1,F2,F3,F4,F5
  --start-phase <0-5>                First phase to run. Default: 0
  --end-phase <0-5>                  Last phase to run. Default: 5
  --preprocess-overwrite <bool>      Whether Phase0 preprocess uses --overwrite. Default: true
  --keep-variant-videos <bool>       Keep Phase5 <exp_id>__G-* videos. Default: false
  --run-tag <tag>                    Reuse a custom timestamp-like tag for all phase exp_id
  --skip-env-checks <bool>           Skip environment preflight checks. Default: false
  --dry-run <bool>                   Print commands only, do not execute. Default: false
  --help                             Show this help

Example:
  bash scripts/run_phase0_to5.sh
  bash scripts/run_phase0_to5.sh --device cpu --strict-sam3-permission false
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

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

run_cmd() {
  local label="$1"
  shift
  log "$label"
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi
  "$@"
}

run_phase() {
  local phase_idx="$1"
  local phase_name="$2"
  local exp_id="$3"
  local fn_name="$4"
  local start_ts end_ts rc

  PHASE_EXP["${phase_idx}"]="${exp_id}"
  log "========== ${phase_name} START (exp_id=${exp_id}) =========="
  start_ts="$(date +%s)"
  set +e
  "${fn_name}"
  rc=$?
  set -e
  end_ts="$(date +%s)"

  PHASE_DURATION["${phase_idx}"]="$((end_ts - start_ts))"
  if [[ ${rc} -eq 0 ]]; then
    PHASE_STATUS["${phase_idx}"]="PASS"
    log "========== ${phase_name} PASS (${PHASE_DURATION[${phase_idx}]}s) =========="
    return 0
  fi

  PHASE_STATUS["${phase_idx}"]="FAIL"
  log "========== ${phase_name} FAIL (${PHASE_DURATION[${phase_idx}]}s) =========="
  return "${rc}"
}

activate_conda_env() {
  if [[ "${ACTIVATE_CONDA}" != "true" ]]; then
    log "Skip conda activation by user option."
    return
  fi

  if [[ "${DRY_RUN}" == "true" ]]; then
    log "Dry-run: would activate conda env '${CONDA_ENV}'."
    return
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "conda command not found. Install Conda or rerun with --activate-conda false." >&2
    exit 1
  fi

  local conda_base conda_sh
  conda_base="$(conda info --base 2>/dev/null || true)"
  if [[ -z "${conda_base}" ]]; then
    echo "Unable to resolve conda base via 'conda info --base'." >&2
    exit 1
  fi

  conda_sh="${conda_base}/etc/profile.d/conda.sh"
  if [[ ! -f "${conda_sh}" ]]; then
    echo "Conda init script not found: ${conda_sh}" >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  source "${conda_sh}"
  conda activate "${CONDA_ENV}"
  ACTIVE_CONDA_ENV="${CONDA_ENV}"
  log "Activated conda env: ${CONDA_ENV}"
  run_cmd "Conda | python version" python3 -V
}

switch_conda_env() {
  local target_env="$1"
  local reason="${2:-conda switch}"

  if [[ "${ACTIVATE_CONDA}" != "true" ]]; then
    log "Skip conda switch (${reason}) because --activate-conda=false."
    return
  fi

  if [[ -z "${target_env}" ]]; then
    echo "switch_conda_env requires a non-empty target env." >&2
    exit 1
  fi

  if [[ "${DRY_RUN}" == "true" ]]; then
    log "Dry-run: would switch conda env to '${target_env}' (${reason})."
    ACTIVE_CONDA_ENV="${target_env}"
    return
  fi

  if [[ "${ACTIVE_CONDA_ENV:-}" == "${target_env}" ]]; then
    log "Conda env already '${target_env}' (${reason})."
    return
  fi

  conda activate "${target_env}"
  ACTIVE_CONDA_ENV="${target_env}"
  log "Switched conda env: ${target_env} (${reason})"
  run_cmd "Conda | python version (${reason})" python3 -V
}

write_summary() {
  {
    echo "run_tag=${RUN_TAG}"
    echo "config=${CONFIG}"
    echo "datasets=${DATASETS}"
    echo "conda_env=${CONDA_ENV}"
    echo "activate_conda=${ACTIVATE_CONDA}"
    echo "phase3_conda_env=${PHASE3_CONDA_ENV}"
    echo "switch_phase3_conda=${SWITCH_PHASE3_CONDA}"
    echo "seed=${SEED}"
    echo "device=${DEVICE}"
    echo "strict_dual_run=${STRICT_DUAL_RUN}"
    echo "strict_sam3_permission=${STRICT_SAM3_PERMISSION}"
    echo "phase2_stages=${PHASE2_STAGES}"
    echo "phase3_stages=${PHASE3_STAGES}"
    echo "phase4_stages=${PHASE4_STAGES}"
    echo "start_phase=${START_PHASE}"
    echo "end_phase=${END_PHASE}"
    echo "preprocess_overwrite=${PREPROCESS_OVERWRITE}"
    echo "keep_variant_videos=${KEEP_VARIANT_VIDEOS}"
    echo "dry_run=${DRY_RUN}"
    echo "log_file=${LOG_FILE}"
    echo "----"
    for idx in 0 1 2 3 4 5; do
      echo "phase${idx}.exp_id=${PHASE_EXP[${idx}]:-}"
      echo "phase${idx}.status=${PHASE_STATUS[${idx}]:-NOT_RUN}"
      echo "phase${idx}.duration_s=${PHASE_DURATION[${idx}]:-0}"
    done
  } > "${SUMMARY_FILE}"

  log "Run summary:"
  for idx in 0 1 2 3 4 5; do
    log "  Phase${idx}: ${PHASE_STATUS[${idx}]:-NOT_RUN} | exp_id=${PHASE_EXP[${idx}]:-} | ${PHASE_DURATION[${idx}]:-0}s"
  done
  log "Summary file: ${SUMMARY_FILE}"
  log "Log file: ${LOG_FILE}"
}

preflight() {
  mkdir -p data/raw data/processed data/gt
  mkdir -p outputs/videos outputs/masks outputs/figures outputs/metrics outputs/logs

  if [[ "${DATASETS}" != "mandatory" && "${DATASETS}" != "all" ]]; then
    log "WARN: datasets='${DATASETS}'. Full acceptance is expected on mandatory datasets."
  fi

  for raw_video in data/raw/wild.mp4 data/raw/bmx-trees.mp4 data/raw/tennis.mp4; do
    if [[ ! -f "${raw_video}" ]]; then
      echo "Missing mandatory raw video: ${raw_video}" >&2
      exit 1
    fi
  done

  if [[ "${SKIP_ENV_CHECKS}" == "true" ]]; then
    log "Skip environment checks by user option."
    return
  fi

  run_cmd "Preflight | python version" python3 -V
  run_cmd "Preflight | stage A import check" \
    python3 -c "import numpy, cv2, yaml, skimage, matplotlib; print('core ok')"
  run_cmd "Preflight | torch check" \
    python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
}

phase0_body() {
  local preprocess_cmd=(bash scripts/preprocess.sh --datasets "${DATASETS}")
  if [[ "${PREPROCESS_OVERWRITE}" == "true" ]]; then
    preprocess_cmd+=(--overwrite)
  fi
  run_cmd "Phase0 | preprocess" "${preprocess_cmd[@]}"
  run_cmd "Phase0 | evaluate" \
    bash scripts/evaluate.sh \
      --config "${CONFIG}" \
      --exp-id "${P0_EXP}" \
      --datasets "${DATASETS}" \
      --pred-root data/processed \
      --gt-root data/gt \
      --allow-missing-gt true
  run_cmd "Phase0 | check" \
    bash scripts/check_phase0.sh --config "${CONFIG}" --exp-id "${P0_EXP}"
}

phase1_body() {
  run_cmd "Phase1 | run_baseline" \
    python3 src/part1/run_baseline.py \
      --config "${CONFIG}" \
      --datasets "${DATASETS}" \
      --exp-id "${P1_EXP}" \
      --seed "${SEED}" \
      --wild-fallback-mask true
  run_cmd "Phase1 | check" \
    bash scripts/check_phase1.sh --config "${CONFIG}" --exp-id "${P1_EXP}"
}

phase2_body() {
  run_cmd "Phase2 | run_sota" \
    python3 src/part2/run_sota.py \
      --config "${CONFIG}" \
      --datasets "${DATASETS}" \
      --exp-id "${P2_EXP}" \
      --phase1-exp-id "${P1_EXP}" \
      --seed "${SEED}" \
      --strict-dual-run "${STRICT_DUAL_RUN}" \
      --stages "${PHASE2_STAGES}"
  run_cmd "Phase2 | check" \
    bash scripts/check_phase2.sh \
      --config "${CONFIG}" \
      --exp-id "${P2_EXP}" \
      --strict-dual-run "${STRICT_DUAL_RUN}"
}

phase3_body() {
  if [[ "${SWITCH_PHASE3_CONDA}" == "true" ]]; then
    switch_conda_env "${PHASE3_CONDA_ENV}" "Phase3"
  fi

  run_cmd "Phase3 | run_explore(E)" \
    python3 src/part3/run_explore.py \
      --config "${CONFIG}" \
      --datasets "${DATASETS}" \
      --exp-id "${P3_EXP}" \
      --phase2-exp-id "${P2_EXP}" \
      --stages "${PHASE3_STAGES}" \
      --sam3-env-name "${PHASE3_CONDA_ENV}" \
      --seed "${SEED}" \
      --strict-sam3-permission "${STRICT_SAM3_PERMISSION}"
  run_cmd "Phase3 | check" \
    bash scripts/check_phase3.sh \
      --config "${CONFIG}" \
      --exp-id "${P3_EXP}" \
      --strict-sam3-permission "${STRICT_SAM3_PERMISSION}"

  if [[ "${SWITCH_PHASE3_CONDA}" == "true" && "${PHASE3_CONDA_ENV}" != "${CONDA_ENV}" ]]; then
    switch_conda_env "${CONDA_ENV}" "Phase3->Phase4 reset"
  fi
}

phase4_body() {
  run_cmd "Phase4 | run_explore(F)" \
    python3 src/part3/run_explore.py \
      --config "${CONFIG}" \
      --datasets "${DATASETS}" \
      --exp-id "${P4_EXP}" \
      --phase2-exp-id "${P2_EXP}" \
      --stages "${PHASE4_STAGES}" \
      --phase phase4 \
      --seed "${SEED}"
  run_cmd "Phase4 | check" \
    bash scripts/check_phase4.sh --config "${CONFIG}" --exp-id "${P4_EXP}"
}

phase5_body() {
  local cmd=(
    python3 src/part3/run_diffusion.py
    --config "${CONFIG}"
    --datasets "${DATASETS}"
    --exp-id "${P5_EXP}"
    --phase2-exp-id "${P2_EXP}"
    --seed "${SEED}"
    --device "${DEVICE}"
  )
  if [[ "${KEEP_VARIANT_VIDEOS}" == "true" ]]; then
    cmd+=(--keep-variant-videos)
  fi
  run_cmd "Phase5 | run_diffusion(G)" "${cmd[@]}"
  run_cmd "Phase5 | check" \
    bash scripts/check_phase5.sh --config "${CONFIG}" --exp-id "${P5_EXP}"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

CONFIG="configs/base.yaml"
DATASETS="mandatory"
CONDA_ENV="aiaa3201"
ACTIVATE_CONDA="true"
PHASE3_CONDA_ENV="sam3"
SWITCH_PHASE3_CONDA="true"
SEED="42"
DEVICE="cuda"
STRICT_DUAL_RUN="false"
STRICT_SAM3_PERMISSION="true"
PHASE2_STAGES="B1,B2,B3,B4,B5"
PHASE3_STAGES="E1,E2,E3,E4"
PHASE4_STAGES="F1,F2,F3,F4,F5"
START_PHASE="0"
END_PHASE="5"
PREPROCESS_OVERWRITE="true"
KEEP_VARIANT_VIDEOS="false"
SKIP_ENV_CHECKS="false"
DRY_RUN="false"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="${2:-}"
      shift 2
      ;;
    --datasets)
      DATASETS="${2:-}"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="${2:-}"
      shift 2
      ;;
    --activate-conda)
      ACTIVATE_CONDA="$(normalize_bool "${2:-}")"
      shift 2
      ;;
    --phase3-conda-env)
      PHASE3_CONDA_ENV="${2:-}"
      shift 2
      ;;
    --switch-phase3-conda)
      SWITCH_PHASE3_CONDA="$(normalize_bool "${2:-}")"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --device)
      DEVICE="${2:-}"
      shift 2
      ;;
    --strict-dual-run)
      STRICT_DUAL_RUN="$(normalize_bool "${2:-}")"
      shift 2
      ;;
    --strict-sam3-permission)
      STRICT_SAM3_PERMISSION="$(normalize_bool "${2:-}")"
      shift 2
      ;;
    --phase2-stages)
      PHASE2_STAGES="${2:-}"
      shift 2
      ;;
    --phase3-stages)
      PHASE3_STAGES="${2:-}"
      shift 2
      ;;
    --phase4-stages)
      PHASE4_STAGES="${2:-}"
      shift 2
      ;;
    --start-phase)
      START_PHASE="${2:-}"
      shift 2
      ;;
    --end-phase)
      END_PHASE="${2:-}"
      shift 2
      ;;
    --preprocess-overwrite)
      PREPROCESS_OVERWRITE="$(normalize_bool "${2:-}")"
      shift 2
      ;;
    --keep-variant-videos)
      KEEP_VARIANT_VIDEOS="$(normalize_bool "${2:-}")"
      shift 2
      ;;
    --run-tag)
      RUN_TAG="${2:-}"
      shift 2
      ;;
    --skip-env-checks)
      SKIP_ENV_CHECKS="$(normalize_bool "${2:-}")"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="$(normalize_bool "${2:-}")"
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

if [[ -z "${RUN_TAG}" ]]; then
  echo "--run-tag cannot be empty" >&2
  exit 1
fi
if ! [[ "${START_PHASE}" =~ ^[0-5]$ && "${END_PHASE}" =~ ^[0-5]$ ]]; then
  echo "--start-phase and --end-phase must be integers in [0,5]" >&2
  exit 1
fi
if (( START_PHASE > END_PHASE )); then
  echo "--start-phase cannot be greater than --end-phase" >&2
  exit 1
fi

mkdir -p outputs/logs
LOG_FILE="outputs/logs/run_phase0_to5_${RUN_TAG}.log"
SUMMARY_FILE="outputs/logs/run_phase0_to5_${RUN_TAG}_summary.txt"

exec > >(tee -a "${LOG_FILE}") 2>&1

declare -A PHASE_STATUS
declare -A PHASE_DURATION
declare -A PHASE_EXP

ACTIVE_CONDA_ENV=""

P0_EXP="phase0_${RUN_TAG}"
P1_EXP="phase1_${RUN_TAG}"
P2_EXP="phase2_${RUN_TAG}"
P3_EXP="phase3_${RUN_TAG}"
P4_EXP="phase4_${RUN_TAG}"
P5_EXP="phase5_${RUN_TAG}"

log "Repository root: ${REPO_ROOT}"
log "Run tag: ${RUN_TAG}"
log "Exp IDs: P0=${P0_EXP}, P1=${P1_EXP}, P2=${P2_EXP}, P3=${P3_EXP}, P4=${P4_EXP}, P5=${P5_EXP}"
log "Conda activation: ${ACTIVATE_CONDA} (env=${CONDA_ENV})"
log "Phase3 conda switch: ${SWITCH_PHASE3_CONDA} (env=${PHASE3_CONDA_ENV})"
log "Phase range: ${START_PHASE}..${END_PHASE}"

activate_conda_env
preflight

run_phase_if_enabled() {
  local phase_idx="$1"
  local phase_name="$2"
  local exp_id="$3"
  local fn_name="$4"
  if (( phase_idx < START_PHASE || phase_idx > END_PHASE )); then
    PHASE_EXP["${phase_idx}"]="${exp_id}"
    PHASE_STATUS["${phase_idx}"]="SKIPPED"
    PHASE_DURATION["${phase_idx}"]="0"
    log "========== ${phase_name} SKIPPED (exp_id=${exp_id}) =========="
    return 0
  fi
  run_phase "${phase_idx}" "${phase_name}" "${exp_id}" "${fn_name}"
}

run_phase_if_enabled 0 "Phase0" "${P0_EXP}" phase0_body || { write_summary; exit 1; }
run_phase_if_enabled 1 "Phase1" "${P1_EXP}" phase1_body || { write_summary; exit 1; }
run_phase_if_enabled 2 "Phase2" "${P2_EXP}" phase2_body || { write_summary; exit 1; }
run_phase_if_enabled 3 "Phase3" "${P3_EXP}" phase3_body || { write_summary; exit 1; }
run_phase_if_enabled 4 "Phase4" "${P4_EXP}" phase4_body || { write_summary; exit 1; }
run_phase_if_enabled 5 "Phase5" "${P5_EXP}" phase5_body || { write_summary; exit 1; }

write_summary
log "All phases completed successfully."
