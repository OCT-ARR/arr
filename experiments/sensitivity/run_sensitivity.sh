#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_A3="${PROJECT_ROOT}/src/grpo/faces_grpo/train_faces_grpo.py"

CUDA_MASK="${SENSITIVITY_CUDA_VISIBLE_DEVICES:-0,1}"
MAX_STEPS="${SENSITIVITY_MAX_STEPS:-0}"
VAL_MAX_SAMPLES="${SENSITIVITY_VAL_MAX_SAMPLES:-0}"
OUTPUT_ROOT="${SENSITIVITY_OUTPUT_ROOT:-${SCRIPT_DIR}}"

BASE_MODEL_PATH="${SENSITIVITY_BASE_MODEL_PATH:-${PROJECT_ROOT}/checkpoints/warmup_merged}"
REWARD_MODEL_PATH="${SENSITIVITY_REWARD_MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3-VL-8B-Instruct}"
VISUAL_MODEL_PATH="${SENSITIVITY_VISUAL_MODEL_PATH:-${PROJECT_ROOT}/models/clip-vit-large-patch14}"
EMOTION_MODEL_PATH="${SENSITIVITY_EMOTION_MODEL_PATH:-${PROJECT_ROOT}/models/paraphrase-multilingual-MiniLM-L12-v2}"
TRADITIONAL_REWARD_MODEL_PATH="${SENSITIVITY_TRADITIONAL_REWARD_MODEL_PATH:-${PROJECT_ROOT}/checkpoints/reward_model/roberta_epitome_multitask/final}"
TRAIN_FILE="${SENSITIVITY_TRAIN_FILE:-${PROJECT_ROOT}/data/MESC-FT/train.jsonl}"
VAL_FILE="${SENSITIVITY_VAL_FILE:-${PROJECT_ROOT}/data/MESC-FT/val.jsonl}"

COMMON_ENV=(
  "CUDA_VISIBLE_DEVICES=${CUDA_MASK}"
  "FORCE_COLOR=1"
  "TRANSFORMERS_NO_ADVISORY_WARNINGS=1"
  "HF_HUB_DISABLE_PROGRESS_BARS=1"
  "TQDM_DISABLE=1"
  "PYTHONUNBUFFERED=1"
)

usage() {
  cat <<'EOF'
Usage:
  bash experiments/sensitivity/run_sensitivity.sh
  bash experiments/sensitivity/run_sensitivity.sh visual_threshold
  bash experiments/sensitivity/run_sensitivity.sh gate_floor
  bash experiments/sensitivity/run_sensitivity.sh endpoint_anchor

Environment overrides:
  SENSITIVITY_CUDA_VISIBLE_DEVICES=0,1
  SENSITIVITY_MAX_STEPS=0
  SENSITIVITY_VAL_MAX_SAMPLES=0
  SENSITIVITY_OUTPUT_ROOT=experiments/sensitivity
  SENSITIVITY_BASE_MODEL_PATH=checkpoints/warmup_merged
  SENSITIVITY_REWARD_MODEL_PATH=models/Qwen3-VL-8B-Instruct
  SENSITIVITY_VISUAL_MODEL_PATH=models/clip-vit-large-patch14
  SENSITIVITY_EMOTION_MODEL_PATH=models/paraphrase-multilingual-MiniLM-L12-v2
  SENSITIVITY_TRADITIONAL_REWARD_MODEL_PATH=checkpoints/reward_model/roberta_epitome_multitask/final
EOF
}

run_variant() {
  local name="$1"
  shift
  local out_dir="${OUTPUT_ROOT}/${name}/checkpoints"
  local log_dir="${OUTPUT_ROOT}/${name}/logs"
  mkdir -p "${out_dir}" "${log_dir}"

  echo "[SENS] Running ${name} on physical GPUs ${CUDA_MASK}"
  env "${COMMON_ENV[@]}" \
    "${PYTHON_BIN}" "${TRAIN_A3}" \
    --base_model_path "${BASE_MODEL_PATH}" \
    --reward_model_path "${REWARD_MODEL_PATH}" \
    --visual_model_path "${VISUAL_MODEL_PATH}" \
    --emotion_model_path "${EMOTION_MODEL_PATH}" \
    --traditional_reward_model_path "${TRADITIONAL_REWARD_MODEL_PATH}" \
    --train_file "${TRAIN_FILE}" \
    --val_file "${VAL_FILE}" \
    --checkpoint_dir "${out_dir}" \
    --policy_device cuda:0 \
    --reward_device cuda:1 \
    --visual_device cpu \
    --emotion_device cpu \
    --temperature 1.3 \
    --top_p 0.95 \
    --max_new_tokens 192 \
    --max_frames 16 \
    --reward_temperature 0.0 \
    --reward_max_new_tokens 512 \
    --val_max_samples "${VAL_MAX_SAMPLES}" \
    --resume auto \
    --max_steps "${MAX_STEPS}" \
    "$@" \
    2>&1 | tee -a "${log_dir}/train.log"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

families=("$@")
if [[ ${#families[@]} -eq 0 ]]; then
  families=(visual_threshold gate_floor endpoint_anchor)
fi

for family in "${families[@]}"; do
  case "${family}" in
    visual_threshold)
      for tau in 0.50 0.60 0.65 0.70; do
        run_variant "tau_${tau}" --visual_threshold "${tau}"
      done
      ;;
    gate_floor)
      for gamma in 0.00 0.10 0.30 1.00; do
        run_variant "gamma_${gamma}" --segment_advantage_gate_floor "${gamma}"
      done
      ;;
    endpoint_anchor)
      run_variant "anchor_m3_lambda0.80" --endpoint_anchor_window 3 --endpoint_anchor_decay 0.80
      run_variant "anchor_m5_lambda0.60" --endpoint_anchor_window 5 --endpoint_anchor_decay 0.60
      run_variant "anchor_m5_lambda0.80" --endpoint_anchor_window 5 --endpoint_anchor_decay 0.80
      run_variant "anchor_m7_lambda0.80" --endpoint_anchor_window 7 --endpoint_anchor_decay 0.80
      run_variant "anchor_m5_lambda1.00" --endpoint_anchor_window 5 --endpoint_anchor_decay 1.00
      ;;
    *)
      echo "Unknown sensitivity family: ${family}" >&2
      usage
      exit 1
      ;;
  esac
done

echo "[SENS] Requested sensitivity runs completed."
