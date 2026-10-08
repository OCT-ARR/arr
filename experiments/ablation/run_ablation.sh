#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_A2="${PROJECT_ROOT}/src/grpo/warmup/train_warmup_grpo.py"
TRAIN_A3="${PROJECT_ROOT}/src/grpo/faces_grpo/train_faces_grpo.py"
MERGE_A2="${PROJECT_ROOT}/src/grpo/warmup/merge_warmup_lora.py"
MERGE_A3="${PROJECT_ROOT}/src/grpo/faces_grpo/merge_faces_lora.py"

CUDA_MASK="${ABLATION_CUDA_VISIBLE_DEVICES:-0,1}"

COMMON_ENV=(
  "CUDA_VISIBLE_DEVICES=${CUDA_MASK}"
  "FORCE_COLOR=1"
  "TRANSFORMERS_NO_ADVISORY_WARNINGS=1"
  "HF_HUB_DISABLE_PROGRESS_BARS=1"
  "TQDM_DISABLE=1"
  "PYTHONUNBUFFERED=1"
)

ALL_EXPERIMENTS=(
  "a2_w_o_format"
  # "a2_w_o_outcome"  # Disabled for now: format is saturated; outcome ablation is low priority.
  "a3_w_o_visual"
  "a3_w_o_emotion"
  "a3_w_o_situation"
  "a3_w_o_strategy"
)

A2_TRAIN_FILE="${PROJECT_ROOT}/data/MESC-FT/train.jsonl"
A3_TRAIN_FILE="${PROJECT_ROOT}/data/MESC-FT/train.jsonl"
A2_BATCH_SIZE=32
A3_BATCH_SIZE=32

merged_model_ready() {
  local merged_dir="$1"
  if [[ ! -d "${merged_dir}" ]]; then
    return 1
  fi
  # "merged_model" is considered ready once it contains any file.
  find "${merged_dir}" -mindepth 1 -maxdepth 1 -type f | read -r _
}

experiment_status() {
  local exp_name="$1"
  local exp_root="${SCRIPT_DIR}/${exp_name}"
  local checkpoint_root="${exp_root}/checkpoints"
  local latest_file="${checkpoint_root}/latest_checkpoint.txt"
  local merged_dir="${exp_root}/merged_model"

  if [[ -f "${latest_file}" ]]; then
    local latest_checkpoint
    latest_checkpoint="$(<"${latest_file}")"
    local latest_base
    latest_base="$(basename "${latest_checkpoint}")"

    # Completed means: trainer wrote a final checkpoint AND we successfully merged.
    if [[ "${latest_base}" == checkpoint-final-* ]] && merged_model_ready "${merged_dir}"; then
      echo "completed"
      return 0
    fi

    # Any non-empty latest checkpoint indicates "started but not completed".
    if [[ -n "${latest_checkpoint}" ]]; then
      echo "incomplete"
      return 0
    fi
  fi

  if [[ -d "${checkpoint_root}" ]] || [[ -d "${merged_dir}" ]]; then
    echo "incomplete"
    return 0
  fi

  echo "missing"
}

select_pending_experiments() {
  SELECTED_EXPERIMENTS=()

  local start_idx=-1
  local idx
  local exp_name
  local status

  for idx in "${!ALL_EXPERIMENTS[@]}"; do
    exp_name="${ALL_EXPERIMENTS[$idx]}"
    status="$(experiment_status "${exp_name}")"
    case "${status}" in
      completed)
        echo "[ABLA] Skipping completed experiment: ${exp_name}"
        ;;
      incomplete|missing)
        start_idx="${idx}"
        if [[ "${status}" == "incomplete" ]]; then
          echo "[ABLA] Resuming unfinished experiment queue from: ${exp_name}"
        else
          echo "[ABLA] Starting pending experiment queue from: ${exp_name}"
        fi
        break
        ;;
      *)
        echo "[ABLA] Unknown status '${status}' for ${exp_name}" >&2
        exit 1
        ;;
    esac
  done

  if (( start_idx < 0 )); then
    echo "[ABLA] All experiments are already completed."
    return 1
  fi

  for (( idx=start_idx; idx<${#ALL_EXPERIMENTS[@]}; idx++ )); do
    SELECTED_EXPERIMENTS+=("${ALL_EXPERIMENTS[$idx]}")
  done
  return 0
}

usage() {
  cat <<'EOF'
Usage:
  ./run_ablation.sh                 # run the default ablation queue
  ./run_ablation.sh list            # list supported experiment names
  ./run_ablation.sh a2_w_o_format   # run one or more selected experiments

Supported experiments:
  a2_w_o_format
  a2_w_o_outcome (disabled)
  a3_w_o_visual
  a3_w_o_emotion
  a3_w_o_situation
  a3_w_o_strategy
EOF
}

select_best_checkpoint() {
  local checkpoint_root="$1"
  "${PYTHON_BIN}" - "$checkpoint_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve()
best_json = root / "best_records.json"
if best_json.exists():
    data = json.loads(best_json.read_text(encoding="utf-8"))
    if isinstance(data, list) and data:
        checkpoint = str(data[0].get("checkpoint", "")).strip()
        if checkpoint:
            print(checkpoint)
            raise SystemExit(0)

latest_file = root / "latest_checkpoint.txt"
if latest_file.exists():
    checkpoint = latest_file.read_text(encoding="utf-8").strip()
    if checkpoint:
        print(checkpoint)
        raise SystemExit(0)

raise SystemExit(f"no checkpoint found under {root}")
PY
}

count_valid_samples() {
  local data_file="$1"
  "${PYTHON_BIN}" - "$data_file" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser().resolve()
count = 0
with path.open("r", encoding="utf-8") as f:
    for raw_line in f:
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        user_text = ""
        gt_full = ""
        for turn in item.get("conversations", []):
            role = turn.get("from")
            if role == "user":
                user_text = str(turn.get("value", ""))
            elif role == "assistant":
                gt_full = str(turn.get("value", ""))
        if user_text and gt_full:
            count += 1
print(count)
PY
}

compute_epoch_steps() {
  local data_file="$1"
  local batch_size="$2"
  "${PYTHON_BIN}" - "$data_file" "$batch_size" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser().resolve()
batch_size = int(sys.argv[2])
count = 0
with path.open("r", encoding="utf-8") as f:
    for raw_line in f:
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        user_text = ""
        gt_full = ""
        for turn in item.get("conversations", []):
            role = turn.get("from")
            if role == "user":
                user_text = str(turn.get("value", ""))
            elif role == "assistant":
                gt_full = str(turn.get("value", ""))
        if user_text and gt_full:
            count += 1
print(int(math.ceil(count / batch_size)))
PY
}

run_a2_experiment() {
  local exp_name="$1"
  local ablation="$2"
  local exp_root="${SCRIPT_DIR}/${exp_name}"
  local checkpoint_dir="${exp_root}/checkpoints"
  local merged_dir="${exp_root}/merged_model"
  local log_dir="${exp_root}/logs"
  local log_file="${log_dir}/train.log"
  local epoch_steps
  epoch_steps="$(compute_epoch_steps "${A2_TRAIN_FILE}" "${A2_BATCH_SIZE}")"

  mkdir -p "${checkpoint_dir}" "${merged_dir}" "${log_dir}"

  echo "[ABLA] Running ${exp_name} on physical GPUs ${CUDA_MASK} (3 epochs)"
  env "${COMMON_ENV[@]}" \
    "${PYTHON_BIN}" "${TRAIN_A2}" \
    --ablation "${ablation}" \
    --checkpoint_dir "${checkpoint_dir}" \
    --policy_device cuda:0 \
    --reward_device cuda:1 \
    --epochs 3 \
    --temperature 1.3 \
    --top_p 0.95 \
    --max_new_tokens 192 \
    --max_frames 16 \
    --reward_temperature 0.0 \
    --reward_max_new_tokens 512 \
    --resume auto \
    2>&1 | tee -a "${log_file}"

  local best_ckpt
  best_ckpt="$(select_best_checkpoint "${checkpoint_dir}")"
  echo "[ABLA] Best checkpoint for ${exp_name}: ${best_ckpt}" | tee -a "${log_file}"

  env "${COMMON_ENV[@]}" \
    "${PYTHON_BIN}" "${MERGE_A2}" \
    --checkpoint-dir "${best_ckpt}" \
    --device cuda:0 \
    --output-dir "${merged_dir}" \
    2>&1 | tee -a "${log_file}"
}

run_a3_experiment() {
  local exp_name="$1"
  local ablation="$2"
  local exp_root="${SCRIPT_DIR}/${exp_name}"
  local checkpoint_dir="${exp_root}/checkpoints"
  local merged_dir="${exp_root}/merged_model"
  local log_dir="${exp_root}/logs"
  local log_file="${log_dir}/train.log"
  local epoch_steps
  local total_steps
  epoch_steps="$(compute_epoch_steps "${A3_TRAIN_FILE}" "${A3_BATCH_SIZE}")"
  total_steps="$((epoch_steps * 3))"

  mkdir -p "${checkpoint_dir}" "${merged_dir}" "${log_dir}"

  echo "[ABLA] Running ${exp_name} (ablation=${ablation}) on physical GPUs ${CUDA_MASK} (3 epochs = ${total_steps} steps)"
  env "${COMMON_ENV[@]}" \
    "${PYTHON_BIN}" "${TRAIN_A3}" \
    --ablation "${ablation}" \
    --checkpoint_dir "${checkpoint_dir}" \
    --policy_device cuda:0 \
    --reward_device cuda:1 \
    --visual_device cuda:1 \
    --emotion_device cuda:1 \
    --epochs 3 \
    --temperature 1.3 \
    --top_p 0.95 \
    --max_new_tokens 192 \
    --max_frames 16 \
    --reward_temperature 0.0 \
    --reward_max_new_tokens 512 \
    --resume auto \
    --val_max_samples 0 \
    2>&1 | tee -a "${log_file}"

  local best_ckpt
  best_ckpt="$(select_best_checkpoint "${checkpoint_dir}")"
  echo "[ABLA] Best checkpoint for ${exp_name}: ${best_ckpt}" | tee -a "${log_file}"

  env "${COMMON_ENV[@]}" \
    "${PYTHON_BIN}" "${MERGE_A3}" \
    --checkpoint-dir "${best_ckpt}" \
    --device cuda:0 \
    --output-dir "${merged_dir}" \
    2>&1 | tee -a "${log_file}"
}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  if ! select_pending_experiments; then
    exit 0
  fi
elif [[ "$1" == "list" ]]; then
  usage
  exit 0
else
  SELECTED_EXPERIMENTS=("$@")
fi

for exp_name in "${SELECTED_EXPERIMENTS[@]}"; do
  case "${exp_name}" in
    a2_w_o_format)
      run_a2_experiment "${exp_name}" "w_o_format"
      ;;
    a2_w_o_outcome)
      echo "[ABLA] Skipping ${exp_name} (temporarily disabled)." >&2
      ;;
    a3_w_o_visual)
      run_a3_experiment "${exp_name}" "w_o_visual"
      ;;
    a3_w_o_emotion)
      run_a3_experiment "${exp_name}" "w_o_emotion"
      ;;
    a3_w_o_situation)
      run_a3_experiment "${exp_name}" "w_o_situation"
      ;;
    a3_w_o_strategy)
      run_a3_experiment "${exp_name}" "w_o_strategy"
      ;;
    *)
      echo "Unknown experiment: ${exp_name}" >&2
      usage
      exit 1
      ;;
  esac
done

echo "[ABLA] All requested experiments completed."
