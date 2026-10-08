#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_A3="${PROJECT_ROOT}/src/grpo/faces_grpo/train_faces_grpo.py"
MERGE_A3="${PROJECT_ROOT}/src/grpo/faces_grpo/merge_faces_lora.py"

CUDA_MASK="${ABLATION_CUDA_VISIBLE_DEVICES:-0,1}"
MAX_STEPS="${ABLATION_MAX_STEPS:-0}"
EVAL_EVERY="${ABLATION_EVAL_EVERY:-5}"
SAVE_EVERY="${ABLATION_SAVE_EVERY:-5}"
VAL_MAX_SAMPLES="${ABLATION_VAL_MAX_SAMPLES:-0}"
MAX_NEW_TOKENS="${ABLATION_MAX_NEW_TOKENS:-192}"
MAX_FRAMES="${ABLATION_MAX_FRAMES:-16}"

COMMON_ENV=(
  "CUDA_VISIBLE_DEVICES=${CUDA_MASK}"
  "FORCE_COLOR=1"
  "TRANSFORMERS_NO_ADVISORY_WARNINGS=1"
  "HF_HUB_DISABLE_PROGRESS_BARS=1"
  "TQDM_DISABLE=1"
  "PYTHONUNBUFFERED=1"
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
)

usage() {
  cat <<'EOF'
Usage:
  ./run_contribution_ablation.sh
  ./run_contribution_ablation.sh a3_w_o_multi_source_reward
  ./run_contribution_ablation.sh a3_w_o_noncomp_gating
  ./run_contribution_ablation.sh a3_w_o_fine_grained_credit

Environment overrides:
  ABLATION_CUDA_VISIBLE_DEVICES=0,1
  ABLATION_MAX_STEPS=0
  ABLATION_EVAL_EVERY=5
  ABLATION_SAVE_EVERY=5
  ABLATION_VAL_MAX_SAMPLES=0
  ABLATION_MAX_NEW_TOKENS=192
  ABLATION_MAX_FRAMES=16
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

run_experiment() {
  local exp_name="$1"
  local ablation="$2"
  local exp_root="${SCRIPT_DIR}/${exp_name}"
  local checkpoint_dir="${exp_root}/checkpoints"
  local merged_dir="${exp_root}/merged_model"
  local log_dir="${exp_root}/logs"
  local log_file="${log_dir}/train.log"

  mkdir -p "${checkpoint_dir}" "${merged_dir}" "${log_dir}"

  echo "[ABLA-CONTRIB] Running ${exp_name} (ablation=${ablation}) on physical GPUs ${CUDA_MASK}; max_steps=${MAX_STEPS}" | tee -a "${log_file}"
  env "${COMMON_ENV[@]}" \
    "${PYTHON_BIN}" "${TRAIN_A3}" \
    --ablation "${ablation}" \
    --checkpoint_dir "${checkpoint_dir}" \
    --policy_device cuda:0 \
    --reward_device cuda:1 \
    --visual_device cuda:1 \
    --emotion_device cuda:1 \
    --epochs 3 \
    --max_steps "${MAX_STEPS}" \
    --temperature 1.3 \
    --top_p 0.95 \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_frames "${MAX_FRAMES}" \
    --reward_temperature 0.0 \
    --reward_max_new_tokens 512 \
    --eval_every "${EVAL_EVERY}" \
    --save_every "${SAVE_EVERY}" \
    --val_max_samples "${VAL_MAX_SAMPLES}" \
    --resume auto \
    2>&1 | tee -a "${log_file}"

  local best_ckpt
  best_ckpt="$(select_best_checkpoint "${checkpoint_dir}")"
  echo "[ABLA-CONTRIB] Best checkpoint for ${exp_name}: ${best_ckpt}" | tee -a "${log_file}"

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

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

if [[ $# -eq 0 ]]; then
  EXPERIMENTS=(a3_w_o_multi_source_reward a3_w_o_noncomp_gating a3_w_o_fine_grained_credit)
else
  EXPERIMENTS=("$@")
fi

for exp_name in "${EXPERIMENTS[@]}"; do
  case "${exp_name}" in
    a3_w_o_multi_source_reward)
      run_experiment "${exp_name}" "w_o_multi_source_reward"
      ;;
    a3_w_o_noncomp_gating)
      run_experiment "${exp_name}" "w_o_noncomp_gating"
      ;;
    a3_w_o_fine_grained_credit)
      run_experiment "${exp_name}" "w_o_fine_grained_credit"
      ;;
    *)
      echo "Unknown experiment: ${exp_name}" >&2
      usage
      exit 1
      ;;
  esac
done

echo "[ABLA-CONTRIB] All requested contribution-level ablations completed."
