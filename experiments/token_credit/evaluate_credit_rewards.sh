#!/usr/bin/env bash
# Evaluate all three merged policies on the same fixed 100 validation examples.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TRAINER="${REPO_ROOT}/src/grpo/faces_grpo/train_faces_grpo.py"
CREDIT_ROOT="${CREDIT_CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/token_credit}"
OUT_DIR="${CREDIT_EVAL_OUTPUT:-${REPO_ROOT}/outputs/token_credit}"
mkdir -p "${OUT_DIR}"
for variant in endpoint linear exponential; do
  if [[ "${variant}" == endpoint ]]; then
    policy_path="${CREDIT_ENDPOINT_MODEL:-${REPO_ROOT}/checkpoints/faces_main}"
  else
    policy_path="${CREDIT_ROOT}/${variant}/merged_model"
  fi
  "${PYTHON_BIN}" "${TRAINER}" \
    --base_model_path "${policy_path}" --ablation full --token_credit_mode "${variant}" \
    --val_file "${REPO_ROOT}/data/MESC-FT/val.jsonl" \
    --checkpoint_dir "${OUT_DIR}/${variant}/checkpoints" \
    --policy_device cuda:0 --reward_device cuda:1 --visual_device cuda:1 --emotion_device cuda:1 \
    --val_max_samples 100 --val_index_file "${OUT_DIR}/validation_indices_100.json" \
    --reward_temperature 0 --reward_max_new_tokens 512 \
    --seed 42 --eval_only --resume none \
    2>&1 | tee "${OUT_DIR}/${variant}_validation.log"
done
