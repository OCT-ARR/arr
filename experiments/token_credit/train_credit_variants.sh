#!/usr/bin/env bash
# Equal training budgets for endpoint, linear, and exponential credit schedules.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TRAINER="${REPO_ROOT}/src/grpo/faces_grpo/train_faces_grpo.py"
MERGER="${REPO_ROOT}/src/grpo/faces_grpo/merge_faces_lora.py"
BASE_MODEL="${CREDIT_BASE_MODEL:-${REPO_ROOT}/checkpoints/warmup_merged}"
CREDIT_ROOT="${CREDIT_CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/token_credit}"
if [[ "${1:-}" == "--help" ]]; then
  echo 'Usage: train_credit_variants.sh [endpoint linear exponential]'
  exit 0
fi
VARIANTS=("$@")
if [[ ${#VARIANTS[@]} -eq 0 ]]; then VARIANTS=(endpoint linear exponential); fi
for variant in "${VARIANTS[@]}"; do
  case "${variant}" in endpoint|linear|exponential) ;; *) echo "Unknown credit mode: ${variant}" >&2; exit 1 ;; esac
  exp_root="${CREDIT_ROOT}/${variant}"
  mkdir -p "${exp_root}/logs"
  "${PYTHON_BIN}" "${TRAINER}" \
    --base_model_path "${BASE_MODEL}" \
    --ablation full --token_credit_mode "${variant}" \
    --train_file "${REPO_ROOT}/data/MESC-FT/train.jsonl" \
    --val_file "${REPO_ROOT}/data/MESC-FT/val.jsonl" \
    --checkpoint_dir "${exp_root}/checkpoints" \
    --policy_device cuda:0 --reward_device cuda:1 \
    --visual_device cuda:1 --emotion_device cuda:1 \
    --epochs "${CREDIT_EPOCHS:-3}" --seed 42 \
    --temperature 1.3 --top_p 0.95 --max_new_tokens 192 --max_frames 16 \
    --reward_temperature 0 --reward_max_new_tokens 512 \
    --endpoint_anchor_window 5 --endpoint_anchor_decay 0.8 \
    --val_max_samples 100 --resume auto \
    2>&1 | tee "${exp_root}/logs/train.log"
  selected_checkpoint="$("${PYTHON_BIN}" - "${exp_root}/checkpoints" <<'SELECT'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
records = json.loads((root / 'best_records.json').read_text()) if (root / 'best_records.json').exists() else []
if records:
    print(records[0]['checkpoint'])
elif (root / 'latest_checkpoint.txt').exists():
    print((root / 'latest_checkpoint.txt').read_text().strip())
else:
    raise SystemExit('No checkpoint found')
SELECT
)"
  "${PYTHON_BIN}" "${MERGER}" --checkpoint-dir "${selected_checkpoint}" \
    --base-model-path "${BASE_MODEL}" --output-dir "${exp_root}/merged_model" --device cuda:0
done
