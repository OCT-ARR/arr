#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

FORCE_COLOR=1 \
TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
HF_HUB_DISABLE_PROGRESS_BARS=1 \
TQDM_DISABLE=1 \
PYTHONUNBUFFERED=1 \
"${PYTHON_BIN}" \
"${SCRIPT_DIR}/train_warmup_grpo.py" \
  --base_model_path "${PROJECT_ROOT}/checkpoints/sft_merged" \
  --reward_model_path "${PROJECT_ROOT}/models/Qwen3-VL-8B-Instruct" \
  --train_file "${PROJECT_ROOT}/data/MESC-FT/train.jsonl" \
  --val_file "${PROJECT_ROOT}/data/MESC-FT/val.jsonl" \
  --checkpoint_dir "${PROJECT_ROOT}/checkpoints/warmup_grpo" \
  --policy_device cuda:0 \
  --reward_device cuda:1 \
  --temperature 1.3 \
  --top_p 0.95 \
  --max_new_tokens 192 \
  --max_frames 16 \
  --reward_temperature 0.0 \
  --reward_max_new_tokens 512 \
  --val_max_samples 0 \
  --resume auto \
  "$@"
