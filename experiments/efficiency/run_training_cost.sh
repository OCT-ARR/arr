#!/usr/bin/env bash
# Reproduce the paper's controlled 100-step FIRES/FACES training-cost protocol.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAINER="${REPO_ROOT}/src/grpo/faces_grpo/train_faces_grpo.py"
OUTPUT="${REPO_ROOT}/outputs/efficiency/training_cost.json"

common=(--base_model_path "${REPO_ROOT}/checkpoints/warmup_merged"
  --train_file "${REPO_ROOT}/data/MESC-FT/train.jsonl"
  --policy_device cuda:0 --reward_device cuda:1
  --batch_size 32 --group_size 4 --max_new_tokens 192 --max_steps 100 --seed 42)

"${PYTHON_BIN}" experiments/efficiency/benchmark_training.py \
  --name FIRES --gpus "${TRAINING_GPUS:-0,1}" --stable_steps 100 --output "${OUTPUT}" -- \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/train_fires_grpo.py" "${common[@]}" \
  --updates_per_rollout 4 \
  --reward_model_path "${REPO_ROOT}/checkpoints/reward_model/epitome" \
  --checkpoint_dir "${REPO_ROOT}/checkpoints/training_cost/fires"

"${PYTHON_BIN}" experiments/efficiency/benchmark_training.py \
  --name FACES --gpus "${TRAINING_GPUS:-0,1}" --stable_steps 100 --output "${OUTPUT}" -- \
  "${PYTHON_BIN}" "${TRAINER}" "${common[@]}" --ablation full --resume none \
  --grpo_updates_per_rollout 4 \
  --val_file "${REPO_ROOT}/data/MESC-FT/val.jsonl" --visual_device cuda:1 --emotion_device cuda:1 \
  --checkpoint_dir "${REPO_ROOT}/checkpoints/training_cost/faces"
