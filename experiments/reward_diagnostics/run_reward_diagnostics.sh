#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
OUT_DIR="${REWARD_DIAGNOSTICS_OUTPUT:-${REPO_ROOT}/outputs/reward_diagnostics}"
mkdir -p "${OUT_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/generate_scored_candidates.py" \
  --policy_model "${FACES_MODEL:-${REPO_ROOT}/checkpoints/faces_main}" \
  --test_data "${DIAGNOSTIC_DATA:-${REPO_ROOT}/data/MESC-FT/test.jsonl}" \
  --limit "${DIAGNOSTIC_PROMPTS:-369}" --k "${DIAGNOSTIC_CANDIDATES:-5}" \
  --seed 42 --policy_device cuda:0 --reward_device cuda:1 \
  --out "${OUT_DIR}/scored_candidates.jsonl" "$@"
"${PYTHON_BIN}" "${SCRIPT_DIR}/compare_reward_aggregations.py" \
  --in "${OUT_DIR}/scored_candidates.jsonl" --out_dir "${OUT_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_reward_geometry.py" \
  --in "${OUT_DIR}/scored_candidates.jsonl" --out_dir "${OUT_DIR}"
