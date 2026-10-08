#!/usr/bin/env bash
# Table 7 defaults to DPO/SimPO/RLOO. Pass ppo explicitly to run the optional control.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${BASELINE_CONFIG:-${REPO_ROOT}/configs/optimization_baselines.json}"
PREFERENCES="${PREFERENCE_FILE:-${REPO_ROOT}/outputs/optimization_baselines/preferences.jsonl}"

if [[ "${1:-}" == "--help" ]]; then
  echo 'Usage: run_baselines.sh [dpo simpo rloo [ppo]]'
  exit 0
fi

METHODS=("$@")
if [[ ${#METHODS[@]} -eq 0 ]]; then METHODS=(dpo simpo rloo); fi
for method in "${METHODS[@]}"; do
  case "${method}" in dpo|simpo|rloo|ppo) ;; *) echo "Unknown method: ${method}" >&2; exit 1 ;; esac
done

need_preferences=0
for method in "${METHODS[@]}"; do
  if [[ "${method}" == dpo || "${method}" == simpo ]]; then need_preferences=1; fi
done
manifest="${PREFERENCES%.*}.manifest.json"
if [[ ${need_preferences} -eq 1 ]]; then
  if [[ ! -f "${PREFERENCES}" && ! -f "${manifest}" ]]; then
    "${PYTHON_BIN}" experiments/optimization_baselines/prepare_preferences.py \
      --config "${CONFIG}" --output "${PREFERENCES}"
  elif [[ ! -f "${PREFERENCES}" || ! -f "${manifest}" ]]; then
    echo "Preference data and manifest must either both exist or both be absent." >&2
    exit 1
  fi
fi

for method in "${METHODS[@]}"; do
  method_dir="${REPO_ROOT}/checkpoints/optimization_baselines/${method}"
  if [[ "${method}" == dpo || "${method}" == simpo ]]; then
    "${PYTHON_BIN}" experiments/optimization_baselines/train_preference_baseline.py \
      --method "${method}" --config "${CONFIG}" --preferences "${PREFERENCES}" \
      --output_dir "${method_dir}"
  else
    "${PYTHON_BIN}" experiments/optimization_baselines/train_online_baseline.py \
      --method "${method}" --config "${CONFIG}" --output_dir "${method_dir}"
  fi
  if [[ "${MERGE_BASELINES:-0}" == 1 ]]; then
    "${PYTHON_BIN}" src/grpo/faces_grpo/merge_faces_lora.py \
      --checkpoint-dir "${method_dir}" \
      --output-dir "${REPO_ROOT}/checkpoints/optimization_baselines/${method}_merged"
  fi
done
