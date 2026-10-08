#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# ================= 1. GPU configuration =================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC="${NPROC:-2}"
MASTER_PORT="${MASTER_PORT:-29505}"

# ================= 2. Path configuration =================
MODEL_PATH="${MODEL_PATH:-${BASE_MODEL_PATH:-model/qwen/Qwen2___5-VL-7B-Instruct}}"
DATA_PATH="${DATA_PATH:-data/MESC-FT/train.jsonl}"
VAL_DATA_PATH="${VAL_DATA_PATH:-data/MESC-FT/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/sft_lora}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ================= 3. Training configuration =================
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
EVAL_STEPS="${EVAL_STEPS:-5}"
SAVE_STEPS="${SAVE_STEPS:-5}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2000}"

echo "[Stage 1] Starting SFT LoRA training"
echo "Visible devices: ${CUDA_VISIBLE_DEVICES}"
echo "Processes: ${NPROC}"
echo "Base model: ${MODEL_PATH}"
echo "Train file: ${DATA_PATH}"
echo "Val file: ${VAL_DATA_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Strategy: evaluate and save every ${SAVE_STEPS} steps; keep saved checkpoints."

command -v torchrun >/dev/null 2>&1 || {
  echo "Missing 'torchrun'. Install PyTorch before running SFT training." >&2
  exit 1
}

"${PYTHON_BIN}" -c "import swift" >/dev/null 2>&1 || {
  echo "Missing ms-swift. Install it with: pip install ms-swift" >&2
  exit 1
}

torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" -m swift.cli.sft \
  --model "${MODEL_PATH}" \
  --model_type qwen2_5_vl \
  --dataset "${DATA_PATH}" \
  --val_dataset "${VAL_DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --target_modules all-linear \
  --warmup_ratio "${WARMUP_RATIO}" \
  --ddp_find_unused_parameters True \
  --predict_with_generate False \
  --logging_steps "${LOGGING_STEPS}" \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --load_best_model_at_end True \
  --metric_for_best_model loss \
  --greater_is_better False \
  "$@"

echo ""
echo "[Warm-up] Generating top-20 checkpoint report from validation loss"

OUTPUT_DIR="${OUTPUT_DIR}" SAVE_STEPS="${SAVE_STEPS}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

base_dir = Path(os.environ["OUTPUT_DIR"])
save_steps = os.environ.get("SAVE_STEPS", "5")
if not base_dir.exists():
    print("Output directory does not exist; skip report generation.")
    raise SystemExit(0)

subdirs = [p for p in base_dir.iterdir() if p.is_dir()]
if not subdirs:
    print("No training run subdirectory found; skip report generation.")
    raise SystemExit(0)

latest_run_dir = max(subdirs, key=lambda p: p.stat().st_mtime)
state_file = latest_run_dir / "trainer_state.json"
output_txt = base_dir / "best.txt"

if not state_file.exists():
    print("trainer_state.json not found; skip report generation.")
    raise SystemExit(0)

with state_file.open("r", encoding="utf-8") as f:
    data = json.load(f)

history = data.get("log_history", [])
eval_records = [x for x in history if "eval_loss" in x]
eval_records.sort(key=lambda x: x["eval_loss"])
top_20 = eval_records[:20]

with output_txt.open("w", encoding="utf-8") as f:
    f.write(f"Checkpoint Parent Location: {latest_run_dir}\n")
    f.write(f"Strategy: saved every {save_steps} steps. Showing top 20 by validation loss.\n")
    f.write("=" * 72 + "\n")
    f.write(f"{'Rank'.ljust(6)} | {'Step'.ljust(8)} | {'Validation Loss'.ljust(15)} | {'Folder Exists?'.ljust(12)}\n")
    f.write("=" * 72 + "\n")
    for i, record in enumerate(top_20):
        step = record.get("step")
        loss = record.get("eval_loss")
        ckpt_path = latest_run_dir / f"checkpoint-{step}"
        exists = "Yes" if ckpt_path.exists() else "No"
        f.write(f"{str(i + 1).ljust(6)} | {str(step).ljust(8)} | {loss:.6f}        | {exists}\n")

print(f"Top-20 checkpoint report saved to: {output_txt}")
print(f"Use the reported step to select a checkpoint under: {latest_run_dir}")
PY
