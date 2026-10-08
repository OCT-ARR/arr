# 🚀 GRPO Reproduction Guide

🔥 This folder contains the manual multimodal GRPO training code used after SFT. 

For clarity, the GRPO training code is separated into independent subfolders:

```text
src/grpo/
|-- warmup/         # format + outcome warm-up GRPO
`-- faces_grpo/      # full FACES GRPO with the three main algorithmic modules
```

The expected reproduction flow is:

```text
checkpoints/sft_merged
        |
        v
Warm-up GRPO: format + outcome rewards
        |
        v
checkpoints/warmup_merged
        |
        v
FACES GRPO: multi-source rewards + gated fusion + token-level credit assignment
        |
        v
checkpoints/faces_main
```

No trained checkpoints, merged models, generated outputs, judge scores, or logs
are included in the release. Reviewers should train all checkpoints
locally.

## 📁 Directory Map

| Path | Purpose |
| --- | --- |
| `warmup/README.md` | Detailed warm-up GRPO instructions |
| `warmup/run_warmup_grpo.sh` | Convenience launcher for warm-up GRPO |
| `warmup/train_warmup_grpo.py` | Manual GRPO trainer with format and outcome rewards |
| `warmup/merge_warmup_lora.py` | Merge a selected Warm-up LoRA checkpoint |
| `warmup/test_warmup_model.py` | Compare the SFT initialization and Warm-up model |
| `faces_grpo/README.md` | Detailed full FACES GRPO instructions |
| `faces_grpo/run_faces_grpo.sh` | Convenience launcher for the full FACES GRPO stage |
| `faces_grpo/train_faces_grpo.py` | FACES trainer with multi-source rewards, gated fusion, and token-level credit assignment |
| `faces_grpo/merge_faces_lora.py` | Merge a selected FACES LoRA checkpoint |
| `faces_grpo/test_faces_model.py` | Compare the Warm-up model and FACES model |
| `faces_grpo/evaluate_faces_checkpoints.py` | Evaluate and compare local FACES checkpoints |

## 📦 Required Inputs

Before GRPO, complete the SFT stage and merge the selected SFT LoRA checkpoint:

```text
checkpoints/sft_merged/
```

The GRPO stages use these local assets:

```text
data/MESC-FT/train.jsonl
data/MESC-FT/val.jsonl
keyframes/
models/Qwen3-VL-8B-Instruct/
models/clip-vit-large-patch14/
models/paraphrase-multilingual-MiniLM-L12-v2/
```

`Qwen3-VL-8B-Instruct` is used as the MLLM reward judge.
`clip-vit-large-patch14` is used for visual grounding reward.
`paraphrase-multilingual-MiniLM-L12-v2` is used for emotion semantic
similarity. These public models are not committed to the repository.

The optional traditional reward module uses a DIFF-EPITOME style support-skill
reward model:

```text
data/EPITOME-Reddit/
checkpoints/reward_model/roberta_epitome_multitask/final/
```

Train that optional model with `src/MiniLM_reward_model/train_epitome_multitask.py`.

## 📌 Warm-up GRPO

Detailed instructions are in `warmup/README.md`.

Run the convenience launcher from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHON_BIN=python bash src/grpo/warmup/run_warmup_grpo.sh
```

Equivalent explicit command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python src/grpo/warmup/train_warmup_grpo.py \
  --base_model_path checkpoints/sft_merged \
  --reward_model_path models/Qwen3-VL-8B-Instruct \
  --train_file data/MESC-FT/train.jsonl \
  --val_file data/MESC-FT/val.jsonl \
  --checkpoint_dir checkpoints/warmup_grpo \
  --policy_device cuda:0 \
  --reward_device cuda:1
```

Merge a selected Warm-up LoRA checkpoint:

```bash
python src/grpo/warmup/merge_warmup_lora.py \
  --checkpoint-dir "checkpoints/warmup_grpo/checkpoint-step-<step>" \
  --base-model-path checkpoints/sft_merged \
  --output-dir checkpoints/warmup_merged
```

## 📌 FACES GRPO

Detailed instructions are in `faces_grpo/README.md`.

Run the convenience launcher from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHON_BIN=python bash src/grpo/faces_grpo/run_faces_grpo.sh
```

Equivalent explicit command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python src/grpo/faces_grpo/train_faces_grpo.py \
  --base_model_path checkpoints/warmup_merged \
  --reward_model_path models/Qwen3-VL-8B-Instruct \
  --visual_model_path models/clip-vit-large-patch14 \
  --emotion_model_path models/paraphrase-multilingual-MiniLM-L12-v2 \
  --traditional_reward_model_path checkpoints/reward_model/roberta_epitome_multitask/final \
  --train_file data/MESC-FT/train.jsonl \
  --val_file data/MESC-FT/val.jsonl \
  --checkpoint_dir checkpoints/faces_grpo \
  --policy_device cuda:0 \
  --reward_device cuda:1 \
  --visual_device cpu \
  --emotion_device cpu
```

Merge a selected FACES LoRA checkpoint:

```bash
python src/grpo/faces_grpo/merge_faces_lora.py \
  --checkpoint-dir "checkpoints/faces_grpo/checkpoint-step-<step>" \
  --base-model-path checkpoints/warmup_merged \
  --output-dir checkpoints/faces_main
```

## 📊 Evaluate checkpoints

After merging, compare adjacent stages:

```bash
python src/grpo/warmup/test_warmup_model.py
python src/grpo/faces_grpo/test_faces_model.py
```

Checkpoint evaluation can be run locally:

```bash
python src/grpo/faces_grpo/evaluate_faces_checkpoints.py
```

[Repository overview](../../README.md)
