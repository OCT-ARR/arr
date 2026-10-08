# 🚀 FACES GRPO

🚀 This folder contains the full FACES GRPO stage after the warm-up model has
been merged into `checkpoints/warmup_merged/`.

The code implements the three main algorithmic components described in the
paper:

- multi-source semantic rewards for visual grounding, emotion, situation,
  strategy, and outcome;
- non-compensatory gated fusion, where strategy and outcome act as strict
  quality gates;
- structured CoT span alignment with token-level credit assignment.

This stage produces local LoRA checkpoints under `checkpoints/faces_grpo/`.
After selecting a checkpoint, merge it into `checkpoints/faces_main/` for local
evaluation.

## 📁 Files

| File | Purpose |
| --- | --- |
| `run_faces_grpo.sh` | Recommended launcher with repository-relative paths |
| `train_faces_grpo.py` | Full FACES GRPO trainer |
| `merge_faces_lora.py` | Merge a selected FACES LoRA checkpoint into the Warm-up model |
| `test_faces_model.py` | Local qualitative comparison between baseline and FACES models |
| `evaluate_faces_checkpoints.py` | Local checkpoint evaluation script |

## 📦 Required Local Assets

Prepare these paths from the repository root:

```text
checkpoints/warmup_merged/
data/MESC-FT/train.jsonl
data/MESC-FT/val.jsonl
keyframes/
models/Qwen3-VL-8B-Instruct/
models/clip-vit-large-patch14/
models/paraphrase-multilingual-MiniLM-L12-v2/
```

For the optional traditional support-skill reward module, also prepare:

```text
data/EPITOME-Reddit/
checkpoints/reward_model/roberta_epitome_multitask/final/
```

The reward model can be trained locally with
`src/MiniLM_reward_model/train_epitome_multitask.py`.

## ⚙️ Method-Specific Hyperparameters

The trainer exposes the paper's sensitivity-analysis controls:

| Paper symbol | Script argument | Default |
| --- | --- | --- |
| `tau_v` | `--visual_threshold` | `0.60` |
| `gamma` | `--segment_advantage_gate_floor` | `0.10` |
| `m` | `--endpoint_anchor_window` | `5` |
| `lambda` | `--endpoint_anchor_decay` | `0.80` |

The corresponding appendix launcher is
`experiments/sensitivity/run_sensitivity.sh`.

The main GRPO defaults follow the paper setup: batch size `32`, group size `4`,
`3` epochs, learning rate `1e-5`, LoRA rank/alpha/dropout `8/32/0.05`, KL beta
`1e-3`, clip epsilon `0.2`, `4` policy updates per rollout, rollout decoding
with temperature `1.3` and top-p `0.95`, rollout max tokens `192`, maximum
sequence length `1024`, maximum image size `384`, maximum video frames `16`,
and reward-judge max tokens `512`. Validation uses the full validation split by
default; use `--val_max_samples N` only for local smoke tests.

## 🚀 Train

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHON_BIN=python bash src/grpo/faces_grpo/run_faces_grpo.sh
```

The launcher expands to:

```bash
python src/grpo/faces_grpo/train_faces_grpo.py \
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
  --emotion_device cpu \
  --temperature 1.3 \
  --top_p 0.95 \
  --max_new_tokens 192 \
  --max_frames 16 \
  --reward_temperature 0.0 \
  --reward_max_new_tokens 512
```

## 🔗 Merge

Choose the highest `val_avg_total` entry in `checkpoints/faces_grpo/best.txt`.
Replace `<step>` below with the selected checkpoint step before merging.

```bash
python src/grpo/faces_grpo/merge_faces_lora.py \
  --checkpoint-dir "checkpoints/faces_grpo/checkpoint-step-<step>" \
  --base-model-path checkpoints/warmup_merged \
  --output-dir checkpoints/faces_main
```

## 📊 Evaluate checkpoints

```bash
python src/grpo/faces_grpo/test_faces_model.py
python src/grpo/faces_grpo/evaluate_faces_checkpoints.py
```

## ⚙️ Rewards and token credit

For format-valid outputs, `R = mean(Visual, Emotion, Situation) * Strategy * Outcome`.
CLIP provides Visual, MiniLM provides Emotion, and the three-prompt Qwen3-VL
critic provides Situation/Strategy/Outcome. Critic fields must be discrete
1–5 scores; invalid fields receive zero. Format-invalid candidates receive
zero reward and zero reward-driven token advantages.

Dimension advantages use centered descending group ranks with average ties.
Only positive advantages are scaled by the overall-reward gate. The last five
tokens of a semantic span retain full credit, while earlier tokens retain 0.8.
The linear/exponential comparison schedules match the endpoint schedule's
per-span total weight.

The EPITOME checkpoint is required only for the traditional-reward ablation;
it is not loaded for full FACES. See the
[judge guide](../../../experiments/llm_blind_judge/README.md) for reference-free
post-hoc judging and the cap-15 evaluation view.

[Repository overview](../../../README.md)
