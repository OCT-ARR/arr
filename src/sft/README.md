# 🧱 SFT Reproduction Guide

✨ This folder contains the supervised fine-tuning entry points used to build
the initial FACES model before GRPO training. The release provides
the clean JSONL splits. Public base models and raw visual assets should be
prepared locally.

## 🚦 Stage Overview

```text
Download base model
        v
Prepare copyrighted image assets locally
        v
Check data/MESC-FT/{train,val,test}.jsonl
        v
Run SFT LoRA training
        v
Select a checkpoint by validation loss
        v
Merge LoRA into Qwen2.5-VL-7B-Instruct
        v
Use checkpoints/sft_merged for the GRPO stages
```

## 📁 Files Used In This Stage

| Purpose | File or directory |
| --- | --- |
| SFT training script | `src/sft/run_sft.sh` |
| SFT LoRA merge script | `src/sft/merge_sft_lora.py` |
| Optional raw-video clip cutting | `src/image_data_synthesis/process_data.py` |
| Optional keyframe extraction | `src/image_data_synthesis/video_to_keyframes.py` |
| Optional data-format conversion | `src/image_data_synthesis/convert_data.py` |
| Provided SFT splits | `data/MESC-FT/train.jsonl`, `data/MESC-FT/val.jsonl`, `data/MESC-FT/test.jsonl` |
| Local base model | `model/qwen/Qwen2___5-VL-7B-Instruct/` |
| Local keyframes | `keyframes/` |
| Generated LoRA checkpoints | `checkpoints/sft_lora/` |
| Generated merged SFT model | `checkpoints/sft_merged/` |

The `data/` directory is shared by all stages and should not be duplicated under
`src/sft/`.

## 📥 1. Download The Base Model

Download the public Qwen2.5-VL-7B-Instruct base model yourself. The default
local path used by the scripts is:

```text
model/qwen/Qwen2___5-VL-7B-Instruct/
```

For example, with Hugging Face:

```bash
mkdir -p model/qwen
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir model/qwen/Qwen2___5-VL-7B-Instruct
```

You may also use a remote model id directly when running training:

```bash
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct bash src/sft/run_sft.sh
```

No base-model files should be committed to the repository.

## 🧱 2. Prepare MESC-FT Image Assets

The JSONL text splits for MESC-FT are already included:

```text
data/MESC-FT/train.jsonl
data/MESC-FT/val.jsonl
data/MESC-FT/test.jsonl
```

The corresponding image/video assets are not included because of copyright
restrictions. Reviewers should obtain the original MESC-FT source assets
separately and place the final extracted frames at the paths referenced by the
JSONL files, for example:

```text
keyframes/dia10utt5/temp_0001.jpg
keyframes/dia10utt5/temp_0002.jpg
...
```

If the keyframes are already available, place them directly under `keyframes/`
at the repository root.

## 📌 3. Optional: Rebuild Keyframes From Raw Videos

If starting from raw MESC-FT episode videos and the source annotation CSV, use
the provided scripts to reproduce the local visual assets.

Expected raw layout:

```text
raw/MESC-FT/MESC.csv
raw/MESC-FT/videos/S01E01.mp4
raw/MESC-FT/videos/S01E02.mp4
...
```

Cut dialogue-history clips:

```bash
python src/image_data_synthesis/process_data.py \
  --csv_file raw/MESC-FT/MESC.csv \
  --video_dir raw/MESC-FT/videos \
  --output_dir clips \
  --gpu_ids 0,1 \
  --workers_per_slot 4
```

Extract I-frame keyframes:

```bash
python src/image_data_synthesis/video_to_keyframes.py \
  --input_dir clips \
  --output_dir keyframes
```

The scripts produce keyframe directories named like `keyframes/dia0utt1/`,
which match the relative image paths used by `data/MESC-FT/*.jsonl`.

## 🚀 4. Run SFT LoRA Training

Install dependencies from the repository root first:

```bash
pip install -r requirements.txt
```

Then run the default two-GPU SFT configuration:

```bash
bash src/sft/run_sft.sh
```

The default paths are:

| Variable | Default |
| --- | --- |
| `MODEL_PATH` | `model/qwen/Qwen2___5-VL-7B-Instruct` |
| `DATA_PATH` | `data/MESC-FT/train.jsonl` |
| `VAL_DATA_PATH` | `data/MESC-FT/val.jsonl` |
| `OUTPUT_DIR` | `checkpoints/sft_lora` |
| `CUDA_VISIBLE_DEVICES` | `0,1` |
| `NPROC` | `2` |

The default hyperparameters are:

| Hyperparameter | Value |
| --- | --- |
| Epochs | `3` |
| Train batch size per GPU | `1` |
| Eval batch size per GPU | `1` |
| Gradient accumulation | `16` |
| Learning rate | `5e-5` |
| LoRA rank / alpha / dropout | `8 / 32 / 0.05` |
| Target modules | `all-linear` |
| Warmup ratio | `0.05` |
| Logging steps | `1` |
| Eval steps | `5` |
| Save steps | `5` |
| Save total limit | `2000` |
| Best-model metric | validation loss |

All defaults can be overridden from the command line:

```bash
CUDA_VISIBLE_DEVICES=0 NPROC=1 \
MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct \
OUTPUT_DIR=checkpoints/sft_lora_single_gpu \
bash src/sft/run_sft.sh
```

The script writes a validation-loss checkpoint ranking to:

```text
checkpoints/sft_lora/best.txt
```

Use this file to select the SFT LoRA checkpoint for merging.

## 🔗 5. Merge The Selected LoRA Checkpoint

After choosing the best SFT LoRA checkpoint, merge it into the base model:

```bash
python src/sft/merge_sft_lora.py \
  --base_model_path model/qwen/Qwen2___5-VL-7B-Instruct \
  --adapter_path "checkpoints/sft_lora/<run_name>/checkpoint-<step>" \
  --output_dir checkpoints/sft_merged
```

The merged model in `checkpoints/sft_merged/` is the initialization used by the
later GRPO stages.

## 💡 Not Required For SFT

The following assets are used only after SFT, during FACES GRPO or ablation
experiments:

- `models/paraphrase-multilingual-MiniLM-L12-v2/`
- `models/Qwen3-VL-8B-Instruct/`
- `models/clip-vit-large-patch14/`
- `data/EPITOME-Reddit/`
- `checkpoints/reward_model/roberta_epitome_multitask/final/`

MiniLM is a public sentence-transformer used by the emotion semantic
reward in FACES GRPO. `data/EPITOME-Reddit/` is for the optional support-skill
reward model, not for SFT and not for training MiniLM. Neither MiniLM nor the
EPITOME reward model is needed to complete the SFT stage.

[Repository overview](../../README.md)
