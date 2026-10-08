# 🎬 Image Data Synthesis

🎬 This directory contains optional utilities for rebuilding the local visual
assets referenced by `data/MESC-FT/*.jsonl`.

The release includes the JSONL splits but not copyrighted videos,
clips, or extracted keyframes. Reviewers who already have the required visual
assets can skip this folder and place frames directly under `keyframes/`.

## 📁 Files

| File | Purpose |
| --- | --- |
| `process_data.py` | Cut dialogue-history video clips from raw episode videos and source annotations |
| `video_to_keyframes.py` | Extract I-frame keyframes from the generated clips |
| `convert_data.py` | Convert intermediate data into the multimodal JSONL format used by training |

## 📦 Expected Inputs

```text
raw/MESC-FT/MESC.csv
raw/MESC-FT/videos/
```

## 🧱 Rebuild Clips

```bash
python src/image_data_synthesis/process_data.py \
  --csv_file raw/MESC-FT/MESC.csv \
  --video_dir raw/MESC-FT/videos \
  --output_dir clips \
  --gpu_ids 0,1 \
  --workers_per_slot 4
```

## 🧱 Extract Keyframes

```bash
python src/image_data_synthesis/video_to_keyframes.py \
  --input_dir clips \
  --output_dir keyframes
```

## 🧱 Optional Format Conversion

If you rebuild an intermediate synthesis JSON locally, convert it to the JSONL
conversation format with:

```bash
python src/image_data_synthesis/convert_data.py \
  --input_file raw/MESC-FT/synthesis_data.json \
  --output_file data/MESC-FT/converted.jsonl \
  --image_root .
```

The released training splits are already provided as
`data/MESC-FT/{train,val,test}.jsonl`, so this conversion step is not required
for normal SFT or GRPO reproduction.

The `raw/`, `clips/`, and `keyframes/` directories are local visual assets and
are covered by `.gitignore`.

[Repository overview](../../README.md)
