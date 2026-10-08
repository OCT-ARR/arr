# 🌍 Generalization Data Synthesis

🌍 This directory contains the optional reconstruction scripts for the three
zero-shot generalization benchmarks described in the submitted paper appendix:
MELD-ESC300, StickerConv, and AvaMERG.

The release already includes the compressed 300-example JSONL split for each
benchmark. Reconstruction recreates visual frames and applies the
deterministic sampling procedure to official raw assets.

## 📁 Files

| File | Purpose |
| --- | --- |
| `build_meld_esc300.py` | Rebuild `data/MELD-ESC300/test.jsonl` and frame paths from raw MELD test files |
| `build_stickerconv_esc.py` | Rebuild the 300-example StickerConv text-cue split used in the manuscript |
| `build_avamerg_esc.py` | Rebuild the 300-example AvaMERG split and extract five speaker-video frames per example |

## 📌 MELD-ESC300

Download the original MELD raw release yourself and place it under:

```text
raw/MELD/MELD.Raw/test_sent_emo.csv
raw/MELD/MELD.Raw/output_repeated_splits_test/
```

Raw MELD videos are not redistributed in this repository.

Rebuild with:

```bash
python src/generalization_data_synthesis/build_meld_esc300.py \
  --csv raw/MELD/MELD.Raw/test_sent_emo.csv \
  --video-dir raw/MELD/MELD.Raw/output_repeated_splits_test \
  --output-dir data/MELD-ESC300 \
  --num-samples 300 \
  --seed 42 \
  --num-frames 5
```

The script follows the paper appendix:

- target response length between 4 and 40 words;
- target sentiment is not negative;
- target emotion is one of `neutral`, `joy`, or `surprise`;
- target speaker differs from the immediately previous speaker;
- visual context comes from the immediately previous utterance video;
- strict samples require negative affect in the immediately previous utterance;
- relaxed samples allow negative affect in one of the three most recent
  preceding utterances;
- five frames are uniformly extracted per sample;
- at most six preceding turns are used as context;
- the random seed is `42`.

Generated frames are saved under `data/MELD-ESC300/frames/`.

## 📌 StickerConv

The included `data/StickerConv/test.jsonl.gz` contains the deterministic
300-example, seed-42 split used in the manuscript. Download the current official
Parquet test split following `data/StickerConv/README.md`, then run:

```bash
python src/generalization_data_synthesis/build_stickerconv_esc.py \
  --src raw/StickerConv/data/test-00000-of-00001.parquet \
  --output-dir data/StickerConv \
  --num-samples 300 \
  --seed 42
```

The paper protocol renders official sticker descriptions and emotion cues as
text. Original sticker pixels are not redistributed and are not required for
this evaluation view.

## 📌 AvaMERG

The included `data/AvaMERG/test.jsonl.gz` contains the deterministic 300-example,
seed-42 split used in the manuscript. Download `test.json` and `test_video.zip`
from the official release as documented in `data/AvaMERG/README.md`, then run:

```bash
python src/generalization_data_synthesis/build_avamerg_esc.py \
  --src raw/AvaMERG/test.json \
  --zip raw/AvaMERG/test_video.zip \
  --output-dir data/AvaMERG \
  --num-samples 300 \
  --seed 42 \
  --num-frames 5 \
  --max-context-turns 6
```

The generated JSONL contains repository-relative frame paths. The upstream
video archive and `data/AvaMERG/frames/` remain local and ignored by Git.

[Repository overview](../../README.md)
