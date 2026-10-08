# 📚 Data Splits

📚 This directory contains the training and evaluation data used by
the released code.

## 📁 Files

| File or folder | Purpose |
| --- | --- |
| `MESC-FT/train.jsonl.gz` | Compressed SFT and GRPO training split |
| `MESC-FT/val.jsonl.gz` | Compressed SFT and GRPO validation split |
| `MESC-FT/test.jsonl.gz` | Compressed main MESC-FT evaluation split |
| `MELD-ESC300/test.jsonl.gz` | Compressed external/generalization test split |
| `StickerConv/test.jsonl.gz` | Compressed 300-example external test split |
| `AvaMERG/test.jsonl.gz` | Compressed 300-example external test split |
| `EPITOME-Reddit/emotional-reactions-reddit.csv.gz` | Compressed optional support-skill reward data |
| `EPITOME-Reddit/interpretations-reddit.csv.gz` | Compressed optional support-skill reward data |
| `EPITOME-Reddit/explorations-reddit.csv.gz` | Compressed optional support-skill reward data |

Unpack the compressed data files before running training or evaluation:

```bash
python tools/unpack_data.py
```

Each sample follows a SWIFT-style multimodal conversation format. MESC-FT
training, validation, and test rows use the paper's structured assistant target:

```json
{
  "system": "...",
  "conversations": [
    {
      "from": "user",
      "value": "<image>Client: ...\nTherapist:",
      "images": ["keyframes/dia.../temp_0001.jpg"]
    },
    {
      "from": "assistant",
      "value": "<think>...</think><response>...</response>"
    }
  ]
}
```

Image paths are relative to the repository root. Prepare the referenced
keyframe/frame images under the paths used by the JSONL files before running
multimodal training or evaluation.

After unpacking, the three zero-shot external generalization splits are:

| Split | Examples | Visual input used by the paper protocol |
| --- | ---: | --- |
| `MELD-ESC300/test.jsonl` | 300 | Five locally reconstructed MELD frames |
| `StickerConv/test.jsonl` | 300 | Official textual sticker descriptions and emotion cues; no pixels |
| `AvaMERG/test.jsonl` | 300 | Five locally reconstructed speaker-video frames |

Their assistant targets are response-only `<response>...</response>` entries,
because all three are final-response generalization evaluations rather than
supervised four-part-reasoning splits. External splits are used only for
zero-shot transfer evaluation.

`EPITOME-Reddit/` is not an SFT dataset. It contains support-skill CSV files for
training the optional DIFF-EPITOME reward model in `src/MiniLM_reward_model/`.

`MELD-ESC300/test.jsonl.gz` is already included for generalization evaluation.
If you want to rebuild it from raw MELD test files, use
`src/generalization_data_synthesis/build_meld_esc300.py`.

The processed StickerConv and AvaMERG JSONL splits are also included. See
`StickerConv/README.md` and `AvaMERG/README.md` for their official download
links, licensing notes, exact reported protocol, and optional reconstruction
commands.

## 📦 Local Assets Not Included

The JSONL files reference image paths such as `keyframes/...`,
`data/MELD-ESC300/frames/...`, and `data/AvaMERG/frames/...`. The corresponding
images are external assets and must be prepared locally before multimodal
training or evaluation. StickerConv uses textual sticker cues in the reported
protocol and therefore does not require local sticker pixels.

[Repository overview](../README.md)
