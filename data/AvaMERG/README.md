# 📚 AvaMERG External Evaluation Split

This folder packages the 300-example AvaMERG split used in the manuscript for
zero-shot external evaluation in Appendix G. It is an evaluation-only split;
FACES is not trained or fine-tuned on AvaMERG.

## 📌 Included File

| File | Description |
| --- | --- |
| `test.jsonl.gz` | The deterministic 300-example, seed-42 response-generation split used by the released evaluation code |

Run `python tools/unpack_data.py` from the repository root to create
`data/AvaMERG/test.jsonl`.

Assistant targets contain only `<response>...</response>`. Each example refers
to five relative frame paths under `data/AvaMERG/frames/`. The JSONL split is
included, while the upstream video archive and extracted frames remain local
because the official AvaMERG release is large.

## 📥 Official Source And Required Assets

- Project page: <https://avamerg.github.io/>
- Official dataset: <https://huggingface.co/datasets/ZhangHanXD/AvaMERG>
- Official baseline: <https://github.com/AvaMERG/AvaMERG-Pipeline>
- Paper: <https://arxiv.org/abs/2502.04976>

The official Hugging Face release is licensed MIT and is roughly 43 GB in
total. Reconstructing this paper split requires only `test.json` and
`test_video.zip` (the video archive is roughly 2.3 GB), not the training data or
audio archives.

Download the required official files:

```bash
huggingface-cli download ZhangHanXD/AvaMERG \
  test.json test_video.zip \
  --repo-type dataset \
  --local-dir raw/AvaMERG
```

## 🧱 Reconstruct Frames And JSONL

FFmpeg and ffprobe must be available on `PATH`. From the repository root, run:

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

This deterministically selects 300 examples and extracts five uniformly spaced
speaker-video frames per example. Raw archives and generated frames are ignored
by Git; only the compressed protocol-format split is distributed here.

[Repository overview](../../README.md)
