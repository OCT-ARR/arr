# 📚 StickerConv External Evaluation Split

This folder packages the 300-example StickerConv split used in the manuscript
for zero-shot external evaluation in Appendix G. It is an evaluation-only
split; FACES is not trained or fine-tuned on StickerConv.

## 📌 Included File

| File | Description |
| --- | --- |
| `test.jsonl.gz` | The deterministic 300-example, seed-42 response-generation split used by the released evaluation code |

Run `python tools/unpack_data.py` from the repository root to create
`data/StickerConv/test.jsonl`.

Each row uses the same `system` plus two-turn `conversations` schema as the
other released splits. Assistant targets contain only
`<response>...</response>`. The paper protocol uses the official textual
sticker descriptions and emotion cues inline in the dialogue; `images` is
therefore empty and no fake `<image>` token is inserted.

## 📥 Official Source And Visual Assets

- Project and code: <https://github.com/ZhangYiqun018/StickerConv>
- Official LLM-formatted release: <https://huggingface.co/datasets/Estwld/StickerConv_llm>
- Sticker-image access procedure: <https://github.com/nku-shengzheliu/SER30K>
- Paper: <https://aclanthology.org/2024.acl-long.417/>

The LLM-formatted release is marked Apache-2.0. The official StickerConv
repository states that sticker pixels are not bundled
because of the underlying SER30K licensing restrictions. They are not required
for the text-cue protocol reported in the paper. Users who need the original
pixels must follow the SER30K access procedure and the upstream licenses.

## 🧱 Optional Reconstruction

Download the official Parquet test split:

```bash
huggingface-cli download Estwld/StickerConv_llm \
  data/test-00000-of-00001.parquet \
  --repo-type dataset \
  --local-dir raw/StickerConv
```

Then rebuild the same protocol-format split:

```bash
python src/generalization_data_synthesis/build_stickerconv_esc.py \
  --src raw/StickerConv/data/test-00000-of-00001.parquet \
  --output-dir data/StickerConv \
  --num-samples 300 \
  --seed 42
```

The reconstruction script also accepts the older official line-delimited JSON
export at `raw/StickerConv/test.json`. It prioritizes conversations with
negative-affect sticker annotations, then deterministically fills the split to
300 examples. Raw upstream files remain local under the ignored `raw/`
directory.

[Repository overview](../../README.md)
