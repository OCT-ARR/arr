# 🌍 Generalization Evaluation

🌍 This folder contains candidate-generation and evaluation entry points for
the three 300-example zero-shot external splits: MELD-ESC300, StickerConv, and
AvaMERG.

## 📁 Files

| File | Purpose |
| --- | --- |
| `stickerconv_eval_emotyda.py` | Run the same MLLM judge protocol on `data/MELD-ESC300/test.jsonl` |
| `generate_external_candidates.py` | Generate aligned FACES/FIRES candidates for MELD-ESC300, StickerConv, or AvaMERG |

## 📌 MELD-ESC300

```bash
CUDA_VISIBLE_DEVICES=0,1 python experiments/generalization/stickerconv_eval_emotyda.py
```

For the paper's response-only, length-controlled pairwise setting on the main
and all external splits, run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/llm_blind_judge/pairwise_controlled_eval.py \
  --datasets mesc emotyda stickerconv avamerg \
  --modes length_controlled \
  --max_words 15
```

Prepare these local paths first:

```text
checkpoints/faces_main/
checkpoints/grpo_baseline/   # FIRES (Baseline)
data/MELD-ESC300/test.jsonl
data/MELD-ESC300/frames/
data/StickerConv/test.jsonl
data/AvaMERG/test.jsonl
data/AvaMERG/frames/
models/judge-32b/
```

`grpo_baseline` is the internal checkpoint/output identifier for the paper's
`FIRES (Baseline)` model.

Outputs are written under `experiments/generalization/outputs_emotyda/`.


## 📌 StickerConv And AvaMERG

The compressed protocol-format JSONL splits are included under
`data/StickerConv/` and `data/AvaMERG/`; run `python tools/unpack_data.py` first.
Prepare raw assets when rebuilding splits or creating AvaMERG frames.
Both reconstruction scripts default to 300 examples and seed 42. AvaMERG uses
five frames and up to six context turns. StickerConv preserves official textual
sticker descriptions/emotion cues because the pixels require separate upstream
authorization.

```bash
python src/generalization_data_synthesis/build_stickerconv_esc.py
python src/generalization_data_synthesis/build_avamerg_esc.py
python experiments/generalization/generate_external_candidates.py --dataset stickerconv
python experiments/generalization/generate_external_candidates.py --dataset avamerg
python experiments/llm_blind_judge/pairwise_controlled_eval.py --datasets stickerconv avamerg
python experiments/automatic_metrics/prepare_metric_inputs.py --datasets stickerconv avamerg
python experiments/api_baselines/run_metrics.py --datasets stickerconv,avamerg --models faces,baseline
```

The upstream raw datasets, AvaMERG visual frames, generated candidates, and
metrics remain local. The compressed 300-example protocol-format JSONL splits
are part of this release; official acquisition details are in each dataset's
README under `data/`.
`generate_external_candidates.py` also accepts `--dataset mesc` or `meld` and
model/output overrides for credit-variant evaluations.

[Repository overview](../../README.md)
