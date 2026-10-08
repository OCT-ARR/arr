# ⚖️ MLLM Blind Judge Evaluation

⚖️ This folder contains blind MLLM judge evaluation for MESC-FT model outputs.
It can generate candidate outputs, run pointwise/pairwise judging, and run
controlled pairwise views over existing generated JSONL files.

## 📁 Files

| File | Purpose |
| --- | --- |
| `stickerconv_eval.py` | Main MESC-FT MLLM judge evaluator |
| `pairwise_controlled_eval.py` | Controlled pairwise views over existing JSONL generations |
| `pointwise_controlled_eval.py` | Response-only cap-15 three-dimensional pointwise evaluation |
| `cross_family_judge.py` | GLM fixed-candidate cross-family pairwise evaluation |

## 🚀 Run Main Judge

```bash
CUDA_VISIBLE_DEVICES=0,1 python experiments/llm_blind_judge/stickerconv_eval.py
```

## 🚀 Run Controlled Pairwise Views

Use this command for the paper's main response-only, length-controlled MLLM
preference setting:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/llm_blind_judge/pairwise_controlled_eval.py \
  --datasets mesc emotyda stickerconv avamerg \
  --modes length_controlled \
  --max_words 15
```

The default `stickerconv_eval.py` entry point is kept for local full-context
pointwise/pairwise diagnostics. Appendix-style controlled views can be rerun by
changing `--max_words` to `25` or `30`, or by adding `response_only` to
`--modes`.

Prepare these local paths first:

```text
checkpoints/faces_main/
checkpoints/grpo_baseline/   # FIRES (Baseline)
data/MESC-FT/test.jsonl
data/MELD-ESC300/test.jsonl
data/StickerConv/test.jsonl
data/AvaMERG/test.jsonl
keyframes/
models/judge-32b/
```

`grpo_baseline` is the internal checkpoint/output identifier for the paper's
`FIRES (Baseline)` model.

Outputs are written under `experiments/llm_blind_judge/outputs/`.


## 📊 Supplementary Controlled Evaluation

`pointwise_controlled_eval.py` evaluates response-only cap-15 views with the
shared three-prompt 1–5 judge. `cross_family_judge.py` runs the same blind
pairwise comparison with GLM-4.6V-Flash through an API; its default sample limit
is zero (full split). Configure `ZHIPU_API_KEY` and `ZHIPU_BASE_URL`
in your local environment. GLM uses temperature zero, disabled thinking,
768 output tokens, and at most eight input images.

```bash
python experiments/llm_blind_judge/pointwise_controlled_eval.py --dataset mesc
python experiments/llm_blind_judge/pointwise_controlled_eval.py --dataset meld
python experiments/llm_blind_judge/cross_family_judge.py --dataset mesc
python experiments/llm_blind_judge/cross_family_judge.py --dataset stickerconv
python experiments/llm_blind_judge/cross_family_judge.py --dataset avamerg
```

The controlled pairwise dataset registry includes all four datasets.
For another opponent, both Qwen and API pairwise entries accept
`--right_name` and `--right_generation_file`; API generation files must include
`source_index` to align with local sample IDs. Each pair gets a distinct output
filename or folder. Full-context, response-only, cap-15, and cap-25 views are
available by selecting the mode and word cap. All generated judge records and
summaries stay in ignored output folders.

Post-hoc judges receive no reference annotations. Main pointwise/pairwise
views remove both reasoning fields and reasoning-related criteria. Each
response is independently prefix-capped; this is not forced equal length.
Win rates include ties in the denominator and sign tests exclude them.
Invalid/unparsable judge output is treated as Tie. Human–Qwen
agreement is analyzed with the tools in `experiments/human_preference/`.

Dataset aliases: `meld` and `emotyda` refer to MELD-ESC300 in the controlled
registry; the metric/API registry calls the same split `meld_esc300`.

[Repository overview](../../README.md)
