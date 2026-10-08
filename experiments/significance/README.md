# 📐 Statistical significance

`significance_bootstrap.py` computes per-sample automatic-metric caches and
paired bootstrap results for the MESC-FT and MELD-ESC300 comparisons.
Use the same generated responses as the shared evaluation pipeline:

```text
experiments/llm_blind_judge/outputs/generations/faces_main.jsonl
experiments/llm_blind_judge/outputs/generations/grpo_baseline.jsonl
experiments/generalization/outputs_emotyda/generations/faces_main.jsonl
experiments/generalization/outputs_emotyda/generations/grpo_baseline.jsonl
```

```bash
python experiments/significance/significance_bootstrap.py --mode all --include_distinct3
```

Defaults: 5,000 resamples, seed 42, final-layer RoBERTa-large (24), no IDF or
baseline rescaling. `--include_distinct3` adds corpus-level Distinct-3 to
METEOR, BERTScore, and DIFF-EPITOME. Model/path overrides are available via
`--help`. Cache and bootstrap CSV files are ignored runtime outputs.

Pairwise exact sign tests exclude ties and are implemented by the controlled
Qwen and GLM judges. Human preference sign tests are computed by
[analyze_annotations.py](../human_preference/analyze_annotations.py).

[Repository overview](../../README.md)
