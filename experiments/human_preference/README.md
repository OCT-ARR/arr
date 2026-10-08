# 👥 Human preferences and MELD validity audit

The manuscript compares FACES with FIRES and, separately, directly prompted
GPT-4o on all 450 MESC-FT and 300 MELD-ESC300 instances. Three independent
annotators assess empathy, multimodal appropriateness, strategy suitability,
and helpfulness, making one A/B/Tie decision per item. Reasoning is hidden and
each response is prefix-truncated to at most 15 words.

| Script | Purpose |
| --- | --- |
| [prepare_blind_annotation.py](prepare_blind_annotation.py) | Randomize items and balance candidate positions; write a blinded questionnaire, private key, and empty three-annotator template |
| [analyze_annotations.py](analyze_annotations.py) | Majority vote, fourth-person adjudication, rates including ties, sign test, pre-adjudication Fleiss kappa, and human–judge Cohen kappa |
| [summarize_meld_audit.py](summarize_meld_audit.py) | Three-annotator majority audit for emotional need, relevance, supportive function, and overall validity |

## 🧱 Prepare questionnaires

Generate candidate responses first, then run:

```bash
python experiments/human_preference/prepare_blind_annotation.py --dataset mesc
python experiments/human_preference/prepare_blind_annotation.py --dataset meld
```

For GPT-4o, supply its saved API generations and model identifier:

```bash
python experiments/human_preference/prepare_blind_annotation.py   --dataset mesc --right_name gpt_4o   --right_predictions experiments/api_baselines/outputs/raw/mesc/gpt_4o.jsonl
```

Outputs are under `outputs/human_preference/<dataset>/<left>_vs_<right>/`.
Give annotators the questionnaire and label template, keeping the model-mapping
key private. Fill `annotator_1`, `annotator_2`, and `annotator_3` with `A`, `B`,
or `TIE`. Fill `adjudicated_label` only when all three labels differ.

## 📊 Aggregate and compare with Qwen

```bash
python experiments/human_preference/analyze_annotations.py   --private_key outputs/human_preference/mesc/faces_main_vs_grpo_baseline/private_key.json   --annotations outputs/human_preference/mesc/faces_main_vs_grpo_baseline/annotations_template.jsonl   --judge_records outputs/human_preference/mesc/qwen_cap15_records.jsonl   --output_dir outputs/human_preference/mesc/analysis
```

`--judge_records` is optional and should point to the actual controlled judge
JSONL: each row needs integer `sample_id` and `winner_name` equal to one of
the compared model identifiers or `TIE`. Rows are aligned by sample ID, not
file order. Duplicate/missing records are rejected. Ties remain a category in
exact agreement and unweighted Cohen kappa. Fleiss kappa always uses the three
independent pre-adjudication labels. The default enforces the paper split size;
`--expected_samples 0` explicitly permits a pilot.

## 📝 MELD validity audit

Collect one JSONL row per sample, using three independent boolean votes per
criterion:

```json
{"sample_id": 0, "subset": "strict", "emotional_need": [true, true, false], "relevant_response": [true, true, true], "supportive_function": [true, false, true]}
```

Each criterion uses majority vote. Overall validity requires all three final
criteria. The full audit enforces 300 samples, with 208 strict and 92 relaxed.

```bash
python experiments/human_preference/summarize_meld_audit.py   --input outputs/human_preference/meld_audit_annotations.jsonl
```

[Repository overview](../../README.md)
