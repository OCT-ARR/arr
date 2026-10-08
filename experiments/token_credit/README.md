# 🎯 Semantic-Span and Token-Credit Experiments

Reproduction code for Table 31 and Appendix F.1, Tables 35–36.
Run from the repository root after preparing the warm-up model and reward assets.

| File | Purpose |
| --- | --- |
| `train_credit_variants.sh` | Train endpoint, linear, and exponential schedules with the same data and budget, then merge each selected checkpoint |
| `evaluate_credit_rewards.sh` | Compare training rewards on the same fixed 100 validation instances |
| `evaluate_span_alignment.py` | Compare the full policy and the span-alignment ablation with the five-dimensional diagnostic on 100 validation instances |

The implementations live in `src/grpo/faces_grpo/train_faces_grpo.py`.
`--token_credit_mode` accepts `endpoint`, `linear`, and `exponential`.
For a span of length L, endpoint uses m=5 and lambda=0.8; exponential uses 0.7.
Both alternative shapes are normalized to the same total weight
`min(m,L) + lambda * max(L-m,0)`, as specified in Appendix F.1.
The launcher uses three epochs and two visible GPUs for each variant, without
an elapsed-time cutoff. Legacy credit-ablation names remain accepted.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash experiments/token_credit/train_credit_variants.sh
CREDIT_ENDPOINT_MODEL=checkpoints/token_credit/endpoint/merged_model \
  CUDA_VISIBLE_DEVICES=0,1 bash experiments/token_credit/evaluate_credit_rewards.sh
```

If the complete FACES policy is already available, train only the alternatives:

```bash
bash experiments/token_credit/train_credit_variants.sh linear exponential
bash experiments/token_credit/evaluate_credit_rewards.sh
```

For span alignment, first train `a3_w_o_fine_grained_credit` using
`experiments/ablation/run_contribution_ablation.sh`, then run:

```bash
python experiments/token_credit/evaluate_span_alignment.py --cuda_devices 0,1
```

The five-dimensional diagnostic uses CLIP Visual, MiniLM Emotion, and the
three-prompt Qwen Situation/Strategy/Response scores. Its weights are
15/20/20/20/25. It uses validation data and is separate from the main 1–5
pointwise evaluation.

For the independent 450-example MESC pointwise comparison, generate each
variant and evaluate its response-only cap-15 view:

```bash
python experiments/generalization/generate_external_candidates.py \
  --dataset mesc --faces_model checkpoints/token_credit/linear/merged_model \
  --faces_name linear --output_dir outputs/token_credit/generations
python experiments/llm_blind_judge/pointwise_controlled_eval.py \
  --dataset mesc --model_name linear \
  --predictions outputs/token_credit/generations/linear.jsonl
```

Repeat with the exponential checkpoint and name. For endpoint/FIRES, omit
`--predictions` and choose `--model_name faces_main` or `grpo_baseline`.
All checkpoints, validation indices, generation records, and scores are local runtime artifacts.

[Repository overview](../../README.md)
