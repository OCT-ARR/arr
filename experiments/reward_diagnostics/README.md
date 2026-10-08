# 🧩 Reward Aggregation and Geometry

Reproduction code for Appendix E.1, Tables 29–30. Run from the repository root.

| File | Purpose |
| --- | --- |
| `generate_scored_candidates.py` | Generate and score candidates with the canonical CLIP/MiniLM/Qwen reward engine; also save independent ROUGE, EPITOME, and MiniLM signals |
| `compare_reward_aggregations.py` | Compare linear, geometric, Tchebycheff, and gated aggregation, top-quartile selection, critical failures, and group maxima |
| `analyze_reward_geometry.py` | Analyze cognitive/critical quadrants, rank correlation, and rank reversals |
| `run_reward_diagnostics.sh` | Run generation and both offline analyses |

Defaults: 369 MESC prompts, five candidates per prompt, seed 42, format threshold
0.5, top/bottom quantile 0.25, high-cognitive threshold 0.6, and low-critical
threshold 0.1. Valid candidate counts are computed from the generated records.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash experiments/reward_diagnostics/run_reward_diagnostics.sh
```

Prepare `checkpoints/faces_main/` and the reward-model paths from
`configs/paths.example.json`. To analyze your own generated records without a GPU:

```bash
python experiments/reward_diagnostics/compare_reward_aggregations.py \
  --in outputs/reward_diagnostics/scored_candidates.jsonl
python experiments/reward_diagnostics/analyze_reward_geometry.py \
  --in outputs/reward_diagnostics/scored_candidates.jsonl
```

[Repository overview](../../README.md)
