# 📏 Reward Reliability

`compute_reliability.py` implements Appendix A.4. It calculates two-way
random-effects absolute-agreement ICC(2,1) and
ICC(2,3) over the three semantically equivalent reward prompts, then Spearman
correlations between reward dimensions and mean human ratings.

The input is JSONL with one candidate per line:

```json
{
  "candidate_id": "candidate-id",
  "prompt_scores": {
    "situation": [0.0, 0.0, 0.0],
    "strategy": [0.0, 0.0, 0.0],
    "outcome": [0.0, 0.0, 0.0]
  },
  "reward_scores": {
    "visual": 0.0, "emotion": 0.0, "situation": 0.0,
    "strategy": 0.0, "outcome": 0.0, "gated_reward": 0.0
  },
  "human_scores": {
    "visual": [1, 1, 1], "emotion": [1, 1, 1],
    "situation": [1, 1, 1], "strategy": [1, 1, 1],
    "outcome": [1, 1, 1], "overall_quality": [1, 1, 1]
  }
}
```

Only the 150-item human subset should contain `human_scores`; every one of the
1,829 format-valid candidates contains `prompt_scores` and `reward_scores`.
The default command enforces both paper counts:

```bash
python experiments/reward_reliability/compute_reliability.py \
  --input outputs/reward_reliability/ratings.jsonl
```

[Repository overview](../../README.md)
