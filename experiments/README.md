# 🧪 Experiment Reproduction Guide

🧪 This folder groups all experiment-side code by experiment type. Runtime
outputs, metric tables, API responses, caches, and logs are written under
ignored local paths.

## 📁 Directory Map

| Folder | Purpose |
| --- | --- |
| `ablation/` | Reward-component and contribution-level ablation launchers |
| `automatic_metrics/` | Shared six-metric runner for complete responses, plus reward diagnostics |
| `generation/` | Local candidate generation for downstream evaluation |
| `llm_blind_judge/` | MLLM blind pairwise and controlled judge evaluation |
| `generalization/` | MELD-ESC300, StickerConv, and AvaMERG candidate generation and evaluation |
| `human_preference/` | Full-split blind questionnaires, vote/adjudication analysis, human–judge agreement, and MELD validity audits |
| `sensitivity/` | Method-specific hyperparameter sensitivity launchers |
| `api_baselines/` | Closed/open API model baselines and metrics |
| `significance/` | Paired bootstrap significance tests |
| `reward_diagnostics/` | Four aggregations, independent signals, and reward geometry |
| `token_credit/` | Semantic-span diagnostic and equal-budget credit schedules |
| `efficiency/` | Controlled inference and training-cost benchmarks |
| `optimization_baselines/` | Reconstructed DPO, SimPO, RLOO, and optional PPO controls |
| `reward_reliability/` | ICC and reward-to-human correlation pipeline |

## 🧭 Recommended Order

1. Generate local model outputs with `generation/`.
2. Run automatic metrics with `automatic_metrics/`.
3. Run MLLM blind judging with `llm_blind_judge/`.
4. Run generalization evaluation with `generalization/`.
5. Run ablation launchers with `ablation/`.
6. Run sensitivity variants with `sensitivity/` when reproducing appendix analysis.
7. Use `human_preference/` to prepare questionnaires and aggregate annotations.
8. Run paired bootstrap tests with `significance/`.
9. Run API baselines with `api_baselines/` if API keys are available.
10. Use `optimization_baselines/` for the Table 7 controls and
    `reward_reliability/` for Appendix A.4 statistics.

All commands should be run from the repository root unless a subfolder README
states otherwise.

Keep runtime folders such as `outputs/`, `outputs_emotyda/`, and `logs/` under
the ignored local experiment paths.

See [SUPPLEMENTARY_CODE_MAP.md](SUPPLEMENTARY_CODE_MAP.md) for source mappings and paper locations.

[Repository overview](../README.md)
