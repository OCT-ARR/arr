# Paper-to-code map

Run commands from the repository root. Section names refer to the current
submitted manuscript. Shared protocol values are in
[paper_protocols.json](../configs/paper_protocols.json).

| Paper experiment | Implementation | Paper location |
| --- | --- | --- |
| SFT initialization | [SFT](../src/sft/README.md) | Experimental Setup: Training Configuration |
| Format-and-outcome warm-up | [warm-up trainer](../src/grpo/warmup/train_warmup_grpo.py) | Methodology: Gated Reward Fusion |
| Five rewards, fusion, positive gate, span credit, clipped GRPO | [FACES trainer](../src/grpo/faces_grpo/train_faces_grpo.py) | Methodology |
| Complete-response automatic metrics | [metric runner](automatic_metrics/run_metrics.py), [shared metrics](api_baselines/api_baselines/metrics.py) | Main Results; External Dataset |
| Greedy local candidate generation | [generation](generalization/generate_external_candidates.py) | Inference and Evaluation Settings |
| Cap-15 pointwise judging | [controlled pointwise](llm_blind_judge/pointwise_controlled_eval.py) | Main Results; Credit Assignment Alternatives |
| Qwen cap-15/cap-25/cap-30 and response-only judging | [controlled pairwise](llm_blind_judge/pairwise_controlled_eval.py) | Controlled Evaluation |
| GLM cross-family fixed-candidate comparison | [cross-family judge](llm_blind_judge/cross_family_judge.py) | Cross-Judge Robustness |
| Six directly prompted MLLMs | [API baselines](api_baselines/README.md) | Main Results; Strong Direct-Prompted MLLMs |
| Blind human questionnaires, majority vote, adjudication, Fleiss/Cohen kappa | [human tools](human_preference/README.md) | Detailed Preference Evaluation |
| Contribution and dimension ablations | [ablation](ablation/README.md) | Ablation Study; Additional Ablation Analysis |
| Four reward aggregation rules and geometry | [reward diagnostics](reward_diagnostics/README.md) | Reward Aggregation Diagnostics |
| Span-alignment diagnostic | [span diagnostic](token_credit/evaluate_span_alignment.py) | Fine-grained Credit Diagnostic |
| Equal-total-weight endpoint/linear/exponential credit | [token credit](token_credit/README.md) | Credit Assignment Alternatives |
| Visual threshold, positive gate floor, endpoint sensitivity | [sensitivity](sensitivity/README.md) | Hyperparameter Sensitivity Analysis |
| DPO/SimPO/RLOO controls | [optimization baselines](optimization_baselines/README.md) | Optimization; Training Configuration |
| Prompt ICC and reward–human correlation | [reliability](reward_reliability/README.md) | Reward Reliability and Cost |
| Inference and training costs | [efficiency](efficiency/README.md) | Reward Reliability and Cost |
| Paired bootstrap for automatic metrics | [significance](significance/README.md) | Generalization Analysis; Statistical Significance |
| MELD, StickerConv, AvaMERG reconstruction | [builders](../src/generalization_data_synthesis/README.md) | External Dataset |
| MELD expert validity audit | [audit summary](human_preference/summarize_meld_audit.py) | Benchmark Construction |
| Four figures and PDF previews | [figures](../figures/README.md) | Figures 1–4 |

Algorithm-specific optimization defaults are configurable. The training-cost
benchmark uses separate FIRES and FACES entry points with measured step timings. See the
[optimization guide](optimization_baselines/README.md) and
[efficiency guide](efficiency/README.md) for the corresponding settings.

Only MESC-FT is used for training. Each external compressed test split has 300
examples; human annotation covers MESC-FT and MELD-ESC300 only. Human analyses
require collecting annotations in the documented input formats.
