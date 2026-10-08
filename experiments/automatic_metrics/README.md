# 📊 Automatic metrics

The main metric set is BLEU-2, ROUGE-L, METEOR, BERTScore-F1, Distinct-3,
and DIFF-EPITOME. Evaluate complete final responses; cap-15 applies to the
main MLLM/human comparisons, not these automatic metrics.

| Script | Purpose |
| --- | --- |
| [prepare_metric_inputs.py](prepare_metric_inputs.py) | Align saved local predictions with dataset-derived IDs and export shared metric inputs |
| [run_metrics.py](run_metrics.py) | Recommended six-metric entry point for both local and API candidates |
| [eval_faces.py](eval_faces.py) | Combined local FACES generation and metric/reward diagnostics |
| [eval_sft.py](eval_sft.py) | Combined SFT generation and evaluation |
| [auto_eval_faces.py](auto_eval_faces.py) | Batch checkpoint diagnostics |
| [test_model_performance.py](test_model_performance.py) | Inspect one sample qualitatively |

## 🧭 Recommended pipeline

Generate candidates using `experiments/generalization/generate_external_candidates.py`,
then run:

```bash
python experiments/automatic_metrics/prepare_metric_inputs.py   --datasets mesc meld_esc300 stickerconv avamerg
python experiments/automatic_metrics/run_metrics.py --datasets all --models faces,baseline
```

The runner reuses the API pipeline's shared metric implementation. Inputs are
under `experiments/api_baselines/outputs/raw/<dataset>/`; outputs are under
its ignored `outputs/` and `reports/` directories. API opponent generations
can be evaluated directly with the same runner.

BERTScore defaults to final-layer RoBERTa-large (24), without IDF or baseline
rescaling. Distinct-3 is corpus-level. DIFF-EPITOME uses MSE between candidate
and reference RoBERTa support-skill score vectors. Prepare that model with
`src/MiniLM_reward_model/` and download NLTK resources before running METEOR.

```bash
python -m nltk.downloader -d nltk_data wordnet omw-1.4 punkt punkt_tab
```

A Qwen critic diagnostic from `eval_faces.py --reward_backend critic` is a
separate diagnostic and must not be relabelled as DIFF-EPITOME. To inspect a
local sample:

```bash
python experiments/automatic_metrics/test_model_performance.py   --model_path checkpoints/faces_main --test_file data/MESC-FT/test.jsonl   --gpus 0 --sample_idx 0
```

[Repository overview](../../README.md)
