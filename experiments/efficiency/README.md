# ⏱️ Efficiency benchmarks

These scripts measure deployment inference and training cost using the
manuscript's evaluation settings.

## 🚀 Deployment

`benchmark_inference.py` reuses shared greedy generation on the same 100
seeded MESC-FT examples for both policies, with batch one, 256 output tokens,
eight images, and one CUDA device. Model loading and warm-up are excluded.

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/efficiency/benchmark_inference.py   --device cuda:0 --samples 100   --faces_model checkpoints/faces_main --fires_model checkpoints/grpo_baseline
```

It measures latency, output length, throughput, and peak allocated memory.
Reward models are not loaded for deployment inference.

## 🚀 Training cost

```bash
bash experiments/efficiency/run_training_cost.sh
```

The launcher runs `train_fires_grpo.py` and the full FACES trainer as separate
training commands. The FIRES entry point uses Format, ROUGE-L, and
DIFF-EPITOME/RoBERTa rewards with group-standardized completion advantages.
Following [FIRES Section 3.4.3](https://aclanthology.org/2025.findings-emnlp.70/),
the weights are 0.1, 1.0, and 0.1; the skill reward is one when DIFF-EPITOME
is at most one, and invalid-format completions receive zero reward. The FIRES
objective follows the completion-level probability ratio in Eqs. (2)-(3),
with one advantage per completion and an importance-weighted sampled sequence
KL penalty. It does not use the FACES token-level clipped objective. Both entry
points share model/data utilities; their rewards and optimization objectives
are separate.

For DIFF-EPITOME, prepare the three pretrained classifier checkpoints linked
by the [cited DIFF-EPITOME implementation](https://github.com/passing2961/EmpGPT-3#4-evaluation)
at:

```text
checkpoints/reward_model/epitome/
|-- finetuned_ER.pth
|-- finetuned_IP.pth
`-- finetuned_EX.pth
```

The loader uses that implementation's RoBERTa bi-encoder and single-head
cross-attention architecture. Context and response are encoded separately,
with at most 64 tokens each. Each classifier predicts its mechanism level
as the argmax class in `{0, 1, 2}`; DIFF-EPITOME is the mean squared difference
of these three levels between the generated and reference responses. The
continuous three-output regression model used by other repository experiments
is not used by this FIRES entry point. The checkpoints and RoBERTa-base
tokenizer/config files must be prepared before running the benchmark; use
`--epitome_roberta_path` when running the entry point with a local tokenizer.

Use two NVIDIA A6000 GPUs. The launcher matches the Qwen2.5-VL-7B policy
initialization in `checkpoints/warmup_merged/`, MESC-FT training split, batch
size 32, group size 4, generation limit 192, four updates per rollout, and
seed 42 over 100 GRPO steps. A measured step is one rollout batch followed by
four optimizer updates, so each run performs 100 rollout batches and 400
optimizer updates. Four updates per rollout and the shared initialization are
settings of this controlled comparison, not settings specified by the original
FIRES paper. The original FIRES paper reports four A6000 GPUs and DeepSpeed
ZeRO-2; this benchmark uses the manuscript's two-GPU allocation.
Training checkpoints are written separately under `checkpoints/training_cost/`.
`TRAINING_GPUS` selects the two allocated physical GPUs (default `0,1`); the
child commands see them as `cuda:0,1`.

`benchmark_training.py` passes `TRAINING_BENCHMARK_STEPS=100` and a fresh
`TRAINING_BENCHMARK_TIMINGS_FILE` to each trainer. Both entry points write
one JSONL row after each measured training step:

```json
{"step": 1, "wall_seconds": 12.3}
```

The duration above only illustrates the format. Measure each complete GRPO
step, including generation, reward computation, and policy updates; synchronize
both GPUs before starting and stopping its timer. Exclude model loading,
validation, checkpoint saving, and any preliminary warm-up from the measured
step durations. Number the 100 measured steps consecutively from 1 to 100.
The benchmark rejects missing or incomplete timing reports and computes
wall-hours from the sum of measured durations, then GPU-hours from the allocated
GPU count. It never treats total subprocess runtime as 100 measured steps.

[Repository overview](../../README.md)
