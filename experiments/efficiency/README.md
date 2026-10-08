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

The shared-trainer wrapper compares the traditional Format/ROUGE-L/EPITOME
reward backend with full FACES over 100 GRPO steps and two allocated GPUs.
`benchmark_training.py` records wall time and computes GPU-hours from the
actual allocated device count. The FIRES-labelled traditional-reward arm is a
reward-cost approximation using the common trainer. It does not reproduce
a separate FIRES training implementation.

[Repository overview](../../README.md)
