# ⚖️ Optimization Baselines

This directory implements DPO, SimPO, and RLOO controls from the training
protocol described in the paper. Algorithm-specific settings not specified
in the manuscript are configurable below.

## 🧭 Scope

- DPO, SimPO, and RLOO correspond to Table 7.
- PPO is an optional control because it is discussed in related work but is not
  a Table 7 row.
- Every method uses Qwen2.5-VL-7B, the same SFT initialization, training
  prompts, three-epoch/global-batch-32 budget, LoRA setup, rollout limits, and
  MESC evaluation protocol.
- Candidate ranking and online scalar rewards use the complete gated FACES
  reward. DPO/SimPO receive the same fixed preference pairs; RLOO/PPO receive
  the same online reward.

Algorithm-specific values absent from the paper are exposed in
`configs/optimization_baselines.json`. The defaults are conventional starting
points: DPO beta 0.1; SimPO beta 2.0 and target margin 1.0; PPO clip 0.2,
GAE lambda 0.95, and value-loss coefficient 0.5. Changing one of these values
changes the reconstructed protocol and should be reported.

Keep each method's configuration unchanged across repeated runs so the
comparison uses consistent settings.

## 🚀 Run

Unpack the released JSONL splits and prepare the local model/frame assets first.
From the repository root, run the three paper baselines with:

```bash
bash experiments/optimization_baselines/run_baselines.sh
```

Set `MERGE_BASELINES=1` to merge each LoRA adapter after training. PPO is only
run when requested explicitly:

```bash
bash experiments/optimization_baselines/run_baselines.sh dpo simpo rloo ppo
```

For DPO and SimPO, `prepare_preferences.py` samples four responses per training
prompt from the untouched SFT policy and retains the highest/lowest FACES
reward pair. It records token IDs and source hashes so both methods consume
identical prompts and responses without silently mixing data versions. A prompt
with tied rewards is resampled up to three rounds; preparation stops rather
than silently dropping a training prompt if no strict preference is obtained.

Generate greedy test outputs after merging a method:

```bash
python experiments/optimization_baselines/generate_baseline_outputs.py --method dpo
python experiments/llm_blind_judge/pointwise_controlled_eval.py \
  --predictions outputs/optimization_baselines/generations/dpo.jsonl \
  --model_name dpo
```

[Repository overview](../../README.md)
