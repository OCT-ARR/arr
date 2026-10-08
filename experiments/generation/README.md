# 💬 Candidate Generation

📝 This folder regenerates local JSONL candidate outputs for downstream
automatic metrics, MLLM judge evaluation, and significance tests.

## 📁 Files

| File | Purpose |
| --- | --- |
| `generate_mesc_outputs.py` | Generate MESC-FT outputs for `faces_main` and `grpo_baseline` |

## 🚀 Run

```bash
CUDA_VISIBLE_DEVICES=0,1 python experiments/generation/generate_mesc_outputs.py
```

Outputs are written under `experiments/generation/outputs/`.

Prepare these local paths first:

```text
checkpoints/faces_main/
checkpoints/grpo_baseline/   # FIRES (Baseline)
data/MESC-FT/test.jsonl
keyframes/
```

`grpo_baseline` is the internal checkpoint/output identifier for the paper's
`FIRES (Baseline)` model.

[Repository overview](../../README.md)
