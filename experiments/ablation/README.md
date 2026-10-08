# 🧩 Ablation Experiments

🧯 This folder contains launchers for local ablation runs. The scripts do not
reimplement the GRPO algorithms; they call the existing trainers under
`src/grpo/` with different `--ablation` switches.

Each experiment writes its own local checkpoints and merged model under
`experiments/ablation/<experiment_name>/`.

## 📁 Files

| File | Purpose |
| --- | --- |
| `run_ablation.sh` | Runs the component ablation queue for warm-up and FACES GRPO |
| `run_contribution_ablation.sh` | Runs contribution-level FACES ablations |

## ⚙️ Ablation Switches

`run_ablation.sh` supports:

```text
a2_w_o_format
a3_w_o_visual
a3_w_o_emotion
a3_w_o_situation
a3_w_o_strategy
```

`run_contribution_ablation.sh` supports:

```text
a3_w_o_multi_source_reward
a3_w_o_noncomp_gating
a3_w_o_fine_grained_credit
```

## 🚀 Run

```bash
PYTHON_BIN=python bash experiments/ablation/run_ablation.sh
PYTHON_BIN=python bash experiments/ablation/run_contribution_ablation.sh
```

Run a single experiment:

```bash
PYTHON_BIN=python bash experiments/ablation/run_ablation.sh a3_w_o_visual
PYTHON_BIN=python bash experiments/ablation/run_contribution_ablation.sh a3_w_o_noncomp_gating
```

The scripts call:

```text
src/grpo/warmup/train_warmup_grpo.py
src/grpo/faces_grpo/train_faces_grpo.py
src/grpo/warmup/merge_warmup_lora.py
src/grpo/faces_grpo/merge_faces_lora.py
```

The ablation runtime folders remain local and are covered by `.gitignore`.

[Repository overview](../../README.md)
