# 🎚️ Hyperparameter Sensitivity

🎚️ This directory provides the launcher for the method-specific sensitivity
analysis reported in the paper appendix. The script reuses
`src/grpo/faces_grpo/train_faces_grpo.py` with one-factor-at-a-time parameter
overrides.

## 📁 Files

| File | Purpose |
| --- | --- |
| `run_sensitivity.sh` | Run one-factor-at-a-time sensitivity variants for FACES GRPO |

## ⚙️ Covered Hyperparameters

The appendix varies one factor at a time and keeps the rest of the FACES GRPO
configuration fixed.

| Paper symbol | Script argument | Default |
| --- | --- | --- |
| `tau_v` | `--visual_threshold` | `0.60` |
| `gamma` | `--segment_advantage_gate_floor` | `0.10` |
| `m` | `--endpoint_anchor_window` | `5` |
| `lambda` | `--endpoint_anchor_decay` | `0.80` |

## ⚙️ Variant Grid

| Family | Values |
| --- | --- |
| Visual threshold `tau_v` | `0.50`, `0.60`, `0.65`, `0.70` |
| Positive gate floor `gamma` | `0.00`, `0.10`, `0.30`, `1.00` |
| Endpoint anchor `(m, lambda)` | `(3, 0.80)`, `(5, 0.60)`, `(5, 0.80)`, `(7, 0.80)`, `(5, 1.00)` |

## 🚀 Run

```bash
PYTHON_BIN=python bash experiments/sensitivity/run_sensitivity.sh
```

Run only one family:

```bash
PYTHON_BIN=python bash experiments/sensitivity/run_sensitivity.sh visual_threshold
PYTHON_BIN=python bash experiments/sensitivity/run_sensitivity.sh gate_floor
PYTHON_BIN=python bash experiments/sensitivity/run_sensitivity.sh endpoint_anchor
```

Prepare the same local assets required by `src/grpo/faces_grpo/README.md`
before running this script.

Useful local overrides:

```bash
SENSITIVITY_CUDA_VISIBLE_DEVICES=0,1 \
SENSITIVITY_VAL_MAX_SAMPLES=100 \
SENSITIVITY_MAX_STEPS=0 \
PYTHON_BIN=python \
bash experiments/sensitivity/run_sensitivity.sh
```

`SENSITIVITY_MAX_STEPS=0` means the trainer uses its normal epoch-based run.
Set a positive value to limit the run to that number of steps.

[Repository overview](../../README.md)
