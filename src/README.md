# 🛠️ Source Code Guide

🛠️ This folder contains the training and data-construction code for FACES.
Runtime artifacts such as local checkpoints, extracted frames, model caches,
logs, and scores are written outside this source tree or under ignored paths.

## 📁 Directory Map

| Folder | Purpose |
| --- | --- |
| `image_data_synthesis/` | Optional scripts for preparing MESC-FT-style visual keyframes from raw videos |
| `generalization_data_synthesis/` | Optional scripts for rebuilding the MELD-ESC300, StickerConv, and AvaMERG external splits |
| `sft/` | SFT LoRA training and LoRA-to-base merge scripts |
| `MiniLM_reward_model/` | Optional DIFF-EPITOME support-skill reward model training and testing |
| `grpo/warmup/` | Warm-up GRPO with format and outcome rewards |
| `grpo/faces_grpo/` | Full FACES GRPO with multi-source rewards, gated fusion, and token-level credit assignment |

Run commands from the repository root unless a subfolder README states
otherwise.

[Repository overview](../README.md)
