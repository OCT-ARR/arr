# ⚙️ Configuration guide

| File | Purpose |
| --- | --- |
| [paths.example.json](paths.example.json) | Relative data, model, and checkpoint paths |
| [paper_protocols.json](paper_protocols.json) | Training, rewards, controlled judging, human evaluation, and metric settings |
| [optimization_baselines.json](optimization_baselines.json) | Shared controls and algorithm-specific DPO/SimPO/RLOO/PPO defaults |

These files describe reproduction settings. Scripts expose their own command-line
options and do not all load the JSON files automatically. BERTScore uses
RoBERTa-large layer 24 without IDF or baseline rescaling.

[Repository overview](../README.md)
