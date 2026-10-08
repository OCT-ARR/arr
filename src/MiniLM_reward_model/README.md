# 🧠 EPITOME Support-skill Reward Model

This folder trains the RoBERTa support-skill model required for DIFF-EPITOME
evaluation and used by traditional-reward controls. Full FACES training uses a
separate frozen MiniLM encoder and does not require this model.

## 📁 Files

| File | Purpose |
| --- | --- |
| `train_epitome_multitask.py` | Train a three-head support-skill reward model |
| `test_reward_model.py` | Evaluate the trained reward model on the support-skill CSV files |

## 📦 What Is Required

The release includes the three EPITOME/Reddit support-skill CSV files
under:

```text
data/EPITOME-Reddit/
|-- emotional-reactions-reddit.csv
|-- interpretations-reddit.csv
`-- explorations-reddit.csv
```

The release does not include trained reward weights.

The trained model is written locally to:

```text
checkpoints/reward_model/roberta_epitome_multitask/final/
```

## 🚀 Train

```bash
python src/MiniLM_reward_model/train_epitome_multitask.py \
  --data_dir data/EPITOME-Reddit \
  --output_dir checkpoints/reward_model/roberta_epitome_multitask \
  --model_checkpoint roberta-base \
  --fp16
```

The model predicts three EPITOME dimensions in this order:

```text
[emotional reaction, interpretation, exploration]
```

## 📊 Evaluate

```bash
python src/MiniLM_reward_model/test_reward_model.py \
  --model_path checkpoints/reward_model/roberta_epitome_multitask/final \
  --data_dir data/EPITOME-Reddit
```

## 🧭 Relationship To MiniLM

`models/paraphrase-multilingual-MiniLM-L12-v2/` is a separate public
sentence-transformer used by the FACES GRPO emotion semantic reward.
It does not need to be trained during SFT. Download it locally before running
the FACES GRPO stage.

The directory name is retained for compatibility. This trainer uses RoBERTa;
it does not train the frozen public MiniLM emotion encoder.

[Repository overview](../../README.md)
